# Path: backend/app/evacuation/ea.py
from typing import Dict, List, Any, Optional, Tuple, Union, Set, Callable
import random
import time
import copy
import csv
from collections import defaultdict
import math
import numpy as np

from .algorithm_interface import EvacuationAlgorithm, AlgorithmResult
from .core import (
    initialize_problem_data, travel_time_for_trip, decode_individual,
    PENALTY_FACTOR, EXTRA_TRIP_PENALTY_FACTOR, STOP_EMPTY_PENALTY,
    STOP_FULL_PENALTY, LATE_PENALTY, DEPOT_OVERFILL_PENALTY
)
from . import visualization
from ..logging_utils import log_evacuation_run, log_generation_metrics
from .metrics import compute_solution_metrics, _simulate_and_get_timings
from .local_search import MemeticImprover
from .runtime_budget import RuntimeBudget


class RevisionaryEvolutionaryAlgorithm(EvacuationAlgorithm):
    """

    """

    # ---- Instance-scoped runtime config (set inside run) ----
    _cap_by_bus: Optional[List[int]] = None
    _origin_by_bus: Optional[List[Dict[str, Any]]] = None
    _depots_runtime: Optional[List[Dict[str, Any]]] = None 
    _facilities_runtime: Optional[List[Dict[str, Any]]] = None
    _node_coords: Optional[Dict[int, Tuple[float, float]]] = None
    _start_to_node_seconds: Optional[Dict[int, Dict[int, float]]] = None
    _avg_speed_kmh: float = 30.0
    _road_factor: float = 1.25
    _buses_count_runtime: int = 0
    _service_params: Dict[str, Any] = {}

    def run(
        self,
        evacuation_zones_input: Optional[List[Dict[str, Any]]] = None,
        buses_count: int = 3,
        bus_capacity: int = 80,
        use_local_search: bool = True,
        vehicles: Optional[List[Dict[str, Any]]] = None,
        start_to_node_seconds: Optional[Dict[int, Dict[int, float]]] = None,
        avg_speed_kmh: float = 30.0,
        road_factor: float = 1.25,
        precomputed_problem_data: Optional[Dict[str, Any]] = None,
        collect_operator_telemetry: bool = False,
        **algorithm_specific_params
    ) -> AlgorithmResult:

        # Monotonic start for the whole solver run.
        run_start_wall = time.monotonic()

        population_size = algorithm_specific_params.get('population_size', 50)
        generations = algorithm_specific_params.get('generations', 100)
        crossover_rate = algorithm_specific_params.get('crossover_rate', 0.2)
        mutation_rate = algorithm_specific_params.get('mutation_rate', 0.8)
        tournament_size = algorithm_specific_params.get('tournament_size', 3)
        penalty_factor = 1000
        lateness_penalty_factor = algorithm_specific_params.get('lateness_penalty_factor', 50)
        latest_evacuation_penalty_factor = algorithm_specific_params.get('latest_evacuation_penalty_factor', 0)

        early_stopping_generations = algorithm_specific_params.get('early_stopping_generations', 1000)
        if not isinstance(early_stopping_generations, int) or early_stopping_generations <= 0:
            early_stopping_generations = None  # Disable if invalid or non-positive
        print(30*"-", early_stopping_generations)

        self._service_params = {
            "use_dynamic_service_time": algorithm_specific_params.get('use_dynamic_service_time', False),
            "service_time_base_min": algorithm_specific_params.get('service_time_base_min', 3.0),
            "service_time_per_person_min": algorithm_specific_params.get('service_time_per_person_min', 20.0 / 60.0),
        }
        print(f"Service Time Params: {self._service_params}")

        # Optional wall-clock budget (seconds). Generation only starts if budget remains.
        time_limit_seconds = algorithm_specific_params.get('time_limit_seconds', None)
        if isinstance(time_limit_seconds, (int, float)) and time_limit_seconds <= 0:
            time_limit_seconds = None
        budget = RuntimeBudget(
            limit_seconds=time_limit_seconds,
            mode=algorithm_specific_params.get("budget_mode", "strict"),
            postprocess_reserve_seconds=float(
                algorithm_specific_params.get("postprocess_reserve_seconds", 0.25)
            ),
            run_started_at=run_start_wall,
        )

        # Allow experiment to toggle generation awareness of memetic
        ls_gen_aware = bool(algorithm_specific_params.get("ls_gen_aware", True))

        # Optional center/buffer parameters 
        default_evac_center_coords = algorithm_specific_params.get('default_evac_center_coords', None)
        buffer_meters = algorithm_specific_params.get('buffer_meters', None)

        print(f"Tournament size: {tournament_size}")
        print(f"Latest evacuation penalty factor: {latest_evacuation_penalty_factor}")
        print(f'using local search: {use_local_search}')
        if precomputed_problem_data:
            print("📦 Using pre-computed problem data (Matrix & Graph)...")
            problem_data = precomputed_problem_data
        else:
            print("🌐 Calculating problem data from scratch (API/OSRM)...")
            problem_data = self.initialize_problem(
                evacuation_zones_input, buses_count, bus_capacity,
                default_evac_center_coords=default_evac_center_coords,
                buffer_meters=buffer_meters
            )

        depots = problem_data['depots']
        self._depots_runtime = depots 
        facilities = problem_data['facilities']
        self._facilities_runtime = facilities
        durations_matrix = problem_data['durations_matrix']
        n_depots = problem_data['n_depots']
        n_facilities = problem_data['n_facilities']
        demand_full = problem_data['demand_full']
        print("Full Demand: ", demand_full)
        deadlines = problem_data['deadlines']
        pickup_nodes = problem_data['pickup_nodes']
        max_trips_per_bus = problem_data['max_trips_per_bus']
        max_stops_per_trip = problem_data['max_stops_per_trip']


        cap_by_bus, origin_by_bus, buses_count_effective, normalized_vehicles = \
            self._resolve_fleet(vehicles, buses_count, bus_capacity, n_depots, pickup_nodes)

        print(f"🚌 Effective Fleet Capacities: {cap_by_bus}")

        # store coords/first-leg providers 
        self._node_coords = problem_data.get("node_coords", None)  # Dict[int -> (lat, lon)]
        self._start_to_node_seconds = start_to_node_seconds or None
        self._avg_speed_kmh = float(avg_speed_kmh)
        self._road_factor = float(road_factor)

        # resolve heterogeneous vehicles/origins and persist on self
        cap_by_bus, origin_by_bus, buses_count_effective, normalized_vehicles = \
            self._resolve_fleet(vehicles, buses_count, bus_capacity, n_depots, pickup_nodes)

        self._cap_by_bus = cap_by_bus
        self._origin_by_bus = origin_by_bus
        self._buses_count_runtime = buses_count_effective

        # Update buses_count for the rest of the run (population dims, etc.)
        buses_count = buses_count_effective

        ls_params = algorithm_specific_params.get("ls_params", {})
        memetic = None
        if use_local_search:
            # MemeticImprover now receives hetero-fleet context + origin-aware first-leg providers.
            memetic = MemeticImprover(
                evaluate_fitness=self._evaluate_fitness,
                repair=self._repair,
                fix_depot_connectivity=self._fix_depot_connectivity,  
                buses_count=buses_count,
                bus_capacity=bus_capacity,  
                depots=depots,
                facilities=facilities,
                n_depots=n_depots,
                pickup_nodes=pickup_nodes,
                durations_matrix=durations_matrix,
                demand_full=demand_full,
                deadlines=deadlines,
                penalty_factor=penalty_factor,
                lateness_penalty_factor=lateness_penalty_factor,
                latest_evacuation_penalty_factor=latest_evacuation_penalty_factor,
                cap_by_bus=cap_by_bus,
                origin_by_bus=origin_by_bus,
                node_coords=self._node_coords,
                start_to_node_seconds=self._start_to_node_seconds,
                avg_speed_kmh=self._avg_speed_kmh,
                road_factor=self._road_factor,
                **self._service_params,
            )

        # LS policy (overridable via params)
        ls_policy = {
            "apply_every": int(algorithm_specific_params.get("ls_every", 3)),
            "burn_in_generations": int(algorithm_specific_params.get("ls_burn_in", 5)),
            "offspring_probability": float(algorithm_specific_params.get("ls_offspring_prob", 0.20)),
            "max_elites_to_polish": int(algorithm_specific_params.get("ls_max_elites", 2)),
        }


        # Separate budgets for elites vs offspring
        ls_params_elite = dict(ls_params)
        ls_params_elite.setdefault("time_limit_seconds", 0.06)
        ls_params_elite.setdefault("max_iterations", 80)
        ls_params_elite.setdefault("candidate_set_size", 16)
        ls_params_elite.setdefault("use_alns_shake", False)

        ls_params_offspring = dict(ls_params)
        ls_params_offspring.setdefault("time_limit_seconds", 0.03)
        ls_params_offspring.setdefault("max_iterations", 40)
        ls_params_offspring.setdefault("candidate_set_size", 12)
        ls_params_offspring.setdefault("use_alns_shake", False)

        population = self._initialize_population(
            population_size, buses_count, bus_capacity, depots, facilities,
            n_depots, pickup_nodes, max_trips_per_bus, max_stops_per_trip,
            durations_matrix, demand_full, deadlines,
            default_evac_center_coords=default_evac_center_coords,
            buffer_meters=buffer_meters,
            precomputed_problem_data=problem_data,
            deadline=budget.deadline if budget.is_strict else None,
        )

        evaluated_population = []
        fitness_values = []
        for individual in population:
            # Always evaluate at least one incumbent, even when initialization
            # consumed an exceptionally short strict budget.
            if budget.is_strict and fitness_values and budget.expired():
                break
            fitness_values.append(self._evaluate_fitness(
                individual, buses_count, bus_capacity, depots, facilities,
                n_depots, durations_matrix, demand_full, deadlines,
                penalty_factor, lateness_penalty_factor, latest_evacuation_penalty_factor
            ))
            evaluated_population.append(individual)
        population = evaluated_population

        if not population:
            raise RuntimeError("Unable to construct an initial feasible incumbent.")

        best_individual_idx = fitness_values.index(min(fitness_values))
        best_individual = population[best_individual_idx]
        best_fitness = fitness_values[best_individual_idx]

        # For early stopping
        last_improvement_gen = 0

        # Stats (include precise timing + LS/GA split)
        algorithm_stats = {
            "generation_costs": [best_fitness],
            "generation_times": [0.0],
            "ls_time_per_generation": [0.0],
            "ls_time_this_gen": [0.0],
            "ga_time_per_generation": [0.0],
            "best_per_generation": [best_fitness],
            "avg_per_generation": [sum(fitness_values) / len(fitness_values)],
            "generation_avg_evacuation_times": [],
            "generation_latest_evacuation_times": [],
            "generation_total_people_evacuated": [],
            "generation_evacuation_efficiencies": [],
            "generation_fitness_std": [],
            "generation_population_diversity": [],
            "time_limit_seconds": time_limit_seconds,
            "stopped_by_time_limit": False,
            "stopped_by_early_stopping": False,
            "preprocessing_runtime": None,
            "optimization_runtime": None,
            "postprocessing_runtime": None,
            "total_runtime": None,
            "collect_operator_telemetry": bool(collect_operator_telemetry),
            **budget.metadata(),
        }

        # Gen 0 metrics
        print(f"🔄 Logging initial generation metrics...")
        _ = self._extract_and_log_generation_metrics(
            0, population, fitness_values, buses_count, bus_capacity,
            depots, facilities, n_depots, durations_matrix, demand_full, deadlines,
            algorithm_stats
        )

        # In legacy_results mode the budget starts here. In strict mode the
        # deadline was fixed at solver entry and initialization is included.
        start_time_wall = budget.start_search()
        algorithm_stats["preprocessing_runtime"] = budget.preprocessing_runtime()

        # --- METRICS SCOREBOARD INIT ---
        if collect_operator_telemetry:
            operator_scoreboard = {
                "ga_crossover": 0.0,
                "ga_crossover_cnt": 0,
            }
        else:
            operator_scoreboard = {}
        # ------------------------------

        # --- CSV LOGGING SETUP ---
        ops_csv_path = None
        all_known_ops = []
        if collect_operator_telemetry:
            output_dir = algorithm_specific_params.get("output_dir")
            # Define stable column order
            all_known_ops = [
                "ga_crossover", 
                "mutate_intra_swap", "mutate_relocate_stop", "mutate_add_remove_trip", 
                "mutate_change_depot", "mutate_swap_trip", "mutate_spatial_ruin",
                "intra_trip", "relocate", "swap_stops", "swap_trips", "move_trip", 
                "change_depot", "consolidate_trips", "balance_makespan", "takeover_gap", 
                "fill_idle_time", "spatial_relocate", "split_mixed", "crumb_extract", "self_consolidate",
                "quantity_rebalance"
            ]
            
            if output_dir:
                from pathlib import Path
                ops_csv_path = Path(output_dir) / "operator_history.csv"
                try:
                    with open(ops_csv_path, "w", newline="") as f:
                        writer = csv.writer(f)
                        # Header: Gen, Op1_CumulativeGain, Op1_CumulativeCount, ...
                        header = ["Generation"]
                        for op in all_known_ops:
                            header.extend([f"{op}_gain", f"{op}_cnt"])
                        writer.writerow(header)
                except Exception as e:
                    print(f"⚠️ Failed to init CSV: {e}")
        import time as _t
        for generation in range(generations):
            if budget.expired():
                algorithm_stats["stopped_by_time_limit"] = True
                print(f"⏱️ Time limit ({time_limit_seconds:.1f}s) reached; stopping before generation {generation}.")
                break

            if early_stopping_generations is not None and (generation - last_improvement_gen) > early_stopping_generations:
                algorithm_stats["stopped_by_early_stopping"] = True
                print(f"🛑 No improvement for {early_stopping_generations} generations. Stopping early before generation {generation}.")
                break

            gen_wall_start = _t.perf_counter()
            gen_ls_time = 0.0
            generation_incomplete = False

            new_population = []

            elite_count = max(2, int(population_size * 0.05))
            elite_indices = sorted(
                range(len(fitness_values)),
                key=lambda i: fitness_values[i]
            )[:elite_count]

            apply_ls_this_gen = (
                use_local_search
                and (generation >= ls_policy["burn_in_generations"])
                and ((generation % ls_policy["apply_every"]) == 0)
            )
            context = {
                "generation": generation,
                "algorithm_start_time": start_time_wall,
                "time_limit_seconds": time_limit_seconds,
                "deadline_monotonic": (
                    budget.deadline if budget.is_strict else None
                ),
            } if use_local_search else None

            # Elites
            for rank, idx in enumerate(elite_indices):
                if budget.is_strict and budget.expired():
                    generation_incomplete = True
                    break
                elite = copy.deepcopy(population[idx])
                if apply_ls_this_gen and rank < ls_policy["max_elites_to_polish"]:
                    t0 = _t.perf_counter()
                    elite = memetic.improve(elite, ls_params_elite, context=context)
                    gen_ls_time += (_t.perf_counter() - t0)
                    # Collect Elite LS stats
                    if collect_operator_telemetry and hasattr(memetic, 'get_last_run_stats'):
                        ls_stats = memetic.get_last_run_stats()
                        for k, v in ls_stats.items():
                            operator_scoreboard[k] = operator_scoreboard.get(k, 0.0) + v
                new_population.append(elite)

            if generation_incomplete:
                algorithm_stats["stopped_by_time_limit"] = True
                break

            # Offspring Loop
            while len(new_population) < population_size:
                if budget.is_strict and budget.expired():
                    generation_incomplete = True
                    break
                parent1 = self._tournament_selection(population, fitness_values, tournament_size)
                parent2 = self._tournament_selection(population, fitness_values, tournament_size)
                
                # Evaluate parents for GA stats (telemetry only)
                parents_min = None
                if collect_operator_telemetry:
                    p1_cost = self._evaluate_fitness(parent1, buses_count, bus_capacity, depots, facilities, n_depots, durations_matrix, demand_full, deadlines, penalty_factor, lateness_penalty_factor, latest_evacuation_penalty_factor)
                    p2_cost = self._evaluate_fitness(parent2, buses_count, bus_capacity, depots, facilities, n_depots, durations_matrix, demand_full, deadlines, penalty_factor, lateness_penalty_factor, latest_evacuation_penalty_factor)
                    parents_min = min(p1_cost, p2_cost)

                # --- Crossover ---
                if random.random() < crossover_rate:
                    offspring1, offspring2 = self._crossover(
                        parent1, parent2, buses_count, bus_capacity, depots, facilities,
                        n_depots, durations_matrix, demand_full, deadlines
                    )
                    if collect_operator_telemetry:
                        # Evaluate Crossover Gain
                        # We use temp copies to avoid messing up the flow if repairs are needed later
                        t_o1 = self._repair(offspring1, buses_count, bus_capacity, depots, facilities, n_depots, pickup_nodes, durations_matrix, demand_full, deadlines)
                        t_o2 = self._repair(offspring2, buses_count, bus_capacity, depots, facilities, n_depots, pickup_nodes, durations_matrix, demand_full, deadlines)
                        o1_c = self._evaluate_fitness(t_o1, buses_count, bus_capacity, depots, facilities, n_depots, durations_matrix, demand_full, deadlines, penalty_factor, lateness_penalty_factor, latest_evacuation_penalty_factor)
                        o2_c = self._evaluate_fitness(t_o2, buses_count, bus_capacity, depots, facilities, n_depots, durations_matrix, demand_full, deadlines, penalty_factor, lateness_penalty_factor, latest_evacuation_penalty_factor)
                        gain = max(0, parents_min - min(o1_c, o2_c))
                        operator_scoreboard["ga_crossover"] += gain
                        if gain > 0: operator_scoreboard["ga_crossover_cnt"] += 1
                else:
                    offspring1, offspring2 = parent1, parent2

                # --- Mutation (Now with Telemetry) ---
                # Helper to handle single offspring mutation
                def process_mutation(ind):
                    if random.random() < mutation_rate:
                        if collect_operator_telemetry:
                            # Eval Before
                            c_before = self._evaluate_fitness(ind, buses_count, bus_capacity, depots, facilities, n_depots, durations_matrix, demand_full, deadlines, penalty_factor, lateness_penalty_factor, latest_evacuation_penalty_factor)

                        # Mutate
                        mutated, op_name = self._mutate(
                            ind, buses_count, bus_capacity, depots, facilities,
                            n_depots, pickup_nodes, durations_matrix, demand_full, deadlines
                        )
                        
                        # Repair
                        mutated = self._repair(
                            mutated, buses_count, bus_capacity, depots, facilities,
                            n_depots, pickup_nodes, durations_matrix, demand_full, deadlines
                        )
                        
                        if collect_operator_telemetry:
                            # Eval After
                            c_after = self._evaluate_fitness(mutated, buses_count, bus_capacity, depots, facilities, n_depots, durations_matrix, demand_full, deadlines, penalty_factor, lateness_penalty_factor, latest_evacuation_penalty_factor)
                            
                            # Record Stats
                            gain = max(0, c_before - c_after)
                            if op_name not in operator_scoreboard:
                                operator_scoreboard[op_name] = 0.0
                                operator_scoreboard[f"{op_name}_cnt"] = 0
                            operator_scoreboard[op_name] += gain
                            if gain > 0: operator_scoreboard[f"{op_name}_cnt"] += 1
                        
                        return mutated
                    else:
                        # Ensure valid even if not mutated
                        return self._repair(ind, buses_count, bus_capacity, depots, facilities, n_depots, pickup_nodes, durations_matrix, demand_full, deadlines)

                offspring1 = process_mutation(offspring1)
                offspring2 = process_mutation(offspring2)

                if budget.is_strict and budget.expired():
                    generation_incomplete = True
                    break

                # --- Local Search ---
                if apply_ls_this_gen and random.random() < ls_policy["offspring_probability"]:
                    t0 = _t.perf_counter()
                    offspring1 = memetic.improve(offspring1, ls_params_offspring, context=context)
                    gen_ls_time += (_t.perf_counter() - t0)
                    if collect_operator_telemetry and hasattr(memetic, 'get_last_run_stats'):
                        for k, v in memetic.get_last_run_stats().items():
                            operator_scoreboard[k] = operator_scoreboard.get(k, 0.0) + v

                if apply_ls_this_gen and random.random() < ls_policy["offspring_probability"]:
                    t0 = _t.perf_counter()
                    offspring2 = memetic.improve(offspring2, ls_params_offspring, context=context)
                    gen_ls_time += (_t.perf_counter() - t0)
                    if collect_operator_telemetry and hasattr(memetic, 'get_last_run_stats'):
                         for k, v in memetic.get_last_run_stats().items():
                            operator_scoreboard[k] = operator_scoreboard.get(k, 0.0) + v

                new_population.append(offspring1)
                if len(new_population) < population_size:
                    new_population.append(offspring2)

            if generation_incomplete:
                algorithm_stats["stopped_by_time_limit"] = True
                break

            new_fitness_values = []
            for individual in new_population:
                if budget.is_strict and budget.expired():
                    generation_incomplete = True
                    break
                new_fitness_values.append(self._evaluate_fitness(
                    individual, buses_count, bus_capacity, depots, facilities,
                    n_depots, durations_matrix, demand_full, deadlines,
                    penalty_factor, lateness_penalty_factor, latest_evacuation_penalty_factor
                ))

            if budget.is_strict and budget.expired():
                generation_incomplete = True
            if generation_incomplete:
                algorithm_stats["stopped_by_time_limit"] = True
                break

            population = new_population
            fitness_values = new_fitness_values

            current_best_idx = fitness_values.index(min(fitness_values))
            if fitness_values[current_best_idx] < best_fitness:
                best_individual = population[current_best_idx]
                best_fitness = fitness_values[current_best_idx]
                last_improvement_gen = generation

            generation_metrics = self._extract_and_log_generation_metrics(
                generation + 1, population, fitness_values, buses_count, bus_capacity,
                depots, facilities, n_depots, durations_matrix, demand_full, deadlines,
                algorithm_stats
            )

            gen_wall = _t.perf_counter() - gen_wall_start
            gen_ga_time = max(0.0, gen_wall - gen_ls_time)

            algorithm_stats["generation_times"].append(gen_wall)
            algorithm_stats["ls_time_per_generation"].append(gen_ls_time)
            algorithm_stats["ls_time_this_gen"].append(gen_ls_time)
            algorithm_stats["ga_time_per_generation"].append(gen_ga_time)

            algorithm_stats["generation_costs"].append(best_fitness)
            algorithm_stats["best_per_generation"].append(min(fitness_values))
            algorithm_stats["avg_per_generation"].append(sum(fitness_values) / len(fitness_values))

            log_generation_metrics(generation + 1, generation_metrics)

            if (generation) % 10 == 0:
                avg_evac_time = generation_metrics.get('gen_avg_evacuation_time')
                latest_evac_time = generation_metrics.get('gen_latest_evacuation_time')
                people_evacuated = generation_metrics.get('gen_total_people_evacuated', 0)

                print(f"Generation {generation :3d}/{generations}: "
                      f"Cost={best_fitness:8.2f}, "
                      f"AvgEvac={avg_evac_time or -1.0:>7.3f}min, "
                      f"Latest={latest_evac_time or 'N/A':>6.1f}min, "
                      f"People={people_evacuated}, "
                      f"gen_time={gen_wall:.3f}s (LS {gen_ls_time:.3f}s)")
                
                if collect_operator_telemetry:
                    # --- PRINT METRICS TABLE ---
                    print("\n--- 🔧 OPERATOR INFLUENCE REPORT (Cumulative Gain) ---")
                    print(f"{'Operator':<25} | {'Total Gain':<15} | {'Successes':<10}")
                    print("-" * 55)
                    sorted_ops = sorted(
                        [k for k in operator_scoreboard.keys() if not k.endswith('_cnt')],
                        key=lambda k: operator_scoreboard[k], reverse=True
                    )
                    for op in sorted_ops:
                        val = operator_scoreboard[op]
                        cnt = operator_scoreboard.get(f"{op}_cnt", "N/A")
                        if val > 1.0:
                            print(f"{op:<25} | {val:15.2f} | {str(cnt):<10}")
                    print("-" * 55)
            
            if collect_operator_telemetry and ops_csv_path:
                try:
                    with open(ops_csv_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        row = [generation]
                        for op in all_known_ops:
                            # 1. Cumulative Gain (Rounded to 2 decimals)
                            gain = operator_scoreboard.get(op, 0.0)
                            row.append(f"{gain:.2f}")
                            
                            # 2. Cumulative Count (Integer)
                            count = operator_scoreboard.get(f"{op}_cnt", 0)
                            row.append(int(count))
                        writer.writerow(row)
                except Exception as e:
                    # Ignore write errors to avoid crashing the run
                    pass


        optimization_end_time = budget.now()
        algorithm_stats["optimization_runtime"] = budget.search_runtime(
            optimization_end_time
        )

        best_solution = self._individual_to_solution(
            best_individual, buses_count, depots, n_depots
        )

        simulation_data = self.create_simulation_data(
            best_solution, buses_count, self._cap_by_bus, depots, facilities,
            n_depots, durations_matrix, demand_full, deadlines,
            **self._service_params
        )

        solution_summary = visualization.create_solution_summary(
            best_solution, buses_count, simulation_data
        )

        algorithm_stats["operator_scoreboard"] = operator_scoreboard

        result = self.create_result_object(
            "revisionary_evolutionary_algorithm", best_fitness, best_solution, simulation_data,
            depots, facilities, buses_count, bus_capacity
        )

        # Expose the actual route plan & heterogeneous fleet
        result["buses_count"] = buses_count
        result["bus_capacity"] = bus_capacity  # kept for compatibility
        result["best_solution"] = best_solution
        result["vehicles"] = normalized_vehicles

        # problem primitives (single source of truth for downstream consumers)
        result["problem_data"] = {
            "n_depots": n_depots,
            "durations_matrix": {str(k): v for k, v in durations_matrix.items()},
            "demand_full": demand_full,
            "deadlines": deadlines,
            "max_trips_per_bus": max_trips_per_bus,
            "max_stops_per_trip": max_stops_per_trip,
            "depots": depots,
            "facilities": facilities,
            "vehicles": normalized_vehicles,  
        }

        # keep existing attachments
        result["algorithm_stats"] = algorithm_stats
        result["stopped_by_time_limit"] = bool(algorithm_stats.get("stopped_by_time_limit", False))
        result["stopped_by_early_stopping"] = bool(algorithm_stats.get("stopped_by_early_stopping", False))
        result["time_limit_seconds"] = time_limit_seconds
        result["budget_mode"] = budget.mode

        # compute and attach metrics (minutes/person-minutes)
        result["metrics"] = compute_solution_metrics(
            solution=best_solution,
            buses_count=buses_count,
            n_depots=n_depots,
            durations_matrix=durations_matrix,
            demand_full=demand_full,
            deadlines=deadlines,
            depots=depots, #  Pass depots to metrics calculation
            service_time_min=10.0, # Kept for legacy; service params are used
            vehicles=normalized_vehicles,
            node_coords=self._node_coords,
            start_to_node_seconds=self._start_to_node_seconds,
            avg_speed_kmh=self._avg_speed_kmh,
            road_factor=self._road_factor,
            **self._service_params,
        )

        run_end_wall = budget.now()
        algorithm_stats["total_runtime"] = budget.total_runtime(run_end_wall)
        if (
            algorithm_stats.get("preprocessing_runtime") is not None
            and algorithm_stats.get("optimization_runtime") is not None
        ):
            algorithm_stats["postprocessing_runtime"] = float(
                algorithm_stats["total_runtime"]
                - algorithm_stats["preprocessing_runtime"]
                - algorithm_stats["optimization_runtime"]
            )

        opt_runtime = algorithm_stats.get("optimization_runtime", algorithm_stats["total_runtime"])
        if algorithm_stats.get("stopped_by_time_limit"):
            print(f"⏱️ Stopped by time limit after {opt_runtime:.2f} seconds")
        elif algorithm_stats.get("stopped_by_early_stopping"):
             print(f"🛑 Stopped due to no improvement after {opt_runtime:.2f} seconds")
        else:
            print(f"Total algorithm runtime: {algorithm_stats['total_runtime']:.2f} seconds")

        algorithm_params = {
            "buses_count": buses_count,
            "bus_capacity": bus_capacity,
            "population_size": population_size,
            "generations": generations,
            "crossover_rate": crossover_rate,
            "mutation_rate": mutation_rate,
            "tournament_size": tournament_size,
            "penalty_factor": penalty_factor,
            "lateness_penalty_factor": lateness_penalty_factor,
            "latest_evacuation_penalty_factor": latest_evacuation_penalty_factor,
            "ls_gen_aware": ls_gen_aware,
            "use_local_search": use_local_search,
            "collect_operator_telemetry": bool(collect_operator_telemetry),
            "time_limit_seconds": time_limit_seconds,
            "budget_mode": budget.mode,
            "early_stopping_generations": early_stopping_generations,
            **self._service_params,
        }

        self.log_algorithm_run(
            "revisionary_evolutionary_algorithm",
            algorithm_params,
            facilities,
            best_fitness,
            solution_summary,
            depots,
            algorithm_stats=algorithm_stats
        )

        if not budget.is_strict:
            self._print_detailed_debug_info(best_solution, n_depots, durations_matrix)

        final_end = budget.now()
        algorithm_stats["total_runtime"] = budget.total_runtime(final_end)
        algorithm_stats["postprocessing_runtime"] = max(
            0.0,
            algorithm_stats["total_runtime"]
            - algorithm_stats["preprocessing_runtime"]
            - algorithm_stats["optimization_runtime"],
        )
        algorithm_stats["budget_overshoot_seconds"] = budget.overshoot_seconds(final_end)
        algorithm_stats["budget_adhered"] = (
            not budget.is_strict
            or algorithm_stats["budget_overshoot_seconds"] <= 1e-9
        )
        result["optimization_runtime"] = algorithm_stats["optimization_runtime"]
        result["total_runtime"] = algorithm_stats["total_runtime"]

        return result

    # ---------- fleet normalization ----------
    def _resolve_fleet(
        self,
        vehicles: Optional[List[Dict[str, Any]]],
        buses_count: int,
        bus_capacity: int,
        n_depots: int,
        pickup_nodes: List[int],
    ) -> Tuple[List[int], List[Dict[str, Any]], int, List[Dict[str, Any]]]:
        """
        Returns:
            cap_by_bus, origin_by_bus, buses_count_effective, normalized_vehicles_list
        Normalized vehicle item:
            {"id": str|None, "capacity": int, "start": {"kind": "depot|node|coord", ...}}
        """
        if not vehicles:
            # Homogeneous fallback (unchanged behavior)
            cap_by_bus = [int(bus_capacity) for _ in range(buses_count)]
            origin_by_bus = [{"kind": "depot", "index": 0} for _ in range(buses_count)]
            normalized = [
                {"id": None, "capacity": int(bus_capacity), "start": {"kind": "depot", "index": 0}}
                for _ in range(buses_count)
            ]
            return cap_by_bus, origin_by_bus, buses_count, normalized

        # Heterogeneous: validate and normalize
        cap_by_bus: List[int] = []
        origin_by_bus: List[Dict[str, Any]] = []
        normalized: List[Dict[str, Any]] = []
        buses_count_effective = len(vehicles)

        pickup_nodes_set = set(pickup_nodes)

        for v in vehicles:
            vid = v.get("id")
            cap = int(v.get("capacity", 0))
            if cap <= 0:
                raise ValueError(f"Vehicle capacity must be > 0 (vehicle id={vid}).")

            has_sd = "start_depot" in v and v["start_depot"] is not None
            has_sn = "start_node" in v and v["start_node"] is not None
            has_sc = "start_coord" in v and v["start_coord"] is not None

            if sum([has_sd, has_sn, has_sc]) > 1:
                raise ValueError(f"Vehicle may specify only one of start_depot/start_node/start_coord (vehicle id={vid}).")

            if has_sd:
                sd = int(v["start_depot"])
                if not (0 <= sd < n_depots):
                    raise ValueError(f"start_depot out of range (vehicle id={vid}).")
                origin = {"kind": "depot", "index": sd}
                start_norm = {"kind": "depot", "index": sd}

            elif has_sn:
                sn = int(v["start_node"])
                if sn not in pickup_nodes_set:
                    raise ValueError(f"start_node {sn} not in pickup_nodes (vehicle id={vid}).")
                origin = {"kind": "node", "index": sn}
                start_norm = {"kind": "node", "index": sn}

            elif has_sc:
                coord = v["start_coord"]
                if not isinstance(coord, dict) or "lat" not in coord or "lon" not in coord:
                    raise ValueError(f"start_coord must be a dict with lat/lon (vehicle id={vid}).")
                lat = float(coord["lat"])
                lon = float(coord["lon"])
                if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                    raise ValueError(f"start_coord lat/lon out of bounds (vehicle id={vid}).")
                origin = {"kind": "coord", "lat": lat, "lon": lon}
                start_norm = {"kind": "coord", "lat": lat, "lon": lon}

            else:
                # Default: depot 0
                origin = {"kind": "depot", "index": 0}
                start_norm = {"kind": "depot", "index": 0}

            cap_by_bus.append(cap)
            origin_by_bus.append(origin)
            normalized.append({"id": vid, "capacity": cap, "start": start_norm})

        return cap_by_bus, origin_by_bus, buses_count_effective, normalized

    # ---------- Population init ----------


    def _initialize_population(self, population_size, buses_count, bus_capacity,
                               depots, facilities, n_depots, pickup_nodes,
                               max_trips_per_bus, max_stops_per_trip, durations_matrix,
                               demand_full, deadlines,
                               default_evac_center_coords=None, buffer_meters=None,
                               precomputed_problem_data=None, deadline=None):

        from ..evacuation.baselines.pendelverkehr import PendelverkehrShuttleAlgorithm

        population = []

        def deadline_reached():
            return deadline is not None and time.monotonic() >= float(deadline)

        # --- Hybrid Seeding Strategy ---
        print("🌱 Seeding initial population with Pendelverkehr heuristic...")
        num_seeded = 1  # Number of direct seeds from the heuristic
        num_clones = max(2, int(population_size * 0.05)) # 5% of population as mutated clones

        try:
            # 1. Run the Pendelverkehr baseline to get a high-quality seed
            pendel_algo = PendelverkehrShuttleAlgorithm()
            # Pass all relevant parameters
            pendel_params = {
                "vehicles": [{"capacity": c, "start": o} for c, o in zip(self._cap_by_bus, self._origin_by_bus)],
                "start_to_node_seconds": self._start_to_node_seconds,
                "node_coords": self._node_coords,
                "avg_speed_kmh": self._avg_speed_kmh,
                "road_factor": self._road_factor,
                **self._service_params,
                "precomputed_problem_data": precomputed_problem_data,
                # Pass problem data directly to avoid re-initialization
                "evacuation_zones_input": [d for d in depots if d.get('capacity') is not None],
                "buses_count": buses_count,
                "bus_capacity": bus_capacity,
                "default_evac_center_coords": default_evac_center_coords,
                "buffer_meters": buffer_meters,
            }
            # Run the algorithm
            pendel_result = pendel_algo.run(**pendel_params)
            pendel_solution = pendel_result['best_solution']
            self._debug_pendel_solution = pendel_solution
            pendel_individual = self._convert_solution_to_individual(pendel_solution)

            # 2. Add the pure seed(s) to the population
            for _ in range(num_seeded):
                population.append(copy.deepcopy(pendel_individual))

            # 3. Add mutated clones to introduce diversity around the good seed
            print(f"🧬 Creating {num_clones} mutated clones of the seed...")
            for _ in range(num_clones):
                if deadline_reached():
                    break
                clone = copy.deepcopy(pendel_individual)
                # Apply 3-5 rounds of mutation to create significant variation
                for _ in range(random.randint(3, 5)):
                    if deadline_reached():
                        break
                    # FIX: Unpack the tuple (individual, op_name)
                    # We only care about the individual here
                    clone, _ = self._mutate(clone, buses_count, bus_capacity, depots, facilities, n_depots,
                                         pickup_nodes, durations_matrix, demand_full, deadlines)

                if deadline_reached():
                    break
                # Repair the heavily mutated clone to ensure it's valid
                repaired_clone = self._repair(clone, buses_count, bus_capacity, depots, facilities, n_depots,
                                              pickup_nodes, durations_matrix, demand_full, deadlines)
                population.append(repaired_clone)

        except Exception as e:
            print(f"⚠️ Warning: Could not seed population with Pendelverkehr. Proceeding with fully random population. Error: {e}")
            import traceback
            traceback.print_exc()
            population = [] # Clear any partial results if seeding failed

        # 4. Fill the rest of the population with random individuals
        num_to_fill = population_size - len(population)
        print(f"🎲 Filling remaining {num_to_fill} individuals randomly...")

        cap_by_bus = self._cap_by_bus or [bus_capacity for _ in range(buses_count)]
        origin_by_bus = self._origin_by_bus or [{"kind": "depot", "index": 0} for _ in range(buses_count)]

        for _ in range(num_to_fill):
            if deadline_reached():
                break
            individual = [[] for _ in range(buses_count)]
            depot_loads = [0] * n_depots
            remaining_people = {node: demand_full[node] for node in pickup_nodes}
            active_pickup_nodes = set(pickup_nodes)
            bus_order = list(range(buses_count))
            random.shuffle(bus_order)

            for bus_idx in bus_order:
                num_trips = random.randint(1, max_trips_per_bus)
                for trip_idx in range(num_trips):
                    if not active_pickup_nodes: break
                    if trip_idx == 0:
                        start_depot = origin_by_bus[bus_idx]["index"] if origin_by_bus[bus_idx]["kind"] == "depot" else 0
                    else:
                        start_depot = random.randint(0, n_depots - 1)

                    num_stops = min(random.randint(1, max_stops_per_trip), len(active_pickup_nodes))
                    if num_stops == 0: continue
                    stops_nodes = random.sample(list(active_pickup_nodes), num_stops)

                    stops = []
                    remaining_capacity = cap_by_bus[bus_idx]
                    for node in stops_nodes:
                        pickup_count = min(remaining_capacity, remaining_people[node])
                        if pickup_count > 0:
                            stops.append((node, pickup_count))
                            remaining_capacity -= pickup_count
                            remaining_people[node] -= pickup_count
                            if remaining_people[node] == 0:
                                active_pickup_nodes.remove(node)

                    if stops:
                        trip_load = sum(count for _, count in stops)
                        last_stop_node = stops[-1][0]
                        best_depot = self._find_best_end_depot(last_stop_node, trip_load, depot_loads, n_depots, durations_matrix)
                        trip = {"start_depot": start_depot, "stops": stops, "end_depot": best_depot}
                        individual[bus_idx].append(trip)
                        depot_loads[best_depot] += trip_load

                    if not active_pickup_nodes: break
                if not active_pickup_nodes: break

            individual = self._repair(individual, buses_count, bus_capacity, depots, facilities, n_depots,
                                      pickup_nodes, durations_matrix, demand_full, deadlines)
            population.append(individual)

        if not population:
            # A strict budget must still return a complete incumbent. Repairing
            # the empty schedule is the cheapest deterministic fallback.
            fallback = self._repair(
                [[] for _ in range(buses_count)],
                buses_count,
                bus_capacity,
                depots,
                facilities,
                n_depots,
                pickup_nodes,
                durations_matrix,
                demand_full,
                deadlines,
            )
            population.append(fallback)

        return population

    
    def _calculate_depot_loads(self, individual: List[List[Dict[str, Any]]]) -> List[int]:
        """Calculates the total number of people assigned to each depot for a given solution."""
        if not self._depots_runtime:
            return []

        depot_loads = [0] * len(self._depots_runtime)
        for bus_schedule in individual:
            for trip in bus_schedule:
                end_depot = trip.get("end_depot", 0)
                if 0 <= end_depot < len(depot_loads):
                    trip_load = sum(count for _, count in trip.get("stops", []))
                    depot_loads[end_depot] += trip_load
        return depot_loads

    # ---------- Fitness ----------
    def _evaluate_fitness(self, individual, buses_count, bus_capacity, depots, facilities,
                          n_depots, durations_matrix, demand_full, deadlines,
                          penalty_factor, lateness_penalty_factor, latest_evacuation_penalty_factor=0.0):
        """
        Calculates fitness score by calling the central simulation helper.
        """
        sim_results = _simulate_and_get_timings(
            individual=individual,
            n_depots=n_depots,
            durations_matrix=durations_matrix,
            deadlines=deadlines,
            origin_by_bus=self._origin_by_bus,
            cap_by_bus=self._cap_by_bus,
            depots=self._depots_runtime, # Pass depot data for capacity checks
            node_coords=self._node_coords,
            start_to_node_seconds=self._start_to_node_seconds,
            avg_speed_kmh=self._avg_speed_kmh,
            road_factor=self._road_factor,
            **self._service_params,
        )

        travel_time_penalty_factor = 0

        fitness = (
            sim_results["total_wait_pm"] / sim_results["total_people_evacuated"]
            +sim_results["latest_evac_min"]
            + sim_results.get("total_overfill", 0)
        )
        return fitness

    # ---------- Tournament ----------
    def _tournament_selection(self, population, fitness_values, tournament_size):
        tournament_indices = random.sample(range(len(population)), tournament_size)

        best_idx = tournament_indices[0]
        for idx in tournament_indices[1:]:
            if fitness_values[idx] < fitness_values[best_idx]:
                best_idx = idx

        return copy.deepcopy(population[best_idx])

    # ---------- Crossover ----------
    # ----------------------------------------------------------------------------------
    # CROSSOVER ARCHITECTURE (REPLACES THE OLD _crossover METHOD)
    # ----------------------------------------------------------------------------------
    def _crossover(self, parent1: List[List[Dict]], parent2: List[List[Dict]],
                buses_count, bus_capacity, depots, facilities,
                n_depots, durations_matrix, demand_full, deadlines) -> Tuple[List[List[Dict]], List[List[Dict]]]:
        """
        Main crossover dispatcher. Probabilistically selects a specialized crossover operator.
        This hybrid approach balances structured recombination, exploitation, and exploration.
        The offspring are intentionally left in a potentially invalid state (regarding demand),
        as the main EA loop will call the _repair function immediately after this step.
        """
        offspring1 = copy.deepcopy(parent1)
        offspring2 = copy.deepcopy(parent2)

        # Operator probabilities
        prob_ssx = 0.50  # Sub-Schedule Crossover (high probability for structured recombination)
        prob_btix = 0.35 # Best Trip Injection Crossover (for exploiting good trips)
        # prob_tsx = 0.15 # Time-Slice Crossover (for temporal exploration)

        choice = random.random()

        if choice < prob_ssx:
            offspring1, offspring2 = self._crossover_sub_schedule_swap(
                offspring1, offspring2, buses_count)
        elif choice < prob_ssx + prob_btix:
            # BTIX is asymmetric; we run it once in each direction
            offspring1 = self._crossover_best_trip_injection(
                offspring1, offspring2, n_depots, durations_matrix)
            offspring2 = self._crossover_best_trip_injection(
                offspring2, offspring1, n_depots, durations_matrix)
        else:
            offspring1, offspring2 = self._crossover_temporal_slice(
                offspring1, offspring2, n_depots, durations_matrix)

        return offspring1, offspring2

    def _crossover_sub_schedule_swap(self, offspring1: List[List[Dict]],
                                    offspring2: List[List[Dict]],
                                    buses_count: int) -> Tuple[List[List[Dict]], List[List[Dict]]]:
        """
        Sub-Schedule Crossover (SSX)
        Exchanges trip sequences between buses of the *same capacity*. This is a much
        more intelligent and less destructive replacement for the original one-point crossover.
        """
        # Group bus indices by capacity to find valid swap partners
        caps_to_buses1 = {}
        caps_to_buses2 = {}
        for i in range(buses_count):
            cap1 = self._cap_by_bus[i]
            caps_to_buses1.setdefault(cap1, []).append(i)
            cap2 = self._cap_by_bus[i]
            caps_to_buses2.setdefault(cap2, []).append(i)

        # Find capacities that are common to both parents and have buses available
        common_caps = [cap for cap in caps_to_buses1 if cap in caps_to_buses2 and caps_to_buses1[cap] and caps_to_buses2[cap]]
        if not common_caps:
            return offspring1, offspring2  # No compatible buses to swap

        # Select a random capacity group to perform the crossover on
        target_cap = random.choice(common_caps)
        bus1_idx = random.choice(caps_to_buses1[target_cap])
        bus2_idx = random.choice(caps_to_buses2[target_cap])

        schedule1 = offspring1[bus1_idx]
        schedule2 = offspring2[bus2_idx]

        # Only perform crossover if both schedules have trips to swap
        if len(schedule1) > 0 and len(schedule2) > 0:
            point1 = random.randint(0, len(schedule1))
            point2 = random.randint(0, len(schedule2))

            tail1 = schedule1[point1:]
            tail2 = schedule2[point2:]

            offspring1[bus1_idx] = schedule1[:point1] + tail2
            offspring2[bus2_idx] = schedule2[:point2] + tail1

        return offspring1, offspring2

    def _crossover_best_trip_injection(self, recipient: List[List[Dict]],
                                    donor: List[List[Dict]], n_depots: int,
                                    durations_matrix: Dict) -> List[List[Dict]]:
        """
        Best Trip Injection Crossover (BTIX)
        Finds a high-quality trip in the donor and injects it into the best possible
        position in the recipient's schedule. This is an exploitative operator.
        """
        best_trip, best_trip_score = None, -1.0
        cap_by_bus = self._cap_by_bus

        # 1. Find the best trip in the donor parent based on a heuristic
        for bus_schedule in donor:
            for trip in bus_schedule:
                if not trip["stops"]: continue
                # Use a helper to estimate trip duration without a full simulation
                trip_sched = self._estimate_trip_schedule(trip, n_depots, durations_matrix)
                duration = trip_sched["trip_time"]
                people = sum(count for _, count in trip["stops"])
                if duration > 1e-6:
                    # Score: people evacuated per minute
                    score = people / duration
                    if score > best_trip_score:
                        best_trip_score = score
                        best_trip = trip

        if best_trip is None:
            return recipient  # No valid trips found in donor

        # 2. Find the best insertion point in the recipient
        best_insertion_cost = float('inf')
        best_bus_idx, best_trip_pos = -1, -1
        best_trip_load = sum(c for _, c in best_trip.get("stops", []))

        for b_idx, bus_schedule in enumerate(recipient):
            # --- MODIFIED: Capacity Check ---
            # Only consider this bus if it has enough capacity for the injected trip.
            if cap_by_bus[b_idx] < best_trip_load:
                continue
            # --- END MODIFIED ---

            # Estimate cost (total time) before insertion
            initial_timeline = self._estimate_bus_timeline(bus_schedule, b_idx, n_depots, durations_matrix)
            initial_cost = initial_timeline[-1][1] if initial_timeline else 0.0

            for t_pos in range(len(bus_schedule) + 1):
                # Create a temporary schedule with the trip inserted
                temp_schedule = bus_schedule[:t_pos] + [best_trip] + bus_schedule[t_pos:]

                # Estimate cost after insertion
                new_timeline = self._estimate_bus_timeline(temp_schedule, b_idx, n_depots, durations_matrix)
                new_cost = new_timeline[-1][1] if new_timeline else 0.0

                insertion_cost = new_cost - initial_cost
                if insertion_cost < best_insertion_cost:
                    best_insertion_cost = insertion_cost
                    best_bus_idx, best_trip_pos = b_idx, t_pos

        # 3. Perform the injection at the best found location
        if best_bus_idx != -1:
            recipient[best_bus_idx].insert(best_trip_pos, best_trip)

        return recipient

    def _crossover_temporal_slice(self, offspring1: List[List[Dict]],
                                offspring2: List[List[Dict]], n_depots: int,
                                durations_matrix: Dict) -> Tuple[List[List[Dict]], List[List[Dict]]]:
        """
        Time-Slice Crossover (TSX)
        Selects a random time window, swaps all trips within that window between parents,
        and re-inserts them. This is an exploratory, temporally-focused operator.
        """
        # 1. Estimate timelines for both parents
        timeline1 = { (b, t): self._estimate_bus_timeline(sched, b, n_depots, durations_matrix)[t]
                    for b, sched in enumerate(offspring1) for t in range(len(sched)) }
        timeline2 = { (b, t): self._estimate_bus_timeline(sched, b, n_depots, durations_matrix)[t]
                    for b, sched in enumerate(offspring2) for t in range(len(sched)) }

        if not timeline1 and not timeline2:
            return offspring1, offspring2

        max_time = 0
        if timeline1: max_time = max(max_time, max(ret for _, ret in timeline1.values()))
        if timeline2: max_time = max(max_time, max(ret for _, ret in timeline2.values()))
        if max_time == 0: return offspring1, offspring2

        # 2. Define a random time slice
        slice_start = random.uniform(0, max_time * 0.8)
        slice_duration = random.uniform(max_time * 0.1, max_time * 0.3)
        slice_end = slice_start + slice_duration

        # 3. Identify and collect trips within the slice
        trips1_in_slice, trips2_in_slice = [], []
        indices_to_remove1, indices_to_remove2 = {}, {}

        for (b, t), (start, _) in timeline1.items():
            if slice_start <= start < slice_end:
                trips1_in_slice.append(offspring1[b][t])
                indices_to_remove1.setdefault(b, []).append(t)
        for (b, t), (start, _) in timeline2.items():
            if slice_start <= start < slice_end:
                trips2_in_slice.append(offspring2[b][t])
                indices_to_remove2.setdefault(b, []).append(t)

        # 4. Remove sliced trips from offspring, ensuring to handle index shifts
        for b_idx in sorted(indices_to_remove1.keys(), reverse=True):
            for t_idx in sorted(indices_to_remove1[b_idx], reverse=True):
                del offspring1[b_idx][t_idx]
        for b_idx in sorted(indices_to_remove2.keys(), reverse=True):
            for t_idx in sorted(indices_to_remove2[b_idx], reverse=True):
                del offspring2[b_idx][t_idx]

        # 5. Swap the pools of trips and re-insert them greedily
        # Parent 1 gets trips from Parent 2, and vice-versa
        self._greedy_insert_trips(offspring1, trips2_in_slice, n_depots, durations_matrix)
        self._greedy_insert_trips(offspring2, trips1_in_slice, n_depots, durations_matrix)

        return offspring1, offspring2

    def _greedy_insert_trips(self, individual: List[List[Dict]], trips_to_insert: List[Dict],
                            n_depots: int, durations_matrix: Dict):
        """
        Helper for TSX: Inserts a list of trips into an individual's schedules.
        --- MODIFIED: This operator is now capacity-aware. ---
        """
        cap_by_bus = self._cap_by_bus

        for trip in trips_to_insert:
            trip_load = sum(c for _, c in trip.get("stops", []))
            
            # Filter for buses that can handle this trip
            valid_buses = []
            for b_idx, sched in enumerate(individual):
                if cap_by_bus[b_idx] >= trip_load:
                    timeline = self._estimate_bus_timeline(sched, b_idx, n_depots, durations_matrix)
                    finish_time = timeline[-1][1] if timeline else 0.0
                    valid_buses.append((finish_time, b_idx))

            if not valid_buses:
                continue  # This trip cannot be inserted on any bus, so it's dropped.

            # Find the best bus among the valid candidates
            _, best_bus_idx = min(valid_buses, key=lambda x: x[0])
            individual[best_bus_idx].append(trip)

    # --- CROSSOVER HELPER UTILITIES ---
    def _estimate_trip_schedule(self, trip: Dict, n_depots: int, durations_matrix: Dict,
                                bus_idx: Optional[int] = None,
                                trip_idx: Optional[int] = None) -> Dict:
        """
        Lightweight, standalone trip time estimator. Does not do a full simulation,
        but provides a reasonable proxy for duration. Is origin-aware.
        """
        if not trip or not trip.get("stops"):
            return {"arrival_times": [], "return_time": 0.0, "trip_time": 0.0}

        t = 0.0
        bus_capacity = 80 # default
        if bus_idx is not None and bus_idx < len(self._cap_by_bus):
            bus_capacity = self._cap_by_bus[bus_idx]

        first_stop_node, _ = trip["stops"][0]

        # First leg travel time calculation
        if trip_idx == 0 and bus_idx is not None:
            origin = self._origin_by_bus[bus_idx]
            kind = origin.get("kind")
            if kind == "depot":
                t += durations_matrix.get((origin["index"], n_depots + first_stop_node), float('inf')) / 60.0
            elif kind == "node":
                if origin["index"] != first_stop_node:
                    t += durations_matrix.get((n_depots + origin["index"], n_depots + first_stop_node), float('inf')) / 60.0
            elif kind == "coord":
                # Use the main algorithm's first-leg helper for consistency
                t += self._first_leg_minutes_from_coord(bus_idx, origin["lat"], origin["lon"], first_stop_node, n_depots)
        else:
            # For subsequent trips, start from the specified depot
            start_depot = trip.get("start_depot", 0)
            t += durations_matrix.get((start_depot, n_depots + first_stop_node), float('inf')) / 60.0

        # Service and travel time for all stops
        total_people = 0
        for i, (node, pickup_count) in enumerate(trip["stops"]):
            total_people += pickup_count
            # Simplified service time based on dynamic model parameters
            if self._service_params.get("use_dynamic_service_time", False):
                base = self._service_params.get("service_time_base_min", 3.0)
                per_person = self._service_params.get("service_time_per_person_min", 20.0/60.0)
                #print(base, per_person)
                t += base + pickup_count * per_person
            else: # Fallback to original simple model
                base = self._service_params.get("service_time_base_min", 3.0)
                per_person = self._service_params.get("service_time_per_person_min", 20.0/60.0)
                #print(base, per_person)
                t += base + bus_capacity * per_person

            if i < len(trip["stops"]) - 1:
                next_node, _ = trip["stops"][i + 1]
                t += durations_matrix.get((n_depots + node, n_depots + next_node), float('inf')) / 60.0

        # Return leg to end depot
        last_node, _ = trip["stops"][-1]
        t += durations_matrix.get((n_depots + last_node, trip.get("end_depot", 0)), float('inf')) / 60.0

        # Offloading service time
        if self._service_params.get("use_dynamic_service_time", False):
            base = self._service_params.get("service_time_base_min", 3.0)
            per_person = self._service_params.get("service_time_per_person_min", 20.0/60.0)
            t += base + total_people * per_person
        else:
            t += 4.0 + 6.0 * (bus_capacity / 40.0)

        # Handle potential infinite travel times
        if not math.isfinite(t): t = 9999.0

        return {"trip_time": t} # Only duration is needed for these heuristics

    def _estimate_bus_timeline(self, bus_schedule: List[Dict], bus_idx: int,
                            n_depots: int, durations_matrix: Dict) -> List[Tuple[float, float]]:
        """
        Estimates the start and end time for each trip in a bus's schedule.
        """
        timeline = []
        current_time = 0.0
        for trip_idx, trip in enumerate(bus_schedule):
            # This is critical: the trip's start depot for calculation must be the previous trip's end depot
            if trip_idx > 0 and timeline:
                # Make a copy to avoid modifying the individual directly during estimation
                trip_copy = copy.deepcopy(trip)
                trip_copy["start_depot"] = bus_schedule[trip_idx - 1]["end_depot"]
                trip_to_eval = trip_copy
            else:
                trip_to_eval = trip

            trip_sched = self._estimate_trip_schedule(
                trip_to_eval, n_depots, durations_matrix, bus_idx=bus_idx, trip_idx=trip_idx)

            trip_duration = trip_sched.get("trip_time", 0.0)
            trip_start = current_time
            trip_end = current_time + trip_duration
            timeline.append((trip_start, trip_end))
            current_time = trip_end
        return timeline


    # ---------- Mutation ----------
    def _mutation_spatial_ruin_recreate(self, individual, buses_count, n_depots, durations_matrix):
        """
        Spatial Ruin & Recreate:
        1. Pick a random 'victim' node.
        2. Remove it and its N nearest neighbors from all trips.
        3. Re-insert them into the best possible positions (Greedy Insertion).
        """
        mutated = copy.deepcopy(individual)
        
        # 1. Select Center of Ruin
        if not self._node_coords: return mutated # Safety
        
        victim_node = random.choice(list(self._node_coords.keys()))
        
        # 2. Find Neighbors (Spatial)
        # Simple Euclidean or Matrix lookup. Let's use Matrix for accuracy.
        neighbors = []
        for other_node in self._node_coords.keys():
            if other_node == victim_node: continue
            dist = durations_matrix.get((n_depots + victim_node, n_depots + other_node), float('inf'))
            neighbors.append((dist, other_node))
        
        neighbors.sort(key=lambda x: x[0])
        # Ruin 15% of nodes or max 10 nodes
        num_to_remove = min(10, max(3, int(len(self._node_coords) * 0.15)))
        targets = {victim_node} | {n for _, n in neighbors[:num_to_remove]}
        
        removed_stops = [] # List of (node, count)

        # 3. Ruin (Remove)
        for b in range(len(mutated)):
            for trip in mutated[b]:
                new_stops = []
                for node, count in trip.get("stops", []):
                    if node in targets:
                        removed_stops.append((node, count))
                    else:
                        new_stops.append((node, count))
                trip["stops"] = new_stops
            # Clean empty trips
            mutated[b] = [t for t in mutated[b] if t["stops"]]

        # Shuffle removed stops to avoid bias
        random.shuffle(removed_stops)

        # 4. Recreate (Greedy Insertion)
        # For every removed stop, try to insert it into every position of every trip
        for node, count in removed_stops:
            best_cost = float('inf')
            best_insertion = None # (bus_idx, trip_idx, stop_idx)
            
            # Try inserting into existing trips
            for b in range(len(mutated)):
                cap = self._cap_by_bus[b]
                for t_idx, trip in enumerate(mutated[b]):
                    # Cap check
                    current_load = sum(c for _, c in trip["stops"])
                    if current_load + count > cap: continue
                    
                    # Try every position
                    stops = trip["stops"]
                    for i in range(len(stops) + 1):
                        # Calculate local delta cost (detour)
                        # Prev node -> inserted node -> next node
                        prev_loc = n_depots + stops[i-1][0] if i > 0 else trip["start_depot"]
                        next_loc = n_depots + stops[i][0] if i < len(stops) else trip["end_depot"]
                        node_loc = n_depots + node
                        
                        # Simple detour calculation (Triangle inequality)
                        # Cost = (Prev->Node + Node->Next) - (Prev->Next)
                        added = durations_matrix.get((prev_loc, node_loc), 0) + durations_matrix.get((node_loc, next_loc), 0)
                        removed = durations_matrix.get((prev_loc, next_loc), 0)
                        detour = added - removed
                        
                        if detour < best_cost:
                            best_cost = detour
                            best_insertion = (b, t_idx, i)

            # If found a place, insert
            if best_insertion:
                b, t, i = best_insertion
                mutated[b][t]["stops"].insert(i, (node, count))
            else:
                # Create a trip if no fit
                # Simple logic: append to random bus with capacity or bus 0
                # (This part relies on _repair to fix depots/optimization later)
                target_b = random.randint(0, len(mutated)-1)
                if self._cap_by_bus[target_b] >= count:
                     mutated[target_b].append({
                         "start_depot": 0, # Will be fixed by repair
                         "stops": [(node, count)],
                         "end_depot": 0 
                     })

        # 5. Repair Connectivity
        for b in range(len(mutated)):
            mutated[b] = self._fix_depot_connectivity(mutated[b], origin=self._origin_by_bus[b])

        return mutated
    
    def _mutate(self, individual, buses_count, bus_capacity, depots, facilities,
                n_depots, pickup_nodes, durations_matrix, demand_full, deadlines) -> Tuple[List[List[Dict]], str]:

        mutated = copy.deepcopy(individual)
        cap_by_bus = self._cap_by_bus or [bus_capacity for _ in range(buses_count)]

        # --- WEIGHTED OPERATOR SELECTION ---
        # Based on Telemetry: favor Intra-Swap (Polish) and Relocate (Balance).
        # Keep Spatial Ruin high enough to break local optima.
        ops = [1, 2, 3, 4, 5, 6]
        weights = [
            0.30,  # 1. Intra-Swap (The King: creates 2-opt efficiencies)
            0.20,  # 2. Relocate Stop (The Balancer: moves crumbs between buses)
            0.15,  # 3. Add/Remove Trip (The Architect: fixes structure)
            0.10,  # 4. Change Depot (The Navigator: fixes Depot 4 issues)
            0.10,  # 5. Swap Trip (Load balancing)
            0.15   # 6. Spatial Ruin (The Battering Ram: prevents bad runs)
        ]
        
        mutation_op = random.choices(ops, weights=weights, k=1)[0]
        op_name = "unknown"

        if mutation_op == 1:
            op_name = "mutate_intra_swap"
            # Stop reordering within a trip
            non_empty_buses = [i for i, trips in enumerate(mutated) if trips]
            if non_empty_buses:
                bus_idx = random.choice(non_empty_buses)
                non_empty_trips = [i for i, trip in enumerate(mutated[bus_idx]) if trip["stops"]]
                if non_empty_trips:
                    trip_idx = random.choice(non_empty_trips)
                    stops = mutated[bus_idx][trip_idx]["stops"]
                    if len(stops) >= 2:
                        idx1, idx2 = random.sample(range(len(stops)), 2)
                        stops[idx1], stops[idx2] = stops[idx2], stops[idx1]

        elif mutation_op == 2:
            op_name = "mutate_relocate_stop"
            # Stop relocation between buses (smarter version)
            source_candidates = []
            for b_idx, bus_schedule in enumerate(mutated):
                for t_idx, trip in enumerate(bus_schedule):
                    if trip.get("stops"):
                         source_candidates.append((b_idx, t_idx))
            
            if source_candidates and buses_count > 1:
                source_bus_idx, source_trip_idx = random.choice(source_candidates)
                source_stops = mutated[source_bus_idx][source_trip_idx]["stops"]

                if source_stops:
                    stop_idx = random.randrange(len(source_stops))
                    stop_to_move = source_stops.pop(stop_idx)
                    _, people_to_move = stop_to_move

                    target_candidates = []
                    for b_idx in range(buses_count):
                        if b_idx == source_bus_idx: continue
                        current_cap = cap_by_bus[b_idx]
                        for t_idx, trip in enumerate(mutated[b_idx]):
                            current_load = sum(c for _, c in trip.get("stops", []))
                            if current_cap >= current_load + people_to_move:
                                target_candidates.append((b_idx, t_idx))
                    
                    if target_candidates:
                        target_bus_idx, target_trip_idx = random.choice(target_candidates)
                        insert_pos = random.randint(0, len(mutated[target_bus_idx][target_trip_idx]["stops"]))
                        mutated[target_bus_idx][target_trip_idx]["stops"].insert(insert_pos, stop_to_move)
                        if not mutated[source_bus_idx][source_trip_idx]["stops"]:
                            mutated[source_bus_idx].pop(source_trip_idx)
                    else:
                        source_stops.insert(stop_idx, stop_to_move)

        elif mutation_op == 3:
            op_name = "mutate_add_remove_trip"
            # Trip addition or removal
            if random.random() < 0.5:
                # Add trip
                bus_idx = random.randint(0, buses_count - 1)
                pickup_demands = {}
                for node in pickup_nodes:
                    total_picked = 0
                    for bus_trips in mutated:
                        for trip in bus_trips:
                            for stop_node, pickup_count in trip["stops"]:
                                if stop_node == node:
                                    total_picked += pickup_count
                    remaining_demand = max(0, demand_full[node] - total_picked)
                    if remaining_demand > 0:
                        pickup_demands[node] = remaining_demand

                if pickup_demands:
                    available_nodes = list(pickup_demands.keys())
                    num_stops = min(random.randint(1, 3), len(available_nodes))
                    selected_nodes = random.sample(available_nodes, num_stops)
                    stops = []
                    remaining_capacity = cap_by_bus[bus_idx]
                    for node in selected_nodes:
                        if remaining_capacity > 0:
                            pickup_amount = min(pickup_demands[node], remaining_capacity)
                            pickup_amount = max(1, pickup_amount)
                            stops.append((node, pickup_amount))
                            remaining_capacity -= pickup_amount

                    if stops:
                        depot_loads = self._calculate_depot_loads(mutated)
                        trip_load = sum(count for _, count in stops)
                        last_stop_node = stops[-1][0]
                        best_depot = self._find_best_end_depot(last_stop_node, trip_load, depot_loads, n_depots, durations_matrix)
                        new_trip = {
                            "start_depot": random.randint(0, n_depots - 1),
                            "stops": stops,
                            "end_depot": best_depot,
                        }
                        mutated[bus_idx].append(new_trip)
            else:
                # Remove trip
                non_empty_buses = [i for i, trips in enumerate(mutated) if trips]
                if non_empty_buses:
                    bus_idx = random.choice(non_empty_buses)
                    if mutated[bus_idx]:
                        trip_idx = random.randint(0, len(mutated[bus_idx]) - 1)
                        mutated[bus_idx].pop(trip_idx)

        elif mutation_op == 4:
            op_name = "mutate_change_depot"
            # Depot change
            if n_depots > 1:
                non_empty_buses = [i for i, trips in enumerate(mutated) if trips]
                if non_empty_buses:
                    bus_idx = random.choice(non_empty_buses)
                    if mutated[bus_idx]:
                        trip_idx = random.randint(0, len(mutated[bus_idx]) - 1)
                        original_end_depot = mutated[bus_idx][trip_idx].get("end_depot", 0)
                        possible_new_depots = [d for d in range(n_depots) if d != original_end_depot]
                        if possible_new_depots:
                            mutated[bus_idx][trip_idx]["end_depot"] = random.choice(possible_new_depots)

        elif mutation_op == 5:
            op_name = "mutate_swap_trip"
            # Trip swapping between buses
            buses_with_trips = [i for i, trips in enumerate(mutated) if trips]
            if len(buses_with_trips) >= 2:
                bus1_idx, bus2_idx = random.sample(buses_with_trips, 2)
                trip1_idx = random.randint(0, len(mutated[bus1_idx]) - 1)
                trip2_idx = random.randint(0, len(mutated[bus2_idx]) - 1)
                trip1 = mutated[bus1_idx][trip1_idx]
                trip2 = mutated[bus2_idx][trip2_idx]

                load1 = sum(c for _, c in trip1.get("stops", []))
                load2 = sum(c for _, c in trip2.get("stops", []))
                cap1 = cap_by_bus[bus1_idx]
                cap2 = cap_by_bus[bus2_idx]

                if load1 <= cap2 and load2 <= cap1:
                    mutated[bus1_idx][trip1_idx] = trip2
                    mutated[bus2_idx][trip2_idx] = trip1

        elif mutation_op == 6:
            op_name = "mutate_spatial_ruin"
            # Spatial Ruin & Recreate
            mutated = self._mutation_spatial_ruin_recreate(mutated, buses_count, n_depots, durations_matrix)

        # Final Cleanup & Connectivity
        origin_by_bus = self._origin_by_bus or [{"kind": "depot", "index": 0} for _ in range(buses_count)]
        for bus_idx in range(buses_count):
            # Filter checks
            mutated[bus_idx] = [t for t in mutated[bus_idx] if isinstance(t, dict) and t.get("stops")]
            if mutated[bus_idx]:
                self._fix_depot_connectivity(mutated[bus_idx], origin=origin_by_bus[bus_idx])

        return mutated, op_name


    def _fix_depot_connectivity(self, bus_trips: List[Dict], origin: Optional[Dict[str, Any]] = None):
        """
        Ensures the first trip starts at the correct origin and subsequent trips 
        chain correctly (start where the previous one ended).
        """
        if not bus_trips:
            return []

        # Resolve origin (default to depot 0 if missing)
        if origin is None:
            origin_kind = "depot"
            origin_index = 0
        else:
            origin_kind = origin.get("kind", "depot")
            origin_index = origin.get("index", 0)

        # --- 1. Fix the First Trip ---
        first_trip = bus_trips[0]
        if origin_kind == "depot":
            # If bus starts at a specific depot, Trip 0 MUST start there
            first_trip["start_depot"] = int(origin_index)
        elif origin_kind in {"node", "coord"}:
            # For node/coord origins, start_depot is usually 0 (proxy) or irrelevant
            if "start_depot" not in first_trip or first_trip["start_depot"] is None:
                first_trip["start_depot"] = 0
        else:
            first_trip["start_depot"] = 0

        # --- 2. Chain subsequent trips ---
        for i in range(1, len(bus_trips)):
            prev_end = bus_trips[i-1]["end_depot"]
            bus_trips[i]["start_depot"] = prev_end

        return bus_trips


    def _repair(self, individual, buses_count, bus_capacity, depots, facilities,
                    n_depots, pickup_nodes, durations_matrix, demand_full, deadlines):
        """
        Repairs an individual to be a valid solution.
        CONTEXT-AWARE: Tracks depot loads, splits over-capacity trips, and respects vehicle origins.
        """
        repaired = copy.deepcopy(individual)
        
        # Resolve fleet details locally
        cap_by_bus = self._cap_by_bus or [bus_capacity for _ in range(buses_count)]
        origin_by_bus = self._origin_by_bus or [{"kind": "depot", "index": 0} for _ in range(buses_count)]

        # --- Phase 1: Sanitize stops (remove empty/invalid entries) ---
        for bus_idx in range(len(repaired)):
            for trip_idx in range(len(repaired[bus_idx])):
                sanitized = []
                for node, count in repaired[bus_idx][trip_idx].get("stops", []):
                    try:
                        c = int(count)
                        if c > 0: sanitized.append((node, c))
                    except (ValueError, TypeError): pass
                repaired[bus_idx][trip_idx]["stops"] = sanitized
                repaired[bus_idx][trip_idx] = self._coalesce_duplicate_stops_in_trip(repaired[bus_idx][trip_idx])
            repaired[bus_idx] = [trip for trip in repaired[bus_idx] if trip.get("stops")]

        # --- Phase 2: Handle Over-Servicing (Picking up too many people) ---
        picked_up = {node: 0 for node in pickup_nodes}
        for bus_trips in repaired:
            for trip in bus_trips:
                for node, pickup_count in trip.get("stops", []):
                    if node in picked_up: picked_up[node] += pickup_count

        for node in pickup_nodes:
            demand = demand_full.get(node, 0)
            over_pickup = picked_up.get(node, 0) - demand
            if over_pickup > 0:
                visiting = []
                for b, bus_trips in enumerate(repaired):
                    for t, trip in enumerate(bus_trips):
                        for s, (sn, sc) in enumerate(trip.get("stops", [])):
                            if sn == node and sc > 0: visiting.append((b, t, s, sc))
                
                visiting.sort(key=lambda x: (x[0], x[1]), reverse=True)
                for b, t, s, sc in visiting:
                    if over_pickup <= 0: break
                    reduction = min(over_pickup, sc)
                    repaired[b][t]["stops"][s] = (node, sc - reduction)
                    over_pickup -= reduction

        # Clean up any stops that became 0
        for b_idx in range(len(repaired)):
            for t_idx in range(len(repaired[b_idx])):
                repaired[b_idx][t_idx]["stops"] = [(n, c) for n, c in repaired[b_idx][t_idx].get("stops", []) if c > 0]
            repaired[b_idx] = [trip for trip in repaired[b_idx] if trip.get("stops")]

        depot_loads = self._calculate_depot_loads(repaired)

        # --- Phase 2.5: STRICT per-trip capacity enforcement (Split Overflow) ---
        for b in range(len(repaired)):
            cap_b = cap_by_bus[b]
            t = 0
            while t < len(repaired[b]):
                trip = repaired[b][t]
                load = sum(c for _, c in trip.get("stops", []))
                
                if load <= cap_b:
                    t += 1
                    continue
                
                # Handle overflow
                original_end_depot = trip.get("end_depot", 0)
                if 0 <= original_end_depot < len(depot_loads):
                    depot_loads[original_end_depot] -= load
                
                overflow_stops = []
                trip["stops"].reverse() # Process from end
                
                while sum(c for _,c in trip["stops"]) > cap_b:
                    node, cnt = trip["stops"][0]
                    overflow = sum(c for _,c in trip["stops"]) - cap_b
                    take = min(cnt, overflow)
                    
                    if take > 0: overflow_stops.append((node, take))
                    
                    if cnt - take > 0:
                        trip["stops"][0] = (node, cnt - take)
                    else:
                        trip["stops"].pop(0)

                trip["stops"].reverse() # Restore order
                
                new_load = sum(c for _, c in trip.get("stops", []))
                if 0 <= original_end_depot < len(depot_loads):
                    depot_loads[original_end_depot] += new_load

                # Assign overflow to the bus with least trips
                bus_trip_counts = [len(s) for s in repaired]
                target_bus_idx = bus_trip_counts.index(min(bus_trip_counts))
                
                for node, qty in overflow_stops:
                    remaining_qty = qty
                    while remaining_qty > 0:
                        target_bus_cap = cap_by_bus[target_bus_idx]
                        take = min(target_bus_cap, remaining_qty)
                        best_depot = self._find_best_end_depot(node, take, depot_loads, n_depots, durations_matrix)
                        
                        # --- CRITICAL FIX START ---
                        # If the target bus has no trips, look up its specific origin
                        if repaired[target_bus_idx]:
                            start_depot = repaired[target_bus_idx][-1]["end_depot"]
                        else:
                            origin = origin_by_bus[target_bus_idx]
                            start_depot = origin['index'] if origin.get('kind') == 'depot' else 0
                        # --- CRITICAL FIX END ---
                        
                        new_trip = {"start_depot": start_depot, "stops": [(node, take)], "end_depot": best_depot}
                        repaired[target_bus_idx].append(new_trip)
                        
                        if 0 <= best_depot < len(depot_loads):
                            depot_loads[best_depot] += take
                        remaining_qty -= take

                if not trip.get("stops"):
                    repaired[b].pop(t)
                else:
                    t += 1
                    
        # --- Phase 3: Correct Under-servicing (Satisfy remaining demand) ---
        picked_up_after = {node: 0 for node in pickup_nodes}
        for bus_trips in repaired:
            for trip in bus_trips:
                for node, c in trip.get("stops", []):
                    picked_up_after[node] += c

        for node in pickup_nodes:
            remaining = demand_full.get(node, 0) - picked_up_after.get(node, 0)
            while remaining > 0:
                bus_trip_counts = [len(s) for s in repaired]
                target_b = bus_trip_counts.index(min(bus_trip_counts))
                cap_b = cap_by_bus[target_b]
                take = min(cap_b, remaining)
                
                best_depot = self._find_best_end_depot(node, take, depot_loads, n_depots, durations_matrix)
                

                if repaired[target_b]:
                    start_depot = repaired[target_b][-1]["end_depot"]
                else:
                    origin = origin_by_bus[target_b]
                    start_depot = origin['index'] if origin.get('kind') == 'depot' else 0

                
                new_trip = {"start_depot": start_depot, "stops": [(node, take)], "end_depot": best_depot}
                repaired[target_b].append(new_trip)
                
                if 0 <= best_depot < len(depot_loads):
                    depot_loads[best_depot] += take
                remaining -= take

        # --- Phase 4: Final Connectivity Check ---
        for b in range(buses_count):
            if repaired[b]:
                repaired[b] = self._fix_depot_connectivity(repaired[b], origin=origin_by_bus[b])
        
        return repaired


    def _coalesce_duplicate_stops_in_trip(self, trip):
        original_stops = trip.get("stops", [])
        order = []
        sums = {}
        for node, count in original_stops:
            if node not in sums:
                order.append(node)
                sums[node] = 0
            sums[node] += max(0, int(count))
        trip["stops"] = [(n, sums[n]) for n in order]
        return trip

    # ---------- Solution formatting ----------
    def _individual_to_solution(self, individual, buses_count, depots, n_depots):
        solution = []
        for bus_idx in range(buses_count):
            bus_trips = []
            for trip in individual[bus_idx]:
                if not trip["stops"]:
                    continue
                node_indices = []
                pickup_counts = {}
                for node, count in trip["stops"]:
                    node_indices.append(node)
                    pickup_counts[node] = count
                formatted_trip = {
                    "start_depot": trip["start_depot"],
                    "stops": node_indices,
                    "end_depot": trip["end_depot"],
                    "pickup_counts": pickup_counts
                }
                bus_trips.append(formatted_trip)
            solution.append(bus_trips)
        return solution

    def _convert_solution_to_individual(self, solution: List[List[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
        """Converts a solution format (from baseline/ILP) to the EA's individual format."""
        individual = []
        for bus_schedule in solution:
            individual_bus_schedule = []
            for trip in bus_schedule:
                stops_with_counts = []
                # Ensure pickup_counts and stops exist and are iterable
                pickup_counts = trip.get("pickup_counts", {})
                stop_nodes = trip.get("stops", [])

                for node_idx in stop_nodes:
                    count = pickup_counts.get(node_idx, 0)
                    if count > 0:
                        stops_with_counts.append((node_idx, count))

                if stops_with_counts: # Only add trips that have pickups
                    individual_trip = {
                        "start_depot": trip.get("start_depot", 0),
                        "stops": stops_with_counts,
                        "end_depot": trip.get("end_depot", 0)
                    }
                    individual_trip = self._coalesce_duplicate_stops_in_trip(individual_trip)
                    if individual_trip.get("stops"):
                        individual_bus_schedule.append(individual_trip)
            individual.append(individual_bus_schedule)
        return individual

    # ---------- Metrics extraction ----------
    def _extract_and_log_generation_metrics(self, generation: int, population, fitness_values,
                                          buses_count, bus_capacity, depots, facilities,
                                          n_depots, durations_matrix, demand_full, deadlines,
                                          algorithm_stats: Dict[str, Any]):
        """
        Extracts metrics for the best individual by calling the central simulation helper.
        This guarantees consistency between logged metrics and the fitness function.
        """
        try:
            best_idx = fitness_values.index(min(fitness_values))
            best_individual = population[best_idx]
            best_fitness = fitness_values[best_idx]

            sim_results = _simulate_and_get_timings(
                individual=best_individual,
                n_depots=n_depots,
                durations_matrix=durations_matrix,
                deadlines=deadlines,
                origin_by_bus=self._origin_by_bus,
                cap_by_bus=self._cap_by_bus,
                depots=self._depots_runtime,
                node_coords=self._node_coords,
                start_to_node_seconds=self._start_to_node_seconds,
                avg_speed_kmh=self._avg_speed_kmh,
                road_factor=self._road_factor,
                **self._service_params,
            )

            pickup_times = sim_results["pickup_times"]
            return_times = sim_results["return_times"]
            total_people_evacuated = sim_results["total_people_evacuated"]

            metrics = {}
            if pickup_times:
                avg_evacuation_time = sum(pickup_times) / len(pickup_times)
                metrics["gen_avg_evacuation_time"] = avg_evacuation_time
                metrics["gen_min_pickup_time"] = min(pickup_times)
                metrics["gen_max_pickup_time"] = max(pickup_times)
                algorithm_stats["generation_avg_evacuation_times"].append(avg_evacuation_time)
            else:
                metrics["gen_avg_evacuation_time"] = float('inf')
                algorithm_stats["generation_avg_evacuation_times"].append(float('inf'))

            if return_times:
                latest_evacuation_time = max(return_times)
                metrics["gen_latest_evacuation_time"] = latest_evacuation_time
                metrics["gen_min_return_time"] = min(return_times)
                algorithm_stats["generation_latest_evacuation_times"].append(latest_evacuation_time)
            else:
                metrics["gen_latest_evacuation_time"] = float('inf')
                algorithm_stats["generation_latest_evacuation_times"].append(float('inf'))

            metrics["gen_algorithm_cost"] = best_fitness
            metrics["gen_total_people_evacuated"] = total_people_evacuated
            metrics["gen_avg_fitness"] = np.mean(fitness_values)
            metrics["gen_fitness_std"] = np.std(fitness_values) if len(fitness_values) > 1 else 0

            algorithm_stats["generation_total_people_evacuated"].append(total_people_evacuated)
            algorithm_stats["generation_fitness_std"].append(metrics["gen_fitness_std"])

            if pickup_times and total_people_evacuated > 0:
                evacuation_efficiency = total_people_evacuated / max(pickup_times)
                metrics["gen_evacuation_efficiency"] = evacuation_efficiency
                algorithm_stats["generation_evacuation_efficiencies"].append(evacuation_efficiency)
            else:
                algorithm_stats["generation_evacuation_efficiencies"].append(0)

            if return_times:
                avg_trip_duration = np.mean(return_times)
                metrics["gen_avg_trip_duration"] = avg_trip_duration

            unique_fitness_count = len(set(fitness_values))
            diversity = unique_fitness_count / len(fitness_values)
            metrics["gen_population_diversity"] = diversity
            algorithm_stats["generation_population_diversity"].append(diversity)

            log_generation_metrics(generation, metrics)
            return metrics

        except Exception as e:
            import traceback
            print(f"Warning: Could not extract generation metrics for generation {generation}: {e}")
            traceback.print_exc()
            return {}

    # ---------- First-leg helpers ----------
    def _first_leg_minutes_from_coord(
        self,
        bus_idx: int,
        lat: float,
        lon: float,
        first_node: int,
        n_depots: int
    ) -> float:
        """
        Compute minutes from arbitrary coordinate (lat, lon) to pickup `first_node`.
        Uses override map if provided; otherwise uses haversine -> road_factor -> avg_speed.
        """
        # Override map takes precedence
        if self._start_to_node_seconds and bus_idx in self._start_to_node_seconds:
            secs_map = self._start_to_node_seconds[bus_idx]
            if first_node in secs_map:
                return float(secs_map[first_node]) / 60.0

        # Fallback: haversine to node coords
        if not self._node_coords:
            raise RuntimeError(
                "node_coords not available in problem_data; provide start_to_node_seconds override for first-leg times."
            )
        if first_node not in self._node_coords:
            raise RuntimeError(f"Missing coordinates for node {first_node} in node_coords.")

        nlat, nlon = self._node_coords[first_node]
        km = self._haversine_km(lat, lon, nlat, nlon)
        minutes = (km / max(1e-6, self._avg_speed_kmh)) * 60.0 * self._road_factor
        return minutes

    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371.0
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c
    
    def _print_distance_matrix(self, durations_matrix, n_depots):
        if not durations_matrix:
            print("  (Distance matrix unavailable)")
            return

        indices = set()
        for i, j in durations_matrix:
            indices.add(i)
            indices.add(j)
        if not indices:
            print("  (Distance matrix unavailable)")
            return

        total_nodes = max(indices) + 1
        labels = []
        legend = []
        for idx in range(total_nodes):
            if idx < n_depots:
                short = f"D{idx}"
                name = None
                if self._depots_runtime and idx < len(self._depots_runtime):
                    name = self._depots_runtime[idx].get("label")
                if not name:
                    name = f"Depot {idx}"
            else:
                node_idx = idx - n_depots
                short = f"N{node_idx}"
                name = None
                if self._facilities_runtime and node_idx < len(self._facilities_runtime):
                    name = self._facilities_runtime[node_idx].get("label")
                if not name:
                    name = f"Node {node_idx}"
            if name and len(name) > 48:
                name = name[:45] + "..."
            labels.append(short)
            legend.append(f"{short}: {name}")

        max_minutes = 0.0
        for val in durations_matrix.values():
            if val is None or not math.isfinite(val):
                continue
            max_minutes = max(max_minutes, val / 60.0)
        val_width = len(f"{max_minutes:.1f}") if max_minutes > 0 else len("0.0")
        col_width = max(val_width, max(len(l) for l in labels), len("inf"))

        header = " " * (col_width + 1) + " ".join(l.rjust(col_width) for l in labels)
        print(header)
        for i in range(total_nodes):
            row_vals = []
            for j in range(total_nodes):
                val = durations_matrix.get((i, j))
                if val is None or not math.isfinite(val):
                    cell = "inf"
                else:
                    cell = f"{(val / 60.0):.1f}"
                row_vals.append(cell.rjust(col_width))
            print(f"{labels[i].rjust(col_width)} " + " ".join(row_vals))

        print("\nLabel Legend:")
        for entry in legend:
            print(f"  {entry}")

    def _format_origin_label(self, bus_idx: int) -> str:
        origin = None
        if self._origin_by_bus and bus_idx < len(self._origin_by_bus):
            origin = self._origin_by_bus[bus_idx]
        if not origin:
            origin = {"kind": "depot", "index": 0}

        kind = origin.get("kind", "depot")
        if kind == "depot":
            idx = int(origin.get("index", 0))
            label = None
            if self._depots_runtime and idx < len(self._depots_runtime):
                label = self._depots_runtime[idx].get("label")
            return f"Depot {idx} ({label})" if label else f"Depot {idx}"
        if kind == "node":
            idx = int(origin.get("index", 0))
            label = None
            if self._facilities_runtime and idx < len(self._facilities_runtime):
                label = self._facilities_runtime[idx].get("label")
            return f"Node {idx} ({label})" if label else f"Node {idx}"
        if kind == "coord":
            lat = origin.get("lat")
            lon = origin.get("lon")
            if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
                return f"Coord({lat:.5f}, {lon:.5f})"
            return "Coord"
        return "Depot 0"

    def _first_leg_minutes_for_bus(self, bus_idx, bus_trips, n_depots, durations_matrix):
        if not bus_trips:
            return None
        first_trip = bus_trips[0]
        stops = first_trip.get("stops", [])
        if not stops:
            return None
        first_node = stops[0][0] if isinstance(stops[0], (list, tuple)) else stops[0]
        try:
            first_node = int(first_node)
        except Exception:
            return None

        origin = None
        if self._origin_by_bus and bus_idx < len(self._origin_by_bus):
            origin = self._origin_by_bus[bus_idx]
        if not origin:
            origin = {"kind": "depot", "index": 0}

        kind = origin.get("kind", "depot")
        if kind == "depot":
            start = int(origin.get("index", 0))
            val = durations_matrix.get((start, n_depots + first_node))
            return (val / 60.0) if (val is not None and math.isfinite(val)) else None
        if kind == "node":
            start_node = int(origin.get("index", 0))
            if start_node == first_node:
                return 0.0
            val = durations_matrix.get((n_depots + start_node, n_depots + first_node))
            return (val / 60.0) if (val is not None and math.isfinite(val)) else None
        if kind == "coord":
            lat = origin.get("lat")
            lon = origin.get("lon")
            if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
                return None
            try:
                return self._first_leg_minutes_from_coord(bus_idx, float(lat), float(lon), int(first_node), n_depots)
            except Exception:
                return None
        return None

    def _origin_to_node_minutes(self, bus_idx, node_idx, n_depots, durations_matrix):
        origin = None
        if self._origin_by_bus and bus_idx < len(self._origin_by_bus):
            origin = self._origin_by_bus[bus_idx]
        if not origin:
            origin = {"kind": "depot", "index": 0}

        kind = origin.get("kind", "depot")
        if kind == "depot":
            start = int(origin.get("index", 0))
            val = durations_matrix.get((start, n_depots + node_idx))
            if val is None or not math.isfinite(val):
                return float("inf")
            return val / 60.0
        if kind == "node":
            start_node = int(origin.get("index", 0))
            val = durations_matrix.get((n_depots + start_node, n_depots + node_idx))
            if val is None or not math.isfinite(val):
                return float("inf")
            return val / 60.0
        if kind == "coord":
            lat = origin.get("lat")
            lon = origin.get("lon")
            if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
                return None
            try:
                return self._first_leg_minutes_from_coord(bus_idx, float(lat), float(lon), int(node_idx), n_depots)
            except Exception:
                return None
        return None

    def _print_first_leg_matrix(self, solution, n_depots, durations_matrix):
        if not solution:
            print("  (No solution data)")
            return

        bus_indices = [b for b, trips in enumerate(solution) if trips]
        if not bus_indices:
            print("  (No active buses)")
            return

        n_nodes = None
        if self._facilities_runtime is not None:
            n_nodes = len(self._facilities_runtime)
        if n_nodes is None:
            indices = set()
            for i, j in durations_matrix:
                indices.add(i)
                indices.add(j)
            if indices:
                total_nodes = max(indices) + 1
                n_nodes = max(0, total_nodes - n_depots)
        if not n_nodes:
            print("  (No pickup nodes)")
            return

        def format_bus_list(buses):
            if not buses:
                return ""
            ranges = []
            start = prev = buses[0]
            for b in buses[1:]:
                if b == prev + 1:
                    prev = b
                    continue
                ranges.append((start, prev))
                start = prev = b
            ranges.append((start, prev))
            parts = []
            for a, b in ranges:
                if a == b:
                    parts.append(f"B{a}")
                else:
                    parts.append(f"B{a}-{b}")
            return ",".join(parts)

        labels = [f"N{i}" for i in range(n_nodes)]
        groups = {}
        max_cell_width = 0

        for b_idx in bus_indices:
            origin_label = self._format_origin_label(b_idx)
            row_cells = []
            for node_idx in range(n_nodes):
                val = self._origin_to_node_minutes(b_idx, node_idx, n_depots, durations_matrix)
                if val is None:
                    cell = "N/A"
                elif not math.isfinite(val):
                    cell = "inf"
                else:
                    cell = f"{val:.1f}"
                row_cells.append(cell)
                max_cell_width = max(max_cell_width, len(cell))

            key = (origin_label, tuple(row_cells))
            entry = groups.get(key)
            if entry is None:
                groups[key] = {
                    "buses": [b_idx],
                    "row": row_cells,
                    "origin": origin_label,
                }
            else:
                entry["buses"].append(b_idx)

        groups_list = list(groups.values())
        for entry in groups_list:
            entry["buses"].sort()
        groups_list.sort(key=lambda e: e["buses"][0])

        row_label_width = max(len(f"G{len(groups_list)}"), 2)
        col_width = max(max_cell_width, max(len(l) for l in labels), len("inf"), len("N/A"))

        header = " " * (row_label_width + 1) + " ".join(l.rjust(col_width) for l in labels)
        print(header)
        for idx, entry in enumerate(groups_list, start=1):
            row_vals = " ".join(cell.rjust(col_width) for cell in entry["row"])
            print(f"{f'G{idx}'.rjust(row_label_width)} " + row_vals)

        print("\nGroup Legend:")
        for idx, entry in enumerate(groups_list, start=1):
            buses = format_bus_list(entry["buses"])
            count = len(entry["buses"])
            print(f"  G{idx}: buses={buses} (count={count}), origin={entry['origin']}")

    def _print_detailed_debug_info(self, final_solution, n_depots, durations_matrix):
        """
        Prints a side-by-side comparison of the Baseline vs Final Solution,
        plus the Distance Matrix to explain the routing decisions.
        """
        print("\n" + "="*80)
        print("🔍 OPTIMIZATION DEEP DIVE")
        print("="*80)

        # --- 1. Print Distance Matrix (Pretty Printed) ---
        print("\n1) DISTANCE MATRIX (Minutes)")
        print("-" * 60)
        self._print_distance_matrix(durations_matrix, n_depots)

        print("\nFIRST LEG MATRIX (Minutes) [Origins -> Nodes]")
        print("-" * 60)
        self._print_first_leg_matrix(final_solution, n_depots, durations_matrix)

        # --- Helper to print a solution ---
        def print_sol(title, sol):
            print(f"\n{title}")
            print("-" * 60)
            if not sol:
                print("  (No solution data)")
                return

            total_trips = 0
            for b_idx, bus_trips in enumerate(sol):
                if not bus_trips:
                    continue
                # Handle case where cap_by_bus might not be set yet or list is empty
                cap = '?'
                if self._cap_by_bus and b_idx < len(self._cap_by_bus):
                    cap = self._cap_by_bus[b_idx]
                
                print(f"  🚌 Bus {b_idx} (Cap: {cap})")
                origin_label = self._format_origin_label(b_idx)
                if origin_label:
                    print(f"    Origin: {origin_label}")
                first_leg = self._first_leg_minutes_for_bus(b_idx, bus_trips, n_depots, durations_matrix)
                if first_leg is not None:
                    print(f"    First leg: {first_leg:.1f} min")
                else:
                    print("    First leg: N/A")
                
                for t_idx, trip in enumerate(bus_trips):
                    total_trips += 1
                    start = trip.get("start_depot")
                    end = trip.get("end_depot")
                    
                    stops_formatted = []
                    stops_data = trip.get("stops", [])
                    counts = trip.get("pickup_counts", {})
                    
                    # Check if stops is list of tuples [(node, count)] (Baseline/Intermediate) 
                    # or list of ints [node] (Final Output Format)
                    if stops_data and isinstance(stops_data[0], (list, tuple)):
                         stops_formatted = [f"Node {n}({c})" for n, c in stops_data]
                    else:
                        # Fallback for standard output format using pickup_counts dict
                        stops_formatted = [f"Node {n}({counts.get(n, '?')})" for n in stops_data]

                    stops_str = " -> ".join(stops_formatted)
                    
                    print(f"    Trip {t_idx}: Depot {start} -> [{stops_str}] -> Depot {end}")
            print(f"  Total Trips: {total_trips}")

        # --- 2. Print Baseline ---
        if self._debug_pendel_solution:
            print_sol("2️⃣  BASELINE SOLUTION (Pendelverkehr)", self._debug_pendel_solution)
        else:
            print("\n2️⃣  BASELINE SOLUTION: (Not Available / Failed)")

        # --- 3. Print Final EA ---
        print_sol("3️⃣  FINAL EVOLUTIONARY SOLUTION", final_solution)
        print("="*80 + "\n")
        

    def _find_best_end_depot(
        self,
        last_stop_node: int,
        trip_load: int,
        depot_loads: List[int],
        n_depots: int,
        durations_matrix: dict
    ) -> int:
        """
        Greedily finds the best end depot with awareness of depot capacities and current loads.
        1. Filters for depots that can accept the trip_load without exceeding capacity.
        2. Finds the nearest depot within that feasible set.
        3. If no depot has capacity, it chooses the one with the minimum overflow as a fallback.
        """
        if n_depots <= 1:
            return 0

        feasible_depots = []
        for i in range(n_depots):
            capacity = self._depots_runtime[i].get('capacity')
            if capacity is None or (depot_loads[i] + trip_load <= capacity):
                feasible_depots.append(i)

        from_location_idx = n_depots + last_stop_node

        # --- Stage 1: Find best among feasible depots ---
        if feasible_depots:
            min_time = float('inf')
            best_depot_idx = feasible_depots[0]
            for depot_idx in feasible_depots:
                travel_time = durations_matrix.get((from_location_idx, depot_idx), float('inf'))
                if travel_time < min_time:
                    min_time = travel_time
                    best_depot_idx = depot_idx
            return best_depot_idx

        # --- Stage 2: Fallback - No feasible depot found, so minimize the overfill ---
        else:
            min_overfill = float('inf')
            best_depot_idx = 0
            for i in range(n_depots):
                capacity = self._depots_runtime[i].get('capacity', float('inf'))
                overfill = (depot_loads[i] + trip_load) - capacity
                if overfill < min_overfill:
                    min_overfill = overfill
                    best_depot_idx = i
            return best_depot_idx


def run_evolutionary_algorithm(
    evacuation_zones_input: Optional[List[Dict[str, Any]]] = None,
    buses_count: int = 3,
    bus_capacity: int = 80,
    **algorithm_specific_params
) -> AlgorithmResult:

    algorithm = RevisionaryEvolutionaryAlgorithm()
    return algorithm.run(
        evacuation_zones_input=evacuation_zones_input,
        buses_count=buses_count,
        bus_capacity=bus_capacity,
        **algorithm_specific_params
    )
