# Path: backend/app/evacuation/baselines/pendelverkehr.py
# -*- coding: utf-8 -*-
"""
Pendelverkehr (Shuttle) Baseline
================================

A simple, deterministic shuttle strategy used as a benchmark:
- Each trip serves at least one pickup node. If the bus is under-utilized
  (e.g., < 60% capacity), it will attempt to add a second stop that minimizes detour.
- Trips return to the nearest depot with available capacity.
- Buses are dispatched based on their availability (event-driven), not round-robin.
- Correctly handles individual vehicle start positions (depot, node, or coordinate).

This class adheres to your EvacuationAlgorithm interface and returns artifacts
compatible with your experiment harness (algorithm_stats, simulation, metrics).
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import time
import math
import copy

import numpy as np

from ..algorithm_interface import EvacuationAlgorithm, AlgorithmResult
from .. import visualization
from ..metrics import compute_solution_metrics, _simulate_and_get_timings
from ..runtime_budget import RuntimeBudget


@dataclass
class PendelParams:
    """Optional knobs for the shuttle baseline."""
    home_depot: int = 0                 # Fallback depot for trips without a clear nearest one.
    use_nearest_depot: bool = True      # If true, buses return to the nearest depot.
    pick_rule: str = "nearest"            # Selection rule for next node.
    max_rounds: int = 10_000            # Hard safety cap to prevent infinite loops.
    secondary_stop_threshold: float = 0.6 # Capacity threshold to trigger search for a second stop.


class PendelverkehrShuttleAlgorithm(EvacuationAlgorithm):
    """
    Enhanced shuttle baseline for evacuation. Adds a second stop to a trip if the bus
    is under-utilized, improving efficiency. Now returns to the nearest depot with capacity.
    Uses an event-driven model to dispatch the next available bus for better load balancing.
    """
    _service_params: Dict[str, Any] = {}
    _node_coords: Optional[Dict[int, Tuple[float, float]]] = None
    _start_to_node_seconds: Optional[Dict[int, Dict[int, float]]] = None
    _avg_speed_kmh: float = 30.0
    _road_factor: float = 1.25

    def _find_best_end_depot(self, from_node_idx: int, trip_load: int, depot_loads: List[int], depots_data: List[dict], n_depots: int, durations_matrix: dict) -> int:
        """Helper to find the nearest depot, prioritizing those with available capacity."""
        if n_depots <= 1:
            return 0

        # 1. Find depots with enough capacity
        available_depots = []
        for i in range(n_depots):
            capacity = depots_data[i].get('capacity', float('inf'))
            if capacity is None or depot_loads[i] + trip_load <= capacity:
                available_depots.append(i)

        # 2. Decide which set of depots to search
        # If there are depots with capacity, find the nearest among them.
        # Otherwise, search all depots (fallback).
        depot_subset_to_check = available_depots if available_depots else list(range(n_depots))
        
        if not depot_subset_to_check: # Safety for edge case n_depots=0
            return 0

        # 3. Find the nearest depot in the chosen subset
        min_time = float('inf')
        best_depot = depot_subset_to_check[0] # Default to the first in the list
        from_location_idx = n_depots + from_node_idx
        
        for depot_idx in depot_subset_to_check:
            key = (from_location_idx, depot_idx)
            time_to_depot = durations_matrix.get(key, float('inf'))
            
            if time_to_depot < min_time:
                min_time = time_to_depot
                best_depot = depot_idx
                
        return best_depot

    def _calculate_trip_duration(
            self,
            trip: Dict[str, Any],
            bus_capacity: int,
            n_depots: int,
            durations_matrix: Dict[Tuple[int, int], float]
        ) -> float:
        """
        Calculates the total time a bus is occupied for a trip (1 or 2 stops),
        including travel, pickup service, and offloading.
        """
        stops = trip.get("stops", [])
        if not stops:
            return 0.0

        total_duration = 0.0
        
        # Determine starting location index
        # Note: Pendelverkehr usually sets start_depot index relative to n_depots
        current_location_is_depot = True
        current_location_idx = trip["start_depot"]

        for node, pickup_count in stops:
            # 1. Travel Time
            if current_location_is_depot:
                travel_time = durations_matrix.get((current_location_idx, n_depots + node), float("inf")) / 60.0
            else: # From node to node
                travel_time = durations_matrix.get((n_depots + current_location_idx, n_depots + node), float("inf")) / 60.0
            
            total_duration += travel_time

            # 2. Pickup Service Time
            if self._service_params.get("use_dynamic_service_time", False):
                # Dynamic: Pay for what you pick up
                base = self._service_params.get("service_time_base_min", 3.0)
                rate = self._service_params.get("service_time_per_person_min", 20.0/60.0)
                total_duration += base + pickup_count * rate
            else:
                # Static (Punitive): Pay for max capacity
                base = self._service_params.get("service_time_base_min", 3.0)
                rate = self._service_params.get("service_time_per_person_min", 20.0/60.0)
                total_duration += base + bus_capacity * rate
            
            current_location_is_depot = False
            current_location_idx = node

        # 3. Return Travel Time
        total_duration += durations_matrix.get((n_depots + current_location_idx, trip["end_depot"]), float("inf")) / 60.0

        # 4. Offloading Service Time
        total_people_in_trip = sum(count for _, count in stops)
        if total_people_in_trip > 0:
            if self._service_params.get("use_dynamic_service_time", False):
                # Dynamic: Pay for what you offload
                base = self._service_params.get("service_time_base_min", 3.0)
                rate = self._service_params.get("service_time_per_person_min", 20.0/60.0)
                total_duration += base + total_people_in_trip * rate
            else:
                # Static (Punitive): Pay for max capacity
                base = self._service_params.get("service_time_base_min", 3.0)
                rate = self._service_params.get("service_time_per_person_min", 20.0/60.0)
                total_duration += base + bus_capacity * rate

        # Fallback for disconnected graph errors
        if not math.isfinite(total_duration):
            return 45.0 + (len(stops) - 1) * 20.0

        return total_duration

    def run(
        self,
        evacuation_zones_input: Optional[List[Dict[str, Any]]] = None,
        buses_count: int = 3,
        bus_capacity: int = 80,
        vehicles: Optional[List[Dict[str, Any]]] = None,
        start_to_node_seconds: Optional[Dict[int, Dict[int, float]]] = None,
        avg_speed_kmh: float = 30.0,
        road_factor: float = 1.25,
        precomputed_problem_data: Optional[Dict[str, Any]] = None,
        **algorithm_specific_params: Any,
    ) -> AlgorithmResult:

        run_started_at = time.monotonic()
        budget = RuntimeBudget(
            limit_seconds=algorithm_specific_params.get(
                "time_limit_seconds",
                None,
            ),
            mode=algorithm_specific_params.get("budget_mode", "strict"),
            postprocess_reserve_seconds=float(
                algorithm_specific_params.get(
                    "postprocess_reserve_seconds",
                    0.25,
                )
            ),
            run_started_at=run_started_at,
        )

        pendel = PendelParams(
            home_depot=int(algorithm_specific_params.get("home_depot", 0)),
            use_nearest_depot=bool(algorithm_specific_params.get("use_nearest_depot", True)),
            pick_rule=str(algorithm_specific_params.get("pick_rule", "nearest")),
            max_rounds=int(algorithm_specific_params.get("max_rounds", 10_000)),
            secondary_stop_threshold=float(algorithm_specific_params.get("secondary_stop_threshold", 0.6)),
        )
        self._service_params = {
            "use_dynamic_service_time": algorithm_specific_params.get('use_dynamic_service_time', False),
            "service_time_base_min": algorithm_specific_params.get('service_time_base_min', 3.0),
            "service_time_per_person_min": algorithm_specific_params.get('service_time_per_person_min', 20.0 / 60.0),
        }
        self._start_to_node_seconds = start_to_node_seconds
        self._avg_speed_kmh = avg_speed_kmh
        self._road_factor = road_factor

        if precomputed_problem_data:
            print("📦 Using pre-computed problem data (Matrix & Graph)...")
            problem_data = precomputed_problem_data
        else:
            print("🌐 Calculating problem data from scratch (API/OSRM)...")
            problem_data = self.initialize_problem(
                evacuation_zones_input, buses_count, bus_capacity,
                default_evac_center_coords=algorithm_specific_params.get('default_evac_center_coords', None),
                buffer_meters=algorithm_specific_params.get('buffer_meters', None)
            )
        depots = problem_data["depots"]
        facilities = problem_data["facilities"]
        durations_matrix: Dict[Tuple[int, int], float] = problem_data["durations_matrix"]
        n_depots: int = problem_data["n_depots"]
        demand_full: Dict[int, int] = problem_data["demand_full"]
        pickup_nodes: List[int] = problem_data["pickup_nodes"]
        max_trips_per_bus: int = problem_data["max_trips_per_bus"]
        max_stops_per_trip: int = problem_data["max_stops_per_trip"]
        self._node_coords = problem_data.get("node_coords")

        if vehicles:
            buses_count = len(vehicles)
            cap_by_bus = [v.get('capacity', bus_capacity) for v in vehicles]
            normalized_vehicles = []
            for v in vehicles:
                start_info = v.get("start")
                if not start_info:
                    start_info = {"kind": "depot", "index": v.get("start_depot", 0)}
                    if v.get("start_node") is not None:
                        start_info = {"kind": "node", "index": v.get("start_node")}
                    elif v.get("start_coord") is not None:
                        start_info = {"kind": "coord", **v.get("start_coord")}
                
                normalized_vehicles.append({
                    "id": v.get("id"), "capacity": v.get('capacity', bus_capacity), "start": start_info
                })
        else:
            cap_by_bus = [bus_capacity] * buses_count
            normalized_vehicles = [
                {"id": None, "capacity": bus_capacity, "start": {"kind": "depot", "index": 0}}
                for _ in range(buses_count)
            ]

        remaining: Dict[int, int] = {node: int(demand_full.get(node, 0)) for node in pickup_nodes}
        active_nodes: List[int] = [node for node in pickup_nodes if remaining[node] > 0]
        schedules: List[List[Dict[str, Any]]] = [[] for _ in range(buses_count)]
        depot_loads: List[int] = [0] * n_depots
        
        bus_available_time = [0.0] * buses_count
        bus_current_location = [v.get('start', {"kind": "depot", "index": pendel.home_depot}) for v in normalized_vehicles]
        budget.start_search()

        def get_travel_time_from_location(location: dict, to_node: int) -> float:
            kind = location.get('kind')
            if kind == 'depot':
                return durations_matrix.get((location['index'], n_depots + to_node), float("inf")) / 60.0
            if kind == 'node':
                if location['index'] == to_node: return 0.0
                return durations_matrix.get((n_depots + location['index'], n_depots + to_node), float("inf")) / 60.0
            if kind == 'coord':
                bus_idx = -1
                for i, v in enumerate(normalized_vehicles):
                    if v['start'] == location:
                        bus_idx = i; break
                return self._calculate_first_leg_minutes(bus_idx, to_node, location, n_depots, durations_matrix)
            return float('inf')

        def next_node_for(current_location: dict) -> Optional[int]:
            if not active_nodes: return None
            if pendel.pick_rule == "nearest":
                return min(active_nodes, key=lambda n: get_travel_time_from_location(current_location, n))
            if pendel.pick_rule == "largest_demand":
                return max(active_nodes, key=lambda n: remaining[n])
            raise ValueError(
                f"Unsupported pick_rule {pendel.pick_rule!r}; use 'nearest' or 'largest_demand'."
            )

        rounds = 0
        while any(r > 0 for r in remaining.values()) and rounds < pendel.max_rounds:
            rounds += 1
            next_bus_idx = min(range(buses_count), key=lambda b: bus_available_time[b])
            start_location = bus_current_location[next_bus_idx]
            
            node = next_node_for(start_location)
            if node is None: break 

            bus_cap = cap_by_bus[next_bus_idx]
            load = min(bus_cap, remaining[node])
            
            trip_load = load
            trip = { 
                "start_depot": start_location['index'] if start_location.get('kind') == 'depot' else pendel.home_depot,
                "stops": [(node, load)], 
            }
            
            remaining[node] -= load
            if remaining[node] <= 0 and node in active_nodes:
                active_nodes.remove(node)

            if load < bus_cap * pendel.secondary_stop_threshold:
                remaining_capacity = bus_cap - load
                best_second_node, min_detour = None, float('inf')
                for candidate_node in active_nodes:
                    if candidate_node == node: continue
                    detour = (durations_matrix.get((n_depots + node, n_depots + candidate_node), float("inf")) / 60.0 + 
                              durations_matrix.get((n_depots + candidate_node, pendel.home_depot), float("inf")) / 60.0 - 
                              durations_matrix.get((n_depots + node, pendel.home_depot), float("inf")) / 60.0)
                    if detour < min_detour:
                        min_detour, best_second_node = detour, candidate_node
                
                if best_second_node is not None:
                    second_load = min(remaining_capacity, remaining[best_second_node])
                    if second_load > 0:
                        trip["stops"].append((best_second_node, second_load))
                        trip_load += second_load
                        remaining[best_second_node] -= second_load
                        if remaining[best_second_node] <= 0 and best_second_node in active_nodes:
                            active_nodes.remove(best_second_node)

            last_stop_node = trip['stops'][-1][0]
            if pendel.use_nearest_depot:
                trip["end_depot"] = self._find_best_end_depot(last_stop_node, trip_load, depot_loads, depots, n_depots, durations_matrix)
            else:
                trip["end_depot"] = pendel.home_depot
            
            depot_loads[trip["end_depot"]] += trip_load

            travel_to_first_stop = get_travel_time_from_location(start_location, node)
            temp_trip_for_calc = {
                "start_depot": pendel.home_depot, "stops": trip['stops'], "end_depot": trip['end_depot']
            }
            trip_body_duration = self._calculate_trip_duration(temp_trip_for_calc, bus_cap, n_depots, durations_matrix)
            trip_body_duration -= durations_matrix.get((pendel.home_depot, n_depots + node), 0) / 60.0
            
            total_mission_time = travel_to_first_stop + trip_body_duration

            bus_available_time[next_bus_idx] += total_mission_time
            bus_current_location[next_bus_idx] = {"kind": "depot", "index": trip["end_depot"]}
            schedules[next_bus_idx].append(trip)
        
        best_solution: List[List[Dict[str, Any]]] = []
        for b in range(buses_count):
            trips = []
            for trip in schedules[b]:
                if not trip.get("stops"): continue
                pickup_counts = {node: cnt for (node, cnt) in trip["stops"]}
                trips.append({
                    "start_depot": int(trip["start_depot"]), "stops": [int(node) for (node, _cnt) in trip["stops"]],
                    "end_depot": int(trip["end_depot"]), "pickup_counts": {int(k): int(v) for k, v in pickup_counts.items()},
                })
            best_solution.append(trips)

        # Get penalty factor from EA params to ensure consistent fitness calculation
        latest_evacuation_penalty_factor = algorithm_specific_params.get('latest_evacuation_penalty_factor', 0.0)
        best_fitness = self._evaluate_fitness_like_ea(
            schedules, cap_by_bus, n_depots, durations_matrix, normalized_vehicles,
            latest_evacuation_penalty_factor, depots
        )

        simulation_data = self.create_simulation_data(
            best_solution, buses_count, cap_by_bus, depots, facilities,
            n_depots, durations_matrix, demand_full,
            **self._service_params
        )
        solution_summary = visualization.create_solution_summary(
            best_solution, buses_count, simulation_data
        )

        gen_metrics = self._extract_per_generation_like_ea(
            schedules, buses_count, cap_by_bus, depots, facilities,
            n_depots, durations_matrix, demand_full, fitness=best_fitness
        )

        optimization_ended_at = budget.now()
        algorithm_stats: Dict[str, Any] = {
            "preprocessing_runtime": budget.preprocessing_runtime(),
            "optimization_runtime": budget.search_runtime(optimization_ended_at),
            "postprocessing_runtime": None,
            "total_runtime": budget.total_runtime(optimization_ended_at),
            "generation_costs": [best_fitness], "best_per_generation": [best_fitness],
            "avg_per_generation": [best_fitness], "generation_avg_evacuation_times": [gen_metrics.get("gen_avg_evacuation_time", float("inf"))],
            "generation_latest_evacuation_times": [gen_metrics.get("gen_latest_evacuation_time", float("inf"))],
            "generation_total_people_evacuated": [gen_metrics.get("gen_total_people_evacuated", 0)],
            "generation_evacuation_efficiencies": [gen_metrics.get("gen_evacuation_efficiency", 0.0)],
            "generation_fitness_std": [0.0], "generation_population_diversity": [1.0],
            "time_limit_seconds": budget.limit_seconds,
            "stopped_by_time_limit": False,
            **budget.metadata(),
        }
        if budget.is_strict:
            algorithm_stats["stopping_rule"] = (
                "complete_constructive_plan_or_reject_overrun"
            )
        
        result = self.create_result_object(
            "pendelverkehr_shuttle", best_fitness, best_solution, simulation_data,
            depots, facilities, buses_count, bus_capacity
        )
        result["vehicles"] = normalized_vehicles
        result["problem_data"] = {
            "n_depots": n_depots, "durations_matrix": {str(k): v for k, v in durations_matrix.items()},
            "demand_full": demand_full, "max_trips_per_bus": max_trips_per_bus,
            "max_stops_per_trip": max_stops_per_trip, "depots": depots, "facilities": facilities,
            "vehicles": normalized_vehicles,
        }
        result["algorithm_stats"] = algorithm_stats
        result["budget_mode"] = budget.mode
        result["time_limit_seconds"] = budget.limit_seconds
        result["metrics"] = compute_solution_metrics(
            best_solution, buses_count=buses_count, n_depots=n_depots, durations_matrix=durations_matrix,
            demand_full=demand_full, vehicles=normalized_vehicles, depots=depots,
            **self._service_params, node_coords=self._node_coords, start_to_node_seconds=self._start_to_node_seconds,
            avg_speed_kmh=self._avg_speed_kmh, road_factor=self._road_factor,
        )

        self.log_algorithm_run(
            "pendelverkehr_shuttle",
            { "buses_count": buses_count, "bus_capacity": bus_capacity, "home_depot": pendel.home_depot,
              "pick_rule": pendel.pick_rule, "use_nearest_depot": pendel.use_nearest_depot, **self._service_params, },
            facilities, best_fitness, solution_summary, depots, algorithm_stats=algorithm_stats,
        )

        if not budget.is_strict:
            self._print_final_summary(result)

        run_ended_at = budget.now()
        algorithm_stats["total_runtime"] = budget.total_runtime(run_ended_at)
        algorithm_stats["postprocessing_runtime"] = max(
            0.0,
            algorithm_stats["total_runtime"]
            - algorithm_stats["preprocessing_runtime"]
            - algorithm_stats["optimization_runtime"],
        )
        algorithm_stats["budget_overshoot_seconds"] = (
            budget.overshoot_seconds(run_ended_at)
        )
        algorithm_stats["budget_adhered"] = (
            not budget.is_strict
            or algorithm_stats["budget_overshoot_seconds"] <= 1e-9
        )
        result["optimization_runtime"] = algorithm_stats[
            "optimization_runtime"
        ]
        result["total_runtime"] = algorithm_stats["total_runtime"]
        return result

    def _calculate_first_leg_minutes(self, bus_idx: int, first_node: int, origin: dict, n_depots: int, durations_matrix: dict) -> float:
        kind = origin.get('kind')
        
        if kind == 'depot':
            depot_idx = origin.get('index', 0)
            return durations_matrix.get((depot_idx, n_depots + first_node), float('inf')) / 60.0
        
        if kind == 'node':
            start_node = origin.get('index')
            if start_node == first_node: return 0.0
            return durations_matrix.get((n_depots + start_node, n_depots + first_node), float('inf')) / 60.0
        
        if kind == 'coord':
            if self._start_to_node_seconds and bus_idx != -1 and bus_idx in self._start_to_node_seconds:
                if first_node in self._start_to_node_seconds[bus_idx]:
                    return self._start_to_node_seconds[bus_idx][first_node] / 60.0
            
            if not self._node_coords or first_node not in self._node_coords:
                return 30.0
            
            nlat, nlon = self._node_coords[first_node]
            lat, lon = origin['lat'], origin['lon']
            
            R = 6371.0
            phi1, phi2 = math.radians(lat), math.radians(nlat)
            dphi = math.radians(nlat - lat)
            dlambda = math.radians(nlon - lon)
            a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            km = R * c
            
            return (km / self._avg_speed_kmh) * 60.0 * self._road_factor

        return float('inf')

    def _print_final_summary(self, result: AlgorithmResult):
        print("\n" + "=" * 50)
        print("📊 PENDELVERKEHR - FINAL SUMMARY")
        print("-" * 50)
        cost = result.get('overall_cost', 0.0)
        metrics = result.get('metrics', {})
        avg_evac_time = metrics.get('wait', {}).get('mean_min', 0.0)
        latest_evac_time = metrics.get('timeline', {}).get('latest_return_min', 0.0)
        print(f"  - Overall Cost (Fitness): {cost:.2f}")
        print(f"  - Average Evacuation Time: {avg_evac_time:.2f} min")
        print(f"  - Latest Evacuation Time: {latest_evac_time:.2f} min")
        print("=" * 50)

    # ---- helpers (parity with EA scoring & per-gen extraction) ----
    def _evaluate_fitness_like_ea(
        self, individual: List[List[Dict[str, Any]]], cap_by_bus: List[int], n_depots: int,
        durations_matrix: Dict[Tuple[int, int], float],
        normalized_vehicles: List[Dict[str, Any]], latest_evacuation_penalty_factor: float,
        depots: List[Dict[str, Any]]
    ) -> float:
        """
        Calculates fitness score using the central simulation helper to match the EA.
        """
        from ..core import PENALTY_FACTOR # Use a consistent, high penalty
        from ..ea import DEPOT_OVERFILL_PENALTY

        origin_by_bus = [v.get("start", {"kind": "depot", "index": 0}) for v in normalized_vehicles]

        sim_results = _simulate_and_get_timings(
            individual=individual,
            n_depots=n_depots,
            durations_matrix=durations_matrix,
            origin_by_bus=origin_by_bus,
            cap_by_bus=cap_by_bus,
            depots=depots,
            node_coords=self._node_coords,
            start_to_node_seconds=self._start_to_node_seconds,
            avg_speed_kmh=self._avg_speed_kmh,
            road_factor=self._road_factor,
            **self._service_params,
        )

        fitness = (
            sim_results["total_wait_pm"] / sim_results["total_people_evacuated"]
            #+ latest_evacuation_penalty_factor * 
            + sim_results["latest_evac_min"]
            #+ DEPOT_OVERFILL_PENALTY * 
            + sim_results["total_overfill"]
        )
        return float(fitness)

    def _extract_per_generation_like_ea(
        self, individual: List[List[Dict[str, Any]]], buses_count: int, cap_by_bus: List[int], depots, facilities,
        n_depots: int, durations_matrix: Dict[Tuple[int, int], float], demand_full: Dict[int, int],
        *, fitness: float,
    ) -> Dict[str, Any]:
        solution: List[List[Dict[str, Any]]] = []
        for bus_sched in individual:
            bus_trips = []
            for trip in bus_sched:
                if not trip.get("stops"): continue
                pickup_counts = {node: cnt for (node, cnt) in trip["stops"]}
                bus_trips.append({
                    "start_depot": trip["start_depot"], "stops": [node for (node, _cnt) in trip["stops"]],
                    "end_depot": trip["end_depot"], "pickup_counts": pickup_counts,
                })
            solution.append(bus_trips)

        sim = visualization.simulate_solution_with_timeline(
            solution, buses_count, cap_by_bus, depots, facilities, n_depots, durations_matrix, demand_full,
            **self._service_params
        )

        pickup_times, return_times, total_people_evacuated = [], [], 0
        for bus_id, bus_trips in sim.items():
            for trip_id, trip_data in bus_trips.items():
                if not isinstance(trip_data, dict): continue
                return_times.append(trip_data.get("return", 0.0))
                departure_time, trip_time = trip_data.get("departure", 0.0), trip_data.get("trip_time", 0.0)
                details = trip_data.get("details", [])
                for detail in details:
                    if "picked up" in detail:
                        try:
                            part = detail.split("picked up")[1]
                            num = int(part.strip())
                            total_people_evacuated += num
                            est = departure_time + (trip_time * 0.5)
                            for _ in range(num): pickup_times.append(est)
                        except Exception:
                            pickup_times.append(departure_time + (trip_time * 0.5))

        out: Dict[str, Any] = {"gen_algorithm_cost": fitness}
        if pickup_times:
            avg_evac = sum(pickup_times) / len(pickup_times)
            out["gen_avg_evacuation_time"] = float(avg_evac)
            out["min_pickup_time"] = float(min(pickup_times))
            out["max_pickup_time"] = float(max(pickup_times))
            if total_people_evacuated > 0 and max(pickup_times) > 0:
                out["gen_evacuation_efficiency"] = float(total_people_evacuated / max(pickup_times))
        else:
            out["gen_avg_evacuation_time"], out["gen_evacuation_efficiency"] = float("inf"), 0.0

        if return_times:
            out["gen_latest_evacuation_time"] = float(max(return_times))
            out["min_return_time"] = float(min(return_times))
        else:
            out["gen_latest_evacuation_time"] = float("inf")

        out["gen_total_people_evacuated"] = int(total_people_evacuated)
        out["gen_avg_fitness"] = float(fitness)
        out["gen_fitness_std"], out["gen_population_diversity"] = 0.0, 1.0
        return out
