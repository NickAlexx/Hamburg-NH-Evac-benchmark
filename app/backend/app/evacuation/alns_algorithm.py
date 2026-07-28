
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import defaultdict
import math
import random
import time

from .runtime_budget import RuntimeBudget

try:
    # Same interfaces used by ea.py
    from .algorithm_interface import EvacuationAlgorithm, AlgorithmResult  # type: ignore
    from .core import initialize_problem_data  # type: ignore
    from .metrics import compute_solution_metrics, _simulate_and_get_timings  # type: ignore

    # Strong seed (dispatcher baseline) used throughout your experiments/paper
    from ..evacuation.baselines.pendelverkehr import PendelverkehrShuttleAlgorithm  # type: ignore

    # Optional intensification = same local search portfolio as your MA
    from .local_search import MemeticImprover  # type: ignore
except Exception:
    # Standalone fallback (minimal, so the file still imports)
    EvacuationAlgorithm = object  # type: ignore
    AlgorithmResult = Dict[str, Any]  # type: ignore
    initialize_problem_data = None  # type: ignore
    compute_solution_metrics = None  # type: ignore
    _simulate_and_get_timings = None  # type: ignore
    PendelverkehrShuttleAlgorithm = None  # type: ignore
    MemeticImprover = None  # type: ignore


Trip = Dict[str, Any]
Individual = List[List[Trip]]
RemovedList = List[Tuple[int, int]]  # [(node, qty), ...]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def _roulette_choice(weights: Dict[str, float]) -> str:
    """Roulette selection robust to non-positive weights."""
    items = list(weights.items())
    if not items:
        raise ValueError("No operators available.")
    total = sum(max(0.0, w) for _, w in items)
    if total <= 0.0:
        return random.choice([k for k, _ in items])
    r = random.random() * total
    acc = 0.0
    for k, w in items:
        acc += max(0.0, w)
        if acc >= r:
            return k
    return items[-1][0]

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def _now() -> float:
    return time.monotonic()


def _fast_clone_individual(ind: Individual) -> Individual:
    """Faster than deepcopy for this nested-but-regular structure."""
    out: Individual = []
    for sched in ind:
        new_sched: List[Trip] = []
        for tr in sched:
            stops = tr.get("stops", [])
            new_sched.append({
                "start_depot": int(tr.get("start_depot", 0)),
                "end_depot": int(tr.get("end_depot", 0)),
                "stops": [(int(n), int(q)) for (n, q) in stops if int(q) > 0],
            })
        out.append(new_sched)
    return out


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ALNSConfig:
    # Termination
    time_limit_seconds: float = 300.0
    max_iterations: int = 10_000_000  # hard cap; time is the real stop
    stall_seconds: float = 999999.0   # optional “give up” if best not improved for this long

    # Logging
    log_every_seconds: float = 5.0
    log_every_iterations: int = 0  # if >0, also logs every N iterations

    # Removal size (fraction of stop-entries in current solution)
    remove_fraction_min: float = 0.08
    remove_fraction_max: float = 0.28
    # If True, linearly shifts from max->min over time (large destroy early, small late).
    anneal_removal_fraction: bool = True

    # Adaptive weights update
    segment_length: int = 50
    reaction_factor: float = 0.2
    min_weight: float = 0.05

    # Scoring 
    sigma_best: float = 10.0       # global best
    sigma_improve: float = 5.0     # improves current
    sigma_accept: float = 1.0      # accepted but not improved

    # SA acceptance
    sa_initial_accept_prob: float = 0.5
    sa_samples_for_T0: int = 20
    sa_cooling_rate: float = 0.999
    sa_min_temp: float = 1e-6
    # If stuck (no improvement for N seconds), reheat temperature by this factor.
    sa_reheat_after_seconds: float = 30.0
    sa_reheat_factor: float = 1.5
    sa_reheat_max_multiplier: float = 10.0

    # Insertion heuristic shaping (cheap proxies)
    insertion_makespan_penalty: float = 0.35  # >0 encourages using “slack” buses
    insertion_late_trip_penalty: float = 0.02 # small preference for earlier trips
    insertion_noise: float = 0.02             # multiplicative noise on insertion score (diversification)

    # Seeding
    seed_with_dispatcher: bool = True
    dispatcher_pick_rule: str = "nearest"     # match your paper experiments
    initial_shake_cycles: int = 3             # diversify around the seed (cheap ALNS moves)

    # Local search “polish” (optional)
    use_memetic_polish: bool = True
    polish_probability_on_accept: float = 0.30
    polish_probability_on_best: float = 1.00
    polish_time_limit_seconds: float = 0.03   # tiny budget per call; increase if you can afford it
    polish_max_iterations: int = 120
    polish_candidate_set_size: int = 8


# ---------------------------------------------------------------------------
# ALNS Algorithm
# ---------------------------------------------------------------------------

class ALNSEvacuationAlgorithm(EvacuationAlgorithm):
    """
    Adaptive Large Neighborhood Search (ALNS) for the NH-Evac-VRP representation.

    Key design choices (to help convergence under 300s):
    - starts from the dispatcher baseline (PendelverkehrShuttleAlgorithm) by default
    - uses multiple destroy operators including *bottleneck-aware* ruin
    - uses regret insertion + quantity-aware “augment existing stop” option
    - (optionally) polishes accepted solutions with a tiny MemeticImprover budget
    """

    # runtime context (set in run)
    _cap_by_bus: Optional[List[int]] = None
    _origin_by_bus: Optional[List[Dict[str, Any]]] = None
    _depots_runtime: Optional[List[Dict[str, Any]]] = None
    _facilities_runtime: Optional[List[Dict[str, Any]]] = None
    _node_coords: Optional[Dict[int, Tuple[float, float]]] = None  # node -> (lat,lon)
    _start_to_node_seconds: Optional[Dict[int, Dict[int, float]]] = None
    _avg_speed_kmh: float = 30.0
    _road_factor: float = 1.25
    _service_params: Dict[str, Any] = {}
    _debug_pendel_solution: Optional[List[List[Dict[str, Any]]]] = None

    # penalty knobs (kept for fitness equivalence)
    _penalty_factor: float = 1000.0
    _latest_evacuation_penalty_factor: float = 0.0

    # cache for node-node travel minutes (pickup graph)
    _node_node_minutes_cache: Dict[Tuple[int, int], float]

    def __init__(self) -> None:
        super().__init__()  # type: ignore[misc]
        self._node_node_minutes_cache = {}
        self._debug_pendel_solution = None

    # -----------------------------------------------------------------------
    # Public entry point
    # -----------------------------------------------------------------------
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
        # match core.initialize_problem_data knobs
        default_evac_center_coords: Optional[Tuple[float, float]] = None,
        buffer_meters: Optional[float] = None,
        # service model (kept consistent with ea.py)
        use_dynamic_service_time: bool = False,
        service_time_base_min: float = 3.0,
        service_time_per_person_min: float = 20.0 / 60.0,
        # penalty knobs (passed in your experiments)
        penalty_factor: float = 1000.0,
        latest_evacuation_penalty_factor: float = 0.0,
        # ALNS settings
        time_limit_seconds: Optional[float] = None,
        max_iterations: Optional[int] = None,
        log_every_iterations: Optional[int] = None,
        log_every_seconds: Optional[float] = None,
        alns_config: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        budget_mode: str = "strict",
        postprocess_reserve_seconds: float = 0.25,
        **_
    ) -> AlgorithmResult:
        if seed is not None:
            random.seed(seed)

        # Wall-clock start for the whole run (init/calibration + main loop + post)
        run_start = _now()
        self._debug_pendel_solution = None

        cfg = ALNSConfig(**(alns_config or {}))
        if time_limit_seconds is not None:
            cfg.time_limit_seconds = float(time_limit_seconds)
        if max_iterations is not None:
            cfg.max_iterations = int(max_iterations)
        if log_every_iterations is not None:
            cfg.log_every_iterations = int(log_every_iterations)
        if log_every_seconds is not None:
            cfg.log_every_seconds = float(log_every_seconds)

        budget = RuntimeBudget(
            limit_seconds=cfg.time_limit_seconds,
            mode=budget_mode,
            postprocess_reserve_seconds=postprocess_reserve_seconds,
            run_started_at=run_start,
        )

        self._service_params = {
            "use_dynamic_service_time": bool(use_dynamic_service_time),
            "service_time_base_min": float(service_time_base_min),
            "service_time_per_person_min": float(service_time_per_person_min),
        }
        self._penalty_factor = float(penalty_factor)
        self._latest_evacuation_penalty_factor = float(latest_evacuation_penalty_factor)

        self._avg_speed_kmh = float(avg_speed_kmh)
        self._road_factor = float(road_factor)
        self._start_to_node_seconds = start_to_node_seconds or None

        # --- Load/compute problem data ---
        if precomputed_problem_data is not None:
            problem_data = precomputed_problem_data
        else:
            if initialize_problem_data is None:
                raise RuntimeError(
                    "initialize_problem_data import failed. Paste this module into your package and fix imports."
                )
            problem_data = initialize_problem_data(
                evacuation_zones_input=evacuation_zones_input,
                buses_count=buses_count,
                bus_capacity=bus_capacity,
                default_evac_center_coords=default_evac_center_coords,
                buffer_meters_input=buffer_meters,
            )

        depots = problem_data["depots"]
        facilities = problem_data["facilities"]
        self._facilities_runtime = facilities
        durations_matrix = problem_data["durations_matrix"]
        n_depots = problem_data["n_depots"]
        pickup_nodes = problem_data["pickup_nodes"]
        demand_full = problem_data["demand_full"]
        self._node_coords = problem_data.get("node_coords")
        self._depots_runtime = depots

        # --- Fleet (hetero) ---
        cap_by_bus, origin_by_bus, buses_count_eff, normalized_vehicles = self._resolve_fleet(
            vehicles, buses_count, bus_capacity, n_depots, pickup_nodes
        )
        self._cap_by_bus = cap_by_bus
        self._origin_by_bus = origin_by_bus
        buses_count = buses_count_eff

        # --- Initial solution ---
        current = self._initial_solution(
            cfg=cfg,
            buses_count=buses_count,
            bus_capacity=bus_capacity,
            n_depots=n_depots,
            pickup_nodes=pickup_nodes,
            demand_full=demand_full,
            durations_matrix=durations_matrix,
            depots=depots,
            facilities=facilities,
            precomputed_problem_data=problem_data,
            normalized_vehicles=normalized_vehicles,
            deadline=budget.deadline if budget.is_strict else None,
        )
        current = self._repair(
            current, buses_count, bus_capacity,
            depots, facilities, n_depots, pickup_nodes, durations_matrix, demand_full
        )

        current_cost = self._evaluate_fitness(
            current, buses_count, bus_capacity, depots, facilities, n_depots,
            durations_matrix, demand_full,
            penalty_factor=self._penalty_factor,
            latest_evacuation_penalty_factor=self._latest_evacuation_penalty_factor,
        )

        best = _fast_clone_individual(current)
        best_cost = current_cost

        # --- Operators ---
        destroy_ops: Dict[str, Callable[..., Tuple[Individual, RemovedList]]] = {
            "random_stop": self._destroy_random_stop,
            "worst_detour": self._destroy_worst_detour,
            "shaw_related": self._destroy_shaw_related,
            "route": self._destroy_route,
            "node_cluster": self._destroy_node_cluster,
            "bottleneck": self._destroy_bottleneck,
        }
        repair_ops: Dict[str, Callable[..., Individual]] = {
            "greedy": self._repair_greedy,
            "regret2": lambda *args, **kwargs: self._repair_regret_k(*args, k=2, **kwargs),
            "regret3": lambda *args, **kwargs: self._repair_regret_k(*args, k=3, **kwargs),
        }

        # Adaptive weights + segment stats
        w_destroy = {k: 1.0 for k in destroy_ops}
        w_repair = {k: 1.0 for k in repair_ops}
        seg_score_destroy: Dict[str, float] = defaultdict(float)
        seg_score_repair: Dict[str, float] = defaultdict(float)
        seg_count_destroy: Dict[str, int] = defaultdict(int)
        seg_count_repair: Dict[str, int] = defaultdict(int)

        # Operator gain “forensics” (sum of best-improvement deltas credited to operators)
        operator_scoreboard: Dict[str, float] = defaultdict(float)

        # Optional Memetic “polish” module
        polisher = self._build_polisher(
            cfg=cfg,
            buses_count=buses_count,
            bus_capacity=bus_capacity,
            depots=depots,
            facilities=facilities,
            n_depots=n_depots,
            pickup_nodes=pickup_nodes,
            durations_matrix=durations_matrix,
            demand_full=demand_full,
        )

        # SA temperature calibration (important for stability)
        T = self._calibrate_initial_temperature(
            cfg=cfg,
            current=current,
            current_cost=current_cost,
            destroy_ops=destroy_ops,
            repair_ops=repair_ops,
            buses_count=buses_count,
            bus_capacity=bus_capacity,
            depots=depots,
            facilities=facilities,
            n_depots=n_depots,
            pickup_nodes=pickup_nodes,
            durations_matrix=durations_matrix,
            demand_full=demand_full,
            deadline=budget.deadline if budget.is_strict else None,
        )

        # Telemetry
        effective_time_limit = float(cfg.time_limit_seconds)
        if not math.isfinite(effective_time_limit) or effective_time_limit <= 0:
            effective_time_limit = math.inf

        stats: Dict[str, Any] = {
            "best_cost": float(best_cost),
            "best_iteration": 0,
            "initial_cost": float(current_cost),
            "iterations": 0,
            "accepted": 0,
            "improved": 0,
            "reheated": 0,
            "history": [],   # (it, current_cost, best_cost, accepted_bool)
            "progress": [],  # snapshots
            "final_destroy_weights": None,
            "final_repair_weights": None,
            "operator_scoreboard": None,
            "time_limit_seconds": None if math.isinf(effective_time_limit) else float(effective_time_limit),
            "stopped_by_time_limit": False,
            "stopped_by_stall": False,
            "preprocessing_runtime": None,
            "optimization_runtime": None,
            "postprocessing_runtime": None,
            "total_runtime": None,
            **budget.metadata(),
        }

        start_time = budget.start_search()
        stats["preprocessing_runtime"] = budget.preprocessing_runtime()
        last_log_time = start_time
        last_best_time = start_time
        last_reheat_time = start_time
        T0 = T
        max_T = T0 * cfg.sa_reheat_max_multiplier

        # Initial log snapshot
        self._maybe_log_snapshot(
            stats=stats,
            iteration=0,
            elapsed_s=0.0,
            best=best,
            best_cost=best_cost,
            n_depots=n_depots,
            durations_matrix=durations_matrix,
        )

        # ------------------ main loop ------------------
        for it in range(1, cfg.max_iterations + 1):
            elapsed = (
                budget.total_runtime()
                if budget.is_strict
                else budget.search_runtime()
            )
            if budget.expired():
                stats["stopped_by_time_limit"] = True
                break
            if (elapsed - (last_best_time - start_time)) >= cfg.stall_seconds:
                stats["stopped_by_stall"] = True
                break

            # Reheat if stuck (helps escape frozen state under aggressive cooling)
            if (
                cfg.sa_reheat_after_seconds > 0
                and (_now() - last_best_time) >= cfg.sa_reheat_after_seconds
                and (_now() - last_reheat_time) >= cfg.sa_reheat_after_seconds
            ):
                # Only reheat if temperature already cooled significantly
                T = min(max_T, max(cfg.sa_min_temp, T * cfg.sa_reheat_factor))
                last_reheat_time = _now()  # avoid immediate repeated reheats
                stats["reheated"] += 1

            # Choose operators
            d_name = _roulette_choice(w_destroy)
            r_name = _roulette_choice(w_repair)
            d_op = destroy_ops[d_name]
            r_op = repair_ops[r_name]

            # Determine removal size q
            total_stops = self._count_total_stops(current)
            if total_stops <= 0:
                current = self._construct_initial_solution_basic(
                    buses_count=buses_count,
                    n_depots=n_depots,
                    pickup_nodes=pickup_nodes,
                    demand_full=demand_full,
                    durations_matrix=durations_matrix,
                )
                current = self._repair(
                    current, buses_count, bus_capacity,
                    depots, facilities, n_depots, pickup_nodes, durations_matrix, demand_full
                )
                current_cost = self._evaluate_fitness(
                    current, buses_count, bus_capacity, depots, facilities, n_depots,
                    durations_matrix, demand_full,
                    penalty_factor=self._penalty_factor,
                    latest_evacuation_penalty_factor=self._latest_evacuation_penalty_factor,
                )
                continue

            if cfg.anneal_removal_fraction and math.isfinite(effective_time_limit):
                progress = _clamp(elapsed / max(1e-9, effective_time_limit), 0.0, 1.0)
                frac_mid = _lerp(cfg.remove_fraction_max, cfg.remove_fraction_min, progress)
                # a little randomization around the annealed mean
                frac = random.uniform(_clamp(frac_mid * 0.85, cfg.remove_fraction_min, cfg.remove_fraction_max),
                                      _clamp(frac_mid * 1.15, cfg.remove_fraction_min, cfg.remove_fraction_max))
            else:
                frac = random.uniform(cfg.remove_fraction_min, cfg.remove_fraction_max)

            q = max(1, int(frac * total_stops))
            q = min(q, total_stops)

            # Destroy -> Repair -> Global repair (safety net)
            partial, removed = d_op(
                current,
                q=q,
                buses_count=buses_count,
                n_depots=n_depots,
                durations_matrix=durations_matrix,
            )

            candidate = r_op(
                partial,
                removed=removed,
                cfg=cfg,
                buses_count=buses_count,
                n_depots=n_depots,
                durations_matrix=durations_matrix,
            )

            # Strict feasibility fix (ensures exact demand satisfaction, capacity, connectivity)
            candidate = self._repair(
                candidate, buses_count, bus_capacity,
                depots, facilities, n_depots, pickup_nodes, durations_matrix, demand_full
            )

            cand_cost = self._evaluate_fitness(
                candidate, buses_count, bus_capacity, depots, facilities, n_depots,
                durations_matrix, demand_full,
                penalty_factor=self._penalty_factor,
                latest_evacuation_penalty_factor=self._latest_evacuation_penalty_factor,
            )

            # A strict run only accepts candidates whose complete evaluation
            # finished before the deadline.
            if budget.is_strict and budget.expired():
                stats["stopped_by_time_limit"] = True
                break

            # SA acceptance
            delta = cand_cost - current_cost
            accept = False
            if delta <= 0:
                accept = True
            else:
                # avoid overflow
                denom = max(cfg.sa_min_temp, T)
                try:
                    accept = (random.random() < math.exp(-delta / denom))
                except OverflowError:
                    accept = False

            # Cool
            T = max(cfg.sa_min_temp, T * cfg.sa_cooling_rate)

            # Reward to update ALNS weights (standard)
            reward = 0.0

            if accept:
                stats["accepted"] += 1

                # (Optional) polish accepted solutions with MA local search (tiny budget)
                if polisher is not None:
                    do_polish = (random.random() < cfg.polish_probability_on_accept)
                    if cand_cost < best_cost:
                        do_polish = (random.random() < cfg.polish_probability_on_best)  # usually 1.0

                    if do_polish:
                        candidate2, cand_cost2, polish_gain, ls_stats = self._polish_with_memetic(
                            polisher=polisher,
                            cfg=cfg,
                            candidate=candidate,
                            candidate_cost=cand_cost,
                            context={
                                "generation": int(it),
                                "algorithm_start_time": start_time,
                                "time_limit_seconds": None if math.isinf(effective_time_limit) else float(effective_time_limit),
                                "deadline_monotonic": (
                                    budget.deadline if budget.is_strict else None
                                ),
                            },
                            buses_count=buses_count,
                            bus_capacity=bus_capacity,
                            depots=depots,
                            facilities=facilities,
                            n_depots=n_depots,
                            pickup_nodes=pickup_nodes,
                            durations_matrix=durations_matrix,
                            demand_full=demand_full,
                        )
                        candidate = candidate2
                        cand_cost = cand_cost2
                        if polish_gain > 0: operator_scoreboard["LS:memetic_polish"] += float(polish_gain)
                        if ls_stats:
                            for k, v in ls_stats.items():
                                operator_scoreboard[k] += float(v)
                            operator_scoreboard["LS:memetic_polish"] += float(polish_gain)

                # A local-search call may finish after the strict deadline.
                # Discard that in-progress iteration and retain the previously
                # completed incumbent.
                if budget.is_strict and budget.expired():
                    stats["stopped_by_time_limit"] = True
                    break

                if cand_cost < best_cost:
                    operator_scoreboard[f"D:{d_name}"] += float(best_cost - cand_cost)
                    operator_scoreboard[f"R:{r_name}"] += float(best_cost - cand_cost)

                    best = _fast_clone_individual(candidate)
                    best_cost = cand_cost
                    stats["best_cost"] = float(best_cost)
                    stats["best_iteration"] = int(it)
                    last_best_time = _now()
                    reward = cfg.sigma_best
                elif cand_cost < current_cost:
                    stats["improved"] += 1
                    reward = cfg.sigma_improve
                else:
                    reward = cfg.sigma_accept

                current = candidate
                current_cost = cand_cost

            # Track segment usage
            seg_count_destroy[d_name] += 1
            seg_count_repair[r_name] += 1
            seg_score_destroy[d_name] += reward
            seg_score_repair[r_name] += reward

            stats["iterations"] = int(it)
            stats["history"].append((int(it), float(current_cost), float(best_cost), bool(accept)))

            # Periodic weight update
            if cfg.segment_length > 0 and it % cfg.segment_length == 0:
                self._update_weights(w_destroy, seg_score_destroy, seg_count_destroy, cfg.reaction_factor, cfg.min_weight)
                self._update_weights(w_repair, seg_score_repair, seg_count_repair, cfg.reaction_factor, cfg.min_weight)
                seg_score_destroy.clear(); seg_score_repair.clear()
                seg_count_destroy.clear(); seg_count_repair.clear()

            # Logging
            if cfg.log_every_iterations and it % cfg.log_every_iterations == 0:
                self._maybe_log_snapshot(
                    stats=stats,
                    iteration=int(it),
                    elapsed_s=float(elapsed),
                    best=best,
                    best_cost=best_cost,
                    n_depots=n_depots,
                    durations_matrix=durations_matrix,
                )

            if cfg.log_every_seconds and (_now() - last_log_time) >= cfg.log_every_seconds:
                last_log_time = _now()
                self._maybe_log_snapshot(
                    stats=stats,
                    iteration=int(it),
                    elapsed_s=float(elapsed),
                    best=best,
                    best_cost=best_cost,
                    n_depots=n_depots,
                    durations_matrix=durations_matrix,
                )

        opt_end = budget.now()
        stats["optimization_runtime"] = budget.search_runtime(opt_end)

        stats["final_destroy_weights"] = dict(w_destroy)
        stats["final_repair_weights"] = dict(w_repair)
        stats["operator_scoreboard"] = dict(operator_scoreboard)

        # Convert to solution output format (stops list + pickup_counts dict)
        best_solution = self._individual_to_solution(best, buses_count)

        simulation_data = self.create_simulation_data(
            best_solution,
            buses_count,
            self._cap_by_bus or bus_capacity,
            depots,
            facilities,
            n_depots,
            durations_matrix,
            demand_full,
            **self._service_params,
        )

        result = self.create_result_object(
            "alns",
            float(best_cost),
            best_solution,
            simulation_data,
            depots,
            facilities,
            buses_count,
            bus_capacity,
        )
        result.update({
            "best_fitness": float(best_cost),
            "buses_count": int(buses_count),
            "vehicles": normalized_vehicles,
            "algorithm_stats": stats,
            "time_limit_seconds": None if math.isinf(effective_time_limit) else float(effective_time_limit),
            "budget_mode": budget.mode,
        })

        if compute_solution_metrics is not None:
            try:
                result["metrics"] = compute_solution_metrics(
                    solution=best_solution,
                    buses_count=buses_count,
                    n_depots=n_depots,
                    durations_matrix=durations_matrix,
                    demand_full=demand_full,
                    depots=depots,
                    service_time_min=10.0,  # legacy arg
                    vehicles=normalized_vehicles,
                    node_coords=self._node_coords,
                    start_to_node_seconds=self._start_to_node_seconds,
                    avg_speed_kmh=self._avg_speed_kmh,
                    road_factor=self._road_factor,
                    **self._service_params,
                )
            except Exception:
                pass

        if not budget.is_strict:
            print("\nDISTANCE MATRIX (Minutes)")
            print("-" * 60)
            self._print_distance_matrix(durations_matrix, n_depots)

            print("\nFIRST LEG MATRIX (Minutes) [Origins -> Nodes]")
            print("-" * 60)
            self._print_first_leg_matrix(best_solution, n_depots, durations_matrix)

            self._print_final_solution(best_solution, n_depots, durations_matrix)

        run_end = budget.now()
        stats["total_runtime"] = budget.total_runtime(run_end)
        stats["postprocessing_runtime"] = max(
            0.0,
            stats["total_runtime"]
            - stats["preprocessing_runtime"]
            - stats["optimization_runtime"],
        )
        stats["budget_overshoot_seconds"] = budget.overshoot_seconds(run_end)
        stats["budget_adhered"] = (
            not budget.is_strict or stats["budget_overshoot_seconds"] <= 1e-9
        )
        result["optimization_runtime"] = stats["optimization_runtime"]
        result["total_runtime"] = stats["total_runtime"]

        return result

    # -----------------------------------------------------------------------
    # Fitness (same as EA / MA)
    # -----------------------------------------------------------------------
    def _evaluate_fitness(
        self,
        individual: Individual,
        buses_count: int,
        bus_capacity: int,
        depots: List[Dict[str, Any]],
        facilities: List[Dict[str, Any]],
        n_depots: int,
        durations_matrix: Dict[Tuple[int, int], float],
        demand_full: Dict[int, int],
        penalty_factor: float = 1000.0,
        latest_evacuation_penalty_factor: float = 0.0,
    ) -> float:
        if _simulate_and_get_timings is None:
            # Fallback: very rough proxy
            return float(self._count_total_stops(individual))

        sim = _simulate_and_get_timings(
            individual=individual,
            n_depots=n_depots,
            durations_matrix=durations_matrix,
            origin_by_bus=self._origin_by_bus,
            cap_by_bus=self._cap_by_bus,
            depots=self._depots_runtime,
            node_coords=self._node_coords,
            start_to_node_seconds=self._start_to_node_seconds,
            avg_speed_kmh=self._avg_speed_kmh,
            road_factor=self._road_factor,
            **self._service_params,
        )

        total_people = max(1, int(sim.get("total_people_evacuated", 1)))
        avg_wait = float(sim.get("total_wait_pm", 0.0)) / float(total_people)
        makespan = float(sim.get("latest_evac_min", 0.0))
        total_overfill = float(sim.get("total_overfill", 0.0))
        late_evac_penalty = float(sim.get("late_evac_penalty", 0.0))
        return float(avg_wait + makespan + total_overfill)# + late_evac_penalty)

    def _summarize_individual(
        self,
        individual: Individual,
        n_depots: int,
        durations_matrix: Dict[Tuple[int, int], float],
    ) -> Optional[Dict[str, Any]]:
        if _simulate_and_get_timings is None:
            return None
        sim = _simulate_and_get_timings(
            individual=individual,
            n_depots=n_depots,
            durations_matrix=durations_matrix,
            origin_by_bus=self._origin_by_bus,
            cap_by_bus=self._cap_by_bus,
            depots=self._depots_runtime,
            node_coords=self._node_coords,
            start_to_node_seconds=self._start_to_node_seconds,
            avg_speed_kmh=self._avg_speed_kmh,
            road_factor=self._road_factor,
            **self._service_params,
        )
        total_people = int(sim.get("total_people_evacuated", 0))
        denom = max(1, total_people)
        avg_wait = float(sim.get("total_wait_pm", 0.0)) / float(denom)
        makespan = float(sim.get("latest_evac_min", 0.0))
        total_overfill = float(sim.get("total_overfill", 0.0))
        late_evac_penalty = float(sim.get("late_evac_penalty", 0.0))
        return {
            "people_evacuated": total_people,
            "avg_wait_min": avg_wait,
            "makespan_min": makespan,
            "total_overfill": total_overfill,
            "late_evac_penalty": late_evac_penalty,
            "cost": avg_wait + makespan + total_overfill + late_evac_penalty,
        }

    def _maybe_log_snapshot(
        self,
        stats: Dict[str, Any],
        iteration: int,
        elapsed_s: float,
        best: Individual,
        best_cost: float,
        n_depots: int,
        durations_matrix: Dict[Tuple[int, int], float],
    ) -> None:
        summary = self._summarize_individual(best, n_depots, durations_matrix)
        if summary is None:
            return
        stats["progress"].append({
            "iteration": int(iteration),
            "elapsed_seconds": float(elapsed_s),
            "best_cost": float(best_cost),
            "avg_wait_min": float(summary["avg_wait_min"]),
            "makespan_min": float(summary["makespan_min"]),
            "people_evacuated": int(summary["people_evacuated"]),
            "total_overfill": float(summary["total_overfill"]),
            "late_evac_penalty": float(summary["late_evac_penalty"]),
        })
        print(
            f"ALNS it={iteration:6d}  "
            f"Cost={best_cost:8.2f}  "
            f"AvgWait={summary['avg_wait_min']:7.3f}  "
            f"Makespan={summary['makespan_min']:7.1f}  "
            f"Overfill={summary['total_overfill']:6.1f}  "
            f"LatePen={summary['late_evac_penalty']:6.1f}  "
            f"People={summary['people_evacuated']:4d}  "
            f"t={elapsed_s:7.2f}s"
        )

    # -----------------------------------------------------------------------
    # Operator weight update
    # -----------------------------------------------------------------------
    @staticmethod
    def _update_weights(
        weights: Dict[str, float],
        scores: Dict[str, float],
        counts: Dict[str, int],
        reaction: float,
        min_weight: float,
    ) -> None:
        for op, w in list(weights.items()):
            c = counts.get(op, 0)
            if c <= 0:
                weights[op] = max(min_weight, (1.0 - reaction) * w)
                continue
            avg_score = scores.get(op, 0.0) / float(c)
            weights[op] = max(min_weight, (1.0 - reaction) * w + reaction * avg_score)

    # -----------------------------------------------------------------------
    # Representation helpers
    # -----------------------------------------------------------------------
    @staticmethod
    def _count_total_stops(individual: Individual) -> int:
        return sum(len(trip.get("stops", [])) for sched in individual for trip in sched)

    def _individual_to_solution(self, individual: Individual, buses_count: int) -> List[List[Dict[str, Any]]]:
        solution: List[List[Dict[str, Any]]] = []
        for b in range(buses_count):
            bus_trips = []
            for trip in individual[b]:
                stops_with_counts = trip.get("stops", [])
                if not stops_with_counts:
                    continue
                
                # Fix: aggregate counts for duplicate nodes
                pickup_counts_dd = defaultdict(int)
                node_list = []
                seen = set()
                for n, c in stops_with_counts:
                    pickup_counts_dd[int(n)] += int(c)
                    if int(n) not in seen:
                        node_list.append(int(n))
                        seen.add(int(n))
                
                bus_trips.append({
                    "start_depot": int(trip.get("start_depot", 0)),
                    "stops": node_list,
                    "end_depot": int(trip.get("end_depot", 0)),
                    "pickup_counts": dict(pickup_counts_dd),
                })
            solution.append(bus_trips)
        return solution
        
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
            if self._start_to_node_seconds and bus_idx in self._start_to_node_seconds:
                secs_map = self._start_to_node_seconds[bus_idx]
                if first_node in secs_map:
                    return float(secs_map[first_node]) / 60.0
            if not self._node_coords or first_node not in self._node_coords:
                return None
            nlat, nlon = self._node_coords[first_node]
            km = _haversine_km(float(lat), float(lon), float(nlat), float(nlon))
            return (km / max(1e-6, self._avg_speed_kmh)) * 60.0 * self._road_factor
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
            if self._start_to_node_seconds and bus_idx in self._start_to_node_seconds:
                secs_map = self._start_to_node_seconds[bus_idx]
                if node_idx in secs_map:
                    return float(secs_map[node_idx]) / 60.0
            if not self._node_coords or node_idx not in self._node_coords:
                return None
            nlat, nlon = self._node_coords[node_idx]
            km = _haversine_km(float(lat), float(lon), float(nlat), float(nlon))
            return (km / max(1e-6, self._avg_speed_kmh)) * 60.0 * self._road_factor
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

    def _print_final_solution(self, final_solution: List[List[Dict[str, Any]]], n_depots, durations_matrix) -> None:
        def print_sol(title: str, sol: List[List[Dict[str, Any]]]) -> None:
            print(f"\n{title}")
            print("-" * 60)
            if not sol:
                print("  (No solution data)")
                return

            total_trips = 0
            for b_idx, bus_trips in enumerate(sol):
                if not bus_trips:
                    continue
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

                    if stops_data and isinstance(stops_data[0], (list, tuple)):
                        stops_formatted = [f"Node {n}({c})" for n, c in stops_data]
                    else:
                        stops_formatted = [f"Node {n}({counts.get(n, '?')})" for n in stops_data]

                    stops_str = " -> ".join(stops_formatted)
                    print(f"    Trip {t_idx}: Depot {start} -> [{stops_str}] -> Depot {end}")

            print(f"  Total Trips: {total_trips}")

        if self._debug_pendel_solution:
            print_sol("2️⃣  BASELINE SOLUTION (Pendelverkehr)", self._debug_pendel_solution)
        else:
            print("\n2️⃣  BASELINE SOLUTION (Pendelverkehr)")
            print("-" * 60)
            print("  (Not available / failed)")

        print_sol("3️⃣  FINAL ALNS SOLUTION", final_solution)

    # -----------------------------------------------------------------------
    # Fleet normalization (compatible with EA)
    # -----------------------------------------------------------------------
    def _resolve_fleet(
        self,
        vehicles: Optional[List[Dict[str, Any]]],
        buses_count: int,
        bus_capacity: int,
        n_depots: int,
        pickup_nodes: List[int],
    ) -> Tuple[List[int], List[Dict[str, Any]], int, List[Dict[str, Any]]]:
        if not vehicles:
            cap_by_bus = [int(bus_capacity) for _ in range(buses_count)]
            origin_by_bus = [{"kind": "depot", "index": 0} for _ in range(buses_count)]
            normalized = [{"id": None, "capacity": int(bus_capacity), "start": {"kind": "depot", "index": 0}} for _ in range(buses_count)]
            return cap_by_bus, origin_by_bus, buses_count, normalized

        pickup_set = set(pickup_nodes)
        cap_by_bus: List[int] = []
        origin_by_bus: List[Dict[str, Any]] = []
        normalized: List[Dict[str, Any]] = []

        for v in vehicles:
            vid = v.get("id")
            cap = int(v.get("capacity", 0))
            if cap <= 0:
                raise ValueError(f"Vehicle capacity must be > 0 (vehicle id={vid}).")

            has_sd = v.get("start_depot") is not None
            has_sn = v.get("start_node") is not None
            has_sc = v.get("start_coord") is not None
            if (has_sd + has_sn + has_sc) > 1:
                raise ValueError(f"Vehicle may specify only one of start_depot/start_node/start_coord (vehicle id={vid}).")

            if has_sd:
                sd = int(v["start_depot"])
                if not (0 <= sd < n_depots):
                    raise ValueError(f"start_depot out of range (vehicle id={vid}).")
                origin = {"kind": "depot", "index": sd}
                start_norm = {"kind": "depot", "index": sd}
            elif has_sn:
                sn = int(v["start_node"])
                if sn not in pickup_set:
                    raise ValueError(f"start_node {sn} not in pickup_nodes (vehicle id={vid}).")
                origin = {"kind": "node", "index": sn}
                start_norm = {"kind": "node", "index": sn}
            elif has_sc:
                coord = v["start_coord"]
                if not isinstance(coord, dict) or "lat" not in coord or "lon" not in coord:
                    raise ValueError(f"start_coord must be a dict with lat/lon (vehicle id={vid}).")
                lat = float(coord["lat"]); lon = float(coord["lon"])
                origin = {"kind": "coord", "lat": lat, "lon": lon}
                start_norm = {"kind": "coord", "lat": lat, "lon": lon}
            else:
                origin = {"kind": "depot", "index": 0}
                start_norm = {"kind": "depot", "index": 0}

            cap_by_bus.append(cap)
            origin_by_bus.append(origin)
            normalized.append({"id": vid, "capacity": cap, "start": start_norm})

        return cap_by_bus, origin_by_bus, len(vehicles), normalized

    def _origin_loc(self, bus_idx: int) -> Dict[str, Any]:
        o = (self._origin_by_bus or [{"kind": "depot", "index": 0}])[bus_idx]
        if o.get("kind") == "depot":
            return {"kind": "depot", "index": int(o.get("index", 0))}
        if o.get("kind") == "node":
            return {"kind": "node", "index": int(o.get("index", 0))}
        if o.get("kind") == "coord":
            return {"kind": "coord", "lat": float(o["lat"]), "lon": float(o["lon"])}
        return {"kind": "depot", "index": 0}

    # -----------------------------------------------------------------------
    # Travel helpers
    # -----------------------------------------------------------------------
    def _first_leg_minutes_from_coord(self, bus_idx: int, node: int) -> float:
        """
        For origin kind 'coord', EA uses:
          - a per-bus override (start_to_node_seconds), else
          - haversine * road_factor / avg_speed.
        Keep consistent here.
        """
        if self._start_to_node_seconds is not None:
            by_bus = self._start_to_node_seconds.get(int(bus_idx))
            if by_bus is not None:
                secs = by_bus.get(int(node))
                if secs is not None and math.isfinite(float(secs)):
                    return float(secs) / 60.0

        if self._node_coords is None or int(node) not in self._node_coords:
            return 9999.0
        lat2, lon2 = self._node_coords[int(node)]
        o = self._origin_loc(bus_idx)
        lat1 = float(o.get("lat", 0.0))
        lon1 = float(o.get("lon", 0.0))
        km = _haversine_km(lat1, lon1, lat2, lon2)
        return (km / max(1e-9, self._avg_speed_kmh)) * 60.0 * self._road_factor

    def _travel_minutes_between(
        self,
        from_loc: Dict[str, Any],
        to_loc: Dict[str, Any],
        n_depots: int,
        durations_matrix: Dict[Tuple[int, int], float],
        bus_idx: Optional[int] = None,
    ) -> float:
        fk = from_loc.get("kind", "depot")
        tk = to_loc.get("kind", "depot")

        # coord -> node via override/haversine
        if fk == "coord" and tk == "node" and bus_idx is not None:
            return self._first_leg_minutes_from_coord(bus_idx, int(to_loc["index"]))

        # coord -> coord/node/depot via haversine (fallback)
        if fk == "coord":
            if tk == "coord":
                km = _haversine_km(from_loc["lat"], from_loc["lon"], to_loc["lat"], to_loc["lon"])
                return (km / max(1e-6, self._avg_speed_kmh)) * 60.0 * self._road_factor
            if tk == "node":
                if not self._node_coords or int(to_loc["index"]) not in self._node_coords:
                    return 9999.0
                lat2, lon2 = self._node_coords[int(to_loc["index"])]
                km = _haversine_km(from_loc["lat"], from_loc["lon"], lat2, lon2)
                return (km / max(1e-6, self._avg_speed_kmh)) * 60.0 * self._road_factor
            if tk == "depot":
                if not self._depots_runtime or int(to_loc["index"]) >= len(self._depots_runtime):
                    return 9999.0
                lon2, lat2 = self._depots_runtime[int(to_loc["index"])]["coords"]
                km = _haversine_km(from_loc["lat"], from_loc["lon"], lat2, lon2)
                return (km / max(1e-6, self._avg_speed_kmh)) * 60.0 * self._road_factor

        # something -> coord (approx symmetric)
        if tk == "coord":
            return self._travel_minutes_between(to_loc, from_loc, n_depots, durations_matrix, bus_idx)

        def idx(loc: Dict[str, Any]) -> int:
            if loc.get("kind") == "depot":
                return int(loc["index"])
            return int(n_depots + int(loc["index"]))

        i = idx(from_loc)
        j = idx(to_loc)
        secs = durations_matrix.get((i, j), float("inf"))
        if not math.isfinite(float(secs)):
            return 9999.0
        return float(secs) / 60.0

    def _node_node_minutes(
        self,
        node_a: int,
        node_b: int,
        n_depots: int,
        durations_matrix: Dict[Tuple[int, int], float],
    ) -> float:
        key = (int(node_a), int(node_b))
        if key in self._node_node_minutes_cache:
            return self._node_node_minutes_cache[key]
        i = int(n_depots + int(node_a))
        j = int(n_depots + int(node_b))
        secs = durations_matrix.get((i, j), float("inf"))
        if not math.isfinite(float(secs)):
            mins = 9999.0
        else:
            mins = float(secs) / 60.0
        self._node_node_minutes_cache[key] = mins
        return mins

    # -----------------------------------------------------------------------
    # Initial solution (dispatcher seed + small shake)
    # -----------------------------------------------------------------------
    def _initial_solution(
        self,
        cfg: ALNSConfig,
        buses_count: int,
        bus_capacity: int,
        n_depots: int,
        pickup_nodes: List[int],
        demand_full: Dict[int, int],
        durations_matrix: Dict[Tuple[int, int], float],
        depots: List[Dict[str, Any]],
        facilities: List[Dict[str, Any]],
        precomputed_problem_data: Dict[str, Any],
        normalized_vehicles: List[Dict[str, Any]],
        deadline: Optional[float] = None,
    ) -> Individual:
        # 1) Seed from dispatcher baseline (strong starting point, used by EA population seeding)
        indiv: Optional[Individual] = None
        if cfg.seed_with_dispatcher and PendelverkehrShuttleAlgorithm is not None:
            try:
                baseline = PendelverkehrShuttleAlgorithm()
                baseline_result = baseline.run(
                    evacuation_zones_input=None,
                    buses_count=buses_count,
                    bus_capacity=bus_capacity,
                    vehicles=normalized_vehicles,
                    start_to_node_seconds=self._start_to_node_seconds,
                    avg_speed_kmh=self._avg_speed_kmh,
                    road_factor=self._road_factor,
                    precomputed_problem_data=precomputed_problem_data,
                    pick_rule=cfg.dispatcher_pick_rule,
                    # keep same service + penalty knobs for apples-to-apples (fitness)
                    **self._service_params,
                    penalty_factor=self._penalty_factor,
                    latest_evacuation_penalty_factor=self._latest_evacuation_penalty_factor,
                )
                best_solution = baseline_result.get("best_solution")
                if best_solution:
                    self._debug_pendel_solution = best_solution
                    indiv = self._convert_solution_to_individual(best_solution, buses_count=buses_count)
            except Exception:
                indiv = None

        # 2) Fallback: simple constructive
        if indiv is None:
            indiv = self._construct_initial_solution_basic(
                buses_count=buses_count,
                n_depots=n_depots,
                pickup_nodes=pickup_nodes,
                demand_full=demand_full,
                durations_matrix=durations_matrix,
            )

        # 3) Small shake around seed to avoid immediate local trap
        if cfg.initial_shake_cycles > 0:
            current = indiv
            for _ in range(int(cfg.initial_shake_cycles)):
                if deadline is not None and _now() >= float(deadline):
                    break
                total_stops = self._count_total_stops(current)
                if total_stops <= 0:
                    break
                q = max(1, int(0.12 * total_stops))
                partial, removed = self._destroy_random_stop(
                    current,
                    q=q,
                    buses_count=buses_count,
                    n_depots=n_depots,
                    durations_matrix=durations_matrix,
                )
                current = self._repair_greedy(
                    partial,
                    removed=removed,
                    cfg=cfg,
                    buses_count=buses_count,
                    n_depots=n_depots,
                    durations_matrix=durations_matrix,
                )
                current = self._repair(
                    current, buses_count, bus_capacity,
                    depots, facilities, n_depots, pickup_nodes, durations_matrix, demand_full
                )
            indiv = current

        return indiv

    @staticmethod
    def _convert_solution_to_individual(best_solution: List[List[Dict[str, Any]]], buses_count: int) -> Individual:
        indiv: Individual = [[] for _ in range(buses_count)]
        for b in range(min(buses_count, len(best_solution))):
            for trip in best_solution[b]:
                stops = []
                pickup_counts = trip.get("pickup_counts", {}) or {}
                for node in trip.get("stops", []) or []:
                    qty = int(pickup_counts.get(node, 0))
                    if qty > 0:
                        stops.append((int(node), int(qty)))
                if stops:
                    indiv[b].append({
                        "start_depot": int(trip.get("start_depot", 0)),
                        "end_depot": int(trip.get("end_depot", 0)),
                        "stops": stops,
                    })
        return indiv

    def _construct_initial_solution_basic(
        self,
        buses_count: int,
        n_depots: int,
        pickup_nodes: List[int],
        demand_full: Dict[int, int],
        durations_matrix: Dict[Tuple[int, int], float],
    ) -> Individual:
        """
        Basic seed if dispatcher baseline unavailable.
        - Create one-stop trips assigned to currently “shortest” bus (estimated finish time).
        """
        indiv: Individual = [[] for _ in range(buses_count)]
        depot_loads = [0 for _ in range(n_depots)]
        cap_by_bus = self._cap_by_bus or [80 for _ in range(buses_count)]
        origin_by_bus = self._origin_by_bus or [{"kind": "depot", "index": 0} for _ in range(buses_count)]

        # crude finish estimate: sum of (trip travel + service) so far
        finish_est = [0.0 for _ in range(buses_count)]

        for node in pickup_nodes:
            remaining = int(demand_full.get(node, 0))
            while remaining > 0:
                b = min(range(buses_count), key=lambda i: finish_est[i])
                cap = cap_by_bus[b]
                take = min(cap, remaining)

                if indiv[b]:
                    start_depot = int(indiv[b][-1].get("end_depot", 0))
                else:
                    o = origin_by_bus[b]
                    start_depot = int(o.get("index", 0)) if o.get("kind") == "depot" else 0

                end_depot = self._find_best_end_depot(
                    last_stop_node=node,
                    trip_load=take,
                    depot_loads=depot_loads,
                    n_depots=n_depots,
                    durations_matrix=durations_matrix,
                )

                indiv[b].append({"start_depot": start_depot, "stops": [(int(node), int(take))], "end_depot": int(end_depot)})
                if 0 <= end_depot < len(depot_loads):
                    depot_loads[end_depot] += take

                # rough duration update
                fin_add = self._estimate_single_stop_trip_minutes(
                    bus_idx=b,
                    start_depot=start_depot,
                    node=node,
                    end_depot=end_depot,
                    qty=take,
                    n_depots=n_depots,
                    durations_matrix=durations_matrix,
                )
                finish_est[b] += fin_add
                remaining -= take

        for b in range(buses_count):
            indiv[b] = self._fix_depot_connectivity(indiv[b], origin=self._origin_loc(b))
        return indiv

    def _estimate_single_stop_trip_minutes(
        self,
        bus_idx: int,
        start_depot: int,
        node: int,
        end_depot: int,
        qty: int,
        n_depots: int,
        durations_matrix: Dict[Tuple[int, int], float],
    ) -> float:
        base = float(self._service_params.get("service_time_base_min", 3.0))
        per = float(self._service_params.get("service_time_per_person_min", 20.0 / 60.0))
        dyn = bool(self._service_params.get("use_dynamic_service_time", False))
        cap = float((self._cap_by_bus or [qty])[bus_idx])

        # start -> node
        if bus_idx is not None and self._origin_loc(bus_idx).get("kind") == "coord" and not (self._origin_loc(bus_idx).get("kind") == "depot"):
            pass
        start_loc = {"kind": "depot", "index": int(start_depot)}
        node_loc = {"kind": "node", "index": int(node)}
        end_loc = {"kind": "depot", "index": int(end_depot)}

        t1 = self._travel_minutes_between(start_loc, node_loc, n_depots, durations_matrix, bus_idx)
        t2 = self._travel_minutes_between(node_loc, end_loc, n_depots, durations_matrix, bus_idx)

        svc_pick = base + (per * qty if dyn else per * cap)
        svc_drop = base + (per * qty if dyn else per * cap)
        return float(t1 + svc_pick + t2 + svc_drop)

    # -----------------------------------------------------------------------
    # SA temperature calibration
    # -----------------------------------------------------------------------
    def _calibrate_initial_temperature(
        self,
        cfg: ALNSConfig,
        current: Individual,
        current_cost: float,
        destroy_ops: Dict[str, Callable[..., Tuple[Individual, RemovedList]]],
        repair_ops: Dict[str, Callable[..., Individual]],
        buses_count: int,
        bus_capacity: int,
        depots: List[Dict[str, Any]],
        facilities: List[Dict[str, Any]],
        n_depots: int,
        pickup_nodes: List[int],
        durations_matrix: Dict[Tuple[int, int], float],
        demand_full: Dict[int, int],
        deadline: Optional[float] = None,
    ) -> float:
        """
        Choose T0 so that a typical uphill move is accepted with probability ~p0.
        This matters a lot for “convergence feel” under heavy evaluation cost.
        """
        if cfg.sa_samples_for_T0 <= 0:
            return max(cfg.sa_min_temp, 0.05 * max(1e-9, current_cost))

        p0 = _clamp(cfg.sa_initial_accept_prob, 1e-3, 0.999)
        deltas: List[float] = []
        # sample a few cheap candidates (small destroy + greedy repair)
        total_stops = self._count_total_stops(current)
        if total_stops <= 0:
            return max(cfg.sa_min_temp, 0.05 * max(1e-9, current_cost))

        q_small = max(1, int(0.08 * total_stops))
        d_keys = list(destroy_ops.keys())
        r_keys = list(repair_ops.keys())

        for _ in range(int(cfg.sa_samples_for_T0)):
            if deadline is not None and _now() >= float(deadline):
                break
            d_name = random.choice(d_keys)
            r_name = random.choice(r_keys)
            partial, removed = destroy_ops[d_name](current, q=q_small, buses_count=buses_count, n_depots=n_depots, durations_matrix=durations_matrix)
            cand = repair_ops[r_name](partial, removed=removed, cfg=cfg, buses_count=buses_count, n_depots=n_depots, durations_matrix=durations_matrix)
            cand = self._repair(
                cand, buses_count, bus_capacity,
                depots, facilities, n_depots, pickup_nodes, durations_matrix, demand_full
            )
            c = self._evaluate_fitness(
                cand, buses_count, bus_capacity, depots, facilities, n_depots,
                durations_matrix, demand_full,
                penalty_factor=self._penalty_factor,
                latest_evacuation_penalty_factor=self._latest_evacuation_penalty_factor,
            )
            delta = c - current_cost
            if delta > 1e-9 and math.isfinite(delta):
                deltas.append(delta)

        if not deltas:
            return max(cfg.sa_min_temp, 0.05 * max(1e-9, current_cost))

        mean_uphill = sum(deltas) / float(len(deltas))
        # Solve p0 = exp(-mean_uphill / T0)  =>  T0 = -mean_uphill / ln(p0)
        T0 = -mean_uphill / max(1e-12, math.log(p0))
        if not math.isfinite(T0) or T0 <= 0:
            T0 = 0.05 * max(1e-9, current_cost)
        return max(cfg.sa_min_temp, float(T0))

    # -----------------------------------------------------------------------
    # Memetic polisher integration
    # -----------------------------------------------------------------------

    def _build_polisher(
        self,
        cfg: ALNSConfig,
        buses_count: int,
        bus_capacity: int,
        depots: List[Dict[str, Any]],
        facilities: List[Dict[str, Any]],
        n_depots: int,
        pickup_nodes: List[int],
        durations_matrix: Dict[Tuple[int, int], float],
        demand_full: Dict[int, int],
    ) -> Optional[Any]:
        """Construct the MemeticImprover exactly like EA/MA does (if available)."""
        if not cfg.use_memetic_polish:
            return None
        if MemeticImprover is None:
            return None
        try:
            polisher = MemeticImprover(
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
                penalty_factor=self._penalty_factor,
                latest_evacuation_penalty_factor=self._latest_evacuation_penalty_factor,
                origin_by_bus=self._origin_by_bus,
                cap_by_bus=self._cap_by_bus,
                node_coords=self._node_coords,
                start_to_node_seconds=self._start_to_node_seconds,
                avg_speed_kmh=self._avg_speed_kmh,
                road_factor=self._road_factor,
                **self._service_params,
            )
            return polisher
        except Exception:
            return None

    def _polish_with_memetic(
        self,
        polisher: Any,
        cfg: ALNSConfig,
        candidate: Individual,
        candidate_cost: float,
        *,
        context: Optional[Dict[str, Any]],
        buses_count: int,
        bus_capacity: int,
        depots: List[Dict[str, Any]],
        facilities: List[Dict[str, Any]],
        n_depots: int,
        pickup_nodes: List[int],
        durations_matrix: Dict[Tuple[int, int], float],
        demand_full: Dict[int, int],
    ) -> Tuple[Individual, float, float, Dict[str, float]]:
        """Tiny-budget MA local search (same MemeticImprover interface as EA/MA)."""
        before = float(candidate_cost)
        ls_params = {
            "time_limit_seconds": float(cfg.polish_time_limit_seconds),
            "max_iterations": int(cfg.polish_max_iterations),
            "candidate_set_size": int(cfg.polish_candidate_set_size),
            # keep the shake off (ALNS already provides large-scale shake)
            "use_alns_shake": False,
        }

        try:
            improved = polisher.improve(candidate, ls_params, context=context)
        except TypeError:
            # Backward compat if context isn't supported
            improved = polisher.improve(candidate, ls_params)

        # Safety net: enforce feasibility invariants.
        improved = self._repair(
            improved, buses_count, bus_capacity,
            depots, facilities, n_depots, pickup_nodes, durations_matrix, demand_full
        )

        improved_cost = self._evaluate_fitness(
            improved, buses_count, bus_capacity, depots, facilities, n_depots,
            durations_matrix, demand_full,
            penalty_factor=self._penalty_factor,
            latest_evacuation_penalty_factor=self._latest_evacuation_penalty_factor,
        )

        gain = max(0.0, before - float(improved_cost))
        stats: Dict[str, float] = {}
        try:
            if hasattr(polisher, "get_last_run_stats"):
                stats = polisher.get_last_run_stats() or {}
        except Exception:
            stats = {}

        return improved, float(improved_cost), float(gain), stats



    # -----------------------------------------------------------------------
    # Destroy operators
    # -----------------------------------------------------------------------
    def _collect_stop_refs(self, individual: Individual) -> List[Tuple[int, int, int]]:
        refs: List[Tuple[int, int, int]] = []
        for b, sched in enumerate(individual):
            for t, trip in enumerate(sched):
                for s, _ in enumerate(trip.get("stops", [])):
                    refs.append((b, t, s))
        return refs

    def _destroy_random_stop(
        self,
        individual: Individual,
        q: int,
        buses_count: int,
        n_depots: int,
        durations_matrix: Dict[Tuple[int, int], float],
    ) -> Tuple[Individual, RemovedList]:
        partial = _fast_clone_individual(individual)
        refs = self._collect_stop_refs(partial)
        if not refs:
            return partial, []
        q = min(int(q), len(refs))
        chosen = random.sample(refs, q)
        chosen.sort(reverse=True)

        removed: RemovedList = []
        for b, t, s in chosen:
            stops = partial[b][t]["stops"]
            if 0 <= s < len(stops):
                node, qty = stops.pop(s)
                removed.append((int(node), int(qty)))

        for b in range(len(partial)):
            partial[b] = [tr for tr in partial[b] if tr.get("stops")]
            partial[b] = self._fix_depot_connectivity(partial[b], origin=self._origin_loc(b))
        return partial, removed

    def _destroy_route(
        self,
        individual: Individual,
        q: int,
        buses_count: int,
        n_depots: int,
        durations_matrix: Dict[Tuple[int, int], float],
    ) -> Tuple[Individual, RemovedList]:
        partial = _fast_clone_individual(individual)
        removed: RemovedList = []
        # random order of trips
        trips = [(b, t) for b, sched in enumerate(partial) for t, tr in enumerate(sched) if tr.get("stops")]
        random.shuffle(trips)
        for (b, t) in trips:
            if len(removed) >= q:
                break
            tr = partial[b][t]
            for node, qty in tr.get("stops", []):
                removed.append((int(node), int(qty)))
            partial[b][t]["stops"] = []
        # cleanup
        for b in range(len(partial)):
            partial[b] = [tr for tr in partial[b] if tr.get("stops")]
            partial[b] = self._fix_depot_connectivity(partial[b], origin=self._origin_loc(b))
        # if not enough removed, top up with random stop
        if len(removed) < q:
            partial, removed2 = self._destroy_random_stop(partial, q=q - len(removed), buses_count=buses_count, n_depots=n_depots, durations_matrix=durations_matrix)
            removed.extend(removed2)
        return partial, removed

    def _destroy_worst_detour(
        self,
        individual: Individual,
        q: int,
        buses_count: int,
        n_depots: int,
        durations_matrix: Dict[Tuple[int, int], float],
    ) -> Tuple[Individual, RemovedList]:
        partial = _fast_clone_individual(individual)
        candidates: List[Tuple[float, Tuple[int, int, int]]] = []
        for b, sched in enumerate(partial):
            for t, trip in enumerate(sched):
                for s in range(len(trip.get("stops", []))):
                    detour = self._estimate_stop_detour(partial, b, t, s, n_depots, durations_matrix)
                    candidates.append((detour, (b, t, s)))
        if not candidates:
            return partial, []

        candidates.sort(key=lambda x: x[0], reverse=True)
        chosen = [ref for _, ref in candidates[: min(int(q), len(candidates))]]
        chosen.sort(reverse=True)

        removed: RemovedList = []
        for b, t, s in chosen:
            stops = partial[b][t]["stops"]
            if 0 <= s < len(stops):
                node, qty = stops.pop(s)
                removed.append((int(node), int(qty)))

        for b in range(len(partial)):
            partial[b] = [tr for tr in partial[b] if tr.get("stops")]
            partial[b] = self._fix_depot_connectivity(partial[b], origin=self._origin_loc(b))
        return partial, removed

    def _destroy_shaw_related(
        self,
        individual: Individual,
        q: int,
        buses_count: int,
        n_depots: int,
        durations_matrix: Dict[Tuple[int, int], float],
    ) -> Tuple[Individual, RemovedList]:
        """
        Shaw removal: pick a seed stop, then remove related stops (spatial + same-route bias).
        """
        partial = _fast_clone_individual(individual)
        refs = self._collect_stop_refs(partial)
        if not refs:
            return partial, []

        # seed
        sb, st, ss = random.choice(refs)
        seed_node, _ = partial[sb][st]["stops"][ss]

        # relatedness score: lower = more related
        rel: List[Tuple[float, Tuple[int, int, int]]] = []
        for b, t, s in refs:
            node, _ = partial[b][t]["stops"][s]
            nn = self._node_node_minutes(int(seed_node), int(node), n_depots, durations_matrix)
            same_bus = 0.0 if b == sb else 2.0
            same_trip = 0.0 if (b == sb and t == st) else 1.0
            # small random tie-breaker
            score = nn + 0.5 * same_bus + 0.2 * same_trip + random.random() * 0.05
            rel.append((score, (b, t, s)))

        rel.sort(key=lambda x: x[0])

        # select q using “biased random” over sorted list (classic Shaw trick)
        p = 6.0  # higher -> stronger bias to most-related
        chosen: List[Tuple[int, int, int]] = []
        rel_refs = [r for _, r in rel]
        while len(chosen) < min(int(q), len(rel_refs)):
            idx = int((random.random() ** p) * len(rel_refs))
            ref = rel_refs.pop(idx)
            chosen.append(ref)
            if not rel_refs:
                break

        chosen = list(dict.fromkeys(chosen))
        chosen.sort(reverse=True)

        removed: RemovedList = []
        for b, t, s in chosen:
            if b >= len(partial) or t >= len(partial[b]):
                continue
            stops = partial[b][t]["stops"]
            if 0 <= s < len(stops):
                node, qty = stops.pop(s)
                removed.append((int(node), int(qty)))

        for b in range(len(partial)):
            partial[b] = [tr for tr in partial[b] if tr.get("stops")]
            partial[b] = self._fix_depot_connectivity(partial[b], origin=self._origin_loc(b))
        return partial, removed

    def _destroy_node_cluster(
        self,
        individual: Individual,
        q: int,
        buses_count: int,
        n_depots: int,
        durations_matrix: Dict[Tuple[int, int], float],
    ) -> Tuple[Individual, RemovedList]:
        """
        Remove (nearly) all occurrences of a spatial cluster of nodes.
        This is important for split-demand problems to allow the algorithm to *re-decide the split*.
        """
        partial = _fast_clone_individual(individual)
        refs = self._collect_stop_refs(partial)
        if not refs:
            return partial, []

        # choose seed node from an existing stop
        sb, st, ss = random.choice(refs)
        seed_node, _ = partial[sb][st]["stops"][ss]

        # build node frequency and pick cluster nodes by nearest travel time
        node_set = {int(partial[b][t]["stops"][s][0]) for (b, t, s) in refs}
        dists = [(self._node_node_minutes(int(seed_node), int(n), n_depots, durations_matrix), int(n)) for n in node_set]
        dists.sort(key=lambda x: x[0])

        # take k nodes in cluster such that we have at least q stop-entries (roughly)
        cluster_nodes: List[int] = []
        removed: RemovedList = []

        for _, node in dists:
            cluster_nodes.append(int(node))
            # check if enough occurrences
            occ = sum(1 for b, t, s in refs if int(partial[b][t]["stops"][s][0]) in cluster_nodes)
            if occ >= q or len(cluster_nodes) >= 6:
                break

        chosen_refs = [(b, t, s) for (b, t, s) in refs if int(partial[b][t]["stops"][s][0]) in cluster_nodes]
        # maybe more than q, randomly subselect but keep diversity
        if len(chosen_refs) > q:
            chosen_refs = random.sample(chosen_refs, int(q))
        chosen_refs.sort(reverse=True)

        for b, t, s in chosen_refs:
            stops = partial[b][t]["stops"]
            if 0 <= s < len(stops):
                node, qty = stops.pop(s)
                removed.append((int(node), int(qty)))

        for b in range(len(partial)):
            partial[b] = [tr for tr in partial[b] if tr.get("stops")]
            partial[b] = self._fix_depot_connectivity(partial[b], origin=self._origin_loc(b))
        return partial, removed

    def _destroy_bottleneck(
        self,
        individual: Individual,
        q: int,
        buses_count: int,
        n_depots: int,
        durations_matrix: Dict[Tuple[int, int], float],
    ) -> Tuple[Individual, RemovedList]:
        """
        Makespan-aware ruin:
        - estimate which bus finishes last
        - remove stop-entries from the *end* of its schedule (late crumbs),
          plus a few random ones on that bus.
        """
        partial = _fast_clone_individual(individual)
        if not partial:
            return partial, []

        finish = self._estimate_bus_finish_times(partial, n_depots, durations_matrix)
        if not finish:
            return self._destroy_random_stop(individual, q=q, buses_count=buses_count, n_depots=n_depots, durations_matrix=durations_matrix)

        b_star = int(max(range(len(finish)), key=lambda i: finish[i]))
        # collect stop refs from that bus, bias towards late (last trips/last stops)
        refs: List[Tuple[int, int, int]] = []
        for t, trip in enumerate(partial[b_star]):
            for s in range(len(trip.get("stops", []))):
                refs.append((b_star, t, s))
        if not refs:
            return self._destroy_random_stop(individual, q=q, buses_count=buses_count, n_depots=n_depots, durations_matrix=durations_matrix)

        # bias: take more from later part
        refs_sorted = sorted(refs, key=lambda x: (x[1], x[2]), reverse=True)
        chosen = refs_sorted[: max(1, int(0.7 * q))]
        # top up with random from same bus (diversification)
        remaining = int(q) - len(chosen)
        if remaining > 0:
            pool = [r for r in refs if r not in chosen]
            if pool:
                chosen.extend(random.sample(pool, min(remaining, len(pool))))

        chosen = list(dict.fromkeys(chosen))
        chosen.sort(reverse=True)

        removed: RemovedList = []
        for b, t, s in chosen:
            stops = partial[b][t]["stops"]
            if 0 <= s < len(stops):
                node, qty = stops.pop(s)
                removed.append((int(node), int(qty)))

        # if still under q, remove random globally
        if len(removed) < q:
            partial2, removed2 = self._destroy_random_stop(partial, q=q - len(removed), buses_count=buses_count, n_depots=n_depots, durations_matrix=durations_matrix)
            partial = partial2
            removed.extend(removed2)

        for b in range(len(partial)):
            partial[b] = [tr for tr in partial[b] if tr.get("stops")]
            partial[b] = self._fix_depot_connectivity(partial[b], origin=self._origin_loc(b))
        return partial, removed

    def _estimate_stop_detour(
        self,
        individual: Individual,
        bus_idx: int,
        trip_idx: int,
        stop_idx: int,
        n_depots: int,
        durations_matrix: Dict[Tuple[int, int], float],
    ) -> float:
        trip = individual[bus_idx][trip_idx]
        stops = trip.get("stops", [])
        if not (0 <= stop_idx < len(stops)):
            return 0.0
        node, _ = stops[stop_idx]

        if stop_idx > 0:
            prev_node, _ = stops[stop_idx - 1]
            prev_loc = {"kind": "node", "index": int(prev_node)}
        else:
            if trip_idx == 0:
                prev_loc = self._origin_loc(bus_idx)
            else:
                prev_loc = {"kind": "depot", "index": int(trip.get("start_depot", 0))}

        if stop_idx < len(stops) - 1:
            next_node, _ = stops[stop_idx + 1]
            next_loc = {"kind": "node", "index": int(next_node)}
        else:
            next_loc = {"kind": "depot", "index": int(trip.get("end_depot", 0))}

        node_loc = {"kind": "node", "index": int(node)}

        a = self._travel_minutes_between(prev_loc, node_loc, n_depots, durations_matrix, bus_idx)
        b = self._travel_minutes_between(node_loc, next_loc, n_depots, durations_matrix, bus_idx)
        c = self._travel_minutes_between(prev_loc, next_loc, n_depots, durations_matrix, bus_idx)
        return float(max(0.0, (a + b - c)))

    # -----------------------------------------------------------------------
    # Repair (ALNS recreate operators)
    # -----------------------------------------------------------------------
    def _repair_greedy(
        self,
        partial: Individual,
        removed: RemovedList,
        cfg: ALNSConfig,
        buses_count: int,
        n_depots: int,
        durations_matrix: Dict[Tuple[int, int], float],
    ) -> Individual:
        candidate = _fast_clone_individual(partial)
        # split large chunks to max capacity for more stable insertion
        max_cap = max(self._cap_by_bus or [80])
        pool = self._split_removed(removed, max_chunk=max_cap)

        # keep a dynamic estimate for slack-based scoring
        bus_finish_est = self._estimate_bus_finish_times(candidate, n_depots, durations_matrix)

        random.shuffle(pool)
        for node, qty in pool:
            self._insert_quantity_best(
                candidate,
                node=int(node),
                qty=int(qty),
                cfg=cfg,
                buses_count=buses_count,
                n_depots=n_depots,
                durations_matrix=durations_matrix,
                bus_finish_est=bus_finish_est,
            )

        # final cleanup
        for b in range(buses_count):
            candidate[b] = [tr for tr in candidate[b] if tr.get("stops")]
            candidate[b] = self._fix_depot_connectivity(candidate[b], origin=self._origin_loc(b))
        return candidate

    def _repair_regret_k(
        self,
        partial: Individual,
        removed: RemovedList,
        cfg: ALNSConfig,
        buses_count: int,
        n_depots: int,
        durations_matrix: Dict[Tuple[int, int], float],
        k: int = 2,
    ) -> Individual:
        candidate = _fast_clone_individual(partial)
        max_cap = max(self._cap_by_bus or [80])
        pool = self._split_removed(removed, max_chunk=max_cap)

        # dynamic slack estimates
        bus_finish_est = self._estimate_bus_finish_times(candidate, n_depots, durations_matrix)

        # regret loop
        pool = [(int(n), int(q)) for (n, q) in pool if int(q) > 0]
        while pool:
            best_regret = -1.0
            best_i = 0
            best_plan = None  # (node, qty, insertion_plan)

            for i, (node, qty) in enumerate(pool):
                options = self._best_insertion_options(
                    candidate,
                    node=node,
                    qty=qty,
                    cfg=cfg,
                    buses_count=buses_count,
                    n_depots=n_depots,
                    durations_matrix=durations_matrix,
                    bus_finish_est=bus_finish_est,
                    top_k=max(2, k),
                )

                if not options:
                    # force trip creation
                    regret = 9999.0
                    plan = None
                else:
                    # regret = sum_{j=2..k} (score_j - score_1)
                    score1 = options[0][0]
                    regret = 0.0
                    for j in range(1, min(k, len(options))):
                        regret += (options[j][0] - score1)
                    plan = options[0][1]

                if regret > best_regret:
                    best_regret = regret
                    best_i = i
                    best_plan = (node, qty, plan)

            node, qty, plan = best_plan  # type: ignore[misc]

            inserted = self._apply_best_plan(
                candidate,
                node=node,
                qty=qty,
                plan=plan,
                cfg=cfg,
                buses_count=buses_count,
                n_depots=n_depots,
                durations_matrix=durations_matrix,
                bus_finish_est=bus_finish_est,
            )

            # remove from pool, re-add remainder if partial insert happened
            pool.pop(best_i)
            if inserted < qty:
                pool.append((node, qty - inserted))

        for b in range(buses_count):
            candidate[b] = [tr for tr in candidate[b] if tr.get("stops")]
            candidate[b] = self._fix_depot_connectivity(candidate[b], origin=self._origin_loc(b))
        return candidate

    @staticmethod
    def _split_removed(removed: RemovedList, max_chunk: int) -> RemovedList:
        out: RemovedList = []
        for n, q in removed:
            q = int(q)
            while q > max_chunk:
                out.append((int(n), int(max_chunk)))
                q -= max_chunk
            if q > 0:
                out.append((int(n), int(q)))
        return out

    # ----- insertion scoring & application -----

    def _service_time_pickup(self, bus_idx: int, qty: int) -> float:
        base = float(self._service_params.get("service_time_base_min", 3.0))
        per = float(self._service_params.get("service_time_per_person_min", 20.0 / 60.0))
        dyn = bool(self._service_params.get("use_dynamic_service_time", False))
        cap = float((self._cap_by_bus or [qty])[bus_idx])
        return float(base + (per * qty if dyn else per * cap))

    def _service_time_dropoff_delta(self, bus_idx: int, qty_delta: int) -> float:
        """Incremental dropoff time when adding qty_delta people to an existing trip."""
        per = float(self._service_params.get("service_time_per_person_min", 20.0 / 60.0))
        dyn = bool(self._service_params.get("use_dynamic_service_time", False))
        if dyn:
            return float(per * qty_delta)
        return 0.0

    def _estimate_bus_finish_times(
        self,
        individual: Individual,
        n_depots: int,
        durations_matrix: Dict[Tuple[int, int], float],
    ) -> List[float]:
        """
        Cheap schedule estimator (minutes).
        Not identical to full simulation, but consistent enough to guide insertion.
        """
        base = float(self._service_params.get("service_time_base_min", 3.0))
        per = float(self._service_params.get("service_time_per_person_min", 20.0 / 60.0))
        dyn = bool(self._service_params.get("use_dynamic_service_time", False))

        caps = self._cap_by_bus or [80 for _ in range(len(individual))]

        finish: List[float] = []
        for b, sched in enumerate(individual):
            t_cur = 0.0
            # first trip start is origin (depot/node/coord), but trip dict stores start_depot anyway.
            # We'll use trip's start_depot and treat origin as “close enough” for a heuristic.
            for ti, trip in enumerate(sched):
                stops = trip.get("stops", [])
                if not stops:
                    continue

                # start location
                if ti == 0:
                    start_loc = self._origin_loc(b)
                    if start_loc.get("kind") == "depot":
                        start_loc = {"kind": "depot", "index": int(start_loc.get("index", 0))}
                    # else node/coord allowed
                else:
                    start_loc = {"kind": "depot", "index": int(trip.get("start_depot", 0))}

                # traverse stops
                prev = start_loc
                trip_load = 0
                for (node, qty) in stops:
                    node_loc = {"kind": "node", "index": int(node)}
                    t_cur += self._travel_minutes_between(prev, node_loc, n_depots, durations_matrix, b)
                    trip_load += int(qty)
                    # pickup service
                    svc = base + (per * int(qty) if dyn else per * caps[b])
                    t_cur += svc
                    prev = node_loc

                # end depot
                end_loc = {"kind": "depot", "index": int(trip.get("end_depot", 0))}
                t_cur += self._travel_minutes_between(prev, end_loc, n_depots, durations_matrix, b)
                # dropoff service (modeled as base + per*load or base + per*cap)
                drop = base + (per * trip_load if dyn else per * caps[b])
                t_cur += drop

            finish.append(float(t_cur))
        return finish

    def _best_insertion_options(
        self,
        individual: Individual,
        node: int,
        qty: int,
        cfg: ALNSConfig,
        buses_count: int,
        n_depots: int,
        durations_matrix: Dict[Tuple[int, int], float],
        bus_finish_est: List[float],
        top_k: int = 3,
    ) -> List[Tuple[float, Tuple[int, int, int, int, bool]]]:
        """
        Returns a list [(score, plan), ...] sorted by score ascending.
        plan = (bus, trip, pos_or_stopidx, take, augment_flag)
        """
        caps = self._cap_by_bus or [80 for _ in range(buses_count)]
        node_loc = {"kind": "node", "index": int(node)}
        makespan_est = max(bus_finish_est) if bus_finish_est else 0.0

        options: List[Tuple[float, Tuple[int, int, int, int, bool]]] = []

        for b in range(buses_count):
            cap = int(caps[b])
            for t, trip in enumerate(individual[b]):
                stops = trip.get("stops", [])
                if not stops:
                    continue
                load = sum(int(c) for _, c in stops)
                free = cap - load
                if free <= 0:
                    continue
                take = min(int(qty), int(free))
                if take <= 0:
                    continue

                # (A) augment existing stop of same node (no travel detour, no extra base)
                for s_idx, (n2, q2) in enumerate(stops):
                    if int(n2) == int(node):
                        delta_time = 0.0
                        # pickup service delta: dyn -> per*delta, else 0
                        dyn = bool(self._service_params.get("use_dynamic_service_time", False))
                        per = float(self._service_params.get("service_time_per_person_min", 20.0/60.0))
                        if dyn:
                            delta_time += per * take
                            delta_time += per * take  # dropoff delta too
                        # makespan penalty
                        pred_finish = bus_finish_est[b] + delta_time
                        pen = max(0.0, pred_finish - makespan_est)
                        score = (delta_time + cfg.insertion_makespan_penalty * pen) / max(1, take)
                        # slight late-trip penalty
                        score += cfg.insertion_late_trip_penalty * float(t)
                        # noise
                        score *= (1.0 + (random.random() - 0.5) * 2.0 * cfg.insertion_noise)
                        options.append((float(score), (b, t, s_idx, take, True)))

                # (B) insert as an additional stop at some position
                for pos in range(len(stops) + 1):
                    prev_loc = self._prev_loc_for_pos(individual, b, t, pos)
                    next_loc = self._next_loc_for_pos(individual, b, t, pos)
                    a = self._travel_minutes_between(prev_loc, node_loc, n_depots, durations_matrix, b)
                    b2 = self._travel_minutes_between(node_loc, next_loc, n_depots, durations_matrix, b)
                    c = self._travel_minutes_between(prev_loc, next_loc, n_depots, durations_matrix, b)
                    detour = max(0.0, (a + b2 - c))

                    delta_time = detour + self._service_time_pickup(b, take) + self._service_time_dropoff_delta(b, take)
                    pred_finish = bus_finish_est[b] + delta_time
                    pen = max(0.0, pred_finish - makespan_est)
                    score = (delta_time + cfg.insertion_makespan_penalty * pen) / max(1, take)
                    score += cfg.insertion_late_trip_penalty * float(t)
                    score *= (1.0 + (random.random() - 0.5) * 2.0 * cfg.insertion_noise)
                    options.append((float(score), (b, t, pos, take, False)))

        if not options:
            return []
        options.sort(key=lambda x: x[0])
        return options[: max(1, int(top_k))]

    def _apply_best_plan(
        self,
        individual: Individual,
        node: int,
        qty: int,
        plan: Optional[Tuple[int, int, int, int, bool]],
        cfg: ALNSConfig,
        buses_count: int,
        n_depots: int,
        durations_matrix: Dict[Tuple[int, int], float],
        bus_finish_est: List[float],
    ) -> int:
        """
        Apply the chosen plan; if plan is None, create a trip.
        Returns the quantity inserted (may be < qty).
        """
        if qty <= 0:
            return 0
        caps = self._cap_by_bus or [80 for _ in range(buses_count)]

        if plan is not None:
            b, t, pos, take, augment = plan
            take = max(0, min(int(take), int(qty)))
            if take <= 0:
                return 0
            trip = individual[b][t]
            if augment:
                # pos is stop index
                n0, q0 = trip["stops"][pos]
                trip["stops"][pos] = (int(n0), int(q0) + int(take))
                bus_finish_est[b] += self._service_time_dropoff_delta(b, take)
                # dyn pickup delta already accounted in same function
                dyn = bool(self._service_params.get("use_dynamic_service_time", False))
                if dyn:
                    per = float(self._service_params.get("service_time_per_person_min", 20.0/60.0))
                    bus_finish_est[b] += per * take
            else:
                trip["stops"].insert(int(pos), (int(node), int(take)))
                bus_finish_est[b] += self._service_time_pickup(b, take)
                bus_finish_est[b] += self._service_time_dropoff_delta(b, take)
            return int(take)

        # No feasible insertion: create a trip on the slackest bus
        b_new = int(min(range(buses_count), key=lambda i: bus_finish_est[i] if i < len(bus_finish_est) else 0.0))
        cap = int(caps[b_new])
        take = min(int(qty), cap)
        if take <= 0:
            return 0

        if individual[b_new]:
            start_depot = int(individual[b_new][-1].get("end_depot", 0))
        else:
            o = self._origin_loc(b_new)
            start_depot = int(o.get("index", 0)) if o.get("kind") == "depot" else 0

        depot_loads = self._calculate_depot_loads(individual)
        end_depot = self._find_best_end_depot(int(node), int(take), depot_loads, n_depots, durations_matrix)
        individual[b_new].append({"start_depot": start_depot, "stops": [(int(node), int(take))], "end_depot": int(end_depot)})

        # rough estimate update
        bus_finish_est[b_new] += self._estimate_single_stop_trip_minutes(
            bus_idx=b_new, start_depot=start_depot, node=node, end_depot=end_depot, qty=take,
            n_depots=n_depots, durations_matrix=durations_matrix
        )
        return int(take)

    def _insert_quantity_best(
        self,
        individual: Individual,
        node: int,
        qty: int,
        cfg: ALNSConfig,
        buses_count: int,
        n_depots: int,
        durations_matrix: Dict[Tuple[int, int], float],
        bus_finish_est: List[float],
    ) -> None:
        remaining = int(qty)
        while remaining > 0:
            options = self._best_insertion_options(
                individual=individual,
                node=int(node),
                qty=int(remaining),
                cfg=cfg,
                buses_count=buses_count,
                n_depots=n_depots,
                durations_matrix=durations_matrix,
                bus_finish_est=bus_finish_est,
                top_k=1,
            )
            plan = options[0][1] if options else None
            inserted = self._apply_best_plan(
                individual=individual,
                node=int(node),
                qty=int(remaining),
                plan=plan,
                cfg=cfg,
                buses_count=buses_count,
                n_depots=n_depots,
                durations_matrix=durations_matrix,
                bus_finish_est=bus_finish_est,
            )
            if inserted <= 0:
                # Emergency fallback: force a trip on some bus
                inserted = self._apply_best_plan(
                    individual=individual,
                    node=int(node),
                    qty=int(remaining),
                    plan=None,
                    cfg=cfg,
                    buses_count=buses_count,
                    n_depots=n_depots,
                    durations_matrix=durations_matrix,
                    bus_finish_est=bus_finish_est,
                )
                if inserted <= 0:
                    break
            remaining -= inserted

        # periodic cleanup
        for b in range(buses_count):
            individual[b] = [tr for tr in individual[b] if tr.get("stops")]
            individual[b] = self._fix_depot_connectivity(individual[b], origin=self._origin_loc(b))

    def _prev_loc_for_pos(self, individual: Individual, bus_idx: int, trip_idx: int, pos: int) -> Dict[str, Any]:
        trip = individual[bus_idx][trip_idx]
        stops = trip.get("stops", [])
        if pos > 0:
            prev_node, _ = stops[pos - 1]
            return {"kind": "node", "index": int(prev_node)}
        if trip_idx == 0:
            return self._origin_loc(bus_idx)
        return {"kind": "depot", "index": int(trip.get("start_depot", 0))}

    def _next_loc_for_pos(self, individual: Individual, bus_idx: int, trip_idx: int, pos: int) -> Dict[str, Any]:
        trip = individual[bus_idx][trip_idx]
        stops = trip.get("stops", [])
        if pos < len(stops):
            next_node, _ = stops[pos]
            return {"kind": "node", "index": int(next_node)}
        return {"kind": "depot", "index": int(trip.get("end_depot", 0))}

    # -----------------------------------------------------------------------
    # Repair / feasibility (strong safety net)
    # -----------------------------------------------------------------------
    def _calculate_depot_loads(self, individual: Individual) -> List[int]:
        if not self._depots_runtime:
            return []
        depot_loads = [0] * len(self._depots_runtime)
        for sched in individual:
            for trip in sched:
                d = int(trip.get("end_depot", 0))
                if 0 <= d < len(depot_loads):
                    depot_loads[d] += sum(int(c) for _, c in trip.get("stops", []))
        return depot_loads

    def _fix_depot_connectivity(self, bus_trips: List[Trip], origin: Optional[Dict[str, Any]] = None) -> List[Trip]:
        if not bus_trips:
            return []

        if origin is None:
            origin_kind = "depot"
            origin_index = 0
        else:
            origin_kind = origin.get("kind", "depot")
            origin_index = origin.get("index", 0)

        if origin_kind == "depot":
            bus_trips[0]["start_depot"] = int(origin_index)
        else:
            bus_trips[0].setdefault("start_depot", 0)
            if bus_trips[0]["start_depot"] is None:
                bus_trips[0]["start_depot"] = 0

        for i in range(1, len(bus_trips)):
            bus_trips[i]["start_depot"] = int(bus_trips[i - 1].get("end_depot", 0))
        return bus_trips

    def _repair(
        self,
        individual: Individual,
        buses_count: int,
        bus_capacity: int,
        depots: List[Dict[str, Any]],
        facilities: List[Dict[str, Any]],
        n_depots: int,
        pickup_nodes: List[int],
        durations_matrix: Dict[Tuple[int, int], float],
        demand_full: Dict[int, int],
    ) -> Individual:
        """
        Feasibility/consistency repair:
        - sanitize stops, remove zeros
        - correct overservice per node
        - split trips exceeding vehicle capacity
        - fill missing demand (tries insertion into existing trips first)
        - fix depot connectivity chain
        """
        repaired: Individual = _fast_clone_individual(individual)
        cap_by_bus = self._cap_by_bus or [bus_capacity for _ in range(buses_count)]
        origin_by_bus = self._origin_by_bus or [{"kind": "depot", "index": 0} for _ in range(buses_count)]

        # Phase 1: sanitize
        for b in range(len(repaired)):
            cleaned_sched: List[Trip] = []
            for trip in repaired[b]:
                stops = []
                for n, c in trip.get("stops", []):
                    c2 = int(c)
                    if c2 > 0:
                        stops.append((int(n), c2))
                if stops:
                    cleaned_sched.append({
                        "start_depot": int(trip.get("start_depot", 0)),
                        "end_depot": int(trip.get("end_depot", 0)),
                        "stops": stops,
                    })
            repaired[b] = cleaned_sched

        # quick depot loads
        depot_loads = self._calculate_depot_loads(repaired)

        # Phase 2: remove overservice (ensure picked <= demand per node)
        picked = {int(n): 0 for n in pickup_nodes}
        for sched in repaired:
            for trip in sched:
                for n, c in trip.get("stops", []):
                    if int(n) in picked:
                        picked[int(n)] += int(c)

        for node in pickup_nodes:
            node = int(node)
            demand = int(demand_full.get(node, 0))
            over = int(picked.get(node, 0)) - demand
            if over <= 0:
                continue

            visits: List[Tuple[int, int, int, int]] = []
            for b, sched in enumerate(repaired):
                for t, trip in enumerate(sched):
                    for s, (n, c) in enumerate(trip.get("stops", [])):
                        if int(n) == node and int(c) > 0:
                            visits.append((b, t, s, int(c)))
            # reduce from later visits first (often least beneficial for wait time)
            visits.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)

            for b, t, s, c in visits:
                if over <= 0:
                    break
                red = min(over, c)
                new_c = c - red
                repaired[b][t]["stops"][s] = (node, new_c)
                over -= red

        # drop zeros again
        for b in range(len(repaired)):
            new_sched: List[Trip] = []
            for trip in repaired[b]:
                trip["stops"] = [(n, c) for (n, c) in trip.get("stops", []) if int(c) > 0]
                if trip["stops"]:
                    new_sched.append(trip)
            repaired[b] = new_sched

        # recompute depot loads (since we changed quantities)
        depot_loads = self._calculate_depot_loads(repaired)

        # Phase 3: split overflow trips (capacity)
        for b in range(len(repaired)):
            cap = int(cap_by_bus[b])
            t = 0
            while t < len(repaired[b]):
                trip = repaired[b][t]
                stops = trip.get("stops", [])
                load = sum(int(c) for _, c in stops)
                if load <= cap:
                    t += 1
                    continue

                end_depot = int(trip.get("end_depot", 0))
                if 0 <= end_depot < len(depot_loads):
                    depot_loads[end_depot] -= load

                # Remove from end until within cap; overflow becomes a separate request set
                overflow: RemovedList = []
                while sum(int(cc) for _, cc in trip["stops"]) > cap and trip["stops"]:
                    n, c = trip["stops"][-1]
                    excess = sum(int(cc) for _, cc in trip["stops"]) - cap
                    take = min(int(c), int(excess))
                    overflow.append((int(n), int(take)))
                    remain = int(c) - int(take)
                    if remain > 0:
                        trip["stops"][-1] = (int(n), remain)
                    else:
                        trip["stops"].pop()

                new_load = sum(int(cc) for _, cc in trip["stops"])
                if 0 <= end_depot < len(depot_loads):
                    depot_loads[end_depot] += new_load

                # reinsert overflow using the same greedy insertion logic (fills spare space first)
                overflow.reverse()
                if overflow:
                    tmp_cfg = ALNSConfig()  # use defaults for insertion shaping during repair
                    bus_finish_est = self._estimate_bus_finish_times(repaired, n_depots, durations_matrix)
                    for n, q in overflow:
                        self._insert_quantity_best(
                            repaired, node=int(n), qty=int(q), cfg=tmp_cfg,
                            buses_count=buses_count, n_depots=n_depots, durations_matrix=durations_matrix,
                            bus_finish_est=bus_finish_est,
                        )
                    depot_loads = self._calculate_depot_loads(repaired)

                if not trip.get("stops"):
                    repaired[b].pop(t)
                else:
                    t += 1

        # Phase 4: fill missing demand (tries insertion into existing trips first)
        picked2 = {int(n): 0 for n in pickup_nodes}
        for sched in repaired:
            for trip in sched:
                for n, c in trip.get("stops", []):
                    if int(n) in picked2:
                        picked2[int(n)] += int(c)

        tmp_cfg = ALNSConfig()  # insertion shaping defaults for repair
        bus_finish_est = self._estimate_bus_finish_times(repaired, n_depots, durations_matrix)
        for node in pickup_nodes:
            node = int(node)
            missing = int(demand_full.get(node, 0)) - int(picked2.get(node, 0))
            if missing <= 0:
                continue
            self._insert_quantity_best(
                repaired,
                node=node,
                qty=missing,
                cfg=tmp_cfg,
                buses_count=buses_count,
                n_depots=n_depots,
                durations_matrix=durations_matrix,
                bus_finish_est=bus_finish_est,
            )

        # Phase 5: final connectivity cleanup
        for b in range(buses_count):
            repaired[b] = [tr for tr in repaired[b] if tr.get("stops")]
            repaired[b] = self._fix_depot_connectivity(repaired[b], origin=self._origin_loc(b))

        return repaired

    def _find_best_end_depot(
        self,
        last_stop_node: int,
        trip_load: int,
        depot_loads: List[int],
        n_depots: int,
        durations_matrix: Dict[Tuple[int, int], float],
    ) -> int:
        if n_depots <= 1:
            return 0

        feasible = []
        for i in range(n_depots):
            cap = None
            if self._depots_runtime and i < len(self._depots_runtime):
                cap = self._depots_runtime[i].get("capacity")
            if cap is None or (depot_loads[i] + trip_load <= cap):
                feasible.append(i)

        from_idx = n_depots + int(last_stop_node)

        def travel_sec(to_depot: int) -> float:
            return float(durations_matrix.get((from_idx, to_depot), float("inf")))

        if feasible:
            return int(min(feasible, key=lambda d: travel_sec(d)))

        # If none feasible, choose minimal overflow
        best = 0
        best_over = float("inf")
        for i in range(n_depots):
            cap = float("inf")
            if self._depots_runtime and i < len(self._depots_runtime):
                cap = float(self._depots_runtime[i].get("capacity", float("inf")))
            over = (depot_loads[i] + trip_load) - cap
            if over < best_over:
                best_over = over
                best = i
        return int(best)


def run_alns_algorithm(
    evacuation_zones_input: Optional[List[Dict[str, Any]]] = None,
    buses_count: int = 3,
    bus_capacity: int = 80,
    **params: Any,
) -> AlgorithmResult:
    algo = ALNSEvacuationAlgorithm()
    return algo.run(
        evacuation_zones_input=evacuation_zones_input,
        buses_count=buses_count,
        bus_capacity=bus_capacity,
        **params,
    )
