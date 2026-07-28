# Path: backend/app/evacuation/local_search/memetic.py
from typing import Dict, List, Any, Optional, Tuple, Callable
import time
import copy
import math
import random
import pprint
import itertools

class MemeticImprover:
    """
    Memetic / Local-Search engine for the evacuation EA.
    Optimized for Dynamic Service Times and Heterogeneous Fleets.
    """

    def __init__(
        self,
        *,
        evaluate_fitness: Callable[..., float],
        repair: Callable[..., Any],
        fix_depot_connectivity: Callable[..., Any],
        buses_count: int,
        bus_capacity: int,
        depots,
        facilities,
        n_depots: int,
        pickup_nodes,
        durations_matrix: Dict[Tuple[int, int], float],
        demand_full: Dict[int, int],
        penalty_factor: float,
        latest_evacuation_penalty_factor: float,
        # hetero + origin-aware context
        cap_by_bus: Optional[List[int]] = None,
        origin_by_bus: Optional[List[Dict[str, Any]]] = None,
        node_coords: Optional[Dict[int, Tuple[float, float]]] = None,
        start_to_node_seconds: Optional[Dict[int, Dict[int, float]]] = None,
        avg_speed_kmh: float = 30.0,
        road_factor: float = 1.25,
        # Dynamic service time params
        use_dynamic_service_time: bool = False,
        service_time_base_min: float = 3.0,
        service_time_per_person_min: float = 20.0/60.0,
    ):
        self._evaluate_fitness = evaluate_fitness
        self._repair = repair
        self._fix_depot_connectivity = fix_depot_connectivity

        self.buses_count = buses_count
        self.bus_capacity = bus_capacity  # legacy default
        self.depots = depots
        self.facilities = facilities
        self.n_depots = n_depots
        self.pickup_nodes = pickup_nodes
        self.durations_matrix = durations_matrix
        self.demand_full = demand_full
        self.penalty_factor = penalty_factor
        self.latest_evacuation_penalty_factor = latest_evacuation_penalty_factor

        # hetero + origin-aware context
        self.cap_by_bus: List[int] = list(cap_by_bus or [bus_capacity for _ in range(buses_count)])
        self.origin_by_bus: List[Dict[str, Any]] = list(origin_by_bus or [{"kind": "depot", "index": 0} for _ in range(buses_count)])
        self.node_coords: Optional[Dict[int, Tuple[float, float]]] = node_coords
        self.start_to_node_seconds: Optional[Dict[int, Dict[int, float]]] = start_to_node_seconds
        self.avg_speed_kmh: float = float(avg_speed_kmh)
        self.road_factor: float = float(road_factor)

        # Store service time params
        self._service_params = {
            "use_dynamic_service_time": use_dynamic_service_time,
            "service_time_base_min": service_time_base_min,
            "service_time_per_person_min": service_time_per_person_min,
        }

        self.last_run_stats: Dict[str, Any] = {}

    # ---------------------------
    # Public API
    # ---------------------------

    def improve(self, individual, ls_params: Dict[str, Any], *, context: Optional[Dict[str, Any]] = None):
        context = context or {}
        deadline = context.get("deadline_monotonic")
        if deadline is not None and time.monotonic() >= float(deadline):
            self.last_run_stats = {
                "time_limit_s": 0.0,
                "iterations": 0,
                "deadline_reached": True,
            }
            return individual
        return self._memetic_improve(
            copy.deepcopy(individual),
            self.buses_count,
            self.bus_capacity,
            self.depots,
            self.facilities,
            self.n_depots,
            self.pickup_nodes,
            self.durations_matrix,
            self.demand_full,
            self.penalty_factor,
            self.latest_evacuation_penalty_factor,
            ls_params,
            context=context,
        )

    # ---------------------------
    # Memetic kernel (with scheduler)
    # ---------------------------

# -----------------------------------------------------------
    # Memetic kernel (with scheduler)
    # -----------------------------------------------------------

    def _memetic_improve(
        self,
        individual,
        buses_count,
        bus_capacity,
        depots,
        facilities,
        n_depots,
        pickup_nodes,
        durations_matrix,
        demand_full,
        penalty_factor,
        latest_evacuation_penalty_factor,
        ls_params: Dict[str, Any],
        *,
        context: Dict[str, Any],
    ):
        # Local-search parameters with defaults.
        max_it = int(ls_params.get("max_iterations", 120))
        time_limit_s = float(ls_params.get("time_limit_seconds", 0.05))
        use_alns_shake = bool(ls_params.get("use_alns_shake", False))
        shake_every = int(ls_params.get("shake_every", 60))
        rcl_size = int(ls_params.get("candidate_set_size", 12))
        max_checks = int(ls_params.get("max_comparisons", 1000)) 
        allow_split_moves = bool(ls_params.get("allow_split_moves", True))

        # Scheduler knobs
        micro_batch_calls = int(ls_params.get("micro_batch_calls", 8))
        stall_limit = int(ls_params.get("stall_limit", 3))
        min_slice_s = float(ls_params.get("min_slice_seconds", 0.003))

        # --- Time-Aware Progress Calculation ---
        # 1. Retrieve generation safely for logging
        generation = int(context.get("generation", 0))

        # 2. Retrieve timing context
        algo_start = context.get("algorithm_start_time")
        algo_limit = context.get("time_limit_seconds")
        global_deadline = context.get("deadline_monotonic")
        
        # 3. Calculate Progress (0.0 to 1.0)
        progress = 0.5 # Default middle-game
        
        if algo_start is not None and algo_limit is not None and algo_limit > 0:
            # Time-based progress
            elapsed = time.monotonic() - float(algo_start)
            progress = min(1.0, max(0.0, elapsed / float(algo_limit)))
        else:
            # Fallback to Generation-based progress (assuming standard 100 gens)
            # If you increase gens to 10000, update the divisor here or pass max_gens in context
            progress = min(1.0, generation / 100.0)

        # 4. Get adaptive weights
        phase_weights = self._get_adaptive_weights(progress, generation)

        # Baseline cost
        best = individual
        best_cost = self._evaluate_fitness(
            best,
            buses_count,
            bus_capacity,
            depots,
            facilities,
            n_depots,
            durations_matrix,
            demand_full,
            penalty_factor,
            latest_evacuation_penalty_factor,
        )

        # Operator wrappers
        def call_intra():
            return self._try_intra_trip_improvements(
                best, buses_count, bus_capacity, depots, facilities, n_depots,
                durations_matrix, demand_full,
                penalty_factor, latest_evacuation_penalty_factor,
            )

        def call_relocate():
            return self._try_relocate_moves(
                best, buses_count, bus_capacity, depots, facilities, n_depots,
                durations_matrix, demand_full,
                penalty_factor, latest_evacuation_penalty_factor,
                rcl_size=rcl_size, allow_split_moves=allow_split_moves,
            )

        def call_swap_stops():
            return self._try_swap_stops(
                best, buses_count, bus_capacity, depots, facilities, n_depots,
                durations_matrix, demand_full,
                penalty_factor, latest_evacuation_penalty_factor,
                rcl_size=rcl_size,
            )

        def call_swap_trips():
            return self._try_swap_trips(
                best, buses_count, bus_capacity, depots, facilities, n_depots,
                durations_matrix, demand_full,
                penalty_factor, latest_evacuation_penalty_factor, max_checks=max_checks
            )

        def call_move_trip():
            return self._try_move_trip(
                best, buses_count, bus_capacity, depots, facilities, n_depots,
                durations_matrix, demand_full,
                penalty_factor, latest_evacuation_penalty_factor, max_checks=max_checks
            )

        def call_quantity():
            return self._try_quantity_rebalance(
                best, buses_count, bus_capacity, depots, facilities, n_depots,
                durations_matrix, demand_full,
            )

        def call_balance_makespan():
            return self._try_balance_makespan(
                best, buses_count, bus_capacity, depots, facilities, n_depots,
                durations_matrix, demand_full,
                penalty_factor, latest_evacuation_penalty_factor,
            )

        def call_takeover_gap():
            return self._try_takeover_near_gap(
                best, buses_count, bus_capacity, depots, facilities, n_depots,
                durations_matrix, demand_full,
                penalty_factor, latest_evacuation_penalty_factor,
                gap_window_minutes=float(ls_params.get("gap_window_minutes", 12.0)),
                rcl_size=int(ls_params.get("gap_rcl_size", 12)),
            )

        def call_fill_idle():
            return self._try_fill_idle_time(
                best, buses_count, bus_capacity, depots, facilities, n_depots,
                durations_matrix, demand_full,
                penalty_factor, latest_evacuation_penalty_factor,
            )

        def call_change_depot():
            return self._try_change_end_depot(
                best, buses_count, bus_capacity, depots, facilities, n_depots,
                durations_matrix, demand_full,
                penalty_factor, latest_evacuation_penalty_factor,
            )

        def call_consolidate_trips():
            return self._try_consolidate_trips(
                best, buses_count, bus_capacity, depots, facilities, n_depots,
                durations_matrix, demand_full,
                penalty_factor, latest_evacuation_penalty_factor,
            )
        
        def call_spatial_relocate():
            return self._try_spatial_relocate(
                best, buses_count, bus_capacity, depots, facilities, n_depots,
                durations_matrix, demand_full,
                penalty_factor, latest_evacuation_penalty_factor,
                deadhead_threshold_min=15.0 # Tunable: How empty does a drive be to trigger this?
            )
            
        def call_split_mixed():
            return self._try_split_mixed_trips(
                best, buses_count, bus_capacity, depots, facilities, n_depots,
                durations_matrix, demand_full,
                penalty_factor, latest_evacuation_penalty_factor,
            )

        def call_crumb_extract():
            return self._try_crumb_extraction(
                best, buses_count, bus_capacity, depots, facilities, n_depots,
                durations_matrix, demand_full,
                penalty_factor, latest_evacuation_penalty_factor,
            )
            
        def call_self_consolidate():
            return self._try_self_consolidate(
                best, buses_count, bus_capacity, depots, facilities, n_depots,
                durations_matrix, demand_full,
                penalty_factor, latest_evacuation_penalty_factor,
            )

        ops = {
            "swap_stops": call_swap_stops,
            "swap_trips": call_swap_trips,
            "move_trip": call_move_trip,
            "relocate": call_relocate,
            "change_depot": call_change_depot,
            "intra_trip": call_intra,
            "quantity_rebalance": call_quantity,
            "balance_makespan": call_balance_makespan,
            "takeover_gap": call_takeover_gap,
            "fill_idle_time": call_fill_idle,
            "consolidate_trips": call_consolidate_trips,
            "spatial_relocate": call_spatial_relocate,
            "split_mixed": call_split_mixed,
            "crumb_extract": call_crumb_extract,
            "self_consolidate": call_self_consolidate,
        }

        # IPS accounting
        acc_time = {k: 0.0 for k in ops}
        acc_gain = {k: 0.0 for k in ops}
        acc_count = {k: 0 for k in ops}
        stalls = {k: 0 for k in ops}
        disabled = {k: False for k in ops}

        def ips(k: str) -> float:
            t = acc_time[k]
            g = acc_gain[k]
            if t <= 1e-9:
                return 0.0
            return max(0.0, g / max(t, 1e-9))

        def priorities() -> List[str]:
            scored = []
            for k in ops:
                if disabled[k]:
                    continue
                w = phase_weights.get(k, 0.0)
                score = (ips(k) + 1e-9) * (w + 1e-9)
                scored.append((score, w, k))
            scored.sort(reverse=True)
            return [k for _, _, k in scored]

        start = time.monotonic()
        local_deadline = start + max(0.0, time_limit_s)
        if global_deadline is not None:
            local_deadline = min(local_deadline, float(global_deadline))
        it = 0

        while it < max_it and time.monotonic() < local_deadline:
            it += 1
            improved_any = False

            ordered_ops = priorities()
            if not ordered_ops:
                break

            remaining = local_deadline - time.monotonic()
            if remaining <= 0:
                break

            active_weights = {k: phase_weights.get(k, 0.0) for k in ordered_ops}
            total_w = sum(active_weights.values()) or 1.0
            per_op_slice = {k: max(min_slice_s, remaining * (active_weights[k] / total_w)) for k in ordered_ops}

            for k in ordered_ops:
                if disabled[k]:
                    continue

                slice_deadline = min(
                    local_deadline,
                    time.monotonic() + per_op_slice[k],
                )
                local_tries = 0

                while time.monotonic() < slice_deadline and local_tries < micro_batch_calls:
                    local_tries += 1
                    t0 = time.monotonic()
                    before = best_cost
                    ok = ops[k]()
                    dt = time.monotonic() - t0
                    acc_time[k] += dt

                    if ok:
                        # Fix connectivity for every bus after a successful move
                        for b in range(buses_count):
                            origin = (self.origin_by_bus[b] if self.origin_by_bus and b < len(self.origin_by_bus)
                                    else {"kind": "depot", "index": 0})
                            if best[b]:
                                best[b] = self._fix_depot_connectivity(best[b], origin=origin)
                        
                        new_cost = self._evaluate_fitness(
                            best, buses_count, bus_capacity, depots, facilities, n_depots,
                            durations_matrix, demand_full,
                            penalty_factor, latest_evacuation_penalty_factor,
                        )
                        gain = before - new_cost
                        if gain > 1e-9:
                            best_cost = new_cost
                            acc_gain[k] += gain
                            acc_count[k] += 1
                            stalls[k] = 0
                            improved_any = True
                        else:
                            stalls[k] += 1
                    else:
                        stalls[k] += 1

                    if stalls[k] >= stall_limit:
                        disabled[k] = True
                        break

                    if time.monotonic() >= local_deadline:
                        break

                if time.monotonic() >= local_deadline:
                    break

            if (
                use_alns_shake
                and not improved_any
                and (it % shake_every == 0)
                and time.monotonic() < local_deadline
            ):
                self._alns_shake(
                    best,
                    buses_count,
                    bus_capacity,
                    depots,
                    facilities,
                    n_depots,
                    self.pickup_nodes,
                    durations_matrix,
                    demand_full,
                )
                best[:] = self._repair(
                    best,
                    buses_count,
                    bus_capacity,
                    depots,
                    facilities,
                    n_depots,
                    self.pickup_nodes,
                    durations_matrix,
                    demand_full,
                )
                stalls = {k: 0 for k in ops}
                disabled = {k: False for k in ops}

            if not improved_any:
                break
        
        # Capture the "Dirty" state before any final cleanup
        dirty_best = copy.deepcopy(best)
        best = self._finalize_after_local_search(best, buses_count)

        # Final feasibility check with VERBOSE reporting on failure
        if not self._is_feasible_individual(best, buses_count, n_depots, durations_matrix, demand_full):
             best = self._repair(best, buses_count, bus_capacity, depots, facilities, n_depots,
                                 self.pickup_nodes, durations_matrix, demand_full)
             best = self._finalize_after_local_search(best, buses_count)
             
             if not self._is_feasible_individual(best, buses_count, n_depots, durations_matrix, demand_full):
                 # Logging removed for brevity in this response, but keep your print statements if needed
                 pass

        flat_stats = {}
        for k in ops:
            flat_stats[k] = acc_gain[k]
            flat_stats[f"{k}_cnt"] = acc_count[k]

        self.last_run_stats = {
            "time_limit_s": time_limit_s,
            "iterations": it,
            "final_cost": best_cost,
            "ips": {k: ips(k) for k in ops},
            "gain": acc_gain,
            "count": acc_count,   
            "flat_stats": flat_stats,  
            "time": acc_time,
            "stalls": stalls,
            "disabled": disabled,
            "phase_weights": phase_weights,
            "generation": generation,  # << THIS WAS CAUSING THE ERROR, NOW FIXED
        }

        return best
    
    def _get_adaptive_weights(self, progress: float, g) -> Dict[str, float]:
        if progress is None or progress < 0:
            progress = -1.0

        if progress < 0 and g < 0:  # Default / Fallback
            return {
                "swap_stops": 0.10, "swap_trips": 0.10, "move_trip": 0.05, "relocate": 0.05,
                "change_depot": 0.05, "intra_trip": 0.15, "quantity_rebalance": 0.00,
                "balance_makespan": 0.15, "takeover_gap": 0.10, "fill_idle_time": 0.05,
                "consolidate_trips": 0.05, "spatial_relocate": 0.05,
                "split_mixed": 0.10, "crumb_extract": 0.05, "self_consolidate": 0.05,
            }

        # Use time-progress when available; fallback to generation thresholds.
        if progress >= 0:
            if progress <= 0.10:  # Early game: Fix sequence & structure
                return {
                    "swap_stops": 0.10, "swap_trips": 0.05, "move_trip": 0.05, "relocate": 0.05,
                    "change_depot": 0.05, "intra_trip": 0.25,
                    "quantity_rebalance": 0.00,
                    "balance_makespan": 0.05, "takeover_gap": 0.05, "fill_idle_time": 0.00,
                    "consolidate_trips": 0.10, "spatial_relocate": 0.05,
                    "split_mixed": 0.15,  # Purify trips early
                    "crumb_extract": 0.05, "self_consolidate": 0.05,
                }
            elif progress <= 0.40:  # Mid game: Load balancing & cleaning
                return {
                    "swap_stops": 0.10, "swap_trips": 0.10, "move_trip": 0.10, "relocate": 0.10,
                    "change_depot": 0.10, "intra_trip": 0.10,
                    "quantity_rebalance": 0.00,
                    "balance_makespan": 0.10, "takeover_gap": 0.10, "fill_idle_time": 0.05,
                    "consolidate_trips": 0.00, "spatial_relocate": 0.05,
                    "split_mixed": 0.05,
                    "crumb_extract": 0.05, "self_consolidate": 0.05,
                }
            else:  # Late game: Squeeze the bottleneck (makespan focus)
                return {
                    "swap_stops": 0.05, "swap_trips": 0.05, "move_trip": 0.05, "relocate": 0.05,
                    "change_depot": 0.10, "intra_trip": 0.10,
                    "quantity_rebalance": 0.00,
                    "balance_makespan": 0.20,
                    "takeover_gap": 0.15,
                    "fill_idle_time": 0.00,
                    "consolidate_trips": 0.00, "spatial_relocate": 0.00,
                    "split_mixed": 0.05,
                    "crumb_extract": 0.20, "self_consolidate": 0.8,  # Aggressively vacuum crumbs
                }

        if g <= 10:  # Early game: Fix sequence & Structure
            return {
                "swap_stops": 0.10, "swap_trips": 0.05, "move_trip": 0.05, "relocate": 0.05,
                "change_depot": 0.05, "intra_trip": 0.25, 
                "quantity_rebalance": 0.00,
                "balance_makespan": 0.05, "takeover_gap": 0.05, "fill_idle_time": 0.00,
                "consolidate_trips": 0.10, "spatial_relocate": 0.05,
                "split_mixed": 0.15, # Purify trips early
                "crumb_extract": 0.05, "self_consolidate": 0.05,
            }
        elif g <= 40:  # Mid game: Load Balancing & Cleaning
            return {
                "swap_stops": 0.10, "swap_trips": 0.10, "move_trip": 0.10, "relocate": 0.10,
                "change_depot": 0.10, "intra_trip": 0.10,
                "quantity_rebalance": 0.00,
                "balance_makespan": 0.10, "takeover_gap": 0.10, "fill_idle_time": 0.05,
                "consolidate_trips": 0.00, "spatial_relocate": 0.05,
                "split_mixed": 0.05, 
                "crumb_extract": 0.05, "self_consolidate": 0.05,
            }
        else:  # Late game: Squeeze the bottleneck (Makespan focus)
            return {
                "swap_stops": 0.05, "swap_trips": 0.05, "move_trip": 0.05, "relocate": 0.05,
                "change_depot": 0.10, "intra_trip": 0.10, 
                "quantity_rebalance": 0.00,
                "balance_makespan": 0.20, 
                "takeover_gap": 0.15, 
                "fill_idle_time": 0.00,
                "consolidate_trips": 0.00, "spatial_relocate": 0.00,
                "split_mixed": 0.05,
                "crumb_extract": 0.20, "self_consolidate": 0.8, # Aggressively vacuum crumbs from the bottleneck
            }
    def _get_adaptive_weights2(self, progress: float) -> Dict[str, float]:
        """
        Adaptive weights based on Runtime Progress (0.0 to 1.0).
        
        0.0 - 0.25: PURIFICATION (Fix structure, break mixed trips)
        0.25 - 0.75: BALANCING (Load balancing, gap filling)
        0.75 - 1.00: COMPACTION (Aggressive merging, cleaning crumbs)
        """
        
        # --- PHASE 1: PURIFICATION & STRUCTURE (First 25% of time) ---
        if progress < 0.25:
            w = {
                "change_depot": 0.15,       
                "spatial_relocate": 0.15,   
                "split_mixed": 0.20,        # Break mixed trips early
                "crumb_extract": 0.10,      
                "intra_trip": 0.05,         
                "relocate": 0.05,
                "swap_stops": 0.05,
                "swap_trips": 0.05,
                "move_trip": 0.05,
                "balance_makespan": 0.05,
                "takeover_gap": 0.05,
                "fill_idle_time": 0.00,
                "consolidate_trips": 0.05,
                "self_consolidate": 0.00,   
                "quantity_rebalance": 0.00
            }

        # --- PHASE 3: THE COMPACTOR (Last 25% of time) ---
        elif progress > 0.75:
            w = {
                "self_consolidate": 0.60,   # AGGRESSIVE MERGING
                "balance_makespan": 0.15,   
                "crumb_extract": 0.10,      
                "intra_trip": 0.05,         
                "relocate": 0.05,           
                "swap_stops": 0.00,
                "swap_trips": 0.00,
                "move_trip": 0.00,
                "change_depot": 0.00,       
                "spatial_relocate": 0.00,
                "split_mixed": 0.00,        
                "consolidate_trips": 0.05,
                "takeover_gap": 0.00,
                "fill_idle_time": 0.00,
                "quantity_rebalance": 0.00
            }

        # --- PHASE 2: BALANCING (Middle 50% of time) ---
        else:
            w = {
                "takeover_gap": 0.15,       
                "balance_makespan": 0.15,   
                "relocate": 0.15,           
                "intra_trip": 0.10,         
                "move_trip": 0.10,
                "swap_trips": 0.10,
                "swap_stops": 0.05,
                "change_depot": 0.05,
                "spatial_relocate": 0.05,
                "crumb_extract": 0.05,
                "split_mixed": 0.05,
                "consolidate_trips": 0.00,
                "self_consolidate": 0.00,
                "fill_idle_time": 0.00,
                "quantity_rebalance": 0.00
            }

        # Normalize weights
        total = sum(w.values())
        if total > 0:
            for k in w:
                w[k] /= total
            
        return w
    
    def _phase_weights_for_generation3(self, g: int, max_gen: int = 100) -> Dict[str, float]:
        """
        Adaptive weights based on the "Garbage Compactor" strategy.
        
        Learnings from successful runs:
        1. Shuttle evacuation is about PACKING, not Routing.
        2. Early game must purify trips (Split Mixed).
        3. Late game must aggressively merge trips (Self Consolidate).
        """
        
        # Calculate progress (0.0 to 1.0)
        # If g is -1 (unknown), assume Mid-game (0.5)
        progress = g / max(1, max_gen) if g >= 0 else 0.5

        # --- PHASE 1: PURIFICATION & STRUCTURE (0% - 25%) ---
        # Goal: Break inefficient mixed trips and fix "Teleportation" (Deadheads).
        if progress < 0.25:
            w = {
                "change_depot": 0.15,       # Fix heterogeneous fleet origins
                "spatial_relocate": 0.15,   # Stop buses from crossing the map empty
                "split_mixed": 0.20,        # [KEY] Break mixed trips so they can be merged better later
                "crumb_extract": 0.10,      # Pull 1-person pickups off big buses
                "intra_trip": 0.05,         # Don't polish garbage
                "relocate": 0.05,
                "swap_stops": 0.05,
                "swap_trips": 0.05,
                "move_trip": 0.05,
                "balance_makespan": 0.05,
                "takeover_gap": 0.05,
                "fill_idle_time": 0.00,
                "consolidate_trips": 0.05,
                "self_consolidate": 0.00,   # Too early to merge
                "quantity_rebalance": 0.00
            }

        # --- PHASE 3: THE COMPACTOR (75% - 100%) ---
        # Goal: Reduce total trip count. 
        # Replicates the "Nuclear Option" that got you 109 trips.
        elif progress > 0.75:
            w = {
                "self_consolidate": 0.60,   # [KEY] Aggressively merge trips on the same bus
                "balance_makespan": 0.15,   # Ensure we don't create one super-late bus
                "crumb_extract": 0.10,      # Vacuum remaining small pickups
                "intra_trip": 0.05,         # Minimal TSP (Shuttles don't need much)
                "relocate": 0.05,           # Minor load tuning
                "swap_stops": 0.00,
                "swap_trips": 0.00,
                "move_trip": 0.00,
                "change_depot": 0.00,       # Stop changing structure
                "spatial_relocate": 0.00,
                "split_mixed": 0.00,        # Stop splitting, we are merging now
                "consolidate_trips": 0.05,
                "takeover_gap": 0.00,
                "fill_idle_time": 0.00,
                "quantity_rebalance": 0.00
            }

        # --- PHASE 2: BALANCING (25% - 75%) ---
        # Goal: Distribute work evenly across the fleet.
        else:
            w = {
                "takeover_gap": 0.15,       # Fill timeline holes
                "balance_makespan": 0.15,   # Level the fleet
                "relocate": 0.15,           # Move stops to fill capacity
                "intra_trip": 0.10,         # Standard optimization
                "move_trip": 0.10,
                "swap_trips": 0.10,
                "swap_stops": 0.05,
                "change_depot": 0.05,
                "spatial_relocate": 0.05,
                "crumb_extract": 0.05,
                "split_mixed": 0.05,
                "consolidate_trips": 0.00,
                "self_consolidate": 0.00,
                "fill_idle_time": 0.00,
                "quantity_rebalance": 0.00
            }

        # Normalize weights to sum to 1.0
        total = sum(w.values())
        if total > 0:
            for k in w:
                w[k] /= total
            
        return w

    # ---------------------------
    # Phase weights per generation (policy)
    # ---------------------------
    def _phase_weights_for_generation1(self, g: int) -> Dict[str, float]:
        """
        Adaptive weights based on generation progress.
        Assumes we can infer progress or just use raw generation count if max is unknown,
        but here we assume the context passed 'max_generations' or we infer from behavior.
        
        Since 'max_generations' isn't stored in self by default, we can add it 
        or just use a heuristic if g < 30 (Early) vs g > 80 (Late).
        """
        
        # NOTE: Ideally pass max_generations into MemeticImprover init, 
        # but for now let's assume a standard run of ~100 gens.
        # If g is small (<25), we are early. If g is large (>75), we are late.
        
        # Default Weights (Balanced)
        w = {
            "swap_stops": 0.05,
            "swap_trips": 0.05,
            "move_trip": 0.05,
            "relocate": 0.10,
            "change_depot": 0.05,
            "intra_trip": 0.20,      # Always good
            "quantity_rebalance": 0.02,
            "balance_makespan": 0.05,
            "takeover_gap": 0.10,
            "fill_idle_time": 0.05,
            "consolidate_trips": 0.05,
            "spatial_relocate": 0.05,
            "split_mixed": 0.05,
            "crumb_extract": 0.08,
            "self_consolidate": 0.05,
        }

        # EARLY GAME: Focus on Structure & Heavy Moves
        if g < 30:
            w["change_depot"] = 0.15     # Fix depots early
            w["spatial_relocate"] = 0.15 # Fix deadheads
            w["move_trip"] = 0.10
            w["crumb_extract"] = 0.15    # Clean up mess early
            w["intra_trip"] = 0.05       # Don't waste time polishing garbage

        # LATE GAME: Focus on Polishing
        elif g > 70:
            w["intra_trip"] = 0.40       # Massive focus on 2-opt
            w["relocate"] = 0.15         # Fine tuning
            w["swap_stops"] = 0.10
            w["change_depot"] = 0.01     # Stop changing depots now
            w["spatial_relocate"] = 0.01
            w["crumb_extract"] = 0.02

        # Normalize (optional, but good practice)
        total = sum(w.values())
        for k in w:
            w[k] /= total
            
        return w
    
    def _phase_weights_for_generation2(self, g: int) -> Dict[str, float]:
        """
        Uniform weights configuration: All 15 operators have equal static priority.
        Adaptive IPS (Improvement per Second) will still naturally bias execution 
        towards operators that perform well during the run.
        """
        # 15 active operators
        w = 1.0 / 15.0  # ≈ 0.0667

        return {
            "swap_stops": w,
            "swap_trips": w,
            "move_trip": w,
            "relocate": w,
            "change_depot": w,
            "intra_trip": w,
            "quantity_rebalance": w,
            "balance_makespan": w,
            "takeover_gap": w,
            "fill_idle_time": w,
            "consolidate_trips": w,
            "spatial_relocate": w,
            "split_mixed": w,
            "crumb_extract": w,
            "self_consolidate": w,
        }
    # ---------------------------
    # Feasibility & finalize
    # ---------------------------

    def _is_feasible_individual(self, individual, buses_count, n_depots, durations_matrix, demand_full) -> bool:
        totals = {node: 0 for node in demand_full.keys()}
        for b in range(buses_count):
            for trip in individual[b]:
                for node, cnt in trip.get("stops", []):
                    if cnt < 0: return False
                    if node not in totals: return False
                    totals[node] += cnt

        for node, need in demand_full.items():
            if totals.get(node, 0) != need: return False

        for b in range(buses_count):
            trips = individual[b]
            if not trips: continue
            origin = self.origin_by_bus[b] if self.origin_by_bus and b < len(self.origin_by_bus) else {"kind": "depot", "index": 0}

            if origin.get("kind") == "depot":
                if trips[0].get("start_depot") != int(origin.get("index", 0)): return False

            for i in range(1, len(trips)):
                if trips[i].get("start_depot") != trips[i - 1].get("end_depot"): return False

            bus_cap = self.cap_by_bus[b] if self.cap_by_bus and b < len(self.cap_by_bus) else self.bus_capacity
            for t_idx, trip in enumerate(trips):
                stops = trip.get("stops", [])
                if not stops: continue
                if sum(c for _, c in stops) > bus_cap: return False
                first_node, _ = stops[0]

                if t_idx == 0:
                    okind = origin.get("kind")
                    if okind == "depot":
                        start_depot = int(origin.get("index", 0))
                        if not math.isfinite(durations_matrix.get((start_depot, n_depots + first_node), float("inf"))): return False
                    elif okind == "node":
                        start_node = int(origin.get("index"))
                        if start_node != first_node:
                            if not math.isfinite(durations_matrix.get((n_depots + start_node, n_depots + first_node), float("inf"))): return False
                else:
                    sd = trip.get("start_depot", 0)
                    if not math.isfinite(durations_matrix.get((sd, n_depots + first_node), float("inf"))): return False

                for (a, _), (bnode, _) in zip(stops, stops[1:]):
                    if not math.isfinite(durations_matrix.get((n_depots + a, n_depots + bnode), float("inf"))): return False

                last_node, _ = stops[-1]
                ed = trip.get("end_depot", 0)
                if not math.isfinite(durations_matrix.get((n_depots + last_node, ed), float("inf"))): return False
        return True

    def _is_feasible_individual_with_reason(self, individual, buses_count, n_depots, durations_matrix, demand_full):
        """Verbose feasibility checker for debugging."""
        def log_and_fail(reason, state):
            print(reason)
            print("--- INDIVIDUAL STATE ---")
            pprint.pprint(state)
            return False

        totals = {node: 0 for node in demand_full.keys()}
        for b in range(buses_count):
            for t_idx, trip in enumerate(individual[b]):
                for s_idx, (node, cnt) in enumerate(trip.get("stops", [])):
                    if cnt < 0:
                        return log_and_fail(f"Negative count at Bus {b}, Trip {t_idx}, Stop {s_idx}", individual)
                    if node not in totals:
                        return log_and_fail(f"Unknown node {node} at Bus {b}, Trip {t_idx}", individual)
                    totals[node] += cnt

        for node, need in demand_full.items():
            if totals.get(node, 0) != need:
                return log_and_fail(f"Demand mismatch for node {node}: Need {need}, Got {totals.get(node, 0)}", individual)

        for b in range(buses_count):
            trips = individual[b]
            if not trips: continue
            origin = self.origin_by_bus[b] if self.origin_by_bus and b < len(self.origin_by_bus) else {"kind": "depot", "index": 0}

            if origin.get("kind") == "depot":
                expected = int(origin.get("index", 0))
                if trips[0].get("start_depot") != expected:
                    return log_and_fail(f"Bus {b} start_depot mismatch. Expected {expected}, Got {trips[0].get('start_depot')}", individual)

            for i in range(1, len(trips)):
                if trips[i].get("start_depot") != trips[i - 1].get("end_depot"):
                    return log_and_fail(f"Bus {b} connectivity broken between trip {i-1} and {i}", individual)

            bus_cap = self.cap_by_bus[b] if self.cap_by_bus and b < len(self.cap_by_bus) else self.bus_capacity
            for t_idx, trip in enumerate(trips):
                stops = trip.get("stops", [])
                if not stops: continue
                if sum(c for _, c in stops) > bus_cap:
                    return log_and_fail(f"Bus {b} Trip {t_idx} over capacity", individual)
                
                first_node, _ = stops[0]
                if t_idx == 0:
                    okind = origin.get("kind")
                    if okind == "depot":
                        start_depot = int(origin.get("index", 0))
                        if not math.isfinite(durations_matrix.get((start_depot, n_depots + first_node), float("inf"))):
                            return log_and_fail(f"Bus {b} Trip 0: No path from start depot {start_depot} to node {first_node}", individual)
                else:
                    sd = trip.get("start_depot", 0)
                    if not math.isfinite(durations_matrix.get((sd, n_depots + first_node), float("inf"))):
                        return log_and_fail(f"Bus {b} Trip {t_idx}: No path from start depot {sd} to node {first_node}", individual)

                for (a, _), (bnode, _) in zip(stops, stops[1:]):
                    if not math.isfinite(durations_matrix.get((n_depots + a, n_depots + bnode), float("inf"))):
                         return log_and_fail(f"Bus {b} Trip {t_idx}: No path between nodes {a} -> {bnode}", individual)

                last_node, _ = stops[-1]
                ed = trip.get("end_depot", 0)
                if not math.isfinite(durations_matrix.get((n_depots + last_node, ed), float("inf"))):
                     return log_and_fail(f"Bus {b} Trip {t_idx}: No path from node {last_node} to end depot {ed}", individual)
        return True

    def _finalize_after_local_search(self, individual, buses_count: int):
        #print(f"\n[DEBUG] Finalize called. Object ID: {id(individual)}")
        def _coalesce_stops(stops):
            order = []
            sums = {}
            for node, count in stops:
                n = int(node)
                c = int(count)
                if n not in sums:
                    order.append(n)
                    sums[n] = 0
                sums[n] += max(0, c)
            return [(n, sums[n]) for n in order if sums[n] > 0]
        
        for b in range(buses_count):
            # 1. Clean empty trips (AND DEEPCOPY TO BREAK REFERENCES)
            cleaned = []
            for trip in individual[b]:
                if trip.get("stops"):
                    # Filter stops
                    valid_stops = [(n, c) for (n, c) in trip["stops"] if c > 0]
                    if valid_stops:
                        valid_stops = _coalesce_stops(valid_stops)
                    if valid_stops:
                        # CRITICAL: Create a fresh copy of the trip dict to prevent aliasing bugs
                        new_trip = copy.deepcopy(trip)
                        new_trip["stops"] = valid_stops
                        cleaned.append(new_trip)
            
            individual[b] = cleaned

            # 2. FORCE CONNECTIVITY REPAIR
            if individual[b]:
                origin = self.origin_by_bus[b] if self.origin_by_bus and b < len(self.origin_by_bus) else {"kind": "depot", "index": 0}
                
                # --- Force First Trip Start ---
                start_idx = 0
                if origin.get("kind") == "depot":
                    start_idx = int(origin.get("index", 0))
                
                # Check and Fix First Trip
                if individual[b][0].get("start_depot") != start_idx:
                    #print(f"[DEBUG] Bus {b} Trip 0 Start Mismatch. Forced {individual[b][0].get('start_depot')} -> {start_idx}")
                    individual[b][0]["start_depot"] = start_idx
                
                # --- Force Chain (The Zipper) ---
                for i in range(1, len(individual[b])):
                    prev_end = individual[b][i-1].get("end_depot", 0)
                    current_start = individual[b][i].get("start_depot")
                    
                    # Check and Fix Chain
                    if current_start != prev_end:
                        #print(f"[DEBUG] Bus {b} Trip {i} Connectivity Gap! PrevEnd={prev_end}, CurrStart={current_start}. FIXED.")
                        individual[b][i]["start_depot"] = prev_end

        return individual

    # ---------------------------
    # Neighborhoods & utilities
    # ---------------------------

    def _calculate_depot_loads(self, individual: List[List[Dict[str, Any]]]) -> List[int]:
        if not self.depots: return []
        depot_loads = [0] * len(self.depots)
        for bus_schedule in individual:
            for trip in bus_schedule:
                end_depot = trip.get("end_depot", 0)
                if 0 <= end_depot < len(depot_loads):
                    trip_load = sum(count for _, count in trip.get("stops", []))
                    depot_loads[end_depot] += trip_load
        return depot_loads

    def _find_best_end_depot(self, last_stop_node: int, trip_load: int, depot_loads: List[int]) -> int:
        if self.n_depots <= 1: return 0
        feasible_depots = []
        for i in range(self.n_depots):
            capacity = self.depots[i].get('capacity')
            if capacity is None or (depot_loads[i] + trip_load <= capacity):
                feasible_depots.append(i)

        from_location_idx = self.n_depots + last_stop_node
        if feasible_depots:
            min_time = float('inf')
            best_depot_idx = feasible_depots[0]
            for depot_idx in feasible_depots:
                travel_time = self.durations_matrix.get((from_location_idx, depot_idx), float('inf'))
                if travel_time < min_time:
                    min_time = travel_time
                    best_depot_idx = depot_idx
            return best_depot_idx
        else:
            min_overfill = float('inf')
            best_depot_idx = 0
            for i in range(self.n_depots):
                capacity = self.depots[i].get('capacity', float('inf'))
                overfill = (depot_loads[i] + trip_load) - capacity
                if overfill < min_overfill:
                    min_overfill = overfill
                    best_depot_idx = i
            return best_depot_idx

    def _trip_load(self, trip) -> int:
        return sum(cnt for _, cnt in trip.get("stops", []))

    def _capacity_left(self, trip, bus_idx: int) -> int:
        cap = self.cap_by_bus[bus_idx] if self.cap_by_bus and bus_idx < len(self.cap_by_bus) else self.bus_capacity
        return max(0, cap - self._trip_load(trip))

    def _compute_trip_schedule(self, trip, n_depots, durations_matrix, *, bus_idx: Optional[int] = None, trip_idx: Optional[int] = None):
        """
        Returns per-trip schedule proxy using consistent Static/Dynamic logic.
        """
        if not trip.get("stops"):
            return {"arrival_times": [], "return_time": 0.0, "departure_time": 0.0, "trip_time": 0.0}

        t = 0.0
        arrival_times: List[float] = []
        
        bus_capacity = 80 # default
        if bus_idx is not None and bus_idx < len(self.cap_by_bus):
            bus_capacity = self.cap_by_bus[bus_idx]

        first_stop_node, _ = trip["stops"][0]

        # First leg travel time
        if trip_idx == 0 and bus_idx is not None:
            origin = self.origin_by_bus[bus_idx] if self.origin_by_bus and bus_idx < len(self.origin_by_bus) else {"kind": "depot", "index": 0}
            okind = origin.get("kind")
            if okind == "depot":
                start_dep = int(origin.get("index", 0))
                t += durations_matrix.get((start_dep, n_depots + first_stop_node), float("inf")) / 60.0
            elif okind == "node":
                start_node = int(origin.get("index"))
                if start_node != first_stop_node:
                    t += durations_matrix.get((n_depots + start_node, n_depots + first_stop_node), float("inf")) / 60.0
            elif okind == "coord":
                t += self._first_leg_minutes_memetic(bus_idx, first_stop_node)
            else:
                t += durations_matrix.get((trip.get("start_depot", 0), n_depots + first_stop_node), float("inf")) / 60.0
        else:
            t += durations_matrix.get((trip.get("start_depot", 0), n_depots + first_stop_node), float("inf")) / 60.0

        arrival_times.append(t)

        # Service Time (Pickup 0)
        people_at_stop = trip["stops"][0][1]
        if self._service_params.get("use_dynamic_service_time", False):
            service_time = self._service_params["service_time_base_min"] + people_at_stop * self._service_params["service_time_per_person_min"]
        else:
            service_time = self._service_params["service_time_base_min"] + bus_capacity * self._service_params["service_time_per_person_min"]
        t += service_time

        # Stops
        for i in range(1, len(trip["stops"])):
            prev_node, _ = trip["stops"][i - 1]
            node, people_at_stop = trip["stops"][i]
            
            t += durations_matrix.get((n_depots + prev_node, n_depots + node), float("inf")) / 60.0
            arrival_times.append(t)

            if self._service_params.get("use_dynamic_service_time", False):
                service_time = self._service_params["service_time_base_min"] + people_at_stop * self._service_params["service_time_per_person_min"]
            else:
                service_time = self._service_params["service_time_base_min"] + bus_capacity * self._service_params["service_time_per_person_min"]
            t += service_time

        # Return
        last_node, _ = trip["stops"][-1]
        t_return = t + durations_matrix.get((n_depots + last_node, trip.get("end_depot", 0)), float("inf")) / 60.0

        # Service Time (Offload)
        total_people_in_trip = self._trip_load(trip)
        if total_people_in_trip > 0:
            if self._service_params.get("use_dynamic_service_time", False):
                offload_service_time = self._service_params["service_time_base_min"] + total_people_in_trip * self._service_params["service_time_per_person_min"]
            else:
                offload_service_time = self._service_params["service_time_base_min"] + bus_capacity * self._service_params["service_time_per_person_min"]
            t_return += offload_service_time

        return {"arrival_times": arrival_times, "return_time": t_return, "departure_time": 0.0, "trip_time": t_return}

    def _bus_finish_times(self, individual, n_depots, durations_matrix) -> List[float]:
        return [self._bus_finish_time(individual[b], n_depots, durations_matrix, bus_idx=b) for b in range(len(individual))]
    
    def _bus_finish_time(self, trips, n_depots, durations_matrix, *, bus_idx: int) -> float:
        total = 0.0
        for t_idx, trip in enumerate(trips):
            sched = self._compute_trip_schedule(trip, n_depots, durations_matrix, bus_idx=bus_idx, trip_idx=t_idx)
            total += sched.get("trip_time", 0.0)
        return total

    def _latest_finish_time(self, individual, n_depots, durations_matrix) -> float:
        times = self._bus_finish_times(individual, n_depots, durations_matrix)
        return max(times) if times else 0.0

    def _bus_trip_timeline(self, trips, n_depots, durations_matrix, *, bus_idx: int):
        timeline = []
        t = 0.0
        for i, trip in enumerate(trips):
            sched = self._compute_trip_schedule(trip, n_depots, durations_matrix, bus_idx=bus_idx, trip_idx=i)
            depart = t
            ret = t + sched.get("trip_time", 0.0)
            timeline.append((depart, ret))
            t = ret
        return timeline

    def _all_bus_timelines(self, individual, n_depots, durations_matrix):
        return [self._bus_trip_timeline(individual[b], n_depots, durations_matrix, bus_idx=b) for b in range(len(individual))]

    # --- Neighborhoods ---

    def _try_intra_trip_improvements(
        self, individual, buses_count, bus_capacity, depots, facilities, n_depots,
        durations_matrix, demand_full, penalty_factor,
        latest_evacuation_penalty_factor
    ) -> bool:
        """
        [COORDINATE & MAKESPAN AWARE]
        Optimizes stop sequence (TSP). 
        Crucially, if this is Trip 0 and the bus starts at a Coordinate, 
        it minimizes distance from (Lat, Lon) -> First Node.
        """
        base_cost = self._evaluate_fitness(
            individual, buses_count, bus_capacity, depots, facilities, n_depots,
            durations_matrix, demand_full,
            penalty_factor, latest_evacuation_penalty_factor,
        )
        improved = False

        # --- Helper: Calculate duration with Origin Awareness ---
        def _local_trip_duration(trip_idx, bus_idx, stops, end_depot):
            t = 0.0
            first_node = stops[0][0]
            
            # 1. Start Leg Calculation
            if trip_idx == 0:
                # Get the specific origin for this bus
                origin = self.origin_by_bus[bus_idx]
                
                if origin['kind'] == 'coord':
                    # Use Haversine/First-Leg helper for Coord -> Node
                    t += self._first_leg_minutes_memetic(bus_idx, first_node)
                elif origin['kind'] == 'node':
                    # Start Node -> First Node
                    start_node = origin['index']
                    if start_node != first_node:
                        t += durations_matrix.get((n_depots + start_node, n_depots + first_node), 1000) / 60.0
                else:
                    # Depot -> First Node (Standard)
                    start_depot = origin.get('index', 0)
                    t += durations_matrix.get((start_depot, n_depots + first_node), 1000) / 60.0
            else:
                # Not the first trip: Start from previous trip's End Depot
                # We need to look up the previous trip in the individual to find where it ended
                # But for efficiency in this local scope, we pass the 'start_depot' implied by the schedule
                start_depot = individual[bus_idx][trip_idx].get("start_depot", 0)
                t += durations_matrix.get((start_depot, n_depots + first_node), 1000) / 60.0

            # 2. Inter-stop legs (The standard TSP part)
            for i in range(len(stops) - 1):
                node_a = stops[i][0]
                node_b = stops[i+1][0]
                t += durations_matrix.get((n_depots + node_a, n_depots + node_b), 1000) / 60.0
            
            # 3. End Leg
            last_node = stops[-1][0]
            t += durations_matrix.get((n_depots + last_node, end_depot), 1000) / 60.0
            
            return t

        for b in range(buses_count):
            for t_idx, trip in enumerate(individual[b]):
                stops = trip.get("stops", [])
                if len(stops) < 2: continue
                
                # Limit depth for speed
                if len(stops) > 6: continue 

                current_stops = list(stops)
                end_depot = trip.get("end_depot", 0)
                
                # Calculate current duration using the Origin-Aware helper
                best_local_duration = _local_trip_duration(t_idx, b, current_stops, end_depot)
                best_local_stops = list(current_stops)
                found_better = False

                # Permute to find Shortest Path (TSP)
                for perm in itertools.permutations(current_stops):
                    # Check candidate duration
                    duration = _local_trip_duration(t_idx, b, perm, end_depot)
                    
                    if duration < best_local_duration - 1e-6:
                        best_local_duration = duration
                        best_local_stops = list(perm)
                        found_better = True
                
                if found_better:
                    trip["stops"] = best_local_stops
                    # Feasibility check
                    if self._is_feasible_individual(individual, buses_count, n_depots, durations_matrix, demand_full):
                        new_cost = self._evaluate_fitness(
                            individual, buses_count, bus_capacity, depots, facilities, n_depots,
                            durations_matrix, demand_full,
                            penalty_factor, latest_evacuation_penalty_factor,
                        )
                        if new_cost < base_cost:
                            base_cost = new_cost
                            improved = True
                        else:
                            trip["stops"] = current_stops # Revert
                    else:
                        trip["stops"] = current_stops
        return improved
    
    def _try_spatial_relocate(self, individual, buses_count, bus_capacity, depots, facilities, n_depots,
                              durations_matrix, demand_full, penalty_factor,
                              latest_evacuation_penalty_factor, deadhead_threshold_min=15.0) -> bool:
        """
        Spatial Relocation: Identifies trips where the bus has to drive empty for a long time 
        (deadhead) to reach the first pickup, and moves them to a closer bus.
        """
        base_cost = self._evaluate_fitness(
            individual, buses_count, bus_capacity, depots, facilities, n_depots,
            durations_matrix, demand_full,
            penalty_factor, latest_evacuation_penalty_factor,
        )

        # 1. Identify "Bad Links" (Long Deadheads)
        bad_moves = []
        for b_idx in range(buses_count):
            trips = individual[b_idx]
            if not trips: continue
            
            # Determine initial location of the bus
            origin = self.origin_by_bus[b_idx] if self.origin_by_bus and b_idx < len(self.origin_by_bus) else {"kind": "depot", "index": 0}
            
            # Current location index for distance calculation
            # Note: If origin is a coordinate/node, we approximate or skip the first check. 
            # Here we focus on Depot->Node deadheads which are the main issue.
            if origin.get("kind") == "depot":
                current_loc_idx = int(origin.get("index", 0))
            else:
                # Skip logic for non-depot start for simplicity, or treat as node index if available
                current_loc_idx = -1 

            for t_idx, trip in enumerate(trips):
                stops = trip.get("stops", [])
                if not stops: continue
                
                first_node = stops[0][0]
                
                # Calculate deadhead
                deadhead_time = 0.0
                if current_loc_idx != -1:
                    deadhead_time = durations_matrix.get((current_loc_idx, n_depots + first_node), 0) / 60.0
                
                # If this is a "Ping-Pong" move (long empty drive), mark it
                if deadhead_time > deadhead_threshold_min:
                    bad_moves.append({
                        'score': deadhead_time,
                        'src_b': b_idx,
                        'src_t': t_idx,
                        'node': first_node,
                        'trip_load': self._trip_load(trip)
                    })
                
                # Update location for next iteration (end of this trip)
                current_loc_idx = trip.get("end_depot", 0)

        if not bad_moves:
            return False

        # Sort by worst deadheads first
        bad_moves.sort(key=lambda x: x['score'], reverse=True)
        
        # 2. Try to re-assign the worst trips
        # We limit to top 5 to keep it fast
        for move in bad_moves[:5]:
            src_b = move['src_b']
            src_t = move['src_t']
            target_node = move['node']
            load_to_move = move['trip_load']
            
            # The trip might have shifted index if we moved previous ones, so we verify
            if src_t >= len(individual[src_b]): continue
            trip_to_move = individual[src_b][src_t]
            if not trip_to_move.get("stops") or trip_to_move["stops"][0][0] != target_node:
                continue # Trip changed position or isn't the one we thought

            best_dst_b = -1
            best_insert_pos = -1
            min_new_deadhead = float('inf')

            # Find a better bus
            for dst_b in range(buses_count):
                if dst_b == src_b: continue
                
                # Capacity Check (Critical for speed)
                # We check if the bus has general room. 
                # Note: Ideally we check specific trips, but this is a heuristic scan.
                # A more rigorous check happens inside the loop or relies on _is_feasible later.
                
                schedule = individual[dst_b]
                
                # Determine start loc of dst bus
                origin = self.origin_by_bus[dst_b] if self.origin_by_bus and dst_b < len(self.origin_by_bus) else {"kind": "depot", "index": 0}
                
                # Check every insertion slot
                for i in range(len(schedule) + 1):
                    # Where is the bus at slot i?
                    if i == 0:
                        if origin.get("kind") == "depot":
                            prev_loc = int(origin.get("index", 0))
                        else:
                            continue # Skip complex origin logic here
                    else:
                        prev_loc = schedule[i-1]["end_depot"]
                    
                    # How far is this bus from the target node?
                    new_dh = durations_matrix.get((prev_loc, n_depots + target_node), 9999) / 60.0
                    
                    # We want a bus that is CLOSE (e.g., < 10 mins away)
                    # and significantly closer than the original bus
                    if new_dh < 10.0 and new_dh < (move['score'] - 5.0) and new_dh < min_new_deadhead:
                        
                        # Check capacity at insertion point? 
                        # For Giant Tour logic, simple cap check is hard. 
                        # We rely on the final feasibility check to reject bad cap moves.
                        # But we can do a quick check: if bus is totally full, skip.
                        if self.cap_by_bus[dst_b] < load_to_move: 
                            continue 

                        min_new_deadhead = new_dh
                        best_dst_b = dst_b
                        best_insert_pos = i

            # 3. Apply Move
            if best_dst_b != -1:
                # Backup
                src_snapshot = copy.deepcopy(individual[src_b])
                dst_snapshot = copy.deepcopy(individual[best_dst_b])
                
                # Execute
                trip_data = individual[src_b].pop(src_t)
                individual[best_dst_b].insert(best_insert_pos, trip_data)
                
                # Fix Connectivity
                individual[src_b] = self._fix_depot_connectivity(individual[src_b], origin=self.origin_by_bus[src_b])
                individual[best_dst_b] = self._fix_depot_connectivity(individual[best_dst_b], origin=self.origin_by_bus[best_dst_b])
                
                # Evaluate
                if self._is_feasible_individual(individual, buses_count, n_depots, durations_matrix, demand_full):
                    new_cost = self._evaluate_fitness(
                        individual, buses_count, bus_capacity, depots, facilities, n_depots,
                        durations_matrix, demand_full,
                        penalty_factor, latest_evacuation_penalty_factor
                    )
                    
                    # Accept if better
                    if new_cost < base_cost:
                        return True
                
                # Revert
                individual[src_b] = src_snapshot
                individual[best_dst_b] = dst_snapshot

        return False
    def _try_relocate_moves(self, individual, buses_count, bus_capacity, depots, facilities, n_depots,
                            durations_matrix, demand_full, penalty_factor,
                            latest_evacuation_penalty_factor, rcl_size=24, allow_split_moves=True) -> bool:
        base_cost = self._evaluate_fitness(
            individual, buses_count, bus_capacity, depots, facilities, n_depots,
            durations_matrix, demand_full,
            penalty_factor, latest_evacuation_penalty_factor,
        )
        candidates = []
        for b in range(buses_count):
            for t_idx, trip in enumerate(individual[b]):
                for s_idx, (node, cnt) in enumerate(trip.get("stops", [])):
                    candidates.append((b, t_idx, s_idx, node, cnt))
        if not candidates: return False

        random.shuffle(candidates)
        candidates = candidates[: min(rcl_size, len(candidates))]

        for (src_b, src_t, src_s, node, cnt) in candidates:
            if src_t >= len(individual[src_b]): continue
            src_trip = individual[src_b][src_t]
            if src_s >= len(src_trip["stops"]): continue
            stop_to_move = src_trip["stops"][src_s]

            src_trip["stops"].pop(src_s)
            for dst_b in range(buses_count):
                for dst_t in range(len(individual[dst_b])):
                    dst_trip = individual[dst_b][dst_t]
                    cap_left = self._capacity_left(dst_trip, dst_b)
                    for pos in range(len(dst_trip["stops"]) + 1):
                        if cap_left >= cnt:
                            dst_trip["stops"].insert(pos, stop_to_move)
                            if self._is_feasible_individual(individual, buses_count, n_depots, durations_matrix, demand_full):
                                new_cost = self._evaluate_fitness(individual, buses_count, bus_capacity, depots, facilities, n_depots, durations_matrix, demand_full, penalty_factor, latest_evacuation_penalty_factor)
                                if new_cost < base_cost: return True
                            dst_trip["stops"].pop(pos)
                        elif allow_split_moves and cap_left > 0:
                            part = (node, cap_left)
                            remainder = (node, cnt - cap_left)
                            dst_trip["stops"].insert(pos, part)
                            src_trip["stops"].append(remainder)
                            if self._is_feasible_individual(individual, buses_count, n_depots, durations_matrix, demand_full):
                                new_cost = self._evaluate_fitness(individual, buses_count, bus_capacity, depots, facilities, n_depots, durations_matrix, demand_full, penalty_factor, latest_evacuation_penalty_factor)
                                if new_cost < base_cost: return True
                            dst_trip["stops"].pop(pos)
                            src_trip["stops"].pop()
            if src_s <= len(src_trip["stops"]):
                src_trip["stops"].insert(src_s, stop_to_move)
            else:
                src_trip["stops"].append(stop_to_move)
        return False

    def _try_swap_stops(self, individual, buses_count, bus_capacity, depots, facilities, n_depots,
                        durations_matrix, demand_full, penalty_factor,
                        latest_evacuation_penalty_factor, rcl_size=24) -> bool:
        base_cost = self._evaluate_fitness(
            individual, buses_count, bus_capacity, depots, facilities, n_depots,
            durations_matrix, demand_full,
            penalty_factor, latest_evacuation_penalty_factor,
        )
        stops = []
        for b in range(buses_count):
            for t_idx, trip in enumerate(individual[b]):
                for s_idx, (node, cnt) in enumerate(trip.get("stops", [])):
                    stops.append((b, t_idx, s_idx, node, cnt))
        if len(stops) < 2: return False
        random.shuffle(stops)
        stops = stops[: min(rcl_size, len(stops))]

        for i in range(len(stops)):
            for j in range(i + 1, len(stops)):
                b1, t1, s1, node1, cnt1 = stops[i]
                b2, t2, s2, node2, cnt2 = stops[j]
                if (b1, t1) == (b2, t2) and s1 == s2: continue
                if t1 >= len(individual[b1]) or t2 >= len(individual[b2]): continue
                trip1 = individual[b1][t1]
                trip2 = individual[b2][t2]
                if s1 >= len(trip1["stops"]) or s2 >= len(trip2["stops"]): continue

                new_load1 = self._trip_load(trip1) - cnt1 + cnt2
                new_load2 = self._trip_load(trip2) - cnt2 + cnt1
                cap1 = self.cap_by_bus[b1] if self.cap_by_bus and b1 < len(self.cap_by_bus) else self.bus_capacity
                cap2 = self.cap_by_bus[b2] if self.cap_by_bus and b2 < len(self.cap_by_bus) else self.bus_capacity
                if new_load1 <= cap1 and new_load2 <= cap2:
                    trip1["stops"][s1], trip2["stops"][s2] = trip2["stops"][s2], trip1["stops"][s1]
                    if self._is_feasible_individual(individual, buses_count, n_depots, durations_matrix, demand_full):
                        new_cost = self._evaluate_fitness(individual, buses_count, bus_capacity, depots, facilities, n_depots, durations_matrix, demand_full, penalty_factor, latest_evacuation_penalty_factor)
                        if new_cost < base_cost: return True
                    trip1["stops"][s1], trip2["stops"][s2] = trip2["stops"][s2], trip1["stops"][s1]
        return False

    def _try_swap_trips(self, individual, buses_count, bus_capacity, depots, facilities, n_depots,
                        durations_matrix, demand_full, penalty_factor,
                        latest_evacuation_penalty_factor, max_checks=1000) -> bool: # <--- Added param
        base_cost = self._evaluate_fitness(
            individual, buses_count, bus_capacity, depots, facilities, n_depots,
            durations_matrix, demand_full,
            penalty_factor, latest_evacuation_penalty_factor,
        )
        
        trip_positions = []
        for b in range(buses_count):
            for t_idx, trip in enumerate(individual[b]):
                if trip.get("stops"): trip_positions.append((b, t_idx))
        
        if len(trip_positions) < 2: return False
        
        # Randomize so we don't always optimize the same buses first
        random.shuffle(trip_positions)
        
        checks_performed = 0  # <--- Counter
        
        for i in range(len(trip_positions)):
            for j in range(i + 1, len(trip_positions)):
                
                # --- SAFETY BRAKE ---
                checks_performed += 1
                if checks_performed > max_checks:
                    return False # Abort: Neighborhood too large, took too long
                # --------------------

                b1, t1 = trip_positions[i]
                b2, t2 = trip_positions[j]
                
                if b1 == b2: continue
                if t1 >= len(individual[b1]) or t2 >= len(individual[b2]): continue
                
                trip1 = individual[b1][t1]
                trip2 = individual[b2][t2]
                
                load1 = self._trip_load(trip1)
                load2 = self._trip_load(trip2)
                
                # Use cached capacities for speed
                cap1 = self.cap_by_bus[b1]
                cap2 = self.cap_by_bus[b2]
                
                if load1 <= cap2 and load2 <= cap1:
                    bus1_before = copy.deepcopy(individual[b1])
                    bus2_before = copy.deepcopy(individual[b2])
                    
                    individual[b1][t1], individual[b2][t2] = trip2, trip1
                    
                    individual[b1] = self._fix_depot_connectivity(individual[b1], origin=self.origin_by_bus[b1])
                    individual[b2] = self._fix_depot_connectivity(individual[b2], origin=self.origin_by_bus[b2])
                    
                    if self._is_feasible_individual(individual, buses_count, n_depots, durations_matrix, demand_full):
                        new_cost = self._evaluate_fitness(individual, buses_count, bus_capacity, depots, facilities, n_depots, durations_matrix, demand_full, penalty_factor, latest_evacuation_penalty_factor)
                        if new_cost < base_cost: 
                            return True
                    
                    individual[b1] = bus1_before
                    individual[b2] = bus2_before
                    
        return False

    def _try_move_trip(self, individual, buses_count, bus_capacity, depots, facilities, n_depots,
                       durations_matrix, demand_full, penalty_factor,
                       latest_evacuation_penalty_factor, max_checks=1000) -> bool: # <--- Added param
        base_cost = self._evaluate_fitness(
            individual, buses_count, bus_capacity, depots, facilities, n_depots,
            durations_matrix, demand_full,
            penalty_factor, latest_evacuation_penalty_factor,
        )
        
        trip_positions = []
        for b in range(buses_count):
            for t_idx, trip in enumerate(individual[b]):
                if trip.get("stops"): trip_positions.append((b, t_idx))
        
        if not trip_positions: return False
        
        random.shuffle(trip_positions)
        checks_performed = 0 # <--- Counter
        
        for src_b, src_t in trip_positions:
            if src_t >= len(individual[src_b]): continue
            
            trip_to_move = individual[src_b][src_t]
            trip_load = self._trip_load(trip_to_move)
            
            src_before = copy.deepcopy(individual[src_b])
            del individual[src_b][src_t]
            individual[src_b] = self._fix_depot_connectivity(individual[src_b], origin=self.origin_by_bus[src_b])
            
            # Randomize destination buses so we don't bias towards Bus 0
            dest_candidates = list(range(buses_count))
            random.shuffle(dest_candidates)

            for dst_b in dest_candidates:
                # Capacity Pre-check
                if self.cap_by_bus[dst_b] < trip_load: continue
                
                dst_before = copy.deepcopy(individual[dst_b])
                
                # Check all insertion positions (Start, Middle, End)
                for pos in range(len(individual[dst_b]) + 1):
                    
                    # --- SAFETY BRAKE ---
                    checks_performed += 1
                    if checks_performed > max_checks:
                        # Revert the source deletion before aborting!
                        individual[src_b] = src_before
                        return False 
                    # --------------------

                    individual[dst_b].insert(pos, copy.deepcopy(trip_to_move))
                    individual[dst_b] = self._fix_depot_connectivity(individual[dst_b], origin=self.origin_by_bus[dst_b])
                    
                    if self._is_feasible_individual(individual, buses_count, n_depots, durations_matrix, demand_full):
                        new_cost = self._evaluate_fitness(individual, buses_count, bus_capacity, depots, facilities, n_depots, durations_matrix, demand_full, penalty_factor, latest_evacuation_penalty_factor)
                        if new_cost < base_cost: 
                            return True
                    
                    # Revert destination for next position check
                    individual[dst_b] = copy.deepcopy(dst_before)
            
            # Revert source for next trip check
            individual[src_b] = src_before
            
        return False

    def _try_change_end_depot(self, individual, buses_count, bus_capacity, depots, facilities, n_depots,
                              durations_matrix, demand_full, penalty_factor,
                              latest_evacuation_penalty_factor) -> bool:
        if n_depots <= 1: return False
        base_cost = self._evaluate_fitness(
            individual, buses_count, bus_capacity, depots, facilities, n_depots,
            durations_matrix, demand_full,
            penalty_factor, latest_evacuation_penalty_factor,
        )
        for b_idx in range(buses_count):
            bus_schedule = individual[b_idx]
            if not bus_schedule: continue
            for t_idx in range(len(bus_schedule)):
                original_schedule_snapshot = copy.deepcopy(bus_schedule)
                original_end_depot = bus_schedule[t_idx]["end_depot"]
                for new_depot in range(n_depots):
                    if new_depot == original_end_depot: continue
                    bus_schedule[t_idx]["end_depot"] = new_depot
                    individual[b_idx] = self._fix_depot_connectivity(bus_schedule, origin=self.origin_by_bus[b_idx])
                    new_cost = self._evaluate_fitness(individual, buses_count, bus_capacity, depots, facilities, n_depots, durations_matrix, demand_full, penalty_factor, latest_evacuation_penalty_factor)
                    if new_cost < base_cost: return True
                    else:
                        individual[b_idx] = copy.deepcopy(original_schedule_snapshot)
                        bus_schedule = individual[b_idx]
        return False

    def _try_quantity_rebalance(self, individual, buses_count, bus_capacity, depots, facilities, n_depots,
                                durations_matrix, demand_full) -> bool:
        occurrences = {}
        for b in range(buses_count):
            for t_idx, trip in enumerate(individual[b]):
                sched = self._compute_trip_schedule(trip, n_depots, durations_matrix, bus_idx=b, trip_idx=t_idx)
                arr = sched["arrival_times"]
                for s_idx, (node, cnt) in enumerate(trip.get("stops", [])):
                    if cnt <= 0: continue
                    t_arr = arr[s_idx] if s_idx < len(arr) else float("inf")
                    occurrences.setdefault(node, []).append((b, t_idx, s_idx, t_arr, cnt))

        for node, occs in occurrences.items():
            if len(occs) < 2: continue
            occs.sort(key=lambda x: x[3])
            for i in range(len(occs) - 1):
                b_e, t_e, s_e, t_early, cnt_e = occs[i]
                for j in range(i + 1, len(occs)):
                    b_l, t_l, s_l, t_late, cnt_l = occs[j]
                    if t_late <= t_early: continue
                    if t_e >= len(individual[b_e]) or t_l >= len(individual[b_l]): continue
                    early_trip = individual[b_e][t_e]
                    late_trip = individual[b_l][t_l]
                    if s_e >= len(early_trip["stops"]) or s_l >= len(late_trip["stops"]): continue
                    spare = self._capacity_left(early_trip, b_e)
                    if spare <= 0 or cnt_l <= 0: continue
                    shift = min(spare, cnt_l)
                    old_e = early_trip["stops"][s_e]
                    old_l = late_trip["stops"][s_l]
                    early_trip["stops"][s_e] = (node, old_e[1] + shift)
                    new_l_cnt = old_l[1] - shift
                    if new_l_cnt > 0:
                        late_trip["stops"][s_l] = (node, new_l_cnt)
                    else:
                        late_trip["stops"].pop(s_l)
                    if self._is_feasible_individual(individual, buses_count, n_depots, durations_matrix, demand_full): return True
                    if new_l_cnt > 0: late_trip["stops"][s_l] = old_l
                    else: late_trip["stops"].insert(min(s_l, len(late_trip["stops"])), old_l)
                    early_trip["stops"][s_e] = old_e
        return False

    def _try_balance_makespan(self, individual, buses_count, bus_capacity, depots, facilities, n_depots,
                              durations_matrix, demand_full, penalty_factor,
                              latest_evacuation_penalty_factor) -> bool:
        base_cost = self._evaluate_fitness(
            individual, buses_count, bus_capacity, depots, facilities, n_depots,
            durations_matrix, demand_full,
            penalty_factor, latest_evacuation_penalty_factor,
        )
        if buses_count <= 1: return False
        finish_times = self._bus_finish_times(individual, n_depots, durations_matrix)
        if not any(math.isfinite(t) for t in finish_times): return False
        b_max_idx = max(range(buses_count), key=lambda b: finish_times[b])
        if not individual[b_max_idx]: return False
        tail_trip_idx = -1
        for i in range(len(individual[b_max_idx]) - 1, -1, -1):
            if individual[b_max_idx][i].get("stops"):
                tail_trip_idx = i
                break
        if tail_trip_idx == -1: return False
        candidate_buses = sorted([b for b in range(buses_count) if b != b_max_idx], key=lambda b: finish_times[b])
        tail_trip_stops = individual[b_max_idx][tail_trip_idx]["stops"]
        for s_idx in range(len(tail_trip_stops) - 1, -1, -1):
            node, count = tail_trip_stops[s_idx]
            if count <= 0: continue
            for bus_idx in candidate_buses:
                if individual[bus_idx]:
                    last_trip = individual[bus_idx][-1]
                    spare_capacity = self._capacity_left(last_trip, bus_idx)
                    if spare_capacity > 0:
                        shift_amount = min(count, spare_capacity)
                        temp_individual = copy.deepcopy(individual)
                        temp_tail_trip = temp_individual[b_max_idx][tail_trip_idx]
                        temp_last_trip = temp_individual[bus_idx][-1]
                        temp_tail_trip["stops"][s_idx] = (node, count - shift_amount)
                        found = False
                        for i in range(len(temp_last_trip["stops"])):
                            if temp_last_trip["stops"][i][0] == node:
                                temp_last_trip["stops"][i] = (node, temp_last_trip["stops"][i][1] + shift_amount)
                                found = True
                                break
                        if not found: temp_last_trip["stops"].append((node, shift_amount))
                        temp_tail_trip["stops"] = [s for s in temp_tail_trip["stops"] if s[1] > 0]
                        if self._is_feasible_individual(temp_individual, buses_count, n_depots, durations_matrix, demand_full):
                            new_cost = self._evaluate_fitness(temp_individual, buses_count, bus_capacity, depots, facilities, n_depots, durations_matrix, demand_full, penalty_factor, latest_evacuation_penalty_factor)
                            if new_cost < base_cost:
                                individual[:] = temp_individual
                                return True
                bus_cap = self.cap_by_bus[bus_idx]
                shift_amount = min(count, bus_cap)
                if shift_amount > 0:
                    temp_individual = copy.deepcopy(individual)
                    temp_tail_trip = temp_individual[b_max_idx][tail_trip_idx]
                    temp_tail_trip["stops"][s_idx] = (node, count - shift_amount)
                    temp_tail_trip["stops"] = [s for s in temp_tail_trip["stops"] if s[1] > 0]
                    start_depot = temp_individual[bus_idx][-1]["end_depot"] if temp_individual[bus_idx] else (self.origin_by_bus[bus_idx]['index'] if self.origin_by_bus[bus_idx]['kind'] == 'depot' else 0)
                    depot_loads = self._calculate_depot_loads(temp_individual)
                    best_depot = self._find_best_end_depot(node, shift_amount, depot_loads)
                    new_trip = {"start_depot": start_depot, "stops": [(node, shift_amount)], "end_depot": best_depot}
                    temp_individual[bus_idx].append(new_trip)
                    if self._is_feasible_individual(temp_individual, buses_count, n_depots, durations_matrix, demand_full):
                        new_cost = self._evaluate_fitness(temp_individual, buses_count, bus_capacity, depots, facilities, n_depots, durations_matrix, demand_full, penalty_factor, latest_evacuation_penalty_factor)
                        if new_cost < base_cost:
                            individual[:] = temp_individual
                            return True
        return False

    def _try_takeover_near_gap(self, individual, buses_count, bus_capacity, depots, facilities, n_depots,
                               durations_matrix, demand_full, penalty_factor,
                               latest_evacuation_penalty_factor, gap_window_minutes=12.0, rcl_size=12) -> bool:
        base_cost = self._evaluate_fitness(
            individual, buses_count, bus_capacity, depots, facilities, n_depots,
            durations_matrix, demand_full,
            penalty_factor, latest_evacuation_penalty_factor,
        )
        base_latest = self._latest_finish_time(individual, n_depots, durations_matrix)
        if buses_count <= 1: return False
        timelines = self._all_bus_timelines(individual, n_depots, durations_matrix)
        finish_times = [tl[-1][1] if tl else 0.0 for tl in timelines]
        candidates = []
        for bi in range(buses_count):
            ti = finish_times[bi]
            if not math.isfinite(ti): continue
            for bj in range(buses_count):
                if bj == bi: continue
                tl_j = timelines[bj]
                for k, (dj_k, _rj_k) in enumerate(tl_j):
                    gap = dj_k - ti
                    if 0.0 <= gap <= gap_window_minutes:
                        candidates.append((gap, bi, bj, k))
        if not candidates: return False
        candidates.sort(key=lambda x: x[0])
        candidates = candidates[: min(rcl_size, len(candidates))]
        for _, bi, bj, k in candidates:
            if k >= len(individual[bj]): continue
            bus_i_before = copy.deepcopy(individual[bi])
            bus_j_before = copy.deepcopy(individual[bj])
            moved = copy.deepcopy(individual[bj][k])
            tmp_j = copy.deepcopy(bus_j_before)
            del tmp_j[k]
            individual[bj] = self._fix_depot_connectivity(tmp_j, origin=self.origin_by_bus[bj])
            tmp_i = copy.deepcopy(bus_i_before)
            tmp_i.append(moved)
            individual[bi] = self._fix_depot_connectivity(tmp_i, origin=self.origin_by_bus[bi])
            if self._is_feasible_individual(individual, buses_count, n_depots, durations_matrix, demand_full):
                new_latest = self._latest_finish_time(individual, n_depots, durations_matrix)
                new_cost = self._evaluate_fitness(individual, buses_count, bus_capacity, depots, facilities, n_depots, durations_matrix, demand_full, penalty_factor, latest_evacuation_penalty_factor)
                if (new_cost + 1e-9 < base_cost) or (new_latest + 1e-9 < base_latest and new_cost <= base_cost + 1e-9): return True
            individual[bi] = bus_i_before
            individual[bj] = bus_j_before
        return False

    def _try_fill_idle_time(self, individual, buses_count, bus_capacity, depots, facilities, n_depots,
                            durations_matrix, demand_full, penalty_factor,
                            latest_evacuation_penalty_factor, idle_threshold_minutes=20.0) -> bool:
        base_cost = self._evaluate_fitness(
            individual, buses_count, bus_capacity, depots, facilities, n_depots,
            durations_matrix, demand_full,
            penalty_factor, latest_evacuation_penalty_factor,
        )
        timelines = self._all_bus_timelines(individual, n_depots, durations_matrix)
        finish_times = [tl[-1][1] if tl else 0.0 for tl in timelines]
        if not finish_times: return False
        T_latest = max(finish_times)
        idle_buses = []
        for b_idx, t_finish in enumerate(finish_times):
            if T_latest - t_finish > idle_threshold_minutes:
                idle_buses.append((b_idx, t_finish))
        if not idle_buses: return False
        latest_pickups = []
        for b_idx, tl in enumerate(timelines):
            if not individual[b_idx]: continue
            for t_idx, (t_start, t_end) in enumerate(tl):
                trip = individual[b_idx][t_idx]
                trip_sched = self._compute_trip_schedule(trip, n_depots, durations_matrix, bus_idx=b_idx, trip_idx=t_idx)
                for s_idx, (node, count) in enumerate(trip["stops"]):
                    arrival_time = t_start + trip_sched["arrival_times"][s_idx]
                    latest_pickups.append({"arrival": arrival_time, "bus": b_idx, "trip": t_idx, "stop_idx": s_idx, "node": node, "count": count})
        if not latest_pickups: return False
        latest_pickups.sort(key=lambda p: p["arrival"], reverse=True)
        for pickup in latest_pickups:
            src_bus, src_trip_idx, src_stop_idx = pickup["bus"], pickup["trip"], pickup["stop_idx"]
            node, count = pickup["node"], pickup["count"]
            for idle_bus_idx, idle_finish_time in idle_buses:
                bus_cap = self.cap_by_bus[idle_bus_idx]
                shift_amount = min(count, bus_cap)
                if shift_amount <= 0: continue
                start_depot = individual[idle_bus_idx][-1]["end_depot"] if individual[idle_bus_idx] else 0
                temp_new_trip = {"start_depot": start_depot, "stops": [(node, shift_amount)], "end_depot": 0}
                trip_duration_est = self._compute_trip_schedule(temp_new_trip, n_depots, durations_matrix, bus_idx=idle_bus_idx, trip_idx=len(individual[idle_bus_idx]))["trip_time"]
                if idle_finish_time + trip_duration_est < T_latest:
                    temp_individual = copy.deepcopy(individual)
                    source_trip = temp_individual[src_bus][src_trip_idx]
                    original_stop = source_trip["stops"][src_stop_idx]
                    source_trip["stops"][src_stop_idx] = (original_stop[0], original_stop[1] - shift_amount)
                    source_trip["stops"] = [s for s in source_trip["stops"] if s[1] > 0]
                    depot_loads = self._calculate_depot_loads(temp_individual)
                    best_depot = self._find_best_end_depot(node, shift_amount, depot_loads)
                    temp_new_trip["end_depot"] = best_depot
                    temp_new_trip["start_depot"] = temp_individual[idle_bus_idx][-1]["end_depot"] if temp_individual[idle_bus_idx] else (self.origin_by_bus[idle_bus_idx]['index'] if self.origin_by_bus[idle_bus_idx]['kind'] == 'depot' else 0)
                    temp_individual[idle_bus_idx].append(temp_new_trip)
                    if self._is_feasible_individual(temp_individual, buses_count, n_depots, durations_matrix, demand_full):
                        new_cost = self._evaluate_fitness(temp_individual, buses_count, bus_capacity, depots, facilities, n_depots, durations_matrix, demand_full, penalty_factor, latest_evacuation_penalty_factor)
                        if new_cost < base_cost:
                            individual[:] = temp_individual
                            return True
        return False

    def _optimize_trip_stop_sequence(self, start_depot, stops, end_depot, bus_idx):
        """
        Optimizes stop sequence using 2-Opt Local Search.
        This uncrosses inefficient paths (e.g., A->C->B->D becomes A->B->C->D).
        """
        if not stops or len(stops) <= 1:
            return stops

        # 1. Helper to calculate total travel time for a sequence
        def get_seq_cost(seq):
            t = 0.0
            # Start Leg
            t += self.durations_matrix.get((start_depot, self.n_depots + seq[0][0]), 1000)
            # Inter-stop legs
            for i in range(len(seq) - 1):
                t += self.durations_matrix.get((self.n_depots + seq[i][0], self.n_depots + seq[i+1][0]), 1000)
            # End Leg
            t += self.durations_matrix.get((self.n_depots + seq[-1][0], end_depot), 1000)
            return t

        best_seq = list(stops)
        best_cost = get_seq_cost(best_seq)
        improved = True

        # 2. Apply 2-Opt Swaps
        # Limit iterations to avoid slow-downs on huge trips
        max_no_improve = 0
        while improved and max_no_improve < 5: 
            improved = False
            for i in range(len(best_seq) - 1):
                for j in range(i + 1, len(best_seq)):
                    if j - i == 1: continue # No point swapping adjacent edges in this specific way
                    
                    # Create sequence with the section reversed
                    new_seq = best_seq[:i] + best_seq[i:j][::-1] + best_seq[j:]
                    
                    new_cost = get_seq_cost(new_seq)
                    if new_cost < best_cost:
                        best_seq = new_seq
                        best_cost = new_cost
                        improved = True
                        max_no_improve = 0
            if not improved:
                max_no_improve += 1
                
        return best_seq

    def _try_consolidate_trips(self, individual, buses_count, bus_capacity, depots, facilities, n_depots,
                               durations_matrix, demand_full, penalty_factor,
                               latest_evacuation_penalty_factor) -> bool:
        """
        Aggressively merges ANY two trips that fit within bus capacity.
        """
        base_cost = self._evaluate_fitness(
            individual, buses_count, bus_capacity, depots, facilities, n_depots,
            durations_matrix, demand_full,
            penalty_factor, latest_evacuation_penalty_factor,
        )

        # Collect all trips with their metadata
        all_trips = []
        for b_idx in range(buses_count):
            for t_idx, trip in enumerate(individual[b_idx]):
                if trip.get("stops"):
                    load = self._trip_load(trip)
                    all_trips.append({
                        'b': b_idx, 't': t_idx, 'load': load, 'trip': trip
                    })

        # Sort by load (smallest first) to pack crumbs into larger trips
        all_trips.sort(key=lambda x: x['load'])

        for src in all_trips:
            # Check if source still exists (we might have deleted it in a previous iter)
            if src['t'] >= len(individual[src['b']]): continue
            
            current_src_trip = individual[src['b']][src['t']]
            # Double check object identity/content to ensure we haven't shifted
            if current_src_trip != src['trip']: continue 

            for dst in all_trips:
                if src == dst: continue
                # Check existence
                if dst['t'] >= len(individual[dst['b']]): continue
                
                # Capacity check
                target_bus_cap = self.cap_by_bus[dst['b']]
                if src['load'] + dst['load'] <= target_bus_cap:
                    
                    # Snapshot
                    src_sched_backup = copy.deepcopy(individual[src['b']])
                    dst_sched_backup = copy.deepcopy(individual[dst['b']])

                    # MERGE LOGIC
                    # 1. Combine stops
                    target_trip = individual[dst['b']][dst['t']]
                    stops_to_add = current_src_trip["stops"]
                    target_trip["stops"].extend(stops_to_add)
                    
                    # 2. Optimize Sequence (Vital!)
                    target_trip["stops"] = self._optimize_trip_stop_sequence(
                        target_trip["start_depot"], target_trip["stops"], target_trip["end_depot"], dst['b']
                    )

                    # 3. Remove Source
                    # Be careful with indices if src and dst are on same bus
                    if src['b'] == dst['b']:
                        # If on same bus, we must pop the higher index first to avoid shifting the lower one incorrectly
                        if src['t'] > dst['t']:
                            individual[src['b']].pop(src['t'])
                        else:
                            individual[src['b']].pop(src['t'])
                            # Note: The 'dst' index has now shifted down by 1
                            # But we already modified the object reference 'target_trip', so the data is safe.
                    else:
                        individual[src['b']].pop(src['t'])

                    # 4. Fix Connectivity
                    individual[src['b']] = self._fix_depot_connectivity(individual[src['b']], origin=self.origin_by_bus[src['b']])
                    individual[dst['b']] = self._fix_depot_connectivity(individual[dst['b']], origin=self.origin_by_bus[dst['b']])

                    # 5. Evaluate
                    if self._is_feasible_individual(individual, buses_count, n_depots, durations_matrix, demand_full):
                        new_cost = self._evaluate_fitness(
                            individual, buses_count, bus_capacity, depots, facilities, n_depots,
                            durations_matrix, demand_full,
                            penalty_factor, latest_evacuation_penalty_factor
                        )
                        if new_cost < base_cost:
                            return True # Success

                    # Revert
                    individual[src['b']] = src_sched_backup
                    individual[dst['b']] = dst_sched_backup
                    
                    # Optimization: If we merged successfully, we returned. 
                    # If we failed, we continue, but usually only a few successful merges happen per gen.

        return False
    
    def _try_split_mixed_trips(self, individual, buses_count, bus_capacity, depots, facilities, n_depots,
                               durations_matrix, demand_full, penalty_factor,
                               latest_evacuation_penalty_factor) -> bool:
        """
        [THE PURIFIER]
        Strategy: High-capacity buses lose speed when making multi-stop trips.
        Action: Finds mixed trips (2+ stops) on big buses and kicks the smallest stop
        to a different bus (preferably a smaller/faster one), prioritizing 'Purity' over 'Capacity'.
        """
        base_cost = self._evaluate_fitness(
            individual, buses_count, bus_capacity, depots, facilities, n_depots,
            durations_matrix, demand_full,
            penalty_factor, latest_evacuation_penalty_factor,
        )

        # 1. Identify Bottleneck to avoid making it worse
        finish_times = self._bus_finish_times(individual, n_depots, durations_matrix)
        makespan = max(finish_times) if finish_times else 0.0

        # Sort candidate target buses by capacity (Smallest first -> "The Swarm")
        candidate_buses = sorted(range(buses_count), key=lambda b: self.cap_by_bus[b])

        for b_idx in range(buses_count):
            # Apply mainly to larger buses (Capacity >= 20) or if fleet is homogenous
            if self.cap_by_bus[b_idx] < 20 and buses_count > 1:
                continue

            for t_idx, trip in enumerate(individual[b_idx]):
                stops = trip.get("stops", [])
                
                # Only split mixed trips
                if len(stops) < 2: continue

                # Identify the "Minor" stop (fewest people)
                # This is the "impurity" we want to remove
                minor_s_idx, (minor_node, minor_cnt) = min(enumerate(stops), key=lambda x: x[1][1])

                # Backup state (Save only the buses involved to save RAM)
                src_backup = copy.deepcopy(individual[b_idx])

                # Remove from source
                trip["stops"].pop(minor_s_idx)
                
                # Try to move to a target bus
                for target_b in candidate_buses:
                    if target_b == b_idx: continue
                    
                    # Constraint: Don't overload the current bottleneck bus
                    if finish_times[target_b] > makespan * 0.95: continue

                    target_backup = copy.deepcopy(individual[target_b])
                    moved = False

                    # A. Try inserting into existing trips (Best for efficiency)
                    for tgt_trip in individual[target_b]:
                        if self._capacity_left(tgt_trip, target_b) >= minor_cnt:
                            tgt_trip["stops"].append((minor_node, minor_cnt))
                            # Re-optimize sequence
                            tgt_trip["stops"] = self._optimize_trip_stop_sequence(
                                tgt_trip["start_depot"], tgt_trip["stops"], tgt_trip["end_depot"], target_b
                            )
                            moved = True
                            break
                    
                    # B. If A failed, append as a trip (best for speed/flow)
                    # Only do this if target bus has plenty of slack (< 80% of makespan)
                    if not moved and finish_times[target_b] < makespan * 0.80:
                        start_dep = individual[target_b][-1]["end_depot"] if individual[target_b] else \
                                    (self.origin_by_bus[target_b]['index'] if self.origin_by_bus[target_b]['kind'] == 'depot' else 0)
                        
                        depot_loads = self._calculate_depot_loads(individual)
                        best_end = self._find_best_end_depot(minor_node, minor_cnt, depot_loads)
                        
                        individual[target_b].append({
                            "start_depot": start_dep,
                            "stops": [(minor_node, minor_cnt)],
                            "end_depot": best_end
                        })
                        moved = True

                    if moved:
                        # Fix connectivity & Evaluate
                        individual[b_idx] = self._fix_depot_connectivity(individual[b_idx], origin=self.origin_by_bus[b_idx])
                        individual[target_b] = self._fix_depot_connectivity(individual[target_b], origin=self.origin_by_bus[target_b])

                        if self._is_feasible_individual(individual, buses_count, n_depots, durations_matrix, demand_full):
                            new_cost = self._evaluate_fitness(
                                individual, buses_count, bus_capacity, depots, facilities, n_depots,
                                durations_matrix, demand_full,
                                penalty_factor, latest_evacuation_penalty_factor
                            )
                            if new_cost < base_cost:
                                return True # Success

                    # Revert target if failed
                    individual[target_b] = target_backup

                # Revert source if failed to move anywhere
                individual[b_idx] = src_backup

        return False
    
    def _try_self_consolidate(self, individual, buses_count, bus_capacity, depots, facilities, n_depots,
                              durations_matrix, demand_full, penalty_factor,
                              latest_evacuation_penalty_factor) -> bool:
        """
        [THE GLUTTON - FIXED]
        Strategy: A bus should never leave a node partially empty if it plans to return 
        to that SAME node later in its schedule.
        Action: Scans a single bus's schedule. Moves passengers from Future Trips -> Earlier Trips
        visiting the same node.
        """
        base_cost = self._evaluate_fitness(
            individual, buses_count, bus_capacity, depots, facilities, n_depots,
            durations_matrix, demand_full,
            penalty_factor, latest_evacuation_penalty_factor,
        )

        improved = False

        for b_idx in range(buses_count):
            trips = individual[b_idx]
            if len(trips) < 2: continue
            
            cap = self.cap_by_bus[b_idx]

            # Use a while loop because the list length 'len(trips)' changes dynamically 
            # as we merge and delete empty future trips.
            t_early_idx = 0
            while t_early_idx < len(trips) - 1:
                trip_early = trips[t_early_idx]
                
                # Check if Early trip has space
                load_early = self._trip_load(trip_early)
                space_early = cap - load_early
                
                # If full, nothing to do, move to next
                if space_early <= 0: 
                    t_early_idx += 1
                    continue

                stops_early = trip_early.get("stops", [])

                # Look ahead at future trips (Iterate BACKWARDS to safely delete)
                # Note: We re-check len(trips) every time because we might have deleted items
                t_late_idx = len(trips) - 1
                
                while t_late_idx > t_early_idx:
                    # Double check bounds just to be safe
                    if t_late_idx >= len(trips):
                        t_late_idx -= 1
                        continue

                    trip_late = trips[t_late_idx]
                    stops_late = trip_late.get("stops", [])
                    
                    if not stops_late:
                        # Defensive: remove already empty trip if found
                        del trips[t_late_idx]
                        t_late_idx -= 1
                        continue
                        
                    if space_early <= 0: 
                        break

                    # Check for matching nodes
                    # Iterate stops on the Late trip backwards (to allow popping)
                    for s_late_idx in range(len(stops_late) - 1, -1, -1):
                        if space_early <= 0: break
                        
                        node_l, cnt_l = stops_late[s_late_idx]
                        
                        # Compare against all stops in Early trip
                        for s_early_idx, (node_e, cnt_e) in enumerate(stops_early):
                            if node_l == node_e:
                                # FOUND MATCH: Merge Late -> Early
                                amount_to_move = min(space_early, cnt_l)
                                
                                if amount_to_move > 0:
                                    # Update Early (Add load)
                                    curr_n, curr_c = stops_early[s_early_idx]
                                    stops_early[s_early_idx] = (curr_n, curr_c + amount_to_move)
                                    
                                    # Update Late (Remove load)
                                    if cnt_l - amount_to_move <= 0:
                                        stops_late.pop(s_late_idx)
                                    else:
                                        stops_late[s_late_idx] = (node_l, cnt_l - amount_to_move)
                                    
                                    space_early -= amount_to_move
                                    improved = True
                                    
                                    # If we emptied the Late Trip completely, delete it
                                    if not stops_late:
                                        del trips[t_late_idx]
                                        # Trip is gone, stop checking stops for this trip
                                        break 
                                break # Stop checking early nodes, we found our match
                    
                    t_late_idx -= 1
                
                t_early_idx += 1

        if improved:
            # Repair and Evaluate
            for b in range(buses_count):
                if individual[b]:
                    individual[b] = self._fix_depot_connectivity(individual[b], origin=self.origin_by_bus[b])
            
            if self._is_feasible_individual(individual, buses_count, n_depots, durations_matrix, demand_full):
                new_cost = self._evaluate_fitness(
                    individual, buses_count, bus_capacity, depots, facilities, n_depots,
                    durations_matrix, demand_full,
                    penalty_factor, latest_evacuation_penalty_factor
                )
                
                # Accept if cost is same or better (prioritize consolidation)
                if new_cost <= base_cost + 1e-6:
                    return True
        
        return False

    def _try_crumb_extraction(self, individual, buses_count, bus_capacity, depots, facilities, n_depots,
                               durations_matrix, demand_full, penalty_factor,
                               latest_evacuation_penalty_factor) -> bool:
        """
        [THE VACUUM - FIXED]
        Strategy: The bus determining the makespan (Bottleneck) is often slowed down by tiny pickups.
        Action: Identifies "Crumbs" (<15% cap) on the Bottleneck Bus and forces them
        onto the Slackest Bus (Min Makespan), usually creating a trip there.
        """
        base_cost = self._evaluate_fitness(
            individual, buses_count, bus_capacity, depots, facilities, n_depots,
            durations_matrix, demand_full,
            penalty_factor, latest_evacuation_penalty_factor,
        )

        finish_times = self._bus_finish_times(individual, n_depots, durations_matrix)
        if not finish_times: return False

        bottleneck_idx = finish_times.index(max(finish_times))
        slack_idx = finish_times.index(min(finish_times))

        if bottleneck_idx == slack_idx: return False

        bottleneck_cap = self.cap_by_bus[bottleneck_idx]
        crumb_threshold = max(2, int(bottleneck_cap * 0.15)) 

        # Scan Bottleneck Bus
        trips = individual[bottleneck_idx]
        for t_idx, trip in enumerate(trips):
            stops = trip.get("stops", [])
            
            # --- SAFE ITERATION: Backwards ---
            # We iterate backwards so that if we pop an item, it doesn't mess up indices 0..n
            for s_idx in range(len(stops) - 1, -1, -1):
                node, cnt = stops[s_idx]
                
                if cnt <= crumb_threshold:
                    # Found a crumb! 
                    
                    # 1. Create Backups (Deepcopy is safest here)
                    bn_backup = copy.deepcopy(individual[bottleneck_idx])
                    slack_backup = copy.deepcopy(individual[slack_idx])

                    # 2. Remove from Bottleneck
                    # This is safe because s_idx is valid for the current list state
                    trip["stops"].pop(s_idx)
                    
                    # If trip becomes empty, delete the trip entirely
                    trip_deleted = False
                    if not trip["stops"]:
                        del individual[bottleneck_idx][t_idx]
                        trip_deleted = True

                    # 3. Add to slack bus (append a trip)
                    # Determine start depot for the slack bus
                    if individual[slack_idx]:
                        start_dep = individual[slack_idx][-1]["end_depot"]
                    else:
                        origin = self.origin_by_bus[slack_idx]
                        start_dep = origin.get('index', 0) if origin.get('kind') == 'depot' else 0
                    
                    depot_loads = self._calculate_depot_loads(individual)
                    best_end = self._find_best_end_depot(node, cnt, depot_loads)

                    individual[slack_idx].append({
                        "start_depot": start_dep,
                        "stops": [(node, cnt)],
                        "end_depot": best_end
                    })

                    # 4. Fix Connectivity & Evaluate
                    # We must re-run connectivity fixes because we might have deleted a trip
                    individual[bottleneck_idx] = self._fix_depot_connectivity(individual[bottleneck_idx], origin=self.origin_by_bus[bottleneck_idx])
                    individual[slack_idx] = self._fix_depot_connectivity(individual[slack_idx], origin=self.origin_by_bus[slack_idx])

                    if self._is_feasible_individual(individual, buses_count, n_depots, durations_matrix, demand_full):
                        new_cost = self._evaluate_fitness(
                            individual, buses_count, bus_capacity, depots, facilities, n_depots,
                            durations_matrix, demand_full,
                            penalty_factor, latest_evacuation_penalty_factor
                        )
                        
                        # ACCEPT only if it helps global fitness (Makespan)
                        if new_cost < base_cost:
                            return True

                    # 5. Revert if not successful
                    individual[bottleneck_idx] = bn_backup
                    individual[slack_idx] = slack_backup
                    
                    # If we deleted the trip and reverted, 'trip' variable is stale/reset.
                    # Since we are inside a loop over stops of a trip, and we just reverted,
                    # we can continue to the next stop index (which is s_idx - 1).
                    # However, strictly speaking, 'trip' is a reference. 
                    # If bn_backup replaced the list in individual, 'trip' variable might point to old object.
                    # To be safe, we refresh the 'trip' reference or break.
                    # Since we usually only want one successful move per call, 
                    # we can just continue to try other stops. 
                    # But to be absolutely safe against object reference issues:
                    trip = individual[bottleneck_idx][t_idx] 
                    
        return False
    
    def get_last_run_stats(self) -> Dict[str, float]:
        """Returns flattened stats: {op_name: gain, op_name_cnt: count}"""
        if not hasattr(self, 'last_run_stats') or not self.last_run_stats:
            return {}
        # Return the flattened dict (compatible with ea.py loop)
        return self.last_run_stats.get("flat_stats", {})
    
    def _alns_shake(self, individual, buses_count, bus_capacity, depots, facilities, n_depots,
                    pickup_nodes, durations_matrix, demand_full, remove_fraction=0.12):
        all_stops = []
        for b in range(buses_count):
            for t_idx, trip in enumerate(individual[b]):
                for s_idx, (node, cnt) in enumerate(trip.get("stops", [])):
                    if cnt > 0: 
                        all_stops.append((b, t_idx, s_idx, node, cnt))
        
        if not all_stops: 
            return

        seed = random.choice(all_stops)
        _, _, _, seed_node, _ = seed
        
        def travel_node_to_node(a, b):
            ta = durations_matrix.get((n_depots + a, n_depots + b), float("inf")) / 60.0
            tb = durations_matrix.get((n_depots + b, n_depots + a), float("inf")) / 60.0
            return min(ta, tb)
        
        sorted_stops = sorted(all_stops, key=lambda x: travel_node_to_node(seed_node, x[3]))
        k = max(1, int(remove_fraction * len(sorted_stops)))
        removed = sorted_stops[:k]
        removed_items = []
        
        for (b, t_idx, _s_idx, node, cnt) in removed:
            if t_idx >= len(individual[b]): continue
            trip = individual[b][t_idx]
            pos = None
            for idx, (n, c) in enumerate(trip.get("stops", [])):
                if n == node and c == cnt:
                    pos = idx
                    break
            if pos is not None:
                removed_items.append((node, cnt))
                trip["stops"].pop(pos)
        
        if not removed_items: return

        # Helper closures for regret insertion
        def arc_time_from_depot(d, node): 
            return durations_matrix.get((d, n_depots + node), float("inf")) / 60.0
        
        def arc_time_node_node(a, b): 
            return durations_matrix.get((n_depots + a, n_depots + b), float("inf")) / 60.0
        
        def arc_time_to_depot(node, d): 
            return durations_matrix.get((n_depots + node, d), float("inf")) / 60.0
        
        def insertion_proxy(trip, pos, node):
            stops = trip.get("stops", [])
            start_dep, end_dep = trip.get("start_depot", 0), trip.get("end_depot", 0)
            pred_is_depot, pred = (True, start_dep) if pos == 0 else (False, stops[pos - 1][0])
            succ_is_depot, succ = (True, end_dep) if pos >= len(stops) else (False, stops[pos][0])
            
            t_pred_node = arc_time_from_depot(pred, node) if pred_is_depot else arc_time_node_node(pred, node)
            t_node_succ = arc_time_to_depot(node, succ) if succ_is_depot else arc_time_node_node(node, succ)
            
            if pred_is_depot and succ_is_depot: t_pred_succ = 0.0
            elif pred_is_depot: t_pred_succ = arc_time_from_depot(pred, succ)
            elif succ_is_depot: t_pred_succ = arc_time_to_depot(pred, succ)
            else: t_pred_succ = arc_time_node_node(pred, succ)
            
            delta = (t_pred_node + t_node_succ) - t_pred_succ
            return max(0.0, delta) if math.isfinite(delta) else float("inf")

        for (node, cnt) in removed_items:
            remaining = cnt
            while remaining > 0:
                candidates = []
                for b in range(buses_count):
                    for t_idx, trip in enumerate(individual[b]):
                        cap_left = self._capacity_left(trip, b)
                        if cap_left <= 0: continue
                        take = min(cap_left, remaining)
                        for pos in range(len(trip.get("stops", [])) + 1):
                            proxy = insertion_proxy(trip, pos, node)
                            candidates.append((proxy, b, t_idx, pos, take))
                
                if candidates:
                    candidates.sort(key=lambda x: x[0])
                    _, b, t_idx, pos, take = candidates[0]
                    individual[b][t_idx]["stops"].insert(pos, (node, take))
                    remaining -= take
                else:
                    b = random.randint(0, buses_count - 1)
                    cap_b = self.cap_by_bus[b] if b < len(self.cap_by_bus) else self.bus_capacity
                    take = min(remaining, cap_b)
                    if take <= 0: break
                    
                    if individual[b]: 
                        start_depot = individual[b][-1].get("end_depot", 0)
                    else:
                        origin = self.origin_by_bus[b] if b < len(self.origin_by_bus) else {"kind": "depot", "index": 0}
                        start_depot = origin.get("index", 0) if origin.get("kind") == "depot" else 0
                    
                    depot_loads = self._calculate_depot_loads(individual)
                    best_depot = self._find_best_end_depot(node, take, depot_loads)
                    new_trip = {"start_depot": start_depot, "stops": [(node, take)], "end_depot": best_depot}
                    individual[b].append(new_trip)
                    remaining -= take
        
        for b in range(buses_count):
            origin = self.origin_by_bus[b] if b < len(self.origin_by_bus) else {"kind": "depot", "index": 0}
            if individual[b]: 
                individual[b] = self._fix_depot_connectivity(individual[b], origin=origin)

    def _first_leg_minutes_memetic(self, bus_idx: int, first_node: int) -> float:
        if self.start_to_node_seconds and bus_idx in self.start_to_node_seconds:
            secs_map = self.start_to_node_seconds[bus_idx]
            if first_node in secs_map: 
                return float(secs_map[first_node]) / 60.0
        
        if not self.node_coords: 
            return 0.0
        
        origin = self.origin_by_bus[bus_idx] if self.origin_by_bus and bus_idx < len(self.origin_by_bus) else {"kind": "depot", "index": 0}
        if origin.get("kind") != "coord": 
            return 0.0
        
        nlat, nlon = self.node_coords.get(first_node, (None, None))
        if nlat is None: 
            return 0.0
        
        lat, lon = float(origin["lat"]), float(origin["lon"])
        R = 6371.0
        phi1, phi2 = math.radians(lat), math.radians(nlat)
        dphi, dlambda = math.radians(nlat - lat), math.radians(nlon - lon)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        km = R * c
        
        minutes = (km / max(1e-6, self.avg_speed_kmh)) * 60.0 * self.road_factor
        return minutes
