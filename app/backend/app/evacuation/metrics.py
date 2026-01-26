# Path: backend/app/evacuation/metrics.py
from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional
import math
import numpy as np # Import numpy for percentile calculation

SERVICE_TIME_MIN_DEFAULT = 10.0

### NEW CENTRAL SIMULATION HELPER ###
def _simulate_and_get_timings(
    individual: List[List[Dict[str, Any]]],
    n_depots: int,
    durations_matrix: Dict[Tuple[int, int], float],
    deadlines: Dict[int, float],
    origin_by_bus: List[Dict[str, Any]],
    cap_by_bus: List[int],
    depots: List[Dict[str, Any]],
    node_coords: Optional[Dict[int, Tuple[float, float]]],
    start_to_node_seconds: Optional[Dict[int, Dict[int, float]]],
    avg_speed_kmh: float,
    road_factor: float,
    use_dynamic_service_time: bool = False,
    service_time_base_min: float = 3.0,
    service_time_per_person_min: float = 20.0 / 60.0,
) -> Dict[str, Any]:
    """
    Central simulation logic for an individual. Single source of truth for timings.
    Returns a dictionary with detailed simulation results.
    """
    total_wait_pm = 0.0
    total_lateness_pm = 0.0
    all_return_times: List[float] = []
    all_pickup_times: List[float] = []
    total_people_evacuated = 0
    
    # --- NEW METRICS INITIALIZATION ---
    late_pickup_count = 0
    total_travel_time_min = 0.0
    bus_finish_times = [0.0] * len(individual)
    depot_loads = [0] * len(depots) if depots else []
    total_overfill = 0
    # --- END NEW ---

    def _first_leg_minutes_from_coord(bus_idx, lat, lon, first_node):
        if start_to_node_seconds and bus_idx in start_to_node_seconds:
            if first_node in start_to_node_seconds[bus_idx]:
                return float(start_to_node_seconds[bus_idx][first_node]) / 60.0
        if not node_coords or first_node not in node_coords:
            return float('inf')
        nlat, nlon = node_coords[first_node]
        km = _haversine_km(lat, lon, nlat, nlon)
        return (km / max(1e-6, avg_speed_kmh)) * 60.0 * road_factor

    for b_idx, bus_schedule in enumerate(individual):
        current_time = 0.0
        bus_capacity = cap_by_bus[b_idx] if b_idx < len(cap_by_bus) else 80

        for t_idx, trip in enumerate(bus_schedule):
            stops = trip.get("stops", [])
            if not stops:
                continue

            first_node, _ = stops[0]
            t = current_time
            
            # --- First leg travel ---
            travel_duration = 0.0
            if t_idx == 0:
                origin = origin_by_bus[b_idx]
                okind = origin.get("kind", "depot")
                if okind == "depot":
                    travel_duration = durations_matrix.get((int(origin.get("index", 0)), n_depots + first_node), float('inf')) / 60.0
                elif okind == "node":
                    start_node = int(origin.get("index"))
                    if start_node != first_node:
                        travel_duration = durations_matrix.get((n_depots + start_node, n_depots + first_node), float('inf')) / 60.0
                elif okind == "coord":
                    travel_duration = _first_leg_minutes_from_coord(b_idx, float(origin["lat"]), float(origin["lon"]), first_node)
                else:
                    travel_duration = durations_matrix.get((trip["start_depot"], n_depots + first_node), float('inf')) / 60.0
            else:
                travel_duration = durations_matrix.get((trip["start_depot"], n_depots + first_node), float('inf')) / 60.0
            t += travel_duration
            total_travel_time_min += travel_duration # Accumulate travel time

            total_people_in_trip = 0
            for i, (node, pickup_count) in enumerate(stops):
                total_wait_pm += pickup_count * t
                total_people_evacuated += pickup_count
                total_people_in_trip += pickup_count
                for _ in range(pickup_count):
                    all_pickup_times.append(t)
                
                dl = float(deadlines.get(node, float('inf')))
                if t > dl and math.isfinite(dl):
                    total_lateness_pm += (t - dl) * pickup_count
                    late_pickup_count += pickup_count # Accumulate late people count
                
                # --- UPDATED SERVICE TIME LOGIC ---
                if use_dynamic_service_time:
                    service_time = service_time_base_min + pickup_count * service_time_per_person_min
                else:
                    # UPDATED: Static now uses bus_capacity to penalize partial loads (make them as expensive as full ones)
                    service_time = service_time_base_min + bus_capacity * service_time_per_person_min
                
                t += service_time

                if i < len(stops) - 1:
                    nxt, _ = stops[i + 1]
                    travel_duration = durations_matrix.get((n_depots + node, n_depots + nxt), float('inf')) / 60.0
                    t += travel_duration
                    total_travel_time_min += travel_duration # Accumulate travel time

            last_node, _ = stops[-1]
            travel_duration = durations_matrix.get((n_depots + last_node, trip.get("end_depot", 0)), float('inf')) / 60.0
            t += travel_duration
            total_travel_time_min += travel_duration # Accumulate travel time
            
            evacuee_return_time = t
            all_return_times.append(evacuee_return_time)
            
            # Add people to the destination depot's load
            end_depot_idx = trip.get("end_depot", 0)
            if depots and end_depot_idx < len(depot_loads):
                depot_loads[end_depot_idx] += total_people_in_trip
            
            if total_people_in_trip > 0:
                # --- UPDATED OFFLOADING LOGIC ---
                if use_dynamic_service_time:
                    offload_service_time = service_time_base_min + total_people_in_trip * service_time_per_person_min
                else:
                     # UPDATED: Static now uses bus_capacity for offloading too
                    offload_service_time = service_time_base_min + bus_capacity * service_time_per_person_min
                t += offload_service_time
            
            current_time = t
        
        bus_finish_times[b_idx] = current_time # Store finish time for this bus

    # Calculate total overfill after all trips are simulated
    if depots:
        for i, load in enumerate(depot_loads):
            capacity = depots[i].get('capacity')
            if capacity is not None and load > capacity:
                total_overfill += (load - capacity)

    return {
        "total_wait_pm": total_wait_pm,
        "total_lateness_pm": total_lateness_pm,
        "latest_evac_min": max(all_return_times) if all_return_times else 0.0,
        "pickup_times": all_pickup_times,
        "return_times": all_return_times,
        "total_people_evacuated": total_people_evacuated,
        "late_pickup_count": late_pickup_count,
        "total_travel_time_min": total_travel_time_min,
        "bus_finish_times": bus_finish_times,
        "total_overfill": total_overfill,
    }


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def _arc_minutes(durations_matrix: Dict[Tuple[int, int], float], a: int, b: int) -> float:
    sec = durations_matrix.get((a, b), float('inf'))
    if not math.isfinite(sec):
        raise ValueError(f"Non-finite travel time for arc {a}->{b}")
    return sec / 60.0

def _weighted_quantiles(pairs: List[Tuple[float, int]], qs: List[float]) -> List[float]:
    if not pairs:
        return [None] * len(qs)
    pairs_sorted = sorted((v, int(w)) for v, w in pairs if w > 0)
    if not pairs_sorted:
        return [None] * len(qs)
    total_w = sum(w for _, w in pairs_sorted)
    out = []
    for q in qs:
        target = q * total_w
        cum = 0
        val = pairs_sorted[-1][0]
        for v, w in pairs_sorted:
            cum += w
            if cum >= target:
                val = v
                break
        out.append(val)
    return out

def _weighted_mean(pairs: List[Tuple[float, int]]) -> float:
    if not pairs:
        return None
    num = sum(v * w for v, w in pairs)
    den = sum(w for _, w in pairs)
    return num / den if den > 0 else None

def _weighted_gini(pairs: List[Tuple[float, int]]) -> float:
    pairs = [(float(x), int(w)) for x, w in pairs if w > 0]
    if not pairs:
        return None
    pairs.sort(key=lambda t: t[0])
    W = sum(w for _, w in pairs)
    tot = sum(x * w for x, w in pairs)
    if tot == 0 or W == 0:
        return 0.0
    cum_income = 0.0
    sum_term = 0.0
    for x, w in pairs:
        L_prev = cum_income / tot
        cum_income += x * w
        L = cum_income / tot
        p = w / W
        sum_term += p * (L_prev + L)
    return max(0.0, min(1.0, 1.0 - sum_term))

def objective_from_metrics(metrics: Dict[str, Any],
                           lateness_weight: float = 1.0,
                           latest_equiv_people: float = 0.0) -> float:
    comp = metrics["objective_components"]
    return (comp["total_wait_person_minutes"]
            + latest_equiv_people * comp["latest_return_minutes"]
            + comp.get("total_overfill", 0.0))

def compute_solution_metrics(
    solution: List[List[Dict[str, Any]]],
    buses_count: int,
    n_depots: int,
    durations_matrix: Dict[Tuple[int, int], float],
    demand_full: Dict[int, int],
    deadlines: Dict[int, float],
    depots: List[Dict[str, Any]],
    service_time_min: float = SERVICE_TIME_MIN_DEFAULT,
    vehicles: Optional[List[Dict[str, Any]]] = None,
    node_coords: Optional[Dict[int, Tuple[float, float]]] = None,
    avg_speed_kmh: float = 30.0,
    road_factor: float = 1.25,
    start_to_node_seconds: Optional[Dict[int, Dict[int, float]]] = None,
    use_dynamic_service_time: bool = False,
    service_time_base_min: float = 3.0,
    service_time_per_person_min: float = 20.0 / 60.0,
) -> Dict[str, Any]:
    individual: List[List[Dict[str, Any]]] = [[] for _ in range(buses_count)]
    for b_idx, bus_trips in enumerate(solution):
        if b_idx >= buses_count: continue
        for trip in bus_trips:
            stops_seq = trip.get("stops", [])
            counts_map = trip.get("pickup_counts", {})
            stops = [(node, int(counts_map.get(node, 0))) for node in stops_seq if int(counts_map.get(node, 0)) > 0]
            if not stops: continue
            individual[b_idx].append({
                "start_depot": int(trip.get("start_depot", 0)),
                "stops": stops,
                "end_depot": int(trip.get("end_depot", 0)),
            })
    
    origin_by_bus = []
    cap_by_bus = []
    if vehicles:
        for v in vehicles:
            origin_by_bus.append(v.get("start", {"kind": "depot", "index": 0}))
            cap_by_bus.append(v.get("capacity", 80))
    else:
        origin_by_bus = [{"kind": "depot", "index": 0} for _ in range(buses_count)]
        cap_by_bus = [80 for _ in range(buses_count)]

    sim_results = _simulate_and_get_timings(
        individual=individual,
        n_depots=n_depots,
        durations_matrix=durations_matrix,
        deadlines=deadlines,
        origin_by_bus=origin_by_bus,
        cap_by_bus=cap_by_bus,
        depots=depots,
        node_coords=node_coords,
        start_to_node_seconds=start_to_node_seconds,
        avg_speed_kmh=avg_speed_kmh,
        road_factor=road_factor,
        use_dynamic_service_time=use_dynamic_service_time,
        service_time_base_min=service_time_base_min,
        service_time_per_person_min=service_time_per_person_min,
    )

    total_wait_pm = sim_results["total_wait_pm"]
    total_lateness_pm = sim_results["total_lateness_pm"]
    latest_return_min = sim_results["latest_evac_min"]
    total_overfill = sim_results["total_overfill"]
    
    people_picked = sim_results["total_people_evacuated"]
    people_total_required = sum(int(v) for v in demand_full.values())
    unserved = max(0, people_total_required - people_picked)

    wait_times = sim_results["pickup_times"]
    wait_pairs = [(t, 1) for t in wait_times]
    
    mean_wait_min = _weighted_mean(wait_pairs)
    p50, p90, p95, p99 = _weighted_quantiles(wait_pairs, [0.5, 0.9, 0.95, 0.99])
    gini_wait = _weighted_gini(wait_pairs)

    bus_finish_times = sim_results["bus_finish_times"]
    utilization_rates = [(t / latest_return_min) * 100 for t in bus_finish_times] if latest_return_min > 0 else [0] * len(bus_finish_times)

    metrics = {
        "units": {"time": "minutes", "objective_terms": "person-minutes"},
        "counts": {
            "people_required": people_total_required,
            "people_picked": people_picked,
            "people_unserved": unserved,
            "late_people": sim_results["late_pickup_count"],
            "late_fraction": sim_results["late_pickup_count"] / people_picked if people_picked > 0 else 0,
            "stops_visited": sum(len(trip.get("stops", [])) for bus in individual for trip in bus),
        },
        "wait": {
            "sum_person_minutes": total_wait_pm, "mean_min": mean_wait_min,
            "p50_min": p50, "p90_min": p90, "p95_min": p95, "p99_min": p99,
            "max_min": max(wait_times) if wait_times else None,
            "gini": gini_wait,
        },
        "lateness": {"sum_person_minutes": total_lateness_pm},
        "timeline": {"latest_return_min": latest_return_min},
        "efficiency": {
            "total_travel_time_min": sim_results["total_travel_time_min"],
            "avg_utilization_percent": np.mean(utilization_rates) if utilization_rates else 0,
            "std_utilization_percent": np.std(utilization_rates) if utilization_rates else 0,
        },
        "objective_components": {
            "total_wait_person_minutes": total_wait_pm,
            "latest_return_minutes": latest_return_min,
            "total_overfill": total_overfill,
        },
    }
    return metrics
