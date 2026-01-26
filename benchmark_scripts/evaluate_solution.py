from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "app" / "backend"
sys.path.append(str(BACKEND_DIR))

from app.evacuation.metrics import compute_solution_metrics, _simulate_and_get_timings

try:
    from app.evacuation.scenarios import ALL_SCENARIOS
except Exception:
    ALL_SCENARIOS = {}


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _matrix_key_from_str(key: str) -> Tuple[int, int]:
    raw = str(key).strip()
    if raw.startswith("(") and raw.endswith(")"):
        raw = raw[1:-1]
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) < 2:
        raise ValueError(f"Invalid matrix key: {key}")
    return int(parts[0]), int(parts[1])


def _deserialize_locations(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for item in items:
        obj = dict(item)
        coords = obj.get("coords")
        if coords is not None:
            obj["coords"] = (coords[0], coords[1])
        out.append(obj)
    return out


def _deserialize_problem_data(data: Dict[str, Any]) -> Dict[str, Any]:
    problem = dict(data)
    problem["depots"] = _deserialize_locations(data.get("depots", []))
    problem["facilities"] = _deserialize_locations(data.get("facilities", []))
    problem["durations_matrix"] = {
        _matrix_key_from_str(k): float(v) for k, v in data.get("durations_matrix", {}).items()
    }
    problem["pickup_nodes"] = [int(n) for n in data.get("pickup_nodes", [])]
    problem["demand_full"] = {int(k): int(v) for k, v in data.get("demand_full", {}).items()}
    problem["deadlines"] = {int(k): float(v) for k, v in data.get("deadlines", {}).items()}
    problem["node_coords"] = {
        int(k): (float(v[0]), float(v[1])) for k, v in data.get("node_coords", {}).items()
    }
    return problem


def _deserialize_first_leg(data: Dict[str, Any]) -> Dict[int, Dict[int, float]]:
    return {int(b): {int(n): float(v) for n, v in nodes.items()} for b, nodes in data.items()}


def _load_problem(problem_path: Path, matrix_path: Optional[Path]) -> Dict[str, Any]:
    data = _read_json(problem_path)
    if not data.get("durations_matrix"):
        if not matrix_path:
            raise ValueError("Problem file missing durations_matrix and no --matrix provided.")
        data = dict(data)
        data["durations_matrix"] = _read_json(matrix_path)
    return _deserialize_problem_data(data)


def _ensure_node_coords(problem: Dict[str, Any]) -> Dict[int, Tuple[float, float]]:
    node_coords = dict(problem.get("node_coords") or {})
    if node_coords:
        problem["node_coords"] = node_coords
        return node_coords
    facilities = problem.get("facilities", [])
    for idx, fac in enumerate(facilities):
        coords = fac.get("coords")
        if coords and len(coords) == 2:
            lon, lat = coords
            node_coords[int(idx)] = (float(lat), float(lon))
    problem["node_coords"] = node_coords
    return node_coords


def _resolve_scenario(name: str) -> Tuple[str, Dict[str, Any]]:
    if not ALL_SCENARIOS:
        raise ValueError("Scenario registry not available. Use --problem instead.")
    if name in ALL_SCENARIOS:
        return name, ALL_SCENARIOS[name]
    lowered = name.lower()
    for key, scen in ALL_SCENARIOS.items():
        if lowered == str(scen.get("name", "")).lower():
            return key, scen
    raise ValueError(f"Unknown scenario '{name}'. Available: {', '.join(ALL_SCENARIOS.keys())}")


def _normalize_start_coord(coord: Any) -> Dict[str, Any]:
    if isinstance(coord, dict):
        lat = float(coord["lat"])
        lon = float(coord["lon"])
        return {"kind": "coord", "lat": lat, "lon": lon}
    if isinstance(coord, (list, tuple)) and len(coord) == 2:
        lon, lat = coord
        return {"kind": "coord", "lat": float(lat), "lon": float(lon)}
    raise ValueError("start_coord must be a dict with lat/lon or a [lon, lat] list.")


def _build_vehicles_from_scenario(scenario: Dict[str, Any], fleet_name: str) -> List[Dict[str, Any]]:
    fleets = scenario.get("fleets", {})
    if fleet_name not in fleets:
        raise ValueError(f"Fleet '{fleet_name}' not found in scenario '{scenario.get('name')}'.")
    start_point = scenario.get("vehicle_start_point")
    default_start = {"kind": "depot", "index": 0}
    if start_point:
        default_start = _normalize_start_coord(start_point)

    vehicles = []
    for spec in fleets[fleet_name]:
        count = int(spec.get("count", 0))
        cap = int(spec.get("capacity", 0))
        if count <= 0 or cap <= 0:
            continue
        start = default_start
        if spec.get("start_coord") is not None:
            start = _normalize_start_coord(spec["start_coord"])
        vehicles.extend([{"capacity": cap, "start": start} for _ in range(count)])
    return vehicles


def _normalize_vehicles_payload(raw: List[Dict[str, Any]], default_start: Dict[str, Any]) -> List[Dict[str, Any]]:
    vehicles = []
    for v in raw:
        if v is None:
            continue
        cap = int(v.get("capacity", 0))
        if cap <= 0:
            raise ValueError("Vehicle capacity must be > 0.")
        if isinstance(v.get("start"), dict):
            start = v["start"]
        elif v.get("start_depot") is not None:
            start = {"kind": "depot", "index": int(v["start_depot"])}
        elif v.get("start_node") is not None:
            start = {"kind": "node", "index": int(v["start_node"])}
        elif v.get("start_coord") is not None:
            start = _normalize_start_coord(v["start_coord"])
        else:
            start = default_start
        vehicles.append({"capacity": cap, "start": start})
    return vehicles


def _normalize_solution(raw_solution: Any) -> Tuple[List[List[Dict[str, Any]]], List[str]]:
    if not isinstance(raw_solution, list):
        raise ValueError("Solution must be a list of bus schedules.")
    warnings = []
    solution = []
    for b_idx, bus_trips in enumerate(raw_solution):
        if not isinstance(bus_trips, list):
            raise ValueError(f"Bus {b_idx} schedule is not a list.")
        bus_out = []
        for t_idx, trip in enumerate(bus_trips):
            if not isinstance(trip, dict):
                raise ValueError(f"Bus {b_idx} trip {t_idx} is not an object.")
            start_depot = int(trip.get("start_depot", 0))
            end_depot = int(trip.get("end_depot", 0))
            raw_counts = trip.get("pickup_counts", {})
            counts_map: Dict[int, int] = {}
            if isinstance(raw_counts, dict):
                for k, v in raw_counts.items():
                    try:
                        counts_map[int(k)] = int(v)
                    except Exception:
                        warnings.append(f"Bus {b_idx} trip {t_idx}: invalid pickup_counts entry {k}:{v}.")
            elif raw_counts:
                warnings.append(f"Bus {b_idx} trip {t_idx}: pickup_counts is not a dict.")

            raw_stops = trip.get("stops", [])
            stops_seq: List[int] = []
            if raw_stops:
                first = raw_stops[0]
                if isinstance(first, (list, tuple)) and len(first) == 2:
                    for node, count in raw_stops:
                        n = int(node)
                        c = int(count)
                        counts_map[n] = c
                        stops_seq.append(n)
                elif isinstance(first, dict):
                    for item in raw_stops:
                        n = int(item.get("node", item.get("id", item.get("stop", 0))))
                        c = int(item.get("count", item.get("pickup", item.get("qty", 0))))
                        counts_map[n] = c
                        stops_seq.append(n)
                else:
                    for node in raw_stops:
                        stops_seq.append(int(node))

            for node in stops_seq:
                if node not in counts_map:
                    warnings.append(
                        f"Bus {b_idx} trip {t_idx}: missing pickup_counts for node {node}; defaulting to 0."
                    )
                    counts_map[node] = 0

            extra_nodes = [n for n in counts_map if n not in stops_seq]
            if extra_nodes:
                warnings.append(
                    f"Bus {b_idx} trip {t_idx}: pickup_counts has nodes not in stops list: {extra_nodes}."
                )

            bus_out.append({
                "start_depot": start_depot,
                "end_depot": end_depot,
                "stops": stops_seq,
                "pickup_counts": counts_map,
            })
        solution.append(bus_out)
    return solution, warnings


def _build_individual(solution: List[List[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
    individual = []
    for bus_trips in solution:
        schedule = []
        for trip in bus_trips:
            stops = []
            for node in trip.get("stops", []):
                count = int(trip.get("pickup_counts", {}).get(node, 0))
                if count > 0:
                    stops.append((node, count))
            schedule.append({
                "start_depot": int(trip.get("start_depot", 0)),
                "end_depot": int(trip.get("end_depot", 0)),
                "stops": stops,
            })
        individual.append(schedule)
    return individual


def _has_arc(durations_matrix: Dict[Tuple[int, int], float], a: int, b: int) -> bool:
    val = durations_matrix.get((a, b))
    return val is not None and math.isfinite(val)


def _has_coord_first_leg(
    bus_idx: int,
    node: int,
    start_to_node_seconds: Dict[int, Dict[int, float]],
    node_coords: Dict[int, Tuple[float, float]],
) -> bool:
    if start_to_node_seconds and bus_idx in start_to_node_seconds:
        val = start_to_node_seconds[bus_idx].get(node)
        if val is not None and math.isfinite(val):
            return True
    return node in node_coords


def _check_constraints(
    solution: List[List[Dict[str, Any]]],
    demand_full: Dict[int, int],
    n_depots: int,
    durations_matrix: Dict[Tuple[int, int], float],
    cap_by_bus: List[int],
    origin_by_bus: List[Dict[str, Any]],
    start_to_node_seconds: Dict[int, Dict[int, float]],
    node_coords: Dict[int, Tuple[float, float]],
    depots: List[Dict[str, Any]],
) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    totals = {node: 0 for node in demand_full.keys()}
    demand_nodes = set(demand_full.keys())
    depot_loads = [0] * n_depots

    for b_idx, bus_trips in enumerate(solution):
        bus_cap = cap_by_bus[b_idx] if b_idx < len(cap_by_bus) else None
        if bus_cap is None:
            errors.append(f"Bus {b_idx}: missing capacity definition.")
            bus_cap = 0
        origin = origin_by_bus[b_idx] if b_idx < len(origin_by_bus) else {"kind": "depot", "index": 0}

        if bus_trips and origin.get("kind") == "depot":
            expected = int(origin.get("index", 0))
            if int(bus_trips[0].get("start_depot", 0)) != expected:
                errors.append(f"Bus {b_idx}: first trip start_depot != origin depot ({expected}).")

        prev_end = None
        for t_idx, trip in enumerate(bus_trips):
            start_depot = int(trip.get("start_depot", 0))
            end_depot = int(trip.get("end_depot", 0))
            if not (0 <= start_depot < n_depots):
                errors.append(f"Bus {b_idx} trip {t_idx}: start_depot out of range ({start_depot}).")
            if not (0 <= end_depot < n_depots):
                errors.append(f"Bus {b_idx} trip {t_idx}: end_depot out of range ({end_depot}).")
            if prev_end is not None and start_depot != prev_end:
                errors.append(f"Bus {b_idx} trip {t_idx}: start_depot does not match previous end_depot.")
            prev_end = end_depot

            stops_seq = trip.get("stops", [])
            counts_map = trip.get("pickup_counts", {})
            trip_load = 0
            seen_nodes = set()

            for node in stops_seq:
                seen_nodes.add(node)
                if node not in demand_nodes:
                    errors.append(f"Bus {b_idx} trip {t_idx}: unknown node {node}.")
                    continue
                count = int(counts_map.get(node, 0))
                if count < 0:
                    errors.append(f"Bus {b_idx} trip {t_idx}: negative pickup count at node {node}.")
                trip_load += count
                totals[node] += count

            extra_nodes = [n for n in counts_map if n not in seen_nodes]
            if extra_nodes:
                errors.append(
                    f"Bus {b_idx} trip {t_idx}: pickup_counts has nodes not in stops list: {extra_nodes}."
                )

            if bus_cap is not None and trip_load > bus_cap:
                errors.append(
                    f"Bus {b_idx} trip {t_idx}: capacity exceeded (load {trip_load} > {bus_cap})."
                )

            if 0 <= end_depot < n_depots:
                depot_loads[end_depot] += trip_load

            if stops_seq:
                first_node = int(stops_seq[0])
                if t_idx == 0:
                    okind = origin.get("kind", "depot")
                    if okind == "depot":
                        sd = int(origin.get("index", 0))
                        if not _has_arc(durations_matrix, sd, n_depots + first_node):
                            errors.append(f"Bus {b_idx} trip 0: no arc depot {sd} -> node {first_node}.")
                    elif okind == "node":
                        sn = int(origin.get("index", 0))
                        if sn != first_node and not _has_arc(
                            durations_matrix, n_depots + sn, n_depots + first_node
                        ):
                            errors.append(
                                f"Bus {b_idx} trip 0: no arc node {sn} -> node {first_node}."
                            )
                    elif okind == "coord":
                        if not _has_coord_first_leg(
                            b_idx, first_node, start_to_node_seconds, node_coords
                        ):
                            errors.append(
                                f"Bus {b_idx} trip 0: missing first-leg path to node {first_node}."
                            )
                else:
                    if not _has_arc(durations_matrix, start_depot, n_depots + first_node):
                        errors.append(
                            f"Bus {b_idx} trip {t_idx}: no arc depot {start_depot} -> node {first_node}."
                        )

                for a, bnode in zip(stops_seq, stops_seq[1:]):
                    if not _has_arc(durations_matrix, n_depots + a, n_depots + bnode):
                        errors.append(
                            f"Bus {b_idx} trip {t_idx}: no arc node {a} -> node {bnode}."
                        )

                last_node = int(stops_seq[-1])
                if not _has_arc(durations_matrix, n_depots + last_node, end_depot):
                    errors.append(
                        f"Bus {b_idx} trip {t_idx}: no arc node {last_node} -> depot {end_depot}."
                    )

    demand_mismatches = []
    for node, need in demand_full.items():
        got = totals.get(node, 0)
        if got != need:
            demand_mismatches.append({"node": node, "required": need, "served": got})
    if demand_mismatches:
        errors.append("Demand mismatch detected (see demand_mismatches).")

    depot_caps = [d.get("capacity") for d in depots] if depots else []
    depot_overfill = []
    for idx, load in enumerate(depot_loads):
        cap = depot_caps[idx] if idx < len(depot_caps) else None
        if cap is not None and load > cap:
            depot_overfill.append({
                "depot": idx,
                "capacity": int(cap),
                "load": int(load),
                "overfill": int(load - cap),
            })
    if depot_overfill:
        warnings.append("Depot capacity exceeded (see depot_overfill).")

    return {
        "errors": errors,
        "warnings": warnings,
        "demand_mismatches": demand_mismatches,
        "depot_loads": depot_loads,
        "depot_overfill": depot_overfill,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate an evacuation solution against benchmark data.")
    parser.add_argument("--solution", required=True, help="Path to solution JSON.")
    parser.add_argument("--scenario", help="Scenario key or name (e.g., bomb, BombThreat).")
    parser.add_argument("--fleet", help="Fleet name (e.g., specialized_only, augmented, default).")
    parser.add_argument("--problem", help="Path to problem JSON (overrides --scenario/--fleet).")
    parser.add_argument("--matrix", help="Path to durations matrix JSON if problem lacks durations_matrix.")
    parser.add_argument("--first-leg", help="Path to first-leg JSON (origins -> nodes).")
    parser.add_argument("--vehicles", help="Path to vehicles JSON (list).")
    parser.add_argument("--bus-capacity", type=int, default=80, help="Default capacity for homogeneous fleets.")
    parser.add_argument("--avg-speed-kmh", type=float, default=30.0, help="Average speed for coord starts.")
    parser.add_argument("--road-factor", type=float, default=1.25, help="Road factor for coord starts.")
    parser.add_argument("--static-service-time", action="store_true", help="Use static service time model.")
    parser.add_argument("--service-time-base-min", type=float, default=3.0)
    parser.add_argument("--service-time-per-person-min", type=float, default=20.0 / 60.0)
    parser.add_argument("--output", help="Optional JSON output path.")
    args = parser.parse_args()

    solution_path = Path(args.solution)
    solution_data = _read_json(solution_path)
    reported_cost = None
    raw_solution = solution_data
    if isinstance(solution_data, dict):
        if "best_solution" in solution_data:
            raw_solution = solution_data["best_solution"]
        elif "solution" in solution_data:
            raw_solution = solution_data["solution"]
        if "overall_cost" in solution_data:
            try:
                reported_cost = float(solution_data["overall_cost"])
            except Exception:
                reported_cost = None

    normalized_solution, solution_warnings = _normalize_solution(raw_solution)

    scenario_name = None
    fleet_name = None
    problem_path = Path(args.problem) if args.problem else None
    matrix_path = Path(args.matrix) if args.matrix else None
    first_leg_path = Path(args.first_leg) if args.first_leg else None

    vehicles = None
    default_start = {"kind": "depot", "index": 0}

    if problem_path is None:
        if not args.scenario:
            raise ValueError("Provide --problem or --scenario/--fleet.")
        scen_key, scen = _resolve_scenario(args.scenario)
        scenario_name = scen.get("name", scen_key)
        fleet_name = args.fleet
        if fleet_name is None:
            if len(scen.get("fleets", {})) == 1:
                fleet_name = next(iter(scen["fleets"].keys()))
            else:
                raise ValueError("Multiple fleets available. Provide --fleet.")
        vehicles = _build_vehicles_from_scenario(scen, fleet_name)
        start_point = scen.get("vehicle_start_point")
        if start_point:
            default_start = _normalize_start_coord(start_point)
        problem_path = (
            REPO_ROOT / "benchmark_data" / "precomputed_matrices" / "matrices"
            / f"{scenario_name}_{fleet_name}_problem.json"
        )
        if first_leg_path is None:
            candidate = (
                REPO_ROOT / "benchmark_data" / "precomputed_matrices" / "matrices"
                / f"{scenario_name}_{fleet_name}_first_leg.json"
            )
            if candidate.exists():
                first_leg_path = candidate

    if not problem_path.exists():
        raise FileNotFoundError(f"Problem file not found: {problem_path}")

    problem = _load_problem(problem_path, matrix_path)
    _ensure_node_coords(problem)

    if args.vehicles:
        vehicles_raw = _read_json(Path(args.vehicles))
        if not isinstance(vehicles_raw, list):
            raise ValueError("--vehicles must point to a JSON list.")
        vehicles = _normalize_vehicles_payload(vehicles_raw, default_start)

    buses_count = len(vehicles) if vehicles else len(normalized_solution)
    if vehicles and len(normalized_solution) != len(vehicles):
        solution_warnings.append(
            f"Solution bus count ({len(normalized_solution)}) != vehicles count ({len(vehicles)})."
        )

    if len(normalized_solution) > buses_count:
        solution_warnings.append(
            f"Solution has {len(normalized_solution)} buses; truncating to {buses_count}."
        )
        normalized_solution = normalized_solution[:buses_count]

    if len(normalized_solution) < buses_count:
        normalized_solution.extend([[] for _ in range(buses_count - len(normalized_solution))])

    if vehicles is None:
        vehicles = [{"capacity": int(args.bus_capacity), "start": default_start} for _ in range(buses_count)]

    cap_by_bus = [int(v.get("capacity", args.bus_capacity)) for v in vehicles]
    origin_by_bus = [v.get("start", {"kind": "depot", "index": 0}) for v in vehicles]

    start_to_node_seconds = {}
    if first_leg_path and Path(first_leg_path).exists():
        start_to_node_seconds = _deserialize_first_leg(_read_json(Path(first_leg_path)))

    n_depots = int(problem.get("n_depots", len(problem.get("depots", []))))
    constraints = _check_constraints(
        solution=normalized_solution,
        demand_full=problem.get("demand_full", {}),
        n_depots=n_depots,
        durations_matrix=problem.get("durations_matrix", {}),
        cap_by_bus=cap_by_bus,
        origin_by_bus=origin_by_bus,
        start_to_node_seconds=start_to_node_seconds,
        node_coords=problem.get("node_coords", {}),
        depots=problem.get("depots", []),
    )

    individual = _build_individual(normalized_solution)

    use_dynamic_service_time = not args.static_service_time

    sim_results = _simulate_and_get_timings(
        individual=individual,
        n_depots=n_depots,
        durations_matrix=problem.get("durations_matrix", {}),
        deadlines=problem.get("deadlines", {}),
        origin_by_bus=origin_by_bus,
        cap_by_bus=cap_by_bus,
        depots=problem.get("depots", []),
        node_coords=problem.get("node_coords", {}),
        start_to_node_seconds=start_to_node_seconds,
        avg_speed_kmh=float(args.avg_speed_kmh),
        road_factor=float(args.road_factor),
        use_dynamic_service_time=use_dynamic_service_time,
        service_time_base_min=float(args.service_time_base_min),
        service_time_per_person_min=float(args.service_time_per_person_min),
    )

    total_people = max(1, int(sim_results.get("total_people_evacuated", 0)))
    avg_wait = float(sim_results.get("total_wait_pm", 0.0)) / total_people
    makespan = float(sim_results.get("latest_evac_min", 0.0))
    total_overfill = float(sim_results.get("total_overfill", 0.0))
    computed_cost = avg_wait + makespan + total_overfill

    metrics = compute_solution_metrics(
        solution=normalized_solution,
        buses_count=buses_count,
        n_depots=n_depots,
        durations_matrix=problem.get("durations_matrix", {}),
        demand_full=problem.get("demand_full", {}),
        deadlines=problem.get("deadlines", {}),
        depots=problem.get("depots", []),
        vehicles=vehicles,
        node_coords=problem.get("node_coords", {}),
        start_to_node_seconds=start_to_node_seconds,
        avg_speed_kmh=float(args.avg_speed_kmh),
        road_factor=float(args.road_factor),
        use_dynamic_service_time=use_dynamic_service_time,
        service_time_base_min=float(args.service_time_base_min),
        service_time_per_person_min=float(args.service_time_per_person_min),
    )

    result = {
        "scenario": scenario_name,
        "fleet": fleet_name,
        "problem_file": str(problem_path),
        "solution_file": str(solution_path),
        "feasible": len(constraints["errors"]) == 0,
        "errors": constraints["errors"],
        "warnings": constraints["warnings"] + solution_warnings,
        "demand_mismatches": constraints["demand_mismatches"],
        "depot_loads": constraints["depot_loads"],
        "depot_overfill": constraints["depot_overfill"],
        "objective": {
            "avg_wait_min": avg_wait,
            "makespan_min": makespan,
            "total_overfill": total_overfill,
            "overall_cost": computed_cost,
        },
        "reported_cost": reported_cost,
        "cost_delta": None if reported_cost is None else computed_cost - reported_cost,
        "metrics": metrics,
    }

    output = json.dumps(result, indent=2, ensure_ascii=True)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
