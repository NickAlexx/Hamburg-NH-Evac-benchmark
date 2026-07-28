import os
import sys
import json
import time
import argparse
import math
import random
from datetime import datetime
from pathlib import Path

# Make repository-qualified imports work both as a script and as a module.
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
import numpy as np

# Import algorithms and scenarios
from app.backend.app.evacuation.ea import (
    RevisionaryEvolutionaryAlgorithm,
    run_evolutionary_algorithm,
)
from app.backend.app.evacuation.alns_algorithm import run_alns_algorithm
from app.backend.app.evacuation.baselines.pendelverkehr import (
    PendelverkehrShuttleAlgorithm,
)
from app.backend.app.evacuation.runtime_budget import (
    LEGACY_RESULTS_BUDGET_MODE,
    STRICT_BUDGET_MODE,
)
from app.backend.app.evacuation.scenarios import ALL_SCENARIOS
from openrouteservice import Client as ORSClient
from app.backend.app.config import ORS_KEY

# --- EXPERIMENT CONFIGURATION ---
NUM_RUNS = 30
TIME_LIMIT_SECONDS = 300
SEED_BASE = 1
USE_CACHED_INPUTS = True
CACHE_VERSION = 1
COLLECT_OPERATOR_TELEMETRY = False
BENCHMARK_DATA_DIR = repo_root / "benchmark_data"
PRECOMPUTED_MATRICES_DIR = BENCHMARK_DATA_DIR / "precomputed_matrices" / "matrices"
FORCE_PRECOMPUTED_MATRICES = True

SHARED_EXECUTION_PARAMS = {
    "use_dynamic_service_time": True,
    "service_time_base_min": 3.0,
    "service_time_per_person_min": 20.0 / 60.0,
}

ALGORITHMS_TO_RUN = {
    "MA": {
        "runner": run_evolutionary_algorithm,
        "params": {
            "use_local_search": True,
            "population_size": 200,
            "generations": 10000, #irrelevant due to 5 min budet
            "crossover_rate": 0.8,
            "mutation_rate": 0.2,
            "collect_operator_telemetry": COLLECT_OPERATOR_TELEMETRY,
        },
    },
    "GA": {
        "runner": run_evolutionary_algorithm,
        "params": {
            "use_local_search": False,
            "population_size": 200,
            "generations": 10000,
            "crossover_rate": 0.8,
            "mutation_rate": 0.2,
            "collect_operator_telemetry": COLLECT_OPERATOR_TELEMETRY
        }
    },
    "ALNS": {
        "runner": run_alns_algorithm,
        "params": {
            "alns_config": {"use_memetic_polish": False}
        }
    },
    "Dispatcher": {
        "runner": PendelverkehrShuttleAlgorithm().run,
        "params": {"pick_rule": "nearest"},
    },
}

def _matrix_key_to_str(key):
    return f"{int(key[0])},{int(key[1])}"

def _matrix_key_from_str(key):
    raw = str(key).strip()
    if raw.startswith("(") and raw.endswith(")"):
        raw = raw[1:-1]
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) < 2:
        raise ValueError(f"Invalid matrix key: {key}")
    return int(parts[0]), int(parts[1])

def _serialize_locations(items):
    out = []
    for item in items:
        obj = dict(item)
        coords = obj.get("coords")
        if coords is not None:
            obj["coords"] = [float(coords[0]), float(coords[1])]
        out.append(obj)
    return out

def _deserialize_locations(items):
    out = []
    for item in items:
        obj = dict(item)
        coords = obj.get("coords")
        if coords is not None:
            obj["coords"] = (coords[0], coords[1])
        out.append(obj)
    return out

def _serialize_problem_data(problem_data):
    data = dict(problem_data)
    data.pop("deadlines", None)  # Ignore obsolete fields in legacy in-memory inputs.
    data["_cache_version"] = CACHE_VERSION
    data["depots"] = _serialize_locations(problem_data.get("depots", []))
    data["facilities"] = _serialize_locations(problem_data.get("facilities", []))
    data["durations_matrix"] = {
        _matrix_key_to_str(k): v for k, v in problem_data.get("durations_matrix", {}).items()
    }
    data["pickup_nodes"] = [int(n) for n in problem_data.get("pickup_nodes", [])]
    data["demand_full"] = {str(k): v for k, v in problem_data.get("demand_full", {}).items()}
    data["node_coords"] = {
        str(k): [float(v[0]), float(v[1])] for k, v in problem_data.get("node_coords", {}).items()
    }
    return data

def _deserialize_problem_data(data):
    problem_data = dict(data)
    problem_data.pop("deadlines", None)  # Legacy cache files remain readable.
    problem_data["depots"] = _deserialize_locations(data.get("depots", []))
    problem_data["facilities"] = _deserialize_locations(data.get("facilities", []))
    problem_data["durations_matrix"] = {
        _matrix_key_from_str(k): v for k, v in data.get("durations_matrix", {}).items()
    }
    problem_data["pickup_nodes"] = [int(n) for n in data.get("pickup_nodes", [])]
    problem_data["demand_full"] = {int(k): v for k, v in data.get("demand_full", {}).items()}
    problem_data["node_coords"] = {
        int(k): (v[0], v[1]) for k, v in data.get("node_coords", {}).items()
    }
    return problem_data

def _serialize_first_leg(data):
    return {str(bus_idx): {str(n): v for n, v in nodes.items()} for bus_idx, nodes in data.items()}

def _deserialize_first_leg(data):
    return {int(bus_idx): {int(n): v for n, v in nodes.items()} for bus_idx, nodes in data.items()}

# --- ROBUST RETRY HELPER (NO FALLBACK) ---
def robust_api_call(func, *args, **kwargs):
    """
    Executes a function. If it raises an exception, it waits and retries 
    indefinitely with exponential backoff.
    """
    delay = 2  # Start with 2 seconds
    attempt = 1
    
    while True:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"\n       API Call Failed (Attempt {attempt}). Error: {str(e)}")
            print(f"       Waiting {delay} seconds before retrying...")
            time.sleep(delay)
            # Exponential backoff, capped at 5 minutes
            delay = min(delay * 2, 300) 
            attempt += 1

# --- LOGGING HELPER ---
class DualLogger(object):
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding='utf-8', buffering=1)
    def write(self, message):
        try:
            self.terminal.write(message)
        except UnicodeEncodeError:
            encoding = self.terminal.encoding or "ascii"
            safe = message.encode(encoding, errors="ignore").decode(encoding, errors="ignore")
            self.terminal.write(safe)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def extract_run_summary(result, scenario_name, fleet_name, algo_name, run_num, run_time, success=True, error=None, seed=None):
    opt_runtime = None
    total_runtime = None
    pre_runtime = None
    post_runtime = None
    wall_runtime = None
    if isinstance(result, dict):
        opt_runtime = result.get("optimization_runtime")
        total_runtime = result.get("total_runtime")
        pre_runtime = result.get("preprocessing_runtime")
        post_runtime = result.get("postprocessing_runtime")
        wall_runtime = result.get("wall_runtime")
        stats = result.get("algorithm_stats", {})
        if opt_runtime is None:
            opt_runtime = stats.get("optimization_runtime")
        if total_runtime is None:
            total_runtime = stats.get("total_runtime")
        if pre_runtime is None:
            pre_runtime = stats.get("preprocessing_runtime")
        if post_runtime is None:
            post_runtime = stats.get("postprocessing_runtime")
        if opt_runtime is None and total_runtime is not None:
            opt_runtime = total_runtime
    if opt_runtime is None:
        opt_runtime = run_time
    if not success:
        return {
            "scenario": scenario_name, "fleet": fleet_name, "algorithm": algo_name,
            "run": run_num, "success": False, "error": str(error), "runtime": opt_runtime,
            "optimization_runtime": opt_runtime,
            "total_runtime": total_runtime,
            "preprocessing_runtime": pre_runtime,
            "postprocessing_runtime": post_runtime,
            "wall_runtime": wall_runtime,
            "seed": seed
        }

    result_seed = result.get("seed", seed) if isinstance(result, dict) else seed
    metrics = result.get('metrics', {})
    alg_stats = result.get('algorithm_stats', {})
    
    # Calculate Total Stops (Nodes visited)
    best_solution = result.get('best_solution', [])
    total_trips = sum(len(bus) for bus in best_solution)
    
    # NEW: Count total stops across all trips
    total_stops = 0
    for bus_schedule in best_solution:
        for trip in bus_schedule:
            # Check if 'stops' is a list of tuples or just nodes (handle both formats)
            stops = trip.get('stops', [])
            total_stops += len(stops)

    op_stats = alg_stats.get('operator_scoreboard', {})

    summary = {
        "scenario": scenario_name, "fleet": fleet_name, "algorithm": algo_name,
        "run": run_num, "success": True, "runtime": opt_runtime,
        "optimization_runtime": opt_runtime,
        "total_runtime": total_runtime,
        "preprocessing_runtime": pre_runtime,
        "postprocessing_runtime": post_runtime,
        "wall_runtime": wall_runtime,
        "seed": result_seed,
        "cost_fitness": result.get('overall_cost', float('nan')),
        "avg_evac_time": metrics.get('wait', {}).get('mean_min', float('nan')),
        "p95_wait_time": metrics.get('wait', {}).get('p95_min', float('nan')),
        "makespan": metrics.get('timeline', {}).get('latest_return_min', float('nan')),
        "trip_count": total_trips,
        "stop_count": total_stops,  # <--- NEW FIELD
        "total_travel_time": metrics.get('efficiency', {}).get('total_travel_time_min', float('nan')),
        "top_operator": max(op_stats, key=op_stats.get) if op_stats else "N/A",
    }
    for op_k, op_v in op_stats.items(): summary[f"op_gain_{op_k}"] = op_v
    return summary

def save_summary_file(output_dir, all_results):
    try:
        with open(output_dir / "all_runs_summary.tmp", 'w') as f:
            json.dump(all_results, f, indent=2)
        os.replace(output_dir / "all_runs_summary.tmp", output_dir / "all_runs_summary.json")
    except Exception as e:
        print(f"     Warning: Failed to save summary file: {e}")

def _recorded_budget_mode(result):
    """Return the explicit mode, or legacy for pre-metadata result files."""
    stats = result.get("algorithm_stats", {}) if isinstance(result, dict) else {}
    return (
        result.get("budget_mode")
        or stats.get("budget_mode")
        or LEGACY_RESULTS_BUDGET_MODE
    )

def _require_matching_budget_mode(result, requested_mode, result_path):
    existing_mode = _recorded_budget_mode(result)
    if existing_mode != requested_mode:
        raise RuntimeError(
            f"{result_path} uses budget_mode={existing_mode!r}, "
            f"but this run requested {requested_mode!r}. "
            "Use a new output directory instead of mixing protocols."
        )

def build_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--budget-mode",
        choices=(STRICT_BUDGET_MODE, LEGACY_RESULTS_BUDGET_MODE),
        default=STRICT_BUDGET_MODE,
        help=(
            "strict includes solver initialization and returns the last fully "
            "evaluated incumbent before the deadline; legacy_results "
            "reproduces the non-preemptive search-loop protocol used by the "
            "stored published results"
        ),
    )
    parser.add_argument(
        "--postprocess-reserve-seconds",
        type=float,
        default=0.25,
        help="Time reserved for solver result construction in strict mode.",
    )
    return parser

# --- MAIN ---
def main():
    args = build_argument_parser().parse_args()

    if args.resume:
        output_base_dir = Path(args.resume)
        if not output_base_dir.exists(): sys.exit(f" Error: {output_base_dir} not found")
        print(f" Resuming: {output_base_dir}")
    else:
        output_base_dir = BENCHMARK_DATA_DIR / "solutions" / f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(output_base_dir, exist_ok=True)
        print(f" Starting: {output_base_dir}")

    if FORCE_PRECOMPUTED_MATRICES and not PRECOMPUTED_MATRICES_DIR.exists():
        sys.exit(f" Error: Precomputed matrices dir not found: {PRECOMPUTED_MATRICES_DIR}")

    sys.stdout = DualLogger(output_base_dir / "experiment_log.txt")
    sys.stderr = sys.stdout
    matrices_dir = output_base_dir / "matrices"
    os.makedirs(matrices_dir, exist_ok=True)
    source_matrices_dir = PRECOMPUTED_MATRICES_DIR if FORCE_PRECOMPUTED_MATRICES else matrices_dir
    saved_matrices = set()
    all_results_summary = []
    
    ors_client = None
    def _get_ors_client():
        nonlocal ors_client
        if ors_client is None:
            if not ORS_KEY:
                raise ValueError("ORS_KEY not set but an ORS call is required.")
            ors_client = ORSClient(key=ORS_KEY)
        return ors_client

    data_loader = RevisionaryEvolutionaryAlgorithm()

    print(
        f"Runs: {NUM_RUNS} | Time Limit: {TIME_LIMIT_SECONDS}s "
        f"| Budget Mode: {args.budget_mode}"
    )

    for s_idx, (s_key, s_conf) in enumerate(ALL_SCENARIOS.items()):
        print(f"\n\n---  SCENARIO: {s_conf['name']} ---")
        #if s_conf['name'] != "Default" and s_conf['name'] != "FloodWilhelmsburg":
        #    continue
        #print(ALL_SCENARIOS.items())
        for f_idx, (f_name, f_spec) in enumerate(s_conf['fleets'].items()):
            print(f"\n  ---  FLEET: {f_name} ---")
            # 1. BUILD VEHICLE PAYLOAD
            vehicles_payload = []
            s_coord = s_conf.get("vehicle_start_point")
            for v_type in f_spec:
                for _ in range(v_type['count']):
                    v = {"capacity": v_type['capacity'], "id": None}
                    if "start_coord" in v_type:
                        v.update({"start_coord": v_type["start_coord"], "start_depot": None, "start_node": None})
                    elif s_coord:
                        v.update({"start_coord": {"lon": s_coord[0], "lat": s_coord[1]}, "start_depot": None, "start_node": None})
                    else:
                        v.update({"start_depot": 0, "start_coord": None, "start_node": None})
                    vehicles_payload.append(v)
            
            max_cap = max(v['capacity'] for v in vehicles_payload) if vehicles_payload else 80
            total_buses = len(vehicles_payload)

            safe_scen = s_conf['name'].replace(' ', '_')
            safe_fleet = f_name.replace(' ', '_')
            matrix_cache_path = matrices_dir / f"{safe_scen}_{safe_fleet}_matrix.json"
            problem_cache_path = matrices_dir / f"{safe_scen}_{safe_fleet}_problem.json"
            first_leg_cache_path = matrices_dir / f"{safe_scen}_{safe_fleet}_first_leg.json"
            source_problem_cache_path = source_matrices_dir / f"{safe_scen}_{safe_fleet}_problem.json"
            source_first_leg_cache_path = source_matrices_dir / f"{safe_scen}_{safe_fleet}_first_leg.json"

            # 2. LOAD STANDARD MATRIX FROM PRECOMPUTED CACHE (NO ORS CALLS)
            cached_data = None
            if FORCE_PRECOMPUTED_MATRICES:
                if not source_problem_cache_path.exists():
                    raise FileNotFoundError(f"Missing precomputed problem data: {source_problem_cache_path}")
                try:
                    with open(source_problem_cache_path, "r") as f:
                        cached_data = _deserialize_problem_data(json.load(f))
                    print(f"    Using precomputed problem data: {source_problem_cache_path.name}")
                except Exception as e:
                    raise RuntimeError(f"Failed to load precomputed problem data: {source_problem_cache_path}") from e
            elif USE_CACHED_INPUTS and problem_cache_path.exists():
                try:
                    with open(problem_cache_path, "r") as f:
                        cached_data = _deserialize_problem_data(json.load(f))
                    print(f"    Using cached problem data: {problem_cache_path.name}")
                except Exception as e:
                    print(f"    Warning: Failed to load cached problem data: {e}")
                    cached_data = None

            if cached_data is None:
                print(f"    Calculating Standard Matrix...")
                cached_data = robust_api_call(
                    data_loader.initialize_problem,
                    evacuation_zones_input=s_conf["evac_centers"],
                    buses_count=total_buses,
                    bus_capacity=max_cap,
                    default_evac_center_coords=s_conf["main_center"],
                    buffer_meters=s_conf["buffer_meters"]
                )
                if USE_CACHED_INPUTS:
                    try:
                        with open(problem_cache_path, "w") as f:
                            json.dump(_serialize_problem_data(cached_data), f, indent=2)
                    except Exception as e:
                        print(f"    Warning: Failed to save problem cache: {e}")
            
            # Save matrix once
            m_key = (s_conf['name'], f_name)
            if m_key not in saved_matrices:
                if not matrix_cache_path.exists():
                    with open(matrix_cache_path, "w") as f:
                        json.dump(
                            {_matrix_key_to_str(k): v for k, v in cached_data['durations_matrix'].items()},
                            f,
                            indent=2
                        )
                saved_matrices.add(m_key)

            # 3. LOAD FIRST-LEG DURATIONS (NO ORS CALLS WHEN FORCED)
            start_to_node_seconds = {}
            node_coords_map = cached_data.get('node_coords', {})
            vehicles_with_coords = [(i, v) for i, v in enumerate(vehicles_payload) if v.get('start_coord')]

            loaded_first_leg = False
            if FORCE_PRECOMPUTED_MATRICES and vehicles_with_coords:
                if not source_first_leg_cache_path.exists():
                    raise FileNotFoundError(f"Missing precomputed first-leg cache: {source_first_leg_cache_path}")
                try:
                    with open(source_first_leg_cache_path, "r") as f:
                        start_to_node_seconds = _deserialize_first_leg(json.load(f))
                    loaded_first_leg = True
                    print(f"    Using precomputed first-leg routes: {source_first_leg_cache_path.name}")
                except Exception as e:
                    raise RuntimeError(f"Failed to load precomputed first-leg cache: {source_first_leg_cache_path}") from e
            elif USE_CACHED_INPUTS and first_leg_cache_path.exists():
                try:
                    with open(first_leg_cache_path, "r") as f:
                        start_to_node_seconds = _deserialize_first_leg(json.load(f))
                    loaded_first_leg = True
                    print(f"    Using cached first-leg routes: {first_leg_cache_path.name}")
                except Exception as e:
                    print(f"    Warning: Failed to load first-leg cache: {e}")
                    loaded_first_leg = False
                    start_to_node_seconds = {}

            if loaded_first_leg and vehicles_with_coords and not start_to_node_seconds:
                if FORCE_PRECOMPUTED_MATRICES:
                    raise ValueError(f"Precomputed first-leg cache is empty: {source_first_leg_cache_path}")
                loaded_first_leg = False

            if not loaded_first_leg and vehicles_with_coords and node_coords_map:
                if FORCE_PRECOMPUTED_MATRICES:
                    raise FileNotFoundError(f"Missing precomputed first-leg cache: {source_first_leg_cache_path}")
                print(f"      Calculating First-Leg ORS Routes ({len(vehicles_with_coords)} vehicles)...")
                unique_starts = {}
                for i, v in vehicles_with_coords:
                    k = (float(v['start_coord']['lon']), float(v['start_coord']['lat']))
                    unique_starts.setdefault(k, []).append(i)
                
                sorted_node_ids = sorted(node_coords_map.keys())
                dest_coords = [[node_coords_map[nid][1], node_coords_map[nid][0]] for nid in sorted_node_ids]

                for (lon, lat), b_indices in unique_starts.items():
                    locations = [[lon, lat]] + dest_coords
                    # ROBUST CALL
                    ors_client = _get_ors_client()
                    res = robust_api_call(
                        ors_client.distance_matrix, locations=locations, sources=[0], 
                        destinations=list(range(1, len(locations))), metrics=["duration"]
                    )
                    durations = res['durations'][0]
                    time_map = {nid: d for nid, d in zip(sorted_node_ids, durations) if d is not None}
                    for i in b_indices: start_to_node_seconds[i] = time_map
                if USE_CACHED_INPUTS:
                    try:
                        with open(first_leg_cache_path, "w") as f:
                            json.dump(_serialize_first_leg(start_to_node_seconds), f, indent=2)
                    except Exception as e:
                        print(f"    Warning: Failed to save first-leg cache: {e}")

            # 4. RUN TRIALS
            for a_idx, (a_name, a_conf) in enumerate(ALGORITHMS_TO_RUN.items()):
                print(f"\n    ---  ALGORITHM: {a_name} ---")
                run_dir = output_base_dir / s_conf['name'] / f_name / a_name
                os.makedirs(run_dir, exist_ok=True)

                for run_num in range(1, NUM_RUNS + 1):
                    f_path = run_dir / f"run_{run_num}.json"
                    if f_path.exists():
                        try:
                            with open(f_path) as f:
                                existing = json.load(f)
                        except Exception:
                            existing = None

                        if existing is not None:
                            _require_matching_budget_mode(
                                existing,
                                args.budget_mode,
                                f_path,
                            )
                            all_results_summary.append(
                                extract_run_summary(
                                    existing, s_conf['name'], f_name, a_name, run_num,
                                    existing.get('runtime', 0), seed=existing.get("seed")
                                )
                            )
                            print(f"        Trial {run_num} SKIPPED")
                            continue

                    print(f"      ▶  Trial {run_num}/{NUM_RUNS}...", end="", flush=True)
                    seed = None
                    t0 = time.time()
                    try:
                        seed = (
                            SEED_BASE
                            + (s_idx * 1000000)
                            + (f_idx * 10000)
                            #+ (a_idx * 100)
                            + run_num
                        )
                        random.seed(seed)
                        np.random.seed(seed)
                        params = {
                            "evacuation_zones_input": s_conf["evac_centers"],
                            "default_evac_center_coords": s_conf["main_center"],
                            "buffer_meters": s_conf["buffer_meters"],
                            "buses_count": total_buses,
                            "bus_capacity": max_cap,
                            "vehicles": vehicles_payload,
                            "precomputed_problem_data": cached_data,     # <--- PASS DATA
                            "start_to_node_seconds": start_to_node_seconds, # <--- PASS DATA
                            "time_limit_seconds": TIME_LIMIT_SECONDS,
                            "budget_mode": args.budget_mode,
                            "postprocess_reserve_seconds": args.postprocess_reserve_seconds,
                            "output_dir": str(run_dir), 
                            **SHARED_EXECUTION_PARAMS,
                            **a_conf["params"]
                        }
                        if a_name.startswith("ALNS"):
                            params["seed"] = seed
                        result = a_conf["runner"](**params)
                        rt = time.time() - t0
                        stats = result.get("algorithm_stats", {})
                        if (
                            args.budget_mode == "strict"
                            and stats.get("budget_adhered") is not True
                        ):
                            overshoot = float(
                                stats.get("budget_overshoot_seconds", 0.0)
                            )
                            raise RuntimeError(
                                "Strict runtime budget was not met "
                                f"(overshoot={overshoot:.6f}s); result not saved."
                            )
                        opt_runtime = (
                            result.get("optimization_runtime")
                            or stats.get("optimization_runtime")
                            or stats.get("total_runtime")
                            or rt
                        )
                        total_runtime = result.get("total_runtime") or stats.get("total_runtime")
                        result["optimization_runtime"] = opt_runtime
                        if total_runtime is not None:
                            result["total_runtime"] = total_runtime
                        result["wall_runtime"] = rt
                        result['runtime'] = opt_runtime
                        result['seed'] = seed
                        if 'problem_data' in result: result['problem_data']['durations_matrix'] = "SAVED_SEPARATELY"

                        with open(f_path, 'w') as f: json.dump(result, f, indent=2)
                        
                        summary = extract_run_summary(result, s_conf['name'], f_name, a_name, run_num, rt, seed=seed)
                        all_results_summary.append(summary)
                        save_summary_file(output_base_dir, all_results_summary)
                        print(f"  ({summary['avg_evac_time']:.1f}m)")

                    except Exception as e:
                        print(f"  ERROR: {e}")
                        all_results_summary.append(
                            extract_run_summary(
                                {}, s_conf['name'], f_name, a_name, run_num,
                                time.time()-t0, False, e, seed=seed
                            )
                        )
                        save_summary_file(output_base_dir, all_results_summary)

    print("\n Done.")

if __name__ == "__main__":
    main()
