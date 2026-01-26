"""
GECCO 2025 Comprehensive Analysis Suite - Merged Version
Generates 6 LaTeX tables, 4 Visualization plots, and 1 Example Solutions file.

Tables:
1. Performance (Wait/Makespan/Late) + Wilcoxon Signed-Rank with Holm Correction
2. Efficiency (Travel Time/Trips/Utility)
3. Robustness (Mean vs Worst-Case Risk Analysis)
4. Convergence Speed (Time to 95% and 99% of final quality)
5. Operator Forensics (Cumulative Gain, LS vs M)
6. Vehicle Type Trip Activity (Stops/Trip + Mean Stops/Trips)

Plots:
1. Convergence (Fitness vs Wall-clock Time with interpolation)
2. Efficiency Breakdown (Driving vs Service Overhead)
3. Robustness Boxplots
4. Operator Impact Bar Chart

Results Files:
1. Example Solutions (Simulation data per scenario/algorithm with timestamps)
"""

import os
import json
import importlib
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 10,
    "figure.titlesize": 16
})

# Metric configuration for convergence plots
METRIC_CONFIG = {
    "cost": {
        "json_key": "generation_costs", 
        "summary_col": "cost_fitness",
        "label": "Total Cost (Fitness)", 
        "log_scale": True
    },
    "makespan": {
        "json_key": "generation_latest_evacuation_times",
        "summary_col": "makespan",
        "label": "Makespan (min)", 
        "log_scale": False
    },
    "wait_time": {
        "json_key": "generation_avg_evacuation_times",
        "summary_col": "avg_evac_time",
        "label": "Avg. Evacuation Time (min)", 
        "log_scale": False
    }
}

# Canonical naming used for the paper tables/figures.
# Keys are taken from EA `operator_scoreboard` (exported as `op_gain_{key}` in the summary JSON).
OPERATOR_META = {
    # --- Local Search (Memetic) ---
    "intra_trip": ("LS", "Intra-Trip TSP"),
    "relocate": ("LS", "Relocate Stop"),
    "swap_stops": ("LS", "Swap Stops"),
    "swap_trips": ("LS", "Swap Trips"),
    "move_trip": ("LS", "Move Trip"),
    "change_depot": ("LS", "Change End Depot"),
    "consolidate_trips": ("LS", "Consolidate Trips"),
    "balance_makespan": ("LS", "Makespan Balancer"),
    "takeover_gap": ("LS", "Takeover Gap"),
    "fill_idle_time": ("LS", "Fill Idle Time"),
    "spatial_relocate": ("LS", "Spatial Relocate"),
    "split_mixed": ("LS", "Split Mixed Trips"),
    "crumb_extract": ("LS", "Crumb Extraction"),
    "self_consolidate": ("LS", "Self-Consolidate"),

    # --- Mutation (GA) ---
    "mutate_intra_swap": ("M", "Intra-Swap"),
    "mutate_relocate_stop": ("M", "Relocate Stop"),
    "mutate_add_remove_trip": ("M", "Add/Remove Trip"),
    "mutate_change_depot": ("M", "Change Depot"),
    "mutate_swap_trip": ("M", "Swap Trip"),
    "mutate_spatial_ruin": ("M", "Spatial Ruin-Recreate"),
}

# Exclude non-LS/non-mutation bookkeeping from operator forensics.
# - `ga_crossover` is not part of the LS vs M breakdown in the paper table.
# - `quantity_rebalance` is a memetic helper and not part of the 14-operator portfolio.
EXCLUDED_OPERATOR_KEYS = {"ga_crossover", "quantity_rebalance"}
MIN_OPERATOR_GAIN = 1.0

ALGO_ORDER = ["Dispatcher", "Shuttle", "GA", "MA", "ALNS"]
BASELINE_ALGOS = ("Dispatcher", "Shuttle")
SCENARIO_FLEET_LABELS = {
    ("Default", "Standard"): "Synthetic Mass Transit",
}
FLEET_LABELS = {
    "default": "Standard",
    "augmented": "Augmented",
    "specialized_only": "Specialized"
}

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def _clean_scenario_name(name: str) -> str:
    return str(name).replace("FloodWilhelmsburg", "Flood").replace("BombThreat", "Bomb")

def _scenario_fleet_display(scenario: str, fleet_label: str) -> str:
    clean = _clean_scenario_name(scenario)
    special = SCENARIO_FLEET_LABELS.get((clean, fleet_label))
    if special:
        return special
    return f"{clean} - {fleet_label}"

def _scenario_fleet_shortstack(scenario: str, fleet_label: str) -> str:
    clean = _clean_scenario_name(scenario)
    special = SCENARIO_FLEET_LABELS.get((clean, fleet_label))
    if special:
        parts = special.split(" ")
        if len(parts) >= 2:
            return " ".join(parts[:-1]) + "\\\\" + parts[-1]
        return special
    return f"{clean}\\\\{fleet_label}"

def _algo_base_name(name: str) -> str:
    if not name:
        return ""
    raw = str(name)
    lower = raw.lower()
    if lower.startswith("dispatcher"):
        return "Dispatcher"
    if lower.startswith("shuttle"):
        return "Shuttle"
    if lower.startswith("ga"):
        return "GA"
    if lower.startswith("ma"):
        return "MA"
    if lower.startswith("alns"):
        return "ALNS"
    return raw

def _is_baseline_algo(name: str) -> bool:
    return _algo_base_name(name) in BASELINE_ALGOS

def order_algorithms(algo_names):
    if algo_names is None:
        return []
    names = list(algo_names)
    if len(names) == 0:
        return []
    ordered = []
    for base in ALGO_ORDER:
        ordered.extend([n for n in names if _algo_base_name(n) == base])
    extras = [n for n in names if _algo_base_name(n) not in ALGO_ORDER]
    ordered.extend(sorted(extras, key=str.lower))
    return ordered

def format_operator_gain(val: float) -> str:
    if val == 0 or abs(val) < 1e-9:
        return "--"
    abs_val = abs(val)
    if abs_val >= 1_000_000:
        return f"{val / 1_000_000:.1f}M"
    if abs_val >= 1_000:
        return f"{val / 1_000:.1f}k"
    if abs_val >= 10:
        return f"{val:.1f}"
    return f"{val:.2f}"

def operator_key_meta(key: str):
    if key in OPERATOR_META:
        return OPERATOR_META[key]
    if key.startswith("D:"):
        return ("D", key[2:].replace("_", " ").title())
    if key.startswith("R:"):
        return ("R", key[2:].replace("_", " ").title())
    if key.startswith("LS:"):
        return ("LS", key[3:].replace("_", " ").title())
    if key.startswith("M:"):
        return ("M", key[2:].replace("_", " ").title())
    if key.startswith("mutate_"):
        return ("M", key.replace("mutate_", "", 1).replace("_", " ").title())
    return ("LS", key.replace("_", " ").title())

def collect_operator_stats(df: pd.DataFrame, min_gain: float = MIN_OPERATOR_GAIN):
    gain_cols = [c for c in df.columns if c.startswith("op_gain_") and not c.endswith("_cnt")]
    if not gain_cols:
        return {}, []
    algo_stats = {}
    for algo, sub in df.groupby("algorithm"):
        sums = sub[gain_cols].fillna(0).sum()
        ops = {}
        for col, val in sums.items():
            if pd.isna(val):
                continue
            if abs(val) <= min_gain:
                continue
            key = col.replace("op_gain_", "", 1)
            if key in EXCLUDED_OPERATOR_KEYS:
                continue
            ops[key] = float(val)
        if ops:
            algo_stats[str(algo)] = ops
    return algo_stats, gain_cols

def iter_scenario_fleets(results_dir: Path):
    for scen_dir in sorted(results_dir.iterdir(), key=lambda p: p.name):
        if not scen_dir.is_dir() or scen_dir.name.startswith("paper_") or scen_dir.name == "analysis_output":
            continue
        for fleet_dir in sorted(scen_dir.iterdir(), key=lambda p: p.name):
            if not fleet_dir.is_dir() or fleet_dir.name.startswith("."):
                continue
            yield scen_dir, fleet_dir

def list_algorithms(fleet_dir: Path):
    algo_dirs = [
        d.name for d in fleet_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ]
    return order_algorithms(algo_dirs)

def get_latest_valid_experiment_dir(base_path: Path):
    """Finds the most recent directory containing 'all_runs_summary.json'."""
    if not base_path.exists():
        print(f"⚠️  Directory not found: {base_path.resolve()}")
        return None
    
    candidates = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("paper_results_")]
    
    if not candidates:
        print(f"⚠️  No 'paper_results_' folders found in {base_path.resolve()}")
        return None

    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    for d in candidates:
        if (d / "all_runs_summary.json").exists():
            return d
    return None

def load_summary_data(results_dir: Path) -> pd.DataFrame:
    """Loads the main summary JSON into a clean DataFrame."""
    summary_file = results_dir / "all_runs_summary.json"
    if not summary_file.exists():
        return pd.DataFrame()
    
    print(f"📂 Loading Summary: {summary_file}")
    df = pd.read_json(summary_file)
    df = df[df['success'] == True].copy()

    if 'optimization_runtime' in df.columns:
        if 'runtime' in df.columns:
            df['runtime'] = df['optimization_runtime'].fillna(df['runtime'])
        else:
            df['runtime'] = df['optimization_runtime']
    
    df['fleet_label'] = df['fleet'].replace(FLEET_LABELS)
    return df

def _json_safe_value(val):
    try:
        if pd.isna(val):
            return None
    except Exception:
        pass
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    return val

def _load_scenario_fleet_map():
    candidates = [
        Path(__file__).resolve().parents[1] / "app" / "evacuation" / "scenarios.py",
        Path.cwd() / "backend" / "app" / "evacuation" / "scenarios.py",
        Path.cwd() / "app" / "evacuation" / "scenarios.py",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            spec = importlib.util.spec_from_file_location("_evacuation_scenarios", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception:
            continue
        scenarios = getattr(module, "ALL_SCENARIOS", None)
        if not isinstance(scenarios, dict):
            continue
        scenario_map = {}
        for data in scenarios.values():
            if not isinstance(data, dict):
                continue
            name = data.get("name")
            if not name:
                continue
            scenario_map[str(name)] = data.get("fleets", {})
        return scenario_map
    return {}

def _capacity_to_vehicle_type_map(fleet_spec):
    mapping = {}
    if not isinstance(fleet_spec, list):
        return mapping
    for vehicle in fleet_spec:
        if not isinstance(vehicle, dict):
            continue
        cap = vehicle.get("capacity")
        if cap is None:
            continue
        try:
            cap = int(cap)
        except Exception:
            continue
        vtype = vehicle.get("type") or f"capacity_{cap}"
        mapping.setdefault(cap, set()).add(str(vtype))
    resolved = {}
    for cap, types in mapping.items():
        if len(types) == 1:
            resolved[cap] = sorted(types)[0]
        else:
            resolved[cap] = f"capacity_{cap}"
    return resolved

def _vehicle_type_capacity_map(fleet_spec):
    mapping = {}
    if not isinstance(fleet_spec, list):
        return mapping
    for vehicle in fleet_spec:
        if not isinstance(vehicle, dict):
            continue
        vtype = vehicle.get("type")
        cap = vehicle.get("capacity")
        if vtype is None or cap is None:
            continue
        try:
            cap = int(cap)
        except Exception:
            continue
        mapping.setdefault(str(vtype), set()).add(cap)
    resolved = {}
    for vtype, caps in mapping.items():
        if len(caps) == 1:
            resolved[vtype] = next(iter(caps))
    return resolved

def _trip_pickup_count(trip):
    if not isinstance(trip, dict):
        return 0
    pickup_counts = trip.get("pickup_counts")
    if isinstance(pickup_counts, dict):
        total = 0
        for val in pickup_counts.values():
            try:
                total += int(val)
            except Exception:
                continue
        return total
    stops = trip.get("stops", [])
    if stops and isinstance(stops[0], (list, tuple)) and len(stops[0]) >= 2:
        total = 0
        for stop in stops:
            if not isinstance(stop, (list, tuple)) or len(stop) < 2:
                continue
            try:
                total += int(stop[1])
            except Exception:
                continue
        return total
    return 0

def _resolve_vehicle_capacity(bus_idx, vehicles, vehicle_type, type_capacity_map, fallback_capacity=None):
    if isinstance(vehicles, list) and bus_idx < len(vehicles):
        vehicle = vehicles[bus_idx]
        if isinstance(vehicle, dict):
            cap = vehicle.get("capacity")
            if cap is not None:
                try:
                    return int(cap)
                except Exception:
                    pass
            vtype = vehicle.get("type") or vehicle.get("vehicle_type")
            if vtype in type_capacity_map:
                return type_capacity_map[vtype]
    cap = type_capacity_map.get(vehicle_type)
    if cap is None and isinstance(vehicle_type, str) and vehicle_type.startswith("capacity_"):
        try:
            cap = int(vehicle_type.split("_", 1)[1])
        except Exception:
            cap = None
    if cap is None and fallback_capacity is not None:
        try:
            cap = int(fallback_capacity)
        except Exception:
            cap = None
    return cap

def _resolve_vehicle_type(bus_idx, vehicles, capacity_map, fallback_capacity=None):
    if isinstance(vehicles, list) and bus_idx < len(vehicles):
        vehicle = vehicles[bus_idx]
        if isinstance(vehicle, dict):
            vtype = vehicle.get("type") or vehicle.get("vehicle_type")
            if vtype:
                return str(vtype)
            cap = vehicle.get("capacity")
            if cap is not None:
                try:
                    cap = int(cap)
                except Exception:
                    cap = None
                if cap is not None:
                    if cap in capacity_map:
                        return capacity_map[cap]
                    return f"capacity_{cap}"
    if capacity_map and len(capacity_map) == 1:
        return next(iter(capacity_map.values()))
    if fallback_capacity is not None:
        return f"capacity_{fallback_capacity}"
    return "unknown"

def _count_vehicle_type_trips(best_solution, vehicles, capacity_map, type_capacity_map, fallback_capacity=None):
    counts = {}
    if not isinstance(best_solution, list):
        return counts
    for bus_idx, bus_schedule in enumerate(best_solution):
        if not isinstance(bus_schedule, list):
            continue
        vehicle_type = _resolve_vehicle_type(bus_idx, vehicles, capacity_map, fallback_capacity)
        capacity = _resolve_vehicle_capacity(
            bus_idx,
            vehicles,
            vehicle_type,
            type_capacity_map,
            fallback_capacity,
        )
        entry = counts.setdefault(
            vehicle_type,
            {"trips": 0, "stops": 0, "fullness_sum": 0.0, "fullness_count": 0},
        )
        trip_count = len(bus_schedule)
        stop_count = 0
        for trip in bus_schedule:
            if not isinstance(trip, dict):
                continue
            stop_count += len(trip.get("stops", []) or [])
            if capacity:
                trip_load = _trip_pickup_count(trip)
                entry["fullness_sum"] += (trip_load / capacity) * 100.0
                entry["fullness_count"] += 1
        entry["trips"] += trip_count
        entry["stops"] += stop_count
    return counts

def collect_vehicle_type_trip_stats(results_dir: Path) -> pd.DataFrame:
    scenario_fleet_map = _load_scenario_fleet_map()
    rows = []

    for scen_dir, fleet_dir in iter_scenario_fleets(results_dir):
        scenario = scen_dir.name
        fleet = fleet_dir.name
        fleet_spec = scenario_fleet_map.get(scenario, {}).get(fleet, [])
        capacity_map = _capacity_to_vehicle_type_map(fleet_spec)
        type_capacity_map = _vehicle_type_capacity_map(fleet_spec)

        algo_dirs = [
            d for d in fleet_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]
        for algo_dir in sorted(algo_dirs, key=lambda p: p.name):
            algo = algo_dir.name
            for run_file in algo_dir.glob("run_*.json"):
                try:
                    with open(run_file) as f:
                        run_data = json.load(f)
                except Exception:
                    continue

                best_solution = run_data.get("best_solution")
                vehicles = run_data.get("vehicles")
                fallback_capacity = run_data.get("bus_capacity")
                try:
                    fallback_capacity = int(fallback_capacity) if fallback_capacity is not None else None
                except Exception:
                    fallback_capacity = None

                counts = _count_vehicle_type_trips(
                    best_solution,
                    vehicles,
                    capacity_map,
                    type_capacity_map,
                    fallback_capacity,
                )
                if not counts:
                    continue
                for vehicle_type, stats in counts.items():
                    trips = stats.get("trips", 0)
                    stops = stats.get("stops", 0)
                    fullness_sum = stats.get("fullness_sum", 0.0)
                    fullness_count = stats.get("fullness_count", 0)
                    fullness_mean = (fullness_sum / fullness_count) if fullness_count else np.nan
                    rows.append({
                        "scenario": scenario,
                        "fleet": fleet,
                        "algorithm": algo,
                        "vehicle_type": vehicle_type,
                        "trip_count": trips,
                        "stop_count": stops,
                        "trips_per_stop": (trips / stops) if stops else np.nan,
                        "stops_per_trip": (stops / trips) if trips else np.nan,
                        "fullness_percent": fullness_mean,
                    })

    return pd.DataFrame(rows)

STOCHASTIC_ALGOS = {"GA", "MA", "ALNS"}

def compute_paired_pvalue(df_sub, metric, best_algo, current_algo):
    """
    Computes Wilcoxon signed-rank p-value for paired runs (one-sided).
    Tests whether current_algo is significantly worse than best_algo.
    Returns None if test cannot be performed.
    """
    df_best = df_sub[df_sub['algorithm'] == best_algo].sort_values('run')
    df_curr = df_sub[df_sub['algorithm'] == current_algo].sort_values('run')
    
    # Find common runs (paired by seed)
    common_runs = sorted(set(df_best['run']) & set(df_curr['run']))
    
    if len(common_runs) < 5:  # Wilcoxon needs sufficient pairs
        return None
    
    vals_best = df_best[df_best['run'].isin(common_runs)].sort_values('run')[metric].values
    vals_curr = df_curr[df_curr['run'].isin(common_runs)].sort_values('run')[metric].values
    
    differences = vals_curr - vals_best
    
    if np.all(differences == 0):
        return 1.0
    if np.std(differences) == 0:
        return 0.0 if differences[0] > 0 else 1.0
    
    try:
        # One-sided: test if current is worse (greater) than best
        _, p = wilcoxon(vals_best, vals_curr, alternative='less')
        return p
    except ValueError:
        return None


def compute_all_significance(df, metrics, scenarios_fleets):
    """
    Computes pairwise significance tests with global Holm-Bonferroni correction.
    
    Only stochastic algorithms (GA, MA, ALNS) are tested. Dispatcher is excluded.
    Within each (scenario, fleet, metric), the best stochastic method is identified
    and compared against each other stochastic method using paired Wilcoxon signed-rank
    tests (one-sided, α=0.05). P-values are corrected globally across all tests.
    
    Returns dict mapping (scenario, fleet, metric, algo) -> marker string.
    """
    all_tests = []
    
    for scen, fleet in scenarios_fleets:
        sub = df[(df['scenario'] == scen) & (df['fleet_label'] == fleet)]
        
        # Filter to stochastic algorithms only
        sub_stochastic = sub[sub['algorithm'].isin(STOCHASTIC_ALGOS)]
        
        if sub_stochastic.empty:
            continue
        
        for metric in metrics:
            agg = sub_stochastic.groupby('algorithm')[metric].mean()
            if agg.empty or len(agg) < 2:
                continue
            
            best_algo = agg.idxmin()
            
            for algo in agg.index:
                if algo == best_algo:
                    all_tests.append((scen, fleet, metric, algo, True, None))
                else:
                    p = compute_paired_pvalue(sub_stochastic, metric, best_algo, algo)
                    all_tests.append((scen, fleet, metric, algo, False, p))
    
    # Extract valid p-values and apply global Holm correction
    valid_indices = [i for i, t in enumerate(all_tests) if t[5] is not None]
    p_values = [all_tests[i][5] for i in valid_indices]
    
    reject = [False] * len(all_tests)
    if p_values:
        rej, _, _, _ = multipletests(p_values, alpha=0.05, method='holm')
        for idx, valid_idx in enumerate(valid_indices):
            reject[valid_idx] = rej[idx]
    
    # Build result dict
    results = {}
    for i, (scen, fleet, metric, algo, is_best, p) in enumerate(all_tests):
        key = (scen, fleet, metric, algo)
        if is_best or p is None:
            results[key] = ""
        else:
            results[key] = "$^\\dagger$" if reject[i] else ""
    
    return results

def format_cell(val, std, is_best, marker):
    """Formats LaTeX cell content."""
    std_str = f"{std:.1f}" if std >= 0.1 else "0.0"
    txt = f"{val:.1f} $\\pm$ {std_str}{marker}"
    return f"\\textbf{{{txt}}}" if is_best else txt

def save_latex_table(content: str, filename: str, output_dir: Path):
    """Saves LaTeX table to file."""
    with open(output_dir / filename, "w") as f:
        f.write(content)
    print(f"  📄 Saved LaTeX: {filename}")

def save_plot(fig, filename: str, output_dir: Path):
    """Saves matplotlib figure to file."""
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    print(f"  📊 Saved Plot:  {filename}")
    plt.close(fig)

# ==============================================================================
# TABLE GENERATORS
# ==============================================================================

def generate_global_operator_plot(results_dir: Path, output_dir: Path):
    """
    Generates a single Global Stacked Area Chart averaging ALL scenarios/fleets.
    
    CRITICAL: Performs Normalization so that every scenario contributes equally 
    (Area Under Curve = 1.0 for each scenario), preventing large-scale scenarios 
    from dominating the average.
    """
    print("\n--- Generating Global Operator Evolution Plot (Normalized) ---")

    all_normalized_runs = []
    max_gen_observed = 0

    # 1. Collect and Normalize Data from ALL folders
    for scen_dir in results_dir.iterdir():
        if not scen_dir.is_dir() or scen_dir.name.startswith("paper_") or scen_dir.name == "analysis_output":
            continue

        for fleet_dir in scen_dir.iterdir():
            if not fleet_dir.is_dir():
                continue

            ma_dir = fleet_dir / "MA"
            if not ma_dir.exists():
                continue

            # Load all runs for this specific Scenario/Fleet configuration
            local_dfs = []
            for run_file in ma_dir.glob("**/operator_history.csv"):
                try:
                    df = pd.read_csv(run_file)
                    
                    # 1a. Convert Cumulative -> Marginal (Gain per Gen)
                    gain_cols = [c for c in df.columns if c.endswith('_gain')]
                    df_marginal = df[gain_cols].diff().fillna(df[gain_cols].iloc[0]).clip(lower=0)
                    
                    # 1b. Normalize this run so Total Area = 1.0
                    total_run_gain = df_marginal.sum().sum()
                    if total_run_gain > 0:
                        df_norm = df_marginal / total_run_gain
                        df_norm['Generation'] = df['Generation']
                        local_dfs.append(df_norm)
                        max_gen_observed = max(max_gen_observed, df['Generation'].max())
                        
                except Exception:
                    continue
            
            # Average the runs for this Scenario first (to represent this scenario as one unit)
            if local_dfs:
                df_scen_avg = pd.concat(local_dfs).groupby('Generation').mean().reset_index()
                all_normalized_runs.append(df_scen_avg)

    if not all_normalized_runs:
        print("  ⚠️ No data found for global plot.")
        return

    # 2. Aggregate ALL Scenarios (Equal Weighting)
    # We concat all the "Scenario Averages" and average them.
    # Since they are already normalized (Sum=1.0), this creates a perfect equal-weight composite.
    df_concat = pd.concat(all_normalized_runs)
    df_global = df_concat.groupby('Generation').mean().reset_index()

    # 3. Identify Top Operators (Global)
    total_gains = df_global.drop('Generation', axis=1).sum().sort_values(ascending=False)
    
    TOP_N = 8  # Show top 8 specific operators
    top_ops = total_gains.head(TOP_N).index.tolist()
    
    plot_data = df_global[['Generation'] + top_ops].copy()
    
    # Calculate "Others"
    other_ops = total_gains.iloc[TOP_N:].index.tolist()
    if other_ops:
        plot_data['Others'] = df_global[other_ops].sum(axis=1)
        top_ops.append('Others')

    # 4. Smoothing (Makes phases distinct)
    WINDOW_SIZE = int(max_gen_observed * 0.05) # Dynamic smoothing window (5% of total time)
    if WINDOW_SIZE < 1: WINDOW_SIZE = 1
    
    plot_data_smoothed = plot_data.rolling(window=WINDOW_SIZE, min_periods=1).mean()
    plot_data_smoothed['Generation'] = plot_data['Generation']

    # 5. Plotting
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = sns.color_palette("turbo", n_colors=len(top_ops))
    if 'Others' in top_ops:
        colors[-1] = (0.8, 0.8, 0.8) # Grey for Others

    # Pretty Labels
    labels = []
    for col in top_ops:
        if col == "Others":
            labels.append("Others")
            continue
        clean = col.replace("_gain", "").replace("mutate_", "M: ")
        # Try to map to the canonical names in OPERATOR_META
        key = col.replace("_gain", "").replace("mutate_", "mutate_") # heuristic to match keys
        if key in OPERATOR_META:
            clean = f"{OPERATOR_META[key][0]}: {OPERATOR_META[key][1]}"
        labels.append(clean)

    ax.stackplot(
        plot_data_smoothed['Generation'],
        [plot_data_smoothed[col] for col in top_ops],
        labels=labels,
        colors=colors,
        alpha=0.9
    )

    plt.title("Global Operator Influence (Averaged across all Scenarios)\nNormalized Contribution Density", fontsize=15)
    plt.ylabel("Relative Contribution Intensity (Normalized)")
    plt.xlabel("Generation")
    plt.xlim(5, max_gen_observed)
    plt.margins(0, 0)
    
    # Legend
    plt.legend(loc='upper right', title="Top Contributors", frameon=True, facecolor='white', framealpha=0.9)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    save_plot(fig, "fig_global_operator_evolution.png", output_dir)

def generate_performance_table(df: pd.DataFrame, output_dir: Path):
    """Table 1: Performance & Statistical Significance (Wilcoxon + Holm)"""
    print("\n--- Generating Table 1: Performance & Significance (Wilcoxon + Holm) ---")
    
    metrics = {
        'avg_evac_time': 'Avg Wait (min)',
        'makespan': 'Makespan (min)',
        'cost_fitness': 'Total Cost'
    }
    
    # Gather all scenario/fleet combinations
    scenarios_fleets = []
    for scen in df['scenario'].unique():
        for fleet in df[df['scenario'] == scen]['fleet_label'].unique():
            scenarios_fleets.append((scen, fleet))
    
    # Compute all significance markers with Holm correction
    significance_markers = compute_all_significance(df, metrics.keys(), scenarios_fleets)
    
    n_significant = sum(1 for v in significance_markers.values() if v)
    print(f"    Significant results: {n_significant}/{len(significance_markers)} (after Holm correction)")
    
    latex = []
    latex.append("\\begin{table*}[t]")
    latex.append("\\centering")
    latex.append("\\caption{Performance Comparison (Mean $\\pm$ Std. Dev.). "
                 "\\textbf{Bold} indicates best result. "
                 "$^\\dagger$: Significantly worse than best (Wilcoxon signed-rank, $p<0.05$, Holm-corrected).}")
    latex.append("\\label{tab:performance}")
    latex.append("\\setlength{\\tabcolsep}{6pt}")
    latex.append("\\begin{tabular}{llccc}")
    latex.append("\\toprule")
    latex.append("\\textbf{Scenario} & \\textbf{Method} & \\textbf{Avg Wait} & \\textbf{Makespan} & \\textbf{Total Cost} \\\\")
    latex.append("\\midrule")
    
    for scen, fleet in scenarios_fleets:
        sub = df[(df['scenario'] == scen) & (df['fleet_label'] == fleet)]
        
        shortstack_label = _scenario_fleet_shortstack(scen, fleet)
        agg = sub.groupby('algorithm').mean(numeric_only=True)
        algos_present = order_algorithms(agg.index)
        
        if not algos_present:
            continue
            
        latex.append(f"\\multirow{{{len(algos_present)}}}{{*}}{{\\shortstack[l]{{{shortstack_label}}}}}")
        
        best_algos = {m: agg[m].idxmin() for m in metrics}

        for algo in algos_present:
            stats = sub[sub['algorithm'] == algo]
            row_str = f" & {algo}"
            
            for m_key in metrics:
                val = stats[m_key].mean()
                std = stats[m_key].std()
                is_best = (algo == best_algos[m_key])
                marker = significance_markers.get((scen, fleet, m_key, algo), "")
                row_str += f" & {format_cell(val, std, is_best, marker)}"
            
            latex.append(row_str + " \\\\")
        latex.append("\\addlinespace")
        
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table*}")
    
    save_latex_table("\n".join(latex), "table_1_performance.tex", output_dir)

def generate_efficiency_table(df: pd.DataFrame, output_dir: Path):
    """Table 2: Operational Efficiency"""
    print("\n--- Generating Table 2: Efficiency ---")
    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{Operational Efficiency. Lower travel time implies smarter routing logic.}")
    latex.append("\\label{tab:efficiency}")
    latex.append("\\begin{tabular}{llccc}")
    latex.append("\\toprule")
    latex.append("\\textbf{Scenario} & \\textbf{Method} & \\textbf{Total Travel (min)} & \\textbf{Total Trips} & \\textbf{Stops/Trip} \\\\")
    latex.append("\\midrule")
    
    for scen in df['scenario'].unique():
        fleets = df[df['scenario'] == scen]['fleet_label'].unique()
        for fleet in fleets:
            sub = df[(df['scenario'] == scen) & (df['fleet_label'] == fleet)].copy()
            
            label = _scenario_fleet_display(scen, fleet)
            latex.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{label}}}}} \\\\")
            
            if 'stop_count' in sub.columns:
                stops = sub['stop_count'].fillna(sub['trip_count'])
            else:
                stops = sub['trip_count']
            trips = sub['trip_count']
            sub['stops_per_trip'] = np.where(trips > 0, stops / trips, np.nan)

            agg = sub.groupby('algorithm')[['total_travel_time', 'trip_count', 'stops_per_trip']].mean()
            algos_present = order_algorithms(agg.index)
            if agg.empty or not algos_present:
                continue
            best_travel = agg['total_travel_time'].idxmin()
            
            for algo in algos_present:
                
                tt = agg.loc[algo, 'total_travel_time']
                trips = agg.loc[algo, 'trip_count']
                stops_per_trip = agg.loc[algo, 'stops_per_trip']
                tt_str = f"{tt:.1f}"
                if algo == best_travel:
                    tt_str = f"\\textbf{{{tt_str}}}"

                if np.isnan(stops_per_trip):
                    stops_per_trip_str = "--"
                else:
                    stops_per_trip_str = f"{stops_per_trip:.2f}"
                
                latex.append(f" & {algo} & {tt_str} & {trips:.1f} & {stops_per_trip_str} \\\\")
            latex.append("\\addlinespace")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    save_latex_table("\n".join(latex), "table_2_efficiency.tex", output_dir)

def generate_robustness_table(df: pd.DataFrame, output_dir: Path):
    """Table 3: Robustness Analysis (Mean vs Worst-Case)"""
    print("\n--- Generating Table 3: Robustness Analysis ---")
    
    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{Robustness Analysis. Stability metric indicates predictability (Mean/Max).}")
    latex.append("\\label{tab:robustness}")
    latex.append("\\begin{tabular}{llcc|c}")
    latex.append("\\toprule")
    latex.append("\\textbf{Scenario} & \\textbf{Method} & \\textbf{Mean Wait} & \\textbf{Max Wait} & \\textbf{Stability (\\%)} \\\\")
    latex.append("\\midrule")
    
    for scen in df['scenario'].unique():
        fleets = df[df['scenario'] == scen]['fleet_label'].unique()
        for fleet in fleets:
            sub = df[(df['scenario'] == scen) & (df['fleet_label'] == fleet)]
            
            # Skip Shuttle (deterministic)
            label = _scenario_fleet_display(scen, fleet)
            latex.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{label}}}}} \\\\")
            
            agg = sub.groupby('algorithm')['avg_evac_time'].agg(['mean', 'max'])
            algos_present = order_algorithms(agg.index)
            if agg.empty or not algos_present:
                continue
            
            for algo in algos_present:
                
                mean_val = agg.loc[algo, 'mean']
                max_val = agg.loc[algo, 'max']
                stability = (mean_val / max_val) * 100 if max_val > 0 else 100.0
                
                latex.append(f" & {algo} & {mean_val:.1f} & {max_val:.1f} & {stability:.1f}\\% \\\\")
            latex.append("\\addlinespace")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    save_latex_table("\n".join(latex), "table_3_robustness.tex", output_dir)

def calculate_convergence_stats(base_path: Path) -> pd.DataFrame:
    """
    Parses individual run JSONs to find wall-clock time required to reach
    95% and 99% of final solution quality.
    """
    results = []
    
    for scen_dir, fleet_dir in iter_scenario_fleets(base_path):
        algo_dirs = [
            d for d in fleet_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]
        for algo_dir in sorted(algo_dirs, key=lambda p: p.name):
            algo = algo_dir.name
            times_95 = []
            times_99 = []

            for run_file in algo_dir.glob("run_*.json"):
                try:
                    with open(run_file) as f:
                        d = json.load(f)

                    stats = d.get("algorithm_stats", {})

                    progress = stats.get("progress", [])
                    pairs = [
                        (p.get("elapsed_seconds"), p.get("best_cost"))
                        for p in progress
                    ]
                    pairs = [(t, v) for t, v in pairs if t is not None and v is not None]
                    if pairs:
                        times, costs = zip(*pairs)
                        times = list(times)
                        costs = list(costs)
                    else:
                        costs = stats.get("generation_costs", [])
                        if not costs and "history_fitness" in stats:
                            costs = stats["history_fitness"]
                        gen_times = stats.get("generation_times", [])

                        if not costs or not gen_times:
                            continue

                        min_len = min(len(costs), len(gen_times))
                        costs = costs[:min_len]
                        gen_times = gen_times[:min_len]
                        times = np.cumsum(gen_times).tolist()

                    start_cost = costs[0]
                    final_cost = costs[-1]
                    total_improvement = start_cost - final_cost

                    if total_improvement <= 0:
                        continue

                    target_95 = start_cost - (0.95 * total_improvement)
                    target_99 = start_cost - (0.99 * total_improvement)

                    idx_95 = next((i for i, x in enumerate(costs) if x <= target_95), len(costs)-1)
                    idx_99 = next((i for i, x in enumerate(costs) if x <= target_99), len(costs)-1)

                    times_95.append(times[idx_95])
                    times_99.append(times[idx_99])

                except Exception:
                    continue

            if times_95:
                results.append({
                    "scenario": scen_dir.name,
                    "fleet": fleet_dir.name,
                    "algorithm": algo,
                    "t95": np.mean(times_95),
                    "t99": np.mean(times_99)
                })
                    
    return pd.DataFrame(results)

def generate_convergence_table(results_dir: Path, output_dir: Path):
    """Table 4: Convergence Speed"""
    print("\n--- Generating Table 4: Convergence Speed ---")
    
    df = calculate_convergence_stats(results_dir)
    if df.empty:
        print("  ⚠️ No convergence data found (requires raw JSON logs).")
        return

    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{Convergence Speed (Seconds). Time to reach 95\\% and 99\\% of final solution quality.}")
    latex.append("\\label{tab:convergence}")
    latex.append("\\begin{tabular}{llcc}")
    latex.append("\\toprule")
    latex.append("\\textbf{Scenario} & \\textbf{Method} & \\textbf{Time to 95\\%} & \\textbf{Time to 99\\%} \\\\")
    latex.append("\\midrule")
    
    for scen_dir, fleet_dir in iter_scenario_fleets(results_dir):
        scen = scen_dir.name
        fleet = fleet_dir.name
        fleet_label = FLEET_LABELS.get(fleet, fleet)
        label = _scenario_fleet_display(scen, fleet_label)
        sub = df[(df['scenario'] == scen) & (df['fleet'] == fleet)]
        algos_present = list_algorithms(fleet_dir)
        if not algos_present:
            continue
        latex.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{label}}}}} \\\\")

        for algo in algos_present:
            row = sub[sub['algorithm'] == algo]
            if row.empty:
                t95_str = "--"
                t99_str = "--"
            else:
                t95 = row['t95'].values[0]
                t99 = row['t99'].values[0]
                t95_str = f"{t95:.1f} s"
                t99_str = f"{t99:.1f} s"

            latex.append(f" & {algo} & {t95_str} & {t99_str} \\\\")
        latex.append("\\addlinespace")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    save_latex_table("\n".join(latex), "table_4_convergence.tex", output_dir)

def generate_operator_table_legacy(df: pd.DataFrame, output_dir: Path):
    """Table 5: Memetic Operator Forensics (Legacy)"""
    print("\n--- Generating Table 5: Operator Forensics ---")
    
    df_ma = df[df['algorithm'] == 'MA']
    if df_ma.empty:
        print("  ⚠️ No MA data found for operator table.")
        return

    gain_cols = [c for c in df_ma.columns if c.startswith("op_gain_") and not c.endswith("_cnt")]
    if not gain_cols:
        return
    
    rows = []
    for col in gain_cols:
        key = col.replace("op_gain_", "", 1)
        if key in EXCLUDED_OPERATOR_KEYS:
            continue
        gain = float(df_ma[col].sum())
        group, name = OPERATOR_META.get(
            key,
            ("M" if key.startswith("mutate_") else "LS", key.replace("mutate_", "", 1).replace("_", " ").title()),
        )
        rows.append({"key": key, "group": group, "name": name, "gain": gain})

    if not rows:
        return

    df_ops = pd.DataFrame(rows).sort_values("gain", ascending=False).reset_index(drop=True)
    total_gain = float(df_ops["gain"].sum()) or 1.0

    def format_gain_millions(val: float) -> str:
        return f"{val / 1_000_000.0:.1f}M"

    top_n = 10
    top_df = df_ops.iloc[:top_n]
    other_df = df_ops.iloc[top_n:]

    latex = []
    latex.append("\\begin{table}[t]")
    latex.append("\\centering")
    latex.append("\\caption{Operator Contributions. LS = Local Search, M = Mutation.}")
    latex.append("\\label{tab:operators}")
    latex.append("\\begin{tabular}{llrr}")
    latex.append("\\toprule")
    latex.append("& \\textbf{Operator} & \\textbf{Gain} & \\textbf{Share} \\\\")
    latex.append("\\midrule")

    for idx, row in top_df.iterrows():
        pct = (float(row["gain"]) / total_gain) * 100.0
        name = str(row["name"])
        name_str = f"\\textbf{{{name}}}" if idx < 2 else name
        latex.append(f"{row['group']} & {name_str} & {format_gain_millions(float(row['gain']))} & {pct:.1f}\\% \\\\")

    if not other_df.empty:
        other_gain = float(other_df["gain"].sum())
        other_pct = (other_gain / total_gain) * 100.0
        other_label = f"Other ({len(other_df)} operators)"
        latex.append(
            f"& \\textit{{{other_label}}} & \\textit{{{format_gain_millions(other_gain)}}} & \\textit{{{other_pct:.1f}\\%}} \\\\"
        )

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    save_latex_table("\n".join(latex), "table_5_operators.tex", output_dir)

def generate_operator_table(df: pd.DataFrame, output_dir: Path):
    """Table 5: Operator Forensics (Top Contributors)"""
    print("\n--- Generating Table 5: Operator Forensics ---")

    algo_stats, _ = collect_operator_stats(df)
    if not algo_stats:
        print("  No operator gain data found for operator table.")
        return

    preferred_algos = ["MA", "GA", "ALNS", "ALNS_nopolish", "ALNS(+polish)"]
    chosen_algo = next((a for a in preferred_algos if a in algo_stats), None)
    if chosen_algo is None:
        chosen_algo = max(algo_stats.keys(), key=lambda a: sum(algo_stats[a].values()))

    rows = []
    for key, gain in algo_stats[chosen_algo].items():
        group, name = operator_key_meta(key)
        rows.append({"key": key, "group": group, "name": name, "gain": gain})

    if not rows:
        return

    df_ops = pd.DataFrame(rows).sort_values("gain", ascending=False).reset_index(drop=True)
    total_gain = float(df_ops["gain"].sum()) or 1.0

    top_n = 10
    top_df = df_ops.iloc[:top_n]
    other_df = df_ops.iloc[top_n:]

    latex = []
    latex.append("\\begin{table}[t]")
    latex.append("\\centering")
    latex.append(
        f"\\caption{{Operator Contributions ({chosen_algo}). Type: D=Destroy, R=Repair, LS=Local Search, M=Mutation.}}"
    )
    latex.append("\\label{tab:operators}")
    latex.append("\\begin{tabular}{llrr}")
    latex.append("\\toprule")
    latex.append("\\textbf{Type} & \\textbf{Operator} & \\textbf{Gain} & \\textbf{Share} \\\\")
    latex.append("\\midrule")

    for idx, row in top_df.iterrows():
        pct = (float(row["gain"]) / total_gain) * 100.0
        name = str(row["name"])
        name_str = f"\\textbf{{{name}}}" if idx < 2 else name
        latex.append(f"{row['group']} & {name_str} & {format_operator_gain(float(row['gain']))} & {pct:.1f}\\% \\\\")

    if not other_df.empty:
        other_gain = float(other_df["gain"].sum())
        other_pct = (other_gain / total_gain) * 100.0
        other_label = f"Other ({len(other_df)} operators)"
        latex.append(
            f"& \\textit{{{other_label}}} & \\textit{{{format_operator_gain(other_gain)}}} & \\textit{{{other_pct:.1f}\\%}} \\\\"
        )

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    save_latex_table("\n".join(latex), "table_5_operators.tex", output_dir)

def generate_operator_comparison_table(df: pd.DataFrame, output_dir: Path):
    """Operator Forensics comparison across algorithms"""
    print("\n--- Generating Table 5: Operator Comparison ---")

    algo_stats, _ = collect_operator_stats(df)
    if not algo_stats:
        print("  No operator gain data found for comparison table.")
        return

    found_algos = order_algorithms(algo_stats.keys())
    all_keys = set()
    for ops in algo_stats.values():
        all_keys.update(ops.keys())

    if not all_keys:
        print("  No operator keys found for comparison table.")
        return

    group_order = {"D": 0, "R": 1, "LS": 2, "M": 3}
    def sort_key(key):
        group, name = operator_key_meta(key)
        return (group_order.get(group, 99), name)

    sorted_keys = sorted(all_keys, key=sort_key)

    latex = []
    latex.append("\\begin{table*}[t]")
    latex.append("\\centering")
    latex.append("\\caption{Operator Gain Contributions (Cumulative). Comparison across algorithms.}")
    latex.append("\\label{tab:operators_comparison}")

    col_def = "l l " + "r " * len(found_algos)
    header = " & ".join([f"\\textbf{{{a}}}" for a in found_algos])

    latex.append(f"\\begin{{tabular}}{{{col_def}}}")
    latex.append("\\toprule")
    latex.append(f"\\textbf{{Type}} & \\textbf{{Operator}} & {header} \\\\")
    latex.append("\\midrule")

    current_group = None
    for key in sorted_keys:
        group, name = operator_key_meta(key)
        if current_group is not None and group != current_group:
            latex.append("\\addlinespace")
        current_group = group

        row_str = f"{group} & {name}"
        for algo in found_algos:
            row_str += f" & {format_operator_gain(algo_stats[algo].get(key, 0.0))}"
        latex.append(row_str + " \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table*}")

    save_latex_table("\n".join(latex), "table_5_operators_comparison.tex", output_dir)

# ==============================================================================
# RESULTS FILE GENERATOR
# ==============================================================================

def generate_example_solutions_file(results_dir: Path, df_summary: pd.DataFrame, output_dir: Path):
    """Example solutions per scenario/algorithm (simulation data only)."""
    print("\n--- Generating Example Solutions File ---")
    if df_summary.empty:
        print("  No summary data available for example solutions.")
        return

    required_cols = {"scenario", "fleet", "algorithm", "run"}
    missing_cols = required_cols - set(df_summary.columns)
    if missing_cols:
        print(f"  Missing required columns: {sorted(missing_cols)}")
        return

    metric_preference = ["cost_fitness", "avg_evac_time", "makespan", "runtime"]
    seed_raw = os.getenv("EXAMPLE_RUN_SEED")
    seed = None
    if seed_raw is not None:
        try:
            seed = int(seed_raw)
        except Exception:
            seed = None

    algo_filter = {"MA", "ALNS"}
    df_filtered = df_summary[df_summary["algorithm"].isin(algo_filter)].copy()
    if df_filtered.empty:
        print(f"  No runs found for algorithms: {sorted(algo_filter)}")
        return

    examples = []
    missing_runs = 0

    grouped = df_filtered.groupby(["scenario", "fleet", "algorithm"], dropna=False)
    for (scenario, fleet, algorithm), group in grouped:
        group_valid = group[group["run"].notnull()].copy()
        if group_valid.empty:
            continue
        # Pick a random run per scenario/fleet/algorithm (seedable via EXAMPLE_RUN_SEED).
        best_row = group_valid.sample(n=1, random_state=seed).iloc[0]

        run_raw = best_row.get("run")
        if pd.isna(run_raw):
            continue
        try:
            run_id = int(run_raw)
        except Exception:
            continue

        run_dir = results_dir / str(scenario) / str(fleet) / str(algorithm)
        run_path = run_dir / f"run_{run_id}.json"
        if not run_path.exists():
            candidates = sorted(run_dir.glob("run_*.json"), key=lambda p: p.name)
            if not candidates:
                missing_runs += 1
                continue
            run_path = candidates[0]
            try:
                run_id = int(run_path.stem.split("_", 1)[1])
            except Exception:
                pass

        try:
            with open(run_path, "r") as f:
                run_data = json.load(f)
        except Exception:
            missing_runs += 1
            continue

        run_timestamp = run_data.get("timestamp")
        if not run_timestamp:
            try:
                run_timestamp = datetime.fromtimestamp(run_path.stat().st_mtime, tz=timezone.utc).isoformat()
            except Exception:
                run_timestamp = None

        try:
            rel_path = str(run_path.relative_to(results_dir))
        except Exception:
            rel_path = str(run_path)

        examples.append({
            "scenario": str(scenario),
            "fleet": str(fleet),
            "fleet_label": _json_safe_value(best_row.get("fleet_label")),
            "algorithm": str(algorithm),
            "run": run_id,
            "run_timestamp_utc": run_timestamp,
            "run_path": rel_path,
            "simulation_data": run_data.get("simulation_data"),
        })

    output = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_results_dir": str(results_dir.resolve()),
        "examples": examples,
    }

    output_path = output_dir / "example_solutions.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=True)

    print(f"  Saved Example Solutions: {output_path.name} ({len(examples)} entries)")
    if missing_runs:
        print(f"  Missing run files: {missing_runs}")

def generate_vehicle_type_trip_stats(results_dir: Path, output_dir: Path):
    """Aggregate trips/stops per vehicle type from run JSONs."""
    print("\n--- Generating Vehicle Type Trip Stats ---")
    df_runs = collect_vehicle_type_trip_stats(results_dir)
    if df_runs.empty:
        print("  No vehicle type trip stats found (missing run files or vehicles).")
        return pd.DataFrame()

    agg = df_runs.groupby(
        ["scenario", "fleet", "algorithm", "vehicle_type"],
        dropna=False
    ).agg(
        runs=("trip_count", "size"),
        trip_count_mean=("trip_count", "mean"),
        stop_count_mean=("stop_count", "mean"),
        trips_per_stop_mean=("trips_per_stop", "mean"),
        stops_per_trip_mean=("stops_per_trip", "mean"),
        fullness_percent_mean=("fullness_percent", "mean"),
    ).reset_index()

    agg = agg.sort_values(["scenario", "fleet", "algorithm", "vehicle_type"])

    entries = []
    for _, row in agg.iterrows():
        entries.append({
            "scenario": _json_safe_value(row["scenario"]),
            "fleet": _json_safe_value(row["fleet"]),
            "fleet_label": _json_safe_value(FLEET_LABELS.get(row["fleet"], row["fleet"])),
            "algorithm": _json_safe_value(row["algorithm"]),
            "vehicle_type": _json_safe_value(row["vehicle_type"]),
            "runs": _json_safe_value(row["runs"]),
            "mean_trip_count": _json_safe_value(row["trip_count_mean"]),
            "mean_stop_count": _json_safe_value(row["stop_count_mean"]),
            "mean_trips_per_stop": _json_safe_value(row["trips_per_stop_mean"]),
            "mean_stops_per_trip": _json_safe_value(row["stops_per_trip_mean"]),
            "mean_fullness_percent": _json_safe_value(row["fullness_percent_mean"]),
        })

    output = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_results_dir": str(results_dir.resolve()),
        "entries": entries,
    }

    output_path = output_dir / "vehicle_type_trip_stats.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=True)

    print(f"  Saved Vehicle Type Trip Stats: {output_path.name} ({len(entries)} entries)")
    return agg

def generate_vehicle_type_trip_stats_table(df_vehicle: pd.DataFrame, output_dir: Path):
    """Table 6: Vehicle type trip activity."""
    print("\n--- Generating Table 6: Vehicle Type Trip Activity ---")
    if df_vehicle is None or df_vehicle.empty:
        print("  No vehicle type trip stats available for the table.")
        return

    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{Vehicle-category trip activity (means per run).}")
    latex.append("\\label{tab:vehicle_type_trips}")
    latex.append("\\begin{tabular}{lllcccc}")
    latex.append("\\toprule")
    latex.append("\\textbf{Scenario} & \\textbf{Method} & \\textbf{Vehicle} & \\textbf{Trips} & \\textbf{Stops} & \\textbf{Stops/Trip} & \\textbf{Fullness/Trip (\\%)} \\\\")
    latex.append("\\midrule")

    scenarios = sorted(df_vehicle["scenario"].dropna().unique(), key=str)
    for scen in scenarios:
        scen_df = df_vehicle[df_vehicle["scenario"] == scen]
        fleets = sorted(scen_df["fleet"].dropna().unique(), key=str)
        for fleet in fleets:
            fleet_label = FLEET_LABELS.get(fleet, fleet)
            label = _scenario_fleet_display(scen, fleet_label)
            latex.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{label}}}}} \\\\")

            sub = scen_df[scen_df["fleet"] == fleet]
            algos_present = order_algorithms(sub["algorithm"].unique())
            for algo in algos_present:
                algo_sub = sub[sub["algorithm"] == algo]
                vehicle_types = sorted(algo_sub["vehicle_type"].dropna().unique(), key=str)
                for idx, vehicle_type in enumerate(vehicle_types):
                    row = algo_sub[algo_sub["vehicle_type"] == vehicle_type].iloc[0]
                    trips = row["trip_count_mean"]
                    stops = row["stop_count_mean"]
                    stops_per_trip = row["stops_per_trip_mean"]
                    fullness = row.get("fullness_percent_mean")

                    trips_str = f"{trips:.1f}" if pd.notna(trips) else "--"
                    stops_str = f"{stops:.1f}" if pd.notna(stops) else "--"
                    stops_per_trip_str = f"{stops_per_trip:.2f}" if pd.notna(stops_per_trip) else "--"
                    fullness_str = f"{fullness:.1f}" if pd.notna(fullness) else "--"

                    algo_label = algo if idx == 0 else ""
                    latex.append(
                        f" & {algo_label} & {vehicle_type} & {trips_str} & {stops_str} & {stops_per_trip_str} & {fullness_str} \\\\"
                    )
            latex.append("\\addlinespace")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    save_latex_table("\n".join(latex), "table_6_vehicle_type_trips.tex", output_dir)

# ==============================================================================
# PLOT GENERATORS
# ==============================================================================


def generate_memetic_focus_plot(results_dir: Path, output_dir: Path):
    """
    Generates a specialized plot focusing ONLY on Memetic (Local Search) operators.
    Shows the relative importance of specific heuristics within the LS phase.
    FIXED: Correctly parses CSV column names ending in '_gain'.
    """
    print("\n--- Generating Memetic-Only (LS) Forensics Plot ---")

    all_runs = []
    max_gen_observed = 0
    
    # 1. Identify valid LS keys from metadata
    ls_keys = {k for k, v in OPERATOR_META.items() if v[0] == "LS"}
    if not ls_keys:
        print("  ⚠️ No LS operators defined in metadata.")
        return

    # 2. Collect Data
    for scen_dir in results_dir.iterdir():
        if not scen_dir.is_dir() or scen_dir.name.startswith("paper_") or scen_dir.name == "analysis_output":
            continue

        for fleet_dir in scen_dir.iterdir():
            ma_dir = fleet_dir / "MA"
            if not ma_dir.exists():
                continue

            local_dfs = []
            for run_file in ma_dir.glob("**/operator_history.csv"):
                try:
                    df = pd.read_csv(run_file)
                    
                    # Identify relevant columns dynamically
                    # We want columns that end in '_gain' where the prefix is a known LS operator
                    target_cols = []
                    rename_map = {}
                    
                    for col in df.columns:
                        if col.endswith("_gain"):
                            key = col[:-5] # Strip '_gain' suffix
                            if key in ls_keys:
                                target_cols.append(col)
                                rename_map[col] = key
                    
                    if not target_cols:
                        continue

                    # Calculate marginal gains (Current Gen - Prev Gen)
                    # fillna(0) ensures Gen 0 isn't treated as a massive spike
                    df_marginal = df[target_cols].diff().fillna(0).clip(lower=0)
                    
                    # Rename columns to clean keys (e.g., 'intra_trip_gain' -> 'intra_trip')
                    df_ls = df_marginal.rename(columns=rename_map)
                    
                    # Normalize against LS TOTAL for this generation
                    # (This highlights "Market Share" of each operator)
                    total_ls_gain = df_ls.sum(axis=1)
                    
                    # Filter out rows with zero LS activity to avoid noise/NaNs
                    # We replace 0 sums with 1.0 to avoid division by zero, resulting in 0 contribution
                    safe_total = total_ls_gain.replace(0, 1.0)
                    
                    df_norm = df_ls.div(safe_total, axis=0)
                    df_norm['Generation'] = df['Generation']
                    
                    local_dfs.append(df_norm)
                    max_gen_observed = max(max_gen_observed, df['Generation'].max())
                        
                except Exception as e:
                    print(f"    ⚠️ Error reading {run_file}: {e}")
                    continue
            
            # Average for this scenario
            if local_dfs:
                df_scen_avg = pd.concat(local_dfs).groupby('Generation').mean().reset_index()
                all_runs.append(df_scen_avg)

    if not all_runs:
        print("  ⚠️ No data found for LS plot (Check CSV column names vs OPERATOR_META).")
        return

    # 3. Aggregate Global Average
    df_concat = pd.concat(all_runs)
    df_global = df_concat.groupby('Generation').mean().reset_index()

    # 4. Identify Top LS Contributors
    total_gains = df_global.drop('Generation', axis=1).sum().sort_values(ascending=False)
    
    TOP_N = 6 
    top_ops = total_gains.head(TOP_N).index.tolist()
    
    plot_data = df_global[['Generation'] + top_ops].copy()
    
    # Bundle rest into "Other LS"
    other_ops = total_gains.iloc[TOP_N:].index.tolist()
    if other_ops:
        plot_data['Other LS'] = df_global[other_ops].sum(axis=1)
        top_ops.append('Other LS')

    # 5. Smoothing
    # Use a larger window because LS usage can be intermittent
    WINDOW_SIZE = max(3, int(max_gen_observed * 0.1))
    
    plot_data_smoothed = plot_data.rolling(window=WINDOW_SIZE, min_periods=1).mean()
    plot_data_smoothed['Generation'] = plot_data['Generation']
    
    # Cutoff start (Warmup phase often has chaotic operator usage)
    START_GEN = 5
    plot_data_smoothed = plot_data_smoothed[plot_data_smoothed['Generation'] >= START_GEN]

    # 6. Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use a distinct palette for LS
    colors = sns.color_palette("flare", n_colors=len(top_ops))
    if 'Other LS' in top_ops:
        colors[-1] = (0.8, 0.8, 0.8)

    labels = []
    for col in top_ops:
        if col == "Other LS":
            labels.append("Other LS")
            continue
        # Use pretty name from meta
        if col in OPERATOR_META:
            labels.append(OPERATOR_META[col][1]) 
        else:
            labels.append(col)

    ax.stackplot(
        plot_data_smoothed['Generation'],
        [plot_data_smoothed[col] for col in top_ops],
        labels=labels,
        colors=colors,
        alpha=0.9
    )

    plt.title("Memetic Forensics: Local Search Operator Share\n(Normalized: Relative Importance within LS Phase)", fontsize=14)
    plt.ylabel("Share of Local Search Improvement")
    plt.xlabel("Generation")
    plt.xlim(START_GEN, max_gen_observed)
    plt.ylim(0, 1.0) # Normalized plot always 0-1
    plt.margins(0, 0)
    
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title="LS Heuristics")
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    save_plot(fig, "fig_memetic_only_breakdown.png", output_dir)

def generate_operator_evolution_plot(results_dir: Path, output_dir: Path):
    """
    Generates a Stacked Area Chart showing how operator influence changes over generations.
    - Aggregates data from all 'operator_history.csv' files in a Scenario/Fleet folder.
    - Calculates Marginal Gain (Gain per Gen) instead of Cumulative.
    - Smooths the data to show trends clearly.
    """
    print("\n--- Generating Operator Evolution Plots ---")

    # Iterate over Scenario -> Fleet -> Algorithm (MA only)
    for scen_dir in results_dir.iterdir():
        if not scen_dir.is_dir() or scen_dir.name.startswith("paper_") or scen_dir.name == "analysis_output":
            continue

        for fleet_dir in scen_dir.iterdir():
            if not fleet_dir.is_dir():
                continue

            # We only care about MA (Memetic Algorithm) for operator history
            ma_dir = fleet_dir / "MA"
            if not ma_dir.exists():
                continue

            print(f"  Processing: {scen_dir.name} - {fleet_dir.name}")

            # 1. Load all CSVs for this config
            dfs = []
            for run_file in ma_dir.glob("**/operator_history.csv"):
                try:
                    df = pd.read_csv(run_file)
                    gen_col = df['Generation']
                    gain_cols = [c for c in df.columns if c.endswith('_gain')]
                    
                    # Calculate diff (Current - Previous) to get gain *this* generation
                    df_marginal = df[gain_cols].diff().fillna(df[gain_cols].iloc[0])
                    df_marginal = df_marginal.clip(lower=0)
                    df_marginal['Generation'] = gen_col
                    dfs.append(df_marginal)
                except Exception as e:
                    print(f"    ⚠️ Error reading {run_file.name}: {e}")

            if not dfs:
                continue

            # 2. Aggregate across runs (Mean)
            df_concat = pd.concat(dfs)
            df_mean = df_concat.groupby('Generation').mean().reset_index()
            
            # Determine max generation for this specific scenario
            local_max_gen = df_mean['Generation'].max()

            # 3. Identify Top Operators
            total_gains = df_mean.drop('Generation', axis=1).sum().sort_values(ascending=False)
            TOP_N = 7
            top_ops = total_gains.head(TOP_N).index.tolist()
            plot_data = df_mean[['Generation'] + top_ops].copy()
            
            other_ops = total_gains.iloc[TOP_N:].index.tolist()
            if other_ops:
                plot_data['Others'] = df_mean[other_ops].sum(axis=1)
                top_ops.append('Others')

            # 4. Smoothing
            # Use local_max_gen here instead of the undefined global variable
            WINDOW_SIZE = max(1, int(local_max_gen * 0.05))
            plot_data_smoothed = plot_data.rolling(window=WINDOW_SIZE, min_periods=1).mean()
            plot_data_smoothed['Generation'] = plot_data['Generation']

            # --- CUTOFF: Start at Gen 5 ---
            START_GEN = 5
            plot_data_smoothed = plot_data_smoothed[plot_data_smoothed['Generation'] >= START_GEN]
            
            if plot_data_smoothed.empty:
                print("    ⚠️ Not enough generations to plot evolution.")
                continue
            # ------------------------------

            # 5. Plotting
            fig, ax = plt.subplots(figsize=(12, 7))
            
            colors = sns.color_palette("viridis", n_colors=len(top_ops))
            if 'Others' in top_ops:
                colors[-1] = (0.7, 0.7, 0.7)

            labels = []
            for col in top_ops:
                if col == "Others": 
                    labels.append("Others")
                    continue
                clean_name = col.replace("_gain", "").replace("mutate_", "M: ").replace("_", " ").title()
                if "Ma" in clean_name: clean_name = clean_name.replace("Ma", "MA")
                labels.append(clean_name)

            ax.stackplot(
                plot_data_smoothed['Generation'],
                [plot_data_smoothed[col] for col in top_ops],
                labels=labels,
                colors=colors,
                alpha=0.85
            )

            # --- CRITICAL FIX: Force Y-Limit based on visible data ---
            stack_heights = plot_data_smoothed[top_ops].sum(axis=1)
            max_visible_y = stack_heights.max()
            if max_visible_y > 0:
                plt.ylim(0, max_visible_y * 1.05)
            # ---------------------------------------------------------

            fleet_label = FLEET_LABELS.get(fleet_dir.name, fleet_dir.name)
            label = _scenario_fleet_display(scen_dir.name, fleet_label)
            plt.title(f"Evolution of Operator Influence (Smoothed)\n{label}", fontsize=14)
            plt.ylabel("Average Fitness Gain per Generation")
            plt.xlabel("Generation")
            plt.xlim(START_GEN, local_max_gen)
            plt.margins(0, 0)
            
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Operators")
            plt.grid(axis='y', alpha=0.3, linestyle='--')
            
            safe_name = f"fig_op_evolution_{_clean_scenario_name(scen_dir.name)}_{fleet_dir.name}.png".replace(" ", "_")
            save_plot(fig, safe_name, output_dir)

def generate_convergence_plot(
    metric_id: str, 
    scenario_name: str,
    fleet_name: str,
    base_path: Path, 
    output_dir: Path, 
    df_summary: pd.DataFrame
):
    """
    Generates convergence plot with Time (seconds) on X-axis.
    Interpolates runs onto common time grid.
    """
    config = METRIC_CONFIG[metric_id]
    json_key = config["json_key"]
    ylabel = config["label"]
    
    if json_key is None or not base_path.exists():
        return

    # Collect raw data per run
    raw_run_data = []
    max_time_observed = 0.0

    for algo_dir in base_path.iterdir():
        if not algo_dir.is_dir():
            continue
        algo_name = algo_dir.name
        
        if _is_baseline_algo(algo_name):
            continue
        
        for run_file in algo_dir.glob("run_*.json"):
            try:
                with open(run_file, 'r') as f:
                    data = json.load(f)
                
                stats = data.get("algorithm_stats", {})
                progress = stats.get("progress", [])
                if isinstance(progress, list) and progress:
                    if metric_id == "cost":
                        pairs = [(p.get("elapsed_seconds"), p.get("best_cost")) for p in progress]
                    elif metric_id == "makespan":
                        pairs = [(p.get("elapsed_seconds"), p.get("makespan_min")) for p in progress]
                    elif metric_id == "wait_time":
                        pairs = [(p.get("elapsed_seconds"), p.get("avg_wait_min")) for p in progress]
                    else:
                        pairs = []
                    pairs = [(t, v) for t, v in pairs if t is not None and v is not None]
                    if not pairs:
                        continue
                    times, values = zip(*pairs)
                    times = np.array(times)
                    values = np.array(values)
                    max_time_observed = max(max_time_observed, times[-1])
                    raw_run_data.append((times, values, algo_name))
                    continue

                values = stats.get(json_key, [])
                
                # Fallback for cost metric
                if not values and metric_id == "cost" and "history_fitness" in stats:
                    values = stats["history_fitness"]

                gen_durations = stats.get("generation_times", [])
                
                if values and gen_durations:
                    min_len = min(len(values), len(gen_durations))
                    values = values[:min_len]
                    gen_durations = gen_durations[:min_len]
                    
                    times = np.cumsum(gen_durations)
                    max_time_observed = max(max_time_observed, times[-1])
                    raw_run_data.append((times, values, algo_name))

            except Exception:
                continue

    if not raw_run_data:
        return

    # Normalize to common time grid
    common_time_grid = np.linspace(0, max_time_observed, num=500)
    plot_rows = []

    for times, values, algo_name in raw_run_data:
        indices = np.searchsorted(times, common_time_grid, side='right') - 1
        indices = np.clip(indices, 0, len(values) - 1)
        interpolated_values = np.array(values)[indices]
        
        for t, v in zip(common_time_grid, interpolated_values):
            plot_rows.append({
                "Time (s)": t,
                "Value": v,
                "Algorithm": algo_name
            })

    df_conv = pd.DataFrame(plot_rows)

    # Get baseline (Dispatcher/Shuttle)
    baseline_rows = df_summary[
        (df_summary['scenario'] == scenario_name) & 
        (df_summary['fleet'] == fleet_name)
    ]
    baseline_rows = baseline_rows[baseline_rows["algorithm"].apply(_is_baseline_algo)]
    
    baseline_val = None
    baseline_algo = None
    if not baseline_rows.empty:
        baseline_rows = baseline_rows.assign(
            _base=baseline_rows["algorithm"].apply(_algo_base_name)
        )
        if "Dispatcher" in baseline_rows["_base"].values:
            preferred = baseline_rows[baseline_rows["_base"] == "Dispatcher"]
        else:
            preferred = baseline_rows
        baseline_algo = str(preferred["algorithm"].iloc[0])
        target_col = config["summary_col"]
        if target_col in preferred.columns:
            baseline_val = preferred[preferred["algorithm"] == baseline_algo][target_col].mean()

    # Plot
    plt.figure(figsize=(8, 5))
    
    algo_order = order_algorithms(df_conv["Algorithm"].unique())
    sns.lineplot(
        data=df_conv, 
        x="Time (s)", 
        y="Value", 
        hue="Algorithm",
        style="Algorithm",
        hue_order=algo_order,
        style_order=algo_order,
        palette="viridis",
        linewidth=2
    )

    if baseline_val is not None and not np.isnan(baseline_val):
        plt.axhline(
            y=baseline_val, 
            color='red', 
            linestyle='--', 
            linewidth=1.5, 
            label=f"{baseline_algo} ({baseline_val:.1f})" if baseline_algo else f"Baseline ({baseline_val:.1f})"
        )

    fleet_label = FLEET_LABELS.get(fleet_name, fleet_name)
    label = _scenario_fleet_display(scenario_name, fleet_label)
    plt.title(f"{ylabel}\n{label}")
    plt.ylabel(ylabel)
    plt.xlabel("Runtime (seconds)")
    plt.xlim(0, max_time_observed)
    
    if config["log_scale"]:
        plt.yscale('log')
        
    plt.legend()
    plt.grid(True, which="major", ls="-", alpha=0.3)
    
    safe_scen = scenario_name.replace(" ", "_")
    safe_fleet = fleet_name.replace(" ", "_")
    
    save_plot(plt.gcf(), f"fig_conv_{safe_scen}_{safe_fleet}_{metric_id}.png", output_dir)

def generate_efficiency_plot(df_summary: pd.DataFrame, output_dir: Path):
    """Stacked bar chart: Pure Driving Time vs Service Overhead"""
    print("\n--- Generating Efficiency Plot ---")
    
    # 1. Heuristic to pick a representative scenario
    target_scen = None
    # Prefer Default or Flood scenario if available
    for s in df_summary['scenario'].unique():
        if "Default" in s or "Flood" in s:
            target_scen = s
            break
    # Fallback to whatever is first
    if not target_scen and not df_summary.empty:
        target_scen = df_summary['scenario'].unique()[0]
        
    if not target_scen:
        return

    df_plot = df_summary[df_summary['scenario'] == target_scen].copy()
    
    # 2. Constants (Must match SHARED_EXECUTION_PARAMS in experiment runner)
    FIXED_TIME_BASE_MIN = 3.0         # 3.0 mins per stop (pickup or dropoff)
    TIME_PER_PAX_MIN = 20.0 / 60.0    # ~0.33 mins per person
    
    # Estimated Total Passengers (Standard Scenario ~1348)
    # We assume successful runs evacuated everyone.
    TOTAL_PAX_EST = 1348 

    # 3. Calculate Service Overhead
    def calc_overhead(row):
        # A. Stop Count Logic
        # Try to use 'stop_count' if it exists (from new experiment runner).
        # Fallback: Use 'trip_count' (Assume 1 pickup per trip minimum) if column missing.
        if 'stop_count' in row.index and pd.notnull(row['stop_count']):
            stops = row['stop_count']
        else:
            # Conservative fallback: Assume stops = trip_count 
            # (In reality, stops > trips for multi-stop routes)
            stops = row['trip_count']
            
        trips = row['trip_count']
        
        # B. Fixed Overhead Calculation
        # Applied at every Pickup (Stop) AND every Dropoff (End of Trip)
        total_operations = stops + trips
        fixed_overhead = total_operations * FIXED_TIME_BASE_MIN
        
        # C. Variable Overhead Calculation
        # Loading Time (Pickups) + Offloading Time (Dropoffs)
        # Total Pax * 0.33 * 2
        variable_overhead = (TOTAL_PAX_EST * TIME_PER_PAX_MIN) * 2
        
        return fixed_overhead + variable_overhead

    df_plot['service_overhead'] = df_plot.apply(calc_overhead, axis=1)
    
    # 4. Prepare Plot Data
    stacked_rows = []
    for _, row in df_plot.iterrows():
        # Sanity check to avoid negative driving times if overhead est > total time
        # (Can happen if using old data with approximated overheads)
        drive_time = row['total_travel_time']
        
        stacked_rows.append({
            "Algorithm": row['algorithm'],
            "Fleet": row['fleet_label'],
            "Pure Driving": max(0, drive_time),
            "Service Overhead": row['service_overhead']
        })
        
    df_stack = pd.DataFrame(stacked_rows)
    df_agg = df_stack.groupby(['Algorithm', 'Fleet'])[['Pure Driving', 'Service Overhead']].mean().reset_index()
    
    # Sort Logic: Dispatcher/Shuttle -> GA -> MA -> ALNS
    algo_order = order_algorithms(df_agg['Algorithm'].unique())
    df_agg['Algorithm'] = pd.Categorical(
        df_agg['Algorithm'],
        categories=algo_order,
        ordered=True
    )
    df_agg = df_agg.sort_values(['Fleet', 'Algorithm'])
    
    # Create Labels for X-Axis
    df_agg['Label'] = df_agg['Fleet'] + "\n" + df_agg['Algorithm'].astype(str)
    
    plot_data = df_agg.set_index('Label')[['Pure Driving', 'Service Overhead']]

    # 5. Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Colors: Driving = Dark Blue/Grey, Service = Light Grey
    colors = ['#34495e', '#bdc3c7'] 
    
    plot_data.plot(
        kind='bar', 
        stacked=True, 
        ax=ax,
        color=colors,
        width=0.75,
        edgecolor='black',
        linewidth=0.5
    )
    
    # Add Value Labels
    for c in ax.containers:
        # Filter out 0 labels
        labels = [f'{v:.0f}' if v > 0 else '' for v in c.datavalues]
        ax.bar_label(c, labels=labels, label_type='center', color='white', weight='bold', fontsize=9)

    clean_title = target_scen.replace("FloodWilhelmsburg", "Flood").replace("BombThreat", "Bomb")
    plt.title(f"Efficiency Breakdown: {clean_title}", fontsize=14)
    plt.ylabel("Cumulative Fleet Minutes", fontsize=12)
    plt.xlabel("")
    plt.xticks(rotation=0, fontsize=10)
    plt.legend(title="Time Component", loc='upper right', frameon=True)
    plt.grid(axis='y', alpha=0.3, linestyle='--')

    save_plot(fig, "fig_efficiency_breakdown.png", output_dir)

def generate_robustness_boxplot(df_summary: pd.DataFrame, output_dir: Path):
    """Boxplot showing performance distribution across runs"""
    print("\n--- Generating Robustness Boxplot ---")
    
    target_scen = 'FloodWilhelmsburg'
    if target_scen not in df_summary['scenario'].unique():
        target_scen = df_summary['scenario'].unique()[0] if not df_summary.empty else None
        
    if not target_scen:
        return

    df_box = df_summary[df_summary['scenario'] == target_scen]
    
    fig = plt.figure(figsize=(10, 6))
    hue_order = order_algorithms(df_box['algorithm'].unique())
    sns.boxplot(
        data=df_box,
        x='fleet_label',
        y='avg_evac_time',
        hue='algorithm',
        hue_order=hue_order,
        palette="viridis"
    )
    plt.title(f"Performance Distribution ({target_scen})")
    plt.ylabel("Avg. Evacuation Time (min)")
    plt.xlabel("Fleet Composition")
    plt.legend(title="Algorithm")
    
    save_plot(fig, "fig_robustness.png", output_dir)

def generate_operator_impact_plot(df_summary: pd.DataFrame, output_dir: Path):
    """Bar chart showing cumulative impact of memetic operators"""
    print("\n--- Generating Operator Impact Plot ---")
    
    df_ma = df_summary[df_summary['algorithm'] == 'MA']
    if df_ma.empty:
        print("  ⚠️ No MA data for operator plot")
        return

    op_cols = [c for c in df_ma.columns if c.startswith("op_gain_") and not c.endswith("_cnt")]
    if not op_cols:
        return
    
    filtered_cols = []
    for col in op_cols:
        key = col.replace("op_gain_", "", 1)
        if key in EXCLUDED_OPERATOR_KEYS:
            continue
        filtered_cols.append(col)
    if not filtered_cols:
        return

    total_gains = df_ma[filtered_cols].sum().sort_values(ascending=True)

    def pretty_label(op_gain_col: str) -> str:
        key = op_gain_col.replace("op_gain_", "", 1)
        group, name = OPERATOR_META.get(
            key,
            ("M" if key.startswith("mutate_") else "LS", key.replace("mutate_", "", 1).replace("_", " ").title()),
        )
        return f"{group}: {name}"

    names = [pretty_label(c) for c in total_gains.index]
    
    fig = plt.figure(figsize=(8, 5))
    plt.barh(names, total_gains.values, color=sns.color_palette("magma", len(names)))
    plt.xscale('log')
    plt.xlabel("Cumulative Fitness Improvement (Log)")
    plt.title("Memetic Operator Impact (Global)")
    
    save_plot(fig, "fig_operators.png", output_dir)

# ==============================================================================
# MAIN ANALYSIS FUNCTION
# ==============================================================================

def analyze_results(results_dir: Path):
    """Main analysis orchestrator"""
    summary_file = results_dir / "all_runs_summary.json"
    
    output_dir = results_dir / "analysis_output"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    df = load_summary_data(results_dir)
    if df.empty:
        print("❌ Summary file is empty or missing.")
        return

    # ========================================
    # GENERATE ALL 5 TABLES + EXAMPLE SOLUTIONS
    # ========================================
    print("\n" + "="*70)
    print("GENERATING TABLES AND EXAMPLE SOLUTIONS")
    print("="*70)
    
    generate_performance_table(df, output_dir)
    generate_efficiency_table(df, output_dir)
    generate_robustness_table(df, output_dir)
    # generate_convergence_table(results_dir, output_dir)
    generate_operator_table(df, output_dir)
    generate_operator_comparison_table(df, output_dir)
    generate_example_solutions_file(results_dir, df, output_dir)
    vehicle_type_stats = generate_vehicle_type_trip_stats(results_dir, output_dir)
    generate_vehicle_type_trip_stats_table(vehicle_type_stats, output_dir)

    # ========================================
    # GENERATE CONVERGENCE PLOTS (3 per scenario/fleet)
    # ========================================
    print("\n" + "="*70)
    print("GENERATING CONVERGENCE PLOTS (WITH RUNTIME INTERPOLATION)")
    print("="*70)
    
    # Find all scenario/fleet combinations by scanning directory structure
    found_combinations = []
    for scen_dir in results_dir.iterdir():
        if scen_dir.is_dir() and scen_dir.name != "analysis_output" and not scen_dir.name.startswith('.'):
            for fleet_dir in scen_dir.iterdir():
                if fleet_dir.is_dir() and not fleet_dir.name.startswith('.'):
                    found_combinations.append((scen_dir.name, fleet_dir.name))

    print(f"\nFound {len(found_combinations)} scenario/fleet combinations")
    
    # Generate all 3 metrics for each combination
    for scenario, fleet in found_combinations:
        print(f"\n  Processing: {scenario} - {fleet}")
        fleet_path = results_dir / scenario / fleet
        
        for metric in ["cost", "makespan", "wait_time"]:
            print(f"    → {metric}")
            generate_convergence_plot(
                metric, scenario, fleet, fleet_path, output_dir, df
            )

    # ========================================
    # GENERATE OTHER PLOTS
    # ========================================
    print("\n" + "="*70)
    print("GENERATING ADDITIONAL PLOTS")
    print("="*70)
    
    generate_efficiency_plot(df, output_dir)
    generate_robustness_boxplot(df, output_dir)
    generate_operator_impact_plot(df, output_dir)
    generate_operator_evolution_plot(results_dir, output_dir)
    generate_global_operator_plot(results_dir, output_dir)  
    generate_memetic_focus_plot(results_dir, output_dir)
    print(f"\n{'='*70}")
    print(f"✅ Analysis Complete!")
    print(f"{'='*70}")
    print(f"Results saved to: {output_dir.resolve()}")
    print(f"\nGenerated Files:")
    print("  Example Solutions (example_solutions.json)")
    print("  Vehicle Type Trip Stats (vehicle_type_trip_stats.json)")
    print("  Vehicle Type Trip Table (table_6_vehicle_type_trips.tex)")
    print(f"  📄 6 LaTeX Tables (table_*.tex)")
    print(f"  📊 {len(found_combinations) * 3} Convergence Plots (fig_conv_*.png)")
    print(f"  📊 3 Additional Plots (fig_efficiency, fig_robustness, fig_operators)")

# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    script_dir = Path(__file__).parent
    base_outputs = script_dir / "outputs"
    
    # Handle different execution contexts
    if not base_outputs.exists():
        potential_path = Path("backend/experiments/outputs")
        if potential_path.exists():
            base_outputs = potential_path
        else:
            potential_path = Path("experiments/outputs")
            if potential_path.exists():
                base_outputs = potential_path

    print(f"🔍 Searching for experiments in: {base_outputs.resolve()}")
    latest_exp = get_latest_valid_experiment_dir(base_outputs)
    
    if latest_exp:
        print(f"✅ Found latest experiment: {latest_exp.name}")
        analyze_results(latest_exp)
    else:
        print("❌ No valid experiment results found.")
