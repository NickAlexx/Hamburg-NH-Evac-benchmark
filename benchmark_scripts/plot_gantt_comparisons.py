"""
Generate Gantt chart comparisons per scenario/fleet configuration.

Each chart overlays multiple algorithms on the same x-axis using a small y-offset.
Default comparison is MA_cr08_mr02 vs ALNS_nopolish (labeled as MA/ALNS).
"""

import argparse
import importlib
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DATA_DIR = REPO_ROOT / "benchmark_data"

FIG_WIDTH_IN = 3.4
BASE_FONT_SIZE = 7
TITLE_FONT_SIZE = 8
TIGHT_LAYOUT_PAD = 0.2
LOAD_LABEL_FONT_SIZE = 5
LEFT_MARGIN = 0.0
Y_TICK_PAD = 0

plt.rcParams.update(
    {
        "font.size": BASE_FONT_SIZE,
        "axes.labelsize": BASE_FONT_SIZE,
        "axes.titlesize": TITLE_FONT_SIZE,
        "axes.titleweight": "bold",
        "xtick.labelsize": BASE_FONT_SIZE,
        "ytick.labelsize": BASE_FONT_SIZE,
        "ytick.major.pad": Y_TICK_PAD,
    }
)

FLEET_LABELS = {
    "default": "Standard",
    "augmented": "Augmented",
    "specialized_only": "Specialized",
}

DEFAULT_ALGOS = ["MA_cr08_mr02", "ALNS_nopolish"]
DEFAULT_RUN_ID = 7
ALGO_LABELS = {
    "MA_cr08_mr02": "MA",
    "ALNS_nopolish": "ALNS",
}


def _safe_float(value):
    try:
        val = float(value)
    except Exception:
        return None
    if not math.isfinite(val):
        return None
    return val


def _safe_int(value):
    try:
        return int(value)
    except Exception:
        return None


def _safe_name(text):
    if text is None:
        return "unknown"
    return "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in str(text))

def _display_algo_name(algo: str) -> str:
    return ALGO_LABELS.get(str(algo), str(algo))


def _build_algo_styles(algo_order: List[str]) -> Dict[str, Dict]:
    algo_colors = {
        "MA": "#1f77b4",
        "ALNS": "#ff7f0e",
    }
    default_colors = plt.rcParams.get("axes.prop_cycle").by_key().get("color", ["#1f77b4"])
    styles: Dict[str, Dict] = {}
    for idx, algo in enumerate(algo_order):
        display_algo = _display_algo_name(algo)
        color = algo_colors.get(display_algo, default_colors[idx % len(default_colors)])
        if display_algo == "ALNS":
            styles[algo] = {
                "display": display_algo,
                "facecolor": "none",
                "edgecolor": color,
                "linewidth": 1.0,
                "linestyle": "--",
                "alpha": 1.0,
            }
        else:
            styles[algo] = {
                "display": display_algo,
                "facecolor": color,
                "edgecolor": color,
                "linewidth": 0.6,
                "linestyle": "-",
                "alpha": 0.35,
            }
    return styles


def _load_scenario_fleet_map() -> Dict[str, Dict]:
    candidates = [
        REPO_ROOT / "app" / "backend" / "app" / "evacuation" / "scenarios.py",
        Path.cwd() / "app" / "backend" / "app" / "evacuation" / "scenarios.py",
        Path.cwd() / "backend" / "app" / "evacuation" / "scenarios.py",
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


def _capacity_to_vehicle_type_map(fleet_spec) -> Dict[int, str]:
    mapping: Dict[int, set] = {}
    if not isinstance(fleet_spec, list):
        return {}
    for vehicle in fleet_spec:
        if not isinstance(vehicle, dict):
            continue
        cap = _safe_int(vehicle.get("capacity"))
        if cap is None:
            continue
        vtype = vehicle.get("type") or vehicle.get("vehicle_type")
        if not vtype:
            continue
        mapping.setdefault(cap, set()).add(str(vtype))
    resolved = {}
    for cap, types in mapping.items():
        if len(types) == 1:
            resolved[cap] = sorted(types)[0]
        else:
            resolved[cap] = f"capacity_{cap}"
    return resolved


def _max_small_vehicles() -> Optional[int]:
    raw = os.getenv("GANTT_SMALL_VEHICLES")
    if raw is None:
        return 3
    raw = raw.strip().lower()
    if raw == "all":
        return None
    try:
        value = int(raw)
    except Exception:
        return 3
    if value < 0:
        return None
    return value


def _get_vehicle_capacities(vehicles, fallback_capacity: Optional[int]) -> List[Optional[int]]:
    capacities: List[Optional[int]] = []
    if not isinstance(vehicles, list):
        return capacities
    for vehicle in vehicles:
        cap = None
        if isinstance(vehicle, dict):
            cap = _safe_int(vehicle.get("capacity"))
        if cap is None:
            cap = fallback_capacity
        capacities.append(cap)
    return capacities


def _select_vehicle_indices(
    vehicles,
    bus_capacity: Optional[int],
) -> Tuple[List[int], List[Optional[int]], Optional[int]]:
    capacities = _get_vehicle_capacities(vehicles, bus_capacity)
    if not capacities:
        return [], capacities, bus_capacity

    if bus_capacity is None:
        known_caps = [cap for cap in capacities if cap is not None]
        bus_capacity = max(known_caps) if known_caps else None

    if bus_capacity is None:
        return list(range(len(capacities))), capacities, None

    bus_indices = [i for i, cap in enumerate(capacities) if cap is not None and cap >= bus_capacity]
    small_by_capacity: Dict[int, List[int]] = {}
    for idx, cap in enumerate(capacities):
        if cap is None or cap >= bus_capacity:
            continue
        small_by_capacity.setdefault(cap, []).append(idx)

    max_small = _max_small_vehicles()
    selected_small: List[int] = []
    for cap in sorted(small_by_capacity):
        indices = small_by_capacity[cap]
        if max_small is None:
            selected_small.extend(indices)
        else:
            selected_small.extend(indices[:max_small])

    selected = sorted(set(bus_indices + selected_small))
    return selected, capacities, bus_capacity


def _vehicle_type_for_capacity(
    capacity: Optional[int],
    capacity_map: Dict[int, str],
    bus_capacity: Optional[int],
) -> str:
    if capacity is None:
        return "unknown"
    if capacity in capacity_map:
        return capacity_map[capacity]
    if bus_capacity is not None and capacity == bus_capacity:
        return "bus"
    return f"capacity_{capacity}"


def _build_vehicle_labels(
    selected_bus_indices: List[int],
    capacities: List[Optional[int]],
    capacity_map: Dict[int, str],
    bus_capacity: Optional[int],
) -> List[str]:
    labels = []
    for bus_idx in selected_bus_indices:
        cap = capacities[bus_idx] if bus_idx < len(capacities) else None
        vtype = _vehicle_type_for_capacity(cap, capacity_map, bus_capacity).replace("-", "")
        parts = [f"V{bus_idx}", vtype]
        if cap is not None:
            parts.append(f"c{cap}")
        labels.append("\n".join(parts))
    return labels


def _find_results_dir(base_outputs: Path, explicit: Optional[str]) -> Optional[Path]:
    if explicit:
        candidate = Path(explicit)
        if candidate.exists():
            return candidate
        print(f"Results directory not found: {candidate}")
        return None

    if not base_outputs.exists():
        print(f"Base outputs directory not found: {base_outputs}")
        return None

    candidates = [
        d for d in base_outputs.iterdir()
        if d.is_dir() and (d / "all_runs_summary.json").exists()
    ]
    if not candidates:
        print(f"No results directories found in {base_outputs}")
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _load_summary_rows(results_dir: Path) -> List[Dict]:
    summary_path = results_dir / "all_runs_summary.json"
    if not summary_path.exists():
        return []
    with summary_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return []
    return [row for row in data if row.get("success") is True]


def _init_rng() -> random.Random:
    seed_raw = os.getenv("GANTT_RUN_SEED")
    if seed_raw is None:
        return random.Random()
    try:
        seed = int(seed_raw)
    except Exception:
        seed = seed_raw
    return random.Random(seed)


def _pick_random_row(rows: List[Dict], rng: random.Random) -> Dict:
    return rng.choice(rows)


def _load_run_data(run_path: Path) -> Optional[Dict]:
    try:
        with run_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _load_stop_times(stop_times_path: Optional[Path]) -> Dict[str, Dict]:
    if stop_times_path is None or not stop_times_path.exists():
        return {}
    try:
        with stop_times_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    runs = data.get("runs")
    if isinstance(runs, dict):
        return runs
    if isinstance(runs, list):
        mapped: Dict[str, Dict] = {}
        for entry in runs:
            if not isinstance(entry, dict):
                continue
            scenario = entry.get("scenario")
            fleet = entry.get("fleet")
            algo = entry.get("algorithm")
            run = entry.get("run")
            if not scenario or not fleet or not algo or run is None:
                continue
            mapped[f"{scenario}|{fleet}|{algo}|{run}"] = entry
        return mapped
    return {}


def _lookup_stop_times(
    stop_times: Dict[str, Dict],
    scenario: str,
    fleet: str,
    algo: str,
    run_id: Optional[int],
    bus_idx: int,
    trip_idx: int,
) -> Optional[Tuple[List[float], List[int], Optional[List[float]], Optional[List[float]]]]:
    if not stop_times or run_id is None:
        return None
    key = f"{scenario}|{fleet}|{algo}|{run_id}"
    run_entry = stop_times.get(key)
    if not isinstance(run_entry, dict):
        return None
    buses = run_entry.get("buses")
    if not isinstance(buses, dict):
        return None
    trips = buses.get(str(bus_idx))
    if not isinstance(trips, dict):
        return None
    trip_entry = trips.get(str(trip_idx))
    if not isinstance(trip_entry, dict):
        return None
    label_times = trip_entry.get("label_times")
    arrival_times = trip_entry.get("arrival_times")
    service_end_times = trip_entry.get("service_end_times")
    stop_loads = trip_entry.get("stop_loads")
    time_marks = label_times if isinstance(label_times, list) else arrival_times
    if not isinstance(time_marks, list) or not isinstance(stop_loads, list):
        return None
    return (
        time_marks,
        stop_loads,
        arrival_times if isinstance(arrival_times, list) else None,
        service_end_times if isinstance(service_end_times, list) else None,
    )


def _extract_stop_loads(details) -> List[int]:
    loads: List[int] = []
    if not isinstance(details, list):
        return loads
    for line in details:
        if not isinstance(line, str):
            continue
        if not line.startswith("Stop "):
            continue
        match = re.search(r"picked up (\d+)", line)
        if match:
            loads.append(int(match.group(1)))
            continue
        if "no evacuees" in line:
            loads.append(0)
    return loads


def _count_trip_stops(details) -> int:
    return len(_extract_stop_loads(details))


def _collect_segments(simulation_data: Dict) -> Tuple[List[Tuple[int, int, float, float, List[int]]], int, float]:
    segments: List[Tuple[int, int, float, float, List[int]]] = []
    max_bus = -1
    max_time = 0.0

    if not isinstance(simulation_data, dict):
        return segments, max_bus, max_time

    for bus_key, trips in simulation_data.items():
        bus_idx = _safe_int(bus_key)
        if bus_idx is None:
            continue
        if not isinstance(trips, dict):
            continue
        for trip_pos, (trip_key, trip) in enumerate(trips.items()):
            trip_idx = _safe_int(trip_key)
            if trip_idx is None:
                trip_idx = trip_pos
            if not isinstance(trip, dict):
                continue
            start = _safe_float(trip.get("departure"))
            end = _safe_float(trip.get("return"))
            if start is None or end is None:
                continue
            if end < start:
                start, end = end, start
            stop_loads = _extract_stop_loads(trip.get("details"))
            segments.append((bus_idx, trip_idx, start, end, stop_loads))
            max_time = max(max_time, end)
            max_bus = max(max_bus, bus_idx)

    return segments, max_bus, max_time


def _count_stops(simulation_data: Dict) -> int:
    total = 0
    if not isinstance(simulation_data, dict):
        return total
    for trips in simulation_data.values():
        if not isinstance(trips, dict):
            continue
        for trip in trips.values():
            if not isinstance(trip, dict):
                continue
            total += _count_trip_stops(trip.get("details"))
    return total


def _count_trips(simulation_data: Dict) -> int:
    total = 0
    if not isinstance(simulation_data, dict):
        return total
    for trips in simulation_data.values():
        if isinstance(trips, dict):
            total += len(trips)
    return total


def _plot_gantt_overlay(
    scenario: str,
    fleet: str,
    algo_data: Dict[str, Dict],
    output_path: Path,
    selected_bus_indices: Optional[List[int]] = None,
    vehicle_labels: Optional[List[str]] = None,
    stop_times: Optional[Dict[str, Dict]] = None,
) -> None:
    algo_order = list(algo_data.keys())
    if not algo_order:
        return

    max_time = 0.0
    raw_segments_by_algo: Dict[str, List[Tuple[int, int, float, float, List[int]]]] = {}
    segments_by_algo: Dict[str, List[Tuple[int, int, int, float, float, List[int]]]] = {}
    makespan_by_algo: Dict[str, float] = {}
    all_bus_indices = set()
    for algo in algo_order:
        segments, _, algo_max_time = _collect_segments(algo_data[algo]["simulation_data"])
        raw_segments_by_algo[algo] = segments
        makespan_by_algo[algo] = algo_max_time
        max_time = max(max_time, algo_max_time)
        for bus_idx, _, _, _, _ in segments:
            all_bus_indices.add(bus_idx)

    if selected_bus_indices is None:
        selected_bus_indices = sorted(all_bus_indices)
    if not selected_bus_indices:
        return

    bus_index_map = {bus_idx: i for i, bus_idx in enumerate(selected_bus_indices)}
    for algo, segments in raw_segments_by_algo.items():
        filtered = []
        for bus_idx, trip_idx, start, end, stop_loads in segments:
            row_idx = bus_index_map.get(bus_idx)
            if row_idx is None:
                continue
            filtered.append((row_idx, bus_idx, trip_idx, start, end, stop_loads))
        segments_by_algo[algo] = filtered

    max_bus = len(selected_bus_indices) - 1
    if max_bus < 0 or max_time <= 0:
        return

    count = len(algo_order)
    spacing = 0.35
    if count > 1:
        start_offset = -spacing * (count - 1) / 2.0
        offsets = [start_offset + i * spacing for i in range(count)]
        bar_height = spacing * 0.85
    else:
        offsets = [0.0]
        bar_height = 0.6

    fig_height = max(4.5, (max_bus + 1) * 0.22 + 1.5)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, fig_height))
    style_by_algo = _build_algo_styles(algo_order)
    for idx, algo in enumerate(algo_order):
        style = style_by_algo[algo]
        run_id = algo_data[algo].get("run")
        for row_idx, bus_idx, trip_idx, start, end, stop_loads in segments_by_algo[algo]:
            y_pos = row_idx + offsets[idx]
            ax.barh(
                y_pos,
                end - start,
                left=start,
                height=bar_height,
                facecolor=style["facecolor"],
                edgecolor=style["edgecolor"],
                linewidth=style["linewidth"],
                linestyle=style["linestyle"],
                alpha=style["alpha"],
                zorder=2,
            )
            if stop_loads:
                plot_loads = stop_loads
                stop_positions = None
                start_times = None
                end_times = None
                if stop_times:
                    lookup = _lookup_stop_times(stop_times, scenario, fleet, algo, run_id, bus_idx, trip_idx)
                    if lookup:
                        time_marks, stop_loads_override, arrival_times, service_end_times = lookup
                        if stop_loads_override:
                            plot_loads = stop_loads_override
                        if len(time_marks) == len(plot_loads) and all(
                            isinstance(v, (int, float)) and math.isfinite(v) for v in time_marks
                        ):
                            stop_positions = time_marks
                        if arrival_times and len(arrival_times) == len(plot_loads) and all(
                            isinstance(v, (int, float)) and math.isfinite(v) for v in arrival_times
                        ):
                            start_times = arrival_times
                        if service_end_times and len(service_end_times) == len(plot_loads) and all(
                            isinstance(v, (int, float)) and math.isfinite(v) for v in service_end_times
                        ):
                            end_times = service_end_times
                duration = end - start
                if stop_positions is None and duration > 0:
                    step = duration / (len(plot_loads) + 1)
                    stop_positions = [start + step * (i + 1) for i in range(len(plot_loads))]
                if stop_positions:
                    for stop_pos, load_count in zip(stop_positions, plot_loads):
                        ax.text(
                            stop_pos,
                            y_pos,
                            str(load_count),
                            ha="center",
                            va="center",
                            fontsize=LOAD_LABEL_FONT_SIZE,
                            color="black",
                            alpha=0.85,
                            zorder=4,
                        )
                tick_half = bar_height * 0.45
                if start_times:
                    ax.vlines(
                        start_times,
                        y_pos - tick_half,
                        y_pos + tick_half,
                        colors="#111111",
                        linewidth=0.6,
                        alpha=0.35,
                        zorder=3,
                    )
                if end_times:
                    ax.vlines(
                        end_times,
                        y_pos - tick_half,
                        y_pos + tick_half,
                        colors="#111111",
                        linewidth=0.6,
                        alpha=0.6,
                        linestyles="--",
                        zorder=3,
                    )
        makespan = makespan_by_algo.get(algo)
        if isinstance(makespan, (int, float)) and makespan > 0:
            ax.axvline(
                x=makespan,
                color=style["edgecolor"],
                linestyle=style["linestyle"],
                linewidth=1.1,
                alpha=0.9,
            )

    ax.set_xlabel("Time (min)")
    if vehicle_labels is None or len(vehicle_labels) != len(selected_bus_indices):
        vehicle_labels = [f"V{i}" for i in selected_bus_indices]
    ax.set_yticks(list(range(max_bus + 1)))
    ax.set_yticklabels(vehicle_labels)
    ax.tick_params(axis="y", pad=Y_TICK_PAD)
    ax.set_xlim(0, max_time)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.invert_yaxis()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=TIGHT_LAYOUT_PAD, rect=(LEFT_MARGIN, 0, 1, 1))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_gantt_comparisons(
    results_dir: Path,
    output_dir: Path,
    algos: List[str],
    scenario_filter: Optional[str],
    fleet_filter: Optional[str],
    pick_random: bool,
    stop_times: Optional[Dict[str, Dict]] = None,
) -> None:
    scenario_fleet_map = _load_scenario_fleet_map()
    summary_rows = _load_summary_rows(results_dir)
    if not summary_rows:
        print("No summary data found.")
        return

    grouped: Dict[Tuple[str, str, str], List[Dict]] = {}
    for row in summary_rows:
        algo = row.get("algorithm")
        scenario = row.get("scenario")
        fleet = row.get("fleet")
        if algo not in algos:
            continue
        if scenario_filter and scenario != scenario_filter:
            continue
        if fleet_filter and fleet != fleet_filter:
            continue
        key = (str(scenario), str(fleet), str(algo))
        grouped.setdefault(key, []).append(row)

    if not grouped:
        print("No matching runs found.")
        return

    rng = _init_rng()
    scenarios_fleets = sorted({(s, f) for (s, f, _) in grouped})
    for scenario, fleet in scenarios_fleets:
        rows_by_algo: Dict[str, Dict[int, List[Dict]]] = {}
        missing_algos = []
        for algo in algos:
            rows = grouped.get((scenario, fleet, algo))
            if not rows:
                missing_algos.append(algo)
                continue
            runs: Dict[int, List[Dict]] = {}
            for row in rows:
                run_id = _safe_int(row.get("run"))
                if run_id is None:
                    continue
                runs.setdefault(run_id, []).append(row)
            if not runs:
                missing_algos.append(algo)
                continue
            rows_by_algo[algo] = runs

        if missing_algos:
            print(f"Skipping {scenario} - {fleet}: missing runs for {', '.join(missing_algos)}.")
            continue

        shared_run_ids = None
        for algo in algos:
            algo_run_ids = set(rows_by_algo[algo].keys())
            shared_run_ids = algo_run_ids if shared_run_ids is None else shared_run_ids.intersection(algo_run_ids)
        if not shared_run_ids:
            print(f"Skipping {scenario} - {fleet}: no shared run id across {', '.join(algos)}.")
            continue

        if pick_random:
            shared_run_id = rng.choice(sorted(shared_run_ids))
        else:
            if DEFAULT_RUN_ID not in shared_run_ids:
                print(
                    f"Skipping {scenario} - {fleet}: run {DEFAULT_RUN_ID} not found for all algos."
                )
                continue
            shared_run_id = DEFAULT_RUN_ID
        algo_data: Dict[str, Dict] = {}
        for algo in algos:
            rows = rows_by_algo.get(algo, {}).get(shared_run_id, [])
            if not rows:
                continue
            best_row = _pick_random_row(rows, rng)
            run_id = shared_run_id
            run_path = results_dir / scenario / fleet / algo / f"run_{run_id}.json"
            if not run_path.exists():
                candidates = sorted((results_dir / scenario / fleet / algo).glob("run_*.json"))
                if not candidates:
                    continue
                run_path = candidates[0]
                run_id = _safe_int(run_path.stem.split("_", 1)[1]) or run_id

            run_data = _load_run_data(run_path)
            if not run_data:
                continue
            simulation_data = run_data.get("simulation_data")
            if not simulation_data:
                continue

            algo_data[algo] = {
                "run": run_id,
                "simulation_data": simulation_data,
                "stops_count": _count_stops(simulation_data),
                "trips_count": _count_trips(simulation_data),
                "vehicles": run_data.get("vehicles"),
                "bus_capacity": _safe_int(run_data.get("bus_capacity")),
            }

        if len(algo_data) < 2:
            print(f"Skipping {scenario} - {fleet}: missing comparison data.")
            continue

        algo_suffix = "_".join(_safe_name(a).lower() for a in algos)
        file_name = f"gantt_{_safe_name(scenario)}_{_safe_name(fleet)}_{algo_suffix}.pdf"
        selected_bus_indices = None
        vehicle_labels = None
        base_vehicles = None
        base_bus_capacity = None
        for data in algo_data.values():
            if isinstance(data.get("vehicles"), list) and data["vehicles"]:
                base_vehicles = data["vehicles"]
                base_bus_capacity = data.get("bus_capacity")
                break
        if base_vehicles:
            fleet_spec = scenario_fleet_map.get(str(scenario), {}).get(str(fleet), [])
            capacity_map = _capacity_to_vehicle_type_map(fleet_spec)
            selected_bus_indices, capacities, base_bus_capacity = _select_vehicle_indices(
                base_vehicles,
                base_bus_capacity,
            )
            vehicle_labels = _build_vehicle_labels(
                selected_bus_indices,
                capacities,
                capacity_map,
                base_bus_capacity,
            )

        _plot_gantt_overlay(
            scenario,
            fleet,
            algo_data,
            output_dir / file_name,
            selected_bus_indices=selected_bus_indices,
            vehicle_labels=vehicle_labels,
            stop_times=stop_times,
        )
        if selected_bus_indices is not None:
            run_pairs = ", ".join(
                f"{_display_algo_name(algo)} {algo_data[algo]['run']}"
                for algo in algo_data
                if "run" in algo_data[algo]
            )
            print(f"Saved: {file_name} ({run_pairs})")
        else:
            print(f"Saved: {file_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Gantt comparisons per scenario/fleet.")
    parser.add_argument("--results-dir", help="Path to a results directory containing all_runs_summary.json")
    parser.add_argument("--output-dir", help="Output directory for generated charts")
    parser.add_argument("--algos", help="Comma-separated algorithm names (default: MA_cr08_mr02,ALNS_nopolish)")
    parser.add_argument("--scenario", help="Filter to a specific scenario name")
    parser.add_argument("--fleet", help="Filter to a specific fleet name")
    parser.add_argument("--random", action="store_true", help="Pick a random shared run id per scenario/fleet")
    parser.add_argument("--stop-times", help="Path to a gantt_stop_times.json file")
    args = parser.parse_args()

    analysis_dir = BENCHMARK_DATA_DIR / "analysis"
    base_outputs = BENCHMARK_DATA_DIR / "solutions"
    if not base_outputs.exists():
        alt = Path.cwd() / "benchmark_data" / "solutions"
        if alt.exists():
            base_outputs = alt
        else:
            alt = Path("backend/experiments/outputs")
            if alt.exists():
                base_outputs = alt

    results_dir = _find_results_dir(base_outputs, args.results_dir)
    if not results_dir:
        return

    algos = DEFAULT_ALGOS
    if args.algos:
        algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    if len(algos) < 2:
        print("Need at least two algorithms to compare.")
        return

    output_dir = Path(args.output_dir) if args.output_dir else (analysis_dir / "gantt_comparisons")

    print(f"Results dir: {results_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Algorithms: {', '.join(algos)}")
    if args.scenario:
        print(f"Scenario filter: {args.scenario}")
    if args.fleet:
        print(f"Fleet filter: {args.fleet}")

    stop_times_path = Path(args.stop_times) if args.stop_times else None
    if stop_times_path is None:
        candidates = [
            results_dir / "analysis_output" / "gantt_stop_times.json",
            analysis_dir / "analysis_output" / "gantt_stop_times.json",
        ]
        stop_times_path = next((c for c in candidates if c.exists()), None)
        if stop_times_path is None:
            matches = sorted(analysis_dir.glob("analysis_output*/gantt_stop_times.json"))
            stop_times_path = matches[0] if matches else candidates[-1]
    stop_times = _load_stop_times(stop_times_path)
    if stop_times:
        print(f"Stop times: {stop_times_path}")
    else:
        print("Stop times: none")

    generate_gantt_comparisons(
        results_dir=results_dir,
        output_dir=output_dir,
        algos=algos,
        scenario_filter=args.scenario,
        fleet_filter=args.fleet,
        pick_random=args.random,
        stop_times=stop_times,
    )


if __name__ == "__main__":
    main()
