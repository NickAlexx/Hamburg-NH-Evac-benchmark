"""
Generate a combined Gantt chart figure with two subplots.

Top: FloodWilhelmsburg (augmented fleet)
Bottom: Default (default fleet)
"""

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt

import plot_gantt_comparisons as base


REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DATA_DIR = REPO_ROOT / "benchmark_data"

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


DEFAULT_TOP_SCENARIO = "FloodWilhelmsburg"
DEFAULT_TOP_FLEET = "augmented"
DEFAULT_BOTTOM_SCENARIO = "Default"
DEFAULT_BOTTOM_FLEET = "default"
DEFAULT_OUTPUT_NAME = "gantt_combined_flood_augmented_default.pdf"
PER_BUS_HEIGHT = 0.22
BASE_HEIGHT = 1.5
MIN_TOP_HEIGHT = 4.5
ALGO_SPACING = 0.35
ALGO_BAR_HEIGHT_FACTOR = 0.85
SINGLE_ALGO_BAR_HEIGHT = 0.6


def _build_algo_styles(algo_order: List[str]) -> Dict[str, Dict]:
    algo_colors = {
        "MA": "#1f77b4",
        "ALNS": "#ff7f0e",
    }
    default_colors = plt.rcParams.get("axes.prop_cycle").by_key().get("color", ["#1f77b4"])
    styles: Dict[str, Dict] = {}
    for idx, algo in enumerate(algo_order):
        display_algo = base._display_algo_name(algo)
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


def _group_rows(
    summary_rows: List[Dict],
    scenario: str,
    fleet: str,
    algos: List[str],
) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = {}
    for row in summary_rows:
        if row.get("scenario") != scenario:
            continue
        if row.get("fleet") != fleet:
            continue
        algo = row.get("algorithm")
        if algo not in algos:
            continue
        grouped.setdefault(algo, []).append(row)
    return grouped


def _build_runs_by_algo(
    rows_by_algo: Dict[str, List[Dict]],
    algos: List[str],
) -> Tuple[Dict[str, Dict[int, List[Dict]]], List[str]]:
    runs_by_algo: Dict[str, Dict[int, List[Dict]]] = {}
    missing_algos = []
    for algo in algos:
        rows = rows_by_algo.get(algo)
        if not rows:
            missing_algos.append(algo)
            continue
        runs: Dict[int, List[Dict]] = {}
        for row in rows:
            run_id = base._safe_int(row.get("run"))
            if run_id is None:
                continue
            runs.setdefault(run_id, []).append(row)
        if not runs:
            missing_algos.append(algo)
            continue
        runs_by_algo[algo] = runs
    return runs_by_algo, missing_algos


def _select_shared_run_id(
    runs_by_algo: Dict[str, Dict[int, List[Dict]]],
    algos: List[str],
    rng,
    pick_random: bool,
) -> Optional[int]:
    shared_run_ids = None
    for algo in algos:
        algo_run_ids = set(runs_by_algo.get(algo, {}).keys())
        shared_run_ids = algo_run_ids if shared_run_ids is None else shared_run_ids.intersection(algo_run_ids)
    if not shared_run_ids:
        return None
    if pick_random:
        return rng.choice(sorted(shared_run_ids))
    if base.DEFAULT_RUN_ID in shared_run_ids:
        return base.DEFAULT_RUN_ID
    return None


def _load_algo_data(
    results_dir: Path,
    scenario: str,
    fleet: str,
    algos: List[str],
    runs_by_algo: Dict[str, Dict[int, List[Dict]]],
    run_id: int,
    rng,
) -> Dict[str, Dict]:
    algo_data: Dict[str, Dict] = {}
    for algo in algos:
        rows = runs_by_algo.get(algo, {}).get(run_id, [])
        if not rows:
            continue
        base._pick_random_row(rows, rng)
        actual_run_id = run_id
        run_path = results_dir / scenario / fleet / algo / f"run_{run_id}.json"
        if not run_path.exists():
            candidates = sorted((results_dir / scenario / fleet / algo).glob("run_*.json"))
            if not candidates:
                continue
            run_path = candidates[0]
            actual_run_id = base._safe_int(run_path.stem.split("_", 1)[1]) or actual_run_id
        run_data = base._load_run_data(run_path)
        if not run_data:
            continue
        simulation_data = run_data.get("simulation_data")
        if not simulation_data:
            continue
        algo_data[algo] = {
            "run": actual_run_id,
            "simulation_data": simulation_data,
            "stops_count": base._count_stops(simulation_data),
            "trips_count": base._count_trips(simulation_data),
            "vehicles": run_data.get("vehicles"),
            "bus_capacity": base._safe_int(run_data.get("bus_capacity")),
        }
    return algo_data


def _prepare_vehicle_labels(
    algo_data: Dict[str, Dict],
    scenario: str,
    fleet: str,
    scenario_fleet_map: Dict[str, Dict],
) -> Tuple[Optional[List[int]], Optional[List[str]]]:
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
        capacity_map = base._capacity_to_vehicle_type_map(fleet_spec)
        selected_bus_indices, capacities, base_bus_capacity = base._select_vehicle_indices(
            base_vehicles,
            base_bus_capacity,
        )
        if selected_bus_indices and capacities and base_bus_capacity is not None:
            small_by_capacity: Dict[int, List[int]] = {}
            for idx in selected_bus_indices:
                if idx >= len(capacities):
                    continue
                cap = capacities[idx]
                if cap is None or cap >= base_bus_capacity:
                    continue
                small_by_capacity.setdefault(cap, []).append(idx)
            drop_indices = {indices[-1] for indices in small_by_capacity.values() if indices}
            if drop_indices:
                selected_bus_indices = [idx for idx in selected_bus_indices if idx not in drop_indices]
        vehicle_labels = base._build_vehicle_labels(
            selected_bus_indices,
            capacities,
            capacity_map,
            base_bus_capacity,
        )
    return selected_bus_indices, vehicle_labels


def _count_buses(
    algo_data: Dict[str, Dict],
    selected_bus_indices: Optional[List[int]],
) -> Tuple[int, bool]:
    if selected_bus_indices is not None:
        return len(selected_bus_indices), True
    all_bus_indices = set()
    for algo in algo_data:
        segments, _, _ = base._collect_segments(algo_data[algo]["simulation_data"])
        for bus_idx, _, _, _, _ in segments:
            all_bus_indices.add(bus_idx)
    if not all_bus_indices:
        return 0, False
    return len(all_bus_indices), True


def _row_span(bus_count: int, algo_count: int) -> float:
    if bus_count <= 0 or algo_count <= 0:
        return 0.0
    if algo_count > 1:
        bar_height = ALGO_SPACING * ALGO_BAR_HEIGHT_FACTOR
        offset_span = ALGO_SPACING * (algo_count - 1)
    else:
        bar_height = SINGLE_ALGO_BAR_HEIGHT
        offset_span = 0.0
    return (bus_count - 1) + offset_span + bar_height


def _estimate_height(
    algo_data: Dict[str, Dict],
    selected_bus_indices: Optional[List[int]],
    min_height: float = MIN_TOP_HEIGHT,
) -> float:
    bus_count, _ = _count_buses(algo_data, selected_bus_indices)
    return max(min_height, (bus_count * PER_BUS_HEIGHT) + BASE_HEIGHT)


def _plot_gantt_overlay_ax(
    ax,
    scenario: str,
    fleet: str,
    algo_data: Dict[str, Dict],
    selected_bus_indices: Optional[List[int]],
    vehicle_labels: Optional[List[str]],
    stop_times: Optional[Dict[str, Dict]] = None,
    style_by_algo: Optional[Dict[str, Dict]] = None,
) -> None:
    algo_order = list(algo_data.keys())
    if not algo_order:
        return
    if style_by_algo is None:
        style_by_algo = _build_algo_styles(algo_order)

    max_time = 0.0
    raw_segments_by_algo: Dict[str, List[Tuple[int, int, float, float, List[int]]]] = {}
    segments_by_algo: Dict[str, List[Tuple[int, int, int, float, float, List[int]]]] = {}
    makespan_by_algo: Dict[str, float] = {}
    all_bus_indices = set()
    for algo in algo_order:
        segments, _, algo_max_time = base._collect_segments(algo_data[algo]["simulation_data"])
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
    spacing = ALGO_SPACING
    if count > 1:
        start_offset = -spacing * (count - 1) / 2.0
        offsets = [start_offset + i * spacing for i in range(count)]
        bar_height = spacing * ALGO_BAR_HEIGHT_FACTOR
    else:
        offsets = [0.0]
        bar_height = SINGLE_ALGO_BAR_HEIGHT

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
                    lookup = base._lookup_stop_times(stop_times, scenario, fleet, algo, run_id, bus_idx, trip_idx)
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
                            fontsize=base.LOAD_LABEL_FONT_SIZE,
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
    ax.tick_params(axis="y", pad=0)
    ax.set_xlim(0, max_time + 0.01*max_time)
    ax.margins(x=0)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.invert_yaxis()


def _build_plot_bundle(
    results_dir: Path,
    summary_rows: List[Dict],
    scenario: str,
    fleet: str,
    algos: List[str],
    scenario_fleet_map: Dict[str, Dict],
    rng,
    pick_random: bool,
) -> Optional[Tuple[Dict[str, Dict], Optional[List[int]], Optional[List[str]]]]:
    rows_by_algo = _group_rows(summary_rows, scenario, fleet, algos)
    runs_by_algo, missing_algos = _build_runs_by_algo(rows_by_algo, algos)
    if missing_algos:
        print(f"Skipping {scenario} - {fleet}: missing runs for {', '.join(missing_algos)}.")
        return None

    shared_run_id = _select_shared_run_id(runs_by_algo, algos, rng, pick_random)
    if shared_run_id is None:
        if pick_random:
            print(f"Skipping {scenario} - {fleet}: no shared run id across {', '.join(algos)}.")
        else:
            print(f"Skipping {scenario} - {fleet}: run {base.DEFAULT_RUN_ID} not found for all algos.")
        return None

    algo_data = _load_algo_data(
        results_dir,
        scenario,
        fleet,
        algos,
        runs_by_algo,
        shared_run_id,
        rng,
    )
    if len(algo_data) < 2:
        print(f"Skipping {scenario} - {fleet}: missing comparison data.")
        return None

    selected_bus_indices, vehicle_labels = _prepare_vehicle_labels(
        algo_data,
        scenario,
        fleet,
        scenario_fleet_map,
    )
    return algo_data, selected_bus_indices, vehicle_labels


def generate_combined_gantt(
    results_dir: Path,
    output_dir: Path,
    algos: List[str],
    top_scenario: str,
    top_fleet: str,
    bottom_scenario: str,
    bottom_fleet: str,
    pick_random: bool,
    output_name: str,
    stop_times: Optional[Dict[str, Dict]] = None,
) -> None:
    summary_rows = base._load_summary_rows(results_dir)
    if not summary_rows:
        print("No summary data found.")
        return

    scenario_fleet_map = base._load_scenario_fleet_map()
    rng = base._init_rng()

    top_bundle = _build_plot_bundle(
        results_dir,
        summary_rows,
        top_scenario,
        top_fleet,
        algos,
        scenario_fleet_map,
        rng,
        pick_random,
    )
    bottom_bundle = _build_plot_bundle(
        results_dir,
        summary_rows,
        bottom_scenario,
        bottom_fleet,
        algos,
        scenario_fleet_map,
        rng,
        pick_random,
    )
    if not top_bundle or not bottom_bundle:
        return

    top_algo_data, top_bus_indices, top_labels = top_bundle
    bottom_algo_data, bottom_bus_indices, bottom_labels = bottom_bundle
    algo_order = [algo for algo in algos if algo in top_algo_data]
    style_by_algo = _build_algo_styles(algo_order)

    top_bus_count, top_count_known = _count_buses(top_algo_data, top_bus_indices)
    bottom_bus_count, bottom_count_known = _count_buses(bottom_algo_data, bottom_bus_indices)
    if not top_count_known:
        print(f"Bus count missing for top ({top_scenario}/{top_fleet}).")
    if not bottom_count_known:
        print(f"Bus count missing for bottom ({bottom_scenario}/{bottom_fleet}).")

    top_height = _estimate_height(top_algo_data, top_bus_indices, min_height=MIN_TOP_HEIGHT)
    top_span = _row_span(top_bus_count, len(top_algo_data))
    bottom_span = _row_span(bottom_bus_count, len(bottom_algo_data))
    if top_span > 0 and bottom_span > 0:
        bottom_height = top_height * (bottom_span / top_span)
    else:
        bottom_height = _estimate_height(
            bottom_algo_data,
            bottom_bus_indices,
            min_height=BASE_HEIGHT + PER_BUS_HEIGHT,
        )
    total_height = top_height + bottom_height

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(base.FIG_WIDTH_IN, total_height/1.7),
        gridspec_kw={"height_ratios": [top_height, bottom_height]},
    )

    _plot_gantt_overlay_ax(
        axes[0],
        top_scenario,
        top_fleet,
        top_algo_data,
        top_bus_indices,
        top_labels,
        stop_times=stop_times,
        style_by_algo=style_by_algo,
    )
    _plot_gantt_overlay_ax(
        axes[1],
        bottom_scenario,
        bottom_fleet,
        bottom_algo_data,
        bottom_bus_indices,
        bottom_labels,
        stop_times=stop_times,
        style_by_algo=style_by_algo,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name
    fig.tight_layout(pad=0)
    fig.savefig(output_path, dpi=200, bbox_inches=None)
    plt.close(fig)

    top_runs = ", ".join(
        f"{base._display_algo_name(algo)} {top_algo_data[algo]['run']}"
        for algo in top_algo_data
        if "run" in top_algo_data[algo]
    )
    bottom_runs = ", ".join(
        f"{base._display_algo_name(algo)} {bottom_algo_data[algo]['run']}"
        for algo in bottom_algo_data
        if "run" in bottom_algo_data[algo]
    )
    print(f"Saved: {output_path.name}")
    print(f"  Top ({top_scenario}/{top_fleet}): {top_runs}")
    print(f"  Bottom ({bottom_scenario}/{bottom_fleet}): {bottom_runs}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a combined Gantt figure with Flood (top) and Default (bottom)."
    )
    parser.add_argument("--results-dir", help="Path to a results directory containing all_runs_summary.json")
    parser.add_argument("--output-dir", help="Output directory for generated charts")
    parser.add_argument("--output-name", help=f"Output filename (default: {DEFAULT_OUTPUT_NAME})")
    parser.add_argument("--algos", help="Comma-separated algorithm names (default: MA_cr08_mr02,ALNS_nopolish)")
    parser.add_argument("--random", action="store_true", help="Pick a random shared run id per scenario/fleet")
    parser.add_argument("--top-scenario", default=DEFAULT_TOP_SCENARIO, help="Top subplot scenario name")
    parser.add_argument("--top-fleet", default=DEFAULT_TOP_FLEET, help="Top subplot fleet name")
    parser.add_argument("--bottom-scenario", default=DEFAULT_BOTTOM_SCENARIO, help="Bottom subplot scenario name")
    parser.add_argument("--bottom-fleet", default=DEFAULT_BOTTOM_FLEET, help="Bottom subplot fleet name")
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

    results_dir = base._find_results_dir(base_outputs, args.results_dir)
    if not results_dir:
        return

    algos = base.DEFAULT_ALGOS
    if args.algos:
        algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    if len(algos) < 2:
        print("Need at least two algorithms to compare.")
        return

    output_dir = Path(args.output_dir) if args.output_dir else (analysis_dir / "gantt_comparisons")
    output_name = args.output_name or DEFAULT_OUTPUT_NAME

    print(f"Results dir: {results_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Output name: {output_name}")
    print(f"Algorithms: {', '.join(algos)}")
    print(f"Top: {args.top_scenario} / {args.top_fleet}")
    print(f"Bottom: {args.bottom_scenario} / {args.bottom_fleet}")

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
    stop_times = base._load_stop_times(stop_times_path)
    if stop_times:
        print(f"Stop times: {stop_times_path}")
    else:
        print("Stop times: none")

    generate_combined_gantt(
        results_dir=results_dir,
        output_dir=output_dir,
        algos=algos,
        top_scenario=args.top_scenario,
        top_fleet=args.top_fleet,
        bottom_scenario=args.bottom_scenario,
        bottom_fleet=args.bottom_fleet,
        pick_random=args.random,
        output_name=output_name,
        stop_times=stop_times,
    )


if __name__ == "__main__":
    main()
