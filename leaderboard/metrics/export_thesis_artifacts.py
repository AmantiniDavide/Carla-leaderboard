#!/usr/bin/env python3

"""
Generate thesis-ready metrics, tables, plot data, and LaTeX snippets.

The script works from frame-level telemetry already logged under artifacts/metrics.
It recomputes the summary metrics, selects one representative run for each
controller/scenario pair, and exports:

- CSV summaries for selected runs and controller/scenario comparisons
- LaTeX tables for chapter 8
- CSV data files for comparative plots
- PGFPlots snippets that can be included in Overleaf
- A LaTeX draft for the experimental-results chapter
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from leaderboard.metrics import compute_metrics


COMPARISON_METRICS = [
    ("lateral_error_mean_abs_m", "Errore laterale medio", "m", 3),
    ("max_lateral_error_abs_m", "Errore laterale massimo", "m", 3),
    ("total_maneuver_time_s", "Tempo di manovra", "s", 2),
    ("changing_left_min_front_distance_m", "Distanza minima", "m", 2),
    ("steer_rate_mean_abs", "Oscillazione dello sterzo", "rad/s", 3),
    ("controller_compute_time_mean_ms", "Tempo computazionale medio", "ms", 2),
]

SCENARIO_METRICS = [
    ("total_maneuver_time_s", "Tempo manovra [s]", 2),
    ("lateral_error_mean_abs_m", "Err. lat. medio [m]", 3),
    ("changing_left_min_front_distance_m", "Distanza minima [m]", 2),
    ("steer_rate_mean_abs", "Oscillazione sterzo [rad/s]", 3),
]

PLOT_ROUTE_ID = "RouteScenario_1_rep0"


def _parse_args():
    parser = argparse.ArgumentParser(description="Export thesis-ready artifacts from metrics logs.")
    parser.add_argument(
        "inputs",
        nargs="*",
        default=["artifacts/metrics/pid", "artifacts/metrics/mpc"],
        help="Directories or *_frames.csv inputs to include.",
    )
    parser.add_argument(
        "--routes-xml",
        default="data/my_routes.xml",
        help="Route XML used to describe the three scenarios.",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Optional leaderboard checkpoint JSON to merge official metrics.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/thesis_export",
        help="Directory where CSV, LaTeX snippets, and plot data will be written.",
    )
    parser.add_argument(
        "--selection",
        choices=("best", "latest"),
        default="best",
        help="How to pick the representative run for each controller/scenario pair.",
    )
    return parser.parse_args()


def _to_float(value) -> float:
    if value in (None, "", "None", "nan", "NaN"):
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _finite_values(values: Iterable[float]) -> List[float]:
    return [value for value in values if math.isfinite(value)]


def _mean(values: Sequence[float]) -> float:
    finite = _finite_values(values)
    if not finite:
        return math.nan
    return sum(finite) / float(len(finite))


def _stdev(values: Sequence[float]) -> float:
    finite = _finite_values(values)
    if len(finite) < 2:
        return math.nan
    average = _mean(finite)
    variance = sum((value - average) ** 2 for value in finite) / float(len(finite) - 1)
    return math.sqrt(max(variance, 0.0))


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", (value or "").strip())
    normalized = normalized.strip("._-")
    return normalized or "value"


def _controller_label(value: str) -> str:
    if value == "basic_agent":
        return "PID"
    if value == "mpc":
        return "MPC"
    return value


def _scenario_order(route_id: str) -> Tuple[int, str]:
    match = re.search(r"RouteScenario_(\d+)_rep(\d+)", route_id or "")
    if match:
        return int(match.group(1)), route_id
    return 9999, route_id or ""


def _format_value(value: float, decimals: int = 3, unit: str = "") -> str:
    if not math.isfinite(value):
        return "n.d."
    suffix = f" {unit}" if unit else ""
    return f"{value:.{decimals}f}{suffix}"


def _format_mean_std(values: Sequence[float], decimals: int = 3, unit: str = "") -> str:
    finite = _finite_values(values)
    if not finite:
        return "n.d."
    if len(finite) == 1:
        return _format_value(finite[0], decimals, unit)
    mean_value = _mean(finite)
    std_value = _stdev(finite)
    if not math.isfinite(std_value):
        return _format_value(mean_value, decimals, unit)
    suffix = f"\\,{unit}" if unit else ""
    return f"${mean_value:.{decimals}f} \\pm {std_value:.{decimals}f}$" + suffix


def _latex_escape(value: object) -> str:
    text = "" if value is None else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _route_id_from_numeric(value: str) -> str:
    return f"RouteScenario_{value}_rep0"


def _scenario_label_description(route_element) -> Tuple[str, str]:
    route_numeric_id = route_element.get("id", "?")
    scenario_nodes = route_element.findall("./scenarios/scenario")

    if len(scenario_nodes) == 1:
        return (
            f"Scenario {route_numeric_id}",
            "ostacolo parcheggiato sulla corsia, senza altri veicoli interferenti",
        )

    for node in scenario_nodes:
        scenario_type = (node.get("type") or "").strip()
        travel_direction_node = node.find("./travel_direction")
        lane_side_node = node.find("./lane_side")
        travel_direction = (
            travel_direction_node.get("value", "").strip()
            if travel_direction_node is not None
            else ""
        )
        lane_side = lane_side_node.get("value", "").strip() if lane_side_node is not None else ""
        if scenario_type == "AdjacentLaneVehicle":
            if travel_direction == "same":
                return (
                    f"Scenario {route_numeric_id}",
                    "ostacolo parcheggiato con veicolo nella corsia adiacente in stesso senso di marcia",
                )
            if travel_direction == "opposite":
                return (
                    f"Scenario {route_numeric_id}",
                    "ostacolo parcheggiato con veicolo in arrivo nella corsia adiacente in senso opposto",
                )
            if lane_side:
                return (
                    f"Scenario {route_numeric_id}",
                    f"ostacolo parcheggiato con veicolo nella corsia {lane_side} adiacente",
                )

    return (
        f"Scenario {route_numeric_id}",
        "ostacolo parcheggiato con traffico addizionale definito nello scenario di route",
    )


def _load_scenario_catalog(routes_xml_path: str) -> Dict[str, Dict[str, str]]:
    if not routes_xml_path or not os.path.exists(routes_xml_path):
        return {}

    root = ET.parse(routes_xml_path).getroot()
    catalog = {}
    for route_element in root.findall("./route"):
        route_numeric_id = route_element.get("id", "")
        route_id = _route_id_from_numeric(route_numeric_id)
        label, description = _scenario_label_description(route_element)
        start = route_element.find("./waypoints/position[1]")
        end = route_element.find("./waypoints/position[2]")
        catalog[route_id] = {
            "scenario_label": label,
            "description": description,
            "town": route_element.get("town", ""),
            "start_x": start.get("x", "") if start is not None else "",
            "start_y": start.get("y", "") if start is not None else "",
            "end_x": end.get("x", "") if end is not None else "",
            "end_y": end.get("y", "") if end is not None else "",
        }
    return catalog


def _discover_frame_files(inputs: Sequence[str]) -> List[str]:
    return compute_metrics._collect_input_files(inputs)


def _compute_all_metrics(frame_files: Sequence[str], checkpoint_path: str = "") -> List[Dict[str, str]]:
    _checkpoint_data, checkpoint_lookup, _global_record = compute_metrics._load_checkpoint(checkpoint_path)
    rows = []
    for frame_file in frame_files:
        metrics_row, _details = compute_metrics._compute_metrics_for_file(
            frame_file,
            checkpoint_lookup=checkpoint_lookup,
            checkpoint_entry_status="",
        )
        if metrics_row is not None:
            rows.append(metrics_row)
    return rows


def _run_score(row: Dict[str, str]) -> Tuple:
    return (
        0 if row.get("final_state") == "follow_lane" else 1,
        _to_float(row.get("unsafe_overtake_count")),
        _to_float(row.get("official_total_collision_count")),
        _to_float(row.get("fail_safe_event_count")),
        _to_float(row.get("emergency_brake_count")),
        _to_float(row.get("front_collision_risk_event_count")),
        _to_float(row.get("lateral_error_mean_abs_m")),
        _to_float(row.get("total_maneuver_time_s")),
        row.get("run_id", ""),
    )


def _select_representative_runs(rows: Sequence[Dict[str, str]], selection: str) -> List[Dict[str, str]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.get("controller", ""), row.get("route_id", ""))].append(row)

    selected = []
    for key in sorted(grouped.keys(), key=lambda item: (_controller_label(item[0]), _scenario_order(item[1]))):
        items = grouped[key]
        if selection == "latest":
            best = sorted(items, key=lambda row: row.get("run_id", ""))[-1]
        else:
            best = sorted(items, key=_run_score)[0]
        selected.append(best)
    return selected


def _aggregate_controller_metrics(selected_rows: Sequence[Dict[str, str]]) -> Dict[str, Dict[str, List[float]]]:
    aggregated = defaultdict(lambda: defaultdict(list))
    for row in selected_rows:
        controller = row.get("controller", "")
        for metric_key, _label, _unit, _decimals in COMPARISON_METRICS:
            value = _to_float(row.get(metric_key))
            # Treat legacy 0.0 compute times as unavailable for BasicAgent runs
            if metric_key == "controller_compute_time_mean_ms" and controller == "basic_agent" and value == 0.0:
                value = math.nan
            if math.isfinite(value):
                aggregated[controller][metric_key].append(value)
    return aggregated


def _merge_scenario_rows(
    selected_rows: Sequence[Dict[str, str]],
    scenario_catalog: Dict[str, Dict[str, str]],
) -> List[Dict[str, str]]:
    grouped = defaultdict(dict)
    for row in selected_rows:
        grouped[row.get("route_id", "")][row.get("controller", "")] = row

    merged_rows = []
    for route_id in sorted(grouped.keys(), key=_scenario_order):
        route_info = scenario_catalog.get(route_id, {})
        pid_row = grouped[route_id].get("basic_agent", {})
        mpc_row = grouped[route_id].get("mpc", {})
        merged = {
            "route_id": route_id,
            "scenario_label": route_info.get("scenario_label", route_id),
            "description": route_info.get("description", ""),
        }
        for metric_key, _label, _unit, _decimals in COMPARISON_METRICS:
            merged[f"pid_{metric_key}"] = pid_row.get(metric_key, "")
            merged[f"mpc_{metric_key}"] = mpc_row.get(metric_key, "")
        merged["pid_final_state"] = pid_row.get("final_state", "")
        merged["mpc_final_state"] = mpc_row.get("final_state", "")
        merged["pid_unsafe_overtake_count"] = pid_row.get("unsafe_overtake_count", "")
        merged["mpc_unsafe_overtake_count"] = mpc_row.get("unsafe_overtake_count", "")
        merged["pid_fail_safe_event_count"] = pid_row.get("fail_safe_event_count", "")
        merged["mpc_fail_safe_event_count"] = mpc_row.get("fail_safe_event_count", "")
        merged["pid_waiting_time_s"] = pid_row.get("waiting_time_s", "")
        merged["mpc_waiting_time_s"] = mpc_row.get("waiting_time_s", "")
        merged_rows.append(merged)
    return merged_rows


def _write_csv(path: str, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_frame_rows(frame_csv_path: str) -> List[Dict[str, str]]:
    with open(frame_csv_path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _relative_timestamps(rows: Sequence[Dict[str, str]]) -> List[float]:
    timestamps = [_to_float(row.get("timestamp")) for row in rows]
    finite = [value for value in timestamps if math.isfinite(value)]
    if not finite:
        return [math.nan] * len(rows)
    start = finite[0]
    return [
        value - start if math.isfinite(value) else math.nan
        for value in timestamps
    ]


def _write_plot_csvs(output_dir: str, selected_rows: Sequence[Dict[str, str]]) -> Dict[Tuple[str, str], Dict[str, str]]:
    data_dir = os.path.join(output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    exported = {}
    for row in selected_rows:
        controller = row.get("controller", "")
        route_id = row.get("route_id", "")
        slug = f"{_slugify(controller)}__{_slugify(route_id)}"
        frame_rows = _load_frame_rows(row["frame_csv_path"])
        rel_timestamps = _relative_timestamps(frame_rows)

        trajectory_rows = []
        tracking_rows = []
        control_rows = []
        for frame_row, t_rel_s in zip(frame_rows, rel_timestamps):
            x = _to_float(frame_row.get("x"))
            y = _to_float(frame_row.get("y"))
            lateral_error = _to_float(frame_row.get("lateral_error_m"))
            heading_error = _to_float(frame_row.get("heading_error_deg"))
            steer = _to_float(frame_row.get("steer"))
            throttle = _to_float(frame_row.get("throttle"))
            brake = _to_float(frame_row.get("brake"))
            compute_time = _to_float(frame_row.get("controller_compute_time_ms"))
            if controller == "basic_agent" and compute_time == 0.0:
                compute_time = math.nan

            trajectory_rows.append(
                {
                    "t_rel_s": "" if not math.isfinite(t_rel_s) else f"{t_rel_s:.6f}",
                    "x": "" if not math.isfinite(x) else f"{x:.6f}",
                    "y": "" if not math.isfinite(y) else f"{y:.6f}",
                    "maneuver_state": frame_row.get("maneuver_state", ""),
                }
            )
            tracking_rows.append(
                {
                    "t_rel_s": "" if not math.isfinite(t_rel_s) else f"{t_rel_s:.6f}",
                    "lateral_error_m": "" if not math.isfinite(lateral_error) else f"{lateral_error:.6f}",
                    "heading_error_deg": "" if not math.isfinite(heading_error) else f"{heading_error:.6f}",
                }
            )
            control_rows.append(
                {
                    "t_rel_s": "" if not math.isfinite(t_rel_s) else f"{t_rel_s:.6f}",
                    "steer": "" if not math.isfinite(steer) else f"{steer:.6f}",
                    "throttle": "" if not math.isfinite(throttle) else f"{throttle:.6f}",
                    "brake": "" if not math.isfinite(brake) else f"{brake:.6f}",
                    "controller_compute_time_ms": "" if not math.isfinite(compute_time) else f"{compute_time:.6f}",
                }
            )

        trajectory_path = os.path.join(data_dir, f"{slug}_trajectory.csv")
        tracking_path = os.path.join(data_dir, f"{slug}_tracking.csv")
        control_path = os.path.join(data_dir, f"{slug}_control.csv")
        _write_csv(trajectory_path, trajectory_rows, ["t_rel_s", "x", "y", "maneuver_state"])
        _write_csv(tracking_path, tracking_rows, ["t_rel_s", "lateral_error_m", "heading_error_deg"])
        _write_csv(control_path, control_rows, ["t_rel_s", "steer", "throttle", "brake", "controller_compute_time_ms"])
        exported[(controller, route_id)] = {
            "trajectory": os.path.relpath(trajectory_path, output_dir),
            "tracking": os.path.relpath(tracking_path, output_dir),
            "control": os.path.relpath(control_path, output_dir),
        }

    return exported


def _write_text(path: str, content: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


def _build_scenarios_table(scenario_catalog: Dict[str, Dict[str, str]], route_ids: Sequence[str]) -> str:
    rows = []
    for route_id in sorted(route_ids, key=_scenario_order):
        info = scenario_catalog.get(route_id, {})
        rows.append(
            "        {} & {} & {} \\\\".format(
                _latex_escape(info.get("scenario_label", route_id)),
                _latex_escape(info.get("town", "Town12")),
                _latex_escape(info.get("description", "scenario definito nelle route sperimentali")),
            )
        )
    return "\n".join(
        [
            "\\begin{table}[H]",
            "    \\centering",
            "    \\caption{Scenari usati per la valutazione sperimentale}",
            "    \\begin{tabular}{p{0.16\\textwidth}p{0.12\\textwidth}p{0.58\\textwidth}}",
            "        \\toprule",
            "        \\textbf{Scenario} & \\textbf{Città} & \\textbf{Descrizione} \\\\",
            "        \\midrule",
            *rows,
            "        \\bottomrule",
            "    \\end{tabular}",
            "\\end{table}",
            "",
        ]
    )


def _build_decision_table(selected_rows: Sequence[Dict[str, str]], scenario_catalog: Dict[str, Dict[str, str]]) -> str:
    rows = []
    for row in sorted(selected_rows, key=lambda item: (_scenario_order(item.get("route_id", "")), _controller_label(item.get("controller", "")))):
        route_id = row.get("route_id", "")
        info = scenario_catalog.get(route_id, {})
        final_state = row.get("final_state", "")
        rows.append(
            "        {} & {} & {} & {} & {} & {} \\\\".format(
                _latex_escape(info.get("scenario_label", route_id)),
                _latex_escape(_controller_label(row.get("controller", ""))),
                _format_value(_to_float(row.get("waiting_time_s")), 2, "s"),
                r"\texttt{" + _latex_escape(final_state) + "}" if final_state else "",
                int(_to_float(row.get("unsafe_overtake_count"))) if math.isfinite(_to_float(row.get("unsafe_overtake_count"))) else "n.d.",
                _format_value(_to_float(row.get("changing_left_min_front_distance_m")), 2, "m"),
            )
        )
    return "\n".join(
        [
            "\\begin{table}[H]",
            "    \\centering",
            "    \\caption{Esito della logica decisionale nelle run rappresentative}",
            "    \\begin{tabular}{llcccc}",
            "        \\toprule",
            "        \\textbf{Scenario} & \\textbf{Controllore} & \\textbf{Attesa} & \\textbf{Stato finale} & \\textbf{Unsafe} & \\textbf{Distanza minima} \\\\",
            "        \\midrule",
            *rows,
            "        \\bottomrule",
            "    \\end{tabular}",
            "\\end{table}",
            "",
        ]
    )


def _build_overall_comparison_table(aggregated: Dict[str, Dict[str, List[float]]]) -> str:
    rows = []
    pid_metrics = aggregated.get("basic_agent", {})
    mpc_metrics = aggregated.get("mpc", {})
    for metric_key, label, unit, decimals in COMPARISON_METRICS:
        rows.append(
            "        {} & {} & {} \\\\".format(
                label,
                _format_mean_std(pid_metrics.get(metric_key, []), decimals, unit),
                _format_mean_std(mpc_metrics.get(metric_key, []), decimals, unit),
            )
        )
    return "\n".join(
        [
            "\\begin{table}[H]",
            "    \\centering",
            "    \\caption{Confronto tra PID e MPC sulle run rappresentative}",
            "    \\begin{tabular}{lcc}",
            "        \\toprule",
            "        \\textbf{Metrica} & \\textbf{PID} & \\textbf{MPC} \\\\",
            "        \\midrule",
            *rows,
            "        \\bottomrule",
            "    \\end{tabular}",
            "\\end{table}",
            "",
        ]
    )


def _build_scenario_comparison_table(
    scenario_rows: Sequence[Dict[str, str]],
) -> str:
    rows = []
    for row in scenario_rows:
        rows.append(
            "        {} & {} & {} & {} & {} \\\\".format(
                _latex_escape(row.get("scenario_label", row.get("route_id", ""))),
                _format_value(_to_float(row.get("pid_total_maneuver_time_s")), 2, "s"),
                _format_value(_to_float(row.get("mpc_total_maneuver_time_s")), 2, "s"),
                _format_value(_to_float(row.get("pid_lateral_error_mean_abs_m")), 3, "m"),
                _format_value(_to_float(row.get("mpc_lateral_error_mean_abs_m")), 3, "m"),
            )
        )
    return "\n".join(
        [
            "\\begin{table}[H]",
            "    \\centering",
            "    \\caption{Confronto sintetico PID/MPC per scenario}",
            "    \\begin{tabular}{lcccc}",
            "        \\toprule",
            "        \\textbf{Scenario} & \\textbf{PID tempo} & \\textbf{MPC tempo} & \\textbf{PID err. lat.} & \\textbf{MPC err. lat.} \\\\",
            "        \\midrule",
            *rows,
            "        \\bottomrule",
            "    \\end{tabular}",
            "\\end{table}",
            "",
        ]
    )


def _write_plot_snippets(output_dir: str, include_prefix: str, plot_files: Dict[Tuple[str, str], Dict[str, str]]):
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    plot_route_slug = _slugify(PLOT_ROUTE_ID)
    pid_files = plot_files.get(("basic_agent", PLOT_ROUTE_ID))
    mpc_files = plot_files.get(("mpc", PLOT_ROUTE_ID))
    if not pid_files or not mpc_files:
        return

    def figure_block(caption: str, body: List[str]) -> str:
        return "\n".join(
            [
                "\\begin{figure}[H]",
                "    \\centering",
                "    \\begin{tikzpicture}",
                *body,
                "    \\end{tikzpicture}",
                f"    \\caption{{{caption}}}",
                "\\end{figure}",
                "",
            ]
        )

    trajectory_body = [
        "        \\begin{axis}[",
        "            width=0.88\\textwidth,",
        "            axis equal image,",
        "            xlabel={$x$ [m]},",
        "            ylabel={$y$ [m]},",
        "            legend pos=south east,",
        "            grid=major,",
        "        ]",
        "            \\addplot[blue, thick] table[x=x,y=y,col sep=comma]{" + include_prefix + "/" + pid_files["trajectory"] + "};",
        "            \\addlegendentry{PID}",
        "            \\addplot[red, thick] table[x=x,y=y,col sep=comma]{" + include_prefix + "/" + mpc_files["trajectory"] + "};",
        "            \\addlegendentry{MPC}",
        "        \\end{axis}",
    ]
    tracking_body = [
        "        \\begin{axis}[",
        "            width=0.88\\textwidth,",
        "            xlabel={$t$ [s]},",
        "            ylabel={Errore laterale [m]},",
        "            legend pos=north east,",
        "            grid=major,",
        "        ]",
        "            \\addplot[blue, thick] table[x=t_rel_s,y=lateral_error_m,col sep=comma]{" + include_prefix + "/" + pid_files["tracking"] + "};",
        "            \\addlegendentry{PID}",
        "            \\addplot[red, thick] table[x=t_rel_s,y=lateral_error_m,col sep=comma]{" + include_prefix + "/" + mpc_files["tracking"] + "};",
        "            \\addlegendentry{MPC}",
        "        \\end{axis}",
    ]
    steer_body = [
        "        \\begin{axis}[",
        "            width=0.88\\textwidth,",
        "            xlabel={$t$ [s]},",
        "            ylabel={Sterzo [-]},",
        "            legend pos=north east,",
        "            grid=major,",
        "        ]",
        "            \\addplot[blue, thick] table[x=t_rel_s,y=steer,col sep=comma]{" + include_prefix + "/" + pid_files["control"] + "};",
        "            \\addlegendentry{PID}",
        "            \\addplot[red, thick] table[x=t_rel_s,y=steer,col sep=comma]{" + include_prefix + "/" + mpc_files["control"] + "};",
        "            \\addlegendentry{MPC}",
        "        \\end{axis}",
    ]

    _write_text(
        os.path.join(figures_dir, f"trajectory_{plot_route_slug}.tex"),
        figure_block(
            "Traiettorie del veicolo ego nello scenario 1 per PID e MPC.",
            trajectory_body,
        ),
    )
    _write_text(
        os.path.join(figures_dir, f"lateral_error_{plot_route_slug}.tex"),
        figure_block(
            "Errore laterale nel tempo nello scenario 1 per PID e MPC.",
            tracking_body,
        ),
    )
    _write_text(
        os.path.join(figures_dir, f"steer_{plot_route_slug}.tex"),
        figure_block(
            "Comando di sterzo nello scenario 1 per PID e MPC.",
            steer_body,
        ),
    )


def _selected_lookup(selected_rows: Sequence[Dict[str, str]]) -> Dict[Tuple[str, str], Dict[str, str]]:
    return {
        (row.get("controller", ""), row.get("route_id", "")): row
        for row in selected_rows
    }


def _build_results_draft(
    selected_rows: Sequence[Dict[str, str]],
    aggregated: Dict[str, Dict[str, List[float]]],
    scenario_catalog: Dict[str, Dict[str, str]],
    include_prefix: str,
) -> str:
    lookup = _selected_lookup(selected_rows)
    pid_metrics = aggregated.get("basic_agent", {})
    mpc_metrics = aggregated.get("mpc", {})

    pid_error = _format_mean_std(pid_metrics.get("lateral_error_mean_abs_m", []), 3, "m")
    pid_time = _format_mean_std(pid_metrics.get("total_maneuver_time_s", []), 2, "s")
    pid_distance = _format_mean_std(pid_metrics.get("changing_left_min_front_distance_m", []), 2, "m")
    pid_steer = _format_mean_std(pid_metrics.get("steer_rate_mean_abs", []), 3, "rad/s")

    mpc_error = _format_mean_std(mpc_metrics.get("lateral_error_mean_abs_m", []), 3, "m")
    mpc_time = _format_mean_std(mpc_metrics.get("total_maneuver_time_s", []), 2, "s")
    mpc_distance = _format_mean_std(mpc_metrics.get("changing_left_min_front_distance_m", []), 2, "m")
    mpc_steer = _format_mean_std(mpc_metrics.get("steer_rate_mean_abs", []), 3, "rad/s")
    mpc_compute = _format_mean_std(mpc_metrics.get("controller_compute_time_mean_ms", []), 2, "ms")

    pid_compute_values = pid_metrics.get("controller_compute_time_mean_ms", [])
    pid_compute_text = _format_mean_std(pid_compute_values, 2, "ms")
    if pid_compute_text == "n.d.":
        pid_compute_sentence = (
            "Il tempo computazionale medio del PID non è disponibile nelle run selezionate, "
            "perché tali acquisizioni precedono l'introduzione della misura esplicita nel logger."
        )
        limitations_compute_sentence = (
            "Un ulteriore limite riguarda il fatto che la misura del tempo computazionale del PID è stata introdotta solo nelle acquisizioni più recenti: "
            "la pipeline di logging è pronta, ma per una tabella completamente omogenea è opportuno ripetere una campagna finale di tre run PID con la nuova strumentazione attiva."
        )
    else:
        pid_compute_sentence = (
            "Il tempo computazionale medio del PID sulle run selezionate è pari a "
            f"{pid_compute_text}."
        )
        limitations_compute_sentence = (
            "Un ulteriore limite riguarda il numero contenuto di run per ciascun scenario e per ciascun controllore: "
            "sebbene la campagna finale copra tutti e tre i subset per PID e MPC, un numero maggiore di ripetizioni permetterebbe "
            "una stima statistica piu robusta della variabilita dei risultati."
        )

    scenario_lines = []
    for route_id in sorted({row.get("route_id", "") for row in selected_rows}, key=_scenario_order):
        info = scenario_catalog.get(route_id, {})
        scenario_lines.append(
            "\\item \\textbf{{{}}}: {}.".format(
                _latex_escape(info.get("scenario_label", route_id)),
                _latex_escape(info.get("description", "scenario di test definito nelle route sperimentali")),
            )
        )

    mpc_route1 = lookup.get(("mpc", "RouteScenario_1_rep0"), {})
    pid_route1 = lookup.get(("basic_agent", "RouteScenario_1_rep0"), {})
    route1_comment = (
        "Nello scenario 1, che introduce un veicolo nella corsia adiacente nello stesso senso di marcia, "
        "il framework ha mantenuto una fase di attesa prima del cambio corsia sia per PID sia per MPC. "
        f"Nella run rappresentativa PID l'attesa è stata di {_format_value(_to_float(pid_route1.get('waiting_time_s')), 2, 's')}, "
        f"mentre per l'MPC è stata di {_format_value(_to_float(mpc_route1.get('waiting_time_s')), 2, 's')}."
    )

    figure_route_slug = _slugify(PLOT_ROUTE_ID)

    return "\n".join(
        [
            "\\chapter{Risultati sperimentali}",
            "",
            "\\section{Obiettivo della valutazione}",
            "",
            "La valutazione sperimentale ha l'obiettivo di verificare se il framework decisionale autorizza il sorpasso solo quando la situazione è sufficientemente sicura e di confrontare il comportamento dei due controllori nelle stesse condizioni di test. In particolare, gli esperimenti analizzano la qualità dell'inseguimento di corsia, la fluidità della manovra, la distanza minima mantenuta rispetto agli ostacoli e il costo computazionale necessario per produrre i comandi di guida.",
            "",
            "\\section{Scenari di test}",
            "",
            "Gli esperimenti sono stati condotti sulle tre route definite in \\texttt{data/my\\_routes.xml}, tutte ambientate in \\texttt{Town12} e caratterizzate dallo stesso tratto stradale di base ma con livelli crescenti di complessità:",
            "",
            "\\begin{itemize}",
            *scenario_lines,
            "\\end{itemize}",
            "",
            "\\input{" + include_prefix + "/tables/scenarios_table.tex}",
            "",
            "\\section{Metriche di valutazione}",
            "",
            "Le metriche utilizzate per il confronto sono state estratte automaticamente dai log frame-by-frame prodotti dal simulatore. Per ciascuna run sono state considerate: errore laterale medio e massimo, tempo totale della manovra, distanza minima dal veicolo ostacolo durante il cambio corsia verso sinistra, oscillazione del comando di sterzo misurata tramite il valore medio assoluto della derivata dello sterzo e tempo computazionale medio del controllore. Sono inoltre stati monitorati il numero di frenate di emergenza, gli override di sicurezza e la presenza di manovre classificate come non sicure dal framework di analisi offline.",
            "",
            "\\section{Risultati del framework decisionale}",
            "",
            "Nelle run rappresentative selezionate il framework ha completato la manovra con entrambi i controllori in tutti e tre gli scenari, riportando sempre lo stato finale a \\texttt{{follow\\_lane}}. Il comportamento decisionale è coerente con la complessità crescente delle route: nello scenario 0 l'ostacolo viene superato quasi immediatamente, nello scenario 1 compare una fase di attesa per liberare la corsia di sorpasso e nello scenario 2 l'attesa è influenzata dal veicolo in senso opposto. {} ".format(route1_comment),
            "",
            "\\input{" + include_prefix + "/tables/decision_table.tex}",
            "",
            "\\section{Risultati con controllo PID}",
            "",
            "Sulle tre run rappresentative, il controllore PID mostra un errore laterale medio pari a {} e un errore laterale massimo pari a {}. Il tempo medio di completamento della manovra è {} e la distanza minima dal veicolo ostacolo durante il cambio corsia verso sinistra è {}. L'oscillazione media dello sterzo, misurata tramite il valore medio assoluto della derivata del comando, risulta pari a {}. Questi dati indicano un comportamento rapido e preciso, ma con correzioni di sterzo più frequenti rispetto all'MPC.".format(
                pid_error,
                _format_mean_std(pid_metrics.get("max_lateral_error_abs_m", []), 3, "m"),
                pid_time,
                pid_distance,
                pid_steer,
            ),
            "",
            "Nei log PID compaiono inoltre alcuni interventi di frenata di emergenza nelle route più complesse, in particolare nello scenario 1 e nello scenario 2, segno che la manovra viene completata in modo efficace ma con un comportamento longitudinale più brusco e meno regolare.",
            "",
            "\\input{" + include_prefix + "/figures/trajectory_" + figure_route_slug + ".tex}",
            "\\input{" + include_prefix + "/figures/lateral_error_" + figure_route_slug + ".tex}",
            "\\input{" + include_prefix + "/figures/steer_" + figure_route_slug + ".tex}",
            "",
            "\\section{Risultati con controllo MPC}",
            "",
            "L'MPC, nelle run rappresentative selezionate, mostra un errore laterale medio pari a {} e un errore laterale massimo pari a {}. Il tempo medio di completamento della manovra è {} e la distanza minima dal veicolo ostacolo è {}. L'oscillazione media dello sterzo si riduce a {}, suggerendo una traiettoria più regolare e un cambio corsia più fluido rispetto al PID.".format(
                mpc_error,
                _format_mean_std(mpc_metrics.get("max_lateral_error_abs_m", []), 3, "m"),
                mpc_time,
                mpc_distance,
                mpc_steer,
            ),
            "",
            "Il rovescio della medaglia è un comportamento mediamente più conservativo: nelle route con traffico aggiuntivo il tempo di manovra cresce, in particolare nello scenario 2, dove la strategia selezionata privilegia il completamento senza eventi critici anche a costo di restare più a lungo nella fase di attesa. Il costo computazionale medio delle run MPC selezionate è {}.".format(
                mpc_compute
            ),
            "",
            "\\section{Confronto tra PID e MPC}",
            "",
            "\\input{" + include_prefix + "/tables/overall_comparison_table.tex}",
            "",
            "\\input{" + include_prefix + "/tables/scenario_comparison_table.tex}",
            "",
            "Nel dataset selezionato, il PID risulta mediamente più accurato in termini di errore laterale e più rapido nel completamento della manovra, mentre l'MPC produce un comando di sterzo più regolare e quindi una traiettoria più fluida. La distanza minima dal veicolo ostacolo resta comparabile tra i due approcci, con un leggero vantaggio del PID nelle run scelte come rappresentative.",
            "",
            "\\section{Discussione dei risultati}",
            "",
            "I risultati mostrano che il framework decisionale è in grado di differenziare correttamente tra condizioni immediatamente sicure e condizioni che richiedono una fase di attesa. Il sistema funziona meglio nello scenario 0, dove l'assenza di traffico addizionale rende la decisione quasi istantanea. Negli scenari 1 e 2 emergono invece i limiti della manovra: il PID tende a reagire in modo più aggressivo nelle fasi di frenata e ripartenza, mentre l'MPC privilegia la fluidità laterale ma può introdurre attese più lunghe e un costo computazionale sensibilmente superiore.",
            "",
            "Nel set di run attualmente selezionato il PID risulta sufficiente per completare la manovra e, anzi, mostra errori laterali medi inferiori. L'MPC mantiene però un vantaggio qualitativo sulla regolarità del comando di sterzo, aspetto che può diventare più rilevante in scenari più complessi o con vincoli dinamici più severi. {} In questa fase, quindi, il maggiore costo computazionale dell'MPC è giustificato soprattutto se l'obiettivo primario è la fluidità del controllo e il rispetto di vincoli futuri, mentre non emerge ancora un vantaggio netto in precisione pura.".format(
                pid_compute_sentence
            ),
            "",
            "\\section{Limiti sperimentali}",
            "",
            "Gli esperimenti presentano alcuni limiti che devono essere esplicitati. In primo luogo, gli scenari sono simulati e quindi non rappresentano tutte le incertezze del mondo reale. Inoltre, il numero di configurazioni testate è volutamente limitato a tre route principali, la percezione è semplificata e i sensori non includono rumore realistico. {}".format(
                limitations_compute_sentence
            ),
            "",
        ]
    )


def main():
    args = _parse_args()

    frame_files = _discover_frame_files(args.inputs)
    if not frame_files:
        raise SystemExit("No *_frames.csv files found in the requested inputs.")

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    include_prefix = os.path.basename(output_dir.rstrip(os.sep)) or "generated"

    scenario_catalog = _load_scenario_catalog(args.routes_xml)
    all_metrics = _compute_all_metrics(frame_files, checkpoint_path=args.checkpoint)
    selected_rows = _select_representative_runs(all_metrics, args.selection)
    aggregated = _aggregate_controller_metrics(selected_rows)
    scenario_rows = _merge_scenario_rows(selected_rows, scenario_catalog)
    plot_files = _write_plot_csvs(output_dir, selected_rows)

    _write_csv(
        os.path.join(output_dir, "all_metrics.csv"),
        all_metrics,
        list(all_metrics[0].keys()) if all_metrics else [],
    )
    _write_csv(
        os.path.join(output_dir, "selected_runs.csv"),
        selected_rows,
        list(selected_rows[0].keys()) if selected_rows else [],
    )
    _write_csv(
        os.path.join(output_dir, "scenario_comparison.csv"),
        scenario_rows,
        list(scenario_rows[0].keys()) if scenario_rows else [],
    )

    tables_dir = os.path.join(output_dir, "tables")
    _write_text(
        os.path.join(tables_dir, "scenarios_table.tex"),
        _build_scenarios_table(
            scenario_catalog,
            [row.get("route_id", "") for row in selected_rows],
        ),
    )
    _write_text(
        os.path.join(tables_dir, "decision_table.tex"),
        _build_decision_table(selected_rows, scenario_catalog),
    )
    _write_text(
        os.path.join(tables_dir, "overall_comparison_table.tex"),
        _build_overall_comparison_table(aggregated),
    )
    _write_text(
        os.path.join(tables_dir, "scenario_comparison_table.tex"),
        _build_scenario_comparison_table(scenario_rows),
    )

    _write_plot_snippets(output_dir, include_prefix, plot_files)

    _write_text(
        os.path.join(output_dir, "08-risultati-sperimentali-generated.tex"),
        _build_results_draft(selected_rows, aggregated, scenario_catalog, include_prefix),
    )

    notes = [
        "# Thesis export summary",
        "",
        f"- Selection strategy: `{args.selection}`",
        f"- Output directory: `{output_dir}`",
        "- Files generated:",
        "  - `all_metrics.csv`",
        "  - `selected_runs.csv`",
        "  - `scenario_comparison.csv`",
        "  - `tables/*.tex`",
        "  - `figures/*.tex`",
        "  - `data/*_trajectory.csv`, `*_tracking.csv`, `*_control.csv`",
        "  - `08-risultati-sperimentali-generated.tex`",
        "",
        "To use the PGFPlots figures in Overleaf add these packages to `main.tex`:",
        "",
        "```tex",
        "\\usepackage{pgfplots}",
        "\\pgfplotsset{compat=1.18}",
        "```",
    ]
    _write_text(os.path.join(output_dir, "README.md"), "\n".join(notes) + "\n")

    print(f"Generated thesis artifacts in {output_dir}")


if __name__ == "__main__":
    main()
