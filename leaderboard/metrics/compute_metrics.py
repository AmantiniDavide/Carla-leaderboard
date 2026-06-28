#!/usr/bin/env python

"""
Offline metric computation from frame-level telemetry with optional leaderboard merge.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from datetime import datetime, timezone


KNOWN_INFRACTION_KEYS = [
    "collisions_layout",
    "collisions_pedestrian",
    "collisions_vehicle",
    "red_light",
    "stop_infraction",
    "outside_route_lanes",
    "min_speed_infractions",
    "yield_emergency_vehicle_infractions",
    "scenario_timeouts",
    "route_dev",
    "vehicle_blocked",
    "route_timeout",
]

DEFAULT_FRONT_EMERGENCY_DISTANCE_M = 5.0
DEFAULT_FRONT_COLLISION_RISK_DISTANCE_M = 4.5


def _parse_args():
    parser = argparse.ArgumentParser(description="Compute thesis-oriented metrics from raw driving CSV logs.")
    parser.add_argument(
        "inputs",
        nargs="*",
        default=["artifacts/metrics"],
        help="CSV files or directories containing '*_frames.csv' logs.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/metrics/metrics_summary.csv",
        help="Destination CSV for aggregated metrics.",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Optional leaderboard checkpoint JSON used to merge official route statistics.",
    )
    parser.add_argument(
        "--details-output",
        default="",
        help="Optional JSON output with detailed merged route records. Defaults next to --output.",
    )
    return parser.parse_args()


def _is_missing(value):
    return value in (None, "", "None", "nan", "NaN")


def _to_float(value):
    if _is_missing(value):
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return False


def _to_int_or_empty(value):
    if _is_missing(value):
        return ""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return value


def _finite_values(values):
    return [value for value in values if math.isfinite(value)]


def _finite_min(values):
    finite_values = _finite_values(values)
    if not finite_values:
        return math.nan
    return min(finite_values)


def _finite_max(values):
    finite_values = _finite_values(values)
    if not finite_values:
        return math.nan
    return max(finite_values)


def _finite_mean(values):
    finite_values = _finite_values(values)
    if not finite_values:
        return math.nan
    return sum(finite_values) / float(len(finite_values))


def _rmse(values):
    finite_values = _finite_values(values)
    if not finite_values:
        return math.nan
    return math.sqrt(sum(value * value for value in finite_values) / float(len(finite_values)))


def _count_rising_edges(bool_values):
    count = 0
    last = False
    for current in bool_values:
        if current and not last:
            count += 1
        last = current
    return count


def _duration_for_mask(mask, timestamps):
    if len(mask) < 2 or len(timestamps) < 2:
        return 0.0

    total_duration = 0.0
    for index in range(len(mask) - 1):
        if not mask[index]:
            continue
        current_ts = timestamps[index]
        next_ts = timestamps[index + 1]
        if not (math.isfinite(current_ts) and math.isfinite(next_ts)):
            continue
        total_duration += max(next_ts - current_ts, 0.0)
    return total_duration


def _derivative(values, timestamps):
    if len(values) < 2 or len(timestamps) < 2:
        return []

    derivatives = []
    for prev_value, next_value, prev_ts, next_ts in zip(values[:-1], values[1:], timestamps[:-1], timestamps[1:]):
        if not (
            math.isfinite(prev_value)
            and math.isfinite(next_value)
            and math.isfinite(prev_ts)
            and math.isfinite(next_ts)
        ):
            continue
        dt = next_ts - prev_ts
        if dt <= 0.0:
            continue
        derivatives.append((next_value - prev_value) / dt)

    return derivatives


def _collect_input_files(inputs):
    frame_files = []
    for path in inputs:
        if os.path.isdir(path):
            for entry in sorted(os.listdir(path)):
                if entry.endswith("_frames.csv"):
                    frame_files.append(os.path.join(path, entry))
        elif path.endswith("_frames.csv"):
            frame_files.append(path)
    return frame_files


def _load_json_if_exists(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_summary(frame_csv_path):
    return _load_json_if_exists(frame_csv_path.replace("_frames.csv", "_summary.json"))


def _load_checkpoint(checkpoint_path):
    if not checkpoint_path:
        return {}, {}, {}

    checkpoint_data = _load_json_if_exists(checkpoint_path)
    route_records = checkpoint_data.get("_checkpoint", {}).get("records", []) if checkpoint_data else []
    route_lookup = {}
    for record in route_records:
        route_id = record.get("route_id")
        if route_id:
            route_lookup[route_id] = record

    global_record = checkpoint_data.get("_checkpoint", {}).get("global_record", {}) if checkpoint_data else {}
    return checkpoint_data, route_lookup, global_record


def _get_route_metadata(rows, summary):
    first_row = rows[0] if rows else {}
    summary_metadata = summary.get("metadata", {}) if isinstance(summary.get("metadata"), dict) else {}

    return {
        "route_id": first_row.get("route_id") or summary.get("route_id") or summary_metadata.get("route_id") or "",
        "route_index": first_row.get("route_index") or summary.get("route_index") or summary_metadata.get("route_index") or "",
        "repetition_index": (
            first_row.get("repetition_index")
            or summary.get("repetition_index")
            or summary_metadata.get("repetition_index")
            or ""
        ),
        "town": first_row.get("town") or summary.get("town") or summary_metadata.get("town") or "",
    }


def _compute_time_by_state(rows):
    time_by_state = {}
    transitions_into_left_change = 0
    for index in range(len(rows) - 1):
        current_state = rows[index].get("maneuver_state") or "unknown"
        next_state = rows[index + 1].get("maneuver_state") or "unknown"
        current_ts = _to_float(rows[index].get("timestamp"))
        next_ts = _to_float(rows[index + 1].get("timestamp"))
        if not math.isfinite(current_ts) or not math.isfinite(next_ts):
            continue
        duration = max(next_ts - current_ts, 0.0)
        time_by_state[current_state] = time_by_state.get(current_state, 0.0) + duration
        if current_state != "changing_left" and next_state == "changing_left":
            transitions_into_left_change += 1
    return time_by_state, transitions_into_left_change


def _count_brake_throttle_switches(throttle, brake, threshold=0.05):
    last_mode = "idle"
    switches = 0
    for current_throttle, current_brake in zip(throttle, brake):
        mode = "idle"
        if math.isfinite(current_throttle) and current_throttle > threshold and current_throttle >= current_brake:
            mode = "throttle"
        elif math.isfinite(current_brake) and current_brake > threshold and current_brake > current_throttle:
            mode = "brake"

        if mode in ("throttle", "brake") and last_mode in ("throttle", "brake") and mode != last_mode:
            switches += 1
        if mode != "idle":
            last_mode = mode
    return switches


def _distance_from_positions(x, y, z):
    points = [
        (x_value, y_value, z_value)
        for x_value, y_value, z_value in zip(x, y, z)
        if math.isfinite(x_value) and math.isfinite(y_value) and math.isfinite(z_value)
    ]
    if len(points) < 2:
        return math.nan

    total_distance = 0.0
    for previous_point, current_point in zip(points[:-1], points[1:]):
        dx = current_point[0] - previous_point[0]
        dy = current_point[1] - previous_point[1]
        dz = current_point[2] - previous_point[2]
        total_distance += math.sqrt(dx * dx + dy * dy + dz * dz)

    return total_distance


def _count_unsafe_overtakes(rows):
    unsafe_count = 0
    for index, row in enumerate(rows):
        state = row.get("maneuver_state") or "unknown"
        previous_state = rows[index - 1].get("maneuver_state") if index > 0 else None
        if state != "changing_left" or previous_state == "changing_left":
            continue

        rear_mode = (row.get("rear_approach_mode") or "").strip().lower()
        unsafe = (
            _to_bool(row.get("left_oncoming_detected"))
            or rear_mode not in ("", "none", "releasing")
            or _to_bool(row.get("safety_override"))
        )
        if unsafe:
            unsafe_count += 1

    return unsafe_count


def _flatten_checkpoint_record(route_record, checkpoint_entry_status):
    if not route_record:
        flattened = {
            "checkpoint_matched": 0,
            "checkpoint_entry_status": checkpoint_entry_status or "",
            "official_status": "",
            "official_success": "",
            "official_num_infractions": "",
            "official_route_score": "",
            "official_penalty_score": "",
            "official_driving_score": "",
            "official_route_length_m": "",
            "official_duration_game_s": "",
            "official_duration_system_s": "",
            "official_total_collision_count": "",
        }
        for key in KNOWN_INFRACTION_KEYS:
            flattened["official_{}_count".format(key)] = ""
        return flattened

    infractions = route_record.get("infractions", {}) or {}
    scores = route_record.get("scores", {}) or {}
    meta = route_record.get("meta", {}) or {}

    flattened = {
        "checkpoint_matched": 1,
        "checkpoint_entry_status": checkpoint_entry_status or "",
        "official_status": route_record.get("status", ""),
        "official_success": int(str(route_record.get("status", "")).startswith(("Completed", "Perfect"))),
        "official_num_infractions": route_record.get("num_infractions", 0),
        "official_route_score": scores.get("score_route", ""),
        "official_penalty_score": scores.get("score_penalty", ""),
        "official_driving_score": scores.get("score_composed", ""),
        "official_route_length_m": meta.get("route_length", ""),
        "official_duration_game_s": meta.get("duration_game", ""),
        "official_duration_system_s": meta.get("duration_system", ""),
    }

    total_collision_count = 0
    for key in KNOWN_INFRACTION_KEYS:
        values = infractions.get(key, []) or []
        count = len(values)
        flattened["official_{}_count".format(key)] = count
        if key.startswith("collisions_"):
            total_collision_count += count

    flattened["official_total_collision_count"] = total_collision_count
    return flattened


def _build_details_record(metrics_row, summary, route_record, checkpoint_path):
    return {
        "metrics": metrics_row,
        "agent_summary": summary,
        "official_route_record": route_record or {},
        "checkpoint_path": os.path.abspath(checkpoint_path) if checkpoint_path else "",
    }


def _compute_metrics_for_file(frame_csv_path, checkpoint_lookup=None, checkpoint_entry_status=""):
    with open(frame_csv_path, "r", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    if not rows:
        return None, None

    summary = _load_summary(frame_csv_path)
    route_metadata = _get_route_metadata(rows, summary)

    timestamps = [_to_float(row.get("timestamp")) for row in rows]
    speed_mps = [_to_float(row.get("speed_mps")) for row in rows]
    target_speed_mps = [_to_float(row.get("target_speed_mps")) for row in rows]
    speed_error_mps = [_to_float(row.get("speed_error_mps")) for row in rows]
    steer = [_to_float(row.get("steer")) for row in rows]
    throttle = [_to_float(row.get("throttle")) for row in rows]
    brake = [_to_float(row.get("brake")) for row in rows]
    controller_compute_time_ms = [_to_float(row.get("controller_compute_time_ms")) for row in rows]
    x = [_to_float(row.get("x")) for row in rows]
    y = [_to_float(row.get("y")) for row in rows]
    z = [_to_float(row.get("z")) for row in rows]
    lateral_error_m = [_to_float(row.get("lateral_error_m")) for row in rows]
    heading_error_deg = [_to_float(row.get("heading_error_deg")) for row in rows]
    target_lane_lateral_error_m = [_to_float(row.get("target_lane_lateral_error_m")) for row in rows]
    target_lane_heading_error_deg = [_to_float(row.get("target_lane_heading_error_deg")) for row in rows]
    target_lane_progress = [_to_float(row.get("target_lane_progress")) for row in rows]
    front_distance_m = [_to_float(row.get("front_distance_m")) for row in rows]
    front_distance_margin_m = [
        _to_float(row.get("front_distance_margin_m"))
        if not _is_missing(row.get("front_distance_margin_m"))
        else (
            _to_float(row.get("front_distance_m")) - DEFAULT_FRONT_EMERGENCY_DISTANCE_M
            if math.isfinite(_to_float(row.get("front_distance_m")))
            else math.nan
        )
        for row in rows
    ]
    front_dynamic_margin_m = [
        _to_float(row.get("front_dynamic_margin_m"))
        if not _is_missing(row.get("front_dynamic_margin_m"))
        else margin_value
        for row, margin_value in zip(rows, front_distance_margin_m)
    ]
    front_ttc_s = [_to_float(row.get("front_ttc_s")) for row in rows]
    front_relative_lateral_abs_m = [_to_float(row.get("front_relative_lateral_abs_m")) for row in rows]
    rear_distance_m = [_to_float(row.get("rear_distance_m")) for row in rows]
    rear_approach_distance_m = [_to_float(row.get("rear_approach_distance_m")) for row in rows]
    rear_approach_ttc_s = [_to_float(row.get("rear_approach_ttc_s")) for row in rows]
    longitudinal_accel_mps2 = [_to_float(row.get("longitudinal_accel_mps2")) for row in rows]

    emergency_brake = [_to_bool(row.get("emergency_brake")) for row in rows]
    safety_override = [_to_bool(row.get("safety_override")) for row in rows]
    controller_fallback = [_to_bool(row.get("controller_fallback")) for row in rows]
    rear_lane_releasing = [_to_bool(row.get("rear_lane_releasing")) for row in rows]
    steer_saturated = [
        _to_bool(row.get("steer_saturated"))
        if not _is_missing(row.get("steer_saturated"))
        else (math.isfinite(steer_value) and abs(steer_value) >= 0.75)
        for row, steer_value in zip(rows, steer)
    ]
    front_collision_risk = [
        _to_bool(row.get("front_collision_risk"))
        if not _is_missing(row.get("front_collision_risk"))
        else (math.isfinite(front_value) and front_value <= DEFAULT_FRONT_COLLISION_RISK_DISTANCE_M)
        for row, front_value in zip(rows, front_distance_m)
    ]
    safety_override_reason = [(row.get("safety_override_reason") or "").strip() for row in rows]

    positive_dt = [
        next_ts - current_ts
        for current_ts, next_ts in zip(timestamps[:-1], timestamps[1:])
        if math.isfinite(current_ts) and math.isfinite(next_ts) and next_ts - current_ts > 0.0
    ]
    total_duration_s = (
        sum(positive_dt)
        if positive_dt
        else (
            timestamps[-1] - timestamps[0]
            if len(timestamps) > 1 and math.isfinite(timestamps[0]) and math.isfinite(timestamps[-1])
            else math.nan
        )
    )

    steer_rate = _derivative(steer, timestamps)
    if any(math.isfinite(value) for value in longitudinal_accel_mps2):
        jerk = _derivative(longitudinal_accel_mps2, timestamps)
    else:
        longitudinal_accel_mps2 = _derivative(speed_mps, timestamps)
        jerk = _derivative(longitudinal_accel_mps2, timestamps[1:])

    time_by_state, lane_change_count = _compute_time_by_state(rows)

    maneuver_states = [row.get("maneuver_state") or "unknown" for row in rows]
    maneuver_active = any(state != "follow_lane" for state in maneuver_states)
    changing_left_mask = [state == "changing_left" for state in maneuver_states]
    changing_left_same_lane_mask = []
    changing_left_other_lane_mask = []
    for row, active in zip(rows, changing_left_mask):
        lane_id_value = row.get("lane_id")
        origin_lane_id_value = row.get("overtake_origin_lane_id")
        same_lane = False
        if active and not _is_missing(lane_id_value) and not _is_missing(origin_lane_id_value):
            try:
                same_lane = int(float(lane_id_value)) == int(float(origin_lane_id_value))
            except (TypeError, ValueError):
                same_lane = False
        changing_left_same_lane_mask.append(active and same_lane)
        changing_left_other_lane_mask.append(active and not same_lane)

    mean_speed_mps = _finite_mean(speed_mps)
    speed_error_series = (
        speed_error_mps
        if any(math.isfinite(value) for value in speed_error_mps)
        else [
            target_value - speed_value
            if math.isfinite(target_value) and math.isfinite(speed_value)
            else math.nan
            for target_value, speed_value in zip(target_speed_mps, speed_mps)
        ]
    )
    combined_rear_distance_m = rear_distance_m + rear_approach_distance_m

    metrics = {
        "run_id": rows[0].get("run_id", ""),
        "run_label": rows[0].get("run_label", ""),
        "route_id": route_metadata["route_id"],
        "route_index": _to_int_or_empty(route_metadata["route_index"]),
        "repetition_index": _to_int_or_empty(route_metadata["repetition_index"]),
        "town": route_metadata["town"],
        "controller": rows[0].get("controller", ""),
        "controller_mode": rows[0].get("controller_mode", ""),
        "frame_csv_path": os.path.abspath(frame_csv_path),
        "summary_json_path": os.path.abspath(frame_csv_path.replace("_frames.csv", "_summary.json")),
        "frame_count": len(rows),
        "route_completion_time_s": total_duration_s,
        "distance_travelled_m": _distance_from_positions(x, y, z),
        "mean_speed_mps": mean_speed_mps,
        "mean_speed_kph": mean_speed_mps * 3.6 if math.isfinite(mean_speed_mps) else math.nan,
        "min_front_distance_m": _finite_min(front_distance_m),
        "min_front_distance_margin_m": _finite_min(front_distance_margin_m),
        "min_front_dynamic_margin_m": _finite_min(front_dynamic_margin_m),
        "min_front_ttc_s": _finite_min(front_ttc_s),
        "min_rear_distance_m": _finite_min(combined_rear_distance_m),
        "min_rear_approach_ttc_s": _finite_min(rear_approach_ttc_s),
        "emergency_brake_count": summary.get("emergency_brake_count", _count_rising_edges(emergency_brake)),
        "safety_override_count": summary.get("safety_override_count", _count_rising_edges(safety_override)),
        "fail_safe_event_count": summary.get("fail_safe_event_count", math.nan),
        "front_blocked_event_count": summary.get("front_blocked_event_count", math.nan),
        "front_collision_risk_event_count": summary.get(
            "front_collision_risk_event_count",
            _count_rising_edges(front_collision_risk),
        ),
        "left_oncoming_event_count": summary.get("left_oncoming_event_count", math.nan),
        "rear_approach_event_count": summary.get("rear_approach_event_count", math.nan),
        "rear_lane_releasing_event_count": summary.get(
            "rear_lane_releasing_event_count",
            _count_rising_edges(rear_lane_releasing),
        ),
        "unsafe_overtake_count": _count_unsafe_overtakes(rows),
        "lateral_error_mean_abs_m": _finite_mean([abs(value) for value in lateral_error_m]),
        "lateral_error_rmse_m": _rmse(lateral_error_m),
        "max_lateral_error_abs_m": _finite_max([abs(value) for value in lateral_error_m]),
        "heading_error_mean_abs_deg": _finite_mean([abs(value) for value in heading_error_deg]),
        "heading_error_rmse_deg": _rmse(heading_error_deg),
        "max_heading_error_abs_deg": _finite_max([abs(value) for value in heading_error_deg]),
        "speed_error_rmse_mps": _rmse(speed_error_series),
        "mean_abs_steer": _finite_mean([abs(value) for value in steer]),
        "steer_rate_mean_abs": _finite_mean([abs(value) for value in steer_rate]),
        "max_steer_rate_abs": _finite_max([abs(value) for value in steer_rate]),
        "longitudinal_accel_mean_abs_mps2": _finite_mean([abs(value) for value in longitudinal_accel_mps2]),
        "jerk_mean_abs_mps3": _finite_mean([abs(value) for value in jerk]),
        "max_jerk_abs_mps3": _finite_max([abs(value) for value in jerk]),
        "controller_compute_time_mean_ms": _finite_mean(controller_compute_time_ms),
        "controller_compute_time_max_ms": _finite_max(controller_compute_time_ms),
        "controller_fallback_count": summary.get("controller_fallback_count", sum(1 for value in controller_fallback if value)),
        "brake_throttle_switches": _count_brake_throttle_switches(throttle, brake),
        "steer_saturation_event_count": _count_rising_edges(steer_saturated),
        "waiting_time_s": time_by_state.get("waiting_left", 0.0),
        "lane_change_left_time_s": time_by_state.get("changing_left", 0.0),
        "passing_time_s": time_by_state.get("passing", 0.0),
        "rejoin_time_s": time_by_state.get("changing_right", 0.0),
        "changing_left_min_front_distance_m": _finite_min(
            [value for value, active in zip(front_distance_m, changing_left_mask) if active]
        ),
        "changing_left_min_front_margin_m": _finite_min(
            [value for value, active in zip(front_distance_margin_m, changing_left_mask) if active]
        ),
        "changing_left_min_front_dynamic_margin_m": _finite_min(
            [value for value, active in zip(front_dynamic_margin_m, changing_left_mask) if active]
        ),
        "changing_left_min_front_ttc_s": _finite_min(
            [value for value, active in zip(front_ttc_s, changing_left_mask) if active]
        ),
        "changing_left_min_front_lateral_abs_m": _finite_min(
            [value for value, active in zip(front_relative_lateral_abs_m, changing_left_mask) if active]
        ),
        "changing_left_min_target_lane_lateral_abs_m": _finite_min(
            [abs(value) for value, active in zip(target_lane_lateral_error_m, changing_left_mask) if active and math.isfinite(value)]
        ),
        "changing_left_max_target_lane_progress": _finite_max(
            [value for value, active in zip(target_lane_progress, changing_left_mask) if active]
        ),
        "changing_left_target_lane_heading_error_mean_abs_deg": _finite_mean(
            [abs(value) for value, active in zip(target_lane_heading_error_deg, changing_left_mask) if active and math.isfinite(value)]
        ),
        "changing_left_steer_saturation_time_s": _duration_for_mask(
            [active and saturated for active, saturated in zip(changing_left_mask, steer_saturated)],
            timestamps,
        ),
        "changing_left_same_lane_time_s": _duration_for_mask(
            changing_left_same_lane_mask,
            timestamps,
        ),
        "changing_left_other_lane_time_s": _duration_for_mask(
            changing_left_other_lane_mask,
            timestamps,
        ),
        "changing_left_same_lane_countersteer_time_s": _duration_for_mask(
            [
                active and math.isfinite(steer_value) and steer_value > 0.05
                for active, steer_value in zip(changing_left_same_lane_mask, steer)
            ],
            timestamps,
        ),
        "changing_left_same_lane_strong_left_steer_time_s": _duration_for_mask(
            [
                active and math.isfinite(steer_value) and steer_value <= -0.5
                for active, steer_value in zip(changing_left_same_lane_mask, steer)
            ],
            timestamps,
        ),
        "changing_left_front_collision_risk_time_s": _duration_for_mask(
            [active and risk for active, risk in zip(changing_left_mask, front_collision_risk)],
            timestamps,
        ),
        "changing_left_front_ttc_risk_time_s": _duration_for_mask(
            [
                active and math.isfinite(ttc_value) and ttc_value <= 1.5
                for active, ttc_value in zip(changing_left_mask, front_ttc_s)
            ],
            timestamps,
        ),
        "changing_left_front_emergency_time_s": _duration_for_mask(
            [active and reason == "front_emergency" for active, reason in zip(changing_left_mask, safety_override_reason)],
            timestamps,
        ),
        "changing_left_launch_time_s": _duration_for_mask(
            [active and reason == "mpc_lane_change_launch" for active, reason in zip(changing_left_mask, safety_override_reason)],
            timestamps,
        ),
        "changing_left_left_oncoming_time_s": _duration_for_mask(
            [active and reason == "left_oncoming" for active, reason in zip(changing_left_mask, safety_override_reason)],
            timestamps,
        ),
        "waiting_left_min_rear_ttc_s": _finite_min(
            [
                value
                for value, state in zip(rear_approach_ttc_s, maneuver_states)
                if state == "waiting_left"
            ]
        ),
        "waiting_left_rear_lane_releasing_time_s": _duration_for_mask(
            [
                state == "waiting_left" and releasing
                for state, releasing in zip(maneuver_states, rear_lane_releasing)
            ],
            timestamps,
        ),
        "total_maneuver_time_s": sum(
            duration for state, duration in time_by_state.items() if state not in ("follow_lane", "unknown")
        ),
        "lane_change_count": lane_change_count,
        "maneuver_observed": int(maneuver_active),
        "final_state": maneuver_states[-1] if maneuver_states else "",
    }

    route_record = {}
    if checkpoint_lookup and route_metadata["route_id"]:
        route_record = checkpoint_lookup.get(route_metadata["route_id"], {})

    metrics.update(_flatten_checkpoint_record(route_record, checkpoint_entry_status))
    return metrics, _build_details_record(metrics, summary, route_record, "")


def _write_output(metrics_rows, output_path):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if not metrics_rows:
        with open(output_path, "w", encoding="utf-8", newline="") as output_file:
            output_file.write("")
        return

    fieldnames = list(metrics_rows[0].keys())
    with open(output_path, "w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics_rows:
            writer.writerow(row)


def _write_details(details_rows, checkpoint_path, global_record, output_path):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": os.path.abspath(checkpoint_path) if checkpoint_path else "",
        "checkpoint_global_record": global_record or {},
        "routes": details_rows,
    }
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True)


def main():
    args = _parse_args()
    frame_files = _collect_input_files(args.inputs)
    checkpoint_data, checkpoint_lookup, global_record = _load_checkpoint(args.checkpoint)
    checkpoint_entry_status = checkpoint_data.get("entry_status", "") if checkpoint_data else ""
    metrics_rows = []
    details_rows = []

    for frame_file in frame_files:
        metrics, details = _compute_metrics_for_file(frame_file, checkpoint_lookup, checkpoint_entry_status)
        if metrics is None:
            continue

        if details is not None:
            details["checkpoint_path"] = os.path.abspath(args.checkpoint) if args.checkpoint else ""
        metrics_rows.append(metrics)
        details_rows.append(details)

    _write_output(metrics_rows, args.output)

    details_output = args.details_output
    if not details_output:
        base, _ext = os.path.splitext(os.path.abspath(args.output))
        details_output = base + ".json"
    _write_details(details_rows, args.checkpoint, global_record, details_output)

    print("Computed {} metric rows -> {}".format(len(metrics_rows), os.path.abspath(args.output)))
    print("Wrote detailed report -> {}".format(os.path.abspath(details_output)))


if __name__ == "__main__":
    main()
