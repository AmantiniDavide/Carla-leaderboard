#!/usr/bin/env python

"""
Runtime logger for frame-by-frame driving data.
"""

from __future__ import annotations

import csv
import json
import os
import re
import uuid
from collections import Counter
from datetime import datetime, timezone


FRAME_FIELDS = [
    "run_id",
    "run_label",
    "route_id",
    "route_index",
    "repetition_index",
    "town",
    "frame",
    "timestamp",
    "controller",
    "controller_mode",
    "controller_compute_time_ms",
    "controller_status",
    "controller_fallback",
    "maneuver_state",
    "waiting_left_reason",
    "speed_mps",
    "speed_kph",
    "target_speed_mps",
    "target_speed_kph",
    "speed_error_mps",
    "steer",
    "throttle",
    "brake",
    "x",
    "y",
    "z",
    "yaw_deg",
    "pitch_deg",
    "roll_deg",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "accel_x",
    "accel_y",
    "accel_z",
    "longitudinal_accel_mps2",
    "lateral_accel_mps2",
    "yaw_rate_dps",
    "road_id",
    "lane_id",
    "target_lane_id",
    "target_lane_lateral_error_m",
    "target_lane_heading_error_deg",
    "target_lane_progress",
    "overtake_origin_lane_id",
    "is_junction",
    "front_distance_m",
    "front_distance_margin_m",
    "front_dynamic_stop_distance_m",
    "front_dynamic_emergency_distance_m",
    "front_dynamic_margin_m",
    "front_ttc_s",
    "front_closing_speed_mps",
    "front_radar_point_count",
    "front_radar_mean_abs_azimuth_rad",
    "front_radar_anchor_azimuth_rad",
    "front_relative_longitudinal_m",
    "front_relative_lateral_m",
    "front_relative_lateral_abs_m",
    "front_blocked",
    "rear_distance_m",
    "rear_approach_distance_m",
    "rear_approach_speed_mps",
    "rear_closing_speed_mps",
    "rear_approach_ttc_s",
    "rear_approach_mode",
    "rear_lane_releasing",
    "left_oncoming_detected",
    "left_oncoming_confidence",
    "left_oncoming_label",
    "lateral_error_m",
    "heading_error_deg",
    "blocked_vehicle",
    "rejoin_lane_clear",
    "safety_override",
    "safety_override_reason",
    "requested_brake",
    "emergency_brake",
    "steer_saturated",
    "front_collision_risk",
]

BOOL_FIELDS = {
    "is_junction",
    "front_blocked",
    "left_oncoming_detected",
    "blocked_vehicle",
    "rejoin_lane_clear",
    "safety_override",
    "emergency_brake",
    "controller_fallback",
    "steer_saturated",
    "front_collision_risk",
    "rear_lane_releasing",
}


def _slugify(value):
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", (value or "").strip())
    normalized = normalized.strip("._-")
    return normalized or "run"


def _to_json_compatible(value):
    if isinstance(value, dict):
        return {str(key): _to_json_compatible(inner_value) for key, inner_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_compatible(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return False


def _as_float(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class MetricsLogger(object):
    """
    Persist frame-level telemetry plus a lightweight run summary.
    """

    def __init__(self, metrics_dir, controller_name="controller", run_label="", metadata=None, flush_interval=25):
        self.metrics_dir = os.path.abspath(metrics_dir)
        self.controller_name = controller_name
        self.run_label = run_label or controller_name
        self.metadata = metadata or {}
        self.flush_interval = max(int(flush_interval), 1)

        os.makedirs(self.metrics_dir, exist_ok=True)

        created_at = datetime.now(timezone.utc)
        slug = _slugify(self.run_label)
        self.run_id = "{}_{}_{}".format(
            created_at.strftime("%Y%m%dT%H%M%SZ"),
            slug,
            uuid.uuid4().hex[:8],
        )
        self.created_at_utc = created_at.isoformat()

        self.frame_path = os.path.join(self.metrics_dir, "{}_frames.csv".format(self.run_id))
        self.summary_path = os.path.join(self.metrics_dir, "{}_summary.json".format(self.run_id))

        self._frame_file = open(self.frame_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._frame_file, fieldnames=FRAME_FIELDS, extrasaction="ignore")
        self._writer.writeheader()

        self._frame_count = 0
        self._start_timestamp = None
        self._end_timestamp = None
        self._state_counts = Counter()
        self._state_transitions = []

        self._min_front_distance_m = None
        self._min_front_distance_margin_m = None
        self._min_rear_distance_m = None
        self._min_front_relative_lateral_abs_m = None
        self._min_front_ttc_s = None
        self._min_rear_approach_ttc_s = None
        self._emergency_brake_count = 0
        self._safety_override_count = 0
        self._front_blocked_event_count = 0
        self._front_collision_risk_event_count = 0
        self._left_oncoming_event_count = 0
        self._rear_approach_event_count = 0
        self._rear_lane_releasing_event_count = 0
        self._fail_safe_event_count = 0
        self._controller_fallback_count = 0

        self._last_state = None
        self._last_emergency_brake = False
        self._last_safety_override = False
        self._last_front_blocked = False
        self._last_front_collision_risk = False
        self._last_left_oncoming = False
        self._last_rear_approach_active = False
        self._last_rear_lane_releasing = False

    def log_frame(self, frame_data):
        row = {field: frame_data.get(field) for field in FRAME_FIELDS}
        row["run_id"] = self.run_id
        row["run_label"] = self.run_label
        row["controller"] = row.get("controller") or self.controller_name

        for field in BOOL_FIELDS:
            row[field] = int(_as_bool(row.get(field)))

        timestamp = _as_float(row.get("timestamp"))
        if self._start_timestamp is None:
            self._start_timestamp = timestamp
        self._end_timestamp = timestamp
        self._frame_count += 1

        state = row.get("maneuver_state") or "unknown"
        self._state_counts[state] += 1
        if state != self._last_state:
            self._state_transitions.append(
                {
                    "timestamp": timestamp,
                    "frame": row.get("frame"),
                    "from_state": self._last_state,
                    "to_state": state,
                }
            )
            self._last_state = state

        front_distance = _as_float(row.get("front_distance_m"))
        if front_distance is not None:
            self._min_front_distance_m = (
                front_distance
                if self._min_front_distance_m is None
                else min(self._min_front_distance_m, front_distance)
            )

        front_distance_margin = _as_float(row.get("front_distance_margin_m"))
        if front_distance_margin is not None:
            self._min_front_distance_margin_m = (
                front_distance_margin
                if self._min_front_distance_margin_m is None
                else min(self._min_front_distance_margin_m, front_distance_margin)
            )

        front_relative_lateral_abs = _as_float(row.get("front_relative_lateral_abs_m"))
        if front_relative_lateral_abs is not None:
            self._min_front_relative_lateral_abs_m = (
                front_relative_lateral_abs
                if self._min_front_relative_lateral_abs_m is None
                else min(self._min_front_relative_lateral_abs_m, front_relative_lateral_abs)
            )

        rear_distance = _as_float(row.get("rear_distance_m"))
        if rear_distance is not None:
            self._min_rear_distance_m = (
                rear_distance
                if self._min_rear_distance_m is None
                else min(self._min_rear_distance_m, rear_distance)
            )

        front_ttc_s = _as_float(row.get("front_ttc_s"))
        if front_ttc_s is not None:
            self._min_front_ttc_s = (
                front_ttc_s
                if self._min_front_ttc_s is None
                else min(self._min_front_ttc_s, front_ttc_s)
            )

        rear_approach_ttc_s = _as_float(row.get("rear_approach_ttc_s"))
        if rear_approach_ttc_s is not None:
            self._min_rear_approach_ttc_s = (
                rear_approach_ttc_s
                if self._min_rear_approach_ttc_s is None
                else min(self._min_rear_approach_ttc_s, rear_approach_ttc_s)
            )

        emergency_brake = _as_bool(row.get("emergency_brake"))
        if emergency_brake and not self._last_emergency_brake:
            self._emergency_brake_count += 1
        self._last_emergency_brake = emergency_brake

        safety_override = _as_bool(row.get("safety_override"))
        override_reason = row.get("safety_override_reason") or ""
        if safety_override and not self._last_safety_override:
            self._safety_override_count += 1
            if override_reason in ("left_oncoming", "front_radar_closing"):
                self._fail_safe_event_count += 1
        self._last_safety_override = safety_override

        front_blocked = _as_bool(row.get("front_blocked"))
        if front_blocked and not self._last_front_blocked:
            self._front_blocked_event_count += 1
        self._last_front_blocked = front_blocked

        front_collision_risk = _as_bool(row.get("front_collision_risk"))
        if front_collision_risk and not self._last_front_collision_risk:
            self._front_collision_risk_event_count += 1
        self._last_front_collision_risk = front_collision_risk

        left_oncoming = _as_bool(row.get("left_oncoming_detected"))
        if left_oncoming and not self._last_left_oncoming:
            self._left_oncoming_event_count += 1
        self._last_left_oncoming = left_oncoming

        controller_fallback = _as_bool(row.get("controller_fallback"))
        if controller_fallback:
            self._controller_fallback_count += 1

        rear_mode = (row.get("rear_approach_mode") or "").strip().lower()
        rear_approach_active = rear_mode in ("approaching", "occupied")
        if rear_approach_active and not self._last_rear_approach_active:
            self._rear_approach_event_count += 1
        self._last_rear_approach_active = rear_approach_active

        rear_lane_releasing = _as_bool(row.get("rear_lane_releasing"))
        if rear_lane_releasing and not self._last_rear_lane_releasing:
            self._rear_lane_releasing_event_count += 1
        self._last_rear_lane_releasing = rear_lane_releasing

        self._writer.writerow(row)
        if self._frame_count % self.flush_interval == 0:
            self._frame_file.flush()

    def close(self, extra_summary=None):
        self._frame_file.flush()
        self._frame_file.close()

        summary = {
            "run_id": self.run_id,
            "run_label": self.run_label,
            "controller": self.controller_name,
            "route_id": self.metadata.get("route_id"),
            "route_index": self.metadata.get("route_index"),
            "repetition_index": self.metadata.get("repetition_index"),
            "town": self.metadata.get("town"),
            "created_at_utc": self.created_at_utc,
            "frame_csv_path": self.frame_path,
            "frame_count": self._frame_count,
            "start_timestamp": self._start_timestamp,
            "end_timestamp": self._end_timestamp,
            "duration_s": (
                None
                if self._start_timestamp is None or self._end_timestamp is None
                else max(self._end_timestamp - self._start_timestamp, 0.0)
            ),
            "min_front_distance_m": self._min_front_distance_m,
            "min_front_distance_margin_m": self._min_front_distance_margin_m,
            "min_front_relative_lateral_abs_m": self._min_front_relative_lateral_abs_m,
            "min_rear_distance_m": self._min_rear_distance_m,
            "min_front_ttc_s": self._min_front_ttc_s,
            "min_rear_approach_ttc_s": self._min_rear_approach_ttc_s,
            "emergency_brake_count": self._emergency_brake_count,
            "safety_override_count": self._safety_override_count,
            "front_blocked_event_count": self._front_blocked_event_count,
            "front_collision_risk_event_count": self._front_collision_risk_event_count,
            "left_oncoming_event_count": self._left_oncoming_event_count,
            "rear_approach_event_count": self._rear_approach_event_count,
            "rear_lane_releasing_event_count": self._rear_lane_releasing_event_count,
            "fail_safe_event_count": self._fail_safe_event_count,
            "controller_fallback_count": self._controller_fallback_count,
            "state_counts": dict(self._state_counts),
            "state_transitions": self._state_transitions,
            "metadata": _to_json_compatible(self.metadata),
        }
        if extra_summary:
            summary.update(_to_json_compatible(extra_summary))

        with open(self.summary_path, "w", encoding="utf-8") as summary_file:
            json.dump(summary, summary_file, indent=2, sort_keys=True)
