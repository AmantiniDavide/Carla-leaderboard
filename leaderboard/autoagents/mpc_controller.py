#!/usr/bin/env python

"""
Lightweight shooting-based lateral MPC controller for CARLA waypoint tracking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class MPCResult(object):
    steer: float
    compute_time_ms: float
    status: str
    candidate_count: int
    reference_count: int


class LateralMPCController(object):
    """
    Finite-horizon shooting MPC for lateral control.

    The controller keeps the longitudinal channel untouched and only optimizes the
    steering command over a short horizon using a kinematic bicycle rollout.
    """

    def __init__(
        self,
        horizon_steps=12,
        prediction_dt=0.10,
        wheel_base_m=2.875,
        max_steer_cmd=0.8,
        max_steer_angle_deg=35.0,
        max_steer_delta_per_cycle=0.18,
        min_reference_spacing_m=1.0,
        position_weight=3.5,
        heading_weight=0.8,
        terminal_position_weight=6.0,
        terminal_heading_weight=1.2,
        steer_weight=0.05,
        steer_rate_weight=0.45,
        steer_stage_weight=0.18,
        lateral_error_gain=0.70,
        lane_change_extra_delta=0.12,
        preview_lateral_points=3,
    ):
        self.horizon_steps = max(int(horizon_steps), 4)
        self.prediction_dt = max(float(prediction_dt), 1e-3)
        self.wheel_base_m = max(float(wheel_base_m), 0.5)
        self.max_steer_cmd = max(float(max_steer_cmd), 0.1)
        self.max_steer_angle_rad = np.deg2rad(max(float(max_steer_angle_deg), 1.0))
        self.max_steer_delta_per_cycle = max(float(max_steer_delta_per_cycle), 0.01)
        self.min_reference_spacing_m = max(float(min_reference_spacing_m), 0.1)
        self.position_weight = float(position_weight)
        self.heading_weight = float(heading_weight)
        self.terminal_position_weight = float(terminal_position_weight)
        self.terminal_heading_weight = float(terminal_heading_weight)
        self.steer_weight = float(steer_weight)
        self.steer_rate_weight = float(steer_rate_weight)
        self.steer_stage_weight = float(steer_stage_weight)
        self.lateral_error_gain = float(lateral_error_gain)
        self.lane_change_extra_delta = max(float(lane_change_extra_delta), 0.0)
        self.preview_lateral_points = max(int(preview_lateral_points), 1)

    def compute_steer(self, vehicle_transform, speed_mps, plan, current_steer, fallback_steer):
        speed_mps = max(float(speed_mps or 0.0), 0.5)
        state = self._build_state(vehicle_transform, speed_mps)
        plan_points = self._extract_plan_points(plan)
        if not plan_points:
            return MPCResult(
                steer=float(np.clip(fallback_steer, -self.max_steer_cmd, self.max_steer_cmd)),
                compute_time_ms=0.0,
                status="plan_unavailable",
                candidate_count=0,
                reference_count=0,
            )

        reference = self._build_reference(plan_points, speed_mps, state)
        if reference["count"] == 0:
            return MPCResult(
                steer=float(np.clip(fallback_steer, -self.max_steer_cmd, self.max_steer_cmd)),
                compute_time_ms=0.0,
                status="reference_unavailable",
                candidate_count=0,
                reference_count=0,
            )

        lateral_error_m = self._compute_preview_lateral_error(state, reference)
        lateral_correction = np.clip(
            self.lateral_error_gain * lateral_error_m / max(speed_mps, 1.0),
            -0.45,
            0.45,
        )
        initial_steer_hint = float(np.clip(reference["initial_steer_hint"] + lateral_correction, -self.max_steer_cmd, self.max_steer_cmd))
        terminal_steer_hint = float(np.clip(reference["terminal_steer_hint"] + 0.5 * lateral_correction, -self.max_steer_cmd, self.max_steer_cmd))
        dynamic_delta = self._compute_dynamic_delta(lateral_error_m, reference["lane_change_active"])
        first_stage_candidates = self._generate_stage_candidates(
            current_steer,
            fallback_steer,
            initial_steer_hint,
            dynamic_delta,
        )

        best_cost = float("inf")
        best_steer = float(np.clip(fallback_steer, -self.max_steer_cmd, self.max_steer_cmd))
        candidate_count = 0
        split_index = max(self.horizon_steps // 2, 1)

        for first_stage in first_stage_candidates:
            second_stage_candidates = self._generate_stage_candidates(
                first_stage,
                fallback_steer,
                terminal_steer_hint,
                dynamic_delta,
            )
            for second_stage in second_stage_candidates:
                candidate_count += 1
                cost = self._evaluate_sequence(
                    state,
                    reference,
                    current_steer=current_steer,
                    first_stage=first_stage,
                    second_stage=second_stage,
                    split_index=split_index,
                )
                if cost < best_cost:
                    best_cost = cost
                    best_steer = first_stage

        return MPCResult(
            steer=float(np.clip(best_steer, -self.max_steer_cmd, self.max_steer_cmd)),
            compute_time_ms=0.0,
            status="mpc_ok" if candidate_count > 0 else "candidate_unavailable",
            candidate_count=candidate_count,
            reference_count=reference["count"],
        )

    def _build_state(self, vehicle_transform, speed_mps):
        location = vehicle_transform.location
        yaw_rad = np.deg2rad(vehicle_transform.rotation.yaw)
        return np.array([float(location.x), float(location.y), float(yaw_rad), float(speed_mps)], dtype=np.float64)

    @staticmethod
    def _extract_plan_points(plan):
        points = []
        for waypoint, road_option in list(plan):
            if waypoint is None:
                continue
            location = waypoint.transform.location
            yaw_rad = np.deg2rad(waypoint.transform.rotation.yaw)
            road_option_name = getattr(road_option, "name", str(road_option))
            points.append((float(location.x), float(location.y), float(yaw_rad), road_option_name))
        return points

    def _build_reference(self, plan_points, speed_mps, state):
        if not plan_points:
            return {"count": 0}

        start_index = self._select_reference_start_index(plan_points, state)
        selected_points = plan_points[start_index:]
        if not selected_points:
            return {"count": 0}

        ego_x = float(state[0])
        ego_y = float(state[1])
        ego_yaw = float(state[2])
        x_path = np.array([ego_x] + [point[0] for point in selected_points], dtype=np.float64)
        y_path = np.array([ego_y] + [point[1] for point in selected_points], dtype=np.float64)
        yaw_path = self._compute_path_yaw(x_path, y_path, ego_yaw)
        option_names = [str(point[3]).upper() for point in selected_points]
        cumulative_s = np.zeros(len(x_path), dtype=np.float64)
        for index in range(1, len(x_path)):
            cumulative_s[index] = cumulative_s[index - 1] + math.hypot(
                x_path[index] - x_path[index - 1],
                y_path[index] - y_path[index - 1],
            )
        if cumulative_s[-1] <= 1e-3:
            return {"count": 0}

        spacing_m = max(speed_mps * self.prediction_dt, self.min_reference_spacing_m)
        sample_s = np.array(
            [min((step + 1) * spacing_m, cumulative_s[-1]) for step in range(self.horizon_steps)],
            dtype=np.float64,
        )

        x_ref = np.interp(sample_s, cumulative_s, x_path)
        y_ref = np.interp(sample_s, cumulative_s, y_path)
        yaw_ref = np.interp(sample_s, cumulative_s, yaw_path)
        initial_heading = math.atan2(y_ref[0] - ego_y, x_ref[0] - ego_x) if len(x_ref) > 0 else yaw_path[0]
        terminal_heading = math.atan2(y_ref[-1] - y_ref[-2], x_ref[-1] - x_ref[-2]) if len(x_ref) > 1 else yaw_ref[-1]

        initial_steer_hint = np.clip(self._wrap_angle(initial_heading - ego_yaw) * 0.85, -self.max_steer_cmd, self.max_steer_cmd)
        terminal_steer_hint = np.clip(self._wrap_angle(terminal_heading - yaw_ref[-1]) * 0.85, -self.max_steer_cmd, self.max_steer_cmd)
        lane_change_active = any(name in ("CHANGELANELEFT", "CHANGELANERIGHT") for name in option_names[: min(len(option_names), 8)])

        return {
            "x": x_ref,
            "y": y_ref,
            "yaw": yaw_ref,
            "count": len(x_ref),
            "initial_steer_hint": float(initial_steer_hint),
            "terminal_steer_hint": float(terminal_steer_hint),
            "lane_change_active": lane_change_active,
        }

    @staticmethod
    def _select_reference_start_index(plan_points, state):
        if len(plan_points) <= 1:
            return 0

        x_pos = float(state[0])
        y_pos = float(state[1])
        yaw_rad = float(state[2])
        forward_x = math.cos(yaw_rad)
        forward_y = math.sin(yaw_rad)

        best_index = 0
        best_score = float("inf")
        for index, point in enumerate(plan_points):
            delta_x = float(point[0]) - x_pos
            delta_y = float(point[1]) - y_pos
            distance_m = math.hypot(delta_x, delta_y)
            longitudinal_m = (forward_x * delta_x) + (forward_y * delta_y)
            behind_penalty = max(-longitudinal_m, 0.0) * 2.5
            score = distance_m + behind_penalty
            if score < best_score:
                best_score = score
                best_index = index

        while best_index + 1 < len(plan_points):
            current_distance = math.hypot(float(plan_points[best_index][0]) - x_pos, float(plan_points[best_index][1]) - y_pos)
            next_distance = math.hypot(
                float(plan_points[best_index + 1][0]) - x_pos,
                float(plan_points[best_index + 1][1]) - y_pos,
            )
            if next_distance + 0.35 < current_distance:
                best_index += 1
                continue
            break

        return best_index

    @staticmethod
    def _compute_path_yaw(x_path, y_path, fallback_yaw):
        if len(x_path) == 0:
            return np.zeros(0, dtype=np.float64)

        yaw_path = np.full(len(x_path), float(fallback_yaw), dtype=np.float64)
        running_yaw = float(fallback_yaw)

        for index in range(len(x_path) - 2, -1, -1):
            delta_x = float(x_path[index + 1] - x_path[index])
            delta_y = float(y_path[index + 1] - y_path[index])
            if math.hypot(delta_x, delta_y) > 1e-4:
                running_yaw = math.atan2(delta_y, delta_x)
            yaw_path[index] = running_yaw

        if len(yaw_path) > 1:
            yaw_path[-1] = yaw_path[-2]

        return np.unwrap(yaw_path)

    def _compute_dynamic_delta(self, lateral_error_m, lane_change_active):
        dynamic_delta = self.max_steer_delta_per_cycle + min(abs(float(lateral_error_m)) * 0.05, 0.10)
        if lane_change_active:
            dynamic_delta += self.lane_change_extra_delta
        return min(dynamic_delta, self.max_steer_cmd)

    def _compute_preview_lateral_error(self, state, reference):
        if reference["count"] == 0:
            return 0.0

        preview_count = min(reference["count"], self.preview_lateral_points)
        yaw_rad = float(state[2])
        x_pos = float(state[0])
        y_pos = float(state[1])
        right_x = math.sin(yaw_rad)
        right_y = -math.cos(yaw_rad)

        lateral_errors = []
        for index in range(preview_count):
            delta_x = float(reference["x"][index]) - x_pos
            delta_y = float(reference["y"][index]) - y_pos
            lateral_errors.append(right_x * delta_x + right_y * delta_y)

        return float(sum(lateral_errors) / float(len(lateral_errors))) if lateral_errors else 0.0

    def _generate_stage_candidates(self, current_steer, fallback_steer, reference_hint, max_delta):
        centers = [current_steer, fallback_steer, reference_hint]
        offsets = (-max_delta, -0.5 * max_delta, 0.0, 0.5 * max_delta, max_delta)

        candidates = set()
        lower = current_steer - max_delta
        upper = current_steer + max_delta
        for center in centers:
            for offset in offsets:
                candidate = float(np.clip(center + offset, -self.max_steer_cmd, self.max_steer_cmd))
                candidate = float(np.clip(candidate, lower, upper))
                candidates.add(round(candidate, 4))

        if not candidates:
            return [float(np.clip(fallback_steer, -self.max_steer_cmd, self.max_steer_cmd))]
        return sorted(candidates)

    def _evaluate_sequence(self, state, reference, current_steer, first_stage, second_stage, split_index):
        predicted = np.array(state, dtype=np.float64)
        total_cost = 0.0
        first_stage = float(first_stage)
        second_stage = float(second_stage)

        for step in range(reference["count"]):
            steer = first_stage if step < split_index else second_stage
            predicted = self._rollout_state(predicted, steer)
            dx = predicted[0] - reference["x"][step]
            dy = predicted[1] - reference["y"][step]
            heading_error = self._wrap_angle(predicted[2] - reference["yaw"][step])
            position_cost = self.position_weight * (dx * dx + dy * dy)
            heading_cost = self.heading_weight * (heading_error * heading_error)
            total_cost += position_cost + heading_cost

        terminal_dx = predicted[0] - reference["x"][-1]
        terminal_dy = predicted[1] - reference["y"][-1]
        terminal_heading_error = self._wrap_angle(predicted[2] - reference["yaw"][-1])
        total_cost += self.terminal_position_weight * (terminal_dx * terminal_dx + terminal_dy * terminal_dy)
        total_cost += self.terminal_heading_weight * (terminal_heading_error * terminal_heading_error)
        total_cost += self.steer_weight * ((first_stage * first_stage) + (second_stage * second_stage))
        total_cost += self.steer_rate_weight * ((first_stage - current_steer) ** 2)
        total_cost += self.steer_stage_weight * ((second_stage - first_stage) ** 2)
        return float(total_cost)

    def _rollout_state(self, state, steer_cmd):
        x_pos, y_pos, yaw_rad, speed_mps = state
        steer_angle = float(np.clip(steer_cmd, -self.max_steer_cmd, self.max_steer_cmd)) * self.max_steer_angle_rad
        yaw_rate = 0.0
        if abs(steer_angle) > 1e-6:
            yaw_rate = speed_mps / self.wheel_base_m * math.tan(steer_angle)

        next_x = x_pos + speed_mps * math.cos(yaw_rad) * self.prediction_dt
        next_y = y_pos + speed_mps * math.sin(yaw_rad) * self.prediction_dt
        next_yaw = yaw_rad + yaw_rate * self.prediction_dt
        return np.array([next_x, next_y, next_yaw, speed_mps], dtype=np.float64)

    @staticmethod
    def _wrap_angle(angle_rad):
        return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class LongitudinalMPCResult(object):
    throttle: float
    brake: float
    acceleration_mps2: float
    status: str
    candidate_count: int
    target_speed_mps: float


class LongitudinalMPCController(object):
    """
    Finite-horizon shooting MPC for 1D longitudinal speed control.

    The controller optimizes a short acceleration sequence around the desired
    speed while penalizing predicted front-gap and TTC violations.
    """

    def __init__(
        self,
        horizon_steps=10,
        prediction_dt=0.20,
        max_accel_mps2=2.2,
        max_decel_mps2=4.8,
        max_accel_delta_per_cycle=0.85,
        speed_weight=1.5,
        terminal_speed_weight=2.5,
        accel_weight=0.08,
        jerk_weight=0.28,
        safety_distance_weight=22.0,
        safety_ttc_weight=7.0,
        stop_buffer_m=0.45,
    ):
        self.horizon_steps = max(int(horizon_steps), 4)
        self.prediction_dt = max(float(prediction_dt), 1e-3)
        self.max_accel_mps2 = max(float(max_accel_mps2), 0.1)
        self.max_decel_mps2 = max(float(max_decel_mps2), 0.1)
        self.max_accel_delta_per_cycle = max(float(max_accel_delta_per_cycle), 0.05)
        self.speed_weight = float(speed_weight)
        self.terminal_speed_weight = float(terminal_speed_weight)
        self.accel_weight = float(accel_weight)
        self.jerk_weight = float(jerk_weight)
        self.safety_distance_weight = float(safety_distance_weight)
        self.safety_ttc_weight = float(safety_ttc_weight)
        self.stop_buffer_m = max(float(stop_buffer_m), 0.0)

    def compute_control(
        self,
        speed_mps,
        target_speed_mps,
        current_accel_cmd=0.0,
        front_distance_m=None,
        front_speed_mps=None,
        dynamic_stop_distance_m=None,
        dynamic_emergency_distance_m=None,
        ttc_threshold_s=2.2,
        fallback_throttle=0.0,
        fallback_brake=0.0,
    ):
        speed_mps = max(float(speed_mps or 0.0), 0.0)
        target_speed_mps = max(float(target_speed_mps or 0.0), 0.0)
        current_accel_cmd = float(
            np.clip(current_accel_cmd, -self.max_decel_mps2, self.max_accel_mps2)
        )
        fallback_accel_cmd = self._control_to_acceleration(fallback_throttle, fallback_brake)
        target_accel_cmd = float(
            np.clip(
                (target_speed_mps - speed_mps) / self.prediction_dt,
                -self.max_decel_mps2,
                self.max_accel_mps2,
            )
        )

        first_stage_candidates = self._generate_accel_candidates(
            current_accel_cmd,
            fallback_accel_cmd,
            target_accel_cmd,
        )
        best_cost = float("inf")
        best_accel_cmd = fallback_accel_cmd
        candidate_count = 0
        split_index = max(self.horizon_steps // 2, 1)

        for first_stage in first_stage_candidates:
            second_stage_candidates = self._generate_accel_candidates(
                first_stage,
                fallback_accel_cmd,
                target_accel_cmd,
            )
            for second_stage in second_stage_candidates:
                candidate_count += 1
                cost = self._evaluate_sequence(
                    speed_mps=speed_mps,
                    target_speed_mps=target_speed_mps,
                    current_accel_cmd=current_accel_cmd,
                    first_stage=first_stage,
                    second_stage=second_stage,
                    split_index=split_index,
                    front_distance_m=front_distance_m,
                    front_speed_mps=front_speed_mps,
                    dynamic_stop_distance_m=dynamic_stop_distance_m,
                    dynamic_emergency_distance_m=dynamic_emergency_distance_m,
                    ttc_threshold_s=ttc_threshold_s,
                )
                if cost < best_cost:
                    best_cost = cost
                    best_accel_cmd = first_stage

        throttle, brake = self._acceleration_to_control(best_accel_cmd)
        return LongitudinalMPCResult(
            throttle=throttle,
            brake=brake,
            acceleration_mps2=best_accel_cmd,
            status="mpc_longitudinal_ok" if candidate_count > 0 else "candidate_unavailable",
            candidate_count=candidate_count,
            target_speed_mps=target_speed_mps,
        )

    def _generate_accel_candidates(self, current_accel_cmd, fallback_accel_cmd, target_accel_cmd):
        centers = [current_accel_cmd, fallback_accel_cmd, target_accel_cmd]
        offsets = (
            -self.max_accel_delta_per_cycle,
            -0.5 * self.max_accel_delta_per_cycle,
            0.0,
            0.5 * self.max_accel_delta_per_cycle,
            self.max_accel_delta_per_cycle,
        )

        candidates = set()
        lower = current_accel_cmd - self.max_accel_delta_per_cycle
        upper = current_accel_cmd + self.max_accel_delta_per_cycle
        for center in centers:
            for offset in offsets:
                candidate = float(
                    np.clip(
                        center + offset,
                        -self.max_decel_mps2,
                        self.max_accel_mps2,
                    )
                )
                candidate = float(np.clip(candidate, lower, upper))
                candidates.add(round(candidate, 4))

        if not candidates:
            return [float(np.clip(fallback_accel_cmd, -self.max_decel_mps2, self.max_accel_mps2))]
        return sorted(candidates)

    def _evaluate_sequence(
        self,
        speed_mps,
        target_speed_mps,
        current_accel_cmd,
        first_stage,
        second_stage,
        split_index,
        front_distance_m=None,
        front_speed_mps=None,
        dynamic_stop_distance_m=None,
        dynamic_emergency_distance_m=None,
        ttc_threshold_s=2.2,
    ):
        predicted_speed_mps = max(float(speed_mps or 0.0), 0.0)
        predicted_gap_m = None if front_distance_m is None else float(front_distance_m)
        front_speed_value = 0.0 if front_speed_mps is None else max(float(front_speed_mps), 0.0)
        stop_distance_m = None
        if dynamic_stop_distance_m is not None:
            stop_distance_m = max(float(dynamic_stop_distance_m) + self.stop_buffer_m, 0.0)
        emergency_distance_m = None
        if dynamic_emergency_distance_m is not None:
            emergency_distance_m = max(float(dynamic_emergency_distance_m), 0.0)

        total_cost = 0.0
        first_stage = float(first_stage)
        second_stage = float(second_stage)

        for step in range(self.horizon_steps):
            accel_cmd = first_stage if step < split_index else second_stage
            next_speed_mps = max(predicted_speed_mps + accel_cmd * self.prediction_dt, 0.0)

            speed_error_mps = next_speed_mps - target_speed_mps
            total_cost += self.speed_weight * (speed_error_mps * speed_error_mps)
            total_cost += self.accel_weight * (accel_cmd * accel_cmd)

            if predicted_gap_m is not None:
                ego_step_m = predicted_speed_mps * self.prediction_dt + 0.5 * accel_cmd * (self.prediction_dt ** 2)
                ego_step_m = max(ego_step_m, 0.0)
                front_step_m = front_speed_value * self.prediction_dt
                predicted_gap_m = predicted_gap_m + front_step_m - ego_step_m

                if emergency_distance_m is not None and predicted_gap_m <= emergency_distance_m:
                    gap_error_m = emergency_distance_m - predicted_gap_m + 0.05
                    total_cost += self.safety_distance_weight * 25.0 * (gap_error_m * gap_error_m)
                elif stop_distance_m is not None and predicted_gap_m <= stop_distance_m:
                    gap_error_m = stop_distance_m - predicted_gap_m
                    total_cost += self.safety_distance_weight * (gap_error_m * gap_error_m)

                closing_speed_mps = max(next_speed_mps - front_speed_value, 0.0)
                if closing_speed_mps > 1e-3 and predicted_gap_m > 0.0:
                    predicted_ttc_s = predicted_gap_m / closing_speed_mps
                    if predicted_ttc_s < ttc_threshold_s:
                        ttc_error_s = ttc_threshold_s - predicted_ttc_s
                        total_cost += self.safety_ttc_weight * (ttc_error_s * ttc_error_s)

            predicted_speed_mps = next_speed_mps

        terminal_speed_error_mps = predicted_speed_mps - target_speed_mps
        total_cost += self.terminal_speed_weight * (terminal_speed_error_mps * terminal_speed_error_mps)
        total_cost += self.jerk_weight * (
            ((first_stage - current_accel_cmd) ** 2) + ((second_stage - first_stage) ** 2)
        )
        return float(total_cost)

    def _control_to_acceleration(self, throttle, brake):
        throttle_value = float(np.clip(throttle, 0.0, 1.0))
        brake_value = float(np.clip(brake, 0.0, 1.0))
        return throttle_value * self.max_accel_mps2 - brake_value * self.max_decel_mps2

    def _acceleration_to_control(self, acceleration_mps2):
        accel_value = float(np.clip(acceleration_mps2, -self.max_decel_mps2, self.max_accel_mps2))
        if accel_value >= 0.0:
            throttle = float(np.clip(accel_value / self.max_accel_mps2, 0.0, 1.0))
            return throttle, 0.0

        brake = float(np.clip((-accel_value) / self.max_decel_mps2, 0.0, 1.0))
        return 0.0, brake
