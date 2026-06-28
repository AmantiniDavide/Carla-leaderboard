#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Simple autonomous agent that follows the leaderboard route using CARLA's BasicAgent.
"""

from __future__ import print_function

import time

import carla
import numpy as np
from ultralytics import YOLO
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.local_planner import RoadOption
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from leaderboard.autoagents.mpc_controller import LateralMPCController, LongitudinalMPCController
from leaderboard.metrics.metrics_logger import MetricsLogger

try:
    import pygame
except ImportError:
    raise RuntimeError("cannot import pygame, make sure pygame package is installed")


def get_entry_point():
    return "MyAgent"


class CameraDisplay(object):
    """
    Lightweight first-person camera viewer for debugging the autonomous agent.
    """

    def __init__(self, width, height, title="MyAgent Camera"):
        self._width = width
        self._height = height
        self._surface = None

        pygame.init()
        pygame.font.init()
        self._font = pygame.font.SysFont("courier", 18)
        self._small_font = pygame.font.SysFont("courier", 14)
        self._display = pygame.display.set_mode((self._width, self._height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption(title)

    @staticmethod
    def _image_to_surface(image):
        if image.shape[2] == 4:
            rgb_image = image[:, :, 2::-1]
        elif image.shape[2] == 3:
            rgb_image = image[:, :, ::-1]
        else:
            raise ValueError("Unsupported image format with {} channels".format(image.shape[2]))
        return pygame.surfarray.make_surface(rgb_image.swapaxes(0, 1))

    @staticmethod
    def _radar_color(velocity):
        if velocity < -0.5:
            return (255, 96, 96)
        if velocity > 0.5:
            return (96, 200, 255)
        return (255, 255, 255)

    def _draw_text_lines(self, lines, origin, font=None):
        x, y = origin
        active_font = font or self._font
        for line in lines:
            self._display.blit(active_font.render(line, True, (255, 255, 255)), (x, y))
            y += active_font.get_height() + 4

    def _draw_radar_panel(self, rect, title, points, max_depth=30.0):
        pygame.draw.rect(self._display, (18, 18, 18), rect)
        pygame.draw.rect(self._display, (235, 235, 235), rect, 1)
        self._display.blit(self._small_font.render(title, True, (255, 255, 255)), (rect.x + 8, rect.y + 6))

        center_x = rect.x + rect.w // 2
        base_y = rect.y + rect.h - 12
        top_y = rect.y + 26
        pygame.draw.line(self._display, (70, 70, 70), (center_x, top_y), (center_x, base_y))
        pygame.draw.line(self._display, (70, 70, 70), (rect.x + 8, base_y), (rect.x + rect.w - 8, base_y))

        if points is None or len(points) == 0:
            self._draw_text_lines(["No detections"], (rect.x + 8, rect.y + 34), self._small_font)
            return

        usable_width = rect.w * 0.42
        usable_height = rect.h - 44

        for depth, _altitude, azimuth, velocity in points:
            depth = min(max(float(depth), 0.0), max_depth)
            lateral = np.sin(float(azimuth)) * depth
            forward = np.cos(float(azimuth)) * depth
            x = int(center_x + (lateral / max_depth) * usable_width)
            y = int(base_y - (forward / max_depth) * usable_height)

            if rect.x + 2 <= x <= rect.x + rect.w - 2 and top_y <= y <= rect.y + rect.h - 2:
                pygame.draw.circle(self._display, self._radar_color(float(velocity)), (x, y), 2)

        summary_lines = [
            "{} pts".format(len(points)),
            "min {:.1f} m".format(float(np.min(points[:, 0]))),
            "vel [{:.1f}, {:.1f}] m/s".format(float(np.min(points[:, 3])), float(np.max(points[:, 3]))),
        ]
        self._draw_text_lines(summary_lines, (rect.x + 8, rect.y + rect.h - 56), self._small_font)

    def _draw_detection_overlay(self, detections, target_rect=None, source_size=None):
        if not detections:
            return

        if target_rect is None:
            target_rect = pygame.Rect(0, 0, self._width, self._height)

        if source_size is None:
            source_width, source_height = target_rect.w, target_rect.h
        else:
            source_width, source_height = source_size

        scale_x = target_rect.w / max(float(source_width), 1.0)
        scale_y = target_rect.h / max(float(source_height), 1.0)

        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            rect = pygame.Rect(
                int(target_rect.x + x1 * scale_x),
                int(target_rect.y + y1 * scale_y),
                max(int((x2 - x1) * scale_x), 1),
                max(int((y2 - y1) * scale_y), 1),
            )
            pygame.draw.rect(self._display, (255, 200, 64), rect, 2)

            label = "{} {:.2f}".format(detection["label"], detection["confidence"])
            if detection["distance_m"] is not None:
                label = "{} {:.1f}m".format(label, detection["distance_m"])

            label_surface = self._small_font.render(label, True, (24, 24, 24), (255, 200, 64))
            label_y = max(rect.y - label_surface.get_height() - 2, target_rect.y)
            self._display.blit(label_surface, (rect.x, label_y))

    def render(self, center_image=None, left_image=None, rear_image=None, speed_mps=None,
               detections=None,
               left_detections=None,
               navigation_state=None,
               front_radar=None, rear_radar=None):
        """
        Render the camera feed plus a small debug HUD for the extra sensors.
        """
        pygame.event.pump()

        if center_image is not None:
            self._surface = self._image_to_surface(center_image)
        elif left_image is not None:
            self._surface = self._image_to_surface(left_image)
        else:
            self._surface = None

        if self._surface is not None:
            self._display.blit(self._surface, (0, 0))
        else:
            self._display.fill((0, 0, 0))

        if center_image is not None:
            self._draw_detection_overlay(detections)

        inset_width = self._width // 4
        inset_height = self._height // 4

        if center_image is not None and left_image is not None:
            left_surface = pygame.transform.smoothscale(
                self._image_to_surface(left_image),
                (inset_width, inset_height),
            )
            inset_rect = pygame.Rect(20, 20, inset_width, inset_height)
            pygame.draw.rect(self._display, (0, 0, 0), inset_rect.inflate(6, 6))
            self._display.blit(left_surface, inset_rect.topleft)
            self._draw_detection_overlay(
                left_detections,
                target_rect=inset_rect,
                source_size=(left_image.shape[1], left_image.shape[0]),
            )
            self._draw_text_lines(["LeftCam"], (inset_rect.x + 8, inset_rect.y + 8), self._small_font)

        if center_image is not None and rear_image is not None:
            rear_surface = pygame.transform.smoothscale(
                self._image_to_surface(rear_image),
                (inset_width, inset_height),
            )
            inset_rect = pygame.Rect(20, 40 + inset_height, inset_width, inset_height)
            pygame.draw.rect(self._display, (0, 0, 0), inset_rect.inflate(6, 6))
            self._display.blit(rear_surface, inset_rect.topleft)
            self._draw_text_lines(["RearCam"], (inset_rect.x + 8, inset_rect.y + 8), self._small_font)

        info_lines = ["Center RGB + debug cams active"]
        if speed_mps is None:
            info_lines.append("Speed: unavailable")
        else:
            info_lines.append("Speed: {:.2f} m/s ({:.1f} km/h)".format(speed_mps, speed_mps * 3.6))
        if left_detections:
            info_lines.append("Left YOLO: {} objs".format(len(left_detections)))
        else:
            info_lines.append("Left YOLO: none")
        if navigation_state is not None:
            info_lines.append("Maneuver: {}".format(navigation_state))
        info_lines.append("Radar format: [depth, altitude, azimuth, velocity]")
        self._draw_text_lines(info_lines, (20, self._height - 164))

        panel_width = 280
        panel_height = 160
        self._draw_radar_panel(
            pygame.Rect(self._width - panel_width - 20, 20, panel_width, panel_height),
            "FrontRdr",
            front_radar,
            max_depth=25.0,
        )
        self._draw_radar_panel(
            pygame.Rect(self._width - panel_width - 20, 200, panel_width, panel_height),
            "RearRdr",
            rear_radar,
            max_depth=20.0,
        )

        pygame.display.flip()

    def close(self):
        if pygame.display.get_init():
            pygame.display.quit()
        pygame.quit()


class MyAgent(AutonomousAgent):
    """
    Minimal autonomous agent that follows the provided route with CARLA's BasicAgent.
    """

    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters.
        """
        self.track = Track.SENSORS
        self._agent = None
        self._hero_actor = None
        self._route_assigned = False
        self._controller_mode = "basic_agent"
        self._controller_mode_explicit = False
        self._controller_name = "basic_agent"
        self._target_speed = 20.0
        self._active_target_speed = self._target_speed
        self.camera_width = 1280
        self.camera_height = 720
        self._display = CameraDisplay(self.camera_width, self.camera_height)
        self._metrics_enabled = True
        self._metrics_dir = "artifacts/metrics"
        self._metrics_run_label = ""
        self._metrics_flush_interval = 25
        self._metrics_logger = None
        self._last_safety_intervention = {
            "override": False,
            "reason": "",
            "requested_brake": 0.0,
        }
        self._last_controller_diagnostics = self._make_controller_diagnostics(status="basic_agent_pid")
        self._last_longitudinal_controller_diagnostics = self._make_controller_diagnostics(
            status="basic_agent_longitudinal"
        )
        self._basic_agent_hazard_brake_threshold = 0.49
        self._mpc_controller = None
        self._mpc_longitudinal_controller = None
        self._mpc_horizon_steps = 12
        self._mpc_prediction_dt = 0.10
        self._mpc_wheel_base_m = 2.875
        self._mpc_max_steer_delta_per_cycle = 0.18
        self._mpc_max_steer_angle_deg = 35.0
        self._mpc_follow_steer_step_limit = 0.08
        self._mpc_lane_change_steer_step_limit = 0.12
        self._mpc_same_lane_change_steer_limit = 0.80
        self._mpc_close_front_steer_limit = 0.80
        self._mpc_lane_change_commitment_min_steer = 0.58
        self._mpc_lane_change_commitment_release_lateral_m = 0.45
        self._mpc_lane_change_commitment_release_distance_m = 50.0
        self._mpc_left_release_speed_mps = 2.60
        self._mpc_left_release_heading_deg = 24.5
        self._mpc_left_release_lateral_error_m = 0.85
        self._mpc_left_release_progress = 0.24
        self._mpc_left_release_max_steer = 0.52
        self._mpc_left_final_release_speed_mps = 3.55
        self._mpc_left_final_release_heading_deg = 28.6
        self._mpc_left_final_release_lateral_error_m = 1.10
        self._mpc_left_final_release_progress = 0.32
        self._mpc_left_final_release_max_steer = 0.12
        self._mpc_rejoin_steer_limit = 0.45
        self._mpc_rejoin_soft_heading_deg = 12.0
        self._mpc_rejoin_soft_lateral_error_m = 1.0
        self._mpc_rejoin_hard_heading_deg = 6.0
        self._mpc_rejoin_hard_lateral_error_m = 0.45
        self._mpc_rejoin_finish_max_heading_error_deg = 5.0
        self._mpc_rejoin_finish_max_lateral_error_m = 0.40
        self._mpc_rejoin_finish_max_abs_steer = 0.18
        self._current_timestamp = 0.0
        self._last_debug_timestamp = -1.0
        self._sensor_snapshot_logged = False
        self._forward_camera_pitch = 0.0
        self._front_radar_min_depth = 1.0
        self._front_radar_max_depth = 25.0
        self._front_radar_max_abs_azimuth = np.deg2rad(10.0)
        self._front_cluster_depth_window = 4.0
        self._front_cluster_azimuth_window = np.deg2rad(5.0)
        self._rear_radar_min_depth = 0.5
        self._rear_radar_max_depth = 20.0
        self._rear_radar_max_abs_azimuth = np.deg2rad(10.0)
        self._rear_cluster_depth_window = 4.0
        self._rear_cluster_azimuth_window = np.deg2rad(5.0)
        self.detector = YOLO("yolov8n.pt")
        self._yolo_conf_threshold = 0.6
        self._yolo_imgsz = 640
        self._yolo_max_draw_detections = 8
        self._yolo_max_summary_detections = 3
        self._vehicle_detection_labels = {"car", "truck", "bus", "motorcycle", "bicycle"}
        self._front_vehicle_block_distance = 14.0
        self._front_vehicle_stop_distance = 8.0
        self._front_vehicle_resume_distance = 18.0
        self._front_vehicle_emergency_distance = 5.0
        self._front_vehicle_min_hold_distance = 2.3
        self._front_vehicle_reaction_time_s = 0.20
        self._front_vehicle_comfort_decel_mps2 = 4.5
        self._front_vehicle_low_speed_mps = 0.75
        self._front_vehicle_restart_speed_mps = 1.2
        self._front_vehicle_ttc_brake_s = 2.2
        self._front_vehicle_ttc_emergency_s = 1.0
        self._front_radar_trigger_distance = 10.0
        self._front_radar_resume_distance = 12.0
        self._front_radar_min_points = 2
        self._front_radar_max_mean_abs_azimuth = np.deg2rad(4.0)
        self._rejoin_radar_max_depth = 18.0
        self._rejoin_radar_max_abs_azimuth = np.deg2rad(35.0)
        self._rejoin_radar_min_forward_distance = 1.5
        self._rejoin_radar_min_points = 2
        self._rejoin_lane_lateral_margin = 1.5
        self._rejoin_clear_hold_time = 0.8
        self._front_vehicle_min_confidence = 0.45
        self._front_vehicle_center_x_min = 0.30
        self._front_vehicle_center_x_max = 0.70
        self._front_vehicle_min_bottom_ratio = 0.35
        self._rear_approach_distance = 18.0
        self._rear_approach_velocity = -2.0
        self._rear_approach_ttc_threshold_s = 4.5
        self._rear_approach_close_distance = 7.5
        self._rear_lane_occupied_distance = 14.0
        self._rear_lane_occupied_abs_velocity = 1.0
        self._rear_lane_occupied_min_points = 4
        self._rear_lane_hold_distance = 8.0
        self._rear_lane_release_distance = 4.5
        self._rear_lane_release_velocity_mps = 0.10
        self._rear_lane_release_closing_speed_mps = 0.35
        self._rear_lane_release_ttc_s = 6.0
        self._left_oncoming_min_confidence = 0.45
        self._left_oncoming_min_area_ratio = 0.0025
        self._left_oncoming_min_center_x_ratio = 0.15
        self._left_oncoming_max_center_x_ratio = 0.90
        self._sensor_soft_brake = 0.18
        self._sensor_stop_brake = 0.38
        self._sensor_close_brake = 0.62
        self._sensor_emergency_brake = 0.95
        self._overtake_fail_safe_distance = 6.0
        self._overtake_fail_safe_velocity = -2.0
        self._blocked_hold_time = 1.0
        self._overtake_speed = min(max(self._target_speed, 15.0), 25.0)
        self._overtake_same_lane_distance = 2.0
        self._overtake_lane_change_distance = 8.0
        self._overtake_other_lane_distance = 20.0
        self._mpc_overtake_same_lane_distance = 0.8
        self._mpc_waiting_left_close_stop_distance = 6.4
        self._mpc_waiting_left_close_stop_brake = 0.78
        self._mpc_waiting_left_oncoming_extra_stop_buffer_m = 1.35
        self._mpc_lane_change_launch_grace_time = 1.0
        self._mpc_lane_change_launch_min_front_distance = 4.0
        self._mpc_lane_change_launch_min_steer = 0.20
        self._mpc_lane_change_launch_min_throttle = 0.20
        self._mpc_lane_change_launch_throttle = 0.30
        self._mpc_lane_change_resume_speed_mps = 0.25
        self._mpc_longitudinal_horizon_steps = 10
        self._mpc_longitudinal_prediction_dt = 0.20
        self._mpc_longitudinal_max_accel_mps2 = 2.2
        self._mpc_longitudinal_max_decel_mps2 = 4.8
        self._mpc_longitudinal_max_accel_delta_per_cycle = 0.85
        self._mpc_longitudinal_stop_buffer_m = 0.45
        self._mpc_waiting_left_target_speed_mps = 0.0
        self._mpc_waiting_left_creep_speed_mps = 0.65
        self._mpc_changing_left_target_speed_mps = 0.0
        self._mpc_passing_target_speed_mps = 0.0
        self._mpc_changing_right_target_speed_mps = 0.0
        self._last_mpc_longitudinal_accel_cmd = 0.0
        self._return_same_lane_distance = 4.0
        self._return_lane_change_distance = 8.0
        self._return_other_lane_distance = 8.0
        self._min_left_lane_time = 1.5
        self._min_right_lane_settle_time = 0.4
        self._post_overtake_cooldown_time = 2.0
        self._route_rejoin_min_distance = 8.0
        self._front_collision_risk_distance = 4.5
        self._overtake_state = "follow_lane"
        self._blocked_vehicle_since = None
        self._overtake_started_at = None
        self._left_lane_entered_at = None
        self._right_lane_entered_at = None
        self._overtake_cooldown_until = None
        self._overtake_origin_lane_id = None
        self._overtake_origin_road_id = None
        self._using_helper_rejoin_plan = False
        self._saved_route_plan = []
        self._waiting_left_reason = "clear"
        self._rejoin_lane_clear_since = None
        self._last_leftcam_filter_debug_timestamp = -1.0
        self._last_overtake_fail_safe_log_timestamp = -1.0

        if path_to_conf_file:
            self._load_config(path_to_conf_file)
        if not self._controller_mode_explicit and self._controller_name.lower() in ("mpc", "mpc_lateral", "shooting_mpc"):
            self._controller_mode = "mpc"
        self._controller_mode = self._normalize_controller_mode(self._controller_mode)
        if self._controller_mode == "mpc" and self._controller_name == "basic_agent":
            self._controller_name = "mpc"
        self._active_target_speed = self._target_speed
        self._initialize_metrics_logger(path_to_conf_file)

    def sensors(self):
        """
        Define the sensor suite required by the agent.
        """
        return [
            {
                "type": "sensor.camera.rgb",
                "x": 2,
                "y": 0.0,
                "z": 1,
                "roll": 0.0,
                "pitch": self._forward_camera_pitch,
                "yaw": 0.0,
                "width": self.camera_width,
                "height": self.camera_height,
                "fov": 100,
                "id": "Center",
            },
            {
                "type": "sensor.speedometer",
                "reading_frequency": 20,
                "id": "Speed",
            },
            {
               "type": "sensor.camera.rgb",
                "x": 0.7,
                "y": -1.5,
                "z": 1.60,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": -45,
                "width": self.camera_width,
                "height": self.camera_height,
                "fov": 100,
                "id": "LeftCam",
            },
            {
               "type": "sensor.other.radar",
                "x": 2,
                "y": 0.0,
                "z": 1.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "horizontal_fov": 25,
                "vertical_fov": 10,
                "id": "FrontRdr",
            },
            {
               "type": "sensor.other.radar",
                "x": -1.7,
                "y": -1,
                "z": 1.60,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": -135,
                "horizontal_fov": 60,
                "vertical_fov": 20,
                "id": "RearRdr",
            },
            {
               "type": "sensor.camera.rgb",
                "x": -1.7,
                "y": -1.0,
                "z": 1.60,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": -135.0,
                "width": self.camera_width,
                "height": self.camera_height,
                "fov": 100,
                "id": "RearCam",
            }
        ]

    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.
        """
        self._current_timestamp = timestamp
        left_camera = input_data.get("LeftCam")
        rear_camera = input_data.get("RearCam")
        center_camera = input_data.get("Center")
        speed_data = input_data.get("Speed")
        front_radar = input_data.get("FrontRdr")
        rear_radar = input_data.get("RearRdr")

        center_image = center_camera[1] if center_camera is not None else None
        left_image = left_camera[1] if left_camera is not None else None
        rear_image = rear_camera[1] if rear_camera is not None else None
        yolo_detections = []
        speed_mps = speed_data[1]["speed"] if speed_data is not None else None
        frame_id = (
            speed_data[0]
            if speed_data is not None
            else center_camera[0]
            if center_camera is not None
            else -1
        )
        raw_front_radar_points = front_radar[1] if front_radar is not None else None
        raw_rear_radar_points = rear_radar[1] if rear_radar is not None else None
        front_radar_points = self._extract_front_cluster(raw_front_radar_points)
        rear_radar_points = self._extract_rear_cluster(raw_rear_radar_points)
        front_vehicle_state = self._estimate_front_vehicle_state(
            front_radar_points,
            raw_front_radar_points,
        )
        should_check_overtake = (
            front_vehicle_state["blocked"]
            or self._overtake_state != "follow_lane"
            or (
                front_vehicle_state["front_radar_depth"] is not None
                and front_vehicle_state["front_radar_depth"] <= self._front_vehicle_block_distance
            )
        )
        left_yolo_detections = self._run_yolo(left_image) if should_check_overtake else []
        left_oncoming = self._detect_left_oncoming_vehicle(left_yolo_detections)
        rear_approaching = self._detect_rear_approaching_vehicle(rear_radar_points)

        self._log_sensor_snapshot(
            center_image,
            left_yolo_detections,
            front_vehicle_state,
            left_oncoming,
            rear_approaching,
            left_image,
            rear_image,
            speed_data[1] if speed_data is not None else None,
            raw_front_radar_points,
            raw_rear_radar_points,
        )
        self._print_sensor_summary(
            timestamp,
            speed_mps,
            left_yolo_detections,
            front_vehicle_state,
            left_oncoming,
            rear_approaching,
            front_radar_points,
            rear_radar_points,
        )

        if self._agent is None:
            self._hero_actor = self._get_hero_actor()
            if self._hero_actor is None:
                return carla.VehicleControl()

            self._agent = BasicAgent(
                self._hero_actor,
                self._target_speed,
                opt_dict={"ignore_vehicles": True},
            )
            self._ensure_runtime_controller()
            self._active_target_speed = self._target_speed

        if not self._route_assigned:
            self._set_agent_route()
            if not self._route_assigned:
                return carla.VehicleControl()

        current_waypoint = self._update_overtake_state(
            timestamp,
            speed_mps,
            front_vehicle_state,
            left_oncoming,
            rear_approaching,
        )
        control = self._run_selected_controller(
            speed_mps,
            current_waypoint=current_waypoint,
            front_vehicle_state=front_vehicle_state,
        )
        control = self._apply_sensor_navigation(
            control,
            current_waypoint,
            front_vehicle_state,
            left_oncoming,
            speed_mps=speed_mps,
        )
        self._record_metrics(
            frame_id,
            speed_mps,
            current_waypoint,
            front_vehicle_state,
            rear_radar_points,
            rear_approaching,
            left_oncoming,
            control,
        )
        self._display.render(
            center_image=center_image,
            left_image=left_image,
            rear_image=rear_image,
            speed_mps=speed_mps,
            detections=[],
            left_detections=left_yolo_detections,
            navigation_state=self._overtake_state,
            front_radar=front_radar_points,
            rear_radar=rear_radar_points,
        )
        return control

    @staticmethod
    def _normalize_controller_mode(value):
        normalized = (value or "").strip().lower()
        if normalized in ("mpc", "mpc_lateral", "shooting_mpc"):
            return "mpc"
        return "basic_agent"

    @staticmethod
    def _compute_speed_mps_from_actor(actor):
        if actor is None:
            return None
        try:
            velocity = actor.get_velocity()
        except RuntimeError:
            return None
        return float(np.linalg.norm([velocity.x, velocity.y, velocity.z]))

    @staticmethod
    def _make_controller_diagnostics(compute_time_ms=0.0, fallback=False, status="", reference_count=0, candidate_count=0):
        return {
            "compute_time_ms": float(compute_time_ms or 0.0),
            "fallback": bool(fallback),
            "status": status or "",
            "reference_count": int(reference_count or 0),
            "candidate_count": int(candidate_count or 0),
        }

    def _set_controller_diagnostics(self, compute_time_ms=0.0, fallback=False, status="", reference_count=0, candidate_count=0):
        self._last_controller_diagnostics = self._make_controller_diagnostics(
            compute_time_ms=compute_time_ms,
            fallback=fallback,
            status=status,
            reference_count=reference_count,
            candidate_count=candidate_count,
        )

    def _set_longitudinal_controller_diagnostics(
        self,
        compute_time_ms=0.0,
        fallback=False,
        status="",
        reference_count=0,
        candidate_count=0,
    ):
        self._last_longitudinal_controller_diagnostics = self._make_controller_diagnostics(
            compute_time_ms=compute_time_ms,
            fallback=fallback,
            status=status,
            reference_count=reference_count,
            candidate_count=candidate_count,
        )

    def _ensure_runtime_controller(self):
        if self._controller_mode != "mpc":
            return

        if self._mpc_controller is None:
            self._mpc_controller = LateralMPCController(
                horizon_steps=self._mpc_horizon_steps,
                prediction_dt=self._mpc_prediction_dt,
                wheel_base_m=self._mpc_wheel_base_m,
                max_steer_delta_per_cycle=self._mpc_max_steer_delta_per_cycle,
                max_steer_angle_deg=self._mpc_max_steer_angle_deg,
            )

        if self._mpc_longitudinal_controller is None:
            self._mpc_longitudinal_controller = LongitudinalMPCController(
                horizon_steps=self._mpc_longitudinal_horizon_steps,
                prediction_dt=self._mpc_longitudinal_prediction_dt,
                max_accel_mps2=self._mpc_longitudinal_max_accel_mps2,
                max_decel_mps2=self._mpc_longitudinal_max_decel_mps2,
                max_accel_delta_per_cycle=self._mpc_longitudinal_max_accel_delta_per_cycle,
                stop_buffer_m=self._mpc_longitudinal_stop_buffer_m,
            )

    def _set_agent_target_speed(self, target_speed_kph):
        target_speed_kph = max(float(target_speed_kph or 0.0), 0.0)
        if self._agent is not None and (
            self._active_target_speed is None
            or abs(float(self._active_target_speed) - target_speed_kph) > 1e-3
        ):
            self._agent.set_target_speed(target_speed_kph)
        self._active_target_speed = target_speed_kph

    def _estimate_control_acceleration_hint(self, throttle, brake):
        throttle_value = float(np.clip(throttle, 0.0, 1.0))
        brake_value = float(np.clip(brake, 0.0, 1.0))
        return (
            throttle_value * self._mpc_longitudinal_max_accel_mps2
            - brake_value * self._mpc_longitudinal_max_decel_mps2
        )

    def _estimate_front_actor_speed_mps(self, front_vehicle_state, ego_speed_mps=None):
        if front_vehicle_state is None:
            return None

        ego_speed_value = 0.0 if ego_speed_mps is None else max(float(ego_speed_mps), 0.0)
        raw_front_velocity = front_vehicle_state.get("front_radar_velocity")
        if raw_front_velocity is None or not np.isfinite(raw_front_velocity):
            return None
        return max(ego_speed_value + float(raw_front_velocity), 0.0)

    def _resolve_mpc_phase_speed_target_mps(self, phase_name, base_target_mps, overtake_target_mps):
        if phase_name == "waiting_left":
            if self._mpc_waiting_left_target_speed_mps > 1e-3:
                return self._mpc_waiting_left_target_speed_mps
            return max(1.2, min(base_target_mps * 0.58, 3.2))

        if phase_name == "changing_left":
            if self._mpc_changing_left_target_speed_mps > 1e-3:
                return self._mpc_changing_left_target_speed_mps
            return max(2.0, min(overtake_target_mps * 0.82, 4.6))

        if phase_name == "passing":
            if self._mpc_passing_target_speed_mps > 1e-3:
                return self._mpc_passing_target_speed_mps
            return max(overtake_target_mps, base_target_mps)

        if phase_name == "changing_right":
            if self._mpc_changing_right_target_speed_mps > 1e-3:
                return self._mpc_changing_right_target_speed_mps
            return max(2.6, min(max(base_target_mps, overtake_target_mps * 0.88), overtake_target_mps))

        return base_target_mps

    def _build_mpc_longitudinal_reference(self, speed_mps, current_waypoint, front_vehicle_state):
        if self._controller_mode != "mpc":
            return None
        if self._overtake_state not in ("waiting_left", "changing_left", "passing", "changing_right"):
            return None

        speed_value = 0.0 if speed_mps is None else max(float(speed_mps), 0.0)
        base_target_mps = max(float(self._target_speed) / 3.6, 0.0)
        overtake_target_mps = max(float(self._overtake_speed) / 3.6, 0.0)
        front_safety = self._build_front_safety_profile(front_vehicle_state, speed_value)
        front_distance_m = front_safety["distance_m"]
        front_speed_mps = self._estimate_front_actor_speed_mps(front_vehicle_state, speed_value)
        target_speed_mps = base_target_mps
        status = self._overtake_state

        if self._overtake_state == "waiting_left":
            waiting_cap_mps = self._resolve_mpc_phase_speed_target_mps(
                "waiting_left",
                base_target_mps,
                overtake_target_mps,
            )
            target_speed_mps = waiting_cap_mps
            if front_distance_m is not None:
                safe_stop_distance_m = front_safety["dynamic_stop_distance_m"] + self._mpc_longitudinal_stop_buffer_m
                if self._waiting_left_reason == "left_oncoming":
                    safe_stop_distance_m += self._mpc_waiting_left_oncoming_extra_stop_buffer_m
                free_distance_m = max(front_distance_m - safe_stop_distance_m, 0.0)
                approach_speed_mps = float(
                    np.sqrt(max(2.0 * self._front_vehicle_comfort_decel_mps2 * free_distance_m, 0.0))
                )
                target_speed_mps = min(waiting_cap_mps, approach_speed_mps)
                if (
                    self._waiting_left_reason != "left_oncoming"
                    and (
                        front_safety["low_speed_restart_safe"]
                        and front_safety["dynamic_margin_m"] is not None
                        and front_safety["dynamic_margin_m"] > 0.2
                    )
                ):
                    target_speed_mps = max(target_speed_mps, min(self._mpc_waiting_left_creep_speed_mps, waiting_cap_mps))
            status = "waiting_left_{}".format(self._waiting_left_reason)

        elif self._overtake_state == "changing_left":
            same_lane_change = (
                current_waypoint is not None
                and self._overtake_origin_lane_id is not None
                and current_waypoint.lane_id == self._overtake_origin_lane_id
            )
            change_left_target_mps = self._resolve_mpc_phase_speed_target_mps(
                "changing_left",
                base_target_mps,
                overtake_target_mps,
            )
            target_speed_mps = change_left_target_mps
            if same_lane_change and front_distance_m is not None:
                launch_min_front_distance = max(
                    front_safety["min_hold_distance_m"] + 0.4,
                    min(
                        self._mpc_lane_change_launch_min_front_distance,
                        front_safety["dynamic_emergency_distance_m"],
                    ),
                )
                launch_free_distance = max(
                    launch_min_front_distance + 1.5,
                    front_safety["dynamic_stop_distance_m"] + 1.0,
                )
                available_span = max(launch_free_distance - launch_min_front_distance, 0.25)
                clearance_ratio = float(
                    np.clip((front_distance_m - launch_min_front_distance) / available_span, 0.0, 1.0)
                )
                minimum_launch_speed_mps = max(self._mpc_lane_change_resume_speed_mps, 0.6)
                target_speed_mps = minimum_launch_speed_mps + clearance_ratio * (
                    change_left_target_mps - minimum_launch_speed_mps
                )
                if self._is_mpc_lane_change_launch_window_active():
                    target_speed_mps = max(target_speed_mps, minimum_launch_speed_mps)
                status = "changing_left_same_lane"
            else:
                target_speed_mps = max(
                    change_left_target_mps,
                    self._resolve_mpc_phase_speed_target_mps("passing", base_target_mps, overtake_target_mps) * 0.92,
                )
                status = "changing_left_committed"

        elif self._overtake_state == "passing":
            target_speed_mps = self._resolve_mpc_phase_speed_target_mps(
                "passing",
                base_target_mps,
                overtake_target_mps,
            )
            status = "passing"

        elif self._overtake_state == "changing_right":
            target_speed_mps = self._resolve_mpc_phase_speed_target_mps(
                "changing_right",
                base_target_mps,
                overtake_target_mps,
            )
            status = "changing_right"

        target_speed_mps = max(float(target_speed_mps), 0.0)
        return {
            "target_speed_mps": target_speed_mps,
            "front_distance_m": front_distance_m,
            "front_speed_mps": front_speed_mps,
            "dynamic_stop_distance_m": front_safety["dynamic_stop_distance_m"],
            "dynamic_emergency_distance_m": front_safety["dynamic_emergency_distance_m"],
            "ttc_threshold_s": self._front_vehicle_ttc_brake_s,
            "status": status,
        }

    def _apply_mpc_longitudinal_control(
        self,
        control,
        speed_mps,
        current_control,
        longitudinal_reference,
    ):
        if longitudinal_reference is None:
            self._last_mpc_longitudinal_accel_cmd = 0.0
            self._set_longitudinal_controller_diagnostics(status="basic_agent_longitudinal")
            return control

        self._ensure_runtime_controller()
        if self._mpc_longitudinal_controller is None:
            self._last_mpc_longitudinal_accel_cmd = self._estimate_control_acceleration_hint(
                control.throttle,
                control.brake,
            )
            self._set_longitudinal_controller_diagnostics(
                fallback=True,
                status="mpc_longitudinal_unavailable",
            )
            return control

        current_accel_hint = 0.0
        if current_control is not None:
            current_accel_hint = self._estimate_control_acceleration_hint(
                current_control.throttle,
                current_control.brake,
            )

        start_time = time.perf_counter()
        try:
            result = self._mpc_longitudinal_controller.compute_control(
                speed_mps=speed_mps,
                target_speed_mps=longitudinal_reference["target_speed_mps"],
                current_accel_cmd=current_accel_hint,
                front_distance_m=longitudinal_reference["front_distance_m"],
                front_speed_mps=longitudinal_reference["front_speed_mps"],
                dynamic_stop_distance_m=longitudinal_reference["dynamic_stop_distance_m"],
                dynamic_emergency_distance_m=longitudinal_reference["dynamic_emergency_distance_m"],
                ttc_threshold_s=longitudinal_reference["ttc_threshold_s"],
                fallback_throttle=control.throttle,
                fallback_brake=control.brake,
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            print("WARNING: MPC longitudinal controller failed: {}".format(exc))
            self._last_mpc_longitudinal_accel_cmd = self._estimate_control_acceleration_hint(
                control.throttle,
                control.brake,
            )
            self._set_longitudinal_controller_diagnostics(
                compute_time_ms=elapsed_ms,
                fallback=True,
                status="mpc_longitudinal_exception",
            )
            return control

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        self._last_mpc_longitudinal_accel_cmd = float(result.acceleration_mps2)
        self._set_longitudinal_controller_diagnostics(
            compute_time_ms=elapsed_ms,
            fallback=False,
            status=longitudinal_reference["status"] + ":" + result.status,
            reference_count=1,
            candidate_count=result.candidate_count,
        )
        control.throttle = float(result.throttle)
        control.brake = float(result.brake)
        control.hand_brake = False
        return control

    def _run_selected_controller(self, speed_mps, current_waypoint=None, front_vehicle_state=None):
        if self._controller_mode != "mpc":
            base_control = self._agent.run_step()
            self._set_controller_diagnostics(status="basic_agent_pid")
            self._set_longitudinal_controller_diagnostics(status="basic_agent_longitudinal")
            return base_control

        longitudinal_reference = self._build_mpc_longitudinal_reference(
            speed_mps,
            current_waypoint,
            front_vehicle_state,
        )
        if longitudinal_reference is not None:
            self._set_agent_target_speed(longitudinal_reference["target_speed_mps"] * 3.6)
        else:
            self._set_agent_target_speed(self._target_speed)
            self._set_longitudinal_controller_diagnostics(status="basic_agent_longitudinal")

        base_control = self._agent.run_step()

        self._ensure_runtime_controller()
        if self._mpc_controller is None:
            self._set_controller_diagnostics(fallback=True, status="mpc_unavailable")
            if longitudinal_reference is not None:
                self._set_longitudinal_controller_diagnostics(
                    fallback=True,
                    status=longitudinal_reference["status"] + ":mpc_unavailable",
                )
            return base_control

        if (
            base_control.brake >= self._basic_agent_hazard_brake_threshold
            and base_control.throttle <= 1e-3
        ):
            self._set_controller_diagnostics(fallback=True, status="basic_agent_hazard_stop")
            if longitudinal_reference is not None:
                self._set_longitudinal_controller_diagnostics(
                    fallback=True,
                    status=longitudinal_reference["status"] + ":basic_agent_hazard_stop",
                )
            return base_control

        local_planner = self._agent.get_local_planner()
        plan = list(local_planner.get_plan()) if local_planner is not None else []
        if not plan:
            self._set_controller_diagnostics(fallback=True, status="planner_empty")
            if longitudinal_reference is not None:
                self._set_longitudinal_controller_diagnostics(
                    fallback=True,
                    status=longitudinal_reference["status"] + ":planner_empty",
                )
            return base_control

        hero_actor = self._hero_actor or self._get_hero_actor()
        if hero_actor is None:
            self._set_controller_diagnostics(fallback=True, status="hero_unavailable")
            if longitudinal_reference is not None:
                self._set_longitudinal_controller_diagnostics(
                    fallback=True,
                    status=longitudinal_reference["status"] + ":hero_unavailable",
                )
            return base_control

        try:
            transform = hero_actor.get_transform()
            current_control = hero_actor.get_control()
        except RuntimeError:
            self._set_controller_diagnostics(fallback=True, status="hero_state_unavailable")
            if longitudinal_reference is not None:
                self._set_longitudinal_controller_diagnostics(
                    fallback=True,
                    status=longitudinal_reference["status"] + ":hero_state_unavailable",
                )
            return base_control

        effective_speed_mps = speed_mps if speed_mps is not None else self._compute_speed_mps_from_actor(hero_actor)
        start_time = time.perf_counter()
        try:
            result = self._mpc_controller.compute_steer(
                vehicle_transform=transform,
                speed_mps=effective_speed_mps,
                plan=plan,
                current_steer=current_control.steer,
                fallback_steer=base_control.steer,
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            print("WARNING: MPC controller failed: {}".format(exc))
            self._set_controller_diagnostics(
                compute_time_ms=elapsed_ms,
                fallback=True,
                status="mpc_exception",
            )
            if longitudinal_reference is not None:
                self._set_longitudinal_controller_diagnostics(
                    fallback=True,
                    status=longitudinal_reference["status"] + ":mpc_exception",
                )
            return base_control

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        self._set_controller_diagnostics(
            compute_time_ms=elapsed_ms,
            fallback=False,
            status=result.status,
            reference_count=result.reference_count,
            candidate_count=result.candidate_count,
        )
        base_control.steer = self._shape_mpc_steer_command(
            raw_steer=result.steer,
            current_steer=current_control.steer,
            current_waypoint=current_waypoint,
            front_vehicle_state=front_vehicle_state,
            vehicle_transform=transform,
            speed_mps=effective_speed_mps,
        )
        base_control = self._apply_mpc_longitudinal_control(
            base_control,
            effective_speed_mps,
            current_control,
            longitudinal_reference,
        )
        return base_control

    def _shape_mpc_steer_command(
        self,
        raw_steer,
        current_steer,
        current_waypoint,
        front_vehicle_state,
        vehicle_transform=None,
        speed_mps=None,
    ):
        target_steer = float(np.clip(raw_steer, -1.0, 1.0))
        step_limit = self._mpc_follow_steer_step_limit
        max_abs_steer = 1.0
        same_lane_left_change = (
            self._overtake_state == "changing_left"
            and current_waypoint is not None
            and self._overtake_origin_lane_id is not None
            and current_waypoint.lane_id == self._overtake_origin_lane_id
        )
        same_lane_right_change = (
            self._overtake_state == "changing_right"
            and current_waypoint is not None
            and self._overtake_origin_lane_id is not None
            and current_waypoint.lane_id == self._overtake_origin_lane_id
        )

        if self._overtake_state in ("changing_left", "changing_right"):
            step_limit = self._mpc_lane_change_steer_step_limit

        if same_lane_left_change:
            max_abs_steer = self._mpc_same_lane_change_steer_limit
        elif same_lane_right_change:
            max_abs_steer = min(max_abs_steer, self._mpc_rejoin_steer_limit)

        target_steer = float(np.clip(target_steer, -max_abs_steer, max_abs_steer))
        steer_delta = float(target_steer - current_steer)
        steer_delta = float(np.clip(steer_delta, -step_limit, step_limit))
        smoothed_steer = float(current_steer + steer_delta)

        front_distance_m = self._get_front_clearance_m(front_vehicle_state)
        if same_lane_left_change and front_distance_m is not None and front_distance_m <= self._front_vehicle_emergency_distance + 0.2:
            close_front_limit = min(self._mpc_same_lane_change_steer_limit, self._mpc_close_front_steer_limit)
            smoothed_steer = float(np.clip(smoothed_steer, -close_front_limit, close_front_limit))

        target_lane_metrics = self._compute_target_lane_tracking_metrics(current_waypoint, vehicle_transform)
        current_lane_metrics = self._compute_current_lane_tracking_metrics(current_waypoint, vehicle_transform)
        if same_lane_left_change and self._should_hold_mpc_lane_change_commitment(
            front_vehicle_state,
            target_lane_metrics,
            current_lane_metrics,
            speed_mps,
        ):
            commitment_steer = -self._compute_mpc_lane_change_commitment_steer(target_lane_metrics)
            smoothed_steer = min(smoothed_steer, commitment_steer)
            smoothed_steer = self._shape_mpc_left_change_release_steer(
                smoothed_steer,
                current_lane_metrics,
                target_lane_metrics,
                speed_mps,
            )
        elif same_lane_left_change:
            smoothed_steer = self._shape_mpc_left_change_release_steer(
                smoothed_steer,
                current_lane_metrics,
                target_lane_metrics,
                speed_mps,
            )
        if same_lane_right_change:
            smoothed_steer = self._shape_mpc_rejoin_settle_steer(smoothed_steer, current_lane_metrics)

        return smoothed_steer

    def _should_hold_mpc_lane_change_commitment(
        self,
        front_vehicle_state,
        target_lane_metrics,
        current_lane_metrics=None,
        speed_mps=None,
    ):
        if self._controller_mode != "mpc":
            return False
        if self._overtake_state != "changing_left":
            return False

        if self._should_release_mpc_left_change_steer(current_lane_metrics, target_lane_metrics, speed_mps):
            return False

        if not target_lane_metrics:
            return True
        progress = target_lane_metrics.get("progress")
        target_lane_lateral_abs_m = target_lane_metrics.get("lateral_error_abs_m")
        if progress is None or target_lane_lateral_abs_m is None:
            return True

        if target_lane_lateral_abs_m > self._mpc_lane_change_commitment_release_lateral_m:
            return True
        if progress < 0.93:
            return True

        front_distance_m = self._get_front_clearance_m(front_vehicle_state)
        if front_distance_m is None:
            return False
        return front_distance_m <= self._mpc_lane_change_commitment_release_distance_m

    def _should_release_mpc_left_change_steer(self, current_lane_metrics, target_lane_metrics, speed_mps):
        if not current_lane_metrics:
            return False

        speed_mps = float(speed_mps or 0.0)
        heading_error_abs_deg = current_lane_metrics.get("heading_error_abs_deg")
        lateral_error_abs_m = current_lane_metrics.get("lateral_error_abs_m")
        progress = None if not target_lane_metrics else target_lane_metrics.get("progress")
        if heading_error_abs_deg is None or lateral_error_abs_m is None:
            return False

        if speed_mps < self._mpc_left_release_speed_mps:
            return False

        return (
            heading_error_abs_deg >= self._mpc_left_release_heading_deg
            or lateral_error_abs_m >= self._mpc_left_release_lateral_error_m
            or (progress is not None and progress >= self._mpc_left_release_progress)
        )

    def _shape_mpc_left_change_release_steer(
        self,
        smoothed_steer,
        current_lane_metrics,
        target_lane_metrics,
        speed_mps,
    ):
        if not current_lane_metrics:
            return smoothed_steer

        speed_mps = float(speed_mps or 0.0)
        heading_error_abs_deg = current_lane_metrics.get("heading_error_abs_deg")
        lateral_error_abs_m = current_lane_metrics.get("lateral_error_abs_m")
        progress = None if not target_lane_metrics else target_lane_metrics.get("progress")
        if heading_error_abs_deg is None or lateral_error_abs_m is None:
            return smoothed_steer

        limited_steer = float(smoothed_steer)
        if self._should_release_mpc_left_change_steer(current_lane_metrics, target_lane_metrics, speed_mps):
            limited_steer = max(limited_steer, -self._mpc_left_release_max_steer)

        if (
            speed_mps >= self._mpc_left_final_release_speed_mps
            and (
                heading_error_abs_deg >= self._mpc_left_final_release_heading_deg
                or lateral_error_abs_m >= self._mpc_left_final_release_lateral_error_m
                or (progress is not None and progress >= self._mpc_left_final_release_progress)
            )
        ):
            limited_steer = max(limited_steer, -self._mpc_left_final_release_max_steer)

        return limited_steer

    def _compute_mpc_lane_change_commitment_steer(self, target_lane_metrics):
        progress = None if not target_lane_metrics else target_lane_metrics.get("progress")
        max_commitment_steer = min(
            self._mpc_same_lane_change_steer_limit,
            self._mpc_close_front_steer_limit,
            0.80,
        )
        if progress is None:
            return max_commitment_steer

        if progress < 0.35:
            return max_commitment_steer
        if progress < 0.55:
            return min(max_commitment_steer, 0.72)
        if progress < 0.72:
            return min(max_commitment_steer, 0.62)
        if progress < 0.85:
            return min(max_commitment_steer, 0.48)
        return min(max_commitment_steer, max(self._mpc_lane_change_commitment_min_steer, 0.30))

    def _get_target_waypoint_for_active_maneuver(self, current_waypoint):
        if current_waypoint is None:
            return None

        if self._overtake_state in ("waiting_left", "changing_left"):
            return self._get_adjacent_driving_lane(current_waypoint, "left")

        if self._overtake_state == "changing_right":
            rejoin_waypoint, _rejoin_direction = self._find_origin_adjacent_lane(current_waypoint)
            return rejoin_waypoint

        return None

    def _compute_target_lane_tracking_metrics(self, current_waypoint, ego_transform):
        target_waypoint = self._get_target_waypoint_for_active_maneuver(current_waypoint)
        if target_waypoint is None or ego_transform is None:
            return {}

        ego_location = ego_transform.location
        target_location = target_waypoint.transform.location
        delta = np.array(
            [
                ego_location.x - target_location.x,
                ego_location.y - target_location.y,
                ego_location.z - target_location.z,
            ],
            dtype=np.float64,
        )
        target_right_vector = target_waypoint.transform.get_right_vector()
        target_right_np = np.array(
            [target_right_vector.x, target_right_vector.y, target_right_vector.z],
            dtype=np.float64,
        )
        lateral_error_m = float(np.dot(delta, target_right_np))
        heading_error_deg = self._normalize_angle_deg(
            ego_transform.rotation.yaw - target_waypoint.transform.rotation.yaw
        )

        lane_spacing_m = current_waypoint.transform.location.distance(target_location)
        if not np.isfinite(lane_spacing_m) or lane_spacing_m <= 1e-3:
            lane_spacing_m = None
        progress = None
        if lane_spacing_m is not None:
            progress = float(np.clip(1.0 - (abs(lateral_error_m) / lane_spacing_m), 0.0, 1.0))

        return {
            "target_lane_id": target_waypoint.lane_id,
            "lateral_error_m": lateral_error_m,
            "lateral_error_abs_m": abs(lateral_error_m),
            "heading_error_deg": heading_error_deg,
            "lane_spacing_m": lane_spacing_m,
            "progress": progress,
        }

    def _compute_current_lane_tracking_metrics(self, current_waypoint, ego_transform):
        if current_waypoint is None or ego_transform is None:
            return {}

        lateral_error_m, heading_error_deg = self._compute_tracking_errors(current_waypoint, ego_transform)
        if lateral_error_m is None or heading_error_deg is None:
            return {}

        return {
            "lateral_error_m": lateral_error_m,
            "lateral_error_abs_m": abs(lateral_error_m),
            "heading_error_deg": heading_error_deg,
            "heading_error_abs_deg": abs(heading_error_deg),
        }

    def _shape_mpc_rejoin_settle_steer(self, smoothed_steer, current_lane_metrics):
        if not current_lane_metrics:
            return float(np.clip(smoothed_steer, -self._mpc_rejoin_steer_limit, self._mpc_rejoin_steer_limit))

        heading_error_abs_deg = current_lane_metrics.get("heading_error_abs_deg")
        lateral_error_abs_m = current_lane_metrics.get("lateral_error_abs_m")
        limited_steer = float(np.clip(smoothed_steer, -self._mpc_rejoin_steer_limit, self._mpc_rejoin_steer_limit))

        if (
            heading_error_abs_deg is not None
            and lateral_error_abs_m is not None
            and heading_error_abs_deg <= self._mpc_rejoin_soft_heading_deg
            and lateral_error_abs_m <= self._mpc_rejoin_soft_lateral_error_m
        ):
            limited_steer = float(np.clip(limited_steer, -0.25, 0.25))
            limited_steer *= 0.65

        if (
            heading_error_abs_deg is not None
            and lateral_error_abs_m is not None
            and heading_error_abs_deg <= self._mpc_rejoin_hard_heading_deg
            and lateral_error_abs_m <= self._mpc_rejoin_hard_lateral_error_m
        ):
            limited_steer = float(np.clip(limited_steer, -0.12, 0.12))
            limited_steer *= 0.5

        if (
            heading_error_abs_deg is not None
            and lateral_error_abs_m is not None
            and heading_error_abs_deg <= 2.0
            and lateral_error_abs_m <= 0.12
        ):
            limited_steer = 0.0

        return limited_steer

    def _is_rejoin_stabilized(self, current_waypoint):
        if current_waypoint is None or self._overtake_origin_lane_id is None:
            return False
        if current_waypoint.lane_id != self._overtake_origin_lane_id:
            return False

        hero_actor = self._hero_actor or self._get_hero_actor()
        if hero_actor is None:
            return False

        try:
            ego_transform = hero_actor.get_transform()
            current_control = hero_actor.get_control()
        except RuntimeError:
            return False

        current_lane_metrics = self._compute_current_lane_tracking_metrics(current_waypoint, ego_transform)
        if not current_lane_metrics:
            return False

        if current_lane_metrics["heading_error_abs_deg"] > self._mpc_rejoin_finish_max_heading_error_deg:
            return False
        if current_lane_metrics["lateral_error_abs_m"] > self._mpc_rejoin_finish_max_lateral_error_m:
            return False
        if abs(float(current_control.steer)) > self._mpc_rejoin_finish_max_abs_steer:
            return False

        return True

    def _log_sensor_snapshot(self, center_image, left_yolo_detections,
                             front_vehicle_state,
                             left_oncoming,
                             rear_approaching,
                             left_image, rear_image, speed_data,
                             front_radar_points, rear_radar_points):
        if self._sensor_snapshot_logged:
            return

        if center_image is not None:
            print("[Sensors] Center image shape={} dtype={}".format(center_image.shape, center_image.dtype))
        if left_yolo_detections:
            print("[YOLO] LeftCam detections={}".format(self._summarize_yolo(left_yolo_detections)))
        if front_vehicle_state["blocked"]:
            print("[Overtake] Front vehicle blocked at {:.2f} m".format(front_vehicle_state["distance_m"]))
        if left_oncoming is not None:
            print("[Overtake] Left lane oncoming candidate={} {:.2f}".format(
                left_oncoming["label"], left_oncoming["confidence"]))
        if rear_approaching is not None:
            print("[Overtake] Rear approach dist={:.2f} m vel={:.2f} m/s".format(
                rear_approaching["distance_m"], rear_approaching["velocity_mps"]))
        if left_image is not None:
            print("[Sensors] LeftCam image shape={} dtype={}".format(left_image.shape, left_image.dtype))
        if rear_image is not None:
            print("[Sensors] RearCam image shape={} dtype={}".format(rear_image.shape, rear_image.dtype))
        if speed_data is not None:
            print("[Sensors] Speed payload={}".format(speed_data))
        if front_radar_points is not None:
            print(
                "[Sensors] FrontRdr shape={} dtype={} sample={}".format(
                    front_radar_points.shape,
                    front_radar_points.dtype,
                    front_radar_points[:3],
                )
            )
        if rear_radar_points is not None:
            print(
                "[Sensors] RearRdr shape={} dtype={} sample={}".format(
                    rear_radar_points.shape,
                    rear_radar_points.dtype,
                    rear_radar_points[:3],
                )
            )

        self._sensor_snapshot_logged = True

    def _print_sensor_summary(self, timestamp, speed_mps, left_yolo_detections,
                              front_vehicle_state, left_oncoming, rear_approaching,
                              front_radar_points, rear_radar_points):
        if timestamp - self._last_debug_timestamp < 1.0:
            return

        self._last_debug_timestamp = timestamp
        speed_text = "unavailable"
        if speed_mps is not None:
            speed_text = "{:.2f} m/s ({:.1f} km/h)".format(speed_mps, speed_mps * 3.6)

        print(
            "[Sensors] Speed={} | LeftYOLO={} | Front={} | LeftOncoming={} | RearApproach={} | Maneuver={} | FrontRdr={} | RearRdr={}".format(
                speed_text,
                self._summarize_yolo(left_yolo_detections),
                self._summarize_front_vehicle_state(front_vehicle_state),
                self._summarize_oncoming(left_oncoming),
                self._summarize_rear_approach(rear_approaching),
                self._summarize_maneuver_state(),
                self._summarize_radar(front_radar_points),
                self._summarize_radar(rear_radar_points),
            )
        )

    @staticmethod
    def _apply_brake_override(control, brake_value):
        control.throttle = 0.0
        control.brake = max(control.brake, brake_value)
        control.hand_brake = False
        return control

    @staticmethod
    def _compute_ttc_s(distance_m, closing_speed_mps):
        if distance_m is None or closing_speed_mps is None:
            return None

        distance_value = float(distance_m)
        closing_speed_value = float(closing_speed_mps)
        if not np.isfinite(distance_value) or not np.isfinite(closing_speed_value):
            return None
        if distance_value <= 0.0 or closing_speed_value <= 1e-3:
            return None

        return distance_value / closing_speed_value

    def _build_front_safety_profile(self, front_vehicle_state, ego_speed_mps=None):
        distance_m = self._get_front_clearance_m(front_vehicle_state)
        raw_front_velocity = front_vehicle_state.get("front_radar_velocity") if front_vehicle_state else None
        closing_speed_mps = None
        if raw_front_velocity is not None and np.isfinite(raw_front_velocity):
            closing_speed_mps = max(-float(raw_front_velocity), 0.0)

        if closing_speed_mps is None:
            ego_speed_mps = 0.0 if ego_speed_mps is None else max(float(ego_speed_mps), 0.0)
            closing_speed_mps = ego_speed_mps

        ego_speed_value = 0.0 if ego_speed_mps is None else max(float(ego_speed_mps), 0.0)
        reaction_distance_m = closing_speed_mps * self._front_vehicle_reaction_time_s
        braking_distance_m = (
            (closing_speed_mps * closing_speed_mps) / (2.0 * max(self._front_vehicle_comfort_decel_mps2, 1e-3))
        )
        min_hold_distance_m = self._front_vehicle_min_hold_distance
        dynamic_stop_distance_m = min_hold_distance_m + reaction_distance_m + 0.55 * braking_distance_m
        dynamic_emergency_distance_m = min_hold_distance_m + reaction_distance_m + braking_distance_m
        ttc_s = self._compute_ttc_s(distance_m, closing_speed_mps)

        low_speed_restart_safe = (
            distance_m is not None
            and distance_m > (min_hold_distance_m + 0.15)
            and ego_speed_value <= self._front_vehicle_restart_speed_mps
            and (ttc_s is None or ttc_s > self._front_vehicle_ttc_emergency_s)
        )
        collision_risk = False
        if distance_m is not None and distance_m <= dynamic_emergency_distance_m:
            collision_risk = True
        if ttc_s is not None and ttc_s <= self._front_vehicle_ttc_emergency_s:
            collision_risk = True

        return {
            "distance_m": distance_m,
            "closing_speed_mps": closing_speed_mps,
            "ttc_s": ttc_s,
            "reaction_distance_m": reaction_distance_m,
            "braking_distance_m": braking_distance_m,
            "min_hold_distance_m": min_hold_distance_m,
            "dynamic_stop_distance_m": dynamic_stop_distance_m,
            "dynamic_emergency_distance_m": dynamic_emergency_distance_m,
            "dynamic_margin_m": (
                None
                if distance_m is None
                else float(distance_m - dynamic_emergency_distance_m)
            ),
            "low_speed_restart_safe": low_speed_restart_safe,
            "collision_risk": collision_risk,
        }

    def _should_block_lane_change_for_front_obstacle(self, front_vehicle_state, ego_speed_mps=None):
        profile = self._build_front_safety_profile(front_vehicle_state, ego_speed_mps)
        distance_m = profile["distance_m"]
        if distance_m is None:
            return False
        if distance_m <= profile["min_hold_distance_m"]:
            return True
        if profile["ttc_s"] is not None and profile["ttc_s"] <= self._front_vehicle_ttc_emergency_s:
            return True

        ego_speed_value = 0.0 if ego_speed_mps is None else max(float(ego_speed_mps), 0.0)
        if ego_speed_value > self._front_vehicle_restart_speed_mps:
            return distance_m <= profile["dynamic_emergency_distance_m"]

        return False

    def _is_front_obstacle_active(self, front_vehicle_state):
        if front_vehicle_state is None:
            return False
        if self._overtake_cooldown_until is not None and self._current_timestamp < self._overtake_cooldown_until:
            return False
        if front_vehicle_state["blocked"]:
            return True

        if self._blocked_vehicle_since is None:
            return False

        front_radar_depth = front_vehicle_state.get("front_radar_depth")
        if front_radar_depth is None:
            return False

        return front_radar_depth <= self._front_radar_resume_distance

    def _compute_front_obstacle_brake(self, front_vehicle_state, ego_speed_mps=None):
        if front_vehicle_state is None:
            return self._sensor_stop_brake

        profile = self._build_front_safety_profile(front_vehicle_state, ego_speed_mps)
        front_radar_depth = profile["distance_m"]
        if front_radar_depth is None:
            return self._sensor_stop_brake

        if front_radar_depth <= profile["min_hold_distance_m"]:
            return self._sensor_close_brake
        if profile["ttc_s"] is not None and profile["ttc_s"] <= self._front_vehicle_ttc_emergency_s:
            return self._sensor_close_brake
        if front_radar_depth <= profile["dynamic_emergency_distance_m"]:
            return self._sensor_close_brake

        ego_speed_value = 0.0 if ego_speed_mps is None else max(float(ego_speed_mps), 0.0)
        if (
            ego_speed_value <= self._front_vehicle_low_speed_mps
            and front_radar_depth > profile["dynamic_stop_distance_m"] + 0.2
        ):
            return 0.0

        trigger_distance = max(
            self._front_radar_trigger_distance,
            profile["dynamic_stop_distance_m"] + 1.2,
        )
        distance_span = max(trigger_distance - profile["dynamic_stop_distance_m"], 0.1)
        closeness = float(np.clip((trigger_distance - front_radar_depth) / distance_span, 0.0, 1.0))
        ttc_pressure = 0.0
        if profile["ttc_s"] is not None and profile["ttc_s"] < self._front_vehicle_ttc_brake_s:
            ttc_span = max(
                self._front_vehicle_ttc_brake_s - self._front_vehicle_ttc_emergency_s,
                0.1,
            )
            ttc_pressure = float(
                np.clip((self._front_vehicle_ttc_brake_s - profile["ttc_s"]) / ttc_span, 0.0, 1.0)
            )
        brake_pressure = max(closeness, ttc_pressure)

        return self._sensor_soft_brake + brake_pressure * (self._sensor_stop_brake - self._sensor_soft_brake)

    def _should_force_mpc_waiting_left_brake_override(self, front_vehicle_state, ego_speed_mps=None):
        if self._controller_mode != "mpc" or self._overtake_state != "waiting_left":
            return True

        profile = self._build_front_safety_profile(front_vehicle_state, ego_speed_mps)
        front_distance_m = profile["distance_m"]
        if front_distance_m is None:
            return True
        if front_distance_m <= profile["min_hold_distance_m"]:
            return True
        if profile["ttc_s"] is not None and profile["ttc_s"] <= self._front_vehicle_ttc_emergency_s:
            return True
        if front_distance_m <= profile["dynamic_emergency_distance_m"]:
            return True
        return False

    def _apply_sensor_navigation(self, control, current_waypoint, front_vehicle_state, left_oncoming, speed_mps=None):
        self._set_safety_intervention(False, "", 0.0)
        launch_assist_applied = False
        if self._overtake_state in ("follow_lane", "waiting_left"):
            if self._is_front_obstacle_active(front_vehicle_state):
                brake_value = self._compute_front_obstacle_brake(front_vehicle_state, ego_speed_mps=speed_mps)
                brake_value = self._adjust_waiting_left_brake(
                    brake_value,
                    front_vehicle_state,
                    ego_speed_mps=speed_mps,
                )
                if (
                    self._controller_mode == "mpc"
                    and self._overtake_state == "waiting_left"
                    and not self._should_force_mpc_waiting_left_brake_override(
                        front_vehicle_state,
                        ego_speed_mps=speed_mps,
                    )
                ):
                    brake_value = 0.0
                if brake_value > 1e-3:
                    self._set_safety_intervention(True, "front_obstacle", brake_value)
                    return self._apply_brake_override(control, brake_value)

        if self._overtake_state == "changing_left" and current_waypoint is not None:
            if self._overtake_origin_lane_id is not None and current_waypoint.lane_id == self._overtake_origin_lane_id:
                if self._should_apply_mpc_lane_change_launch_assist(
                    control,
                    current_waypoint,
                    front_vehicle_state,
                    ego_speed_mps=speed_mps,
                ):
                    control = self._apply_mpc_lane_change_launch_assist(
                        control,
                        front_vehicle_state,
                        ego_speed_mps=speed_mps,
                    )
                    launch_assist_applied = True
                if self._should_block_lane_change_for_front_obstacle(front_vehicle_state, ego_speed_mps=speed_mps):
                    if (not launch_assist_applied) and self._should_apply_mpc_lane_change_launch_assist(
                        control,
                        current_waypoint,
                        front_vehicle_state,
                        ego_speed_mps=speed_mps,
                    ):
                        control = self._apply_mpc_lane_change_launch_assist(
                            control,
                            front_vehicle_state,
                            ego_speed_mps=speed_mps,
                        )
                        launch_assist_applied = True
                    else:
                        self._set_safety_intervention(True, "front_emergency", self._sensor_close_brake)
                        return self._apply_brake_override(control, self._sensor_close_brake)

        if self._overtake_state in ("changing_left", "passing"):
            if left_oncoming is not None:
                self._log_overtake_fail_safe("left_oncoming", front_vehicle_state, left_oncoming)
                self._set_safety_intervention(True, "left_oncoming", self._sensor_emergency_brake)
                return self._apply_brake_override(control, self._sensor_emergency_brake)

            front_radar_depth = front_vehicle_state.get("front_radar_depth") if front_vehicle_state else None
            front_radar_velocity = front_vehicle_state.get("front_radar_velocity") if front_vehicle_state else None
            if (
                front_radar_depth is not None
                and front_radar_velocity is not None
                and front_radar_depth <= self._overtake_fail_safe_distance
                and front_radar_velocity <= self._overtake_fail_safe_velocity
            ):
                self._log_overtake_fail_safe("front_radar_closing", front_vehicle_state, left_oncoming)
                self._set_safety_intervention(True, "front_radar_closing", self._sensor_emergency_brake)
                return self._apply_brake_override(control, self._sensor_emergency_brake)

        if launch_assist_applied:
            self._set_safety_intervention(True, "mpc_lane_change_launch", 0.0)
        return control

    @staticmethod
    def _get_front_clearance_m(front_vehicle_state):
        if not front_vehicle_state:
            return None

        distance_candidates = []
        for key in ("distance_m", "front_radar_depth"):
            value = front_vehicle_state.get(key)
            if value is not None:
                distance_candidates.append(float(value))

        if not distance_candidates:
            return None
        return min(distance_candidates)

    def _should_apply_mpc_lane_change_launch_assist(
        self,
        control,
        current_waypoint,
        front_vehicle_state,
        ego_speed_mps=None,
    ):
        if self._controller_mode != "mpc":
            return False
        if self._overtake_state != "changing_left":
            return False
        if self._overtake_started_at is None:
            return False
        if current_waypoint is None or self._overtake_origin_lane_id is None:
            return False
        if current_waypoint.lane_id != self._overtake_origin_lane_id:
            return False
        if not self._is_mpc_lane_change_launch_window_active():
            return False
        if abs(float(control.steer)) < self._mpc_lane_change_launch_min_steer:
            return False

        front_safety = self._build_front_safety_profile(front_vehicle_state, ego_speed_mps)
        front_distance_m = front_safety["distance_m"]
        if front_distance_m is None:
            return False
        launch_min_front_distance = max(
            front_safety["min_hold_distance_m"] + 0.4,
            min(
                self._mpc_lane_change_launch_min_front_distance,
                front_safety["dynamic_emergency_distance_m"],
            ),
        )
        if front_distance_m < launch_min_front_distance:
            return False

        return True

    def _adjust_waiting_left_brake(self, brake_value, front_vehicle_state, ego_speed_mps=None):
        if self._controller_mode != "mpc":
            return brake_value
        if self._overtake_state != "waiting_left":
            return brake_value

        front_safety = self._build_front_safety_profile(front_vehicle_state, ego_speed_mps)
        front_distance_m = self._get_front_clearance_m(front_vehicle_state)
        if front_distance_m is None:
            return brake_value

        if (
            front_safety["low_speed_restart_safe"]
            and front_safety["dynamic_margin_m"] is not None
            and front_safety["dynamic_margin_m"] > 0.2
        ):
            return min(float(brake_value), self._sensor_soft_brake)

        dynamic_waiting_stop_distance = max(
            self._mpc_waiting_left_close_stop_distance,
            front_safety["dynamic_stop_distance_m"] + 0.2,
        )
        if front_distance_m <= dynamic_waiting_stop_distance:
            return max(float(brake_value), self._mpc_waiting_left_close_stop_brake)
        return brake_value

    def _is_mpc_lane_change_launch_window_active(self):
        if self._overtake_started_at is None:
            return False
        if self._current_timestamp - self._overtake_started_at <= self._mpc_lane_change_launch_grace_time:
            return True
        return self._is_ego_nearly_stationary(self._mpc_lane_change_resume_speed_mps)

    def _is_ego_nearly_stationary(self, speed_threshold_mps):
        hero_actor = self._hero_actor or self._get_hero_actor()
        speed_mps = self._compute_speed_mps_from_actor(hero_actor)
        if speed_mps is None:
            return False
        return speed_mps <= max(float(speed_threshold_mps), 0.0)

    def _apply_mpc_lane_change_launch_assist(self, control, front_vehicle_state, ego_speed_mps=None):
        launch_throttle = self._mpc_lane_change_launch_throttle
        front_safety = self._build_front_safety_profile(front_vehicle_state, ego_speed_mps)
        front_distance_m = front_safety["distance_m"]
        if front_distance_m is not None:
            launch_min_front_distance = max(
                front_safety["min_hold_distance_m"] + 0.4,
                min(
                    self._mpc_lane_change_launch_min_front_distance,
                    front_safety["dynamic_emergency_distance_m"],
                ),
            )
            launch_max_front_distance = max(
                launch_min_front_distance + 0.5,
                front_safety["dynamic_stop_distance_m"] + 0.8,
                self._mpc_lane_change_launch_min_front_distance,
            )
            available_span = max(launch_max_front_distance - launch_min_front_distance, 0.25)
            clearance_ratio = np.clip(
                (front_distance_m - launch_min_front_distance) / available_span,
                0.0,
                1.0,
            )
            launch_throttle = (
                self._mpc_lane_change_launch_min_throttle
                + clearance_ratio * (self._mpc_lane_change_launch_throttle - self._mpc_lane_change_launch_min_throttle)
            )

        control.brake = 0.0
        control.throttle = max(float(control.throttle), float(launch_throttle))
        control.hand_brake = False
        return control

    def _log_overtake_fail_safe(self, reason, front_vehicle_state, left_oncoming):
        if self._current_timestamp - self._last_overtake_fail_safe_log_timestamp < 0.5:
            return

        self._last_overtake_fail_safe_log_timestamp = self._current_timestamp
        front_radar_depth = front_vehicle_state.get("front_radar_depth") if front_vehicle_state else None
        front_radar_velocity = front_vehicle_state.get("front_radar_velocity") if front_vehicle_state else None
        oncoming_text = self._summarize_oncoming(left_oncoming)
        front_text = "clear"
        if front_radar_depth is not None:
            if front_radar_velocity is None:
                front_text = "{:.1f}m".format(front_radar_depth)
            else:
                front_text = "{:.1f}m {:.1f}m/s".format(front_radar_depth, front_radar_velocity)

        print(
            "[OvertakeFailSafe] state={} reason={} oncoming={} front={}".format(
                self._overtake_state,
                reason,
                oncoming_text,
                front_text,
            )
        )

    def _get_current_waypoint(self):
        hero_actor = self._hero_actor or self._get_hero_actor()
        carla_map = CarlaDataProvider.get_map()
        if hero_actor is None or carla_map is None:
            return None
        return carla_map.get_waypoint(hero_actor.get_location(), lane_type=carla.LaneType.Driving)

    def _update_overtake_state(self, timestamp, speed_mps, front_vehicle_state, left_oncoming, rear_approaching):
        del speed_mps
        current_waypoint = self._get_current_waypoint()
        if self._overtake_cooldown_until is not None and timestamp >= self._overtake_cooldown_until:
            self._overtake_cooldown_until = None
        front_obstacle_active = self._is_front_obstacle_active(front_vehicle_state)

        if self._overtake_state in ("follow_lane", "waiting_left"):
            if front_obstacle_active:
                self._blocked_vehicle_since = timestamp if self._blocked_vehicle_since is None else self._blocked_vehicle_since
                self._overtake_state = "waiting_left"
                if self._can_start_overtake(current_waypoint, timestamp, left_oncoming, rear_approaching):
                    self._start_lane_change(current_waypoint, "left", timestamp)
            else:
                self._blocked_vehicle_since = None
                self._overtake_state = "follow_lane"

        elif self._overtake_state == "changing_left":
            self._rejoin_lane_clear_since = None
            if current_waypoint is not None and self._overtake_origin_lane_id is not None:
                if current_waypoint.lane_id != self._overtake_origin_lane_id:
                    self._overtake_state = "passing"
                    self._left_lane_entered_at = timestamp

        elif self._overtake_state == "passing":
            self._update_rejoin_clear_state(current_waypoint, front_vehicle_state, timestamp)
            if self._should_return_right(current_waypoint, front_vehicle_state, timestamp):
                self._start_lane_change(current_waypoint, "right", timestamp)

        elif self._overtake_state == "changing_right":
            self._rejoin_lane_clear_since = None
            if current_waypoint is not None and self._overtake_origin_lane_id is not None:
                if self._is_rejoin_stabilized(current_waypoint):
                    self._right_lane_entered_at = (
                        timestamp if self._right_lane_entered_at is None else self._right_lane_entered_at
                    )
                    if timestamp - self._right_lane_entered_at >= self._min_right_lane_settle_time:
                        self._finish_overtake(current_waypoint, timestamp)
                else:
                    self._right_lane_entered_at = None

        return current_waypoint

    def _can_start_overtake(self, current_waypoint, timestamp, left_oncoming, rear_approaching):
        if self._agent is None or current_waypoint is None:
            self._waiting_left_reason = "no_waypoint"
            return False
        if current_waypoint.is_junction:
            self._waiting_left_reason = "junction"
            return False
        if self._blocked_vehicle_since is None or timestamp - self._blocked_vehicle_since < self._blocked_hold_time:
            self._waiting_left_reason = "hold"
            return False
        if left_oncoming is not None:
            self._waiting_left_reason = "left_oncoming"
            return False
        if rear_approaching is not None and rear_approaching.get("mode") not in ("releasing",):
            self._waiting_left_reason = "rear_approach"
            return False

        left_waypoint = current_waypoint.get_left_lane()
        if left_waypoint is None or left_waypoint.lane_type != carla.LaneType.Driving:
            self._waiting_left_reason = "no_left_lane"
            return False

        self._waiting_left_reason = "clear"
        return True

    def _start_lane_change(self, current_waypoint, direction, timestamp):
        if self._agent is None or current_waypoint is None:
            return False

        if direction == "left":
            self._saved_route_plan = self._capture_remaining_route_plan()
            plan = self._build_lane_shift_plan_from_saved_route(
                current_waypoint,
                [
                    {
                        "distance_m": self._get_left_lane_change_same_lane_distance(),
                        "lane_mode": "base",
                        "road_option": RoadOption.LANEFOLLOW,
                    },
                    {
                        "distance_m": self._overtake_lane_change_distance,
                        "lane_mode": "left",
                        "road_option": RoadOption.CHANGELANELEFT,
                    },
                    {
                        "distance_m": self._overtake_other_lane_distance,
                        "lane_mode": "left",
                        "road_option": RoadOption.LANEFOLLOW,
                    },
                ],
            )
            if not plan:
                return False

            self._overtake_origin_lane_id = current_waypoint.lane_id
            self._overtake_origin_road_id = current_waypoint.road_id
            self._overtake_started_at = timestamp
            self._left_lane_entered_at = None
            self._right_lane_entered_at = None
            self._using_helper_rejoin_plan = False
            self._rejoin_lane_clear_since = None
            self._overtake_state = "changing_left"
            self._agent.set_global_plan(plan)
            self._agent.set_target_speed(self._overtake_speed)
            self._active_target_speed = self._overtake_speed
            return True

        if direction == "right":
            rejoin_waypoint, rejoin_direction = self._find_origin_adjacent_lane(current_waypoint)
            if rejoin_waypoint is None:
                return False
            plan = self._build_helper_rejoin_plan(current_waypoint, rejoin_waypoint, rejoin_direction)
            using_helper_rejoin_plan = bool(plan)

            if not plan:
                plan = self._build_rejoin_plan_from_saved_route(current_waypoint)
                using_helper_rejoin_plan = bool(plan)

            if not plan:
                plan = self._build_rejoin_plan_from_global_route(current_waypoint)
                using_helper_rejoin_plan = bool(plan)

            if not plan:
                plan = self._build_lane_shift_plan_from_saved_route(
                    current_waypoint,
                    [
                        {
                            "distance_m": self._return_same_lane_distance,
                            "lane_mode": "left",
                            "road_option": RoadOption.LANEFOLLOW,
                        },
                        {
                            "distance_m": self._return_lane_change_distance,
                            "lane_mode": "base",
                            "road_option": (
                                RoadOption.CHANGELANERIGHT
                                if rejoin_direction == "right"
                                else RoadOption.CHANGELANELEFT
                            ),
                        },
                        {
                            "distance_m": self._return_other_lane_distance,
                            "lane_mode": "base",
                            "road_option": RoadOption.LANEFOLLOW,
                        },
                    ],
                )
                using_helper_rejoin_plan = False

            if not plan:
                return False

            self._overtake_state = "changing_right"
            self._right_lane_entered_at = None
            self._using_helper_rejoin_plan = using_helper_rejoin_plan
            self._rejoin_lane_clear_since = None
            self._agent.set_global_plan(plan)
            self._agent.set_target_speed(self._overtake_speed)
            self._active_target_speed = self._overtake_speed
            return True

        return False

    def _should_return_right(self, current_waypoint, front_vehicle_state, timestamp):
        del front_vehicle_state
        if current_waypoint is None:
            return False
        if self._left_lane_entered_at is None or timestamp - self._left_lane_entered_at < self._min_left_lane_time:
            return False
        if self._rejoin_lane_clear_since is None:
            return False
        if timestamp - self._rejoin_lane_clear_since < self._rejoin_clear_hold_time:
            return False

        rejoin_waypoint, _rejoin_direction = self._find_origin_adjacent_lane(current_waypoint)
        if rejoin_waypoint is None:
            return False

        return True

    def _get_left_lane_change_same_lane_distance(self):
        if self._controller_mode == "mpc":
            return self._mpc_overtake_same_lane_distance
        return self._overtake_same_lane_distance

    def _update_rejoin_clear_state(self, current_waypoint, front_vehicle_state, timestamp):
        if self._is_rejoin_lane_clear(current_waypoint, front_vehicle_state):
            if self._rejoin_lane_clear_since is None:
                self._rejoin_lane_clear_since = timestamp
        else:
            self._rejoin_lane_clear_since = None

    def _is_rejoin_lane_clear(self, current_waypoint, front_vehicle_state):
        if current_waypoint is None or front_vehicle_state is None:
            return False

        rejoin_waypoint, _rejoin_direction = self._find_origin_adjacent_lane(current_waypoint)
        if rejoin_waypoint is None:
            return False

        raw_front_radar_points = front_vehicle_state.get("raw_front_radar_points")
        rejoin_lane_points = self._extract_rejoin_lane_radar_points(
            current_waypoint,
            rejoin_waypoint,
            raw_front_radar_points,
        )
        return len(rejoin_lane_points) < self._rejoin_radar_min_points

    def _finish_overtake(self, current_waypoint, timestamp):
        using_helper_rejoin_plan = self._using_helper_rejoin_plan
        self._overtake_state = "follow_lane"
        self._blocked_vehicle_since = None
        self._overtake_started_at = None
        self._left_lane_entered_at = None
        self._right_lane_entered_at = None
        self._overtake_cooldown_until = timestamp + self._post_overtake_cooldown_time
        self._overtake_origin_lane_id = None
        self._overtake_origin_road_id = None
        self._using_helper_rejoin_plan = False
        self._waiting_left_reason = "clear"
        self._rejoin_lane_clear_since = None

        if self._agent is not None:
            self._agent.set_target_speed(self._target_speed)
            self._active_target_speed = self._target_speed

        if using_helper_rejoin_plan:
            self._saved_route_plan = []
            self._route_assigned = True
            return

        self._restore_main_route(current_waypoint)

    def _capture_remaining_route_plan(self):
        if self._agent is None:
            return []

        local_planner = self._agent.get_local_planner()
        if local_planner is None:
            return []

        return list(local_planner.get_plan())

    def _is_waypoint_ahead(self, current_waypoint, target_waypoint):
        if current_waypoint is None or target_waypoint is None:
            return False

        hero_actor = self._hero_actor or self._get_hero_actor()
        if hero_actor is not None:
            ego_transform = hero_actor.get_transform()
            current_location = ego_transform.location
            forward = ego_transform.get_forward_vector()
        else:
            current_location = current_waypoint.transform.location
            forward = current_waypoint.transform.get_forward_vector()

        target_location = target_waypoint.transform.location
        direction = carla.Vector3D(
            x=target_location.x - current_location.x,
            y=target_location.y - current_location.y,
            z=target_location.z - current_location.z,
        )
        dot_product = (forward.x * direction.x) + (forward.y * direction.y) + (forward.z * direction.z)
        return dot_product > 0.0

    @staticmethod
    def _get_adjacent_driving_lane(waypoint, direction):
        if waypoint is None:
            return None

        adjacent_waypoint = waypoint.get_left_lane() if direction == "left" else waypoint.get_right_lane()
        if adjacent_waypoint is None or adjacent_waypoint.lane_type != carla.LaneType.Driving:
            return None

        return adjacent_waypoint

    def _get_ego_relative_lane_direction(self, current_waypoint, target_waypoint):
        if current_waypoint is None or target_waypoint is None:
            return None

        hero_actor = self._hero_actor or self._get_hero_actor()
        if hero_actor is not None:
            ego_transform = hero_actor.get_transform()
            current_location = ego_transform.location
            right_vector = ego_transform.get_right_vector()
        else:
            current_location = current_waypoint.transform.location
            right_vector = current_waypoint.transform.get_right_vector()

        target_location = target_waypoint.transform.location
        lateral_vector = carla.Vector3D(
            x=target_location.x - current_location.x,
            y=target_location.y - current_location.y,
            z=target_location.z - current_location.z,
        )
        lateral_offset = (
            right_vector.x * lateral_vector.x
            + right_vector.y * lateral_vector.y
            + right_vector.z * lateral_vector.z
        )

        if lateral_offset > 0.1:
            return "right"
        if lateral_offset < -0.1:
            return "left"
        return None

    def _find_origin_adjacent_lane(self, current_waypoint):
        if current_waypoint is None:
            return None, None

        for direction in ("right", "left"):
            adjacent_waypoint = self._get_adjacent_driving_lane(current_waypoint, direction)
            if adjacent_waypoint is None:
                continue

            if self._overtake_origin_road_id is not None and adjacent_waypoint.road_id != self._overtake_origin_road_id:
                continue
            if self._overtake_origin_lane_id is not None and adjacent_waypoint.lane_id != self._overtake_origin_lane_id:
                continue

            ego_relative_direction = self._get_ego_relative_lane_direction(current_waypoint, adjacent_waypoint)
            return adjacent_waypoint, ego_relative_direction or direction

        return None, None

    def _map_route_waypoint_to_lane(self, route_waypoint, lane_mode):
        if route_waypoint is None:
            return None

        if lane_mode == "base":
            return route_waypoint
        if lane_mode == "left":
            return self._get_adjacent_driving_lane(route_waypoint, "left")
        if lane_mode == "right":
            return self._get_adjacent_driving_lane(route_waypoint, "right")

        return None

    def _extract_rejoin_lane_radar_points(self, current_waypoint, rejoin_waypoint, raw_front_radar_points):
        if current_waypoint is None or rejoin_waypoint is None:
            return np.empty((0, 4), dtype=np.float32)
        if raw_front_radar_points is None or len(raw_front_radar_points) == 0:
            return np.empty((0, 4), dtype=np.float32)

        filtered_points = self._filter_radar_points(
            raw_front_radar_points,
            min_depth=self._front_radar_min_depth,
            max_depth=self._rejoin_radar_max_depth,
            max_abs_azimuth=self._rejoin_radar_max_abs_azimuth,
        )
        if filtered_points is None or len(filtered_points) == 0:
            return np.empty((0, 4), dtype=np.float32)

        hero_actor = self._hero_actor or self._get_hero_actor()
        if hero_actor is not None:
            ego_transform = hero_actor.get_transform()
            current_location = ego_transform.location
            right_vector = ego_transform.get_right_vector()
        else:
            current_location = current_waypoint.transform.location
            right_vector = current_waypoint.transform.get_right_vector()

        rejoin_location = rejoin_waypoint.transform.location
        lane_center_lateral = (
            right_vector.x * (rejoin_location.x - current_location.x)
            + right_vector.y * (rejoin_location.y - current_location.y)
            + right_vector.z * (rejoin_location.z - current_location.z)
        )
        lane_width = float(
            rejoin_waypoint.lane_width
            if getattr(rejoin_waypoint, "lane_width", None) is not None
            else current_waypoint.lane_width
        )
        lateral_margin = max(self._rejoin_lane_lateral_margin, 0.45 * lane_width)

        lane_points = []
        for point in filtered_points:
            depth = float(point[0])
            azimuth = float(point[2])
            forward_m = depth * np.cos(azimuth)
            lateral_m = depth * np.sin(azimuth)

            if forward_m < self._rejoin_radar_min_forward_distance:
                continue
            if abs(lateral_m - lane_center_lateral) > lateral_margin:
                continue

            lane_points.append(point)

        if not lane_points:
            return np.empty((0, 4), dtype=np.float32)

        return np.asarray(lane_points, dtype=filtered_points.dtype)

    def _build_lane_shift_plan_from_saved_route(self, current_waypoint, segments):
        if current_waypoint is None or not self._saved_route_plan:
            return []

        total_distance_m = sum(max(float(segment["distance_m"]), 0.0) for segment in segments)
        if total_distance_m <= 0.0:
            return []

        segment_index = 0
        segment_end_distance_m = max(float(segments[segment_index]["distance_m"]), 0.0)
        cumulative_distance_m = 0.0
        previous_route_waypoint = current_waypoint
        plan = []

        plan.append((current_waypoint, segments[0]["road_option"]))

        for route_waypoint, _route_option in self._saved_route_plan:
            if route_waypoint is None:
                continue

            cumulative_distance_m += route_waypoint.transform.location.distance(
                previous_route_waypoint.transform.location
            )
            while segment_index < len(segments) - 1 and cumulative_distance_m > segment_end_distance_m:
                segment_index += 1
                segment_end_distance_m += max(float(segments[segment_index]["distance_m"]), 0.0)

            segment = segments[segment_index]
            target_waypoint = self._map_route_waypoint_to_lane(route_waypoint, segment["lane_mode"])
            if target_waypoint is None:
                return []

            if plan[-1][0].id != target_waypoint.id or plan[-1][1] != segment["road_option"]:
                plan.append((target_waypoint, segment["road_option"]))
            previous_route_waypoint = route_waypoint

            if cumulative_distance_m >= total_distance_m and len(plan) >= 3:
                break

        return plan

    def _build_rejoin_plan_from_saved_route(self, current_waypoint):
        return self._build_rejoin_plan_from_saved_route_starting_at(current_waypoint, current_waypoint)

    def _build_rejoin_plan_from_saved_route_starting_at(self, current_waypoint, trace_start_waypoint):
        if self._agent is None or current_waypoint is None or not self._saved_route_plan:
            return []
        if trace_start_waypoint is None:
            return []

        hero_actor = self._hero_actor or self._get_hero_actor()
        current_location = hero_actor.get_location() if hero_actor is not None else current_waypoint.transform.location
        preferred_index = None
        fallback_index = None

        for index, (route_waypoint, _road_option) in enumerate(self._saved_route_plan):
            if route_waypoint is None:
                continue

            distance = current_location.distance(route_waypoint.transform.location)
            if distance < self._route_rejoin_min_distance:
                continue
            if not self._is_waypoint_ahead(current_waypoint, route_waypoint):
                continue

            if fallback_index is None:
                fallback_index = index

            same_lane = (
                route_waypoint.road_id == current_waypoint.road_id
                and route_waypoint.lane_id == current_waypoint.lane_id
            )
            if same_lane:
                preferred_index = index
                break

        target_index = preferred_index if preferred_index is not None else fallback_index
        if target_index is None:
            return []

        target_waypoint = self._saved_route_plan[target_index][0]
        join_plan = self._agent.trace_route(trace_start_waypoint, target_waypoint)
        if not join_plan:
            return []

        restored_plan = list(join_plan)
        if target_index + 1 < len(self._saved_route_plan):
            restored_plan.extend(self._saved_route_plan[target_index + 1:])

        return restored_plan

    def _build_rejoin_plan_from_global_route(self, current_waypoint):
        return self._build_rejoin_plan_from_global_route_starting_at(current_waypoint, current_waypoint)

    def _build_rejoin_plan_from_global_route_starting_at(self, current_waypoint, trace_start_waypoint):
        if self._agent is None or current_waypoint is None or not self._global_plan_world_coord:
            return []
        if trace_start_waypoint is None:
            return []

        carla_map = CarlaDataProvider.get_map()
        if carla_map is None:
            return []

        anchor_waypoint = None
        hero_actor = self._hero_actor or self._get_hero_actor()
        current_location = hero_actor.get_location() if hero_actor is not None else current_waypoint.transform.location

        for transform, _road_option in self._global_plan_world_coord:
            candidate_waypoint = carla_map.get_waypoint(transform.location)
            if candidate_waypoint is None:
                continue

            if current_location.distance(candidate_waypoint.transform.location) < self._route_rejoin_min_distance:
                continue
            if not self._is_waypoint_ahead(current_waypoint, candidate_waypoint):
                continue

            anchor_waypoint = candidate_waypoint
            break

        if anchor_waypoint is None:
            return []

        plan = self._agent.trace_route(trace_start_waypoint, anchor_waypoint)
        if not plan:
            return []

        previous_wp = anchor_waypoint
        appended_anchor = False
        for transform, _road_option in self._global_plan_world_coord:
            waypoint = carla_map.get_waypoint(transform.location)
            if waypoint is None:
                continue

            if not appended_anchor:
                if waypoint.id != anchor_waypoint.id:
                    continue
                appended_anchor = True
                continue

            traced_route = self._agent.trace_route(previous_wp, waypoint)
            if traced_route:
                plan.extend(traced_route)
                previous_wp = waypoint

        return plan

    def _build_helper_rejoin_plan(self, current_waypoint, rejoin_waypoint, rejoin_direction):
        if self._agent is None or current_waypoint is None or rejoin_waypoint is None:
            return []

        road_option = (
            RoadOption.CHANGELANERIGHT
            if rejoin_direction == "right"
            else RoadOption.CHANGELANELEFT
        )

        continued_plan = self._build_rejoin_plan_from_saved_route_starting_at(
            current_waypoint,
            rejoin_waypoint,
        )
        if not continued_plan:
            continued_plan = self._build_rejoin_plan_from_global_route_starting_at(
                current_waypoint,
                rejoin_waypoint,
            )
        if not continued_plan:
            return []

        helper_plan = [(current_waypoint, road_option)]

        if continued_plan[0][0].id == rejoin_waypoint.id:
            helper_plan.append((continued_plan[0][0], road_option))
            helper_plan.extend(continued_plan[1:])
        else:
            helper_plan.append((rejoin_waypoint, road_option))
            helper_plan.extend(continued_plan)

        return helper_plan

    def _restore_main_route(self, current_waypoint):
        if self._agent is None:
            self._route_assigned = False
            self._saved_route_plan = []
            return False

        restored_plan = self._build_rejoin_plan_from_saved_route(current_waypoint)
        if not restored_plan:
            restored_plan = self._build_rejoin_plan_from_global_route(current_waypoint)

        self._saved_route_plan = []
        if not restored_plan:
            self._route_assigned = False
            return False

        self._agent.set_global_plan(restored_plan)
        self._route_assigned = True
        return True

    @staticmethod
    def _filter_radar_points(points, min_depth=0.0, max_depth=None, max_abs_azimuth=None):
        if points is None or len(points) == 0:
            return points

        mask = np.isfinite(points).all(axis=1)
        mask &= points[:, 0] >= min_depth

        if max_depth is not None:
            mask &= points[:, 0] <= max_depth

        if max_abs_azimuth is not None:
            mask &= np.abs(points[:, 2]) <= max_abs_azimuth

        return points[mask]

    @staticmethod
    def _cluster_nearest_target(filtered_points, depth_window, azimuth_window):
        if filtered_points is None or len(filtered_points) == 0:
            return filtered_points

        anchor = filtered_points[np.argmin(filtered_points[:, 0])]
        cluster_mask = np.abs(filtered_points[:, 0] - anchor[0]) <= depth_window
        cluster_mask &= np.abs(filtered_points[:, 2] - anchor[2]) <= azimuth_window
        cluster_points = filtered_points[cluster_mask]

        if len(cluster_points) == 0:
            return filtered_points

        return cluster_points[np.argsort(cluster_points[:, 0])]

    def _extract_front_cluster(self, points):
        filtered_points = self._filter_radar_points(
            points,
            min_depth=self._front_radar_min_depth,
            max_depth=self._front_radar_max_depth,
            max_abs_azimuth=self._front_radar_max_abs_azimuth,
        )
        return self._cluster_nearest_target(
            filtered_points,
            self._front_cluster_depth_window,
            self._front_cluster_azimuth_window,
        )

    def _extract_rear_cluster(self, points):
        filtered_points = self._filter_radar_points(
            points,
            min_depth=self._rear_radar_min_depth,
            max_depth=self._rear_radar_max_depth,
            max_abs_azimuth=self._rear_radar_max_abs_azimuth,
        )
        return self._cluster_nearest_target(
            filtered_points,
            self._rear_cluster_depth_window,
            self._rear_cluster_azimuth_window,
        )

    def _select_lead_vehicle_detection(self, detections):
        if not detections:
            return None

        candidates = []
        for detection in detections:
            label = detection["label"].lower()
            if label not in self._vehicle_detection_labels:
                continue
            if detection["confidence"] < self._front_vehicle_min_confidence:
                continue

            x1, y1, x2, y2 = detection["bbox"]
            center_x_ratio = 0.5 * (x1 + x2) / float(self.camera_width)
            bottom_ratio = y2 / float(self.camera_height)
            if center_x_ratio < self._front_vehicle_center_x_min or center_x_ratio > self._front_vehicle_center_x_max:
                continue
            if bottom_ratio < self._front_vehicle_min_bottom_ratio:
                continue

            area = max((x2 - x1) * (y2 - y1), 1.0)
            candidates.append((detection, area))

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: (
                item[0]["distance_m"] is None,
                item[0]["distance_m"] if item[0]["distance_m"] is not None else 1e9,
                -item[1],
            )
        )
        return candidates[0][0]

    def _estimate_front_vehicle_state(self, front_radar_points, raw_front_radar_points=None):
        front_radar_depth = None
        front_radar_velocity = None
        front_radar_count = 0
        front_radar_mean_abs_azimuth = None
        front_radar_anchor_azimuth = None
        front_relative_longitudinal_m = None
        front_relative_lateral_m = None
        if front_radar_points is not None and len(front_radar_points) > 0:
            front_radar_depth = float(np.min(front_radar_points[:, 0]))
            front_radar_velocity = float(np.median(front_radar_points[:, 3]))
            front_radar_count = len(front_radar_points)
            front_radar_mean_abs_azimuth = float(np.mean(np.abs(front_radar_points[:, 2])))
            anchor_point = front_radar_points[np.argmin(front_radar_points[:, 0])]
            front_radar_anchor_azimuth = float(anchor_point[2])
            front_relative_longitudinal_m = float(anchor_point[0] * np.cos(anchor_point[2]))
            front_relative_lateral_m = float(anchor_point[0] * np.sin(anchor_point[2]))

        distance_m = front_radar_depth
        blocked = (
            front_radar_depth is not None
            and front_radar_depth <= self._front_radar_trigger_distance
            and front_radar_count >= self._front_radar_min_points
            and (
                front_radar_mean_abs_azimuth is None
                or front_radar_mean_abs_azimuth <= self._front_radar_max_mean_abs_azimuth
            )
        )

        return {
            "lead_vehicle": None,
            "distance_m": distance_m,
            "blocked": blocked,
            "emergency": distance_m is not None and distance_m <= self._front_vehicle_emergency_distance,
            "front_radar_depth": front_radar_depth,
            "front_radar_velocity": front_radar_velocity,
            "front_radar_count": front_radar_count,
            "front_radar_mean_abs_azimuth": front_radar_mean_abs_azimuth,
            "front_radar_anchor_azimuth": front_radar_anchor_azimuth,
            "front_relative_longitudinal_m": front_relative_longitudinal_m,
            "front_relative_lateral_m": front_relative_lateral_m,
            "raw_front_radar_points": raw_front_radar_points,
        }

    def _detect_left_oncoming_vehicle(self, detections):
        if not detections:
            return None

        best_detection = None
        best_area = 0.0
        frame_area = float(self.camera_width * self.camera_height)
        debug_items = []

        for detection in detections:
            label = detection["label"].lower()
            x1, y1, x2, y2 = detection["bbox"]
            area_ratio = max((x2 - x1) * (y2 - y1), 0.0) / frame_area
            center_x_ratio = 0.5 * (x1 + x2) / float(self.camera_width)

            reason = "accepted"
            if label not in self._vehicle_detection_labels:
                reason = "label"
                debug_items.append(self._format_leftcam_debug_item(detection, area_ratio, center_x_ratio, reason))
                continue
            if detection["confidence"] < self._left_oncoming_min_confidence:
                reason = "low_conf"
                debug_items.append(self._format_leftcam_debug_item(detection, area_ratio, center_x_ratio, reason))
                continue

            if area_ratio < self._left_oncoming_min_area_ratio:
                reason = "small_area"
                debug_items.append(self._format_leftcam_debug_item(detection, area_ratio, center_x_ratio, reason))
                continue
            if center_x_ratio < self._left_oncoming_min_center_x_ratio:
                reason = "too_left"
                debug_items.append(self._format_leftcam_debug_item(detection, area_ratio, center_x_ratio, reason))
                continue
            if center_x_ratio > self._left_oncoming_max_center_x_ratio:
                reason = "too_right"
                debug_items.append(self._format_leftcam_debug_item(detection, area_ratio, center_x_ratio, reason))
                continue

            debug_items.append(self._format_leftcam_debug_item(detection, area_ratio, center_x_ratio, reason))
            if area_ratio > best_area:
                best_detection = detection
                best_area = area_ratio

        self._log_leftcam_filter_debug(debug_items, best_detection)
        return best_detection

    def _format_leftcam_debug_item(self, detection, area_ratio, center_x_ratio, reason):
        return "{} {:.2f} area={:.4f} cx={:.3f} {}".format(
            detection["label"],
            detection["confidence"],
            area_ratio,
            center_x_ratio,
            reason,
        )

    def _log_leftcam_filter_debug(self, debug_items, best_detection):
        if not debug_items:
            return
        if self._current_timestamp - self._last_leftcam_filter_debug_timestamp < 0.5:
            return

        self._last_leftcam_filter_debug_timestamp = self._current_timestamp
        selected = "none" if best_detection is None else "{} {:.2f}".format(
            best_detection["label"],
            best_detection["confidence"],
        )
        print(
            "[LeftCamDebug] selected={} | {}".format(
                selected,
                " | ".join(debug_items[:4]),
            )
        )

    def _detect_rear_approaching_vehicle(self, rear_radar_points):
        if rear_radar_points is None or len(rear_radar_points) == 0:
            return None

        valid_points = rear_radar_points[np.isfinite(rear_radar_points).all(axis=1)]
        if len(valid_points) == 0:
            return None

        approaching_candidates = valid_points[
            (valid_points[:, 0] <= self._rear_approach_distance)
            & (valid_points[:, 3] <= self._rear_approach_velocity)
        ]
        if len(approaching_candidates) > 0:
            nearest = approaching_candidates[np.argmin(approaching_candidates[:, 0])]
            closing_speed_mps = max(-float(nearest[3]), 0.0)
            ttc_s = self._compute_ttc_s(float(nearest[0]), closing_speed_mps)
            if float(nearest[0]) <= self._rear_approach_close_distance or (
                ttc_s is not None and ttc_s <= self._rear_approach_ttc_threshold_s
            ):
                return {
                    "distance_m": float(nearest[0]),
                    "velocity_mps": float(nearest[3]),
                    "closing_speed_mps": closing_speed_mps,
                    "ttc_s": ttc_s,
                    "mode": "approaching",
                }

        occupied_candidates = valid_points[
            (valid_points[:, 0] <= self._rear_lane_occupied_distance)
            & (np.abs(valid_points[:, 3]) <= self._rear_lane_occupied_abs_velocity)
        ]
        if len(occupied_candidates) < self._rear_lane_occupied_min_points:
            return None

        nearest = occupied_candidates[np.argmin(occupied_candidates[:, 0])]
        distance_m = float(nearest[0])
        velocity_mps = float(nearest[3])
        closing_speed_mps = max(-velocity_mps, 0.0)
        ttc_s = self._compute_ttc_s(distance_m, closing_speed_mps)

        if velocity_mps >= self._rear_lane_release_velocity_mps and distance_m >= self._rear_lane_release_distance:
            return {
                "distance_m": distance_m,
                "velocity_mps": velocity_mps,
                "closing_speed_mps": closing_speed_mps,
                "ttc_s": ttc_s,
                "mode": "releasing",
            }

        if (
            distance_m > self._rear_lane_hold_distance
            and closing_speed_mps <= self._rear_lane_release_closing_speed_mps
            and (ttc_s is None or ttc_s >= self._rear_lane_release_ttc_s)
        ):
            return {
                "distance_m": distance_m,
                "velocity_mps": velocity_mps,
                "closing_speed_mps": closing_speed_mps,
                "ttc_s": ttc_s,
                "mode": "releasing",
            }

        if (
            closing_speed_mps >= abs(self._rear_approach_velocity)
            and (distance_m <= self._rear_approach_close_distance or (ttc_s is not None and ttc_s <= self._rear_approach_ttc_threshold_s))
        ):
            return {
                "distance_m": distance_m,
                "velocity_mps": velocity_mps,
                "closing_speed_mps": closing_speed_mps,
                "ttc_s": ttc_s,
                "mode": "approaching",
            }

        return {
            "distance_m": distance_m,
            "velocity_mps": velocity_mps,
            "closing_speed_mps": closing_speed_mps,
            "ttc_s": ttc_s,
            "mode": "occupied",
        }

    def _run_yolo(self, center_image):
        if center_image is None:
            return []

        frame_bgr = center_image[:, :, :3]
        try:
            results = self.detector(
                frame_bgr,
                verbose=False,
                conf=self._yolo_conf_threshold,
                imgsz=self._yolo_imgsz,
            )
        except Exception as exc:
            print("WARNING: YOLO inference failed: {}".format(exc))
            return []

        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        detections = []
        names = getattr(result, "names", {})
        for box in boxes:
            bbox = box.xyxy[0].tolist()
            class_id = int(box.cls[0].item()) if box.cls is not None else -1
            confidence = float(box.conf[0].item()) if box.conf is not None else 0.0
            label = str(class_id)
            if isinstance(names, dict):
                label = names.get(class_id, label)
            elif isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
                label = names[class_id]

            detections.append({
                "label": str(label),
                "confidence": confidence,
                "bbox": bbox,
                "distance_m": None,
            })

        detections.sort(key=lambda detection: detection["confidence"], reverse=True)
        return detections

    def _summarize_yolo(self, detections):
        if not detections:
            return "0 objs"

        summary_items = []
        for detection in detections[:self._yolo_max_summary_detections]:
            item = "{} {:.2f}".format(detection["label"], detection["confidence"])
            if detection["distance_m"] is not None:
                item = "{} {:.1f}m".format(item, detection["distance_m"])
            summary_items.append(item)

        return ", ".join(summary_items)

    @staticmethod
    def _summarize_front_vehicle_state(front_vehicle_state):
        if front_vehicle_state is None or not front_vehicle_state["blocked"]:
            return "clear"

        distance_text = "?"
        if front_vehicle_state["distance_m"] is not None:
            distance_text = "{:.1f}m".format(front_vehicle_state["distance_m"])

        count = front_vehicle_state.get("front_radar_count")
        if count is None:
            return "radar {}".format(distance_text)
        return "radar {} ({} pts)".format(distance_text, count)

    @staticmethod
    def _summarize_oncoming(left_oncoming):
        if left_oncoming is None:
            return "clear"
        return "{} {:.2f}".format(left_oncoming["label"], left_oncoming["confidence"])

    def _summarize_maneuver_state(self):
        if self._overtake_state == "waiting_left":
            return "{} ({})".format(self._overtake_state, self._waiting_left_reason)
        return self._overtake_state

    @staticmethod
    def _summarize_rear_approach(rear_approaching):
        if rear_approaching is None:
            return "clear"
        mode = rear_approaching.get("mode", "approaching")
        ttc_s = rear_approaching.get("ttc_s")
        if ttc_s is None:
            return "{} {:.1f}m {:.1f}m/s".format(
                mode,
                rear_approaching["distance_m"],
                rear_approaching["velocity_mps"],
            )
        return "{} {:.1f}m {:.1f}m/s ttc={:.1f}s".format(
            mode,
            rear_approaching["distance_m"],
            rear_approaching["velocity_mps"],
            ttc_s,
        )


    @staticmethod
    def _summarize_radar(points):
        if points is None:
            return "missing"
        if len(points) == 0:
            return "0 pts"

        return "{count} pts, min={min_depth:.1f} m, mean={mean_depth:.1f} m, vel=[{min_vel:.1f}, {max_vel:.1f}] m/s".format(
            count=len(points),
            min_depth=float(np.min(points[:, 0])),
            mean_depth=float(np.mean(points[:, 0])),
            min_vel=float(np.min(points[:, 3])),
            max_vel=float(np.max(points[:, 3])),
        )

    def _initialize_metrics_logger(self, path_to_conf_file):
        if not self._metrics_enabled:
            return

        route_metadata = self.get_route_metadata()
        run_label = self._metrics_run_label or route_metadata.get("route_id") or self._controller_name
        try:
            self._metrics_logger = MetricsLogger(
                metrics_dir=self._metrics_dir,
                controller_name=self._controller_name,
                run_label=run_label,
                metadata={
                    "agent_entry_point": get_entry_point(),
                    "config_path": path_to_conf_file,
                    "controller_mode": self._controller_mode,
                    "target_speed_kph": self._target_speed,
                    "camera_width": self.camera_width,
                    "camera_height": self.camera_height,
                    "mpc_horizon_steps": self._mpc_horizon_steps if self._controller_mode == "mpc" else None,
                    "mpc_prediction_dt": self._mpc_prediction_dt if self._controller_mode == "mpc" else None,
                    "mpc_longitudinal_horizon_steps": (
                        self._mpc_longitudinal_horizon_steps if self._controller_mode == "mpc" else None
                    ),
                    "mpc_longitudinal_prediction_dt": (
                        self._mpc_longitudinal_prediction_dt if self._controller_mode == "mpc" else None
                    ),
                    "route_id": route_metadata.get("route_id"),
                    "route_index": route_metadata.get("route_index"),
                    "repetition_index": route_metadata.get("repetition_index"),
                    "town": route_metadata.get("town"),
                },
                flush_interval=self._metrics_flush_interval,
            )
            print("[Metrics] Logging raw telemetry to {}".format(self._metrics_logger.frame_path))
        except (OSError, ValueError) as exc:
            self._metrics_logger = None
            print("WARNING: Couldn't initialize metrics logger: {}".format(exc))

    def _record_metrics(self, frame_id, speed_mps, current_waypoint, front_vehicle_state,
                        rear_radar_points, rear_approaching, left_oncoming, control):
        if self._metrics_logger is None:
            return

        hero_actor = self._hero_actor or self._get_hero_actor()
        if hero_actor is None:
            return

        try:
            transform = hero_actor.get_transform()
            velocity = hero_actor.get_velocity()
            acceleration = hero_actor.get_acceleration()
            angular_velocity = hero_actor.get_angular_velocity()
        except RuntimeError:
            return

        if current_waypoint is None:
            current_waypoint = self._get_current_waypoint()

        target_speed_mps = self._active_target_speed / 3.6 if self._active_target_speed is not None else None
        longitudinal_accel_mps2, lateral_accel_mps2 = self._project_to_vehicle_axes(acceleration, transform)
        lateral_error_m, heading_error_deg = self._compute_tracking_errors(current_waypoint, transform)
        target_lane_metrics = self._compute_target_lane_tracking_metrics(current_waypoint, transform)
        route_metadata = self.get_route_metadata()
        rear_distance_m = None
        if rear_radar_points is not None and len(rear_radar_points) > 0:
            rear_distance_m = float(np.min(rear_radar_points[:, 0]))
        front_distance_m = front_vehicle_state.get("distance_m") if front_vehicle_state else None
        front_safety = self._build_front_safety_profile(front_vehicle_state, speed_mps)
        front_distance_margin_m = (
            front_distance_m - self._front_vehicle_emergency_distance
            if front_distance_m is not None
            else None
        )
        front_relative_lateral_m = front_vehicle_state.get("front_relative_lateral_m") if front_vehicle_state else None
        front_relative_longitudinal_m = (
            front_vehicle_state.get("front_relative_longitudinal_m") if front_vehicle_state else None
        )

        metrics_row = {
            "route_id": route_metadata.get("route_id"),
            "route_index": route_metadata.get("route_index"),
            "repetition_index": route_metadata.get("repetition_index"),
            "town": route_metadata.get("town"),
            "frame": frame_id,
            "timestamp": self._current_timestamp,
            "controller": self._controller_name,
            "controller_mode": self._controller_mode,
            "controller_compute_time_ms": self._last_controller_diagnostics["compute_time_ms"],
            "controller_status": self._last_controller_diagnostics["status"],
            "controller_fallback": self._last_controller_diagnostics["fallback"],
            "longitudinal_controller_compute_time_ms": (
                self._last_longitudinal_controller_diagnostics["compute_time_ms"]
            ),
            "longitudinal_controller_status": self._last_longitudinal_controller_diagnostics["status"],
            "longitudinal_controller_fallback": self._last_longitudinal_controller_diagnostics["fallback"],
            "longitudinal_accel_cmd_mps2": self._last_mpc_longitudinal_accel_cmd,
            "maneuver_state": self._overtake_state,
            "waiting_left_reason": self._waiting_left_reason,
            "speed_mps": speed_mps,
            "speed_kph": speed_mps * 3.6 if speed_mps is not None else None,
            "target_speed_mps": target_speed_mps,
            "target_speed_kph": self._active_target_speed,
            "speed_error_mps": (
                target_speed_mps - speed_mps
                if target_speed_mps is not None and speed_mps is not None
                else None
            ),
            "steer": control.steer,
            "throttle": control.throttle,
            "brake": control.brake,
            "x": transform.location.x,
            "y": transform.location.y,
            "z": transform.location.z,
            "yaw_deg": transform.rotation.yaw,
            "pitch_deg": transform.rotation.pitch,
            "roll_deg": transform.rotation.roll,
            "velocity_x": velocity.x,
            "velocity_y": velocity.y,
            "velocity_z": velocity.z,
            "accel_x": acceleration.x,
            "accel_y": acceleration.y,
            "accel_z": acceleration.z,
            "longitudinal_accel_mps2": longitudinal_accel_mps2,
            "lateral_accel_mps2": lateral_accel_mps2,
            "yaw_rate_dps": angular_velocity.z,
            "road_id": current_waypoint.road_id if current_waypoint is not None else None,
            "lane_id": current_waypoint.lane_id if current_waypoint is not None else None,
            "target_lane_id": self._get_target_lane_id_for_metrics(current_waypoint),
            "target_lane_lateral_error_m": target_lane_metrics.get("lateral_error_m"),
            "target_lane_heading_error_deg": target_lane_metrics.get("heading_error_deg"),
            "target_lane_progress": target_lane_metrics.get("progress"),
            "overtake_origin_lane_id": self._overtake_origin_lane_id,
            "is_junction": current_waypoint.is_junction if current_waypoint is not None else None,
            "front_distance_m": front_distance_m,
            "front_distance_margin_m": front_distance_margin_m,
            "front_dynamic_stop_distance_m": front_safety["dynamic_stop_distance_m"],
            "front_dynamic_emergency_distance_m": front_safety["dynamic_emergency_distance_m"],
            "front_dynamic_margin_m": front_safety["dynamic_margin_m"],
            "front_ttc_s": front_safety["ttc_s"],
            "front_closing_speed_mps": front_vehicle_state.get("front_radar_velocity") if front_vehicle_state else None,
            "front_radar_point_count": front_vehicle_state.get("front_radar_count") if front_vehicle_state else None,
            "front_radar_mean_abs_azimuth_rad": (
                front_vehicle_state.get("front_radar_mean_abs_azimuth") if front_vehicle_state else None
            ),
            "front_radar_anchor_azimuth_rad": (
                front_vehicle_state.get("front_radar_anchor_azimuth") if front_vehicle_state else None
            ),
            "front_relative_longitudinal_m": front_relative_longitudinal_m,
            "front_relative_lateral_m": front_relative_lateral_m,
            "front_relative_lateral_abs_m": (
                abs(front_relative_lateral_m) if front_relative_lateral_m is not None else None
            ),
            "front_blocked": front_vehicle_state.get("blocked") if front_vehicle_state else False,
            "rear_distance_m": rear_distance_m,
            "rear_approach_distance_m": rear_approaching.get("distance_m") if rear_approaching else None,
            "rear_approach_speed_mps": rear_approaching.get("velocity_mps") if rear_approaching else None,
            "rear_closing_speed_mps": rear_approaching.get("closing_speed_mps") if rear_approaching else None,
            "rear_approach_ttc_s": rear_approaching.get("ttc_s") if rear_approaching else None,
            "rear_approach_mode": rear_approaching.get("mode", "none") if rear_approaching else "none",
            "rear_lane_releasing": rear_approaching is not None and rear_approaching.get("mode") == "releasing",
            "left_oncoming_detected": left_oncoming is not None,
            "left_oncoming_confidence": left_oncoming.get("confidence") if left_oncoming else None,
            "left_oncoming_label": left_oncoming.get("label") if left_oncoming else None,
            "lateral_error_m": lateral_error_m,
            "heading_error_deg": heading_error_deg,
            "blocked_vehicle": self._is_front_obstacle_active(front_vehicle_state),
            "rejoin_lane_clear": self._rejoin_lane_clear_since is not None,
            "safety_override": self._last_safety_intervention["override"],
            "safety_override_reason": self._last_safety_intervention["reason"],
            "requested_brake": self._last_safety_intervention["requested_brake"],
            "emergency_brake": self._last_safety_intervention["requested_brake"] >= self._sensor_emergency_brake,
            "steer_saturated": abs(float(control.steer)) >= 0.75,
            "front_collision_risk": front_safety["collision_risk"],
        }

        try:
            self._metrics_logger.log_frame(metrics_row)
        except OSError as exc:
            print("WARNING: Failed to write metrics row: {}".format(exc))
            self._metrics_logger = None

    def _get_target_lane_id_for_metrics(self, current_waypoint):
        if current_waypoint is None:
            return self._overtake_origin_lane_id

        if self._overtake_state == "changing_right":
            return self._overtake_origin_lane_id

        if self._overtake_state in ("waiting_left", "changing_left"):
            adjacent_waypoint = self._get_adjacent_driving_lane(current_waypoint, "left")
            if adjacent_waypoint is not None:
                return adjacent_waypoint.lane_id

        return current_waypoint.lane_id

    @staticmethod
    def _project_to_vehicle_axes(vector, transform):
        if vector is None or transform is None:
            return None, None

        forward_vector = transform.get_forward_vector()
        right_vector = transform.get_right_vector()
        vector_np = np.array([vector.x, vector.y, vector.z], dtype=np.float64)
        forward_np = np.array([forward_vector.x, forward_vector.y, forward_vector.z], dtype=np.float64)
        right_np = np.array([right_vector.x, right_vector.y, right_vector.z], dtype=np.float64)

        return float(np.dot(vector_np, forward_np)), float(np.dot(vector_np, right_np))

    def _compute_tracking_errors(self, current_waypoint, ego_transform):
        if current_waypoint is None or ego_transform is None:
            return None, None

        waypoint_location = current_waypoint.transform.location
        ego_location = ego_transform.location
        delta = np.array(
            [
                ego_location.x - waypoint_location.x,
                ego_location.y - waypoint_location.y,
                ego_location.z - waypoint_location.z,
            ],
            dtype=np.float64,
        )
        right_vector = current_waypoint.transform.get_right_vector()
        right_np = np.array([right_vector.x, right_vector.y, right_vector.z], dtype=np.float64)
        lateral_error_m = float(np.dot(delta, right_np))
        heading_error_deg = self._normalize_angle_deg(
            ego_transform.rotation.yaw - current_waypoint.transform.rotation.yaw
        )

        return lateral_error_m, heading_error_deg

    @staticmethod
    def _normalize_angle_deg(angle_deg):
        return (float(angle_deg) + 180.0) % 360.0 - 180.0

    def _set_safety_intervention(self, override, reason, requested_brake):
        self._last_safety_intervention = {
            "override": bool(override),
            "reason": reason or "",
            "requested_brake": float(requested_brake or 0.0),
        }

    def destroy(self):
        """
        Cleanup agent state.
        """
        if getattr(self, "_metrics_logger", None) is not None:
            self._metrics_logger.close(
                extra_summary={
                    "route_id": self.get_route_metadata().get("route_id"),
                    "route_index": self.get_route_metadata().get("route_index"),
                    "repetition_index": self.get_route_metadata().get("repetition_index"),
                    "town": self.get_route_metadata().get("town"),
                    "final_state": self._overtake_state,
                    "target_speed_kph": self._target_speed,
                    "active_target_speed_kph": self._active_target_speed,
                }
            )
            self._metrics_logger = None
        if hasattr(self, "_display") and self._display is not None:
            self._display.close()
            self._display = None
        self._agent = None
        self._hero_actor = None
        self._mpc_controller = None
        self._mpc_longitudinal_controller = None
        self._route_assigned = False

    def _load_config(self, path_to_conf_file):
        """
        Load an optional plain-text config file.
        Supported format:
            target_speed: 20
        """
        try:
            with open(path_to_conf_file, "r", encoding="utf-8") as conf_file:
                for raw_line in conf_file:
                    line = raw_line.strip()
                    if not line or line.startswith("#") or ":" not in line:
                        continue

                    key, value = [part.strip() for part in line.split(":", 1)]
                    if key == "target_speed":
                        self._target_speed = float(value)
                        self._overtake_speed = min(max(self._target_speed, 15.0), 25.0)
                    elif key == "controller_mode":
                        self._controller_mode = value or self._controller_mode
                        self._controller_mode_explicit = True
                    elif key == "controller_name":
                        self._controller_name = value or self._controller_name
                    elif key == "mpc_horizon_steps":
                        self._mpc_horizon_steps = max(int(value), 4)
                    elif key == "mpc_prediction_dt":
                        self._mpc_prediction_dt = max(float(value), 1e-3)
                    elif key == "mpc_wheel_base_m":
                        self._mpc_wheel_base_m = max(float(value), 0.5)
                    elif key == "mpc_max_steer_delta_per_cycle":
                        self._mpc_max_steer_delta_per_cycle = max(float(value), 0.01)
                    elif key == "mpc_max_steer_angle_deg":
                        self._mpc_max_steer_angle_deg = max(float(value), 1.0)
                    elif key == "mpc_follow_steer_step_limit":
                        self._mpc_follow_steer_step_limit = max(float(value), 0.01)
                    elif key == "mpc_lane_change_steer_step_limit":
                        self._mpc_lane_change_steer_step_limit = max(float(value), 0.01)
                    elif key == "mpc_same_lane_change_steer_limit":
                        self._mpc_same_lane_change_steer_limit = min(max(float(value), 0.1), 1.0)
                    elif key == "mpc_close_front_steer_limit":
                        self._mpc_close_front_steer_limit = min(max(float(value), 0.1), 1.0)
                    elif key == "mpc_lane_change_commitment_min_steer":
                        self._mpc_lane_change_commitment_min_steer = min(max(float(value), 0.0), 1.0)
                    elif key == "mpc_lane_change_commitment_release_lateral_m":
                        self._mpc_lane_change_commitment_release_lateral_m = max(float(value), 0.0)
                    elif key == "mpc_lane_change_commitment_release_distance_m":
                        self._mpc_lane_change_commitment_release_distance_m = max(float(value), 0.0)
                    elif key == "mpc_left_release_speed_mps":
                        self._mpc_left_release_speed_mps = max(float(value), 0.0)
                    elif key == "mpc_left_release_heading_deg":
                        self._mpc_left_release_heading_deg = max(float(value), 0.0)
                    elif key == "mpc_left_release_lateral_error_m":
                        self._mpc_left_release_lateral_error_m = max(float(value), 0.0)
                    elif key == "mpc_left_release_progress":
                        self._mpc_left_release_progress = min(max(float(value), 0.0), 1.0)
                    elif key == "mpc_left_release_max_steer":
                        self._mpc_left_release_max_steer = min(max(float(value), 0.0), 1.0)
                    elif key == "mpc_left_final_release_speed_mps":
                        self._mpc_left_final_release_speed_mps = max(float(value), 0.0)
                    elif key == "mpc_left_final_release_heading_deg":
                        self._mpc_left_final_release_heading_deg = max(float(value), 0.0)
                    elif key == "mpc_left_final_release_lateral_error_m":
                        self._mpc_left_final_release_lateral_error_m = max(float(value), 0.0)
                    elif key == "mpc_left_final_release_progress":
                        self._mpc_left_final_release_progress = min(max(float(value), 0.0), 1.0)
                    elif key == "mpc_left_final_release_max_steer":
                        self._mpc_left_final_release_max_steer = min(max(float(value), 0.0), 1.0)
                    elif key == "mpc_rejoin_steer_limit":
                        self._mpc_rejoin_steer_limit = min(max(float(value), 0.05), 1.0)
                    elif key == "mpc_rejoin_soft_heading_deg":
                        self._mpc_rejoin_soft_heading_deg = max(float(value), 0.0)
                    elif key == "mpc_rejoin_soft_lateral_error_m":
                        self._mpc_rejoin_soft_lateral_error_m = max(float(value), 0.0)
                    elif key == "mpc_rejoin_hard_heading_deg":
                        self._mpc_rejoin_hard_heading_deg = max(float(value), 0.0)
                    elif key == "mpc_rejoin_hard_lateral_error_m":
                        self._mpc_rejoin_hard_lateral_error_m = max(float(value), 0.0)
                    elif key == "mpc_rejoin_finish_max_heading_error_deg":
                        self._mpc_rejoin_finish_max_heading_error_deg = max(float(value), 0.0)
                    elif key == "mpc_rejoin_finish_max_lateral_error_m":
                        self._mpc_rejoin_finish_max_lateral_error_m = max(float(value), 0.0)
                    elif key == "mpc_rejoin_finish_max_abs_steer":
                        self._mpc_rejoin_finish_max_abs_steer = min(max(float(value), 0.0), 1.0)
                    elif key == "mpc_overtake_same_lane_distance":
                        self._mpc_overtake_same_lane_distance = max(float(value), 0.0)
                    elif key == "mpc_waiting_left_close_stop_distance":
                        self._mpc_waiting_left_close_stop_distance = max(float(value), 0.0)
                    elif key == "mpc_waiting_left_close_stop_brake":
                        self._mpc_waiting_left_close_stop_brake = min(max(float(value), 0.0), 1.0)
                    elif key == "mpc_waiting_left_oncoming_extra_stop_buffer_m":
                        self._mpc_waiting_left_oncoming_extra_stop_buffer_m = max(float(value), 0.0)
                    elif key == "mpc_lane_change_launch_grace_time":
                        self._mpc_lane_change_launch_grace_time = max(float(value), 0.0)
                    elif key == "mpc_lane_change_launch_min_front_distance":
                        self._mpc_lane_change_launch_min_front_distance = max(float(value), 0.0)
                    elif key == "mpc_lane_change_launch_min_steer":
                        self._mpc_lane_change_launch_min_steer = max(float(value), 0.0)
                    elif key == "mpc_lane_change_launch_min_throttle":
                        self._mpc_lane_change_launch_min_throttle = min(max(float(value), 0.0), 1.0)
                    elif key == "mpc_lane_change_launch_throttle":
                        self._mpc_lane_change_launch_throttle = min(max(float(value), 0.0), 1.0)
                    elif key == "mpc_lane_change_resume_speed_mps":
                        self._mpc_lane_change_resume_speed_mps = max(float(value), 0.0)
                    elif key == "mpc_longitudinal_horizon_steps":
                        self._mpc_longitudinal_horizon_steps = max(int(value), 4)
                    elif key == "mpc_longitudinal_prediction_dt":
                        self._mpc_longitudinal_prediction_dt = max(float(value), 1e-3)
                    elif key == "mpc_longitudinal_max_accel_mps2":
                        self._mpc_longitudinal_max_accel_mps2 = max(float(value), 0.1)
                    elif key == "mpc_longitudinal_max_decel_mps2":
                        self._mpc_longitudinal_max_decel_mps2 = max(float(value), 0.1)
                    elif key == "mpc_longitudinal_max_accel_delta_per_cycle":
                        self._mpc_longitudinal_max_accel_delta_per_cycle = max(float(value), 0.05)
                    elif key == "mpc_longitudinal_stop_buffer_m":
                        self._mpc_longitudinal_stop_buffer_m = max(float(value), 0.0)
                    elif key == "mpc_waiting_left_target_speed_mps":
                        self._mpc_waiting_left_target_speed_mps = max(float(value), 0.0)
                    elif key == "mpc_waiting_left_creep_speed_mps":
                        self._mpc_waiting_left_creep_speed_mps = max(float(value), 0.0)
                    elif key == "mpc_changing_left_target_speed_mps":
                        self._mpc_changing_left_target_speed_mps = max(float(value), 0.0)
                    elif key == "mpc_passing_target_speed_mps":
                        self._mpc_passing_target_speed_mps = max(float(value), 0.0)
                    elif key == "mpc_changing_right_target_speed_mps":
                        self._mpc_changing_right_target_speed_mps = max(float(value), 0.0)
                    elif key == "front_vehicle_min_hold_distance":
                        self._front_vehicle_min_hold_distance = max(float(value), 0.5)
                    elif key == "front_vehicle_reaction_time_s":
                        self._front_vehicle_reaction_time_s = max(float(value), 0.0)
                    elif key == "front_vehicle_comfort_decel_mps2":
                        self._front_vehicle_comfort_decel_mps2 = max(float(value), 0.5)
                    elif key == "front_vehicle_low_speed_mps":
                        self._front_vehicle_low_speed_mps = max(float(value), 0.0)
                    elif key == "front_vehicle_restart_speed_mps":
                        self._front_vehicle_restart_speed_mps = max(float(value), 0.0)
                    elif key == "front_vehicle_ttc_brake_s":
                        self._front_vehicle_ttc_brake_s = max(float(value), 0.1)
                    elif key == "front_vehicle_ttc_emergency_s":
                        self._front_vehicle_ttc_emergency_s = max(float(value), 0.1)
                    elif key == "rear_approach_ttc_threshold_s":
                        self._rear_approach_ttc_threshold_s = max(float(value), 0.1)
                    elif key == "rear_approach_close_distance":
                        self._rear_approach_close_distance = max(float(value), 0.0)
                    elif key == "rear_lane_hold_distance":
                        self._rear_lane_hold_distance = max(float(value), 0.0)
                    elif key == "rear_lane_release_distance":
                        self._rear_lane_release_distance = max(float(value), 0.0)
                    elif key == "rear_lane_release_velocity_mps":
                        self._rear_lane_release_velocity_mps = float(value)
                    elif key == "rear_lane_release_closing_speed_mps":
                        self._rear_lane_release_closing_speed_mps = max(float(value), 0.0)
                    elif key == "rear_lane_release_ttc_s":
                        self._rear_lane_release_ttc_s = max(float(value), 0.1)
                    elif key == "metrics_enabled":
                        self._metrics_enabled = value.lower() in ("1", "true", "yes", "on")
                    elif key == "metrics_dir":
                        self._metrics_dir = value or self._metrics_dir
                    elif key in ("metrics_run_label", "run_label"):
                        self._metrics_run_label = value
                    elif key == "metrics_flush_interval":
                        self._metrics_flush_interval = max(int(value), 1)
        except OSError:
            print("WARNING: Couldn't read agent config file '{}'".format(path_to_conf_file))
        except ValueError as exc:
            print("WARNING: Invalid agent config '{}': {}".format(path_to_conf_file, exc))

    @staticmethod
    def _get_hero_actor():
        """
        Search for the ego vehicle in the CARLA world.
        """
        world = CarlaDataProvider.get_world()
        if world is None:
            return None

        for actor in world.get_actors():
            if actor.attributes.get("role_name") == "hero":
                return actor

        return None

    def _set_agent_route(self):
        """
        Convert the leaderboard route into a BasicAgent global plan.
        """
        if not self._global_plan_world_coord:
            return

        carla_map = CarlaDataProvider.get_map()
        if carla_map is None:
            return

        plan = []
        previous_wp = None

        for transform, _ in self._global_plan_world_coord:
            waypoint = carla_map.get_waypoint(transform.location)
            if waypoint is None:
                continue

            if previous_wp is not None:
                traced_route = self._agent.trace_route(previous_wp, waypoint)
                if traced_route:
                    plan.extend(traced_route)

            previous_wp = waypoint

        if plan:
            self._agent.set_global_plan(plan)
            self._route_assigned = True
