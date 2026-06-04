#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Simple autonomous agent that follows the leaderboard route using CARLA's BasicAgent.
"""

from __future__ import print_function

import carla
import numpy as np
from ultralytics import YOLO
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.local_planner import RoadOption
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track

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

    def _draw_detection_overlay(self, detections):
        if not detections:
            return

        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            rect = pygame.Rect(int(x1), int(y1), max(int(x2 - x1), 1), max(int(y2 - y1), 1))
            pygame.draw.rect(self._display, (255, 200, 64), rect, 2)

            label = "{} {:.2f}".format(detection["label"], detection["confidence"])
            if detection["distance_m"] is not None:
                label = "{} {:.1f}m".format(label, detection["distance_m"])

            label_surface = self._small_font.render(label, True, (24, 24, 24), (255, 200, 64))
            label_y = max(rect.y - label_surface.get_height() - 2, 0)
            self._display.blit(label_surface, (rect.x, label_y))

    def render(self, center_image=None, left_image=None, speed_mps=None,
               detections=None,
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
            self._draw_text_lines(["LeftCam"], (inset_rect.x + 8, inset_rect.y + 8), self._small_font)

        info_lines = ["Center RGB + LeftCam active"]
        if speed_mps is None:
            info_lines.append("Speed: unavailable")
        else:
            info_lines.append("Speed: {:.2f} m/s ({:.1f} km/h)".format(speed_mps, speed_mps * 3.6))
        if detections:
            info_lines.append("YOLO: {} objs".format(len(detections)))
        else:
            info_lines.append("YOLO: none")
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
        self._target_speed = 20.0
        self.camera_width = 1280
        self.camera_height = 720
        self._display = CameraDisplay(self.camera_width, self.camera_height)
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
        self._front_radar_trigger_distance = 10.0
        self._front_radar_resume_distance = 12.0
        self._front_radar_min_points = 2
        self._front_radar_max_mean_abs_azimuth = np.deg2rad(4.0)
        self._front_vehicle_min_confidence = 0.45
        self._front_vehicle_center_x_min = 0.30
        self._front_vehicle_center_x_max = 0.70
        self._front_vehicle_min_bottom_ratio = 0.35
        self._rear_approach_distance = 18.0
        self._rear_approach_velocity = -2.0
        self._left_oncoming_min_confidence = 0.45
        self._left_oncoming_min_area_ratio = 0.0025
        self._left_oncoming_min_center_x_ratio = 0.15
        self._left_oncoming_max_center_x_ratio = 0.90
        self._sensor_stop_brake = 0.60
        self._sensor_emergency_brake = 0.95
        self._overtake_fail_safe_distance = 6.0
        self._overtake_fail_safe_velocity = -2.0
        self._blocked_hold_time = 1.0
        self._overtake_speed = min(max(self._target_speed, 15.0), 25.0)
        self._overtake_same_lane_distance = 2.0
        self._overtake_lane_change_distance = 8.0
        self._overtake_other_lane_distance = 20.0
        self._return_same_lane_distance = 4.0
        self._return_lane_change_distance = 8.0
        self._return_other_lane_distance = 8.0
        self._min_left_lane_time = 1.5
        self._min_right_lane_settle_time = 0.8
        self._post_overtake_cooldown_time = 2.0
        self._route_rejoin_min_distance = 8.0
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
        self._last_leftcam_filter_debug_timestamp = -1.0
        self._last_overtake_fail_safe_log_timestamp = -1.0

        if path_to_conf_file:
            self._load_config(path_to_conf_file)

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
            }
        ]

    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.
        """
        self._current_timestamp = timestamp
        left_camera = input_data.get("LeftCam")
        center_camera = input_data.get("Center")
        speed_data = input_data.get("Speed")
        front_radar = input_data.get("FrontRdr")
        rear_radar = input_data.get("RearRdr")

        center_image = center_camera[1] if center_camera is not None else None
        left_image = left_camera[1] if left_camera is not None else None
        yolo_detections = []
        speed_mps = speed_data[1]["speed"] if speed_data is not None else None
        raw_front_radar_points = front_radar[1] if front_radar is not None else None
        raw_rear_radar_points = rear_radar[1] if rear_radar is not None else None
        front_radar_points = self._extract_front_cluster(raw_front_radar_points)
        rear_radar_points = self._extract_rear_cluster(raw_rear_radar_points)
        front_vehicle_state = self._estimate_front_vehicle_state(front_radar_points)
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
        control = self._agent.run_step()
        control = self._apply_sensor_navigation(
            control,
            current_waypoint,
            front_vehicle_state,
            left_oncoming,
        )
        self._display.render(
            center_image=center_image,
            left_image=left_image,
            speed_mps=speed_mps,
            detections=[],
            navigation_state=self._overtake_state,
            front_radar=front_radar_points,
            rear_radar=rear_radar_points,
        )
        return control

    def _log_sensor_snapshot(self, center_image, left_yolo_detections,
                             front_vehicle_state,
                             left_oncoming,
                             rear_approaching,
                             left_image, speed_data,
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

    def _apply_sensor_navigation(self, control, current_waypoint, front_vehicle_state, left_oncoming):
        if self._overtake_state in ("follow_lane", "waiting_left"):
            if self._is_front_obstacle_active(front_vehicle_state):
                brake_value = self._sensor_emergency_brake if front_vehicle_state["emergency"] else self._sensor_stop_brake
                return self._apply_brake_override(control, brake_value)

        if self._overtake_state == "changing_left" and current_waypoint is not None:
            if self._overtake_origin_lane_id is not None and current_waypoint.lane_id == self._overtake_origin_lane_id:
                if front_vehicle_state["distance_m"] is not None and front_vehicle_state["distance_m"] < self._front_vehicle_emergency_distance:
                    return self._apply_brake_override(control, self._sensor_stop_brake)

        if self._overtake_state in ("changing_left", "passing"):
            if left_oncoming is not None:
                self._log_overtake_fail_safe("left_oncoming", front_vehicle_state, left_oncoming)
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
                return self._apply_brake_override(control, self._sensor_emergency_brake)

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
        del speed_mps  # Behavior is distance-driven for now.
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
            if current_waypoint is not None and self._overtake_origin_lane_id is not None:
                if current_waypoint.lane_id != self._overtake_origin_lane_id:
                    self._overtake_state = "passing"
                    self._left_lane_entered_at = timestamp

        elif self._overtake_state == "passing":
            if self._should_return_right(current_waypoint, front_vehicle_state, timestamp):
                self._start_lane_change(current_waypoint, "right", timestamp)

        elif self._overtake_state == "changing_right":
            if current_waypoint is not None and self._overtake_origin_lane_id is not None:
                if current_waypoint.lane_id == self._overtake_origin_lane_id:
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
        if rear_approaching is not None:
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
                        "distance_m": self._overtake_same_lane_distance,
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
            self._overtake_state = "changing_left"
            self._agent.set_global_plan(plan)
            self._agent.set_target_speed(self._overtake_speed)
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
            self._agent.set_global_plan(plan)
            self._agent.set_target_speed(self._overtake_speed)
            return True

        return False

    def _should_return_right(self, current_waypoint, front_vehicle_state, timestamp):
        if current_waypoint is None:
            return False
        if self._left_lane_entered_at is None or timestamp - self._left_lane_entered_at < self._min_left_lane_time:
            return False
        if front_vehicle_state["distance_m"] is not None and front_vehicle_state["distance_m"] < self._front_vehicle_resume_distance:
            return False

        rejoin_waypoint, _rejoin_direction = self._find_origin_adjacent_lane(current_waypoint)
        if rejoin_waypoint is None:
            return False

        return True

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

        if self._agent is not None:
            self._agent.set_target_speed(self._target_speed)

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

    def _estimate_front_vehicle_state(self, front_radar_points):
        front_radar_depth = None
        front_radar_velocity = None
        front_radar_count = 0
        front_radar_mean_abs_azimuth = None
        if front_radar_points is not None and len(front_radar_points) > 0:
            front_radar_depth = float(np.min(front_radar_points[:, 0]))
            front_radar_velocity = float(np.median(front_radar_points[:, 3]))
            front_radar_count = len(front_radar_points)
            front_radar_mean_abs_azimuth = float(np.mean(np.abs(front_radar_points[:, 2])))

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

        candidates = valid_points[
            (valid_points[:, 0] <= self._rear_approach_distance)
            & (valid_points[:, 3] <= self._rear_approach_velocity)
        ]
        if len(candidates) == 0:
            return None

        nearest = candidates[np.argmin(candidates[:, 0])]
        return {
            "distance_m": float(nearest[0]),
            "velocity_mps": float(nearest[3]),
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
        return "{:.1f}m {:.1f}m/s".format(
            rear_approaching["distance_m"],
            rear_approaching["velocity_mps"],
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

    def destroy(self):
        """
        Cleanup agent state.
        """
        if hasattr(self, "_display") and self._display is not None:
            self._display.close()
            self._display = None
        self._agent = None
        self._hero_actor = None
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
