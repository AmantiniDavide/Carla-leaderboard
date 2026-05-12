#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Simple autonomous agent that follows the leaderboard route using CARLA's BasicAgent.
"""

from __future__ import print_function

import carla
import numpy as np
from agents.navigation.basic_agent import BasicAgent
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
        rgb_image = image[:, :, -2::-1]
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

    def _draw_radar_panel(self, rect, title, points):
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

        max_depth = min(max(float(np.max(points[:, 3])), 1.0), 50.0)
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
            "min {:.1f} m".format(float(np.min(points[:, 3]))),
            "vel [{:.1f}, {:.1f}] m/s".format(float(np.min(points[:, 0])), float(np.max(points[:, 0]))),
        ]
        self._draw_text_lines(summary_lines, (rect.x + 8, rect.y + rect.h - 56), self._small_font)

    def render(self, center_image=None, left_image=None, speed_mps=None, front_radar=None, rear_radar=None):
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

        if center_image is not None and left_image is not None:
            left_surface = pygame.transform.smoothscale(
                self._image_to_surface(left_image),
                (self._width // 4, self._height // 4),
            )
            inset_rect = pygame.Rect(20, 20, self._width // 4, self._height // 4)
            pygame.draw.rect(self._display, (0, 0, 0), inset_rect.inflate(6, 6))
            self._display.blit(left_surface, inset_rect.topleft)
            self._draw_text_lines(["LeftCam"], (inset_rect.x + 8, inset_rect.y + 8), self._small_font)

        info_lines = ["Center + LeftCam active"]
        if speed_mps is None:
            info_lines.append("Speed: unavailable")
        else:
            info_lines.append("Speed: {:.2f} m/s ({:.1f} km/h)".format(speed_mps, speed_mps * 3.6))
        info_lines.append("Radar format: [velocity, altitude, azimuth, depth]")
        self._draw_text_lines(info_lines, (20, self._height - 86))

        panel_width = 280
        panel_height = 160
        self._draw_radar_panel(
            pygame.Rect(self._width - panel_width - 20, 20, panel_width, panel_height),
            "FrontRdr",
            front_radar,
        )
        self._draw_radar_panel(
            pygame.Rect(self._width - panel_width - 20, 200, panel_width, panel_height),
            "RearRdr",
            rear_radar,
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
        self._last_debug_timestamp = -1.0
        self._sensor_snapshot_logged = False

        if path_to_conf_file:
            self._load_config(path_to_conf_file)

    def sensors(self):
        """
        Define the sensor suite required by the agent.
        """
        return [
            {
                "type": "sensor.camera.rgb",
                "x": 0.7,
                "y": 0.0,
                "z": 1.60,
                "roll": 0.0,
                "pitch": 0.0,
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
                "yaw": 0.0,
                "width": self.camera_width,
                "height": self.camera_height,
                "fov": 100,
                "id": "LeftCam",
            },
            {
               "type": "sensor.other.radar",
                "x": 1.7,
                "y": 0.0,
                "z": 1.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "horizontal_fov": 25,
                "vertical_fov": 15,
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
        left_camera = input_data.get("LeftCam")
        center_camera = input_data.get("Center")
        speed_data = input_data.get("Speed")
        front_radar = input_data.get("FrontRdr")
        rear_radar = input_data.get("RearRdr")

        center_image = center_camera[1] if center_camera is not None else None
        left_image = left_camera[1] if left_camera is not None else None
        speed_mps = speed_data[1]["speed"] if speed_data is not None else None
        front_radar_points = front_radar[1] if front_radar is not None else None
        rear_radar_points = rear_radar[1] if rear_radar is not None else None

        self._log_sensor_snapshot(
            center_image,
            left_image,
            speed_data[1] if speed_data is not None else None,
            front_radar_points,
            rear_radar_points,
        )
        self._print_sensor_summary(timestamp, speed_mps, front_radar_points, rear_radar_points)
        self._display.render(center_image, left_image, speed_mps, front_radar_points, rear_radar_points)

        if self._agent is None:
            self._hero_actor = self._get_hero_actor()
            if self._hero_actor is None:
                return carla.VehicleControl()

            self._agent = BasicAgent(self._hero_actor, self._target_speed)

        if not self._route_assigned:
            self._set_agent_route()
            if not self._route_assigned:
                return carla.VehicleControl()

        return self._agent.run_step()

    def _log_sensor_snapshot(self, center_image, left_image, speed_data, front_radar_points, rear_radar_points):
        if self._sensor_snapshot_logged:
            return

        if center_image is not None:
            print("[Sensors] Center image shape={} dtype={}".format(center_image.shape, center_image.dtype))
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

    def _print_sensor_summary(self, timestamp, speed_mps, front_radar_points, rear_radar_points):
        if timestamp - self._last_debug_timestamp < 1.0:
            return

        self._last_debug_timestamp = timestamp
        speed_text = "unavailable"
        if speed_mps is not None:
            speed_text = "{:.2f} m/s ({:.1f} km/h)".format(speed_mps, speed_mps * 3.6)

        print(
            "[Sensors] Speed={} | FrontRdr={} | RearRdr={}".format(
                speed_text,
                self._summarize_radar(front_radar_points),
                self._summarize_radar(rear_radar_points),
            )
        )

    @staticmethod
    def _summarize_radar(points):
        if points is None:
            return "missing"
        if len(points) == 0:
            return "0 pts"

        return "{count} pts, min={min_depth:.1f} m, mean={mean_depth:.1f} m, vel=[{min_vel:.1f}, {max_vel:.1f}] m/s".format(
            count=len(points),
            min_depth=float(np.min(points[:, 3])),
            mean_depth=float(np.mean(points[:, 3])),
            min_vel=float(np.min(points[:, 0])),
            max_vel=float(np.max(points[:, 0])),
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
