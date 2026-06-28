"""
Local runtime patches for debugging leaderboard scenarios.
"""

import os


def _is_enabled(env_var):
    return os.environ.get(env_var, "").strip().lower() in ("1", "true", "yes", "on")


def _get_env_number(env_var, cast):
    raw_value = os.environ.get(env_var, "").strip()
    if not raw_value:
        return None

    try:
        return cast(raw_value)
    except ValueError:
        print("WARNING: Invalid value '{}' for {}".format(raw_value, env_var))
        return None


if _is_enabled("VISIBLE_ADJACENT_LANE_VEHICLE"):
    try:
        from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
        from srunner.scenarios.adjacent_lane_vehicle import AdjacentLaneVehicle
    except Exception:
        # If the scenario runner is not available in the current process, skip the patch.
        pass
    else:
        if not getattr(AdjacentLaneVehicle, "_visible_spawn_patch", False):
            def _initialize_actors_visible(self, config):
                lane_waypoint = self._get_adjacent_lane_waypoint()

                drive_operation = self._get_operation_for_direction(lane_waypoint)
                spawn_operation = "previous" if drive_operation == "next" else "next"

                spawn_waypoint = self._advance_waypoint(lane_waypoint, spawn_operation, self._spawn_distance)
                self._plan = self._build_plan(spawn_waypoint, drive_operation)
                self._spawn_transform = self._make_transform(self._plan[0][0], self._plan[1][0])

                actor_model = config.other_actors[0].model if config.other_actors else "vehicle.*"
                actor = CarlaDataProvider.request_new_actor(
                    actor_model,
                    self._spawn_transform,
                    rolename="scenario no lights",
                    attribute_filter={"base_type": "car", "generation": 2},
                )
                if actor is None:
                    raise ValueError("Couldn't spawn the adjacent lane vehicle")

                # Keep the actor visible in its spawn point until the trigger starts the behavior.
                actor.set_simulate_physics(False)
                self.other_actors.append(actor)

            AdjacentLaneVehicle._initialize_actors = _initialize_actors_visible
            AdjacentLaneVehicle._visible_spawn_patch = True


try:
    from srunner.scenarios.route_obstacles import ParkedObstacle
    from srunner.scenarios.route_obstacles import get_value_parameter as _get_route_obstacle_value
except Exception:
    # If the scenario runner is not available in the current process, skip the patch.
    pass
else:
    if not getattr(ParkedObstacle, "_xml_end_distance_patch", False):
        _parked_obstacle_original_init = ParkedObstacle.__init__

        def _parked_obstacle_init_with_end_distance(
            self,
            world,
            ego_vehicles,
            config,
            randomize=False,
            debug_mode=False,
            criteria_enable=True,
            timeout=180,
        ):
            _parked_obstacle_original_init(
                self,
                world,
                ego_vehicles,
                config,
                randomize,
                debug_mode,
                criteria_enable,
                timeout,
            )
            self._end_distance = _get_route_obstacle_value(config, "end_distance", float, self._end_distance)

        ParkedObstacle.__init__ = _parked_obstacle_init_with_end_distance
        ParkedObstacle._xml_end_distance_patch = True


_route_completion_percentage = _get_env_number("LEADERBOARD_ROUTE_COMPLETION_PERCENTAGE", float)
_route_completion_distance = _get_env_number("LEADERBOARD_ROUTE_COMPLETION_DISTANCE_M", float)

if _route_completion_percentage is not None or _route_completion_distance is not None:
    try:
        from srunner.scenariomanager.scenarioatomics.atomic_criteria import RouteCompletionTest
    except Exception:
        # If the scenario runner is not available in the current process, skip the patch.
        pass
    else:
        if not getattr(RouteCompletionTest, "_leaderboard_completion_patch", False):
            if _route_completion_percentage is not None:
                RouteCompletionTest.PERCENTAGE_THRESHOLD = float(_route_completion_percentage)
            if _route_completion_distance is not None:
                RouteCompletionTest.DISTANCE_THRESHOLD = float(_route_completion_distance)

            RouteCompletionTest._leaderboard_completion_patch = True
