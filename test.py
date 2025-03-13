import numpy as np
from highway_env.envs import HighwayEnv
from highway_env.vehicle.controller import LaneChangeWithTargetSpeedVehicle
from highway_env.utils import are_polygons_intersecting  # are_polygons_intersecting(poly1, poly2, vel1, vel2)

def polygon(length, width, heading, position) -> np.ndarray:
    points = np.array(
        [
            [-length / 2, -width / 2],
            [-length / 2, +width / 2],
            [+length / 2, +width / 2],
            [+length / 2, -width / 2],
        ]
    ).T
    c, s = np.cos(heading), np.sin(heading)
    rotation = np.array([[c, -s], [s, c]])
    points = (rotation @ points).T + np.tile(position, (4, 1))
    return np.vstack([points, points[0:1]])


config = HighwayEnv.default_config()

print(config)
config['action']['type'] = 'LaneChangeWithTargetSpeedAction'
config['observation']['type'] = 'Kinematics'
config['observation']['features'] = ["presence", "x", "y", "vx", "vy", "heading", "length", "width"]

env = HighwayEnv(config, render_mode='human')

vehicle = LaneChangeWithTargetSpeedVehicle.create_from(env.controlled_vehicles[0])
