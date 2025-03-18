import numpy as np
from scipy.differentiate import derivative, jacobian, hessian
EPS = np.finfo(np.float32).eps**0.5
from scipy.optimize import approx_fprime

from highway_env.envs import HighwayEnv
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.utils import Minkowski_sum, signed_distance

def CBF_QP(obs, eom):
    pass



config = HighwayEnv.default_config()
config['action']['type'] = 'LaneChangeWithTargetSpeedAction'
#config['action']['type'] = 'LaneChangeWithThrottleAction'
config['observation']['type'] = 'Kinematics'
config['observation']['features'] = ["presence", "x", "y", "vx", "vy", "heading", "length", "width", "lane_index", "target_lane_index"]



if __name__=="__main__":

    env = HighwayEnv(config, render_mode='human')

    lane_change = 1  # {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT"}
    if config['action']['type']=='LaneChangeWithTargetSpeedAction':
        target_speed = 20.0
        target_speed_normalized = target_speed / env.controlled_vehicles[0].MAX_SPEED
        action = (lane_change, target_speed_normalized)
    elif config['action']['type']=='LaneChangeWithThrottleAction':
        acc = 1.5
        acc_normalized = acc / env.controlled_vehicles[0].ACCELERATION_RANGE[-1]
        action = (lane_change, acc_normalized)

    max_time = 10.0
    max_steps = int(max_time / config['policy_frequency'])

    for k in range(max_steps):
        obs, reward, terminated, truncated, info = env.step(action)
        print(obs)
        if terminated:
            break

