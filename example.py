from highway_env.envs import HighwayEnv
from highway_env.utils import save_video
from numpy.random import choice


config = HighwayEnv.default_config()
config['action']['type'] = 'LaneChangeWithTargetSpeedAction'
#config['action']['type'] = 'LaneChangeWithThrottleAction'
config['observation']['type'] = 'Kinematics'
config['observation']['features'] = ["presence", "x", "y", "vx", "vy", "heading", "length", "width", "lane_index", "target_lane_index", "lat_off"]
config['policy_frequency'] = config['simulation_frequency']


def rotation(P, angle):
    return P @ np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])

def draw_polygon(P, *args, **kwds):
    plt.plot([*P[:, 0], P[0, 0]], [*P[:, 1], P[0, 1]], *args, **kwds)

if __name__=="__main__":

    env = HighwayEnv(config, render_mode='human')
    #env = HighwayEnv(config, render_mode='rgb_array')

    lane_change = 1  # {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT"}
    if config['action']['type']=='LaneChangeWithTargetSpeedAction':
        target_speed = 27.0
        target_speed_normalized = target_speed / env.controlled_vehicles[0].MAX_SPEED
        action = (lane_change, target_speed_normalized)
    elif config['action']['type']=='LaneChangeWithThrottleAction':
        acc = 1.0
        acc_normalized = acc / env.controlled_vehicles[0].ACCELERATION_RANGE[-1]
        action = (lane_change, acc_normalized)

    max_time = 10.0
    max_steps = int(max_time * config['policy_frequency'])

    buffer = []
    imsize = None
    dpi = 50
    import numpy as np
    import matplotlib.pyplot as plt
    P = np.array(
        [
            [ env.controlled_vehicles[0].LENGTH/2, env.controlled_vehicles[0].WIDTH/2],
            [-env.controlled_vehicles[0].LENGTH/2, env.controlled_vehicles[0].WIDTH/2],
            [-env.controlled_vehicles[0].LENGTH/2,-env.controlled_vehicles[0].WIDTH/2],
            [ env.controlled_vehicles[0].LENGTH/2,-env.controlled_vehicles[0].WIDTH/2]
        ]
    )
    position_init = np.array(env.controlled_vehicles[0].position)
    for k in range(max_steps):
        obs, reward, terminated, truncated, info = env.step(action)
        print(obs)
        if abs(obs[0][-1]) < 0.01:
            action = (choice([0, 1, 2]), action[1])
        else:
            action = (1, action[1])
        if terminated:
            break
        if env.render_mode=='rgb_array':
            buffer.append(env.render())
        #env.controlled_vehicles[0].position
        print({0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT"}[action[0]])
        print(env.controlled_vehicles[0].heading)
        print(env.controlled_vehicles[0].position)
        plt.plot()
        draw_polygon(rotation(P, env.controlled_vehicles[0].heading) + env.controlled_vehicles[0].position)
        plt.scatter(env.controlled_vehicles[0].position[0], position_init[1])
        plt.axis("equal")
        plt.show()
            
    if env.render_mode=='rgb_array':
        save_video('test', buffer, extension='mp4')

