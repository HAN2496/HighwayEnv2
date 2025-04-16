import numpy as np
from highway_env.envs import HighwayEnv
from highway_env.utils import save_video, lmap
from controller.dcbf import DCBFIS
from controller.utils import check_possible_lane_changes



config = HighwayEnv.default_config()
#config['action']['type'] = 'LaneChangeWithTargetSpeedAction'
config['action']['type'] = 'LaneChangeWithThrottleAction'
config['observation']['type'] = 'Kinematics'
config['observation']['absolute'] = True
config['observation']['features'] = ["presence", "x", "y", "vx", "vy", "heading", "speed", "length", "width", "lane_index", "target_lane_index", "steering"]
config['policy_frequency'] = 5



if __name__=="__main__":

    env = HighwayEnv(config, render_mode='human')
    #env = HighwayEnv(config, render_mode='rgb_array')

    max_time = 30.0
    max_steps = int(max_time * config['policy_frequency'])

    obs = env.observe()
    dt = 1.0
    sim_steps = int(config['simulation_frequency']/config['policy_frequency'])
    ref_speed = 35.0
    lane_change_frequency = 0.2
    lane_change_time_count = 0.0

    dcbf = DCBFIS(env.controlled_vehicles[0], dt=1.0, ref_speed=35.0, sim_steps=3)

    data_buffer = []
    if env.render_mode=='rgb_array':
        img_buffer = []

    for k in range(max_steps):
        obs = [{key: o[idx] for idx, key in enumerate(config['observation']['features'])} for o in obs]

        # Check possible lane change
        if lane_change_time_count > 1/lane_change_frequency:
            possible_lane_changes = check_possible_lane_changes(env.controlled_vehicles[0])
        else:
            possible_lane_changes = ["IDLE"]

        min_cost = np.inf
        current_speed = obs[0]['speed']
        action = [1, lmap(np.clip(-current_speed/dt, *env.controlled_vehicles[0].ACCELERATION_RANGE), env.controlled_vehicles[0].ACCELERATION_RANGE, (-1.0, 1.0))]
        action = dcbf.solve(obs, possible_lane_changes)
        if action[0]!=1:
            lane_change_time_count = 0.0
        else:
            lane_change_time_count += 1/config['policy_frequency']

        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            break
        data_buffer.append([k/config['policy_frequency'], obs[0, 6], lmap(action[1], (-1.0, 1.0), env.controlled_vehicles[0].ACCELERATION_RANGE)])
        if env.render_mode=='rgb_array':
            img_buffer.append(env.render())

    if env.render_mode=='rgb_array':
        save_video('dcbf_test', img_buffer, extension='gif')

    import matplotlib.pyplot as plt

    data_buffer = np.array(data_buffer)
    plt.figure()
    plt.subplot(211)
    plt.plot(data_buffer[:, 0], ref_speed*np.ones(len(data_buffer)), 'k:')
    plt.plot(data_buffer[:, 0], data_buffer[:, 1], 'k')
    plt.ylim([0.0, 40.0])
    plt.ylabel('v')
    plt.subplot(212)
    plt.plot(data_buffer[:, 0], env.controlled_vehicles[0].ACCELERATION_RANGE[0]*np.ones(len(data_buffer)), 'k:')
    plt.plot(data_buffer[:, 0], env.controlled_vehicles[0].ACCELERATION_RANGE[1]*np.ones(len(data_buffer)), 'k:')
    plt.plot(data_buffer[:, 0], data_buffer[:, 2], 'k')
    plt.ylabel('u')
    plt.xlabel('time [s]')

    plt.show()
