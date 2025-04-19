import numpy as np
from highway_env.envs import HighwayEnv
from highway_env.utils import save_video, lmap
from controller.hocbf import HOCBFQP
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

    max_time = 25.0
    max_steps = int(max_time * config['policy_frequency'])

    obs = env.observe()
    #obs, _ = env.reset(seed=2)
    dt = 1.0/config['policy_frequency']
    sim_steps = int(config['simulation_frequency']/config['policy_frequency'])
    ref_speed = 35.0
    lane_change_frequency = 0.2
    lane_change_time_count = 0.0

    #hocbf = HOCBFQP(env.controlled_vehicles[0], dt, ref_speed, alpha1=lambda x:3.6*np.sqrt(x), alpha2=lambda x:2.7*x)
    hocbf = HOCBFQP(env.controlled_vehicles[0], dt, ref_speed, alpha1=lambda x:2.5*np.sqrt(x), alpha2=lambda x:3.0*x)
    #hocbf = HOCBFQP(env.controlled_vehicles[0], dt, ref_speed, alpha1=lambda x:0.5*x, alpha2=lambda x:3.0*x)

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
        #possible_lane_changes = ["IDLE"]
        
        params = []
        for lane_change in possible_lane_changes:
            ego_target_lane_index = obs[0]['target_lane_index']
            if lane_change=='LANE_LEFT':
                ego_target_lane_index -= 1
            if lane_change=='LANE_RIGHT':
                ego_target_lane_index += 1
            slack_penalties = []
            for o in obs[1:]:
                if np.any([o['lane_index']==obs[0]['lane_index'], o['target_lane_index']==obs[0]['lane_index'], o['lane_index']==ego_target_lane_index, o['target_lane_index']==ego_target_lane_index]):
                    slack_penalties.append(4.0)
                else:
                    size = o['width'] * o['length']
                    if size < 8.0:
                        slack_penalties.append(0.01)
                    elif size < 11.0:
                        slack_penalties.append(0.2)
                    elif size < 18.0:
                        slack_penalties.append(0.7)
                    elif size < 25.0:
                        slack_penalties.append(1.5)
                    else:
                        slack_penalties.append(2.6)
            params.append((lane_change, 0.5, slack_penalties,))  # (lane change, speed feedback gain, penalties for slack variables)

        action = hocbf.solve(obs, *params)
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
        save_video('hocbf_varying_alpha', img_buffer, extension='gif')

    import matplotlib.pyplot as plt

    data_buffer = np.array(data_buffer)
    plt.figure(figsize=(11,5))
    plt.subplot(211)
    plt.plot(data_buffer[:, 0], ref_speed*np.ones(len(data_buffer)), 'k:')
    plt.plot(data_buffer[:, 0], data_buffer[:, 1], 'k')
    plt.xlim([data_buffer[0, 0], data_buffer[-1, 0]])
    plt.ylim([0.0, env.controlled_vehicles[0].MAX_SPEED * 1.05])
    plt.ylabel('v [m/s]', fontsize=16)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.subplot(212)
    plt.plot(data_buffer[:, 0], env.controlled_vehicles[0].ACCELERATION_RANGE[0]*np.ones(len(data_buffer)), 'k:')
    plt.plot(data_buffer[:, 0], env.controlled_vehicles[0].ACCELERATION_RANGE[1]*np.ones(len(data_buffer)), 'k:')
    plt.plot(data_buffer[:, 0], data_buffer[:, 2], 'k')
    plt.xlim([data_buffer[0, 0], data_buffer[-1, 0]])
    plt.ylim(np.array(env.controlled_vehicles[0].ACCELERATION_RANGE) * 1.1)
    plt.ylabel('u [m/s^2]', fontsize=16)
    plt.xlabel('time [s]', fontsize=16)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.tight_layout()

    plt.show()
