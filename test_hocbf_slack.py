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

    dt = 1.0/config['policy_frequency']
    sim_steps = int(config['simulation_frequency']/config['policy_frequency'])
    ref_speed = 35.0
    lane_change_frequency = 0.2
    lane_change_time_count = 0.0

    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,7))
    plt.plot([0.0, 25.0], ref_speed*np.ones(2), 'k:')
    s_list = list(np.linspace(1e-1, 5.0, 11))
    #s_list = list(np.linspace(1e-2**0.5, 5.0**0.5, 11)**2)
    for s in s_list:
        obs, _ = env.reset(seed=2)
        #hocbf = HOCBFQP(env.controlled_vehicles[0], dt, ref_speed, alpha1=lambda x:2.5*np.sqrt(x), alpha2=lambda x:3.0*x)
        hocbf = HOCBFQP(env.controlled_vehicles[0], dt, ref_speed, alpha1=lambda x:2.5*np.sign(x)*np.sqrt(np.abs(x)), alpha2=lambda x:6.0*np.sign(x)*np.sqrt(np.abs(x)))
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
            possible_lane_changes = ["IDLE"]
            
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
                        slack_penalties.append(5e0)
                    else:
                        slack_penalties.append(s)
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


        data_buffer = np.array(data_buffer)
        plt.plot(data_buffer[:, 0], data_buffer[:, 1])
    plt.ylim([0.0, 40.0])
    plt.ylabel('v [m/s]', fontsize=16)
    plt.xlabel('time [s]', fontsize=16)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.legend(['v_ref'] + [f'{s:.2f}' for s in s_list])
    plt.tight_layout()

    plt.show()
