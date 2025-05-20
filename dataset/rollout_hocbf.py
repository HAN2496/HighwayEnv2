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
config['observation']['features'] = [
    "presence", "x", "y", "vx", "vy", "heading", "speed", "length", "width",
    "lane_index", "lane_curv", "lat_off", "ang_off",
    "target_lane_index", "target_lane_curv", "target_lat_off", "target_ang_off"
]
config['policy_frequency'] = 5



if __name__=="__main__":

    env = HighwayEnv(config, render_mode='human')

    max_time = 25.0
    max_steps = int(max_time * config['policy_frequency'])

    dt = 1.0/config['policy_frequency']
    sim_steps = int(config['simulation_frequency']/config['policy_frequency'])
    ref_speed = 35.0
    lane_change_frequency = 0.2
    lane_change_time_count = 0.0
    for i, seed in enumerate(range(100)):
        obs, _ = env.reset(seed=seed)
        #hocbf = HOCBFQP(env.controlled_vehicles[0], dt, ref_speed, alpha1=lambda x:2.5*np.sqrt(x), alpha2=lambda x:3.0*x)
        hocbf = HOCBFQP(env.controlled_vehicles[0], dt, ref_speed, alpha1=lambda x:2.5*np.sign(x)*np.sqrt(np.abs(x)), alpha2=lambda x:6.0*np.sign(x)*np.sqrt(np.abs(x)))
        episode = []
        for k in range(max_steps):
            episode.append({"timestamp": k*dt, "observation":obs.copy()})
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
                alpha_scales = []
                for o in obs[1:]:
                    if np.any([o['lane_index']==obs[0]['lane_index'], o['target_lane_index']==obs[0]['lane_index'], o['lane_index']==ego_target_lane_index, o['target_lane_index']==ego_target_lane_index]):
                        slack_penalties.append(5.0)
                        alpha_scales.append(1.0)
                        #slack_penalties.append(1e6)
                        #alpha_scales.append(1.0)
                    else:
                        size = o['width'] * o['length']
                        if size < 8.0:
                            slack_penalties.append(0.1)
                            alpha_scales.append(1.0)
                            #slack_penalties.append(1e6)
                            #alpha_scales.append(4.0)
                        elif size < 11.0:
                            slack_penalties.append(0.8)
                            alpha_scales.append(1.0)
                            #slack_penalties.append(1e6)
                            #alpha_scales.append(3.5)
                        elif size < 18.0:
                            slack_penalties.append(1.5)
                            alpha_scales.append(1.0)
                            #slack_penalties.append(1e6)
                            #alpha_scales.append(3.0)
                        elif size < 25.0:
                            slack_penalties.append(2.2)
                            alpha_scales.append(1.0)
                            #slack_penalties.append(1e6)
                            #alpha_scales.append(2.5)
                        else:
                            slack_penalties.append(2.9)
                            alpha_scales.append(1.0)
                            #slack_penalties.append(1e6)
                            #alpha_scales.append(2.0)
                params.append((lane_change, 0.5, slack_penalties, alpha_scales,))  # (lane change, speed feedback gain, penalties for slack variables)
            episode[-1]["parameters"] = params

            action, QP = hocbf.solve(obs, *params, return_QP=True)
            episode[-1]["action"] = action
            episode[-1]["QP"] = QP
            if action[0]!=1:
                lane_change_time_count = 0.0
            else:
                lane_change_time_count += 1/config['policy_frequency']

            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break
        np.save(f"rollout_hocbf/episode_{i}.npy", np.array(episode, dtype=object), allow_pickle=True)
