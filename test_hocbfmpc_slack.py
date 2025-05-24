import numpy as np
from highway_env.envs import HighwayEnv
from highway_env.utils import save_video, lmap
# from controller.hocbfmpc_slack import HOCBFMPC              # 변경
# from controller.hocbfmpc_slack2 import HOCBFMPC              # 변경
from controller.hocbfmpc_slack3 import HOCBFMPC              # 변경
from controller.utils import check_possible_lane_changes
import matplotlib.pyplot as plt

config = HighwayEnv.default_config()
# config['action']['type'] = 'LaneChangeWithThrottleAction'
config['action']['type'] = 'ContinuousAction' #ContinuousAction
config['observation']['type'] = 'Kinematics'
config['observation']['absolute'] = True
config['observation']['features'] = [
    "presence", "x", "y", "vx", "vy", "heading", "speed",
    "length", "width", "lane_index", "target_lane_index", "steering"
]
config['policy_frequency'] = 5

if __name__ == "__main__":
    env = HighwayEnv(config, render_mode='human')
    obs, _ = env.reset(seed=2)

    vehicle = env.controlled_vehicles[0]
    dt = 1.0 / config['policy_frequency']

    mpc = HOCBFMPC(vehicle, dt, ref_speed=35.0, ref_lane= env.controlled_vehicles[0].lane_index,
                   init_state = obs[0][[1,2,3,5]],
                   horizon=20, ds_safe=3, dn_safe = 0.2, k1=2.2, k2=2.4, W_cb=2e3,
                   mu=np.array([-2.5, 0.0]), Sigma=np.diag([7.5**2, 0.1**2]), confidence=0.98)

    max_time = 50.0
    max_steps = int(max_time * config['policy_frequency'])
    sim_steps = int(config['simulation_frequency'] / config['policy_frequency'])

    data_buffer = []
    if env.render_mode == 'rgb_array':
        img_buffer = []

    lane_change_time_count = 0.0
    lane_change_frequency = 0.2

    for k in range(max_steps):
        obs = [{key: o[idx] for idx, key in enumerate(config['observation']['features'])} for o in obs]
        
        action = mpc.run( obs=obs)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        data_buffer.append([
            k * dt,
            env.controlled_vehicles[0].speed,
            action[1]
        
        ])
        if env.render_mode == 'rgb_array':
            img_buffer.append(env.render())
        if terminated:
            break

    data_buffer = np.array(data_buffer)
    plt.figure(figsize=(11, 5))
    plt.subplot(211)
    plt.plot(data_buffer[:, 0], 35 * np.ones(len(data_buffer)), 'k:')
    plt.plot(data_buffer[:, 0], data_buffer[:, 1], 'k')
    plt.ylabel('Speed [m/s]', fontsize=14)
    plt.subplot(212)
    plt.plot(data_buffer[:, 0], data_buffer[:, 2], 'k')
    plt.xlabel('Time [s]', fontsize=14)
    plt.ylabel('Acceleration [m/s²]', fontsize=14)
    plt.tight_layout()
    plt.show()

    if env.render_mode == 'rgb_array':
        save_video('mpc_tracking', img_buffer, extension='gif')
