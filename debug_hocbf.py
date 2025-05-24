import numpy as np
import pandas as pd
from highway_env.envs import HighwayEnv
from highway_env.utils import save_video, lmap
from controller.hocbfmpc_slack import HOCBFMPC
from controller.utils import check_possible_lane_changes
import matplotlib.pyplot as plt

config = HighwayEnv.default_config()
config['action']['type'] = 'ContinuousAction'
config['observation']['type'] = 'Kinematics'
config['observation']['absolute'] = True
config['observation']['features'] = [
    "presence", "x", "y", "vx", "vy", "heading", "speed",
    "length", "width", "lane_index", "target_lane_index", "steering"
]
config['policy_frequency'] = 5

if __name__ == "__main__":
    env = HighwayEnv(config, render_mode='human')
    obs, _ = env.reset(seed=5)

    vehicle = env.controlled_vehicles[0]
    dt = 1.0 / config['policy_frequency']

    # CBF‐MPC 생성 시 전달한 하이퍼파라미터를 여기도 기록
    ds_safe = 3
    dn_safe = 0.1
    # (alpha1, alpha2, W_cb 도 직접 지정하셨다면 여기에 적어 두세요)
    alpha1_k = 1.0
    alpha2_k = 2.4
    W_cb     = 1e2

    mpc = HOCBFMPC(
        vehicle, dt, ref_speed=35.0,
        ref_lane=vehicle.lane_index,
        init_state=obs[0][[1,2,3,5]],
        horizon=20, ds_safe=ds_safe, dn_safe=dn_safe,
        k1=alpha1_k, k2=alpha2_k, W_cb=W_cb
    )

    max_time = 50.0
    max_steps = int(max_time * config['policy_frequency'])

    # — 여기에 디버깅용 로그 버퍼 —
    log = []
    data_buffer = []
    obs_dist_buffer = []

    for k in range(max_steps):
        obs_dict = [{key: o[idx] for idx, key in enumerate(config['observation']['features'])} for o in obs]
        action = mpc.run(obs_dict)
        obs, reward, terminated, truncated, info = env.step(action)

        ego_x = env.controlled_vehicles[0].position[0]
        ego_y = env.controlled_vehicles[0].position[1]


        # 1. 현재 obstacle과의 거리 계산
        min_obs_dist = None
        if len(obs_dict) > 1:
            dists = [np.hypot(o['x'] - ego_x, o['y'] - ego_y) for o in obs_dict[1:]]
            min_obs_dist = np.min(dists)
        else:
            min_obs_dist = np.nan


        # 실제 trajectory 저장
        data_buffer.append([
            k * dt,
            ego_x,
            ego_y,
            env.controlled_vehicles[0].speed,
            action[1]
        ])

        # 3. cost 비율 (stage cost / slack cost)
        # (acados solver에서 cost 정보 추출 필요, 예시로 None 처리)
        stage_cost = None
        slack_cost = None
        if hasattr(mpc.solver, 'get_cost'):
            stage_cost = mpc.solver.get_cost()
            # slack_cost는 별도 추출 필요 (예시로 None)
        # log 저장
        log.append({
            'time': k * dt,
            'speed': env.controlled_vehicles[0].speed,
            'acceleration': action[1],
            'ds_safe': ds_safe,
            'dn_safe': dn_safe,
            'alpha1_k': alpha1_k,
            'alpha2_k': alpha2_k,
            'W_cb': W_cb,
            'min_obs_dist': min_obs_dist,
            'stage_cost': stage_cost,
            'slack_cost': slack_cost,
            'ego_x': ego_x,
            'ego_y': ego_y,
        })
        obs_dist_buffer.append(min_obs_dist)

        if terminated:
            break


    # DataFrame으로 변환 후 CSV 저장
    df = pd.DataFrame(log)
    df.to_csv('debug_data.csv', index=False)
    print(f"Saved debug log to debug_data.csv ({len(df)} rows)")

    data_buffer = np.array(data_buffer)
    obs_dist_buffer = np.array(obs_dist_buffer)

    # 1. 속도/가속도/장애물 거리 플롯
    plt.figure(figsize=(12, 7))
    plt.subplot(311)
    plt.plot(data_buffer[:, 0], 35 * np.ones(len(data_buffer)), 'k:', label='Ref Speed')
    plt.plot(data_buffer[:, 0], data_buffer[:, 3], 'k', label='Ego Speed')
    plt.ylabel('Speed [m/s]')
    plt.legend()
    plt.subplot(312)
    plt.plot(data_buffer[:, 0], data_buffer[:, 4], 'k')
    plt.ylabel('Acceleration [m/s²]')
    plt.subplot(313)
    plt.plot(data_buffer[:, 0], obs_dist_buffer, 'r')
    plt.ylabel('Min Obs Dist [m]')
    plt.xlabel('Time [s]')
    plt.tight_layout()
    plt.show()

    # 3. cost 비율 플롯 (가능한 경우)
    if 'stage_cost' in df.columns and df['stage_cost'].notnull().any():
        plt.figure()
        plt.plot(df['time'], df['stage_cost'], label='Stage Cost')
        if 'slack_cost' in df.columns and df['slack_cost'].notnull().any():
            plt.plot(df['time'], df['slack_cost'], label='Slack Cost')
            plt.plot(df['time'], df['stage_cost'] / (df['slack_cost'] + 1e-6), label='Cost Ratio')
        plt.xlabel('Time [s]')
        plt.ylabel('Cost')
        plt.legend()
        plt.title('Cost and Slack Cost over Time')
        plt.tight_layout()
        plt.show()