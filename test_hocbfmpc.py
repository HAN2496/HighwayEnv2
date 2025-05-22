import numpy as np
from highway_env.envs import HighwayEnv
from highway_env.utils import save_video, lmap
from controller.hocbfmpc import HOCBFMPC              # 변경
from controller.utils import check_possible_lane_changes

config = HighwayEnv.default_config()
config['action']['type'] = 'LaneChangeWithThrottleAction'
config['observation']['type'] = 'Kinematics'
config['observation']['absolute'] = True
config['observation']['features'] = [
    "presence", "x", "y", "vx", "vy", "heading", 
    "speed", "length", "width", "lane_index", 
    "target_lane_index", "steering"
]
config['policy_frequency'] = 5

if __name__ == "__main__":
    env = HighwayEnv(config, render_mode='human')
    max_time = 50.0
    max_steps = int(max_time * config['policy_frequency'])

    dt = 1.0 / config['policy_frequency']
    ref_speed = 35.0

    # MPC horizon (예: 10 스텝)
    horizon = 10

    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 7))
    plt.plot([0.0, max_time], ref_speed * np.ones(2), 'k:', label='v_ref')

    # alpha 함수를 s에 따라 바꿔가며 실험
    s_list = list(np.linspace(0.1, 5.0, 11))
    for s in s_list:
        obs, _ = env.reset(seed=2)
        # HOCBFMPC 객체 생성
        mpc = HOCBFMPC(
            vehicle=env.controlled_vehicles[0],
            dt=dt,
            ref_speed=ref_speed,
            horizon=horizon,
            alpha1=lambda x: 2.5 * np.sign(x) * np.sqrt(np.abs(x)),
            alpha2=lambda x: 6.0 * np.sign(x) * np.sqrt(np.abs(x)),
            slack_penalty_max=3
        )

        data_buffer = []
        lane_change_timer = 0.0

        for k in range(max_steps):
            # obs 포맷 맞추기
            obs = [
                {key: o[idx] for idx, key in enumerate(config['observation']['features'])}
                for o in obs
            ]

            # 가능한 차선 변경 결정 (여기선 실험을 위해 항상 IDLE)
            # 실제로는 check_possible_lane_changes() 사용 가능
            possible_lane_changes = ["IDLE"]

            # MPC solve: 여러 후보를 다뤄야 한다면 for문 안에서 각각 호출·비교 가능
            # 여기서는 하나뿐이므로 바로 호출
            action = mpc.solve(obs, possible_lane_changes[0])

            # 차선 변경 타이머 업데이트
            if action[0] != 1:
                lane_change_timer = 0.0
            else:
                lane_change_timer += dt

            # 한 스텝 시뮬레이션
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break

            # 시간, 실제 속도, 명령 가속도 기록
            actual_acc = lmap(
                action[1], (-1.0, 1.0), 
                env.controlled_vehicles[0].ACCELERATION_RANGE
            )
            data_buffer.append([
                k / config['policy_frequency'],
                obs[0, 6],  # speed
                actual_acc
            ])

        # 결과 플롯
        data = np.array(data_buffer)
        plt.plot(data[:, 0], data[:, 1], label=f'slack={s:.2f}')

    plt.ylim([0.0, 40.0])
    plt.ylabel('v [m/s]', fontsize=16)
    plt.xlabel('time [s]', fontsize=16)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()
