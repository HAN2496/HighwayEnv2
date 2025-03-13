from highway_env.envs import HighwayEnv

config = HighwayEnv.default_config()
config['action']['type'] = 'LaneChangeWithTargetSpeedAction'
#config['action']['type'] = 'LaneChangeWithThrottleAction'
config['observation']['type'] = 'Kinematics'
config['observation']['features'] = ["presence", "x", "y", "vx", "vy", "heading", "length", "width"]



if __name__=="__main__":

    env = HighwayEnv(config, render_mode='human')

    lane_change = 1  # {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT"}
    if config['action']['type']=='LaneChangeWithTargetSpeedAction':
        target_speed = 20.0
        target_speed_normalized = target_speed / env.controlled_vehicles[0].MAX_SPEED
        action = (target_speed_normalized, lane_change)
    elif config['action']['type']=='LaneChangeWithThrottleAction':
        acc = 1.5
        acc_normalized = acc / env.controlled_vehicles[0].ACCELERATION_RANGE[-1]
        action = (acc_normalized, lane_change)

    max_time = 10.0
    max_steps = int(max_time / config['policy_frequency'])

    
    for k in range(max_steps):
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            break

