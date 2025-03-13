from highway_env.envs import HighwayEnv

config = HighwayEnv.default_config()

print(config)
config['action']['type'] = 'LaneChangeWithTargetSpeedAction'

env = HighwayEnv(config, render_mode='human')

