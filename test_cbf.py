from multiprocess.pool import Pool
import numpy as np
from scipy.optimize import minimize, approx_fprime, NonlinearConstraint

from highway_env.envs import HighwayEnv
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.utils import save_video
from control.utils import get_polygon, Minkowski_sum, signed_distance, check_possible_lane_changes, predict_state



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

    lane_change = 1  # {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT"}
    lane_change_dict = {"LANE_LEFT": 0, "IDLE": 1, "LANE_RIGHT": 2}
    if config['action']['type']=='LaneChangeWithTargetSpeedAction':
        target_speed = 30.0
        target_speed_normalized = target_speed / env.controlled_vehicles[0].MAX_SPEED
        action = (lane_change, target_speed_normalized)
    elif config['action']['type']=='LaneChangeWithThrottleAction':
        acc = 1.5
        acc_normalized = acc / env.controlled_vehicles[0].ACCELERATION_RANGE[-1]
        action = (lane_change, acc_normalized)

    max_time = 50.0
    max_steps = int(max_time * config['policy_frequency'])

    obs = env.observe()
    dt = 1.0
    sim_steps = int(config['simulation_frequency']/config['policy_frequency'])
    ref_speed = 25.0
    alpha = 0.8
    d = 1.0
    lane_change_frequency = 2
    lane_change_hist = [1] * int(lane_change_frequency * config['policy_frequency'])

    data_buffer = []
    if env.render_mode=='rgb_array':
        img_buffer = []

    for k in range(max_steps):
        obs = [{key: o[idx] for idx, key in enumerate(config['observation']['features'])} for o in obs]

        current_speed = obs[0]['speed']

        current_ego_polygon = get_polygon(obs[0]['length'], obs[0]['width'], obs[0]['heading'])
        other_polygon_list = [get_polygon(o['length'], o['width'], o['heading']) for o in obs[1:]]
        current_h_list = [
            signed_distance(
                Minkowski_sum(
                    current_ego_polygon, other_polygon
                ), np.array([obs[0]['x'] - o['x'], obs[0]['y'] - o['y']]) 
            ) - d for o, other_polygon in zip(obs[1:], other_polygon_list)
        ]
        #print(current_h_list)

        # Check possible lane change
        if np.all(np.array(lane_change_hist)==1):
            possible_lane_changes = check_possible_lane_changes(env.controlled_vehicles[0])
        else:
            possible_lane_changes = ["IDLE"]

        #cost = np.inf
        #action = [1, np.clip(-current_speed/dt, *env.controlled_vehicles[0].ACCELERATION_RANGE)/5.0]
        #res_list = []
        #for lane_change in possible_lane_changes:
        #    cbf_constraint_list = []
        #    for o, other_polygon, current_h in zip(obs[1:], other_polygon_list, current_h_list):
        #        def next_V(u):
        #            next_speed = current_speed + u * dt
        #            return (ref_speed - next_speed)**2 + 1e-1*u**2
        #            #return (ref_speed - u)**2
        #        def next_h(u):
        #            x, y, heading, speed = predict_state(
        #                dt, sim_steps, env.controlled_vehicles[0], lane_change=lane_change,
        #                acceleration=u
        #                #target_speed=u
        #            )
        #            next_ego_polygon = get_polygon(obs[0]['length'], obs[0]['width'], heading)
        #            return signed_distance(Minkowski_sum(next_ego_polygon, other_polygon), np.array([x - (o['x'] + dt * o['vx']), y - (o['y'] + dt * o['vy'])])) - d - max((1-alpha) * current_h, 0.0)
        #        cbf_constraint_list.append(
        #            #NonlinearConstraint(next_h, lb=0.0, ub=np.inf)
        #            {'type':'ineq', 'fun':next_h}
        #        )
        #    res = minimize(
        #        fun=next_V, x0=np.array([action[1]*5.0]),
        #        method='SLSQP',
        #        bounds=[env.controlled_vehicles[0].ACCELERATION_RANGE,],
        #        #bounds=[(env.controlled_vehicles[0].MIN_SPEED, env.controlled_vehicles[0].MAX_SPEED),],
        #        constraints=cbf_constraint_list
        #    )
        #    res_list.append(res)
        #for res in res_list:
        #    x, y, heading, speed = predict_state(
        #        dt, sim_steps, env.controlled_vehicles[0], lane_change=lane_change,
        #        acceleration=res.x
        #        #target_speed=res.x
        #    )
        #    next_ego_polygon = get_polygon(obs[0]['length'], obs[0]['width'], heading)
        #    next_h_list = np.array([signed_distance(Minkowski_sum(next_ego_polygon, other_polygon), np.array([x - (o['x'] + dt * o['vx']), y - (o['y'] + dt * o['vy'])])) - d for o, other_polygon in zip(obs[1:], other_polygon_list)])
        #    if np.all(next_h_list > 0.0):
        #        if cost > res.fun + 1e0*np.fabs(lane_change_dict[lane_change]-1):
        #            cost = res.fun + 1e0*np.fabs(lane_change_dict[lane_change]-1)
        #            action[0]=lane_change_dict[lane_change]
        #            action[1] = res.x[0] / 5.0
        #            #action[1] = res.x[0] / 40.0
        #print(action, cost)
        #x, y, heading, speed = predict_state(
        #    dt, sim_steps, env.controlled_vehicles[0], lane_change=lane_change,
        #    acceleration=action[1]*5.0
        #    #target_speed=u
        #)
        #next_ego_polygon = get_polygon(obs[0]['length'], obs[0]['width'], heading)
        #print([signed_distance(Minkowski_sum(next_ego_polygon, other_polygon), np.array([x - (o['x'] + dt * o['vx']), y - (o['y'] + dt * o['vy'])])) - d - max((1-alpha) * current_h, 0.0) for o, other_polygon in zip(obs[1:], other_polygon_list)])

        lam = 1e2
        cost = np.inf
        uref = np.clip(-current_speed/dt, *env.controlled_vehicles[0].ACCELERATION_RANGE)/5.0
        action = [1, uref]
        res_list = []
        print(possible_lane_changes)
        for lane_change in possible_lane_changes:
            #cost_list = []
            #u_list = []
            #for u in np.linspace(-5, 5, 21):
            #    x, y, heading, speed = predict_state(
            #        dt, int(dt * config['simulation_frequency']), env.controlled_vehicles[0], lane_change=lane_change,
            #        acceleration=u
            #        #target_speed=u
            #    )
            #    next_ego_polygon = get_polygon(obs[0]['length'], obs[0]['width'], heading)
            #    cbf_constraint_list = []
            #    for o, other_polygon, current_h in zip(obs[1:], other_polygon_list, current_h_list):
            #        next_h = signed_distance(Minkowski_sum(next_ego_polygon, other_polygon), np.array([x - (o['x'] + dt * o['vx']), y - (o['y'] + dt * o['vy'])])) - d
            #        cbf_constraint_list.append(next_h >= (1-alpha) * current_h)
            #    if np.all(cbf_constraint_list):
            #        u_list.append(u)
            #        cost_list.append((ref_speed - speed)**2 + 1e-6*(u-uref)**2)
            def target(u):
                x, y, heading, speed = predict_state(
                    dt, int(dt * config['simulation_frequency']), env.controlled_vehicles[0], lane_change=lane_change,
                    acceleration=u
                    #target_speed=u
                )
                next_ego_polygon = get_polygon(obs[0]['length'], obs[0]['width'], heading)
                cbf_constraint_list = []
                for o, other_polygon, current_h in zip(obs[1:], other_polygon_list, current_h_list):
                    next_h = signed_distance(Minkowski_sum(next_ego_polygon, other_polygon), np.array([x - (o['x'] + dt * o['vx']), y - (o['y'] + dt * o['vy'])])) - d
                    cbf_constraint_list.append(next_h >= (1-alpha) * current_h)
                return np.all(cbf_constraint_list), u, (ref_speed - speed)**2 + 1e-6*(u-uref)**2
            with Pool(10) as p:
                res = p.map(target, np.linspace(-5, 5, 29))
            feasible, u_list, cost_list = zip(*res)
            feasible = np.array(feasible)
            #if cost_list:
            if np.any(feasible):
                cost_list = np.array(cost_list)
                min_cost = np.min(cost_list)
                w = np.exp(-(cost_list - min_cost)/lam) * feasible
                u = np.sum(w * np.array(u_list))/ np.sum(w)
                x, y, heading, speed = predict_state(
                    dt, sim_steps, env.controlled_vehicles[0], lane_change=lane_change,
                    acceleration=u
                    #target_speed=u
                )
                if cost > (ref_speed - speed)**2 + 1e-6*(u-uref)**2:
                    cost = (ref_speed - speed)**2 + 1e-6*(u-uref)**2
                    action[0]=lane_change_dict[lane_change]
                    action[1] = u / 5.0
                print(lane_change, speed, cost)
        lane_change_hist.append(action[0])
        if len(lane_change_hist) > int(lane_change_frequency * config['policy_frequency']):
            lane_change_hist.pop(0)

        obs, reward, terminated, truncated, info = env.step(action)
        print(action, obs[0, 6])
        if terminated:
            break
        data_buffer.append([k/config['policy_frequency'], obs[0, 6], action[1]*5.0])
        if env.render_mode=='rgb_array':
            img_buffer.append(env.render())

    if env.render_mode=='rgb_array':
        save_video('cbf_test', img_buffer, extension='gif')

    import matplotlib.pyplot as plt

    data_buffer = np.array(data_buffer)
    plt.figure()
    plt.subplot(211)
    plt.plot(data_buffer[:, 0], ref_speed*np.ones(len(data_buffer)), 'k:')
    plt.plot(data_buffer[:, 0], data_buffer[:, 1], 'k')
    plt.ylim([0.0, 40.0])
    plt.ylabel('v')
    plt.subplot(212)
    plt.plot(data_buffer[:, 0], -5.0*np.ones(len(data_buffer)), 'k:')
    plt.plot(data_buffer[:, 0],  5.0*np.ones(len(data_buffer)), 'k:')
    plt.plot(data_buffer[:, 0], data_buffer[:, 2], 'k')
    plt.ylabel('u')
    plt.xlabel('time [s]')

    plt.show()
