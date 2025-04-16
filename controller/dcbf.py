from multiprocess.pool import Pool
import numpy as np
from scipy.stats import norm
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.utils import lmap
from .utils import get_polygon, Minkowski_sum, signed_distance, check_possible_lane_changes, predict_state

lane_change_dict = {"LANE_LEFT": 0, "IDLE": 1, "LANE_RIGHT": 2}



class DCBFIS:

    def __init__(self, vehicle, dt, ref_speed, sim_steps=5, safe_dist=0.2, alpha=0.9, Sigma=np.diag([3.0**2, 0.1**2]), confidence=0.98, lam=1e2, n_sample=27):
        self.vehicle = vehicle
        self.dt = dt
        self.ref_speed = ref_speed
        self.sim_steps = sim_steps
        self.safe_dist = safe_dist
        self.alpha = alpha
        self.Sigma = Sigma
        self.quantile = norm.ppf(confidence)
        self.lam = lam
        self.n_sample = n_sample


    def solve(self, obs, possible_lane_changes=None):
        
        current_speed = obs[0]['speed']

        current_ego_polygon = get_polygon(obs[0]['length'], obs[0]['width'], obs[0]['heading'])
        other_polygon_list = [get_polygon(o['length'], o['width'], o['heading']) for o in obs[1:]]
        current_h_list = [
            signed_distance(
                Minkowski_sum(
                    current_ego_polygon, other_polygon
                ), np.array([obs[0]['x'] - o['x'], obs[0]['y'] - o['y']]) 
            ) - self.safe_dist for o, other_polygon in zip(obs[1:], other_polygon_list)
        ]

        cost = np.inf
        uref = np.clip(-current_speed/self.dt, *self.vehicle.ACCELERATION_RANGE)
        action = [1, lmap(uref, self.vehicle.ACCELERATION_RANGE, (-1.0, 1.0))]
        if possible_lane_changes is None:
            possible_lane_changes = ['IDLE']
        for lane_change in possible_lane_changes:
            def target(u):
                x, y, heading, speed = predict_state(
                    self.dt, self.sim_steps, self.vehicle, lane_change=lane_change,
                    acceleration=u
                    #target_speed=u
                )
                next_ego_polygon = get_polygon(obs[0]['length'], obs[0]['width'], heading)
                cbf_constraint_list = []
                for o, other_polygon, current_h in zip(obs[1:], other_polygon_list, current_h_list):
                    #next_h = signed_distance(Minkowski_sum(next_ego_polygon, other_polygon), np.array([x - (o['x'] + dt * o['vx']), y - (o['y'] + dt * o['vy'])])) - d
                    confidence_interval = get_polygon(*(self.quantile*np.sqrt(np.diag(self.Sigma)*self.dt)), np.arctan2(o['vy'], o['vx']))
                    robust_other_polygon = Minkowski_sum(other_polygon, confidence_interval)
                    next_h = signed_distance(Minkowski_sum(next_ego_polygon, robust_other_polygon), np.array([x - (o['x'] + self.dt * o['vx']), y - (o['y'] + self.dt * o['vy'])])) - self.safe_dist
                    cbf_constraint_list.append(next_h >= (1-self.alpha) * current_h)
                return np.all(cbf_constraint_list), u, (self.ref_speed - speed)**2 + 1e-6*(u-uref)**2
            with Pool(4) as pool:
                res = pool.map(target, np.linspace(*self.vehicle.ACCELERATION_RANGE, self.n_sample))
            feasible, u_list, cost_list = zip(*res)
            feasible = np.array(feasible)
            if np.any(feasible):
                cost_list = np.array(cost_list)
                min_cost = np.min(cost_list)
                w = np.exp(-(cost_list - min_cost)/self.lam) * feasible
                u = np.sum(w * np.array(u_list))/ np.sum(w)
                x, y, heading, speed = predict_state(
                    self.dt, self.sim_steps, self.vehicle, lane_change=lane_change,
                    acceleration=u
                    #target_speed=u
                )
                current_cost = (self.ref_speed - speed)**2 + 1e-6*(u-uref)**2
                if current_cost < cost:
                    cost = current_cost
                    action[0] = lane_change_dict[lane_change]
                    action[1] = lmap(np.clip(u, *self.vehicle.ACCELERATION_RANGE), self.vehicle.ACCELERATION_RANGE, (-1.0, 1.0))

        return action