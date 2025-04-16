import numpy as np
from scipy.stats import norm
from autograd import jacobian
from qpsolvers import solve_problem, Problem
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.utils import lmap
from .utils import get_polygon, Minkowski_sum, signed_distance, rotation

lane_change_dict = {"LANE_LEFT": 0, "IDLE": 1, "LANE_RIGHT": 2}



class CBFQP:

    def __init__(self, vehicle, dt, ref_speed, safe_dist=0.2, alpha=0.5, Sigma=np.diag([3.0**2, 0.1**2]), confidence=0.98):
        self.vehicle = vehicle
        self.dt = dt
        self.ref_speed = ref_speed
        self.safe_dist = safe_dist
        self.alpha = alpha
        self.Sigma = Sigma
        self.quantile = norm.ppf(confidence)


    def solve(self, obs, possible_lane_changes=None):
        
        current_speed = obs[0]['speed']

        ego_polygon = get_polygon(obs[0]['length'], obs[0]['width'], obs[0]['heading'])
        other_polygon_list = [get_polygon(o['length'], o['width'], o['heading']) for o in obs[1:]]
        h_list = []
        dhdx_list = []
        for o, other_polygon in zip(obs[1:], other_polygon_list):
            h = lambda x: signed_distance(
                Minkowski_sum(ego_polygon, other_polygon),
                x[:2] + np.array([obs[0]['x'] - o['x'], obs[0]['y'] - o['y']])
            )
            h_list.append(h(np.zeros(4)))
            dhdx_list.append(jacobian(h)(np.zeros(4)))

        cost = np.inf
        uref = np.clip(
            current_speed + np.clip((self.ref_speed - current_speed)/self.dt, *self.vehicle.ACCELERATION_RANGE) * self.dt,
            self.vehicle.MIN_SPEED, self.vehicle.MAX_SPEED
        )
        action = [1, 0.0]
        if possible_lane_changes is None:
            possible_lane_changes = ['IDLE']
        for lane_change in possible_lane_changes:
            v = ControlledVehicle.create_from(self.vehicle)
            v.act(lane_change)
            beta = np.arctan(1 / 2 * np.tan(v.action['steering']))
            tire_slip_angle = np.abs(v.action['steering'] - beta)
            gx = np.array(
                [
                    np.cos(v.heading + beta),
                    np.sin(v.heading + beta),
                    np.sin(beta) / (v.LENGTH / 2),
                    -np.sin(tire_slip_angle) * v.LATERAL_TIRE_COEF * tire_slip_angle
                ]
            )
            fx = gx * v.speed
            A = []
            b = []
            for o, h, dhdx in zip(obs[1:], h_list, dhdx_list):
                dhdo = -dhdx
                fo = np.array([o['vx'], o['vy'], 0.0, 0.0])
                go = np.zeros((4, 2))
                go[:2, :] = rotation(o['heading'])
                A.append(-dhdx @ gx)
                b.append(dhdx @ fx + dhdo @ fo + self.alpha * h - 2.0 * np.sqrt((dhdo@go) @ self.Sigma @ (dhdo@go)))
            sol = solve_problem(
                Problem(
                    P=np.ones(1), q=np.array([-2*uref]),
                    G=np.array(A)[:, np.newaxis], h=np.array(b),
                    A=None, b=None,
                    lb=np.array([self.vehicle.ACCELERATION_RANGE[0]*self.dt]), ub=np.array([self.vehicle.ACCELERATION_RANGE[1]*self.dt])
                ),
                solver='quadprog'
            )
            if sol.obj is not None:
                if sol.obj < cost:
                    cost = sol.obj
                    action[0] = lane_change_dict[lane_change]
                    action[1] = lmap(current_speed + sol.x[0], (self.vehicle.MIN_SPEED, self.vehicle.MAX_SPEED), (-1.0, 1.0))

        return action