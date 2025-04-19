import numpy as np
from scipy.stats import norm
from autograd import jacobian
from qpsolvers import solve_qp
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.utils import lmap
from .utils import get_polygon, Minkowski_sum, signed_distance, rotation

lane_change_dict = {"LANE_LEFT": 0, "IDLE": 1, "LANE_RIGHT": 2}



class CBFQP:

    def __init__(self, vehicle, dt, ref_speed, safe_dist=0.1, alpha=lambda x:0.5*x, mu=np.array([-0.5, 0.0]), Sigma=np.diag([1.5**2, 0.02**2]), confidence=0.98, slack_panelty_max=1e6):
        self.vehicle = vehicle
        self.dt = dt
        self.ref_speed = ref_speed
        self.safe_dist = safe_dist
        self.alpha = alpha
        self.mu = mu
        self.Sigma = Sigma
        self.quantile = norm.ppf(confidence)
        self.slack_panelty_max = slack_panelty_max


    def solve(self, obs, *params):
        
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
        if len(params) < 1:
            params = ((lane_change, 1.0, None),)
        for lane_change, K, slack_penalties in params:
            uref = np.clip(
                current_speed + np.clip(K*(self.ref_speed-current_speed)/self.dt, *self.vehicle.ACCELERATION_RANGE) * self.dt,
                self.vehicle.MIN_SPEED, self.vehicle.MAX_SPEED
            )
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
            ns = len(h_list)
            A = -np.eye(ns, ns + 1, k=1)
            b = np.zeros(ns)
            for i, (o, h, dhdx) in enumerate(zip(obs[1:], h_list, dhdx_list)):
                dhdo = -dhdx
                fo = np.array([o['vx'], o['vy'], 0.0, 0.0])
                go = np.zeros((4, 2))
                go[:2, :] = rotation(o['heading'])
                A[i, 0] = -dhdx @ gx
                b[i] = dhdx @ fx + dhdo @ fo + dhdo @ go @ self.mu + self.alpha(h) - self.quantile * np.sqrt((dhdo@go) @ self.Sigma @ (dhdo@go))
            sol = solve_qp(
                P = 0.5 * np.diag([1.0] + slack_penalties) if isinstance(slack_penalties, list) else 0.5 * np.diag([1.0] + [self.slack_panelty_max] * ns),
                q = np.array([-uref] + slack_penalties) if isinstance(slack_penalties, list) else np.array([-uref] + [self.slack_panelty_max] * ns),
                G=A, h=b,
                A=None, b=None,
                lb=np.array([self.vehicle.ACCELERATION_RANGE[0]*self.dt] + [0.0] * ns), ub=np.array([self.vehicle.ACCELERATION_RANGE[1]*self.dt] + [np.inf] * ns),
                solver='quadprog'
            )
            if sol is not None:
                u = sol[0]
                slack_variables = sol[1:]
                current_cost = 0.5 * (u - uref) ** 2
                if current_cost < cost:
                    cost = current_cost
                    action[0] = lane_change_dict[lane_change]
                    action[1] = lmap(current_speed + u, (self.vehicle.MIN_SPEED, self.vehicle.MAX_SPEED), (-1.0, 1.0))

        return action