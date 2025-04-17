import numpy as np
from scipy.stats import norm
from scipy.linalg import block_diag
from autograd import jacobian, hessian
from qpsolvers import solve_qp
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.utils import lmap
from .utils import get_polygon, Minkowski_sum, signed_distance, rotation

lane_change_dict = {"LANE_LEFT": 0, "IDLE": 1, "LANE_RIGHT": 2}



class HOCBFQP:

    def __init__(self, vehicle, dt, ref_speed, safe_dist=0.1, alpha1=lambda x:0.6*x, alpha2=lambda x:2.4*x, mu=np.array([-2.5, 0.0]), Sigma=np.diag([7.5**2, 0.1**2]), confidence=0.98, slack_panelty=1e6):
        self.vehicle = vehicle
        self.dt = dt
        self.ref_speed = ref_speed
        self.safe_dist = safe_dist
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.mu = mu
        self.Sigma = Sigma
        self.quantile = norm.ppf(confidence)
        self.slack_panelty = slack_panelty


    def solve(self, obs, possible_lane_changes=None):
        
        current_speed = obs[0]['speed']

        ego_polygon = get_polygon(obs[0]['length'], obs[0]['width'], obs[0]['heading'])
        other_polygon_list = [get_polygon(o['length'], o['width'], o['heading']) for o in obs[1:]]
        h_list = []
        dhdx_list = []
        d2hdx2_list = []
        for o, other_polygon in zip(obs[1:], other_polygon_list):
            h = lambda x: signed_distance(
                Minkowski_sum(ego_polygon, other_polygon),
                x[:2] + np.array([obs[0]['x'] - o['x'], obs[0]['y'] - o['y']])
            )
            h_list.append(h(np.zeros(4)))
            dhdx_list.append(jacobian(h)(np.zeros(4)))
            d2hdx2_list.append(hessian(h)(np.zeros(4)))

        cost = np.inf
        uref = np.clip((self.ref_speed - current_speed)/self.dt, *self.vehicle.ACCELERATION_RANGE)
        action = [1, lmap(np.clip(-current_speed/self.dt, *self.vehicle.ACCELERATION_RANGE), self.vehicle.ACCELERATION_RANGE, (-1.0, 1.0))]
        if possible_lane_changes is None:
            possible_lane_changes = ['IDLE']
        for lane_change in possible_lane_changes:
            v = ControlledVehicle.create_from(self.vehicle)
            v.act(lane_change)
            beta = np.arctan(1 / 2 * np.tan(v.action['steering']))
            tire_slip_angle = np.abs(v.action['steering'] - beta)
            dfxds = np.array(
                [
                    np.cos(v.heading + beta),
                    np.sin(v.heading + beta),
                    np.sin(beta) / (v.LENGTH / 2),
                    -np.sin(tire_slip_angle) * v.LATERAL_TIRE_COEF * tire_slip_angle
                ]
            )
            fx = dfxds * v.speed
            dfxdx = np.zeros((4, 4))
            dfxdx[:, 2] = np.array([-fx[1], fx[0], 0.0, 0.0])
            dfxdx[:, 3] = dfxds
            gx = np.array([0.0, 0.0, 0.0, np.cos(beta)])
            ns = len(h_list)
            A = -np.eye(ns, ns+1, k=1)
            b = np.zeros(ns)
            for i, (o, h, dhdx, d2hdx2) in enumerate(zip(obs[1:], h_list, dhdx_list, d2hdx2_list)):
                fz = np.hstack([fx, np.array([o['vx'], o['vy'], 0.0, 0.0])])
                dfdz = block_diag(dfxdx, np.eye(4, k=2))
                go = np.zeros((4, 2))
                go[2:, :] = rotation(o['heading'])
                gz = block_diag(gx[:, np.newaxis], go)
                dhdz = np.hstack([dhdx, -dhdx])
                d2hdz2 = np.block([[d2hdx2, -d2hdx2], [-d2hdx2, d2hdx2]])
                dLfhdz = dhdz @ dfdz + d2hdz2 @ fz
                L2fh = dLfhdz @ fz
                LgLfh = dLfhdz @ gz
                A[i, 0] = -LgLfh[0]
                b[i] = L2fh + self.alpha2(dhdz @ fz + self.alpha1(h)) + LgLfh[1:] @ self.mu - self.quantile * np.sqrt(LgLfh[1:] @ self.Sigma @ LgLfh[1:])
            sol = solve_qp(
                P=np.diag([0.5] + [self.slack_panelty] * ns), q=np.array([-uref] + [self.slack_panelty] * ns),
                G=A, h=b,
                A=None, b=None,
                lb=np.array([self.vehicle.ACCELERATION_RANGE[0]] + [0.0] * ns), ub=np.array([self.vehicle.ACCELERATION_RANGE[1]] + [np.inf] * ns),
                #lb=None, ub=None
                solver='quadprog'
            )
            if sol is not None:
                u = sol[0]
                slack_variables = sol[1:]
                current_cost = 0.5 * (u - uref) ** 2
                if current_cost < cost:
                    cost = current_cost
                    action[0] = lane_change_dict[lane_change]
                    action[1] = lmap(np.clip(u, *self.vehicle.ACCELERATION_RANGE), self.vehicle.ACCELERATION_RANGE, (-1.0, 1.0))

        return action