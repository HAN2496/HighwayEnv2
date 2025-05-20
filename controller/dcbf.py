from multiprocess.pool import Pool
import numpy as np
from scipy.stats import norm
from scipy.linalg import expm
from scipy.optimize import minimize, minimize_scalar
from autograd import jacobian
from qpsolvers import solve_qp
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.utils import lmap
from .utils import get_polygon, Minkowski_sum, signed_distance, predict_state, rotation

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



class DCBFQP:

    def __init__(self, vehicle, dt, ref_speed, safe_dist=0.2, alpha=lambda x:0.8*x, mu=np.array([-2.5, 0.0]), Sigma=np.diag([7.5**2, 0.1**2]), confidence=0.98, slack_panelty_max=1e6):
        self.vehicle = vehicle
        self.dt = dt
        self.ref_speed = ref_speed
        self.safe_dist = safe_dist
        self.alpha = alpha
        self.mu = mu
        self.Sigma = Sigma
        self.quantile = norm.ppf(confidence)
        self.slack_panelty_max = slack_panelty_max


    def solve(self, obs, *params, return_QP=False):
        
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
        nh = len(h_list)

        cost = np.inf
        action = [1, lmap(np.clip(-current_speed/self.dt, *self.vehicle.ACCELERATION_RANGE), self.vehicle.ACCELERATION_RANGE, (-1.0, 1.0))]
        if len(params) < 1:
            params = ((lane_change, 1.0, [self.slack_panelty_max] * nh, [1.0] * nh),)
        for lane_change, K, slack_penalties, alpha_scales in params:
            v = ControlledVehicle.create_from(self.vehicle)
            uprev = v.action['acceleration']
            #umin = max(uprev-1.5, self.vehicle.ACCELERATION_RANGE[0])
            #umax = min(uprev+0.5, self.vehicle.ACCELERATION_RANGE[1])
            umin = self.vehicle.ACCELERATION_RANGE[0]
            umax = self.vehicle.ACCELERATION_RANGE[1]
            v.act(lane_change)
            uref = np.clip(K*(self.ref_speed-current_speed)/self.dt, umin, umax)
            uref += 0.8*(uprev - uref)
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

            sysx = np.zeros((6, 6))
            sysx[:4, :4] = dfxdx
            sysx[:4, 4] = gx
            sysx[:4, 5] = fx
            sysxd = expm(sysx * self.dt)
            gxd = sysxd[:4, 4]
            fxd = sysxd[:4, 5]

            A = -np.eye(nh, nh+1, k=1)
            b = np.zeros(nh)
            for i, (o, h, dhdx) in enumerate(zip(obs[1:], h_list, dhdx_list)):
                fo = np.array([o['vx'], o['vy'], 0.0, 0.0])
                dfodo = np.eye(4, k=2)
                go = np.zeros((4, 2))
                go[2:, :] = rotation(o['heading'])
                syso = np.zeros((7, 7))
                syso[:4, :4] = dfodo
                syso[:4, 4:6] = go
                syso[:4, 6] = fo
                sysod = expm(syso * self.dt)
                god = sysod[:4, 4:6]
                fod = sysod[:4, 6]
                Lfxh = dhdx @ fxd
                Lgxh = dhdx @ gxd
                Lfoh = -dhdx @ fod
                Lgoh = -dhdx @ god
                A[i, 0] = -Lgxh
                b[i] = Lfxh + Lfoh + Lgoh @ self.mu + alpha_scales[i] * self.alpha(h) - self.quantile * np.sqrt(Lgoh @ self.Sigma @ Lgoh)
            P = 0.5 * np.diag([1.0] + slack_penalties)**2
            #q = np.array([-uref] + slack_penalties)
            q = np.array([-uref] + [0.0] * nh)
            lb=np.array([umin] + [0.0] * nh)
            ub=np.array([umax] + [np.inf] * nh)
            sol = solve_qp(
                P=P, q=q, G=A, h=b,
                A=None, b=None,
                lb=lb, ub=ub,
                solver='quadprog'
            )
            if sol is not None:
                u = sol[0]
                slack_variables = sol[1:]
                #print(slack_variables)
                current_cost = 0.5 * (u - uref) ** 2
                if current_cost < cost:
                    cost = current_cost
                    action[0] = lane_change_dict[lane_change]
                    action[1] = lmap(np.clip(u, *self.vehicle.ACCELERATION_RANGE), self.vehicle.ACCELERATION_RANGE, (-1.0, 1.0))
            #else:
            #    print("Infeasible")

        if return_QP:
            return action, {"P": P, "q": q, "G":A, "h": b, "lb": lb, "ub": ub}
        else:
            return action

    

class DCBFNLP:

    def __init__(self, vehicle, dt, ref_speed, sim_steps=5, safe_dist=0.2, alpha=lambda x:0.8*x, mu=np.array([-2.5, 0.0]), Sigma=np.diag([7.5**2, 0.1**2]), confidence=0.98, slack_panelty_max=1e6):
        self.vehicle = vehicle
        self.dt = dt
        self.ref_speed = ref_speed
        self.sim_steps = sim_steps
        self.safe_dist = safe_dist
        self.alpha = alpha
        self.mu = mu
        self.Sigma = Sigma
        self.quantile = norm.ppf(confidence)
        self.slack_panelty_max = slack_panelty_max


    def solve(self, obs, *params):
        
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
        nh = len(current_h_list)
        umin = max(self.vehicle.action['acceleration']-1.5, self.vehicle.ACCELERATION_RANGE[0])
        umax = min(self.vehicle.action['acceleration']+0.5, self.vehicle.ACCELERATION_RANGE[1])

        cost = np.inf
        action = [1, lmap(np.clip(-current_speed/self.dt, *self.vehicle.ACCELERATION_RANGE), self.vehicle.ACCELERATION_RANGE, (-1.0, 1.0))]
        if len(params) < 1:
            params = ((lane_change, 1.0, [self.slack_panelty_max] * nh, [1.0] * nh),)
        for lane_change, K, slack_penalties, alpha_scales in params:
            uref = np.clip(K*(self.ref_speed-current_speed)/self.dt, umin, umax)
            def target(u):
                x, y, heading, speed = predict_state(
                    self.dt, self.sim_steps, self.vehicle, lane_change=lane_change,
                    acceleration=u
                    #target_speed=u
                )
                next_ego_polygon = get_polygon(obs[0]['length'], obs[0]['width'], heading)
                cost = 0.5 * (u - uref) ** 2
                for o, other_polygon, current_h, slack_penalty, alpha_scale in zip(obs[1:], other_polygon_list, current_h_list, slack_penalties, alpha_scales):
                    mudx, mudy = rotation(o['heading']) @ self.mu
                    confidence_interval = get_polygon(*(self.quantile*0.5*np.sqrt(np.diag(self.Sigma))*self.dt**2), o['heading'])
                    robust_other_polygon = Minkowski_sum(other_polygon, confidence_interval)
                    next_h = signed_distance(Minkowski_sum(next_ego_polygon, robust_other_polygon), np.array([x - (o['x'] + self.dt * (o['vx']+mudx)), y - (o['y'] + self.dt * (o['vy']+mudy))])) - self.safe_dist
                    cost += 0.5 * slack_penalty * min(0.0, next_h - current_h  + alpha_scale*self.alpha(current_h)) ** 2
                return cost
            sol = minimize_scalar(fun=target, bounds=(umin, umax), method='bounded')
            #sol = minimize(fun=target, x0=uref, method='L-BFGS-B', bounds=[(umin, umax),])
            if sol is not None:
                u = sol.x
                #u = sol.x[0]
                current_cost = 0.5 * (u - uref) ** 2
                if current_cost < cost:
                    cost = current_cost
                    action[0] = lane_change_dict[lane_change]
                    action[1] = lmap(np.clip(u, *self.vehicle.ACCELERATION_RANGE), self.vehicle.ACCELERATION_RANGE, (-1.0, 1.0))

        return action