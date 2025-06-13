import numpy as np
import casadi as ca
from scipy.stats import norm
from highway_env.utils import lmap

from numpy import zeros, diag, array, radians, hstack, clip, ones, hypot
from casadi import SX, vertcat, Function, atan, fabs, jacobian, hessian
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel

from .utils import get_polygon, Minkowski_sum, signed_distance_cbf

"""
HOCBF-MPC + h_parallel(x, z)/h_perpendicular(x, z) 꼴
"""
class HOCBFMPC:
    def __init__(self, vehicle, dt, ref_speed, ref_lane, init_state,
                 horizon=10, safe_dist=0.1, k1=0.06, k2=0.24, W_cb=1e2,
                 mu=np.array([0.0, 0.0]), Sigma=np.diag([1.0, 1.0]), confidence=0.98, stoch=True):
        self.vehicle = vehicle
        self.dt = dt
        self.ref_speed = ref_speed
        self.ref_lane = ref_lane
        self.init_state = init_state
        self.N = horizon
        self.safe_dist = safe_dist
        self.k1 = k1
        self.k2 = k2
        self.W_cb = W_cb

        self.mu = mu
        self.Sigma = Sigma
        self.quantile = norm.ppf(confidence)

        self.stoch = stoch

        self.state_n = 4
        self.control_n = 2

        self.Length = vehicle.LENGTH
        self.Width = vehicle.WIDTH

        self.max_vel = 40.0
        self.max_a = 5.0
        self.max_steering_angle = radians(5)

        self.max_difflaneobs_n = 3
        self.max_samelaneobs_n = 3
        self.total_obs = self.max_samelaneobs_n + self.max_difflaneobs_n


        self.ocp = self._build_ocp()
        self.solver = AcadosOcpSolver(self.ocp, json_file='acados_ocp.json')
        self.trajectory = zeros((self.N+1, self.state_n))

    def _make_cbf(self, ego_state_sym, obs_pos_sym):
        # ---- signed distance ----
        h = signed_distance_cbf(
                ego_state_sym,
                vertcat(obs_pos_sym[0], obs_pos_sym[1], obs_pos_sym[2]), # psi 0으로 설정 (추후 수정해야함)
                (self.Length, self.Width),                  # ego length, width
                (self.Length, self.Width)                   # obs length, width
            )
        # ---- Lie derivatives ----
        Lf_h    = jacobian(h, self.ocp.model.x) @ self.ocp.model.f_expl_expr
        Lf2_h = jacobian(Lf_h, self.ocp.model.x) @ self.ocp.model.f_expl_expr
        LgLf_h = jacobian(Lf_h, self.ocp.model.u)       # 1×2
        grad_z_h  = jacobian(h, obs_pos_sym)              # 1×2
        return h, Lf_h  , Lf2_h, LgLf_h, grad_z_h

    def _build_ocp(self):
        ocp = AcadosOcp()
        ocp.model.name = "KinematicTrackingMPC"

        # set up states & controls
        x = SX.sym('x')
        y = SX.sym('y')
        v = SX.sym('v')
        psi = SX.sym('psi')

        state = ca.vertcat(x, y, v, psi)
        ocp.model.x = state
        nx = ocp.model.x.size1()

        # controls
        a = SX.sym('a')
        delta = SX.sym('delta')
        control = ca.vertcat(a, delta)
        ocp.model.u = control
        nu = ocp.model.u.size1()

        # Params
        xref = SX.sym('xref')
        yref = SX.sym('yref')
        vref = SX.sym('vref')
        psiref = SX.sym('psiref')
        ref = ca.vertcat(xref, yref, vref, psiref)
        s_obstacles = []
        d_obstacles = []
        for i in range(self.max_samelaneobs_n):
            s_obs_x = SX.sym(f's_obs_x_{i}')
            s_obs_y = SX.sym(f's_obs_y_{i}')
            s_obs_vx = SX.sym(f'obs_vx_{i}')
            s_obs_vy = SX.sym(f'obs_vy_{i}')
            s_obs_psi = SX.sym(f'obs_psi_{i}')
            s_obs_flag = SX.sym(f'obs_flag_{i}')
            s_obstacles.append(ca.vertcat(s_obs_x, s_obs_y))        
        for i in range(self.max_difflaneobs_n):
            d_obs_x = SX.sym(f'd_obs_x_{i}')
            d_obs_y = SX.sym(f'd_obs_y_{i}')
            d_obs_vx = SX.sym(f'obs_vx_{i}')
            d_obs_vy = SX.sym(f'obs_vy_{i}')
            d_obs_psi = SX.sym(f'obs_psi_{i}')
            d_obs_flag = SX.sym(f'obs_flag_{i}')
            d_obstacles.append(ca.vertcat(d_obs_x, d_obs_y))
    
        ocp.model.p = ca.vertcat(ref, *s_obstacles, *d_obstacles)
        np = ocp.model.p.size1()
        ocp.parameter_values = zeros(np)

        # algebraic variables
        z = ca.vertcat([])
        ocp.model.z = z

        # Dynamics
        beta = atan(0.5 * SX.tan(delta))
        dx = v * SX.cos(psi + beta)
        dy = v * SX.sin(psi + beta)
        dv = a
        dpsi = v * SX.sin(beta) / (self.Length/2)
        f_expl = vertcat(dx, dy, dv, dpsi) 
        ocp.model.f_expl_expr = f_expl

        # Costs 
        ocp.cost.cost_type   = "EXTERNAL"
        ocp.cost.cost_type_e = "EXTERNAL" 
        Q = diag([5e-1, 30, 1, 1e-3])
        R = diag([6, 25])

        ocp.model.cost_expr_ext_cost = \
            Q[0,0]*(xref-x)**2 + \
            Q[1,1]*(yref-y)**2 + \
            Q[2,2]*(vref-v)**2 + \
            Q[3,3]*(psiref-psi)**2 + \
            R[0,0]*a*a + \
            R[1,1]*delta*delta
        
        ocp.model.cost_expr_ext_cost_e = \
            Q[0,0]*(xref-x)**2 + \
            Q[1,1]*(yref-y)**2 + \
            Q[2,2]*(vref-v)**2 + \
            Q[3,3]*(psiref-psi)**2
        
        # Initial State
        ocp.constraints.x0 = self.init_state

        # Constraints
        ocp.constraints.lbx = array([0.0])
        ocp.constraints.ubx = array([self.max_vel])
        ocp.constraints.idxbx = array([2])

        ocp.constraints.lbu = array([-self.max_a, -self.max_steering_angle])
        ocp.constraints.ubu = array([ self.max_a,   self.max_steering_angle])
        ocp.constraints.idxbu = array([0, 1])

        # === HOCBF soft‐constraint ===
        x, y, v, psi = ocp.model.x[0], ocp.model.x[1], ocp.model.x[2], ocp.model.x[3]
        a, delta = ocp.model.u[0], ocp.model.u[1]

        obs_syms   = []
        lane_flags = []
        idx        = 4
        for _ in range(self.total_obs):
            obs_syms.append(p[idx:idx+2]); idx += 2
            lane_flags.append(p[idx]);     idx += 1

        cbf_exprs = []
        for obs, flag in zip(obs_syms, lane_flags):

            h, Lfh, Lf2h, LgLfh, dz_h = self._make_cbf(ocp.model.x, obs)

            # Robust / stochastic term
            if self.stoch:
                robust = (dz_h @ self.mu
                        - self.quantile * ca.sqrt(dz_h @ self.Sigma @ dz_h.T))
            else:
                robust = 0.0

            ψ2 = (Lf2h
                + self.k2 * (Lfh + self.k1 * h)
                + LgLfh[0]*a + LgLfh[1]*delta
                + robust)

            cbf_exprs.append(-ψ2)          # ≥0 제한 → −ψ2 ≤ 0

        ocp.model.con_h_expr = ca.vertcat(*cbf_exprs)
        nh = ocp.model.con_h_expr.size1()

        # soft constraint 및 slack setting
        ocp.constraints.lh = zeros(nh)
        ocp.constraints.uh = 1e6 * ones(nh)
        ocp.constraints.lsh = zeros(nh)
        ocp.constraints.ush = 1e6 * ones(nh)
        ocp.constraints.idxsh = array(range(nh))

        # slack penalty
        ocp.cost.zl = self.W_cb * ones(nh)
        ocp.cost.zu = self.W_cb * ones(nh)
        ocp.cost.Zl = self.W_cb * ones(nh)
        ocp.cost.Zu = self.W_cb * ones(nh)
        # === HOCBF soft‐constraint 끝~ ===

        ocp.solver_options.tf = self.N * self.dt
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.qp_solver_warm_start = 1
        ocp.solver_options.hpipm_mode = 'BALANCE'
        ocp.solver_options.sim_method_num_stages = 4
        ocp.solver_options.sim_method_num_steps = 3
        ocp.solver_options.nlp_solver_max_iter = 100
        ocp.solver_options.nlp_solver_step_length = 1.0
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.dims.N = self.N
        ocp.solver_options.tf = self.N * self.dt

        return ocp

    def solve(self, x0, ref_traj, s_obs_traj, d_obs_traj):
        self.solver.set(0, "lbx", x0)
        self.solver.set(0, "ubx", x0)
        # if first solve
        if self.solver.get(0, "u") is None:
            for k in range(self.N):
                self.solver.set(k,'x',x0)

        s_padded_obs = zeros((self.max_samelaneobs_n, self.N, self.state_n))
        s_padded_obs[:s_obs_traj.shape[0],:,:] = s_obs_traj

        d_padded_obs = zeros((self.max_difflaneobs_n, self.N, self.state_n))
        d_padded_obs[:d_obs_traj.shape[0],:,:] = d_obs_traj

        # for k in range(self.N+1):
        #     self.solver.set(k, "p", hstack([*ref_traj[k],*s_padded_obs[:,k-1,:2],*d_padded_obs[:,k-1,:2]]))

        for k in range(self.N+1):
            idx = min(k, self.N-1)
            param_k = list(ref_traj[k])
            for i in range(self.max_samelaneobs_n):
                x_i, y_i, psi = s_padded_obs[i, idx, 0, 1, 4]
                param_k += [x_i, y_i, psi]
            for i in range(self.max_difflaneobs_n):
                x_i, y_i = d_padded_obs[i, idx, 0, 1, 4]
                param_k += [x_i, y_i, 1.0]
            self.solver.set(k, "p", np.array(param_k))

        status = self.solver.solve()
        if status != 0:
            print("Acados failed with status", status)
            return [0.0, 0.0]

        u0 = self.solver.get(0, "u")

        for k in range(self.N+1):
            self.trajectory[k, :] = self.solver.get(k, "x")

        return u0

    def run(self, obs):
        ego_x = obs[0]['x']
        ego_y = obs[0]['y']

        lane = self.vehicle.road.network.get_lane(self.ref_lane)
        local_s = lane.local_coordinates([ego_x, ego_y])[0]

        # reference trajectory 생성
        future_ref = zeros((self.N+1, self.state_n))
        delta_s = self.ref_speed * self.dt
        for i in range(self.N+1):
            s_i = local_s + delta_s * (i)
            x_ref, y_ref = lane.position(s_i, 0.0)
            heading_i = lane.heading_at(s_i)
            future_ref[i, 0] = x_ref
            future_ref[i, 1] = y_ref
            future_ref[i, 2] = self.ref_speed
            future_ref[i, 3] = heading_i

        # TEST: WITH Surrounding vehicles
        closest_obs = None
        min_dist = float('inf')
        pos = [0, 0]

        samelane_obs = []
        difflane_obs = []
        for o in obs[1:]:
            dx = o['x'] - ego_x
            dy = o['y'] - ego_y
            # dx = ego_x - o['x'] - o['length'] / 2.0
            # dy = ego_y - o['y'] - o['width'] / 2.0
            dist = hypot(dx, dy)
            if dist < min_dist:
                min_dist = dist
                closest_obs = o
                pos[0], pos[1] = ego_x - o['x'], ego_y - o['y']
            if dist <= 50.0:
                if o['lane_index'] == self.vehicle.lane_index[-1]:
                    samelane_obs.append(o)
                else:
                    difflane_obs.append(o)
        
        # 가장 가까운 obstacle이 same lane인지 아닌지 프린트
        if closest_obs is not None:
            if closest_obs['lane_index'] == self.vehicle.lane_index[-1]:
                print("Closest obstacle is in the SAME lane.", min_dist - self.vehicle.LENGTH / 2.0, pos)
            else:
                print("Closest obstacle is in a DIFFERENT lane.", min_dist - self.vehicle.WIDTH / 2.0, pos)
        else:
            print("No obstacles detected.")        

        samelane_obs = samelane_obs[:self.max_samelaneobs_n]
        difflane_obs = difflane_obs[:self.max_difflaneobs_n]

        s_obs_trajectories = zeros((len(samelane_obs), self.N, self.state_n))
        d_obs_trajectories = zeros((len(difflane_obs), self.N, self.state_n))

        # 예측 trajectory 생성
        for i, o in enumerate(samelane_obs):
            s_obs_trajectories[i] = self.predict_obstacle(o)

        for i, o in enumerate(difflane_obs):
            d_obs_trajectories[i] = self.predict_obstacle(o)


        x0 = array([obs[0]['x'], obs[0]['y'], obs[0]['vx'], obs[0]['heading']])

        result = self.solve(x0, future_ref, s_obs_trajectories, d_obs_trajectories)

        a = result[0]
        delta = result[1]
        print(f"Action: a={a}, delta={delta}")

        action = [
            lmap(np.clip(a, *self.vehicle.ACCELERATION_RANGE), self.vehicle.ACCELERATION_RANGE, (-1.0, 1.0)),
            # clip(a, *self.vehicle.ACCELERATION_RANGE),
            lmap(np.clip(delta, -self.max_steering_angle, self.max_steering_angle), (-self.max_steering_angle, self.max_steering_angle), (-1.0, 1.0)),
            clip(delta, -self.max_steering_angle, self.max_steering_angle)
        ]
        self.vehicle.action['steering'] = clip(delta, -self.max_steering_angle, self.max_steering_angle)
        return action
    

    def predict_obstacle(self, obs_state):
        traj = np.zeros((self.N, 4))
        x_k, y_k, vx_k, vy_k, psi = obs_state['x'], obs_state['y'], obs_state['vx'], obs_state['vy'], obs_state['heading']
        for k in range(self.N):
            vx_k += self.mu[0] * self.dt
            vy_k += self.mu[1] * self.dt
            x_k += vx_k * self.dt
            y_k += vy_k * self.dt
            traj[k] = [x_k, y_k, vx_k, vy_k, psi]
        return traj