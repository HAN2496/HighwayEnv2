from numpy import zeros, diag, array, radians, hstack, clip, ones, hypot
from casadi import SX, vertcat, Function, atan, fabs
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import casadi as ca


class MPC:
    def __init__(self, vehicle, dt, ref_speed, ref_lane, init_state, horizon=10, ds_safe=3, dn_safe = 0.1):
        self.vehicle = vehicle
        self.dt = dt
        self.ref_speed = ref_speed
        self.ref_lane = ref_lane
        self.N = horizon
        self.state_n = 4
        self.control_n = 2
        self.Length = vehicle.LENGTH
        self.Width = vehicle.WIDTH
        self.init_state = init_state
        
        self.max_vel = 40.0
        self.max_a = 5.0
        self.max_steering_angle = radians(5)
        self.ds_safe = ds_safe
        self.dn_safe = dn_safe

        self.max_difflaneobs_n = 3
        self.max_samelaneobs_n = 3


        self.ocp = self._build_ocp()
        self.solver = AcadosOcpSolver(self.ocp, json_file='acados_ocp.json')
        self.trajectory = zeros((self.N+1, self.state_n))

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
            s_obstacles.append(ca.vertcat(s_obs_x, s_obs_y))        
        for i in range(self.max_difflaneobs_n):
            d_obs_x = SX.sym(f'd_obs_x_{i}')
            d_obs_y = SX.sym(f'd_obs_y_{i}')
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
        Q = diag([5e-1, 10, 1, 1e-3])
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
        ocp.constraints.lbx =   array([ 0])
        ocp.constraints.ubx =   array([self.max_vel])
        ocp.constraints.idxbx = array([2])

        ocp.constraints.lbu = array([-self.max_a, -self.max_steering_angle])
        ocp.constraints.ubu = array([ self.max_a,   self.max_steering_angle])
        ocp.constraints.idxbu = array([0, 1])

        # Nonlinear constraints -> Need to be changed for surrounding vehicles
        s_obstacle_dist = []
        d_obstacle_dist = []
        for obs in s_obstacles:
            s_obstacle_dist.append(fabs(x-obs[0]))
        for obs in d_obstacles:
            dx = x - obs[0]
            dy = y - obs[1]
            perp_dist = fabs(dx * ca.sin(psi) - dy * ca.cos(psi))
            d_obstacle_dist.append(perp_dist)
        
        ocp.model.con_h_expr = ca.vertcat(*s_obstacle_dist, * d_obstacle_dist)

        lh_obstacles = array([self.Length+self.ds_safe] * self.max_samelaneobs_n + [self.Width+self.dn_safe] * self.max_difflaneobs_n)
        uh_obstacles = array([1e+6] * self.max_samelaneobs_n + [1e+6] * self.max_difflaneobs_n)
        ocp.constraints.lh = lh_obstacles 
        ocp.constraints.uh = uh_obstacles
        nh = ocp.model.con_h_expr.size1()

        ocp.constraints.lsh = zeros(nh)
        ocp.constraints.ush = zeros(nh)
        ocp.constraints.idxsh = array(range(nh))

        ocp.cost.zl = ocp.cost.zu = ocp.cost.Zl = ocp.cost.Zu = 1e2 * ones(nh)

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

        for k in range(self.N+1):
            self.solver.set(k, "p", hstack([*ref_traj[k],*s_padded_obs[:,k-1,:2],*d_padded_obs[:,k-1,:2]]))

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
            s_i = local_s + delta_s * (i )
            x_ref, y_ref = lane.position(s_i, 0.0)
            heading_i = lane.heading_at(s_i)
            future_ref[i, 0] = x_ref
            future_ref[i, 1] = y_ref
            future_ref[i, 2] = self.ref_speed
            future_ref[i, 3] = heading_i

        # TEST: WITH Surrounding vehicles
        samelane_obs = []
        difflane_obs = []
        for o in obs[1:]:
            dx = o['x'] - ego_x
            dy = o['y'] - ego_y
            dist = hypot(dx, dy)
            if dist <= 50.0:
                if o['lane_index'] == self.vehicle.lane_index[-1]:
                    samelane_obs.append(o)
                else:
                    difflane_obs.append(o)
        samelane_obs = samelane_obs[:self.max_samelaneobs_n]
        difflane_obs = difflane_obs[:self.max_difflaneobs_n]

        s_obs_trajectories = zeros((len(samelane_obs), self.N, self.state_n))
        d_obs_trajectories = zeros((len(difflane_obs), self.N, self.state_n))

        # 예측 trajectory 생성
        for obs_id, o in enumerate(samelane_obs):
            for stepN in range(self.N):
                s_obs_trajectories[obs_id, stepN, 0] = o['x'] + o['vx'] * stepN * self.dt
                s_obs_trajectories[obs_id, stepN, 1] = o['y'] + o['vy'] * stepN * self.dt
                s_obs_trajectories[obs_id, stepN, 2] = o['vx']
                s_obs_trajectories[obs_id, stepN, 3] = o['heading']
        for obs_id, o in enumerate(difflane_obs):
            for stepN in range(self.N):
                d_obs_trajectories[obs_id, stepN, 0] = o['x'] + o['vx'] * stepN * self.dt
                d_obs_trajectories[obs_id, stepN, 1] = o['y'] + o['vy'] * stepN * self.dt
                d_obs_trajectories[obs_id, stepN, 2] = o['vx']
                d_obs_trajectories[obs_id, stepN, 3] = o['heading']

        x0 = array([
            obs[0]['x'],
            obs[0]['y'],
            obs[0]['vx'],
            obs[0]['heading']
        ])

        result = self.solve(x0, future_ref, s_obs_trajectories, d_obs_trajectories)

        a = result[0]
        delta = result[1]

        action = [
            clip(a, *self.vehicle.ACCELERATION_RANGE),
            clip(delta, -self.max_steering_angle, self.max_steering_angle)
        ]
        self.vehicle.action['steering'] = clip(delta, -self.max_steering_angle, self.max_steering_angle)
        return action
    
