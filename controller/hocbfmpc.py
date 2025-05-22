import numpy as np
from scipy.stats import norm
from scipy.linalg import block_diag
from autograd import jacobian, hessian
from qpsolvers import solve_qp
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.utils import lmap
from .utils import get_polygon, Minkowski_sum, signed_distance, rotation

lane_change_dict = {"LANE_LEFT": 0, "IDLE": 1, "LANE_RIGHT": 2}

class HOCBFMPC:

    def __init__(self, vehicle, dt, ref_speed, horizon=10, safe_dist=0.1, alpha1=lambda x: 0.6 * x, alpha2=lambda x: 2.4 * x, mu=np.array([-2.5, 0.0]), Sigma=np.diag([7.5**2, 0.1**2]), confidence=0.98, slack_penalty_max=1e6):
        self.vehicle = vehicle
        self.dt = dt
        self.ref_speed = ref_speed
        self.horizon = horizon
        self.safe_dist = safe_dist
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.mu = mu
        self.Sigma = Sigma
        self.quantile = norm.ppf(confidence)
        self.slack_penalty_max = slack_penalty_max

    def _compute_cbf_terms(self, obs, state, lane_change):
        # ego/other polygon
        ego_poly = get_polygon(self.vehicle.LENGTH, self.vehicle.WIDTH, state[2])
        other_polys = [get_polygon(o['length'], o['width'], o['heading']) for o in obs[1:]]

        # linearized dynamics
        v_ctrl = ControlledVehicle.create_from(self.vehicle)
        v_ctrl.state = state.copy()
        v_ctrl.act(lane_change)
        beta = np.arctan(0.5 * np.tan(v_ctrl.action['steering']))
        tire_slip_angle = np.abs(v_ctrl.action['steering'] - beta)
        dfxds = np.array([
            np.cos(v_ctrl.heading + beta),
            np.sin(v_ctrl.heading + beta),
            np.sin(beta) / (v_ctrl.LENGTH / 2),
            -np.sin(tire_slip_angle) * v_ctrl.LATERAL_TIRE_COEF * tire_slip_angle
        ])
        fx = dfxds * state[3]
        dfxdx = np.zeros((4,4))
        dfxdx[:,2] = [-fx[1], fx[0], 0.0, 0.0]
        dfxdx[:,3] = dfxds
        gx = np.array([0.0, 0.0, 0.0, np.cos(beta)])

        # 3) 각 오브젝트마다 A_i, b_i
        A_rows, b_rows = [], []
        for o, other_poly in zip(obs[1:], other_polys):
            # signed distance h, gradient dhdx, Hessian d2hdx2 at zero relative state
            h_func = lambda x: signed_distance(
                Minkowski_sum(ego_poly, other_poly),
                x[:2] + np.array([state[0]-o['x'], state[1]-o['y']])
            )
            h0 = h_func(np.zeros(4))
            dhdx0 = jacobian(h_func)(np.zeros(4))
            d2hdx20 = hessian(h_func)(np.zeros(4))

            # 상대 속도/제어 영향
            fz = np.hstack([fx, np.array([o['vx'], o['vy'], 0.0, 0.0])])
            dfdz = block_diag(dfxdx, np.eye(4, k=2))
            go = np.zeros((4,2)); go[2:,:] = rotation(o['heading'])
            gz = block_diag(gx[:,None], go)
            dhdz = np.hstack([dhdx0, -dhdx0])
            d2hdz2 = np.block([[d2hdx20, -d2hdx20],[-d2hdx20, d2hdx20]])
            dLfhdz = dhdz @ dfdz + d2hdz2 @ fz
            L2fh = dLfhdz @ fz
            LgLfh = dLfhdz @ gz

            Ai = -LgLfh[0]
            bi = L2fh \
                 + self.alpha2(dhdz @ fz + self.alpha1(h0)) \
                 + LgLfh[1:] @ self.mu \
                 - self.quantile * np.sqrt(LgLfh[1:] @ self.Sigma @ LgLfh[1:])
            A_rows.append(Ai)
            b_rows.append(bi)

        A = np.vstack(A_rows)   # shape (ns, )
        b = np.array(b_rows)
        return A, b

    def solve(self, obs, lane_change="IDLE"):
        # 1) 초기 상태
        x0 = np.array([
            obs[0]['x'], obs[0]['y'],
            obs[0]['heading'], obs[0]['speed']
        ])

        ns = len(obs) - 1 # 장애물 개수
        N  = self.horizon # 예측 단계 수
        n_u = N           # 제어 입력 변수 개수
        n_s = N * ns      # slack 변수 개수

        # 2) Objective P, q
        P = block_diag(np.eye(n_u), np.eye(n_s) * self.slack_penalty_max) * 0.5
        uref = np.clip((self.ref_speed - x0[3]) / self.dt,
                       *self.vehicle.ACCELERATION_RANGE)
        q = np.hstack([-np.ones(n_u) * uref, np.zeros(n_s)])

        # 3) G, h 스택
        G_rows = []
        h_rows = []
        x = x0.copy()
        for t in range(N):
            # 시점 t에서 CBF 계수 A_small, b_small 계산
            A_small, b_small = self._compute_cbf_terms(obs, x, lane_change)
            A_vec = A_small.flatten()   # shape (ns,)

            # 장애물 j마다 하나의 제약 row 생성
            for j in range(ns):
                row = np.zeros(n_u + n_s)
                # u_t 계수
                row[t] = A_vec[j]
                # slack_{t,j} 계수
                row[n_u + t*ns + j] = -1.0
                G_rows.append(row)
                h_rows.append(b_small[j])

            # 단순 Euler 예측 (heading 변화 무시한 예시)
            x[:2] += x[3] * self.dt * np.array([np.cos(x[2]), np.sin(x[2])])
            x[3]  += uref * self.dt

        G = np.vstack(G_rows) # shape (N·ns, n_u+n_s)
        h = np.array(h_rows)  # shape (N·ns,)

        lb = np.hstack([np.ones(n_u)*self.vehicle.ACCELERATION_RANGE[0], np.zeros(n_s)])
        ub = np.hstack([np.ones(n_u)*self.vehicle.ACCELERATION_RANGE[1], np.ones(n_s)*np.inf])

        sol = solve_qp(P=P, q=q, G=G, h=h, lb=lb, ub=ub, solver='quadprog')
        if sol is None:
            return [lane_change_dict["IDLE"], 0.0]

        u0 = sol[0]
        acc_cmd = lmap(np.clip(u0,
                               *self.vehicle.ACCELERATION_RANGE),
                       self.vehicle.ACCELERATION_RANGE,
                       (-1.0, 1.0))
        return [lane_change_dict[lane_change], acc_cmd]