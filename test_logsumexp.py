import numpy as np
import matplotlib.pyplot as plt
from controller.utils import Minkowski_sum, logsumexp_distance

def rotation(P, angle):
    return P @ np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])

def draw_polygon(P, *args, **kwds):
    plt.plot([*P[:, 0], P[0, 0]], [*P[:, 1], P[0, 1]], *args, **kwds)



if __name__=="__main__":

    L = 5.0
    W = 2.0

    A_p = np.array(
        [
            [ L/2, W/2],
            [-L/2, W/2],
            [-L/2,-W/2],
            [ L/2,-W/2]
        ]
    )
    A = A_p
    B_p = rotation(A_p, 0.5)
    B_t = np.array([12, 3])
    B = B_p + B_t

    S_p = Minkowski_sum(A_p, B_p)
    S = Minkowski_sum(A, B)

    resoution = 0.1
    xmin = np.min([*A[:, 0], *B[:, 0], *S[:, 0]])
    xmax = np.max([*A[:, 0], *B[:, 0], *S[:, 0]])
    xlen = xmax - xmin
    xpad = xlen * 0.1
    nx = int((xlen + 2*xpad)/resoution)
    ymin = np.min([*A[:, 1], *B[:, 1], *S[:, 1]])
    ymax = np.max([*A[:, 1], *B[:, 1], *S[:, 1]])
    ylen = ymax - ymin
    ypad = ylen * 0.1
    ny = int((ylen + 2*ypad)/resoution)

    xx, yy = np.meshgrid(np.linspace(xmin - xpad, xmax + xpad, nx), np.linspace(ymin - ypad, ymax + ypad, ny))
    plt.contourf(xx, yy, np.reshape([logsumexp_distance(S_p, np.array([x, y]) - B_t) for x, y in zip(xx.flatten(), yy.flatten())], (ny, nx)), levels=32)
    plt.colorbar()
    plt.contour(xx, yy, np.reshape([logsumexp_distance(S_p, np.array([x, y]) - B_t) - 0.3 for x, y in zip(xx.flatten(), yy.flatten())], (ny, nx)), 'k--', levels=[0.0])

    from autograd import jacobian
    jac = jacobian(lambda x: logsumexp_distance(S_p, x))
    print(jac(-B_t), jac(0.1 * np.ones(2) - B_t))

    draw_polygon(A, 'k')
    plt.scatter(0, 0, s=8, c='k')
    draw_polygon(B, 'k:')
    draw_polygon(S, 'k--')
    plt.legend(["ego vehicle", "ego vehicle center", "target vehicle", "Minkowski sum"])

    plt.axis('equal')
    plt.show()
