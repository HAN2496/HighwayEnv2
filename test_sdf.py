import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

def rotation(P, angle):
    return P @ np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])

def draw_polygon(P, *args, **kwds):
    plt.plot([*P[:, 0], P[0, 0]], [*P[:, 1], P[0, 1]], *args, **kwds)

def Minkowski_sum(P1, P2):
    hull = ConvexHull(np.repeat(P1, len(P2), axis=0) + np.tile(P2, (len(P1), 1)))
    return hull.points[hull.vertices]

def signed_distance(polygon, point):
    sqdist = np.inf
    sign = 1.0
    n = len(polygon)
    for i in range(n):
        j = (i + n - 1) % n
        e = polygon[j] - polygon[i]
        w = point - polygon[i]
        b = w - e * np.clip(np.dot(w, e)/np.dot(e, e), 0.0, 1.0)
        sqdist = min(sqdist, np.dot(b, b))
        c = np.array([point[1]>=polygon[i, 1], point[1]<polygon[j, 1], e[0]*w[1]>e[1]*w[0]])
        sign = np.where(np.all(c) | np.all(~c), -sign, sign)
    return sign * np.sqrt(sqdist)



if __name__=="__main__":

    L = 5.0
    W = 2.0

    A = np.array(
        [
            [ L/2, W/2],
            [-L/2, W/2],
            [-L/2,-W/2],
            [ L/2,-W/2]
        ]
    )

    B = rotation(A, 0.5) + np.array([7, 5])

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
    plt.contourf(xx, yy, np.reshape([signed_distance(S, np.array([x, y])) for x, y in zip(xx.flatten(), yy.flatten())], (ny, nx)), levels=24)
    plt.colorbar()

    draw_polygon(A, 'k')
    draw_polygon(B, 'k')
    draw_polygon(S, 'k--')

    plt.axis('equal')
    plt.show()
