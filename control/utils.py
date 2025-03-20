from typing import Iterable
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull



def Minkowski_sum(polygon1: Iterable[NDArray], polygon2: Iterable[NDArray]) -> Iterable[NDArray]:
    """
    Compute the Minkowski sum of two polygons.

    :param polygon1: polygon 1, as a list of [x, y] points
    :param polygon2: polygon 2, as a list of [x, y] points
    :return: polygon of Minkowski sum
    """
    hull = ConvexHull(np.repeat(polygon1, len(polygon2), axis=0) + np.tile(polygon2, (len(polygon1), 1)))
    return hull.points[hull.vertices]



def signed_distance(polygon: Iterable[NDArray], point: NDArray) -> float:
    """
    Compute the signed distance between polygon and point.

    :param polygon: polygon, as a list of [x, y] points
    :param point: point, as [x, y] point
    :return: signed distance
    """
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


