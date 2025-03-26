from typing import Iterable
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull
from scipy.optimize import approx_fprime
#from ..highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.controller import ControlledVehicle



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



def check_possible_lane_changes(vehicle):
    """
    Compute the signed distance between polygon and point.

    :param polygon: polygon, as a list of [x, y] points
    :param point: point, as [x, y] point
    :return: list of str in {"LANE_LEFT", "IDLE", "LANE_RIGHT"}
    """
    possible_lane_changes = ["IDLE"]
    _from, _to, _id = vehicle.target_lane_index
    target_lane_index = (
        _from,
        _to,
        np.clip(_id - 1, 0, len(vehicle.road.network.graph[_from][_to]) - 1),
    )
    if target_lane_index[-1] != vehicle.target_lane_index[-1]:
        if vehicle.road.network.get_lane(target_lane_index).is_reachable_from(
            vehicle.position
        ):
            possible_lane_changes.append("LANE_LEFT")
    target_lane_index = (
        _from,
        _to,
        np.clip(_id + 1, 0, len(vehicle.road.network.graph[_from][_to]) - 1),
    )
    if target_lane_index[-1] != vehicle.target_lane_index[-1]:
        if vehicle.road.network.get_lane(target_lane_index).is_reachable_from(
            vehicle.position
        ):
            possible_lane_changes.append("LANE_RIGHT")
    
    return possible_lane_changes


def predict_state(dt, steps, vehicle, lane_change, **kwds):
    v = ControlledVehicle.create_from(vehicle)
    v.act(lane_change, **kwds)
    for _ in range(steps):
        v.step(dt/steps)
    return np.array([*v.position, v.heading, v.speed])


def get_polygon(length=5.0, width=2.0, heading=0.0):
    return np.array(
        [
            [ length/2, width/2],
            [-length/2, width/2],
            [-length/2,-width/2],
            [ length/2,-width/2]
        ]
    ) @ np.array([[np.cos(heading), np.sin(heading)], [-np.sin(heading), np.cos(heading)]])
