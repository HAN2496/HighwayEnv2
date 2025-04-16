from typing import Iterable, Optional
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull
from autograd import numpy as anp
import torch
from highway_env.vehicle.controller import ControlledVehicle

EXP_INPUT_MAX = np.log(np.finfo(float).max)
EXP_INPUT_MIN = np.log(np.finfo(float).tiny)



def Minkowski_sum(polygon1: Iterable[NDArray], polygon2: Iterable[NDArray]) -> Iterable[NDArray]:
    """
    Compute the Minkowski sum of two polygons.

    :param polygon1: polygon 1, as a list of [x, y] points
    :param polygon2: polygon 2, as a list of [x, y] points
    :return: polygon of Minkowski sum
    """
    hull = ConvexHull(np.repeat(polygon1, len(polygon2), axis=0) + np.tile(polygon2, (len(polygon1), 1)))
    return hull.points[hull.vertices]



#def signed_distance(polygon: Iterable[NDArray], point: NDArray) -> float:
#    """
#    Compute the signed distance between polygon and point.
#
#    :param polygon: polygon, as a list of [x, y] points
#    :param point: point, as [x, y] point
#    :return: signed distance
#    """
#    #sqdist = np.inf
#    #sign = 1.0
#    sqdist = []
#    sign = []
#    n = len(polygon)
#    c = np.mean(polygon, axis=0)
#    for i in range(n):
#        j = (i + n - 1) % n
#        e = polygon[j] - polygon[i]
#        w = point - polygon[i]
#        e *= np.clip(np.dot(w, e)/np.dot(e, e), 0.0, 1.0)
#        b = w - e
#        sqdist.append(np.dot(b, b))
#        sign.append(np.sign(-np.dot(b, c - polygon[i] - e)))
#    i = np.argmin(sqdist)
#    return sign[i] * np.sqrt(sqdist[i])



def signed_distance(polygon: Iterable[NDArray], point: NDArray) -> float:
    """
    Compute the signed distance between polygon and point.

    :param polygon: polygon, as a list of [x, y] points
    :param point: point, as [x, y] point
    :return: signed distance
    """
    c = anp.mean(polygon, axis=0)
    e = anp.r_[polygon[-1:, :], polygon[:-1, :]] - polygon
    w = point - polygon
    e *= anp.clip(anp.sum(w * e, axis=1, keepdims=True) / anp.sum(e * e, axis=1, keepdims=True), 0.0, 1.0)
    b = w - e
    dist = anp.sqrt(anp.sum(b*b, axis=1))
    sign = anp.sign(-anp.sum(b * (c - polygon - e), axis=1))
    i = anp.argmin(dist)
    return sign[i] * dist[i]



#def logsumexp_distance(polygon: Iterable[NDArray], point: NDArray) -> float:
#    """
#    Compute the approx. distance between polygon and point using logsumexp trick.
#
#    :param polygon: polygon, as a list of [x, y] points
#    :param point: point, as [x, y] point
#    :return: (approx.) distance
#    """
#    sqdist = []
#    n = len(polygon)
#    for i in range(n):
#        j = (i + n - 1) % n
#        e = polygon[j] - polygon[i]
#        w = point - polygon[i]
#        e *= np.clip(np.dot(w, e)/np.dot(e, e), 0.0, 1.0)
#        b = w - e
#        sqdist.append(np.dot(b, b))
#    dist = np.sqrt(sqdist)
#    scale = (EXP_INPUT_MAX / n - EXP_INPUT_MIN) / np.max(dist)
#    return (
#        EXP_INPUT_MAX / n - np.log(
#            np.sum(
#                np.exp(EXP_INPUT_MAX / n - scale * dist)
#            )
#        )
#    ) / scale



def logsumexp_distance(polygon: Iterable[NDArray], point: NDArray):
    """
    Compute the approx. distance between polygon and point using logsumexp trick.

    :param polygon: polygon, as a list of [x, y] points
    :param point: point, as [x, y] point
    :return: (approx.) distance
    """
    n = polygon.shape[0]
    e = anp.r_[polygon[-1:, :], polygon[:-1, :]] - polygon
    w = point - polygon
    e *= anp.clip(anp.sum(w * e, axis=1, keepdims=True) / anp.sum(e * e, axis=1, keepdims=True), 0.0, 1.0)
    b = w - e
    dist = anp.sqrt(anp.sum(b*b, axis=1))
    scale = (EXP_INPUT_MAX / n - EXP_INPUT_MIN) / anp.max(dist)
    return (
        EXP_INPUT_MAX / n - anp.log(
            anp.sum(
                anp.exp(EXP_INPUT_MAX / n - scale * dist)
            )
        )
    ) / scale



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
    ) @ rotation(heading)


def rotation(heading):
    return np.array([[np.cos(heading), np.sin(heading)], [-np.sin(heading), np.cos(heading)]])
