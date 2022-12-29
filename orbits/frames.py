from math import sin, cos, atan2, sqrt

import networkx as nx
import numpy as np


def rotations(pairs):
    dcms = {
        1: lambda a: np.array([[1, 0, 0], [0, cos(a), sin(a)], [0, -sin(a), cos(a)]]),
        2: lambda a: np.array([[cos(a), 0, -sin(a)], [0, 1, 0], [sin(a), 0, cos(a)]]),
        3: lambda a: np.array([[cos(a), sin(a), 0], [-sin(a), cos(a), 0], [0, 0, 1]]),
    }
    mat = np.identity(3)
    for ax, theta in pairs:
        dcm = dcms[ax](theta)
        mat = dcm @ mat
    return mat


def _get_dcm(transform, **params) -> np.ndarray:
    if callable(transform):
        # a function that returns either a matrix or axis, angle pairs
        return _get_dcm(transform(**params), **params)
    if isinstance(transform, np.ndarray) and transform.shape == (3, 3):
        # full matrix
        return transform
    # iterable of (axis, angle) pairs
    return rotations(transform)


def _get_vector(transform, **params) -> np.ndarray:
    if callable(transform):
        return _get_vector(transform(**params), **params)
    if isinstance(transform, np.ndarray) and transform.shape == (3, 1):
        return transform
    vec = np.array(transform)
    return vec.reshape((3, 1))


class FrameSystem(nx.DiGraph):
    def add_frame(self, name, parent=None, rotation=None, translation=None, inv_translation=None, default_units=None):
        # translation is expressed in the parent frame and is a vector from parent to child
        # inv_translation is also vector from parent to child, but in the child frame
        self.add_node(name)
        if not parent or not rotation:
            return
        if translation is None and inv_translation is None:
            if default_units:
                translation = _get_vector([default_units(0), default_units(0), default_units(0)])
            else:
                translation = _get_vector([0, 0, 0])
        if translation is not None:
            # translation = lambda **p: _get_vector(translation, **p)
            inverse_translation = lambda **p: _get_dcm(rotation, **p) @ -_get_vector(translation, **p)
        elif inv_translation is not None:
            inverse_translation = lambda **p: -_get_vector(inv_translation, **p)
            translation = lambda **p: _get_dcm(rotation, **p).transpose() @ _get_vector(inv_translation, **p)
        else:
            raise ValueError("Must have none or exactly one of translation and inv_translation")
        self.add_edge(parent, name, tr=translation, rot=rotation)
        inverse_rotation = lambda **p: _get_dcm(rotation, **p).transpose()
        self.add_edge(name, parent, tr=inverse_translation, rot=inverse_rotation)

    def transform(self, a, b, default_units=float, **params):
        path = nx.shortest_path(self, a, b)
        rotation = np.identity(3)
        translation = np.array([default_units(0), default_units(0), default_units(0)]).reshape((3, 1))
        for edge in zip(path[:-1], path[1:]):
            attrs = self.get_edge_data(*edge)
            matrix = _get_dcm(attrs["rot"], **params)
            translate = _get_vector(attrs["tr"], **params)
            translation = matrix @ (translate + translation)
            rotation = matrix @ rotation
        return rotation, translation


def obs_angles(r):
    r = np.vectorize(lambda k: float(k.to_meters().value))(r)
    azimuth = atan2(r[2], r[1])
    altitude = atan2(r[0], sqrt(r[2] ** 2 + r[1] ** 2))
    return altitude, azimuth
