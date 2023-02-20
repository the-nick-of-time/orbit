from math import sin, cos, atan2, sqrt

import networkx as nx
import numpy as np


def rotations(pairs):
    """Calculate the rotation matrix from a sequence of (axis, angle) pairs.

    :param pairs: An ordered sequence of (axis, angle) pairs where axis is in
        {1, 2, 3} and angle is given in radians. The rotations are applied left
        to right, so [(3, pi), (1, -pi/2)] represents a counterclockwise
        rotation around Z by 180 degrees then a clockwise rotation around 1 by
        90 degrees. This example puts x'=-x, y'=z, z'=y.
    :return: A 3x3 numpy matrix for the full rotation.
    """
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
    """Calculate rotation matrix either directly or lazily.

    :param transform: One of three shapes:
        1. A 3x3 rotation matrix already calculated. Sometimes it's just easier
           to manually calculate a simple flip.
        2. A list of (axis, angle) pairs. This is fed to :py:`rotations`.
        3. A function that takes only keyword arguments. This is called with
           the keyword arguments passed into this function. The return value
           must be one of the two options above.
    :keyword params: A set of keyword arguments if the transform must be lazily
        evaluated.
    :return: A 3x3 numpy matrix for the rotation.
    """
    if callable(transform):
        # a function that returns either a matrix or axis, angle pairs
        return _get_dcm(transform(**params), **params)
    if isinstance(transform, np.ndarray) and transform.shape == (3, 3):
        # full matrix
        return transform
    # iterable of (axis, angle) pairs
    return rotations(transform)


def _get_vector(transform, **params) -> np.ndarray:
    """Calculate offset vector either directly or lazily.

    :param transform: One of three shapes:
        1. A numpy 3x1 (column) vector.
        2. A 3-element numeric sequence, which will be turned into a column
            vector.
        3. A function that takes only keyword arguments. This is called with
           the keyword arguments passed into this function. The return value
           must be one of the two options above.
    :keyword params: A set of keyword arguments if the transform must be lazily
        evaluated.
    :return: A 3x1 numpy vector representing an offset.
    """
    if callable(transform):
        return _get_vector(transform(**params), **params)
    if isinstance(transform, np.ndarray) and transform.shape == (3, 1):
        return transform
    vec = np.array(transform)
    return vec.reshape((3, 1))


class FrameSystem:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_frame(self, name, parent=None, rotation=None, translation=None, inv_translation=None):
        """Define a reference frame within the system.

        For rotation and translation/inv_translation, you must use the function
        style for any time-variant values. These functions must take any number
        of keyword arguments, but only care about particular named ones that
        are relevant to the calculation. The set of names that are used is very
        important for evaluating a transform later.

        :param name: A unique name for this frame.
        :param parent: The name of the existing frame that this new one is
            defined against.
        :param rotation: The rotation matrix from the parent frame to the new
            frame. Can take anything that :py:`_get_dcm` can.
        :param translation: The vector from the origin of the parent frame to
            the origin of the daughter frame, given in the parent frame.
            Mutually exclusive with inv_translation.
        :param inv_translation: The vector from the origin of the parent frame
            to the origin of the daughter frame, given in the daughter frame.
            Mutually exclusive with translation.
        """
        self.graph.add_node(name)
        if not parent:
            return
        if not rotation and not translation:
            self.graph.add_edge(parent, name, tr=_get_vector([0, 0, 0]), rot=np.identity(3))
            self.graph.add_edge(name, parent, tr=_get_vector([0, 0, 0]), rot=np.identity(3))
            return
        if translation is None and inv_translation is None:
            translation = _get_vector([0, 0, 0])
        if translation is not None:
            inverse_translation = lambda **p: _get_dcm(rotation, **p) @ -_get_vector(translation, **p)
        elif inv_translation is not None:
            inverse_translation = lambda **p: -_get_vector(inv_translation, **p)
            translation = lambda **p: _get_dcm(rotation, **p).transpose() @ _get_vector(inv_translation, **p)
        else:
            raise ValueError("Must have none or exactly one of translation and inv_translation")
        self.graph.add_edge(parent, name, tr=translation, rot=rotation)
        inverse_rotation = lambda **p: _get_dcm(rotation, **p).transpose()
        self.graph.add_edge(name, parent, tr=inverse_translation, rot=inverse_rotation)

    def transform(self, a, b, **params):
        """Calculate the total rotation and translation from one frame to another.

        You must pass all keyword arguments that are required by any frame
        between the start and end.

        :param a: The name of the frame to transform from.
        :param b: The name of the frame to transform to.
        :keyword params: Any number of keyword arguments, that will be passed to
            any transforms that require dynamic evaluation.
        :return: A tuple of:
            1. The total rotation matrix between the two frames. That is,
               if you have a vector in the start frame *v* and multiply it by
               this matrix *R* like *R @ v*, you will get the equivalent vector
               seen from the end frame.
            2. The translation between the origin of the start frame and the
               origin of the end frame, expressed in the end frame.
        """
        path = nx.shortest_path(self.graph, a, b)
        rotation = np.identity(3)
        translation = np.array([0, 0, 0]).reshape((3, 1))
        for edge in zip(path[:-1], path[1:]):
            attrs = self.graph.get_edge_data(*edge)
            matrix = _get_dcm(attrs["rot"], **params)
            translate = _get_vector(attrs["tr"], **params)
            translation = matrix @ (translate + translation)
            rotation = matrix @ rotation
        return rotation, translation


def altitude(r):
    """Calculate the altitude angle defined by the vector's direction.

    :param r: The column vector of position.
    :return: The altitude, in radians. This is defined as the angle between the
        vector and the y-z plane.
    """
    return atan2(r[0], sqrt(r[2] ** 2 + r[1] ** 2))


def azimuth(r):
    """Calculate the azimuth angle defined by the vector's direction.

    :param r: The column vector of position.
    :return: The azimuth, in radians. This is defined as the angle from y in the
        y-z plane.
    """
    return atan2(r[2], r[1])
