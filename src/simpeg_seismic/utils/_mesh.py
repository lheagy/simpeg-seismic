"""Geometry helpers for axis-aligned (tensor and tree) meshes."""

import numpy as np


def mesh_bounds(mesh):
    """Lower and upper corners of the mesh.

    Returns
    -------
    bounds_min, bounds_max : (dim,) numpy.ndarray
    """
    bounds_min = np.asarray(mesh.origin, dtype=float)
    bounds_max = bounds_min + np.array([np.sum(h) for h in mesh.h])
    return bounds_min, bounds_max


def axis_nodes(mesh, axis):
    """Every coordinate along ``axis`` where a cell face can be.

    For a TensorMesh these are the node coordinates. For a TreeMesh,
    ``mesh.h`` holds the widths of the finest level, so these are all of the
    potential face locations.
    """
    return mesh.origin[axis] + np.r_[0.0, np.cumsum(mesh.h[axis])]


def points_inside(points, bounds_min, bounds_max):
    """Whether each row of ``points`` lies in the box, boundary included."""
    return np.all((points >= bounds_min) & (points <= bounds_max), axis=1)


def check_points_inside(mesh, name, points, tol=0.0):
    """Raise a ValueError if any of ``points`` is outside the mesh.

    ``tol`` widens the mesh on every side, and ``name`` describes the points
    in the error message.
    """
    bounds_min, bounds_max = mesh_bounds(mesh)
    outside = ~points_inside(points, bounds_min - tol, bounds_max + tol)
    if np.any(outside):
        raise ValueError(
            f"{outside.sum()} {name} location(s) are outside of the mesh, "
            f"e.g. {points[np.flatnonzero(outside)[0]]}."
        )
