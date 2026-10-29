"""
Lengths of straight rays inside the cells of an axis-aligned mesh.

This module only depends on numpy, scipy and discretize so that it can be
tested and reused independently of the SimPEG simulation classes.
"""

import itertools

import numpy as np
import scipy.sparse as sp
from discretize import TensorMesh, TreeMesh

try:
    from ._ray_geometry_numba import tensor_mesh_lengths as _numba_tensor_mesh_lengths
except ImportError:
    _numba_tensor_mesh_lengths = None

__all__ = ["ray_cell_lengths", "ray_cell_length_matrix"]

ENGINES = ("numba", "numpy")
DEFAULT_ENGINE = "numba" if _numba_tensor_mesh_lengths is not None else "numpy"


def ray_cell_lengths(mesh, x0, x1, rtol=1e-10, engine=None):
    """Compute the length of each straight ray inside each cell it crosses.

    Parameters
    ----------
    mesh : discretize.TensorMesh or discretize.TreeMesh
        A 2D or 3D mesh.
    x0, x1 : (n_rays, dim) array_like
        Start and end points of each ray. All points must lie inside the mesh.
    rtol : float, optional
        Tolerance, relative to the largest mesh dimension, used to decide
        whether a point is inside the mesh and whether a ray lies on a cell
        face.
    engine : {"numba", "numpy"}, optional
        Implementation used for tensor meshes. Defaults to ``"numba"`` when
        numba is installed. Tree meshes always use the numpy implementation.

    Returns
    -------
    ray_index : (n_entries,) numpy.ndarray of int
    cell_index : (n_entries,) numpy.ndarray of int
    length : (n_entries,) numpy.ndarray of float
        Coordinate (COO) triplets of the ray-cell length matrix. A
        (ray, cell) pair may appear more than once; duplicate entries are
        meant to be summed.

    Notes
    -----
    A ray that lies exactly on a face (or edge) shared by several cells has
    its length split equally between those cells. On the outer boundary of
    the mesh, all of the length goes to the cells inside the mesh.
    """
    copy_ray, copy_index, cell_index, length = _trace(mesh, x0, x1, rtol, engine)
    return copy_ray[copy_index], cell_index, length


def ray_cell_length_matrix(mesh, x0, x1, rtol=1e-10, engine=None):
    """Sparse matrix of ray lengths in each cell.

    Parameters
    ----------
    mesh : discretize.TensorMesh or discretize.TreeMesh
        A 2D or 3D mesh.
    x0, x1 : (n_rays, dim) array_like
        Start and end points of each ray.
    rtol : float, optional
        See :func:`ray_cell_lengths`.
    engine : {"numba", "numpy"}, optional
        See :func:`ray_cell_lengths`.

    Returns
    -------
    (n_rays, n_cells) scipy.sparse.csr_matrix
        Entry ``(i, j)`` is the length of ray ``i`` inside cell ``j``. Column
        indices within a row are in the order the ray crosses the cells, not
        sorted.
    """
    copy_ray, copy_index, cell_index, length = _trace(mesh, x0, x1, rtol, engine)
    n_rays = np.atleast_2d(x0).shape[0]
    shape = (n_rays, mesh.n_cells)

    if copy_ray.size == n_rays and np.all(copy_ray == np.arange(n_rays)):
        # Each copy is a ray, in order, so the segments are already grouped by
        # row, and a single copy never crosses the same cell twice.
        indptr = np.zeros(n_rays + 1, dtype=np.int64)
        np.cumsum(np.bincount(copy_index, minlength=n_rays), out=indptr[1:])
        return sp.csr_matrix((length, cell_index, indptr), shape=shape)

    # Copies of a split ray can land in the same cell (e.g. on a TreeMesh
    # when a fine-level face runs through a coarse cell), so sum duplicates.
    ray_index = copy_ray[copy_index]
    G = sp.csr_matrix((length, (ray_index, cell_index)), shape=shape)
    G.sum_duplicates()
    return G


def _trace(mesh, x0, x1, rtol, engine):
    """Validate inputs, split face-aligned rays and trace all copies.

    Returns
    -------
    copy_ray : (n_copies,) numpy.ndarray of int
        Ray of each copy. Usually ``arange(n_rays)``; see
        :func:`_split_face_aligned_rays`.
    copy_index, cell_index, length : (n_entries,) numpy.ndarray
        Segments of each copy, grouped by copy in increasing order, with the
        copy weights already applied to the lengths.
    """
    _validate_mesh(mesh)
    engine = _validate_engine(engine)
    x0 = _validate_points("x0", x0, mesh.dim)
    x1 = _validate_points("x1", x1, mesh.dim)
    if x0.shape != x1.shape:
        raise ValueError(
            f"x0 and x1 must have the same shape, got {x0.shape} and {x1.shape}."
        )

    bounds_min, bounds_max = _mesh_bounds(mesh)
    tol = rtol * np.max(bounds_max - bounds_min)
    outside = ~(
        _points_inside(x0, bounds_min - tol, bounds_max + tol)
        & _points_inside(x1, bounds_min - tol, bounds_max + tol)
    )
    if np.any(outside):
        raise ValueError(
            f"{outside.sum()} ray(s) have endpoints outside of the mesh, "
            f"e.g. ray {np.flatnonzero(outside)[0]}. "
            "Extend the mesh to contain all sources and receivers."
        )
    # Snap points that are within tolerance of the boundary onto it.
    x0 = np.clip(x0, bounds_min, bounds_max)
    x1 = np.clip(x1, bounds_min, bounds_max)

    copy_ray, copy_weight, x0, x1 = _split_face_aligned_rays(mesh, x0, x1, tol)

    if isinstance(mesh, TensorMesh) and engine == "numba":
        copy_index, cell_index, length = _numba_tensor_mesh_lengths(
            tuple(_axis_nodes(mesh, axis) for axis in range(mesh.dim)),
            mesh.shape_cells,
            x0,
            x1,
            tol,
        )
    elif isinstance(mesh, TensorMesh):
        copy_index, cell_index, length = _tensor_mesh_lengths(mesh, x0, x1, tol)
    else:
        copy_index, cell_index, length = _tree_mesh_lengths(mesh, x0, x1, tol)

    if np.any(copy_weight != 1.0):
        length = length * copy_weight[copy_index]
    return copy_ray, copy_index, cell_index, length


def _validate_mesh(mesh):
    if not isinstance(mesh, (TensorMesh, TreeMesh)):
        raise TypeError(
            "mesh must be a discretize.TensorMesh or discretize.TreeMesh, "
            f"got {type(mesh).__name__}."
        )
    if mesh.dim not in (2, 3):
        raise ValueError(f"mesh must be 2D or 3D, got a {mesh.dim}D mesh.")


def _validate_engine(engine):
    if engine is None:
        return DEFAULT_ENGINE
    if engine not in ENGINES:
        raise ValueError(f"engine must be one of {ENGINES}, got {engine!r}.")
    if engine == "numba" and _numba_tensor_mesh_lengths is None:
        raise ImportError("engine='numba' requires numba to be installed.")
    return engine


def _validate_points(name, points, dim):
    points = np.atleast_2d(np.asarray(points, dtype=float))
    if points.ndim != 2 or points.shape[1] != dim:
        raise ValueError(f"{name} must have shape (n_rays, {dim}), got {points.shape}.")
    return points


def _mesh_bounds(mesh):
    bounds_min = np.asarray(mesh.origin, dtype=float)
    bounds_max = bounds_min + np.array([np.sum(h) for h in mesh.h])
    return bounds_min, bounds_max


def _axis_nodes(mesh, axis):
    """Every coordinate along ``axis`` where a cell face can be.

    For a TreeMesh, ``mesh.h`` holds the widths of the finest level, so these
    are all of the potential face locations.
    """
    return mesh.origin[axis] + np.r_[0.0, np.cumsum(mesh.h[axis])]


def _points_inside(points, bounds_min, bounds_max):
    return np.all((points >= bounds_min) & (points <= bounds_max), axis=1)


def _split_face_aligned_rays(mesh, x0, x1, tol):
    """Replace rays lying on cell faces by copies shifted off the face.

    A ray parallel to axis ``d`` whose ``d`` coordinate is on a node plane is
    copied twice, shifted by a quarter of the finest cell width to either side
    of the plane, and each copy is given half the weight. In 3D a ray on an
    edge is on two planes and gets four copies. Shifting a ray parallel to an
    axis does not change its length in any cell, so the weighted copies give
    an equal split of the length between the neighbouring cells. Copies that
    leave the mesh are dropped and the weights renormalized.

    Returns
    -------
    copy_ray : (n_copies,) numpy.ndarray of int
        Index of the original ray for each copy.
    copy_weight : (n_copies,) numpy.ndarray of float
    x0, x1 : (n_copies, dim) numpy.ndarray
    """
    n_rays, dim = x0.shape
    on_face = np.zeros((n_rays, dim), dtype=bool)
    shift = np.empty(dim)
    for axis in range(dim):
        nodes = _axis_nodes(mesh, axis)
        parallel = np.abs(x1[:, axis] - x0[:, axis]) <= tol
        on_face[:, axis] = parallel & (_distance_to_nearest(nodes, x0[:, axis]) <= tol)
        shift[axis] = 0.25 * np.min(mesh.h[axis])

    if not np.any(on_face):
        return np.arange(n_rays), np.ones(n_rays), x0, x1

    copy_ray, copy_x0, copy_x1 = [], [], []
    for signs in itertools.product((1.0, -1.0), repeat=dim):
        signs = np.array(signs)
        # Axes that are not on a face only take the + sign, so each distinct
        # copy is made once.
        use = np.all(on_face | (signs > 0), axis=1)
        offset = np.where(on_face[use], signs * shift, 0.0)
        copy_ray.append(np.flatnonzero(use))
        copy_x0.append(x0[use] + offset)
        copy_x1.append(x1[use] + offset)
    copy_ray = np.concatenate(copy_ray)
    copy_x0 = np.vstack(copy_x0)
    copy_x1 = np.vstack(copy_x1)

    bounds_min, bounds_max = _mesh_bounds(mesh)
    inside = _points_inside(copy_x0, bounds_min, bounds_max) & _points_inside(
        copy_x1, bounds_min, bounds_max
    )
    copy_ray = copy_ray[inside]
    n_copies = np.bincount(copy_ray, minlength=n_rays)
    copy_weight = 1.0 / n_copies[copy_ray]
    return copy_ray, copy_weight, copy_x0[inside], copy_x1[inside]


def _distance_to_nearest(sorted_values, x):
    i = np.clip(np.searchsorted(sorted_values, x), 1, len(sorted_values) - 1)
    return np.minimum(np.abs(x - sorted_values[i - 1]), np.abs(x - sorted_values[i]))


def _tensor_mesh_lengths(mesh, x0, x1, tol):
    """Siddon's algorithm, vectorized over all rays.

    Each ray is parametrized as ``x0 + alpha * (x1 - x0)`` with ``alpha`` in
    [0, 1]. The values of ``alpha`` where the ray crosses node planes split it
    into segments that each lie in a single cell.
    """
    n_rays, dim = x0.shape
    delta = x1 - x0
    ray_length = np.linalg.norm(delta, axis=1)
    ray_ids = np.arange(n_rays)

    rays = [ray_ids, ray_ids]
    alphas = [np.zeros(n_rays), np.ones(n_rays)]
    for axis in range(dim):
        nodes = _axis_nodes(mesh, axis)
        lo = np.minimum(x0[:, axis], x1[:, axis])
        hi = np.maximum(x0[:, axis], x1[:, axis])
        # node planes strictly between the two endpoints
        first = np.searchsorted(nodes, lo, side="right")
        n_cross = np.maximum(np.searchsorted(nodes, hi, side="left") - first, 0)

        crossing_ray = np.repeat(ray_ids, n_cross)
        offsets = np.cumsum(n_cross) - n_cross
        node_index = first[crossing_ray] + (
            np.arange(crossing_ray.size) - offsets[crossing_ray]
        )
        rays.append(crossing_ray)
        alphas.append(
            (nodes[node_index] - x0[crossing_ray, axis]) / delta[crossing_ray, axis]
        )

    rays = np.concatenate(rays)
    alphas = np.concatenate(alphas)
    order = np.lexsort((alphas, rays))
    rays, alphas = rays[order], alphas[order]

    same_ray = rays[1:] == rays[:-1]
    segment_ray = rays[:-1][same_ray]
    alpha_start = alphas[:-1][same_ray]
    alpha_end = alphas[1:][same_ray]
    length = (alpha_end - alpha_start) * ray_length[segment_ray]

    keep = length > tol
    segment_ray = segment_ray[keep]
    length = length[keep]
    alpha_mid = 0.5 * (alpha_start[keep] + alpha_end[keep])
    midpoints = x0[segment_ray] + alpha_mid[:, None] * delta[segment_ray]

    cell_subscripts = tuple(
        np.clip(
            np.searchsorted(_axis_nodes(mesh, axis), midpoints[:, axis], side="right")
            - 1,
            0,
            mesh.shape_cells[axis] - 1,
        )
        for axis in range(dim)
    )
    cell_index = np.ravel_multi_index(cell_subscripts, mesh.shape_cells, order="F")
    return segment_ray, cell_index, length


def _tree_mesh_lengths(mesh, x0, x1, tol):
    """Candidate cells from the tree, then clip each ray to each candidate."""
    candidates = [
        np.asarray(mesh.get_cells_on_line(np.stack([a, b])), dtype=np.int64)
        for a, b in zip(x0, x1)
    ]
    n_candidates = np.array([c.size for c in candidates], dtype=np.int64)
    ray_index = np.repeat(np.arange(x0.shape[0]), n_candidates)
    if ray_index.size == 0:
        empty = np.array([], dtype=np.int64)
        return empty, empty, np.array([])
    cell_index = np.concatenate(candidates)

    bounds = mesh.cell_bounds.reshape(mesh.n_cells, mesh.dim, 2)
    length = _slab_clip_lengths(
        x0[ray_index],
        x1[ray_index],
        bounds[cell_index, :, 0],
        bounds[cell_index, :, 1],
    )
    keep = length > tol
    return ray_index[keep], cell_index[keep], length[keep]


def _slab_clip_lengths(x0, x1, cell_min, cell_max):
    """Length of each segment ``x0 -> x1`` inside the matching box.

    All arguments are ``(n, dim)``; row ``i`` of the output is the length of
    segment ``i`` inside box ``i``.
    """
    delta = x1 - x0
    t_in = np.zeros(x0.shape[0])
    t_out = np.ones(x0.shape[0])
    for axis in range(x0.shape[1]):
        d = delta[:, axis]
        parallel = d == 0
        inside = (cell_min[:, axis] <= x0[:, axis]) & (x0[:, axis] <= cell_max[:, axis])
        with np.errstate(divide="ignore", invalid="ignore"):
            t_a = (cell_min[:, axis] - x0[:, axis]) / d
            t_b = (cell_max[:, axis] - x0[:, axis]) / d
        t_lo = np.where(
            parallel, np.where(inside, -np.inf, np.inf), np.minimum(t_a, t_b)
        )
        t_hi = np.where(
            parallel, np.where(inside, np.inf, -np.inf), np.maximum(t_a, t_b)
        )
        t_in = np.maximum(t_in, t_lo)
        t_out = np.minimum(t_out, t_hi)
    return np.maximum(t_out - t_in, 0.0) * np.linalg.norm(delta, axis=1)
