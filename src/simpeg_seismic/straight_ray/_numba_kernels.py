"""
Numba implementation of Siddon's algorithm for tensor meshes.

Each ray is traced on its own: the node-plane crossings along each axis are
already in order, so they are merged in a single pass instead of sorted.
Rays are processed in parallel. Importing this module requires numba.
"""

import numpy as np
from numba import njit, prange


def tensor_mesh_lengths(nodes, shape_cells, x0, x1, tol):
    """Lengths of rays in the cells of a tensor mesh.

    Parameters
    ----------
    nodes : tuple of numpy.ndarray
        Node coordinates along each axis.
    shape_cells : tuple of int
        Number of cells along each axis.
    x0, x1 : (n_rays, dim) numpy.ndarray
        Start and end points of each ray, inside the mesh.
    tol : float
        Segments shorter than this are dropped.

    Returns
    -------
    ray_index, cell_index, length : numpy.ndarray
        COO triplets, grouped by ray.
    """
    dim = x0.shape[1]
    # pad to three axes so numba compiles a single specialization
    nodes = tuple(np.ascontiguousarray(n, dtype=np.float64) for n in nodes)
    nodes = nodes + (np.zeros(2),) * (3 - dim)
    shape = np.array(tuple(shape_cells) + (1,) * (3 - dim), dtype=np.int64)
    x0 = np.ascontiguousarray(x0, dtype=np.float64)
    x1 = np.ascontiguousarray(x1, dtype=np.float64)

    capacity = _segment_capacity(nodes, x0, x1)
    offsets = np.zeros(capacity.size + 1, dtype=np.int64)
    np.cumsum(capacity, out=offsets[1:])
    cells = np.empty(offsets[-1], dtype=np.int64)
    lengths = np.empty(offsets[-1], dtype=np.float64)
    counts = np.empty(capacity.size, dtype=np.int64)
    _trace_rays(nodes, shape, x0, x1, tol, offsets, cells, lengths, counts)

    out_offsets = np.zeros(counts.size + 1, dtype=np.int64)
    np.cumsum(counts, out=out_offsets[1:])
    cell_index, length = _compact(offsets, counts, out_offsets, cells, lengths)
    ray_index = np.repeat(np.arange(counts.size), counts)
    return ray_index, cell_index, length


@njit(parallel=True, cache=True)
def _segment_capacity(nodes, x0, x1):
    """Upper bound on the number of segments of each ray."""
    n_rays, dim = x0.shape
    capacity = np.empty(n_rays, dtype=np.int64)
    for r in prange(n_rays):
        # one segment more than the number of interior crossings, plus one
        # spare for a crossing that rounding puts at the far endpoint
        n = 2
        for d in range(dim):
            lo = min(x0[r, d], x1[r, d])
            hi = max(x0[r, d], x1[r, d])
            n += max(
                np.searchsorted(nodes[d], hi, side="left")
                - np.searchsorted(nodes[d], lo, side="right"),
                0,
            )
        capacity[r] = n
    return capacity


@njit(cache=True)
def _next_alpha(axis_nodes, i, x0, dx):
    if i < 0 or i >= axis_nodes.size:
        return np.inf
    return (axis_nodes[i] - x0) / dx


@njit(parallel=True, cache=True)
def _trace_rays(nodes, shape, x0, x1, tol, offsets, cells, lengths, counts):
    n_rays, dim = x0.shape
    for r in prange(n_rays):
        dx = np.zeros(3)
        alpha_next = np.full(3, np.inf)
        node_index = np.zeros(3, dtype=np.int64)
        step = np.zeros(3, dtype=np.int64)
        ray_length = 0.0
        for d in range(dim):
            dx[d] = x1[r, d] - x0[r, d]
            ray_length += dx[d] * dx[d]
            if dx[d] > 0:
                node_index[d] = np.searchsorted(nodes[d], x0[r, d], side="right")
                step[d] = 1
            elif dx[d] < 0:
                node_index[d] = np.searchsorted(nodes[d], x0[r, d], side="left") - 1
                step[d] = -1
            else:
                continue
            alpha_next[d] = _next_alpha(nodes[d], node_index[d], x0[r, d], dx[d])
        ray_length = np.sqrt(ray_length)

        k = offsets[r]
        end = offsets[r + 1]
        alpha_prev = 0.0
        while True:
            alpha = min(alpha_next[0], alpha_next[1], alpha_next[2], 1.0)
            segment = (alpha - alpha_prev) * ray_length
            if segment > tol and k < end:
                alpha_mid = 0.5 * (alpha + alpha_prev)
                cell = 0
                stride = 1
                for d in range(dim):
                    x = x0[r, d] + alpha_mid * dx[d]
                    i = np.searchsorted(nodes[d], x, side="right") - 1
                    i = min(max(i, 0), shape[d] - 1)
                    cell += i * stride
                    stride *= shape[d]
                cells[k] = cell
                lengths[k] = segment
                k += 1
            if alpha >= 1.0:
                break
            # step every axis that crosses here, so corners are handled once
            for d in range(dim):
                if alpha_next[d] == alpha:
                    node_index[d] += step[d]
                    alpha_next[d] = _next_alpha(
                        nodes[d], node_index[d], x0[r, d], dx[d]
                    )
            alpha_prev = alpha
        counts[r] = k - offsets[r]


@njit(parallel=True, cache=True)
def _compact(offsets, counts, out_offsets, cells, lengths):
    n = out_offsets[-1]
    out_cells = np.empty(n, dtype=np.int64)
    out_lengths = np.empty(n, dtype=np.float64)
    for r in prange(counts.size):
        src = offsets[r]
        dst = out_offsets[r]
        for j in range(counts[r]):
            out_cells[dst + j] = cells[src + j]
            out_lengths[dst + j] = lengths[src + j]
    return out_cells, out_lengths
