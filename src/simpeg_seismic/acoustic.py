r"""
Discrete operators for the constant-density acoustic wave equation.

Shared by the frequency-domain (and future time-domain) simulations. Pressure
lives on the nodes of a 2D or 3D tensor mesh and slowness on the cells. The
weak form of

.. math::
    -\nabla^2 p - \omega^2 s^2 p = q

discretizes to

.. math::
    \left(\mathbf{G}^T \mathbf{M}_e(\Lambda) \mathbf{G}
    - \omega^2 \mathbf{M}_n(\gamma s^2)\right) \mathbf{p} = \mathbf{q},

with :math:`\mathbf{G}` the nodal gradient, :math:`\mathbf{M}_e` the edge
inner product and :math:`\mathbf{M}_n` a lumped (diagonal) nodal mass matrix:
each cell gives ``volume / 2**dim`` of its weight to each of its corners.

Absorbing boundaries use a perfectly matched layer (PML) by complex
coordinate stretching. With SimPEG's :math:`e^{i\omega t}` convention, the
stretching along axis :math:`d` is :math:`\gamma_d = 1 - i \sigma_d / \omega`,
and multiplying the stretched equation by
:math:`\gamma = \gamma_x \gamma_y \gamma_z` gives a symmetric weak form: the
stiffness along axis :math:`d` is weighted by
:math:`\Lambda_d = \gamma / \gamma_d^2` and the mass by :math:`\gamma`.
"""

import numpy as np
import scipy.sparse as sp


def pml_damping(mesh, n_cells, velocity, reflection=1e-6, free_surface=False):
    r"""Damping profile :math:`\sigma_d` of the PML at the cell centers.

    Quadratic in the distance into the layer,
    :math:`\sigma = \sigma_{max} (\ell / L)^2`, with
    :math:`\sigma_{max} = 3 v \ln(1 / R) / (2 L)` chosen so that a wave at
    normal incidence returns with amplitude ``reflection``.

    Parameters
    ----------
    mesh : discretize.TensorMesh
        The mesh. The PML occupies its outer ``n_cells`` cells on each side.
    n_cells : int
        Thickness of the layer in cells.
    velocity : float
        Reference velocity (m/s) used to set the damping strength.
    reflection : float, optional
        Target reflection coefficient at normal incidence.
    free_surface : bool, optional
        No layer on the top (maximum of the last axis), where a free surface
        is imposed instead.

    Returns
    -------
    (n_cells, dim) numpy.ndarray
        Damping (1/s) along each axis.
    """
    sigma = np.zeros((mesh.n_cells, mesh.dim))
    if n_cells == 0:
        return sigma
    centers = mesh.cell_centers
    for d in range(mesh.dim):
        h = mesh.h[d]
        if 2 * n_cells >= len(h):
            raise ValueError(
                f"The PML ({n_cells} cells on each side) does not fit in the "
                f"{len(h)} cells along axis {d}."
            )
        nodes = mesh.origin[d] + np.r_[0.0, np.cumsum(h)]
        inner_lo = nodes[n_cells]
        inner_hi = nodes[-1 - n_cells]
        depth_lo = np.clip(inner_lo - centers[:, d], 0, None)
        depth_hi = np.clip(centers[:, d] - inner_hi, 0, None)
        if free_surface and d == mesh.dim - 1:
            depth_hi[:] = 0.0
        for depth, thickness in (
            (depth_lo, inner_lo - nodes[0]),
            (depth_hi, nodes[-1] - inner_hi),
        ):
            sigma_max = 3.0 * velocity * np.log(1.0 / reflection) / (2.0 * thickness)
            sigma[:, d] += sigma_max * (depth / thickness) ** 2
    return sigma


def pml_inside(mesh, n_cells, points, free_surface=False):
    """Whether each point lies in the PML.

    Parameters
    ----------
    mesh : discretize.TensorMesh
    n_cells : int
        Thickness of the layer in cells.
    points : (n, dim) numpy.ndarray
    free_surface : bool, optional
        No layer on the top.

    Returns
    -------
    (n,) numpy.ndarray of bool
    """
    inside = np.zeros(len(points), dtype=bool)
    if n_cells == 0:
        return inside
    for d in range(mesh.dim):
        nodes = mesh.origin[d] + np.r_[0.0, np.cumsum(mesh.h[d])]
        inside |= points[:, d] < nodes[n_cells]
        if not (free_surface and d == mesh.dim - 1):
            inside |= points[:, d] > nodes[-1 - n_cells]
    return inside


def stretching(sigma, omega):
    r"""Coordinate stretching :math:`\gamma_d = 1 - i \sigma_d / \omega` per cell.

    Parameters
    ----------
    sigma : (n_cells, dim) numpy.ndarray
        PML damping from :func:`pml_damping`.
    omega : float
        Angular frequency (rad/s).

    Returns
    -------
    (n_cells, dim) numpy.ndarray of complex
    """
    return 1.0 - 1j * sigma / omega


def stiffness(mesh, gamma):
    r"""Stretched stiffness matrix :math:`\mathbf{G}^T \mathbf{M}_e(\Lambda) \mathbf{G}`.

    Parameters
    ----------
    mesh : discretize.TensorMesh
    gamma : (n_cells, dim) numpy.ndarray of complex
        Coordinate stretching from :func:`stretching`.

    Returns
    -------
    (n_nodes, n_nodes) scipy.sparse.csr_matrix of complex
        Complex symmetric.
    """
    weights = np.prod(gamma, axis=1)[:, None] / gamma**2
    G = mesh.nodal_gradient
    return (G.T @ mesh.get_edge_inner_product(model=weights) @ G).tocsr()


def mass_weights(mesh, gamma):
    r"""Matrix taking a cell property to lumped nodal mass weights.

    The diagonal of :math:`\mathbf{M}_n(\gamma m)` is ``B @ m`` for a cell
    property ``m`` (e.g. squared slowness), where ``B`` spreads each cell's
    stretched volume ``gamma * volume`` equally over its corners.

    Parameters
    ----------
    mesh : discretize.TensorMesh
    gamma : (n_cells, dim) numpy.ndarray of complex
        Coordinate stretching from :func:`stretching`.

    Returns
    -------
    B : (n_nodes, n_cells) scipy.sparse.csr_matrix of complex
    """
    weights = mesh.cell_volumes * np.prod(gamma, axis=1)
    return (mesh.average_node_to_cell.T @ sp.diags(weights)).tocsr()
