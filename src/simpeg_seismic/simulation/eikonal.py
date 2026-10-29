import warnings

import numpy as np
import scipy.sparse as sp
from discretize import TensorMesh
from simpeg.utils import validate_float, validate_integer, validate_type

from .base import BaseTraveltimeSimulation


def _kernels():
    try:
        from ..utils import _eikonal_numba
    except ImportError as err:
        raise ImportError("SimulationEikonal requires numba.") from err
    return _eikonal_numba


class SimulationEikonal(BaseTraveltimeSimulation):
    r"""First-arrival traveltime simulation from the eikonal equation.

    Traveltimes :math:`T` from each source satisfy

    .. math::
        |\nabla T(\mathbf{x})| = s(\mathbf{x}),

    discretized on the mesh nodes with a first-order upwind scheme and solved
    by fast sweeping. Slowness is defined on cells; each node takes the mean
    slowness of the cells touching it.

    By default the traveltime is factored as :math:`T = D \tau`, with
    :math:`D` the distance to the source, and the scheme solves for
    :math:`\tau` (Fomel, Luo & Zhao, 2009). This removes the large error
    near the source caused by the curvature of the wavefront, and is exact in
    a homogeneous medium. The corners of the cell containing a source are set
    to the straight-ray time ``s_cell * distance``. At the receivers,
    :math:`\tau` is interpolated linearly from the nodes and multiplied by
    the exact distance.

    Sensitivities are the exact derivatives of the discrete forward problem,
    computed with the adjoint-state method: one triangular solve per source
    for ``Jvec`` and ``Jtvec``. The upwind dependencies have a few small cycles
    next to the planes through the source; those nodes are solved together in
    small dense systems.

    Parameters
    ----------
    mesh : discretize.TensorMesh
        A 2D or 3D tensor mesh that contains every source and receiver.
    survey : simpeg_seismic.Survey
        The survey.
    slowness : numpy.ndarray, optional
        Slowness (s/m) on the cells.
    slownessMap : simpeg.maps.IdentityMap, optional
        Mapping from the model to slowness on the cells. Use a
        :class:`simpeg.maps.ReciprocalMap` to invert for velocity.
    tolerance : float, optional
        Fast sweeping stops when the largest change in an iteration is below
        ``tolerance`` times the largest traveltime.
    max_iterations : int, optional
        Maximum number of fast sweeping iterations. Each iteration sweeps the
        mesh in all ``2**dim`` orderings.
    factored : bool, optional
        Solve the factored eikonal equation. With ``False``, solve for the
        traveltime directly (Godunov's upwind scheme), whose error converges
        like ``h log(h)`` because of the point source.

    Notes
    -----
    Requires numba.
    """

    _mesh_types = (TensorMesh,)

    def __init__(
        self,
        mesh,
        survey=None,
        slowness=None,
        slownessMap=None,
        tolerance=1e-10,
        max_iterations=100,
        factored=True,
        **kwargs,
    ):
        _kernels()
        super().__init__(
            mesh,
            survey=survey,
            slowness=slowness,
            slownessMap=slownessMap,
            **kwargs,
        )
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.factored = factored

    @property
    def tolerance(self):
        """Relative convergence tolerance of fast sweeping.

        Returns
        -------
        float
        """
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value):
        self._tolerance = validate_float("tolerance", value, min_val=0.0)

    @property
    def max_iterations(self):
        """Maximum number of fast sweeping iterations.

        Returns
        -------
        int
        """
        return self._max_iterations

    @max_iterations.setter
    def max_iterations(self, value):
        self._max_iterations = validate_integer("max_iterations", value, min_val=1)

    @property
    def factored(self):
        """Whether the factored eikonal equation is solved.

        Returns
        -------
        bool
        """
        return self._factored

    @factored.setter
    def factored(self, value):
        self._factored = validate_type("factored", value, bool)
        self._reset()

    def _reset(self):
        self._source_geometry = None
        self._projection = None
        self._cell_to_node = None

    @property
    def cell_to_node(self):
        """Averaging from cells to nodes.

        Each node gets the mean slowness of the cells that touch it.

        Returns
        -------
        (n_nodes, n_cells) scipy.sparse.csr_matrix
        """
        if getattr(self, "_cell_to_node", None) is None:
            touching = (self.mesh.average_node_to_cell.T != 0).astype(float)
            n_touching = np.asarray(touching.sum(axis=1)).ravel()
            self._cell_to_node = sp.diags(1.0 / n_touching) @ touching.tocsr()
        return self._cell_to_node

    @property
    def _nodes(self):
        nodes = [
            self.mesh.origin[d] + np.r_[0.0, np.cumsum(self.mesh.h[d])]
            for d in range(self.mesh.dim)
        ]
        if self.mesh.dim == 2:
            nodes.append(np.zeros(1))
        return tuple(nodes)

    def _check_inside(self, name, points):
        lo = self.mesh.origin
        hi = lo + np.array([np.sum(h) for h in self.mesh.h])
        outside = np.any((points < lo) | (points > hi), axis=1)
        if np.any(outside):
            raise ValueError(
                f"{outside.sum()} {name} location(s) are outside of the mesh."
            )

    @property
    def source_geometry(self):
        """Fixed nodes around each source.

        Returns
        -------
        dict
            ``source_xyz`` (n_sources, 3) source locations padded to 3D,
            ``fixed_ptr`` (n_sources + 1,), and per fixed node ``fixed_idx``
            (node index), ``fixed_cell`` (cell containing the source) and
            ``fixed_coefficient``, with the fixed value equal to
            ``slowness[fixed_cell] * fixed_coefficient``.
        """
        if getattr(self, "_source_geometry", None) is None:
            mesh = self.mesh
            locations = np.vstack([src.location for src in self.survey.source_list])
            self._check_inside("source", locations)
            nodes = self._nodes
            shape_nodes = [n.size for n in nodes]

            cell_sub = np.column_stack(
                [
                    np.clip(
                        np.searchsorted(nodes[d], locations[:, d], side="right") - 1,
                        0,
                        mesh.shape_cells[d] - 1,
                    )
                    for d in range(mesh.dim)
                ]
            )
            cells = np.ravel_multi_index(cell_sub.T, mesh.shape_cells, order="F")

            corners = np.array(np.meshgrid(*[[0, 1]] * mesh.dim, indexing="ij"))
            corners = corners.reshape(mesh.dim, -1).T  # (2**dim, dim)
            node_sub = cell_sub[:, None, :] + corners[None, :, :]
            fixed_idx = np.ravel_multi_index(
                tuple(node_sub[..., d] for d in range(mesh.dim)),
                shape_nodes[: mesh.dim],
                order="F",
            )
            corner_xyz = np.stack(
                [nodes[d][node_sub[..., d]] for d in range(mesh.dim)], axis=-1
            )
            distance = np.linalg.norm(corner_xyz - locations[:, None, :], axis=-1)

            n_corners = corners.shape[0]
            source_xyz = np.zeros((len(locations), 3))
            source_xyz[:, : mesh.dim] = locations
            self._source_geometry = {
                "source_xyz": source_xyz,
                "fixed_ptr": np.arange(len(locations) + 1, dtype=np.int64) * n_corners,
                "fixed_idx": fixed_idx.ravel().astype(np.int64),
                "fixed_cell": np.repeat(cells, n_corners),
                # tau = T / D = slowness at the fixed nodes when factored
                "fixed_coefficient": (
                    np.ones(distance.size) if self.factored else distance.ravel()
                ),
            }
        return self._source_geometry

    @property
    def projection(self):
        """Interpolation from the nodes of every source to the data.

        When factored, interpolates ``tau`` and multiplies by the distance
        from the receiver to the source.

        Returns
        -------
        (nD, n_sources * n_nodes) scipy.sparse.csr_matrix
            Block diagonal, one block per source.
        """
        if getattr(self, "_projection", None) is None:
            blocks = []
            for src in self.survey.source_list:
                locations = np.vstack(
                    [np.atleast_2d(rx.locations) for rx in src.receiver_list]
                )
                self._check_inside("receiver", locations)
                P = self.mesh.get_interpolation_matrix(locations, location_type="nodes")
                if self.factored:
                    distance = np.linalg.norm(locations - src.location, axis=1)
                    P = sp.diags(distance) @ P
                blocks.append(P)
            self._projection = sp.block_diag(blocks, format="csr")
        return self._projection

    def _check_active_cells(self):
        if not np.all(self.active_cells):
            raise NotImplementedError(
                "SimulationEikonal does not support inactive cells yet."
            )

    def fields(self, m):
        """Traveltimes on the nodes for every source.

        Parameters
        ----------
        m : (n_param,) numpy.ndarray
            The model.

        Returns
        -------
        dict
            ``T`` (n_sources, n_nodes) traveltimes; ``tau`` the quantity
            solved for (``T`` divided by the distance to the source when
            factored, otherwise ``T``); ``branch`` the upwind branch of
            each node; ``order`` and ``component_start``, the nodes grouped
            into strongly connected components of the upwind dependencies,
            dependencies first; and ``slowness_nodes`` (n_nodes,).
        """
        self._check_active_cells()
        self.model = m
        geometry = self.source_geometry
        slowness_nodes = self.cell_to_node @ self.slowness
        fixed_val = self.slowness[geometry["fixed_cell"]] * geometry["fixed_coefficient"]
        tau, T, branch, order, component_start, n_iterations, _ = _kernels().solve(
            slowness_nodes,
            *self._nodes,
            self.mesh.dim,
            geometry["source_xyz"],
            self.factored,
            geometry["fixed_ptr"],
            geometry["fixed_idx"],
            fixed_val,
            self.tolerance,
            self.max_iterations,
        )
        if np.any(n_iterations >= self.max_iterations):
            warnings.warn(
                f"Fast sweeping did not converge in {self.max_iterations} "
                f"iterations for {np.sum(n_iterations >= self.max_iterations)} "
                "source(s).",
                stacklevel=2,
            )
        return {
            "tau": tau,
            "T": T,
            "branch": branch,
            "order": order,
            "component_start": component_start,
            "slowness_nodes": slowness_nodes,
        }

    def dpred(self, m=None, f=None):
        # Docstring inherited from BaseSimulation.
        if f is None:
            f = self.fields(self.model if m is None else m)
        return self.projection @ f["tau"].ravel()

    def _tangent(self, f, ds):
        geometry = self.source_geometry
        dtau = _kernels().tangent(
            f["tau"],
            f["branch"],
            f["order"],
            f["component_start"],
            f["slowness_nodes"],
            self.cell_to_node @ ds,
            *self._nodes,
            geometry["source_xyz"],
            self.factored,
            geometry["fixed_ptr"],
            geometry["fixed_idx"],
            ds[geometry["fixed_cell"]] * geometry["fixed_coefficient"],
        )
        return self.projection @ dtau.ravel()

    def _adjoint(self, f, g, row_source):
        """Gradient on the cells for each row of ``g`` (on the nodes of ``row_source``)."""
        geometry = self.source_geometry
        grad_nodes, fixed_adjoint = _kernels().adjoint(
            f["tau"],
            f["branch"],
            f["order"],
            f["component_start"],
            f["slowness_nodes"],
            g,
            row_source,
            *self._nodes,
            geometry["source_xyz"],
            self.factored,
            geometry["fixed_ptr"],
            geometry["fixed_idx"],
        )
        grad_cells = (self.cell_to_node.T @ grad_nodes.T).T
        # contribution of the fixed nodes, through the slowness of the source cell
        n_fixed = np.diff(geometry["fixed_ptr"])[row_source]
        rows = np.repeat(np.arange(len(row_source)), n_fixed)
        k = np.concatenate(
            [np.arange(geometry["fixed_ptr"][j], geometry["fixed_ptr"][j + 1]) for j in row_source]
        )
        local = k - geometry["fixed_ptr"][row_source][rows]
        np.add.at(
            grad_cells,
            (rows, geometry["fixed_cell"][k]),
            geometry["fixed_coefficient"][k] * fixed_adjoint[rows, local],
        )
        return grad_cells

    def Jvec(self, m, v, f=None):
        # Docstring inherited from BaseSimulation.
        if f is None:
            f = self.fields(m)
        self.model = m
        return self._tangent(f, self.slownessDeriv @ v)

    def Jtvec(self, m, v, f=None):
        # Docstring inherited from BaseSimulation.
        if f is None:
            f = self.fields(m)
        self.model = m
        n_sources, n_nodes = f["T"].shape
        g = (self.projection.T @ v).reshape(n_sources, n_nodes)
        grad_cells = self._adjoint(f, g, np.arange(n_sources)).sum(axis=0)
        return self.slownessDeriv.T @ grad_cells

    def getJtJdiag(self, m, W=None, f=None, batch_size=64):
        r"""Diagonal of :math:`\mathbf{J}^T \mathbf{W}^T \mathbf{W} \mathbf{J}`.

        Computed exactly from one adjoint solve per datum, in batches.

        Parameters
        ----------
        m : (n_param,) numpy.ndarray
            The model parameters.
        W : (nD, nD) scipy.sparse matrix, optional
            Diagonal matrix with the square root of the data weights.
        f : dict, optional
            Fields from :meth:`fields`.
        batch_size : int, optional
            Number of data handled per adjoint batch.

        Returns
        -------
        (n_param,) numpy.ndarray
        """
        if f is None:
            f = self.fields(m)
        self.model = m
        n_nodes = f["T"].shape[1]
        weights = np.ones(self.survey.nD) if W is None else W.diagonal()
        P = self.projection
        data_source = np.repeat(
            np.arange(len(self.survey.source_list)),
            [src.nD for src in self.survey.source_list],
        )
        deriv = self.slownessDeriv

        diagonal = 0.0
        for start in range(0, self.survey.nD, batch_size):
            rows = P[start : start + batch_size].tocoo()
            g = np.zeros((rows.shape[0], n_nodes))
            g[rows.row, rows.col % n_nodes] = rows.data
            J_cells = self._adjoint(f, g, data_source[start : start + batch_size])
            J = (deriv.T @ J_cells.T).T
            J *= weights[start : start + batch_size, None]
            diagonal = diagonal + np.sum(J**2, axis=0)
        return np.asarray(diagonal).ravel()
