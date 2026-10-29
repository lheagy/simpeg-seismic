import warnings

import numpy as np
import scipy.sparse as sp
from discretize import TensorMesh
from simpeg.fields import Fields
from simpeg.utils import validate_float, validate_integer, validate_type

from ..utils._mesh import axis_nodes, check_points_inside
from .base import BaseTraveltimeSimulation


def _kernels():
    try:
        from ..utils import _eikonal_numba
    except ImportError as err:
        raise ImportError("SimulationEikonal requires numba.") from err
    return _eikonal_numba


class EikonalFields(Fields):
    r"""Fields of :class:`SimulationEikonal`.

    Stores the solution ``tauSolution`` on the nodes, one column per source.
    When the simulation is factored the traveltime is :math:`T = D \tau`,
    with :math:`D` the distance from the node to the source; otherwise
    :math:`\tau` is the traveltime.

    Fields that can be extracted, each of shape (n_nodes, n_sources):

    - ``'T'``: traveltime.
    - ``'tau'``: the quantity solved for.

    .. code-block:: python

        f = simulation.fields(m)
        T = f[:, "T"]
        T_first = f[survey.source_list[0], "T"]

    The upwind structure of the solve, which ``Jvec`` and ``Jtvec`` reuse, is
    kept in the attributes below. Arrays with a source dimension are
    (n_sources, n_nodes).

    Attributes
    ----------
    slowness_nodes : (n_nodes,) numpy.ndarray
        Slowness on the nodes.
    branch : numpy.ndarray of int8
        Upwind branch code of each node.
    order, component_start : numpy.ndarray
        Nodes grouped into strongly connected components of the upwind
        dependencies, dependencies first, and where each component starts.
    n_iterations : (n_sources,) numpy.ndarray of int
        Fast sweeping iterations for each source.
    """

    knownFields = {"tauSolution": "N"}
    aliasFields = {
        "tau": ["tauSolution", "N", "_tau"],
        "T": ["tauSolution", "N", "_T"],
    }

    def _tau(self, tau_solution, source_list):
        return tau_solution

    def _T(self, tau_solution, source_list):
        if not self.simulation.factored:
            return tau_solution
        locations = np.vstack([src.location for src in source_list])
        distance = np.linalg.norm(
            self.mesh.nodes[:, None, :] - locations[None, :, :], axis=-1
        )
        return distance * tau_solution

    def _store(self, tau, slowness_nodes, branch, order, component_start, n_iterations):
        """Store a solve; ``tau`` is the (n_sources, n_nodes) kernel output."""
        # The kernels work source by source on C-ordered (n_sources, n_nodes)
        # arrays; the transpose is the (n_nodes, n_sources) storage Fields
        # expects, in Fortran order, without a copy.
        self._fields["tauSolution"] = tau.T
        self.slowness_nodes = slowness_nodes
        self.branch = branch
        self.order = order
        self.component_start = component_start
        self.n_iterations = n_iterations

    @property
    def _tau_by_source(self):
        """``tauSolution`` as a C-ordered (n_sources, n_nodes) array for the kernels."""
        return np.ascontiguousarray(self._fields["tauSolution"].T)


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
        Mapping from the model to slowness on all cells of the mesh. Use a
        :class:`simpeg.maps.ReciprocalMap` to invert for velocity. As for other
        PDE simulations in SimPEG, topography is handled through the map: an
        :class:`simpeg.maps.InjectActiveCells` that gives the air cells a
        large fixed slowness keeps first arrivals in the ground.
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
        """Node coordinates along each axis, padded to three axes for 2D."""
        nodes = [axis_nodes(self.mesh, d) for d in range(self.mesh.dim)]
        if self.mesh.dim == 2:
            nodes.append(np.zeros(1))
        return tuple(nodes)

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
            check_points_inside(self.mesh, "source", locations)
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
        """The receivers' projections of every source, as one matrix.

        Applying it to the stored solution of all sources (source by source)
        gives all the data in one sparse product.

        Returns
        -------
        (nD, n_sources * n_nodes) scipy.sparse.csr_matrix
            Block diagonal, one block per source; see
            :meth:`simpeg_seismic.receivers.Traveltime.projection`.
        """
        if getattr(self, "_projection", None) is None:
            blocks = []
            for src in self.survey.source_list:
                for rx in src.receiver_list:
                    check_points_inside(self.mesh, "receiver", rx.locations)
                blocks.append(
                    sp.vstack([rx.projection(src, self) for rx in src.receiver_list])
                )
            self._projection = sp.block_diag(blocks, format="csr")
        return self._projection

    def fields(self, m):
        """Traveltimes on the nodes for every source.

        Parameters
        ----------
        m : (n_param,) numpy.ndarray
            The model.

        Returns
        -------
        EikonalFields
        """
        self.model = m
        geometry = self.source_geometry
        slowness_nodes = self.cell_to_node @ self.slowness
        fixed_val = (
            self.slowness[geometry["fixed_cell"]] * geometry["fixed_coefficient"]
        )
        tau, branch, order, component_start, n_iterations, _ = _kernels().solve(
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
        f = EikonalFields(self)
        f._store(tau, slowness_nodes, branch, order, component_start, n_iterations)
        return f

    def dpred(self, m=None, f=None):
        # Docstring inherited from BaseSimulation.
        if f is None:
            f = self.fields(self.model if m is None else m)
        return self.projection @ f._tau_by_source.ravel()

    def _tangent(self, f, ds):
        geometry = self.source_geometry
        dtau = _kernels().tangent(
            f._tau_by_source,
            f.branch,
            f.order,
            f.component_start,
            f.slowness_nodes,
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
            f._tau_by_source,
            f.branch,
            f.order,
            f.component_start,
            f.slowness_nodes,
            g,
            row_source,
            *self._nodes,
            geometry["source_xyz"],
            self.factored,
            geometry["fixed_ptr"],
            geometry["fixed_idx"],
        )
        grad_cells = (self.cell_to_node.T @ grad_nodes.T).T
        # The fixed nodes are the 2**dim corners of the source cell, and their
        # values are slowness[source_cell] * fixed_coefficient.
        n_corners = 2**self.mesh.dim
        coefficient = geometry["fixed_coefficient"].reshape(-1, n_corners)
        source_cell = geometry["fixed_cell"][::n_corners]
        grad_cells[np.arange(len(row_source)), source_cell[row_source]] += np.sum(
            coefficient[row_source] * fixed_adjoint, axis=1
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
        n_sources = len(self.survey.source_list)
        g = (self.projection.T @ v).reshape(n_sources, self.mesh.n_nodes)
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
        f : EikonalFields, optional
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
        n_nodes = self.mesh.n_nodes
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
