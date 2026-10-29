import warnings

import numpy as np
import scipy.sparse as sp
from discretize import TensorMesh
from simpeg import props
from simpeg.base import BasePDESimulation
from simpeg.utils import validate_float, validate_integer, validate_type

from .. import acoustic
from ..utils._mesh import check_points_inside
from .fields import FieldsAcoustic
from .survey import Survey

# below this many nodes per wavelength the second-order scheme is dispersive
_MIN_POINTS_PER_WAVELENGTH = 10


class SimulationAcoustic(BasePDESimulation):
    r"""Frequency-domain constant-density acoustic simulation (Helmholtz).

    Solves

    .. math::
        -\nabla^2 p - \omega^2 s^2 p = q

    for the pressure :math:`p` on the nodes of a 2D or 3D tensor mesh, with
    slowness :math:`s` on the cells and SimPEG's :math:`e^{i\omega t}` time
    convention; see :mod:`simpeg_seismic.acoustic` for the discretization.
    In 2D, sources are line sources.

    The system matrix is factored once per frequency and all sources at that
    frequency are solved together. It is complex symmetric, so the adjoint
    solves of ``Jtvec`` reuse the same factorization. Factorizations are kept
    until the model changes.

    Parameters
    ----------
    mesh : discretize.TensorMesh
        A 2D or 3D tensor mesh. The PML occupies its outer ``pml_cells`` cells.
    survey : simpeg_seismic.frequency_domain.Survey
        The survey.
    slowness : numpy.ndarray, optional
        Slowness (s/m) on the cells.
    slownessMap : simpeg.maps.IdentityMap, optional
        Mapping from the model to slowness on the cells, e.g.
        ``maps.ReciprocalMap() * maps.ExpMap()`` for log-velocity.
    pml_velocity : float
        Reference velocity (m/s) setting the PML damping; typically the
        largest velocity next to the boundary. Required when
        ``pml_cells > 0``. It is fixed, so the PML does not change as the model
        is updated.
    pml_cells : int, optional
        Thickness of the PML in cells on each side. Absorption depends on the
        thickness in wavelengths: aim for about one wavelength at the lowest
        frequency; a warning is raised below half a wavelength.
    pml_reflection : float, optional
        Target reflection coefficient of the PML at normal incidence.
    free_surface : bool, optional
        Pressure-free (p = 0) top boundary, the maximum of the last axis, with
        no PML there.
    solver : type[pymatsolver.solvers.Base], optional
        Solver for the complex symmetric system; defaults to SimPEG's default
        solver.
    solver_opts : dict, optional
        Options passed to the solver.
    """

    slowness, slownessMap, slownessDeriv = props.Invertible("Slowness (s/m)")

    def __init__(
        self,
        mesh,
        survey=None,
        slowness=None,
        slownessMap=None,
        pml_velocity=None,
        pml_cells=10,
        pml_reflection=1e-6,
        free_surface=False,
        **kwargs,
    ):
        super().__init__(mesh=mesh, survey=survey, **kwargs)
        self.slowness = slowness
        self.slownessMap = slownessMap
        self.pml_velocity = pml_velocity
        self.pml_cells = pml_cells
        self.pml_reflection = pml_reflection
        self.free_surface = free_surface

    @BasePDESimulation.mesh.setter
    def mesh(self, value):
        value = validate_type("mesh", value, TensorMesh, cast=False)
        if value.dim not in (2, 3):
            raise ValueError(
                f"{type(self).__name__} mesh must be 2D or 3D, "
                f"received a {value.dim}D mesh."
            )
        self._mesh = value
        self._reset()

    @BasePDESimulation.survey.setter
    def survey(self, value):
        if value is not None:
            value = validate_type("survey", value, Survey, cast=False)
        BasePDESimulation.survey.fset(self, value)
        self._reset()

    @property
    def pml_velocity(self):
        """Reference velocity (m/s) setting the PML damping.

        Returns
        -------
        float or None
        """
        return self._pml_velocity

    @pml_velocity.setter
    def pml_velocity(self, value):
        if value is not None:
            value = validate_float(
                "pml_velocity", value, min_val=0.0, inclusive_min=False
            )
        self._pml_velocity = value
        self._reset()

    @property
    def pml_cells(self):
        """Thickness of the PML in cells on each side.

        Returns
        -------
        int
        """
        return self._pml_cells

    @pml_cells.setter
    def pml_cells(self, value):
        self._pml_cells = validate_integer("pml_cells", value, min_val=0)
        self._reset()

    @property
    def pml_reflection(self):
        """Target reflection coefficient of the PML at normal incidence.

        Returns
        -------
        float
        """
        return self._pml_reflection

    @pml_reflection.setter
    def pml_reflection(self, value):
        self._pml_reflection = validate_float(
            "pml_reflection",
            value,
            min_val=0.0,
            max_val=1.0,
            inclusive_min=False,
            inclusive_max=False,
        )
        self._reset()

    @property
    def free_surface(self):
        """Whether the top boundary is pressure-free.

        Returns
        -------
        bool
        """
        return self._free_surface

    @free_surface.setter
    def free_surface(self, value):
        self._free_surface = validate_type("free_surface", value, bool)
        self._reset()

    def _reset(self):
        """Clear quantities that depend on the mesh, survey or PML settings."""
        self._operators = {}
        self._Ainv = {}
        self._Ainv_slowness = None
        self._pml_sigma = None
        self._free_projection = None
        self._checked_locations = False

    @property
    def _delete_on_model_update(self):
        # the factorizations depend on the model; the operators do not
        return super()._delete_on_model_update + ["_Ainv", "_Ainv_slowness"]

    @property
    def pml_sigma(self):
        """PML damping (1/s) along each axis at the cell centers.

        Returns
        -------
        (n_cells, dim) numpy.ndarray
        """
        if getattr(self, "_pml_sigma", None) is None:
            if self.pml_cells > 0 and self.pml_velocity is None:
                raise ValueError(
                    "pml_velocity is required to set the PML damping when "
                    "pml_cells > 0."
                )
            self._pml_sigma = acoustic.pml_damping(
                self.mesh,
                self.pml_cells,
                self.pml_velocity,
                reflection=self.pml_reflection,
                free_surface=self.free_surface,
            )
        return self._pml_sigma

    @property
    def _free_nodes(self):
        """Selection of the nodes that are solved for (all but a free surface)."""
        if getattr(self, "_free_projection", None) is None:
            keep = np.ones(self.mesh.n_nodes, dtype=bool)
            if self.free_surface:
                top = self.mesh.origin[-1] + np.sum(self.mesh.h[-1])
                keep = ~np.isclose(self.mesh.nodes[:, -1], top)
            self._free_projection = sp.identity(self.mesh.n_nodes, format="csr")[keep]
        return self._free_projection

    def _operators_at(self, frequency):
        """Stiffness and mass-weight matrices at one frequency (model independent)."""
        if frequency not in self._operators:
            gamma = acoustic.stretching(self.pml_sigma, 2 * np.pi * frequency)
            self._operators[frequency] = (
                acoustic.stiffness(self.mesh, gamma),
                acoustic.mass_weights(self.mesh, gamma),
            )
        return self._operators[frequency]

    def getA(self, frequency):
        r"""System matrix at one frequency for the current model.

        Parameters
        ----------
        frequency : float
            Frequency (Hz).

        Returns
        -------
        (n_nodes, n_nodes) scipy.sparse.csr_matrix of complex
            :math:`\mathbf{K} - \omega^2 \operatorname{diag}(\mathbf{B} s^2)`,
            before removing the free-surface nodes.
        """
        K, B = self._operators_at(frequency)
        omega = 2 * np.pi * frequency
        return (K - omega**2 * sp.diags(B @ self.slowness**2)).tocsr()

    def _solve(self, frequency, rhs):
        """Solve ``A x = rhs`` on the free nodes; ``rhs`` is (n_nodes, k)."""
        # SimPEG deletes the cache when the model changes, but it decides that
        # with np.allclose and its default absolute tolerance (1e-8), which
        # misses small changes of a slowness of order 1e-3. So also check the
        # slowness the factorizations were made with.
        factored_for = getattr(self, "_Ainv_slowness", None)
        if (
            getattr(self, "_Ainv", None) is None
            or factored_for is None
            or not np.array_equal(factored_for, self.slowness)
        ):
            self._Ainv = {}
            self._Ainv_slowness = np.array(self.slowness, copy=True)
        if frequency not in self._Ainv:
            F = self._free_nodes
            A = F @ self.getA(frequency) @ F.T
            self._Ainv[frequency] = self.solver(A.tocsc(), **self.solver_opts)
        F = self._free_nodes
        b = F @ rhs
        if b.shape[1] == 1:  # pymatsolver expects a vector for one right-hand side
            b = b[:, 0]
        x = self._Ainv[frequency] * b
        return F.T @ np.asarray(x).reshape(F.shape[0], -1)

    def _check_locations(self):
        """Warn about sources or receivers in the PML or a coarse mesh."""
        if self._checked_locations:
            return
        sources = np.vstack([src.location for src in self.survey.source_list])
        receivers = np.vstack(
            [
                rx.locations
                for src in self.survey.source_list
                for rx in src.receiver_list
            ]
        )
        check_points_inside(self.mesh, "source", sources)
        check_points_inside(self.mesh, "receiver", receivers)
        self._check_pml()
        for name, points in (("source", sources), ("receiver", receivers)):
            n_in = acoustic.pml_inside(
                self.mesh, self.pml_cells, points, free_surface=self.free_surface
            ).sum()
            if n_in:
                warnings.warn(
                    f"{n_in} {name} location(s) are inside the PML.", stacklevel=3
                )
        self._checked_locations = True

    def _check_pml(self):
        """Warn when the PML is thin compared with the longest wavelength."""
        if self.pml_cells == 0:
            return
        self.pml_sigma  # raises if pml_velocity is missing
        longest = self.pml_velocity / min(self.survey.frequencies)
        thinnest = min(
            min(np.sum(h[: self.pml_cells]), np.sum(h[-self.pml_cells :]))
            for h in self.mesh.h
        )
        if thinnest < 0.5 * longest:
            warnings.warn(
                f"The PML is {thinnest:.3g} m thick, less than half the longest "
                f"wavelength ({longest:.3g} m at {min(self.survey.frequencies)} Hz); "
                "reflections from it may be significant. Aim for about one "
                "wavelength.",
                stacklevel=3,
            )

    def _check_sampling(self):
        """Warn when the mesh inside the PML has too few nodes per wavelength.

        Cells in the PML are left out, so the layer can use expanding
        padding cells.
        """
        n = self.pml_cells
        h_max = max(np.max(h[n : len(h) - n]) for h in self.mesh.h)
        interior = ~acoustic.pml_inside(
            self.mesh, n, self.mesh.cell_centers, free_surface=self.free_surface
        )
        v_min = 1.0 / np.max(self.slowness[interior])
        points = v_min / (max(self.survey.frequencies) * h_max)
        if points < _MIN_POINTS_PER_WAVELENGTH:
            warnings.warn(
                f"Only {points:.1f} nodes per shortest wavelength (at "
                f"{max(self.survey.frequencies)} Hz); use at least "
                f"{_MIN_POINTS_PER_WAVELENGTH} to limit numerical dispersion.",
                stacklevel=3,
            )

    def fields(self, m=None):
        """Complex pressure on the nodes for every source.

        Parameters
        ----------
        m : (n_param,) numpy.ndarray, optional
            The model.

        Returns
        -------
        simpeg_seismic.frequency_domain.FieldsAcoustic
        """
        if m is not None:
            self.model = m
        self._check_locations()
        self._check_sampling()
        f = FieldsAcoustic(self)
        for frequency in self.survey.frequencies:
            sources = self.survey.get_sources_by_frequency(frequency)
            rhs = np.column_stack([src.evaluate(self) for src in sources])
            f[sources, "pSolution"] = self._solve(frequency, rhs)
        return f

    def dpred(self, m=None, f=None):
        # Docstring inherited from BaseSimulation.
        if f is None:
            f = self.fields(m)
        data = np.empty(self.survey.nD)
        for src in self.survey.source_list:
            for rx in src.receiver_list:
                data[self.survey.get_slice(src, rx)] = rx.eval(src, self.mesh, f)
        return data

    def _mass_derivative(self, frequency):
        r"""Matrix taking a slowness change to the change in ``diag(A)`` per node.

        ``A = K - w^2 diag(B s^2)``, so ``dA[u] = -w^2 diag(u) B diag(2 s) ds``.
        Returns ``w^2 B diag(2 s)``; the change in ``A u`` is then
        ``-u * (this @ ds)``.
        """
        _, B = self._operators_at(frequency)
        omega = 2 * np.pi * frequency
        return omega**2 * B @ sp.diags(2 * self.slowness)

    def Jvec(self, m, v, f=None):
        # Docstring inherited from BaseSimulation.
        if f is None:
            f = self.fields(m)
        self.model = m
        ds = self.slownessDeriv @ v
        out = np.empty(self.survey.nD)
        for frequency in self.survey.frequencies:
            sources = self.survey.get_sources_by_frequency(frequency)
            u = f[sources, "pSolution"]
            # A du = -dA[u] = u * (w^2 B 2s ds)
            dmass = self._mass_derivative(frequency) @ ds
            du = self._solve(frequency, u * dmass[:, None])
            for j, src in enumerate(sources):
                for rx in src.receiver_list:
                    out[self.survey.get_slice(src, rx)] = rx.evalDeriv(
                        src, self.mesh, f, du_dm_v=du[:, j]
                    )
        return out

    def Jtvec(self, m, v, f=None):
        # Docstring inherited from BaseSimulation.
        if f is None:
            f = self.fields(m)
        self.model = m
        gradient = np.zeros(self.mesh.n_cells)
        for frequency in self.survey.frequencies:
            sources = self.survey.get_sources_by_frequency(frequency)
            g = np.zeros((self.mesh.n_nodes, len(sources)), dtype=complex)
            for j, src in enumerate(sources):
                for rx in src.receiver_list:
                    g[:, j] += rx.evalDeriv(
                        src,
                        self.mesh,
                        f,
                        v=v[self.survey.get_slice(src, rx)],
                        adjoint=True,
                    )
            # A is complex symmetric, so A^-T = A^-1: reuse the factorization
            adjoint_field = self._solve(frequency, g)
            u = f[sources, "pSolution"]
            gradient += np.real(
                self._mass_derivative(frequency).T @ np.sum(u * adjoint_field, axis=1)
            )
        return self.slownessDeriv.T @ gradient
