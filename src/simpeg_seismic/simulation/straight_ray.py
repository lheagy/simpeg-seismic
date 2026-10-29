import warnings

import numpy as np
import scipy.sparse as sp

from simpeg.utils import validate_active_indices

from ..utils import ray_cell_length_matrix
from .base import BaseTraveltimeSimulation


class SimulationStraightRay(BaseTraveltimeSimulation):
    r"""Straight-ray traveltime simulation.

    The traveltime of each datum is the integral of slowness along the
    straight line from the source to the receiver,

    .. math::
        t_i = \int_{\text{ray}_i} s(\mathbf{x}) \, dl
        \quad \Rightarrow \quad \mathbf{d} = \mathbf{G} \, \mathbf{s},

    where :math:`G_{ij}` is the length of ray :math:`i` in cell :math:`j`.

    Parameters
    ----------
    mesh : discretize.TensorMesh or discretize.TreeMesh
        A 2D or 3D mesh that contains every source and receiver.
    survey : simpeg_seismic.Survey
        The survey.
    slowness : numpy.ndarray, optional
        Slowness (s/m) on the active cells.
    slownessMap : simpeg.maps.IdentityMap, optional
        Mapping from the model to slowness on the active cells. Use a
        :class:`simpeg.maps.ReciprocalMap` to invert for velocity.
    active_cells : numpy.ndarray of bool or int, optional
        Cells that are part of the model. Defaults to all cells. As for other
        integral simulations in SimPEG, the model lives on the active cells
        only and ``G`` has one column per active cell.
    """

    def __init__(
        self,
        mesh,
        survey=None,
        slowness=None,
        slownessMap=None,
        active_cells=None,
        **kwargs,
    ):
        super().__init__(
            mesh, survey=survey, slowness=slowness, slownessMap=slownessMap, **kwargs
        )
        self.active_cells = active_cells

    @property
    def active_cells(self):
        """Active cells of the mesh.

        Returns
        -------
        (n_cells,) numpy.ndarray of bool
        """
        return self._active_cells

    @active_cells.setter
    def active_cells(self, value):
        if value is None:
            value = np.ones(self.mesh.n_cells, dtype=bool)
        self._active_cells = validate_active_indices(
            "active_cells", value, self.mesh.n_cells
        )
        self._reset()

    def _reset(self):
        self._G = None

    @property
    def G(self):
        """Ray length in each active cell.

        Returns
        -------
        (nD, n_active_cells) scipy.sparse.csr_matrix
        """
        if getattr(self, "_G", None) is None:
            if self.survey is None or self.survey.nD == 0:
                return None
            source_locations, receiver_locations = self.survey.ray_endpoints
            G = ray_cell_length_matrix(self.mesh, source_locations, receiver_locations)
            if not np.all(self.active_cells):
                n_through_inactive = np.count_nonzero(
                    G[:, ~self.active_cells].getnnz(axis=1)
                )
                if n_through_inactive:
                    warnings.warn(
                        f"{n_through_inactive} ray(s) pass through inactive cells. "
                        "Their length in those cells is ignored.",
                        stacklevel=2,
                    )
                G = G[:, self.active_cells]
            self._G = G.tocsr()
        return self._G

    def fields(self, m):
        # Docstring inherited from BaseSimulation.
        self.model = m
        return self.G @ self.slowness

    def getJ(self, m, f=None):
        r"""Sensitivity matrix :math:`\mathbf{G} \, \partial \mathbf{s} / \partial \mathbf{m}`.

        Parameters
        ----------
        m : (n_param,) numpy.ndarray
            The model parameters.
        f : Ignored
            Not used, present for API consistency.

        Returns
        -------
        (nD, n_param) scipy.sparse matrix
        """
        self.model = m
        return self.G @ self.slownessDeriv

    def Jvec(self, m, v, f=None):
        # Docstring inherited from BaseSimulation.
        self.model = m
        return self.G @ (self.slownessDeriv @ v)

    def Jtvec(self, m, v, f=None):
        # Docstring inherited from BaseSimulation.
        self.model = m
        return self.slownessDeriv.T @ (self.G.T @ v)

    def getJtJdiag(self, m, W=None, f=None):
        r"""Diagonal of :math:`\mathbf{J}^T \mathbf{W}^T \mathbf{W} \mathbf{J}`.

        Parameters
        ----------
        m : (n_param,) numpy.ndarray
            The model parameters.
        W : (nD, nD) scipy.sparse matrix, optional
            Diagonal matrix with the square root of the data weights.
        f : Ignored
            Not used, present for API consistency.

        Returns
        -------
        (n_param,) numpy.ndarray
        """
        J = sp.csr_matrix(self.getJ(m))
        if W is not None:
            J = sp.csr_matrix(W @ J)
        return np.asarray(J.multiply(J).sum(axis=0)).ravel()
