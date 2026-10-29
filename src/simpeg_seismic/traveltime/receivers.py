import numpy as np
import scipy.sparse as sp
from simpeg.survey import BaseRx


class Traveltime(BaseRx):
    """First-arrival traveltime receivers.

    :class:`simpeg_seismic.traveltime.SimulationStraightRay` only uses the locations.
    For simulations that solve for traveltimes on the mesh, such as
    :class:`simpeg_seismic.traveltime.SimulationEikonal`, the receiver evaluates the
    fields with :meth:`eval` and :meth:`evalDeriv`.

    Parameters
    ----------
    locations : (n_loc, dim) numpy.ndarray
        Receiver locations. Each location gives one datum per source.
    """

    def projection(self, src, simulation):
        """Linear map from the stored solution of ``src`` to these data.

        The stored solution ``tauSolution`` lives on the nodes. It is
        interpolated linearly to the receiver locations and, when the
        simulation is factored (``T = D tau``), multiplied by the exact
        distance from each receiver to the source.

        Parameters
        ----------
        src : simpeg_seismic.traveltime.sources.PointSource
            The source.
        simulation : simpeg_seismic.traveltime.SimulationEikonal
            The simulation.

        Returns
        -------
        (n_loc, n_nodes) scipy.sparse.csr_matrix
        """
        P = self.getP(simulation.mesh, "N")
        if simulation.factored:
            distance = np.linalg.norm(self.locations - src.location, axis=1)
            P = sp.diags(distance) @ P
        return sp.csr_matrix(P)

    def eval(self, src, mesh, f):  # noqa: A003
        """Traveltimes at the receiver locations.

        Parameters
        ----------
        src : simpeg_seismic.traveltime.sources.PointSource
            The source.
        mesh : discretize.TensorMesh
            The mesh (taken from the fields' simulation; kept for SimPEG's
            receiver signature).
        f : simpeg_seismic.traveltime.simulation.EikonalFields
            The fields.

        Returns
        -------
        (n_loc,) numpy.ndarray
        """
        return self.projection(src, f.simulation) @ f[src, "tauSolution"][:, 0]

    def evalDeriv(self, src, mesh, f, v, adjoint=False):
        """Derivative of :meth:`eval` with respect to the stored solution.

        Parameters
        ----------
        src : simpeg_seismic.traveltime.sources.PointSource
            The source.
        mesh : discretize.TensorMesh
            The mesh (kept for SimPEG's receiver signature).
        f : simpeg_seismic.traveltime.simulation.EikonalFields
            The fields.
        v : numpy.ndarray
            Vector on the nodes, or on the data when ``adjoint`` is True.
        adjoint : bool, optional
            Apply the transpose.

        Returns
        -------
        numpy.ndarray
        """
        P = self.projection(src, f.simulation)
        return P.T @ v if adjoint else P @ v
