import numpy as np
from simpeg.survey import BaseRx
from simpeg.utils import validate_string


class Pressure(BaseRx):
    """Pressure receivers.

    The data are the real or imaginary part of the pressure, interpolated
    linearly from the nodes, as for the frequency-domain EM receivers in
    SimPEG.

    Parameters
    ----------
    locations : (n_loc, dim) numpy.ndarray
        Receiver locations.
    component : {"real", "imag"}
        Part of the complex pressure that is recorded.
    """

    def __init__(self, locations, component="real", **kwargs):
        super().__init__(locations, **kwargs)
        self.component = component

    @property
    def component(self):
        """Part of the complex pressure that is recorded.

        Returns
        -------
        str
            "real" or "imag".
        """
        return self._component

    @component.setter
    def component(self, value):
        self._component = validate_string("component", value, ("real", "imag"))

    def eval(self, src, mesh, f):  # noqa: A003
        """Recorded part of the pressure at the receiver locations.

        Parameters
        ----------
        src : simpeg_seismic.frequency_domain.sources.PointSource
            The source.
        mesh : discretize.TensorMesh
            The mesh.
        f : simpeg_seismic.frequency_domain.FieldsAcoustic
            The fields.

        Returns
        -------
        (n_loc,) numpy.ndarray
        """
        p = self.getP(mesh, "N") @ f[src, "p"][:, 0]
        return getattr(p, self.component)

    def evalDeriv(self, src, mesh, f, du_dm_v=None, v=None, adjoint=False):
        """Derivative of the data with respect to the pressure on the nodes.

        Parameters
        ----------
        src : simpeg_seismic.frequency_domain.sources.PointSource
            The source.
        mesh : discretize.TensorMesh
            The mesh.
        f : simpeg_seismic.frequency_domain.FieldsAcoustic
            The fields (not used; kept for SimPEG's receiver signature).
        du_dm_v : (n_nodes,) numpy.ndarray of complex, optional
            Change in pressure; required when ``adjoint`` is False.
        v : (n_loc,) numpy.ndarray, optional
            Vector on the data; required when ``adjoint`` is True.
        adjoint : bool, optional
            Apply the transpose.

        Returns
        -------
        numpy.ndarray
            Real data when ``adjoint`` is False. When True, the complex vector
            on the nodes ``P^T c v`` with ``c = 1`` for the real part and
            ``c = -i`` for the imaginary part (``Im(z) = Re(-i z)``); the
            caller takes the real part of the final gradient.
        """
        P = self.getP(mesh, "N")
        if not adjoint:
            return getattr(P @ du_dm_v, self.component)
        weight = 1.0 if self.component == "real" else -1j
        return P.T @ (weight * np.asarray(v, dtype=float))
