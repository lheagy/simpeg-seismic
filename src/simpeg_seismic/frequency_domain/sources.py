import numpy as np
from simpeg.survey import BaseSrc
from simpeg.utils import validate_float, validate_type


class Point(BaseSrc):
    """Acoustic point source at one frequency.

    In 2D this is a line source (invariant out of the plane).

    Parameters
    ----------
    receiver_list : list of simpeg_seismic.frequency_domain.receivers.Pressure
        Receivers that record this source.
    location : (dim,) numpy.ndarray
        Location of the source.
    frequency : float
        Frequency (Hz).
    amplitude : complex, optional
        Source spectrum at this frequency, e.g. from
        :func:`simpeg_seismic.utils.wavelets.ricker_spectrum`.
    """

    def __init__(
        self, receiver_list=None, location=None, frequency=None, amplitude=1.0
    ):
        if location is None:
            raise TypeError("Point requires a location.")
        if frequency is None:
            raise TypeError("Point requires a frequency.")
        super().__init__(receiver_list=receiver_list, location=location)
        self.frequency = frequency
        self.amplitude = amplitude

    @property
    def frequency(self):
        """Frequency (Hz).

        Returns
        -------
        float
        """
        return self._frequency

    @frequency.setter
    def frequency(self, value):
        self._frequency = validate_float(
            "frequency", value, min_val=0.0, inclusive_min=False
        )

    @property
    def amplitude(self):
        """Source spectrum at this frequency.

        Returns
        -------
        complex
        """
        return self._amplitude

    @amplitude.setter
    def amplitude(self, value):
        self._amplitude = complex(
            validate_type("amplitude", value, (complex, float, int, np.number))
        )

    def evaluate(self, simulation):
        """Right-hand side on the nodes.

        The amplitude is spread to the nodes with the transpose of linear
        interpolation, so the discrete source integrates to ``amplitude``.

        Parameters
        ----------
        simulation : simpeg_seismic.frequency_domain.SimulationAcoustic

        Returns
        -------
        (n_nodes,) numpy.ndarray of complex
        """
        P = simulation.mesh.get_interpolation_matrix(
            np.atleast_2d(self.location), location_type="nodes"
        )
        return self.amplitude * np.asarray(P.T.todense()).ravel()
