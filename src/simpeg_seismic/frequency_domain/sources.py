import numpy as np
from simpeg.survey import BaseSrc
from simpeg.utils import validate_float, validate_type

from ..utils.wavelets import BaseWaveform


class PointSource(BaseSrc):
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
        Source spectrum at this frequency. Defaults to 1 when no
        ``waveform`` is given.
    waveform : simpeg_seismic.utils.wavelets.BaseWaveform, optional
        Source waveform; the amplitude is then its spectrum at ``frequency``.
        Give either ``amplitude`` or ``waveform``, not both.
    """

    def __init__(
        self,
        receiver_list=None,
        location=None,
        frequency=None,
        amplitude=None,
        waveform=None,
    ):
        if location is None:
            raise TypeError("PointSource requires a location.")
        if frequency is None:
            raise TypeError("PointSource requires a frequency.")
        if amplitude is not None and waveform is not None:
            raise ValueError("Give either amplitude or waveform, not both.")
        super().__init__(receiver_list=receiver_list, location=location)
        self.frequency = frequency
        self.waveform = waveform
        if waveform is None:
            self.amplitude = 1.0 if amplitude is None else amplitude

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
    def waveform(self):
        """Source waveform, if the amplitude comes from one.

        Returns
        -------
        simpeg_seismic.utils.wavelets.BaseWaveform or None
        """
        return self._waveform

    @waveform.setter
    def waveform(self, value):
        if value is not None:
            value = validate_type("waveform", value, BaseWaveform, cast=False)
        self._waveform = value

    @property
    def amplitude(self):
        """Source spectrum at this frequency.

        The waveform's spectrum at ``frequency`` when a waveform is set.

        Returns
        -------
        complex
        """
        if self.waveform is not None:
            return complex(self.waveform.spectrum(self.frequency))
        return self._amplitude

    @amplitude.setter
    def amplitude(self, value):
        if self.waveform is not None:
            raise ValueError(
                "The amplitude comes from the waveform; set waveform=None first."
            )
        self._amplitude = complex(
            validate_type("amplitude", value, (complex, float, int, np.number))
        )

    def eval(self, simulation):  # noqa: A003
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
