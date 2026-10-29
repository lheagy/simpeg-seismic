r"""
Source wavelets in time and their spectra.

Waveform objects follow SimPEG's waveform interface: ``eval(time)`` gives the
source time function. They also give its spectrum, ``spectrum(frequency)``,
so the same object can drive a frequency-domain or a time-domain simulation.

Spectra use SimPEG's :math:`e^{i\omega t}` convention,
:math:`W(f) = \int w(t) e^{-i 2 \pi f t} dt`, so a frequency-domain
simulation with source amplitudes ``ricker_spectrum(f, peak_frequency)`` is
the Fourier transform of a time-domain simulation with ``ricker(t,
peak_frequency)``.
"""

import numpy as np
from simpeg.utils import validate_float


def ricker(t, peak_frequency, delay=None):
    r"""Ricker wavelet (second derivative of a Gaussian, peak value 1).

    .. math::
        w(t) = (1 - 2 \pi^2 f_p^2 \tau^2) e^{-\pi^2 f_p^2 \tau^2},
        \quad \tau = t - t_0

    Parameters
    ----------
    t : numpy.ndarray
        Times (s).
    peak_frequency : float
        Peak frequency :math:`f_p` (Hz).
    delay : float, optional
        Time of the peak :math:`t_0`. Defaults to ``1.5 / peak_frequency``,
        which makes the wavelet effectively causal.

    Returns
    -------
    numpy.ndarray
    """
    if delay is None:
        delay = 1.5 / peak_frequency
    a = (np.pi * peak_frequency * (np.asarray(t) - delay)) ** 2
    return (1 - 2 * a) * np.exp(-a)


def ricker_spectrum(frequency, peak_frequency, delay=None):
    r"""Fourier transform of :func:`ricker`.

    .. math::
        W(f) = \frac{2 f^2}{\sqrt{\pi} f_p^3} e^{-f^2 / f_p^2}
        e^{-i 2 \pi f t_0}

    Parameters
    ----------
    frequency : float or numpy.ndarray
        Frequencies (Hz).
    peak_frequency : float
        Peak frequency :math:`f_p` (Hz).
    delay : float, optional
        Time of the peak; see :func:`ricker`.

    Returns
    -------
    complex or numpy.ndarray of complex
    """
    if delay is None:
        delay = 1.5 / peak_frequency
    f = np.asarray(frequency, dtype=float)
    amplitude = 2 * f**2 / (np.sqrt(np.pi) * peak_frequency**3)
    out = (
        amplitude
        * np.exp(-((f / peak_frequency) ** 2))
        * np.exp(-2j * np.pi * f * delay)
    )
    return out[()] if out.ndim == 0 else out


class BaseWaveform:
    """Base class for source waveforms.

    Subclasses implement :meth:`eval` (the source time function) and
    :meth:`spectrum` (its Fourier transform).
    """

    def eval(self, time):  # noqa: A003
        """Source time function.

        Parameters
        ----------
        time : float or numpy.ndarray
            Times (s).

        Returns
        -------
        float or numpy.ndarray
        """
        raise NotImplementedError(f"eval is not implemented for {type(self).__name__}")

    def spectrum(self, frequency):
        """Fourier transform of the source time function.

        Parameters
        ----------
        frequency : float or numpy.ndarray
            Frequencies (Hz).

        Returns
        -------
        complex or numpy.ndarray of complex
        """
        raise NotImplementedError(
            f"spectrum is not implemented for {type(self).__name__}"
        )


class RickerWaveform(BaseWaveform):
    """Ricker wavelet; see :func:`ricker` and :func:`ricker_spectrum`.

    Parameters
    ----------
    peak_frequency : float
        Peak frequency (Hz).
    delay : float, optional
        Time of the peak (s). Defaults to ``1.5 / peak_frequency``.
    """

    def __init__(self, peak_frequency, delay=None):
        self.peak_frequency = peak_frequency
        self.delay = delay

    @property
    def peak_frequency(self):
        """Peak frequency (Hz).

        Returns
        -------
        float
        """
        return self._peak_frequency

    @peak_frequency.setter
    def peak_frequency(self, value):
        self._peak_frequency = validate_float(
            "peak_frequency", value, min_val=0.0, inclusive_min=False
        )

    @property
    def delay(self):
        """Time of the peak (s).

        Returns
        -------
        float
        """
        return self._delay

    @delay.setter
    def delay(self, value):
        if value is None:
            value = 1.5 / self.peak_frequency
        self._delay = validate_float("delay", value)

    def eval(self, time):  # noqa: A003
        # Docstring inherited from BaseWaveform.
        return ricker(time, self.peak_frequency, delay=self.delay)

    def spectrum(self, frequency):
        # Docstring inherited from BaseWaveform.
        return ricker_spectrum(frequency, self.peak_frequency, delay=self.delay)
