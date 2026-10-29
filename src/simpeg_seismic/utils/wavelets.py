r"""
Source wavelets in time and their spectra.

Spectra use SimPEG's :math:`e^{i\omega t}` convention,
:math:`W(f) = \int w(t) e^{-i 2 \pi f t} dt`, so a frequency-domain
simulation with source amplitudes ``ricker_spectrum(f, peak_frequency)`` is
the Fourier transform of a time-domain simulation with ``ricker(t,
peak_frequency)``.
"""

import numpy as np


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
