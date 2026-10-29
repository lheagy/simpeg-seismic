import numpy as np

from simpeg_seismic.utils import wavelets


def test_ricker_peak():
    t = np.linspace(0, 0.2, 2001)
    w = wavelets.ricker(t, 25.0)
    assert np.isclose(t[np.argmax(w)], 1.5 / 25.0)
    assert np.isclose(np.max(w), 1.0)


def test_ricker_spectrum_matches_fourier_transform():
    """W(f) = integral of w(t) exp(-i 2 pi f t) dt."""
    peak = 20.0
    dt = 1e-4
    t = np.arange(0, 0.5, dt)
    w = wavelets.ricker(t, peak)
    for f in (5.0, 20.0, 40.0):
        numerical = np.sum(w * np.exp(-2j * np.pi * f * t)) * dt
        np.testing.assert_allclose(
            wavelets.ricker_spectrum(f, peak), numerical, rtol=1e-6, atol=1e-9
        )
    assert wavelets.ricker_spectrum(np.r_[5.0, 20.0], peak).shape == (2,)
