import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from simpeg_seismic import traveltime


def survey(dim):
    rng = np.random.default_rng(0)
    source_list = [
        traveltime.sources.PointSource(
            receiver_list=[
                traveltime.receivers.Traveltime(rng.uniform(size=(3, dim))),
                traveltime.receivers.Traveltime(rng.uniform(size=(2, dim))),
            ],
            location=rng.uniform(size=dim),
        )
        for _ in range(2)
    ]
    return traveltime.Survey(source_list)


@pytest.mark.parametrize("dim", [2, 3])
def test_ray_endpoints_order(dim):
    s = survey(dim)
    src, rx = s.ray_endpoints
    assert s.nD == 10
    assert src.shape == rx.shape == (s.nD, dim)
    np.testing.assert_equal(src[:5], np.tile(s.source_list[0].location, (5, 1)))
    np.testing.assert_equal(src[5:], np.tile(s.source_list[1].location, (5, 1)))
    np.testing.assert_equal(
        rx[5:],
        np.vstack([r.locations for r in s.source_list[1].receiver_list]),
    )


@pytest.mark.parametrize("dim", [2, 3])
def test_plot(dim):
    ax = survey(dim).plot()
    assert ax.name == ("3d" if dim == 3 else "rectilinear")


def test_point_source_requires_location():
    with pytest.raises(TypeError, match="location"):
        traveltime.sources.PointSource(receiver_list=[])


def test_aliases():
    assert traveltime.Src is traveltime.sources
    assert traveltime.Rx is traveltime.receivers
