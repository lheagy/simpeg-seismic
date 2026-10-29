import warnings

import discretize
import numpy as np
import pytest
from helpers import assert_adjoint, assert_derivative
from scipy.special import hankel2
from simpeg import maps
from simpeg.utils import get_default_solver

from simpeg_seismic import frequency_domain as fd
from simpeg_seismic.utils import wavelets

VELOCITY = 1500.0
FREQUENCY = 30.0  # wavelength 50 m
K = 2 * np.pi * FREQUENCY / VELOCITY


def make_survey(source_locations, receiver_locations, frequencies=(FREQUENCY,)):
    """Every source at every frequency, recording real and imaginary parts."""
    receivers = [
        fd.receivers.Pressure(receiver_locations, component="real"),
        fd.receivers.Pressure(receiver_locations, component="imag"),
    ]
    return fd.Survey(
        [
            fd.sources.PointSource(receivers, location=loc, frequency=f)
            for f in frequencies
            for loc in source_locations
        ]
    )


def simulation(mesh, survey, mapping=None, **kwargs):
    kwargs.setdefault("pml_velocity", VELOCITY)
    kwargs.setdefault("solver", get_default_solver())
    return fd.SimulationAcoustic(
        mesh,
        survey=survey,
        slownessMap=maps.IdentityMap(mesh) if mapping is None else mapping,
        **kwargs,
    )


def complex_data(d, n_rx):
    """Data of one source with a real and an imaginary receiver."""
    return d[:n_rx] + 1j * d[n_rx : 2 * n_rx]


def green(dim, r):
    """Outgoing Green's function of -lap(p) - k^2 p = delta, e^{iwt}."""
    if dim == 3:
        return np.exp(-1j * K * r) / (4 * np.pi * r)
    return -0.25j * hankel2(0, K * r)


def homogeneous_error(dim, h, extent, pml_cells):
    mesh = discretize.TensorMesh(
        [np.full(int(round(extent / h)), h)] * dim, origin="C" * dim
    )
    source = np.full(dim, 0.3)
    rng = np.random.default_rng(0)
    rx = rng.uniform(-50, 50, (12, dim))
    rx = rx[np.linalg.norm(rx - source, axis=1) > 25]
    sim = simulation(mesh, make_survey([source], rx), pml_cells=pml_cells)
    p = complex_data(sim.dpred(np.full(mesh.n_cells, 1 / VELOCITY)), len(rx))
    exact = green(dim, np.linalg.norm(rx - source, axis=1))
    return np.max(np.abs(p - exact) / np.abs(exact))


def test_green_function_2d_second_order():
    # PML one wavelength thick at every resolution
    errors = [homogeneous_error(2, h, 300, int(50 / h)) for h in (2.5, 1.25)]
    assert errors[0] < 0.03
    assert errors[1] < 0.3 * errors[0]


def test_green_function_3d():
    # 10 nodes per wavelength: dispersion error of a few percent
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the PML is half a wavelength thick
        assert homogeneous_error(3, 5.0, 200, 10) < 0.1


def test_pml_absorbs():
    """The same receivers on a small and a much larger mesh agree."""
    h = 2.5
    source = np.array([0.3, 0.3])
    rx = np.array([[40.0, 10.0], [-30.0, 35.0], [5.0, -45.0]])
    data = []
    for extent in (200, 400):
        mesh = discretize.TensorMesh([np.full(int(extent / h), h)] * 2, origin="CC")
        sim = simulation(mesh, make_survey([source], rx), pml_cells=20)
        data.append(complex_data(sim.dpred(np.full(mesh.n_cells, 1 / VELOCITY)), 3))
    np.testing.assert_allclose(data[0], data[1], rtol=5e-3)


def heterogeneous_problem(free_surface=False, mapping="slowness", seed=1):
    """Small 2D problem: 2 sources, 2 frequencies, random model."""
    h = 5.0
    mesh = discretize.TensorMesh([np.full(36, h), np.full(30, h)], origin="CC")
    rng = np.random.default_rng(seed)
    sources = np.array([[-30.0, 10.0], [25.0, -15.0]])
    rx = rng.uniform(-35, 35, (5, 2))
    survey = make_survey(sources, rx, frequencies=(10.0, 15.0))
    velocity = rng.uniform(1400.0, 1800.0, mesh.n_cells)
    if mapping == "slowness":
        sim = simulation(mesh, survey, pml_cells=6, free_surface=free_surface)
        model = 1 / velocity
    else:  # log-velocity
        sim = simulation(
            mesh,
            survey,
            mapping=maps.ReciprocalMap(mesh) * maps.ExpMap(mesh),
            pml_cells=6,
            free_surface=free_surface,
        )
        model = np.log(velocity)
    return sim, model, rng


@pytest.mark.parametrize("free_surface", [False, True])
@pytest.mark.parametrize("mapping", ["slowness", "log_velocity"])
def test_derivative_and_adjoint(free_surface, mapping):
    sim, model, rng = heterogeneous_problem(free_surface, mapping)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # small test mesh: thin PML
        assert_derivative(
            sim, model, dx=1e-2 * np.abs(model) * rng.standard_normal(model.size)
        )
        assert_adjoint(sim, model, rng, f=sim.fields(model))


def test_reciprocity():
    """Swapping source and receiver gives the same datum (A is symmetric)."""
    sim, model, _ = heterogeneous_problem()
    a, b = np.array([-30.0, 10.0]), np.array([22.0, 17.0])
    data = []
    for src, rx in ((a, b), (b, a)):
        swapped = simulation(sim.mesh, make_survey([src], rx[None]), pml_cells=6)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data.append(swapped.dpred(model))
    np.testing.assert_allclose(data[0], data[1], rtol=1e-10)


def test_free_surface_matches_image_source():
    h = 1.25
    mesh = discretize.TensorMesh(
        [np.full(240, h), np.full(160, h)], origin=[-150.0, -200.0]
    )  # top at z = 0
    source = np.array([0.3, -20.0])
    image = np.array([0.3, 20.0])
    rx = np.array([[40.0, -5.0], [-30.0, -40.0], [60.0, -25.0]])
    sim = simulation(mesh, make_survey([source], rx), pml_cells=40, free_surface=True)
    p = complex_data(sim.dpred(np.full(mesh.n_cells, 1 / VELOCITY)), 3)
    exact = green(2, np.linalg.norm(rx - source, axis=1)) - green(
        2, np.linalg.norm(rx - image, axis=1)
    )
    np.testing.assert_allclose(p, exact, rtol=0.02)


def test_fields_and_receivers():
    sim, model, _ = heterogeneous_problem()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = sim.fields(model)
    assert isinstance(f, fd.FieldsAcoustic)
    p = f[:, "p"]
    assert p.shape == (sim.mesh.n_nodes, len(sim.survey.source_list))
    assert np.iscomplexobj(p)
    src = sim.survey.source_list[1]
    rx_real, rx_imag = src.receiver_list
    P = rx_real.getP(sim.mesh, "N")
    np.testing.assert_allclose(rx_real.eval(src, sim.mesh, f), (P @ p[:, 1]).real)
    np.testing.assert_allclose(rx_imag.eval(src, sim.mesh, f), (P @ p[:, 1]).imag)


def test_free_surface_pressure_is_zero():
    sim, model, _ = heterogeneous_problem(free_surface=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = sim.fields(model)[:, "p"]
    top = np.isclose(sim.mesh.nodes[:, -1], sim.mesh.nodes[:, -1].max())
    assert np.all(p[top] == 0)
    assert np.all(np.abs(p[~top]) > 0)


def test_factorization_cleared_on_model_update():
    sim, model, _ = heterogeneous_problem()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d0 = sim.dpred(model)
        assert len(sim._Ainv) == 2  # one per frequency
        d1 = sim.dpred(model * 1.1)
    assert not np.allclose(d0, d1)
    sim.model = model
    assert getattr(sim, "_Ainv", None) in (None, {})


def test_survey_groups_by_frequency():
    rx = [fd.receivers.Pressure(np.zeros((1, 2)))]
    sources = [
        fd.sources.PointSource(rx, location=np.zeros(2), frequency=f)
        for f in (5.0, 2.0, 5.0)
    ]
    survey = fd.Survey(sources)
    assert survey.frequencies == [2.0, 5.0]
    assert survey.num_frequencies == 2
    assert survey.num_sources_by_frequency == {2.0: 1, 5.0: 2}
    assert survey.get_sources_by_frequency(5.0) == [sources[0], sources[2]]
    with pytest.raises(KeyError):
        survey.get_sources_by_frequency(3.0)


def test_validation():
    with pytest.raises(TypeError, match="location"):
        fd.sources.PointSource([], frequency=1.0)
    with pytest.raises(TypeError, match="frequency"):
        fd.sources.PointSource([], location=np.zeros(2))
    with pytest.raises(ValueError):
        fd.receivers.Pressure(np.zeros((1, 2)), component="amplitude")
    with pytest.raises(TypeError):
        fd.SimulationAcoustic(discretize.TreeMesh([np.ones(8), np.ones(8)]))


def test_warnings_and_errors():
    mesh = discretize.TensorMesh([np.full(40, 5.0), np.full(40, 5.0)], origin="CC")
    sim = simulation(mesh, make_survey([[0.3, 0.3]], [[90.0, 0.0]]), pml_cells=10)
    with pytest.warns(UserWarning, match="inside the PML"):
        sim.dpred(np.full(mesh.n_cells, 1 / VELOCITY))

    sim = simulation(mesh, make_survey([[0.3, 0.3]], [[20.0, 0.0]]), pml_cells=10)
    with pytest.warns(UserWarning, match="nodes per shortest wavelength"):
        sim.dpred(np.full(mesh.n_cells, 1 / 300.0))

    sim = simulation(
        mesh, make_survey([[0.3, 0.3]], [[20.0, 0.0]]), pml_velocity=None, pml_cells=10
    )
    with pytest.raises(ValueError, match="pml_velocity"):
        sim.dpred(np.full(mesh.n_cells, 1 / VELOCITY))


def test_small_model_change_is_not_ignored():
    """SimPEG's model setter treats changes below np.allclose's tolerance as no
    change; the factorizations must still be redone for any new slowness."""
    sim, model, _ = heterogeneous_problem()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d0 = sim.dpred(model)
        d1 = sim.dpred(model * (1 + 1e-7))
    assert not np.array_equal(d0, d1)


def test_source_waveform():
    waveform = wavelets.RickerWaveform(10.0)
    src = fd.sources.PointSource(
        [], location=np.zeros(2), frequency=7.0, waveform=waveform
    )
    assert src.amplitude == waveform.spectrum(7.0)
    with pytest.raises(ValueError, match="waveform"):
        src.amplitude = 2.0
    with pytest.raises(ValueError, match="not both"):
        fd.sources.PointSource(
            [], location=np.zeros(2), frequency=7.0, amplitude=1.0, waveform=waveform
        )
    assert (
        fd.sources.PointSource([], location=np.zeros(2), frequency=7.0).amplitude == 1.0
    )
