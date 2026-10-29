import discretize
import numpy as np
import pytest
import scipy.sparse as sp
from simpeg import maps
from discretize.tests import check_derivative

import simpeg_seismic as seismic


def crosswell_survey(dim, n_src=4, n_rx=6):
    depths_src = np.linspace(-9.0, -1.0, n_src)
    depths_rx = np.linspace(-9.5, -0.5, n_rx)
    if dim == 2:
        src_locs = np.c_[np.full(n_src, 1.0), depths_src]
        rx_locs = np.c_[np.full(n_rx, 9.0), depths_rx]
    else:
        src_locs = np.c_[np.full(n_src, 1.0), np.full(n_src, 2.0), depths_src]
        rx_locs = np.c_[np.full(n_rx, 9.0), np.full(n_rx, 7.0), depths_rx]
    source_list = [
        seismic.sources.PointSource(
            receiver_list=[seismic.receivers.Traveltime(rx_locs)], location=loc
        )
        for loc in src_locs
    ]
    return seismic.Survey(source_list)


def mesh(dim, mesh_type):
    h = [np.ones(10), np.ones(10), np.ones(10)][:dim]
    origin = [0.0, 0.0, -10.0][:dim] if dim == 3 else [0.0, -10.0]
    if mesh_type == "tensor":
        return discretize.TensorMesh(h, origin=origin)
    tree = discretize.TreeMesh(
        [np.full(16, 10 / 16)] * dim, origin=origin, diagonal_balance=False
    )
    center = np.r_[[5.0] * (dim - 1), -5.0]
    tree.refine(tree.max_level - 2, finalize=False)
    tree.refine_ball([center], [2.0], [tree.max_level], finalize=False)
    tree.finalize()
    return tree


CASES = [(dim, t) for dim in (2, 3) for t in ("tensor", "tree")]


@pytest.mark.parametrize("dim, mesh_type", CASES)
def test_homogeneous_traveltimes(dim, mesh_type):
    m = mesh(dim, mesh_type)
    survey = crosswell_survey(dim)
    sim = seismic.SimulationStraightRay(
        m, survey=survey, slownessMap=maps.IdentityMap(m)
    )
    s = 1 / 1500.0
    src, rx = survey.ray_endpoints
    np.testing.assert_allclose(
        sim.dpred(np.full(m.n_cells, s)),
        s * np.linalg.norm(rx - src, axis=1),
        rtol=1e-12,
    )


@pytest.mark.parametrize("dim, mesh_type", CASES)
def test_velocity_derivative_and_adjoint(dim, mesh_type):
    m = mesh(dim, mesh_type)
    sim = seismic.SimulationStraightRay(
        m, survey=crosswell_survey(dim), slownessMap=maps.ReciprocalMap(m)
    )
    rng = np.random.default_rng(42)
    model = rng.uniform(1000.0, 3000.0, m.n_cells)

    assert check_derivative(
        lambda x: (sim.dpred(x), lambda v: sim.Jvec(x, v)),
        model,
        dx=model * 1e-2,
        num=4,
        plotIt=False,
        random_seed=0,
    )

    v = rng.standard_normal(m.n_cells)
    w = rng.standard_normal(sim.survey.nD)
    np.testing.assert_allclose(
        w @ sim.Jvec(model, v), v @ sim.Jtvec(model, w), rtol=1e-10
    )


def test_getJtJdiag():
    m = mesh(3, "tensor")
    sim = seismic.SimulationStraightRay(
        m, survey=crosswell_survey(3), slownessMap=maps.ReciprocalMap(m)
    )
    model = np.linspace(1000.0, 3000.0, m.n_cells)
    W = sp.diags(np.linspace(1.0, 2.0, sim.survey.nD))
    J = sim.getJ(model).toarray()
    np.testing.assert_allclose(
        sim.getJtJdiag(model, W=W), np.sum((W @ J) ** 2, axis=0), rtol=1e-12
    )


def test_active_cells():
    m = mesh(2, "tensor")
    active = m.cell_centers[:, 1] < -2.0
    survey = crosswell_survey(2)
    sim = seismic.SimulationStraightRay(
        m,
        survey=survey,
        active_cells=active,
        slownessMap=maps.IdentityMap(nP=int(active.sum())),
    )
    with pytest.warns(UserWarning, match="inactive cells"):
        G = sim.G
    assert G.shape == (survey.nD, active.sum())


def test_survey_change_rebuilds_G():
    m = mesh(2, "tensor")
    sim = seismic.SimulationStraightRay(m, survey=crosswell_survey(2, n_rx=3))
    assert sim.G.shape[0] == 12
    sim.survey = crosswell_survey(2, n_rx=5)
    assert sim.G.shape[0] == 20


def test_rejects_wrong_survey_type():
    from simpeg.survey import BaseSurvey

    with pytest.raises(TypeError):
        seismic.SimulationStraightRay(mesh(2, "tensor"), survey=BaseSurvey([]))


def test_rejects_1d_mesh():
    with pytest.raises(ValueError, match="2D or 3D"):
        seismic.SimulationStraightRay(discretize.TensorMesh([5]))
