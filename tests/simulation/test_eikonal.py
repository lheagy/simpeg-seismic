import discretize
import numpy as np
import pytest
import scipy.sparse as sp
from discretize.tests import check_derivative
from simpeg import maps

pytest.importorskip("numba")

import simpeg_seismic as seismic  # noqa: E402


def make_survey(source_locations, receiver_locations):
    receivers = [seismic.receivers.Traveltime(receiver_locations)]
    return seismic.Survey(
        [
            seismic.sources.PointSource(receiver_list=receivers, location=loc)
            for loc in source_locations
        ]
    )


def simulation(mesh, survey, mapping=None, **kwargs):
    return seismic.SimulationEikonal(
        mesh,
        survey=survey,
        slownessMap=maps.IdentityMap(mesh) if mapping is None else mapping,
        **kwargs,
    )


def godunov_residual(sim, f, j):
    """Residual of Godunov's upwind equation at the free nodes of source ``j``."""
    T = f["T"][j]
    s = f["slowness_nodes"]
    nodes = sim._nodes
    shape = [n.size for n in nodes]
    T3 = T.reshape(shape, order="F")
    total = np.zeros(shape)
    for d in range(3):
        if shape[d] == 1:
            continue
        h = np.diff(nodes[d])
        pad = [(0, 0)] * 3
        pad[d] = (1, 1)
        Tp = np.pad(T3, pad, constant_values=np.inf)
        sl = [slice(None)] * 3
        sl[d] = slice(0, -2)
        minus = Tp[tuple(sl)]
        sl[d] = slice(2, None)
        plus = Tp[tuple(sl)]
        hshape = [1, 1, 1]
        hshape[d] = -1
        h_minus = np.r_[np.inf, h].reshape(hshape)
        h_plus = np.r_[h, np.inf].reshape(hshape)
        # Godunov: the steeper of the two upwind one-sided slopes
        with np.errstate(invalid="ignore"):
            slope = np.fmax((T3 - minus) / h_minus, (T3 - plus) / h_plus)
        total += np.maximum(np.nan_to_num(slope, nan=0.0, neginf=0.0), 0) ** 2
    fixed = np.zeros(T.size, dtype=bool)
    g = sim.source_geometry
    fixed[g["fixed_idx"][g["fixed_ptr"][j] : g["fixed_ptr"][j + 1]]] = True
    residual = np.sqrt(total.ravel(order="F")) - s
    return residual[~fixed] / s[~fixed]


def homogeneous_error(h, dim=2, slowness=1 / 2000.0, factored=True):
    n = int(round(40 / h))
    mesh = discretize.TensorMesh([np.full(n, h)] * dim)
    src = np.full((1, dim), 10.3)
    rng = np.random.default_rng(0)
    rx = rng.uniform(25, 39, size=(20, dim))
    sim = simulation(mesh, make_survey(src, rx), factored=factored)
    d = sim.dpred(np.full(mesh.n_cells, slowness))
    exact = slowness * np.linalg.norm(rx - src, axis=1)
    return np.max(np.abs(d - exact) / exact)


def random_problem(dim, factored, seed=1):
    mesh = discretize.TensorMesh([np.r_[np.full(6, 2.0), np.ones(10)]] * dim)
    rng = np.random.default_rng(seed)
    survey = make_survey(rng.uniform(0, 16, (3, dim)), rng.uniform(0, 16, (4, dim)))
    sim = simulation(mesh, survey, factored=factored)
    return sim, rng.uniform(1 / 3000, 1 / 1000, mesh.n_cells)


@pytest.mark.parametrize("dim", [2, 3])
def test_plain_scheme_solves_godunov_equations(dim):
    """Independent numpy check of the unfactored discrete equations."""
    sim, model = random_problem(dim, factored=False)
    f = sim.fields(model)
    assert np.all(np.isfinite(f["T"]))
    np.testing.assert_array_equal(f["T"], f["tau"])
    for j in range(3):
        assert np.max(np.abs(godunov_residual(sim, f, j))) < 1e-8


@pytest.mark.parametrize("factored", [False, True])
@pytest.mark.parametrize("dim", [2, 3])
def test_fixed_point_of_local_solver(dim, factored):
    """tau at every free node is the local solution given its neighbours."""
    from simpeg_seismic.utils import _eikonal_numba as kernels

    sim, model = random_problem(dim, factored)
    f = sim.fields(model)
    g = sim.source_geometry
    coef = np.zeros((8, 3))
    grad = np.empty(3)
    for j in range(3):
        fixed = g["fixed_idx"][g["fixed_ptr"][j] : g["fixed_ptr"][j + 1]]
        for i in np.setdiff1d(np.arange(sim.mesh.n_nodes), fixed):
            kernels._stencil(
                f["tau"][j], i, *sim._nodes, g["source_xyz"][j], factored, coef, grad
            )
            t, _ = kernels._local_solve(coef, f["slowness_nodes"][i])
            assert t == pytest.approx(f["tau"][j, i], rel=1e-9)


@pytest.mark.parametrize("dim", [2, 3])
def test_factored_is_exact_in_homogeneous_model(dim):
    assert homogeneous_error(1.0, dim=dim) < 1e-11


def test_plain_scheme_convergence():
    # h log(h) because of the point source: about 0.65 per halving of h
    errors = [homogeneous_error(h, factored=False) for h in (2.0, 1.0, 0.5)]
    assert errors[0] < 0.07
    assert errors[1] < 0.75 * errors[0]
    assert errors[2] < 0.75 * errors[1]


def gradient_error(h, factored=True):
    v0, gradient = 1500.0, 20.0
    mesh = discretize.TensorMesh(
        [np.full(int(80 / h), h), np.full(int(40 / h), h)], origin=[0.0, -40.0]
    )
    velocity = v0 - gradient * mesh.cell_centers[:, 1]
    src = np.array([[5.0, -1.0]])
    rx = np.c_[np.linspace(30, 75, 10), np.full(10, -1.0)]
    sim = simulation(
        mesh, make_survey(src, rx), mapping=maps.ReciprocalMap(mesh), factored=factored
    )
    d = sim.dpred(velocity)

    def v(z):
        return v0 - gradient * z

    r = np.linalg.norm(rx - src, axis=1)
    exact = (
        np.arccosh(1 + gradient**2 * r**2 / (2 * v(src[0, 1]) * v(rx[:, 1]))) / gradient
    )
    # rays bend downwards into faster material, so they beat the straight ray
    assert np.all(d < r / v(-1.0))
    return np.max(np.abs(d - exact) / exact)


def test_linear_velocity_gradient():
    errors = [gradient_error(h) for h in (1.0, 0.5)]
    assert errors[0] < 1e-3
    assert errors[1] < 0.5 * errors[0]
    assert gradient_error(0.5, factored=False) < 0.01


def test_head_wave():
    v1, v2, thickness = 1000.0, 3000.0, 10.0
    h = 0.5
    mesh = discretize.TensorMesh([np.full(200, h), np.full(80, h)], origin=[0.0, -40.0])
    z = mesh.cell_centers[:, 1]
    velocity = np.where(z > -thickness, v1, v2)
    src = np.array([[2.0, 0.0]])
    offsets = np.linspace(50, 95, 10)
    rx = np.c_[src[0, 0] + offsets, np.zeros(10)]
    sim = simulation(mesh, make_survey(src, rx), mapping=maps.ReciprocalMap(mesh))
    d = sim.dpred(velocity)
    head_wave = offsets / v2 + 2 * thickness * np.sqrt(1 / v1**2 - 1 / v2**2)
    assert np.all(head_wave < offsets / v1)  # beyond the crossover distance
    # limited by the slowness averaged onto the nodes at the interface
    np.testing.assert_allclose(d, head_wave, rtol=0.01)


@pytest.mark.parametrize("factored", [False, True])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("velocity", [False, True])
def test_derivative_and_adjoint(dim, velocity, factored):
    n = 12 if dim == 2 else 7
    mesh = discretize.TensorMesh([np.r_[np.full(3, 1.5), np.ones(n)]] * dim)
    rng = np.random.default_rng(2)
    extent = 4.5 + n
    survey = make_survey(
        rng.uniform(0, extent, (3, dim)), rng.uniform(0, extent, (5, dim))
    )
    mapping = maps.ReciprocalMap(mesh) if velocity else maps.IdentityMap(mesh)
    sim = simulation(mesh, survey, mapping=mapping, factored=factored)
    model = rng.uniform(1000.0, 3000.0, mesh.n_cells)
    if not velocity:
        model = 1 / model

    assert check_derivative(
        lambda x: (sim.dpred(x), lambda v: sim.Jvec(x, v)),
        model,
        dx=model * 1e-3 * rng.standard_normal(mesh.n_cells),
        num=4,
        plotIt=False,
        random_seed=0,
    )

    f = sim.fields(model)
    v = rng.standard_normal(mesh.n_cells)
    w = rng.standard_normal(survey.nD)
    np.testing.assert_allclose(
        w @ sim.Jvec(model, v, f=f), v @ sim.Jtvec(model, w, f=f), rtol=1e-10
    )


@pytest.mark.parametrize("dim", [2, 3])
def test_adjoint_with_dependency_cycles(dim):
    """Nodes straddling the source planes depend on each other; check the
    block solves of those cycles against the transpose."""
    from simpeg_seismic.utils import _eikonal_numba as kernels

    n = 20 if dim == 2 else 9
    mesh = discretize.TensorMesh([np.ones(n)] * dim)
    rng = np.random.default_rng(5)
    survey = make_survey(rng.uniform(2, n - 2, (2, dim)), rng.uniform(0, n, (6, dim)))
    sim = simulation(mesh, survey, mapping=maps.ReciprocalMap(mesh))
    model = rng.uniform(1500.0, 2500.0, mesh.n_cells)
    f = sim.fields(model)
    g = sim.source_geometry
    *_, n_cyclic = kernels.solve(
        f["slowness_nodes"],
        *sim._nodes,
        dim,
        g["source_xyz"],
        True,
        g["fixed_ptr"],
        g["fixed_idx"],
        model[g["fixed_cell"]] ** -1 * g["fixed_coefficient"],
        sim.tolerance,
        sim.max_iterations,
    )
    assert np.all(n_cyclic > 0)
    v = rng.standard_normal(mesh.n_cells)
    w = rng.standard_normal(survey.nD)
    np.testing.assert_allclose(
        w @ sim.Jvec(model, v, f=f), v @ sim.Jtvec(model, w, f=f), rtol=1e-10
    )
    # and against finite differences
    dm = 1e-4 * model * rng.standard_normal(mesh.n_cells)
    fd = (sim.dpred(model + dm) - sim.dpred(model - dm)) / 2
    np.testing.assert_allclose(sim.Jvec(model, dm, f=f), fd, rtol=1e-4, atol=1e-12)


def test_getJtJdiag():
    mesh = discretize.TensorMesh([np.ones(10), np.ones(8)])
    rng = np.random.default_rng(3)
    survey = make_survey(rng.uniform(0, 8, (2, 2)), rng.uniform(0, 8, (6, 2)))
    sim = simulation(mesh, survey, mapping=maps.ReciprocalMap(mesh))
    model = rng.uniform(1000.0, 3000.0, mesh.n_cells)
    f = sim.fields(model)
    J = np.column_stack([sim.Jvec(model, e, f=f) for e in np.eye(mesh.n_cells)])
    W = sp.diags(rng.uniform(1, 2, survey.nD))
    np.testing.assert_allclose(
        sim.getJtJdiag(model, W=W, f=f, batch_size=5),
        np.sum((W @ J) ** 2, axis=0),
        rtol=1e-10,
        atol=1e-20,
    )


@pytest.mark.parametrize("location", [[5.0, 5.0], [0.0, 3.0], [10.0, 8.0], [4.3, 2.0]])
def test_source_positions(location):
    """Sources on nodes, edges, corners and inside cells."""
    mesh = discretize.TensorMesh([np.ones(10), np.ones(8)])
    rx = np.array([[9.5, 0.5], [0.5, 7.5], [9.5, 4.0], [5.2, 4.9]])
    sim = simulation(mesh, make_survey([location], rx))
    s = 1 / 1500.0
    d = sim.dpred(np.full(mesh.n_cells, s))
    exact = s * np.linalg.norm(rx - np.array(location), axis=1)
    np.testing.assert_allclose(d, exact, rtol=1e-11)


def test_matches_straight_ray_in_homogeneous_model():
    mesh = discretize.TensorMesh([np.full(40, 0.5), np.full(60, 0.5)], origin=[0, -30])
    survey = make_survey(
        np.c_[np.zeros(4), np.linspace(-28, -2, 4)],
        np.c_[np.full(6, 20.0), np.linspace(-29, -1, 6)],
    )
    s = np.full(mesh.n_cells, 1 / 2500.0)
    sim_sr = seismic.SimulationStraightRay(
        mesh, survey=survey, slownessMap=maps.IdentityMap(mesh)
    )
    np.testing.assert_allclose(
        simulation(mesh, survey).dpred(s), sim_sr.dpred(s), rtol=1e-11
    )


def test_not_converged_warns():
    mesh = discretize.TensorMesh([np.ones(10), np.ones(8)])
    sim = simulation(mesh, make_survey([[1.0, 1.0]], [[9.0, 7.0]]), max_iterations=1)
    with pytest.warns(UserWarning, match="did not converge"):
        sim.fields(np.full(mesh.n_cells, 1e-3))


def test_rejects_tree_mesh():
    mesh = discretize.TreeMesh([np.ones(8), np.ones(8)])
    with pytest.raises(TypeError):
        seismic.SimulationEikonal(mesh)


def test_inactive_cells_not_supported():
    mesh = discretize.TensorMesh([np.ones(10), np.ones(8)])
    active = mesh.cell_centers[:, 1] < 6
    sim = seismic.SimulationEikonal(
        mesh,
        survey=make_survey([[1.0, 1.0]], [[9.0, 5.0]]),
        active_cells=active,
        slownessMap=maps.IdentityMap(nP=int(active.sum())),
    )
    with pytest.raises(NotImplementedError):
        sim.dpred(np.full(active.sum(), 1e-3))


def test_receiver_outside_mesh_raises():
    mesh = discretize.TensorMesh([np.ones(10), np.ones(8)])
    sim = simulation(mesh, make_survey([[1.0, 1.0]], [[11.0, 5.0]]))
    with pytest.raises(ValueError, match="outside of the mesh"):
        sim.dpred(np.full(mesh.n_cells, 1e-3))
