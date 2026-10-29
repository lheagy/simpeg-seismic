import discretize
import numpy as np
import pytest
import scipy.sparse as sp

from simpeg_seismic.straight_ray import _ray_geometry
from simpeg_seismic.straight_ray._ray_geometry import (
    _mesh_bounds,
    _slab_clip_lengths,
    ray_cell_length_matrix,
)


@pytest.fixture(autouse=True, params=_ray_geometry.ENGINES)
def engine(request, monkeypatch):
    """Run every test with each tensor-mesh implementation."""
    if request.param == "numba":
        pytest.importorskip("numba")
    monkeypatch.setattr(_ray_geometry, "DEFAULT_ENGINE", request.param)
    return request.param


def tensor_2d():
    return discretize.TensorMesh(
        [[(2.0, 4, -1.3), (1.0, 12), (2.0, 4, 1.3)], [(1.0, 10), (2.0, 5, 1.2)]],
        origin="CN",
    )


def tensor_3d():
    return discretize.TensorMesh(
        [[(1.5, 3, -1.3), (1.0, 8)], [(1.0, 9)], [(0.5, 6), (1.0, 4, 1.4)]],
        origin=[-3.0, 2.0, -10.0],
    )


def adaptive_tree(h, origin, center, radius):
    mesh = discretize.TreeMesh(h, origin=origin, diagonal_balance=False)
    mesh.refine(mesh.max_level - 2, finalize=False)
    mesh.refine_ball([center], [radius], [mesh.max_level], finalize=False)
    mesh.finalize()
    return mesh


def tree_2d():
    return adaptive_tree(
        [np.ones(32), np.full(32, 0.5)], [-16.0, -8.0], np.r_[0.0, 0.0], 4.0
    )


def tree_3d():
    return adaptive_tree(
        [np.ones(16), np.ones(16), np.full(16, 0.5)],
        [0.0, 0.0, -8.0],
        np.r_[8.0, 8.0, -4.0],
        3.0,
    )


MESHES = {
    "tensor_2d": tensor_2d,
    "tensor_3d": tensor_3d,
    "tree_2d": tree_2d,
    "tree_3d": tree_3d,
}


def random_rays(mesh, n_rays, seed=0):
    rng = np.random.default_rng(seed)
    lo, hi = _mesh_bounds(mesh)
    x0 = rng.uniform(lo, hi, size=(n_rays, mesh.dim))
    x1 = rng.uniform(lo, hi, size=(n_rays, mesh.dim))
    return x0, x1


def dense_reference(mesh, x0, x1):
    """Clip every ray against every cell. Exact but O(n_rays * n_cells)."""
    bounds = mesh.cell_bounds.reshape(mesh.n_cells, mesh.dim, 2)
    G = np.zeros((x0.shape[0], mesh.n_cells))
    for i in range(x0.shape[0]):
        G[i] = _slab_clip_lengths(
            np.tile(x0[i], (mesh.n_cells, 1)),
            np.tile(x1[i], (mesh.n_cells, 1)),
            bounds[:, :, 0],
            bounds[:, :, 1],
        )
    return G


@pytest.mark.parametrize("mesh_name", MESHES)
def test_row_sums_are_ray_lengths(mesh_name):
    mesh = MESHES[mesh_name]()
    x0, x1 = random_rays(mesh, 200)
    G = ray_cell_length_matrix(mesh, x0, x1)
    np.testing.assert_allclose(
        np.asarray(G.sum(axis=1)).ravel(), np.linalg.norm(x1 - x0, axis=1), rtol=1e-12
    )


@pytest.mark.parametrize("mesh_name", MESHES)
def test_matches_dense_slab_clip(mesh_name):
    mesh = MESHES[mesh_name]()
    x0, x1 = random_rays(mesh, 25, seed=1)
    G = ray_cell_length_matrix(mesh, x0, x1).toarray()
    np.testing.assert_allclose(G, dense_reference(mesh, x0, x1), atol=1e-10)


@pytest.mark.parametrize("dim", [2, 3])
def test_uniform_tree_matches_tensor(dim):
    n = [8, 4, 4][:dim]
    tensor = discretize.TensorMesh([np.ones(k) for k in n])
    tree = discretize.TreeMesh([np.ones(k) for k in n])
    tree.refine(tree.max_level, finalize=True)

    x0, x1 = random_rays(tensor, 50, seed=2)

    def slowness(mesh):
        return 1.0 + np.sin(mesh.cell_centers @ np.arange(1, dim + 1))

    np.testing.assert_allclose(
        ray_cell_length_matrix(tree, x0, x1) @ slowness(tree),
        ray_cell_length_matrix(tensor, x0, x1) @ slowness(tensor),
        rtol=1e-12,
    )


@pytest.mark.parametrize("mesh_type", ["tensor", "tree"])
class TestFaceAlignedRays:
    def mesh(self, mesh_type):
        h = [np.ones(4)] * 3
        if mesh_type == "tensor":
            return discretize.TensorMesh(h)
        mesh = discretize.TreeMesh(h)
        mesh.refine(mesh.max_level, finalize=True)
        return mesh

    def lengths_by_cell(self, mesh, x0, x1):
        G = ray_cell_length_matrix(mesh, [x0], [x1])
        centers = mesh.cell_centers[G.indices]
        return {tuple(c): v for c, v in zip(centers, G.data)}

    def test_on_face_split_between_two_cells(self, mesh_type):
        mesh = self.mesh(mesh_type)
        lengths = self.lengths_by_cell(mesh, [0.0, 2.0, 1.5], [4.0, 2.0, 1.5])
        assert len(lengths) == 8
        for (x, y, z), v in lengths.items():
            assert y in (1.5, 2.5) and z == 1.5
            assert v == pytest.approx(0.5)

    def test_on_edge_split_between_four_cells(self, mesh_type):
        mesh = self.mesh(mesh_type)
        lengths = self.lengths_by_cell(mesh, [1.0, 2.0, 0.0], [1.0, 2.0, 3.0])
        assert len(lengths) == 12
        assert all(v == pytest.approx(0.25) for v in lengths.values())

    def test_on_outer_boundary_stays_inside(self, mesh_type):
        mesh = self.mesh(mesh_type)
        lengths = self.lengths_by_cell(mesh, [0.0, 0.0, 0.5], [4.0, 0.0, 0.5])
        assert len(lengths) == 4
        assert all(c[1] == 0.5 for c in lengths)
        assert all(v == pytest.approx(1.0) for v in lengths.values())

    def test_on_outer_corner_edge(self, mesh_type):
        mesh = self.mesh(mesh_type)
        lengths = self.lengths_by_cell(mesh, [0.0, 4.0, 4.0], [4.0, 4.0, 4.0])
        assert len(lengths) == 4
        assert all(v == pytest.approx(1.0) for v in lengths.values())

    def test_diagonal_through_nodes(self, mesh_type):
        mesh = self.mesh(mesh_type)
        lengths = self.lengths_by_cell(mesh, [0.0, 0.0, 0.0], [4.0, 4.0, 4.0])
        assert len(lengths) == 4
        assert all(v == pytest.approx(np.sqrt(3)) for v in lengths.values())


@pytest.mark.parametrize("mesh_name", MESHES)
def test_zero_length_ray(mesh_name):
    mesh = MESHES[mesh_name]()
    x0, _ = random_rays(mesh, 1)
    G = ray_cell_length_matrix(mesh, x0, x0)
    assert G.shape == (1, mesh.n_cells)
    assert G.nnz == 0


@pytest.mark.parametrize("mesh_name", MESHES)
def test_endpoint_outside_mesh_raises(mesh_name):
    mesh = MESHES[mesh_name]()
    x0, x1 = random_rays(mesh, 3)
    x1[1] = _mesh_bounds(mesh)[1] + 1.0
    with pytest.raises(ValueError, match="outside of the mesh"):
        ray_cell_length_matrix(mesh, x0, x1)


def test_endpoint_on_boundary_within_tolerance():
    mesh = tensor_3d()
    lo, hi = _mesh_bounds(mesh)
    x0 = lo - 1e-12
    x1 = hi + 1e-12
    G = ray_cell_length_matrix(mesh, [x0], [x1])
    assert G.sum() == pytest.approx(np.linalg.norm(hi - lo))


def test_matches_simpeg_2d_straight_ray():
    from simpeg.seismic import straight_ray_tomography as old

    mesh = tensor_2d()
    x0, x1 = random_rays(mesh, 30, seed=3)
    source_list = [
        old.Src(location=a, receiver_list=[old.Rx(b[None, :])]) for a, b in zip(x0, x1)
    ]
    sim = old.Simulation(mesh, survey=old.Survey(source_list))
    np.testing.assert_allclose(
        ray_cell_length_matrix(mesh, x0, x1).toarray(), sim.A.toarray(), atol=1e-10
    )


def test_rejects_cylindrical_mesh():
    mesh = discretize.CylindricalMesh([4, 1, 4])
    with pytest.raises(TypeError):
        ray_cell_length_matrix(mesh, [[0.1, 0, 0.1]], [[0.5, 0, 0.5]])


@pytest.mark.parametrize("mesh_name", ["tensor_2d", "tensor_3d"])
def test_numba_matches_numpy(mesh_name):
    pytest.importorskip("numba")
    mesh = MESHES[mesh_name]()
    x0, x1 = random_rays(mesh, 300, seed=4)
    # add rays along node planes, through nodes and of zero length
    nodes = [_ray_geometry._axis_nodes(mesh, d) for d in range(mesh.dim)]
    corner = np.array([n[2] for n in nodes])
    x0 = np.vstack([x0, corner, corner, x0[0]])
    x1 = np.vstack([x1, np.array([n[-3] for n in nodes]), corner, x0[0]])
    x1[-2, 0] = nodes[0][-1]
    G_numba = ray_cell_length_matrix(mesh, x0, x1, engine="numba")
    G_numpy = ray_cell_length_matrix(mesh, x0, x1, engine="numpy")
    assert abs(G_numba - G_numpy).max() < 1e-10


def test_bad_engine():
    with pytest.raises(ValueError, match="engine"):
        ray_cell_length_matrix(tensor_2d(), [[0.0, 1.0]], [[1.0, 2.0]], engine="cuda")


@pytest.mark.parametrize("mesh_name", MESHES)
def test_matrix_matches_coo_triplets(mesh_name):
    """CSR built directly agrees with the COO triplets, with and without splits."""
    mesh = MESHES[mesh_name]()
    lo, hi = _mesh_bounds(mesh)
    x0, x1 = random_rays(mesh, 20, seed=5)
    # ray 3 on the upper boundary along the first axis: only the copy shifted
    # in the negative direction survives, which puts the copies out of ray order
    x0[3], x1[3] = hi.copy(), hi.copy()
    x1[3, 0] = lo[0]
    for split in (False, True):
        if split:
            # ray 7 on an interior node plane: two copies
            node = _ray_geometry._axis_nodes(mesh, 1)[2]
            x0[7, 1] = x1[7, 1] = node
            x0[7, 0], x1[7, 0] = lo[0], hi[0]
            x0[7, 2:] = x1[7, 2:]
        G = ray_cell_length_matrix(mesh, x0, x1)
        rays, cells, lengths = _ray_geometry.ray_cell_lengths(mesh, x0, x1)
        G_coo = sp.coo_matrix((lengths, (rays, cells)), shape=G.shape).tocsr()
        assert abs(G - G_coo).max() < 1e-12
        np.testing.assert_allclose(
            np.asarray(G.sum(axis=1)).ravel(), np.linalg.norm(x1 - x0, axis=1)
        )
