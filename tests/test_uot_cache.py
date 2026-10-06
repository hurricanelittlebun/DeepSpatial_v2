import numpy as np


def test_sparse_coupling_cache_roundtrip(tmp_path):
    from deepspatial.data_utils.uot_cache import (
        load_sparse_coupling,
        save_sparse_coupling,
    )
    from deepspatial.data_utils.uot_solver import SparseCoupling

    coupling = SparseCoupling(
        row=np.array([0, 0, 1], dtype=np.int64),
        col=np.array([0, 1, 1], dtype=np.int64),
        mass=np.array([0.2, 0.1, 0.3], dtype=np.float64),
        shape=(2, 2),
    )
    path = tmp_path / "pair.npz"
    save_sparse_coupling(path, coupling, {"source": "a", "target": "b"})

    restored, metadata = load_sparse_coupling(path)
    np.testing.assert_array_equal(restored.row, coupling.row)
    np.testing.assert_array_equal(restored.col, coupling.col)
    np.testing.assert_allclose(restored.mass, coupling.mass)
    assert restored.shape == coupling.shape
    assert metadata == {"source": "a", "target": "b"}


def test_sparse_coupling_summary_reports_support_and_weighted_distance():
    from deepspatial.data_utils.uot_cache import summarize_sparse_coupling
    from deepspatial.data_utils.uot_solver import SparseCoupling

    coupling = SparseCoupling(
        row=np.array([0, 0, 1], dtype=np.int64),
        col=np.array([0, 1, 1], dtype=np.int64),
        mass=np.array([0.2, 0.1, 0.3], dtype=np.float64),
        shape=(2, 2),
    )
    source_xy = np.array([[0.0, 0.0], [10.0, 0.0]])
    target_xy = np.array([[1.0, 0.0], [12.0, 0.0]])

    summary = summarize_sparse_coupling(coupling, source_xy, target_xy)

    assert summary["source_support_fraction"] == 1.0
    assert summary["target_support_fraction"] == 1.0
    assert summary["edge_count"] == 3
    assert summary["dense_pair_count"] == 4
    assert np.isclose(summary["total_mass"], 0.6)
    assert np.isclose(summary["weighted_distance_mean_um"], 2.0 / 0.6)
