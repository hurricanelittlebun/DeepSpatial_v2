import numpy as np
import pytest


def test_bidirectional_topk_candidates_have_bounded_support_and_cover_both_sides():
    from deepspatial.data_utils.uot_solver import build_topk_candidates

    x0 = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    x1 = np.array([[0.1, 0.0], [9.9, 0.0], [20.1, 0.0], [100.0, 0.0]])

    rows, cols, distances = build_topk_candidates(
        x0, x1, top_k=1, bidirectional=True
    )

    assert rows.dtype == np.int64
    assert cols.dtype == np.int64
    assert distances.shape == rows.shape
    assert len(rows) <= len(x0) + len(x1)
    assert set(np.unique(rows)) == set(range(len(x0)))
    assert set(np.unique(cols)) == set(range(len(x1)))
    assert np.isfinite(distances).all()


def test_sparse_uot_returns_only_candidate_edges_and_supports_sampling():
    from deepspatial.data_utils.uot_solver import (
        SparseCoupling,
        compute_uot_coupling,
    )

    x0 = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    x1 = np.array([[0.1, 0.0], [1.1, 0.0], [2.1, 0.0], [3.1, 0.0], [20.0, 0.0]])
    g0 = np.eye(4, dtype=np.float32)
    g1 = np.vstack([np.eye(4, dtype=np.float32), np.ones((1, 4), dtype=np.float32)])
    c0 = np.ones((4, 1), dtype=np.float32)
    c1 = np.ones((5, 1), dtype=np.float32)

    coupling = compute_uot_coupling(
        x0,
        g0,
        c0,
        x1,
        g1,
        c1,
        solver="sparse_topk",
        top_k=2,
        bidirectional=True,
        class_weight=0.0,
        uot_reg=0.2,
        uot_tau=0.05,
    )

    assert isinstance(coupling, SparseCoupling)
    assert coupling.shape == (4, 5)
    assert len(coupling) <= 4 * 2 + 5 * 2
    assert coupling.mass.shape == coupling.row.shape == coupling.col.shape
    assert np.isfinite(coupling.mass).all()
    assert coupling.mass.sum() > 0

    rows, cols = coupling.sample(32, rng=np.random.default_rng(7))
    assert rows.shape == cols.shape == (32,)
    assert np.isin(rows, coupling.row).all()
    assert np.isin(cols, coupling.col).all()


def test_sparse_uot_does_not_use_dense_entry_guard():
    from deepspatial.data_utils.uot_solver import compute_uot_coupling

    x = np.arange(20, dtype=np.float32).reshape(10, 2)
    g = np.eye(10, dtype=np.float32)
    c = np.ones((10, 1), dtype=np.float32)

    coupling = compute_uot_coupling(
        x,
        g,
        c,
        x,
        g,
        c,
        solver="sparse_topk",
        top_k=2,
        max_uot_entries=1,
        class_weight=0.0,
    )
    assert coupling.mass.sum() > 0


def test_sparse_full_support_matches_dense_pot_reference():
    from deepspatial.data_utils.uot_solver import compute_uot_coupling

    rng = np.random.default_rng(11)
    x0 = rng.normal(size=(5, 2))
    x1 = rng.normal(size=(6, 2))
    g0 = rng.random((5, 3), dtype=np.float32)
    g1 = rng.random((6, 3), dtype=np.float32)
    c0 = np.ones((5, 1), dtype=np.float32)
    c1 = np.ones((6, 1), dtype=np.float32)

    dense = compute_uot_coupling(
        x0, g0, c0, x1, g1, c1, class_weight=0.0, uot_reg=0.4, uot_tau=0.2
    )
    sparse = compute_uot_coupling(
        x0,
        g0,
        c0,
        x1,
        g1,
        c1,
        class_weight=0.0,
        uot_reg=0.4,
        uot_tau=0.2,
        solver="sparse_topk",
        top_k=len(x1),
        bidirectional=True,
    )

    np.testing.assert_allclose(sparse.to_dense(), dense, rtol=1e-6, atol=1e-8)


def test_invalid_morphology_queries_are_excluded_from_sparse_cost():
    from deepspatial.data_utils.uot_solver import compute_uot_coupling

    x = np.array([[0.0, 0.0], [1.0, 0.0]])
    g = np.eye(2, dtype=np.float32)
    c = np.ones((2, 1), dtype=np.float32)
    h = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)

    coupling = compute_uot_coupling(
        x,
        g,
        c,
        x,
        g,
        c,
        solver="sparse_topk",
        top_k=2,
        class_weight=0.0,
        histology_weight=1.0,
        h0=h,
        h1=h,
        h0_valid=np.array([True, False]),
        h1_valid=np.array([True, False]),
    )

    assert np.isfinite(coupling.mass).all()
    assert coupling.mass.sum() > 0


def test_sparse_uot_requires_topk_and_rejects_unknown_solver():
    from deepspatial.data_utils.uot_solver import compute_uot_coupling

    x = np.zeros((2, 2), dtype=np.float32)
    g = np.eye(2, dtype=np.float32)
    c = np.ones((2, 1), dtype=np.float32)

    with pytest.raises(ValueError, match="top_k"):
        compute_uot_coupling(
            x, g, c, x, g, c, solver="sparse_topk", class_weight=0.0
        )
    with pytest.raises(ValueError, match="solver"):
        compute_uot_coupling(
            x, g, c, x, g, c, solver="not-a-solver", class_weight=0.0
        )


def test_deepspatial_setup_data_can_build_sparse_trajectories():
    from deepspatial import DeepSpatial
    from test_integration import anchors

    model = DeepSpatial()
    model.setup_data(
        anchors(),
        n_samples_base=12,
        num_workers=0,
        uot_solver="sparse_topk",
        uot_top_k=2,
    )

    assert len(model.dataset) == 12
    assert model.uot_config["solver"] == "sparse_topk"
    assert model.uot_config["top_k"] == 2
