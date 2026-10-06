import numpy as np
import pytest
import torch


def test_revision_invalidates_paths_and_persistent_cache(tmp_path):
    from test_histology import make_store
    from deepspatial.histology import MorphologyPathCache, FeatureStore

    s = make_store(tmp_path)
    path = tmp_path / "paths.h5"
    p = MorphologyPathCache(path, s, candidate_radius_um=0)
    a = np.array([[102.0, 203.0]])
    b = np.array([[106.0, 209.0]])
    ids = p.prepare("a", "b", a, b, ["id"])
    q = MorphologyPathCache(path, FeatureStore(s.path), candidate_radius_um=0)
    x, v = q.evaluate(ids, torch.tensor([0.5]))
    np.testing.assert_allclose(x, [[104.0, 206.0]], atol=1e-5)
    np.testing.assert_allclose(v, [[4.0, 6.0]], atol=1e-5)
    s.add_section(
        "extra",
        np.ones((5, 5, 2)),
        z_um=30,
        origin_um=(100, 200),
        spacing_um=(2, 3),
        patch_size_um=20,
        mpp=(0.5, 0.5),
        coordinate_frame="toy",
    )
    assert p.prepare("a", "b", a, b, ["id"]) != ids


def test_uot_guard_prevents_dense_allocation():
    from deepspatial.data_utils.uot_solver import compute_uot_coupling

    a = np.ones((10, 2))
    with pytest.raises(MemoryError):
        compute_uot_coupling(a, a, a, a, a, a, max_uot_entries=50)


def test_dataset_guard_before_densifying_expression(tmp_path, monkeypatch):
    import scipy.sparse as sp
    from deepspatial import DeepSpatial
    from deepspatial.histology import HistologyConfig
    from test_histology import make_store
    from test_integration import anchors

    store = make_store(tmp_path)
    a = anchors()
    for section in a:
        section.X = sp.csr_matrix(section.X)

    def forbidden(*args, **kwargs):
        raise AssertionError("Must reject before densifying expression")

    monkeypatch.setattr(sp.csr_matrix, "toarray", forbidden)
    with pytest.raises(MemoryError, match="before expression"):
        DeepSpatial().setup_data(
            a,
            n_samples_base=4,
            num_workers=0,
            histology=HistologyConfig(
                use_histology=True,
                feature_path=store.path,
                coordinate_frame="toy",
                max_uot_entries=4,
            ),
        )


def test_no_candidates_not_silent_linear_fallback(tmp_path):
    from test_histology import make_store
    from deepspatial.histology import MorphologyPathCache

    p = MorphologyPathCache(
        tmp_path / "paths.h5", make_store(tmp_path), candidate_radius_um=0
    )
    with pytest.raises(ValueError, match="No valid"):
        p.prepare("a", "b", np.array([[0.0, 0.0]]), np.array([[0.0, 0.0]]), ["bad"])


def test_out_of_grid_path_can_explicitly_use_linear_fallback(tmp_path):
    """An uncovered endpoint can opt into an auditable spatial-only path."""
    from test_histology import make_store
    from deepspatial.histology import MorphologyPathCache

    p = MorphologyPathCache(
        tmp_path / "fallback_paths.h5",
        make_store(tmp_path),
        candidate_radius_um=0,
        allow_linear_fallback=True,
    )
    ids = p.prepare(
        "a", "b", np.array([[0.0, 0.0]]), np.array([[10.0, 10.0]]), ["fallback"]
    )
    x, v = p.evaluate(ids * 3, torch.tensor([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(x.numpy(), [[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]])
    np.testing.assert_allclose(v.numpy(), [[10.0, 10.0]] * 3)


def test_celltype_loss_is_live_and_missing_mode_zero(tmp_path):
    from deepspatial import DeepSpatial
    from test_integration import anchors

    for labels in [True, False]:
        ds = DeepSpatial()
        ds.setup_data(
            anchors(labels), use_celltype=labels, num_workers=0, n_samples_base=4
        )
        ds.build_model(hidden_size=16, depth=1, num_heads=2)
        b = next(iter(ds.train_loader))
        loss = ds.module._shared_step(b)
        loss["loss"].backward()
        grad = ds.model.c_head.weight.grad
        if labels:
            assert loss["loss_c"] > 0 and grad.abs().sum() > 0
        else:
            assert (
                loss["loss_c"] == 0
                and grad.abs().sum() == 0
                and b["c0"].abs().sum() == 0
            )


def test_lightning_checkpoint_roundtrip(tmp_path):
    from deepspatial import DeepSpatial
    from test_integration import anchors

    ds = DeepSpatial()
    ds.setup_data(anchors(), num_workers=0, n_samples_base=4)
    ds.build_model(hidden_size=16, depth=1, num_heads=2, sampling_method="euler")
    ds.fit(
        max_epochs=1,
        save_dir=str(tmp_path),
        accelerator="cpu",
        devices=1,
        save_ckpt=True,
    )
    ckpt = next(tmp_path.glob("*.ckpt"))
    loaded = DeepSpatial()
    loaded.load_checkpoint(str(ckpt), sampling_method="euler")
    b = next(iter(ds.train_loader))
    torch.testing.assert_close(
        ds.module.sample(b, steps=3)["x_traj"],
        loaded.module.sample(b, steps=3)["x_traj"],
    )
