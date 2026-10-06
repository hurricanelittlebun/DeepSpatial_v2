import numpy as np
import pandas as pd
import pytest
import torch
import anndata as ad


def anchors(labels=True):
    result = []
    for sid, z in [("a", 10.0), ("b", 20.0)]:
        obs = pd.DataFrame(
            {"section_id": [sid] * 3, "z_coord": [z] * 3},
            index=[f"{sid}{i}" for i in range(3)],
        )
        if labels:
            obs["cell_class"] = ["A", "B", "A"]
        result.append(
            ad.AnnData(
                np.array([[1.0, 0.2], [0.2, 1.0], [0.4, 0.8]], np.float32),
                obs=obs,
                obsm={
                    "spatial": np.array(
                        [[102.0, 203.0], [104.0, 206.0], [106.0, 209.0]], np.float32
                    )
                },
            )
        )
    return result


@pytest.mark.parametrize("mode", ["baseline", "uot", "path"])
@pytest.mark.parametrize("labels", [True, False])
def test_real_pipeline(tmp_path, mode, labels):
    from deepspatial import DeepSpatial
    from deepspatial.histology import HistologyConfig
    from test_histology import make_store

    store = make_store(tmp_path)
    config = (
        HistologyConfig()
        if mode == "baseline"
        else HistologyConfig(
            use_histology=True,
            use_morphology_uot=True,
            use_morphology_path=mode == "path",
            feature_path=store.path,
            path_cache=str(tmp_path / "paths.h5"),
            coordinate_frame="toy",
            candidate_radius_um=0,
        )
    )
    torch.manual_seed(9)
    np.random.seed(9)
    a = anchors(labels)
    ds = DeepSpatial()
    ds.setup_data(
        a,
        n_samples_base=8,
        batch_size=4,
        num_workers=0,
        use_celltype=labels,
        histology=config,
    )
    ds.build_model(
        hidden_size=16, depth=1, num_heads=2, patch_size=1, sampling_method="euler"
    )
    batch = next(iter(ds.train_loader))
    losses = ds.module._shared_step(batch)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    opt = ds.module.configure_optimizers()
    opt.step()
    ds.module._update_ema()
    sampled = ds.module.sample(batch, steps=3)
    assert sampled["x_traj"].shape == (3, 4, 2)
    assert sampled["c_traj_discrete"].shape == (3, 4)
    out = ds.reconstruct_between_slices(
        a[0], a[1], thickness=10, steps=3, chunk_size=4, device="cpu"
    )
    assert out.n_obs == 3 and out.n_vars == 2
    assert np.isfinite(out.obsm["spatial"]).all()
    assert "cell_class" in out.obs
    ds._save_config(str(tmp_path))
    torch.save({"state_dict": ds.module.state_dict()}, tmp_path / "model.ckpt")
    loaded = DeepSpatial()
    loaded.load_checkpoint(str(tmp_path / "model.ckpt"), sampling_method="euler")
    again = loaded.module.sample(batch, steps=3)
    torch.testing.assert_close(again["x_traj"], sampled["x_traj"])
    # A fresh H5AD load has no internal spatial_norm/z_norm from setup_data.
    fresh = anchors(labels)
    rebuilt = loaded.reconstruct_between_slices(
        fresh[0], fresh[1], thickness=10, steps=3, device="cpu"
    )
    assert rebuilt.n_obs == 3 and np.isfinite(rebuilt.obsm["spatial"]).all()


def test_reconstruction_rejects_different_gene_order():
    from deepspatial import DeepSpatial

    ds = DeepSpatial()
    ds.setup_data(anchors(), num_workers=0, n_samples_base=4)
    ds.build_model(hidden_size=16, depth=1, num_heads=2)
    fresh = [a[:, ::-1].copy() for a in anchors()]
    with pytest.raises(ValueError, match="gene"):
        ds.reconstruct_between_slices(
            fresh[0], fresh[1], thickness=10, steps=3, device="cpu"
        )


def test_predict_on_he_cells_returns_one_row_per_query_cell():
    from deepspatial import DeepSpatial

    anchors_data = anchors(labels=True)
    ds = DeepSpatial()
    ds.setup_data(
        anchors_data,
        num_workers=0,
        n_samples_base=4,
        use_celltype=True,
    )
    ds.build_model(hidden_size=16, depth=1, num_heads=2, sampling_method="euler")

    queries = ad.AnnData(
        np.zeros((4, 2), dtype=np.float32),
        obs=pd.DataFrame(
            {"z_coord": [12.0, 12.0, 15.0, 18.0]},
            index=["he0", "he1", "he2", "he3"],
        ),
        obsm={
            "spatial": np.array(
                [[102.0, 203.0], [104.0, 206.0], [105.0, 207.0], [106.0, 209.0]],
                dtype=np.float32,
            )
        },
    )

    out = ds.predict_on_he_cells(
        queries,
        anchors_data[0],
        anchors_data[1],
        steps=3,
        chunk_size=2,
        device="cpu",
    )

    assert out.n_obs == queries.n_obs
    assert out.n_vars == anchors_data[0].n_vars
    np.testing.assert_allclose(out.obsm["spatial"], queries.obsm["spatial"])
    np.testing.assert_allclose(out.obs["z_coord"], queries.obs["z_coord"])
    assert out.X.shape == (4, 2)
    assert np.isfinite(out.X.toarray()).all()
    assert np.isfinite(out.obs["prediction_distance_um"]).all()
    assert "cell_class" in out.obs


def test_reconstruction_requires_complete_propagated_domain_niche_labels():
    from deepspatial import DeepSpatial

    anchors_data = anchors(labels=True)
    for i, anchor in enumerate(anchors_data):
        anchor.obs["pred_domain"] = pd.Categorical(["Tumor", "Stroma", "Tumor"])
        anchor.obs["pred_niche"] = pd.Categorical(["Tumor nest", "CAF enriched", "Tumor nest"])

    model = DeepSpatial()
    assert model._resolve_propagated_label_keys(anchors_data[0], anchors_data[1]) == (
        "pred_domain", "pred_niche"
    )

    del anchors_data[1].obs["pred_niche"]
    with pytest.raises(ValueError, match="pred_niche"):
        model._resolve_propagated_label_keys(anchors_data[0], anchors_data[1])


def test_original_weights_losses_and_sampling_regression():
    """Compare the release implementation, not a second copy of new formulas."""
    import subprocess
    from deepspatial.models.git import GiT
    from deepspatial.module import DeepSpatialModule

    ns = {
        "__name__": "deepspatial.models.release_git",
        "__package__": "deepspatial.models",
    }
    exec(
        compile(
            subprocess.check_output(
                ["git", "show", "fdb9865:deepspatial/models/git.py"], text=True
            ),
            "release_git",
            "exec",
        ),
        ns,
    )
    ms = {"__name__": "deepspatial.release_module", "__package__": "deepspatial"}
    exec(
        compile(
            subprocess.check_output(
                ["git", "show", "fdb9865:deepspatial/module.py"], text=True
            ),
            "release_module",
            "exec",
        ),
        ms,
    )
    torch.manual_seed(4)
    old = ns["GiT"](2, 1, 16, 1, 2, 2)
    torch.manual_seed(4)
    new = GiT(2, 1, 16, 1, 2, 2)
    for k, v in old.state_dict().items():
        torch.testing.assert_close(v, new.state_dict()[k], rtol=0, atol=0)
    args = {
        "path_type": "Linear",
        "prediction": "velocity",
        "lr": 1e-4,
        "sampling_method": "euler",
    }
    m0 = ms["DeepSpatialModule"](args, old)
    m1 = DeepSpatialModule(args, new)
    b = {k: torch.rand(4, 2) for k in ["x0", "x1", "g0", "g1", "c0", "c1"]}
    b.update(z0=torch.zeros(4, 1), z1=torch.ones(4, 1), delta_z=torch.ones(4, 1))
    torch.manual_seed(7)
    a = m0._shared_step(b)
    torch.manual_seed(7)
    c = m1._shared_step(b)
    for k in a:
        torch.testing.assert_close(a[k], c[k], rtol=0, atol=0)
    a = m0.sample(b, steps=3)
    c = m1.sample(b, steps=3)
    for k in a:
        torch.testing.assert_close(a[k], c[k], rtol=0, atol=0)


def test_default_celltype_requires_labels():
    from deepspatial import DeepSpatial

    with pytest.raises(ValueError, match="use_celltype=False"):
        DeepSpatial().setup_data(anchors(False), num_workers=0, n_samples_base=4)


def test_adaptive_ode_and_reverse_field(tmp_path):
    from deepspatial import DeepSpatial
    from deepspatial.histology import HistologyConfig
    from test_histology import make_store

    store = make_store(tmp_path)
    ds = DeepSpatial()
    ds.setup_data(
        anchors(),
        num_workers=0,
        n_samples_base=4,
        histology=HistologyConfig(
            use_histology=True,
            feature_path=store.path,
            coordinate_frame="toy",
        ),
    )
    ds.build_model(hidden_size=16, depth=1, num_heads=2, patch_size=1)

    class Field(torch.nn.Module):
        def forward(self, xt, gt, t, zt, delta_z, ct):
            assert (delta_z > 0).all()
            # Known v_x=t/10; both forward/reverse displacements have magnitude .05.
            return (
                t[:, None].expand_as(xt) / 10,
                torch.zeros_like(gt),
                torch.zeros_like(ct),
            )

    ds.module.ema_model = Field()
    b = next(iter(ds.train_loader))
    b["x0"] = torch.full_like(b["x0"], 0.5)
    forward = ds.module.sample(b, steps=3)["x_traj"]
    reverse = dict(b, z0=b["z1"], z1=b["z0"], delta_z=-b["delta_z"], reverse=True)
    backward = ds.module.sample(reverse, steps=3)["x_traj"]
    torch.testing.assert_close(
        forward[-1] - forward[0], torch.full_like(b["x0"], 0.05), atol=2e-4, rtol=0
    )
    torch.testing.assert_close(
        backward[-1] - backward[0], torch.full_like(b["x0"], -0.05), atol=2e-4, rtol=0
    )
