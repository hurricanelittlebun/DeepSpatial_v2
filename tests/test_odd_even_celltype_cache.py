import anndata as ad
import numpy as np
import pandas as pd


def _toy_anchors():
    result = []
    for sid, z in [("a", 10.0), ("b", 20.0)]:
        obs = pd.DataFrame(
            {
                "section_id": [sid] * 3,
                "z_coord": [z] * 3,
                "cell_class": pd.Categorical(["A", "B", "A"]),
            },
            index=[f"{sid}{i}" for i in range(3)],
        )
        result.append(
            ad.AnnData(
                np.array([[1.0, 0.2], [0.2, 1.0], [0.4, 0.8]], dtype=np.float32),
                obs=obs,
                obsm={
                    "spatial": np.array(
                        [[102.0, 203.0], [104.0, 206.0], [106.0, 209.0]],
                        dtype=np.float32,
                    )
                },
            )
        )
    return result


def test_alternating_holdout_uses_only_strictly_bracketed_targets():
    from scripts.run_00029_odd_even_celltype_cached import bracketing_pair

    training = [1, 21, 41, 61, 81]
    assert bracketing_pair(training, 31) == (21, 41)
    assert bracketing_pair(training, 91) is None


def test_celltype_labels_can_be_materialized_from_cached_trajectory():
    from scripts.run_00029_odd_even_celltype_cached import labels_from_cached_trajectory

    cache = {
        "steps": 3,
        "planes": {
            "plane": {
                "query_ids": np.array([0, 1, 2, 3]),
                "t_values": np.full(4, 0.5, dtype=np.float32),
                "c_traj_cont": np.array(
                [
                    [[3.0, 0.0], [0.0, 3.0], [3.0, 0.0], [0.0, 3.0]],
                    [[0.0, 4.0], [4.0, 0.0], [0.0, 5.0], [5.0, 0.0]],
                    [[0.0, 5.0], [5.0, 0.0], [0.0, 6.0], [6.0, 0.0]],
            ],
            dtype=np.float32,
                ),
            }
        },
    }

    labels, confidence = labels_from_cached_trajectory(
        cache,
        n_query=4,
        categories=np.array(["A", "B"], dtype=object),
    )

    assert labels.tolist() == ["B", "A", "B", "A"]
    np.testing.assert_allclose(
        confidence,
        [0.98201376, 0.98201376, 0.99330717, 0.99330717],
        rtol=1e-5,
    )


def test_predict_on_he_cells_populates_celltype_trajectory_cache():
    from deepspatial import DeepSpatial

    model = DeepSpatial()
    anchors = _toy_anchors()
    model.setup_data(anchors, num_workers=0, n_samples_base=4)
    model.build_model(hidden_size=16, depth=1, num_heads=2, sampling_method="euler")
    query = ad.AnnData(
        np.zeros((2, 2), dtype=np.float32),
        obs=pd.DataFrame({"z_coord": [15.0, 15.0]}),
        obsm={"spatial": np.array([[103.0, 204.0], [105.0, 207.0]], dtype=np.float32)},
    )
    cache = {}

    model.predict_on_he_cells(
        query,
        anchors[0],
        anchors[1],
        steps=3,
        chunk_size=2,
        device="cpu",
        trajectory_cache=cache,
    )

    assert cache["version"] == 1
    assert len(cache["planes"]) == 1
    plane = next(iter(cache["planes"].values()))
    assert plane["c_traj_cont"].shape == (3, 2, 2)
    assert plane["query_ids"].tolist() == [0, 1]


def test_fixed_depth_matching_materializes_labels_from_reconstruction_cache():
    from scripts.run_00029_section31_3d_control import match_fixed_depth

    reconstructed = ad.AnnData(
        X=np.array([[9.0, 0.0], [0.0, 9.0], [2.0, 2.0]], dtype=np.float32),
        obs=pd.DataFrame(
            {
                "z_um": [15.0, 15.0, 25.0],
                "is_reconstructed": [True, True, True],
                "cell_class": pd.Categorical(["A", "B", "A"]),
            },
            index=["v0", "v1", "v2"],
        ),
        var=pd.DataFrame(index=["g0", "g1"]),
        obsm={
            "spatial_st_corrected": np.array(
                [[100.0, 100.0], [200.0, 200.0], [500.0, 500.0]],
                dtype=np.float32,
            ),
            "celltype_flow_state": np.array(
                [[5.0, 0.0], [0.0, 5.0], [5.0, 0.0]], dtype=np.float32
            ),
        },
    )
    target = ad.AnnData(
        X=np.zeros((2, 2), dtype=np.float32),
        obs=pd.DataFrame({"z_um": [15.0, 15.0]}, index=["t0", "t1"]),
        var=reconstructed.var.copy(),
        obsm={
            "spatial_st_corrected": np.array(
                [[101.0, 101.0], [199.0, 199.0]], dtype=np.float32
            )
        },
    )

    prediction, spatial_summary = match_fixed_depth(
        reconstructed,
        target,
        target_z=15.0,
        categories=["A", "B"],
        tolerance_um=5.0,
    )

    assert spatial_summary["generated_layer_cells"] == 2
    assert prediction.obs["cell_class"].astype(str).tolist() == ["A", "B"]
    assert prediction.obs["celltype_confidence"].min() > 0.99
    np.testing.assert_allclose(
        prediction.obsm["target_spatial"], target.obsm["spatial_st_corrected"]
    )


def test_full_reconstruction_can_cache_continuous_celltype_state():
    from deepspatial import DeepSpatial

    model = DeepSpatial()
    anchors = _toy_anchors()
    model.setup_data(
        anchors,
        num_workers=0,
        n_samples_base=4,
        use_celltype=True,
    )
    model.build_model(hidden_size=16, depth=1, num_heads=2, sampling_method="euler")

    reconstructed = model.reconstruct_between_slices(
        anchors[0],
        anchors[1],
        thickness=5.0,
        steps=3,
        chunk_size=2,
        device="cpu",
        cache_celltype_trajectory=True,
    )

    assert "celltype_flow_state" in reconstructed.obsm
    assert reconstructed.obsm["celltype_flow_state"].shape == (
        reconstructed.n_obs,
        model.num_classes,
    )
    assert np.isfinite(reconstructed.obsm["celltype_flow_state"]).all()
