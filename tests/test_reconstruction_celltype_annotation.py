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


def test_attach_celltype_predictions_preserves_inherited_labels_and_marks_source():
    from scripts.annotate_reconstruction_celltype import attach_celltype_predictions

    source = ad.AnnData(
        np.zeros((2, 2), dtype=np.float32),
        obs=pd.DataFrame(
            {
                "cell_class": pd.Categorical(["A", "A"]),
                "is_xenium_anchor": [True, False],
            },
            index=["anchor", "virtual"],
        ),
    )

    result = attach_celltype_predictions(
        source,
        predicted_labels=["B", "B"],
        confidence=[0.99, 0.75],
    )

    assert result.obs["cell_class"].astype(str).tolist() == ["A", "B"]
    assert result.obs["cell_class_inherited"].astype(str).tolist() == ["A", "A"]
    assert result.obs["celltype_label_source"].tolist() == [
        "observed_anchor",
        "celltype_true_checkpoint",
    ]
    assert result.obs["celltype_model_used"].tolist() == [False, True]
    np.testing.assert_allclose(result.obs["celltype_confidence"], [1.0, 0.75])


def test_attach_celltype_predictions_rejects_wrong_length():
    from scripts.annotate_reconstruction_celltype import attach_celltype_predictions

    source = ad.AnnData(
        np.zeros((2, 2), dtype=np.float32),
        obs=pd.DataFrame({"is_xenium_anchor": [True, False]}),
    )

    try:
        attach_celltype_predictions(source, predicted_labels=["A"], confidence=[0.5])
    except ValueError as error:
        assert "length" in str(error)
    else:
        raise AssertionError("wrong prediction length was accepted")


def test_celltype_model_sampling_exposes_continuous_state_for_confidence():
    from deepspatial import DeepSpatial

    model = DeepSpatial()
    model.setup_data(_toy_anchors(), num_workers=0, n_samples_base=4)
    model.build_model(hidden_size=16, depth=1, num_heads=2, sampling_method="euler")
    batch = next(iter(model.train_loader))
    result = model.module.sample(batch, steps=3)

    assert result["c_traj_cont"].shape == (3, 4, 2)
