def test_even_to_odd_direction_uses_strictly_bracketed_targets():
    from scripts.run_00029_odd_to_even_3d_control import direction_spec

    spec = direction_spec("even_to_odd")

    assert spec["training_sections"] == (11, 31, 51, 71, 91)
    assert spec["target_sections"] == (21, 41, 61, 81)
    assert spec["boundary_not_scored"] == (1,)


def test_odd_to_even_direction_remains_backward_compatible():
    from scripts.run_00029_odd_to_even_3d_control import direction_spec

    spec = direction_spec("odd_to_even")

    assert spec["training_sections"] == (1, 21, 41, 61, 81)
    assert spec["target_sections"] == (11, 31, 51, 71)
    assert spec["boundary_not_scored"] == (91,)


def test_no_celltype_reconstruction_inherits_source_annotations():
    import anndata as ad
    import numpy as np
    import pandas as pd

    from scripts.run_00029_odd_to_even_3d_control import inherit_source_annotations

    anchor = ad.AnnData(
        X=np.zeros((2, 1), dtype=np.float32),
        obs=pd.DataFrame(
            {
                "source_cell_id": ["a", "b"],
                "cell_class": pd.Categorical(["A", "B"]),
                "pred_domain": ["D1", "D2"],
                "pred_niche": ["N1", "N2"],
            },
            index=["a", "b"],
        ),
    )
    generated = ad.AnnData(
        X=np.zeros((2, 1), dtype=np.float32),
        obs=pd.DataFrame(
            {
                "source_cell_id": ["b", "a"],
                "cell_class": pd.Categorical(["unknown", "unknown"]),
            },
            index=["v0", "v1"],
        ),
    )

    result = inherit_source_annotations(generated, [anchor])

    assert result.obs["cell_class"].astype(str).tolist() == ["B", "A"]
    assert result.obs["pred_domain"].tolist() == ["D2", "D1"]
    assert result.obs["pred_niche"].tolist() == ["N2", "N1"]
    assert result.obs["celltype_model_used"].tolist() == [False, False]
    assert result.obs["celltype_label_source"].tolist() == [
        "inherited_from_source_anchor",
        "inherited_from_source_anchor",
    ]
