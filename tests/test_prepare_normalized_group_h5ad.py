from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

import anndata as ad

from scripts.prepare_normalized_group_h5ad import materialize_group


def _source() -> ad.AnnData:
    obs = pd.DataFrame(
        {
            "joint_series_id": ["00029__g0"] * 2 + ["9589__g1"] * 2,
            "joint_cell_id": ["j0", "j1", "j2", "j3"],
            "source_cell_id": ["s0", "s1", "s2", "s3"],
            "sample_id": ["00029", "00029", "9589", "9589"],
            "tissue_id": ["00029", "00029", "9589", "9589"],
            "group_id": ["g0", "g0", "g1", "g1"],
            "section_id": [1, 11, 1, 11],
        },
        index=["row0", "row1", "row2", "row3"],
    )
    return ad.AnnData(
        X=sparse.csr_matrix(np.arange(20, dtype=np.int32).reshape(4, 5)),
        obs=obs,
        var=pd.DataFrame(index=[f"g{i}" for i in range(5)]),
        obsm={
            "spatial_registered": np.array(
                [[10, 20], [11, 21], [30, 40], [31, 41]], dtype=np.float64
            ),
            "spatial_raw": np.zeros((4, 2), dtype=np.float64),
        },
    )


def test_materialize_group_uses_registered_coordinates_and_physical_z():
    result = materialize_group(_source(), "9589__g1", section_thickness_um=5.0)

    assert result.obs_names.tolist() == ["j2", "j3"]
    assert result.var_names.tolist() == [f"g{i}" for i in range(5)]
    np.testing.assert_array_equal(result.obsm["spatial"], [[30, 40], [31, 41]])
    assert result.obs["z_um"].tolist() == [0.0, 50.0]
    assert result.obs["x_global"].tolist() == [30.0, 31.0]
    assert result.obs["is_xenium_anchor"].tolist() == [True, True]
    assert result.uns["deepspatial_group_h5ad"]["coordinate_units"] == "micrometer"


def test_materialize_group_copies_00029_labels_by_stable_id():
    annotation = _source()[[True, True, False, False]].copy()
    annotation.obs["assigned_celltype"] = pd.Categorical(["A", "B"])
    annotation.obs["pred_domain"] = pd.Categorical(["D1", "D2"])
    annotation.obs["pred_niche"] = pd.Categorical(["N1", "N2"])
    annotation.obs["pred_domain_conf"] = [0.8, 0.9]
    annotation.obs["pred_niche_conf"] = [0.7, 0.6]

    result = materialize_group(
        _source(), "00029__g0", annotation=annotation, section_thickness_um=5.0
    )
    assert result.obs["cell_class"].astype(str).tolist() == ["A", "B"]
    assert result.obs["pred_domain"].astype(str).tolist() == ["D1", "D2"]
    assert result.uns["deepspatial_group_h5ad"]["labels"]["status"] == "available"


def test_materialize_group_exposes_cell_class_from_source_annotation():
    source = _source()
    source.obs["assigned_celltype"] = pd.Categorical(["A", "B", "C", "D"])
    source.obs["pred_domain"] = pd.Categorical(["D1", "D2", "D1", "D2"])
    source.obs["pred_niche"] = pd.Categorical(["N1", "N2", "N1", "N2"])
    result = materialize_group(source, "9589__g1", section_thickness_um=5.0)
    assert result.obs["cell_class"].astype(str).tolist() == ["C", "D"]
    assert result.uns["deepspatial_group_h5ad"]["labels"]["status"] == "available"
