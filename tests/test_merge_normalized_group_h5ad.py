from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

import anndata as ad

from scripts.merge_normalized_group_h5ad import merge_group_h5ads


def _write(path, series, offset):
    obs = pd.DataFrame(
        {
            "series_id": [series, series],
            "section_id": [1, 11],
            "g_id": [series.split("__")[1]] * 2,
        },
        index=[f"{series}-a", f"{series}-b"],
    )
    value = ad.AnnData(
        X=sparse.csr_matrix(np.array([[offset, 0, 1], [0, offset + 1, 0]], dtype=np.int32)),
        obs=obs,
        var=pd.DataFrame(index=["A", "B", "C"]),
        obsm={"spatial": np.array([[1, 2], [3, 4]], dtype=float), "spatial_registered": np.array([[1, 2], [3, 4]], dtype=float)},
    )
    value.write_h5ad(path)


def test_merge_preserves_rows_and_series_metadata(tmp_path):
    first = tmp_path / "a.h5ad"
    second = tmp_path / "b.h5ad"
    output = tmp_path / "merged.h5ad"
    _write(first, "9589__g1", 1)
    _write(second, "9625__g0", 3)

    manifest = merge_group_h5ads([first, second], output)
    result = ad.read_h5ad(output)
    assert result.shape == (4, 3)
    assert result.obs_names.is_unique
    assert set(result.obs["series_id"]) == {"9589__g1", "9625__g0"}
    assert result.X.format == "csr"
    assert manifest["n_obs"] == 4
    assert result.uns["deepspatial_merged_groups"]["group_count"] == 2
