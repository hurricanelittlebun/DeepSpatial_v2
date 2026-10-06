from pathlib import Path
import subprocess
import sys

import anndata as ad
import numpy as np
import pandas as pd


def _write_anchor(path: Path, section: int, n_obs: int = 2) -> Path:
    value = ad.AnnData(
        X=np.ones((n_obs, 1), dtype=np.float32),
        obs=pd.DataFrame(
            {"section_id": [section] * n_obs, "z_um": [float(section)] * n_obs},
            index=[f"cell-{section}-{i}" for i in range(n_obs)],
        ),
        var=pd.DataFrame(index=["g1"]),
    )
    value.obsm["spatial_st_corrected"] = np.arange(n_obs * 2, dtype=float).reshape(
        n_obs, 2
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    value.write_h5ad(path)
    return path


def _write_uot(path: Path, feature_revision: str = "rev1") -> Path:
    from deepspatial.data_utils.uot_solver import SparseCoupling
    from deepspatial.data_utils.uot_cache import save_sparse_coupling

    coupling = SparseCoupling(
        row=np.asarray([0, 1], dtype=np.int64),
        col=np.asarray([1, 0], dtype=np.int64),
        mass=np.asarray([0.4, 0.6], dtype=np.float64),
        shape=(2, 2),
    )
    save_sparse_coupling(
        path,
        coupling,
        metadata={
            "source_section": "1",
            "target_section": "11",
            "spatial_key": "spatial_st_corrected",
            "feature_store_revision": feature_revision,
        },
    )
    return path


def test_uot_materializer_adds_canonical_frame_without_changing_edges(tmp_path):
    from deepspatial.data_utils.uot_cache import load_sparse_coupling
    from scripts.materialize_00029_st_only_uot_cache import (
        materialize_verified_uot_cache,
    )

    anchors = tmp_path / "anchors"
    _write_anchor(anchors / "section-001.h5ad", 1)
    _write_anchor(anchors / "section-011.h5ad", 11)
    source = tmp_path / "source"
    _write_uot(source / "pair-001-011.npz")

    result = materialize_verified_uot_cache(
        source,
        anchor_dir=anchors,
        output_dir=tmp_path / "out",
        coordinate_frame="00029__g0_registered",
        feature_revision="rev1",
    )
    old = load_sparse_coupling(source / "pair-001-011.npz")[0]
    new, metadata = load_sparse_coupling(tmp_path / "out/pair-001-011.npz")
    np.testing.assert_array_equal(old.row, new.row)
    np.testing.assert_array_equal(old.col, new.col)
    np.testing.assert_allclose(old.mass, new.mass)
    assert result["coordinate_frame"] == "00029__g0_registered"
    assert metadata["coordinate_frame"] == "00029__g0_registered"
    assert metadata["spatial_key"] == "spatial_st_corrected"
    assert metadata["h_and_e_residual_applied"] is False


def test_uot_materializer_rejects_wrong_coordinate_key(tmp_path):
    from deepspatial.data_utils.uot_cache import save_sparse_coupling
    from deepspatial.data_utils.uot_solver import SparseCoupling
    from scripts.materialize_00029_st_only_uot_cache import (
        materialize_verified_uot_cache,
    )

    anchors = tmp_path / "anchors"
    _write_anchor(anchors / "section-001.h5ad", 1)
    _write_anchor(anchors / "section-011.h5ad", 11)
    source_file = tmp_path / "source/pair-001-011.npz"
    save_sparse_coupling(
        source_file,
        SparseCoupling(
            row=np.asarray([0]),
            col=np.asarray([0]),
            mass=np.asarray([1.0]),
            shape=(2, 2),
        ),
        metadata={
            "source_section": "1",
            "target_section": "11",
            "spatial_key": "spatial_registered",
            "feature_store_revision": "rev1",
        },
    )

    import pytest

    with pytest.raises(ValueError, match="spatial_st_corrected"):
        materialize_verified_uot_cache(
            tmp_path / "source",
            anchor_dir=anchors,
            output_dir=tmp_path / "out",
            coordinate_frame="00029__g0_registered",
            feature_revision="rev1",
        )


def test_uot_cli_can_start_when_invoked_by_script_path():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/materialize_00029_st_only_uot_cache.py"),
            "--help",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
