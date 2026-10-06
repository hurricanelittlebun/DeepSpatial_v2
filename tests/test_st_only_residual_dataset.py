from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest


def _write_anchor(path: Path, coordinates=(10.0, 20.0)) -> Path:
    value = ad.AnnData(
        X=np.asarray([[1.0, 2.0]], dtype=np.float32),
        obs=pd.DataFrame(
            {"section_id": [1], "z_um": [0.0]},
            index=["cell-1"],
        ),
        var=pd.DataFrame(index=["g1", "g2"]),
    )
    value.obsm["spatial_registered"] = np.asarray([coordinates], dtype=np.float64)
    value.obsm["spatial_he_local"] = np.asarray([coordinates], dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    value.write_h5ad(path)
    return path


def _write_residual(root: Path, section: int = 1) -> Path:
    path = root / "transforms" / (
        f"section-{section:03d}__st_to_fixed_he_residual.npz"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        matrix_current_g_to_fixed_he=np.asarray(
            [[1.0, 0.0, 5.0], [0.0, 1.0, -2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
    )
    return path


def test_apply_st_residual_changes_st_only():
    from scripts.build_00029_st_only_residual_dataset import apply_st_residual

    result = apply_st_residual(
        np.array([[10.0, 20.0]]),
        np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -2.0], [0.0, 0.0, 1.0]]),
    )
    np.testing.assert_allclose(result, [[15.0, 18.0]])


def test_materialized_anchor_preserves_original_and_adds_corrected_key(tmp_path):
    from scripts.build_00029_st_only_residual_dataset import materialize_anchor_h5ad

    source = _write_anchor(tmp_path / "source" / "section-001.h5ad")
    matrix_path = _write_residual(tmp_path / "residual")
    destination = tmp_path / "out" / "section-001.h5ad"

    record = materialize_anchor_h5ad(
        source,
        destination,
        matrix_path,
        coordinate_frame="00029__g0_registered",
    )

    value = ad.read_h5ad(destination)
    np.testing.assert_allclose(value.obsm["spatial_registered"], [[10.0, 20.0]])
    np.testing.assert_allclose(value.obsm["spatial_st_corrected"], [[15.0, 18.0]])
    assert record["h_and_e_residual_applied"] is False
    contract = value.uns["deepspatial_coordinate_contract"]
    assert bool(contract["st_residual_applied"]) is True
    assert bool(contract["h_and_e_residual_applied"]) is False
    assert contract["coordinate_frame"] == "00029__g0_registered"


def test_he_contract_for_every_section_is_identity_residual_policy():
    from scripts.build_00029_st_only_residual_dataset import build_he_table_contract

    table = pd.DataFrame(
        {
            "section_id": [1, 3],
            "source_kind": ["xenium_companion_he", "serial_he"],
        }
    )
    result = build_he_table_contract(table)
    assert result["residual_applied_to_he"].tolist() == [False, False]
    assert result["he_coordinate_policy"].tolist() == [
        "original_registered_only",
        "original_registered_only",
    ]


def test_builder_rejects_missing_anchor_residual(tmp_path):
    from scripts.build_00029_st_only_residual_dataset import build_anchor_manifest

    source_dir = tmp_path / "anchors"
    _write_anchor(source_dir / "section-001.h5ad")
    with pytest.raises(FileNotFoundError, match="section-001"):
        build_anchor_manifest(
            source_dir=source_dir,
            residual_dir=tmp_path / "residual",
            output_dir=tmp_path / "out",
            sections=[1],
            coordinate_frame="00029__g0_registered",
            he_table_path=None,
        )


def test_builder_marks_only_st_as_corrected(tmp_path):
    from scripts.build_00029_st_only_residual_dataset import build_anchor_manifest

    source_dir = tmp_path / "anchors"
    _write_anchor(source_dir / "section-001.h5ad")
    residual_dir = tmp_path / "residual"
    _write_residual(residual_dir)
    he_table_path = tmp_path / "he_sections.parquet"
    pd.DataFrame(
        {
            "section_id": [1],
            "source_kind": ["xenium_companion_he"],
            "z_um": [0.0],
            "coordinate_frame": ["00029__g0_registered"],
        }
    ).to_parquet(he_table_path)

    result = build_anchor_manifest(
        source_dir=source_dir,
        residual_dir=residual_dir,
        output_dir=tmp_path / "out",
        sections=[1],
        coordinate_frame="00029__g0_registered",
        he_table_path=he_table_path,
    )
    assert result["h_and_e_policy"] == "identity_residual_for_all_he_sections"
    assert result["st_policy"] == "materialized_residual_affine_per_anchor"


def test_builder_keeps_all_he_sections_not_only_anchor_sections(tmp_path):
    from scripts.build_00029_st_only_residual_dataset import build_anchor_manifest

    source_dir = tmp_path / "anchors"
    _write_anchor(source_dir / "section-001.h5ad")
    residual_dir = tmp_path / "residual"
    _write_residual(residual_dir)
    he_table_path = tmp_path / "he_sections.parquet"
    pd.DataFrame(
        {
            "section_id": [1, 2, 3],
            "source_kind": ["xenium_companion_he", "serial_he", "serial_he"],
            "z_um": [0.0, 5.0, 10.0],
            "coordinate_frame": [
                "00029__g0_registered",
                "00029__g0_registered",
                "00029__g0_registered",
            ],
        }
    ).to_parquet(he_table_path)

    build_anchor_manifest(
        source_dir=source_dir,
        residual_dir=residual_dir,
        output_dir=tmp_path / "out",
        sections=[1],
        coordinate_frame="00029__g0_registered",
        he_table_path=he_table_path,
    )

    output = pd.read_parquet(tmp_path / "out" / "he_sections.parquet")
    assert output["section_id"].tolist() == [1, 2, 3]
