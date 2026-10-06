from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _write_residual(root: Path, section: int, matrix: np.ndarray) -> Path:
    path = root / "transforms" / (
        f"section-{section:03d}__st_to_fixed_he_residual.npz"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, matrix_current_g_to_fixed_he=matrix)
    return path


def test_build_section_affines_uses_residual_for_companion_and_identity_for_serial(
    tmp_path,
):
    from scripts.materialize_00029_fixed_he_feature_stores import (
        build_section_affines,
    )

    residual = np.array(
        [[1.0, 0.0, 12.0], [0.0, 1.0, -3.0], [0.0, 0.0, 1.0]]
    )
    residual_path = _write_residual(tmp_path, 1, residual)
    table = pd.DataFrame(
        {
            "section_id": [1, 3],
            "z_um": [0.0, 10.0],
            "source_kind": ["xenium_companion_he", "serial_he"],
        }
    )

    matrices, records = build_section_affines(table, tmp_path)

    np.testing.assert_allclose(matrices["1"], residual)
    np.testing.assert_allclose(matrices["3"], np.eye(3))
    by_section = {row["section_id"]: row for row in records}
    assert by_section["1"]["transform_type"] == "residual_affine"
    assert by_section["1"]["path"] == str(residual_path)
    assert by_section["3"]["transform_type"] == "identity"
    assert by_section["3"]["path"] is None


def test_build_section_affines_rejects_unknown_source_kind(tmp_path):
    from scripts.materialize_00029_fixed_he_feature_stores import (
        build_section_affines,
    )

    table = pd.DataFrame(
        {
            "section_id": [1],
            "z_um": [0.0],
            "source_kind": ["unverified"],
        }
    )
    with pytest.raises(ValueError, match="source_kind"):
        build_section_affines(table, tmp_path)


def test_build_section_affines_requires_companion_transform(tmp_path):
    from scripts.materialize_00029_fixed_he_feature_stores import (
        build_section_affines,
    )

    table = pd.DataFrame(
        {
            "section_id": [1],
            "z_um": [0.0],
            "source_kind": ["xenium_companion_he"],
        }
    )
    with pytest.raises(FileNotFoundError, match="section 1"):
        build_section_affines(table, tmp_path)


def test_build_section_affines_can_interpolate_serial_frames_between_anchors(tmp_path):
    from scripts.materialize_00029_fixed_he_feature_stores import (
        build_section_affines,
    )

    first = np.array(
        [[1.0, 0.0, 10.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    second = np.array(
        [[1.0, 0.0, 30.0], [0.0, 1.0, 20.0], [0.0, 0.0, 1.0]]
    )
    _write_residual(tmp_path, 1, first)
    _write_residual(tmp_path, 11, second)
    table = pd.DataFrame(
        {
            "section_id": [1, 3, 11],
            "z_um": [0.0, 10.0, 50.0],
            "source_kind": [
                "xenium_companion_he",
                "serial_he",
                "xenium_companion_he",
            ],
        }
    )

    matrices, records = build_section_affines(
        table, tmp_path, interpolate_serial_affine=True
    )

    np.testing.assert_allclose(
        matrices["3"],
        [[1.0, 0.0, 14.0], [0.0, 1.0, 4.0], [0.0, 0.0, 1.0]],
    )
    serial_record = {row["section_id"]: row for row in records}["3"]
    assert serial_record["transform_type"] == "interpolated_affine"
    assert serial_record["bracket_sections"] == ["1", "11"]
    assert serial_record["interpolation_fraction"] == 0.2
