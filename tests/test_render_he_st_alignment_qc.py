from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).parents[1] / "scripts" / "render_he_st_alignment_qc.py"
SPEC = importlib.util.spec_from_file_location("render_he_st_alignment_qc", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_resolve_data_path_uses_v2_root_for_relative_paths() -> None:
    v2_root = Path("/tmp/deepspatial_v2")
    assert MODULE.resolve_data_path("outputs/a.tif", v2_root) == v2_root / "outputs/a.tif"
    absolute = Path("/tmp/a.tif")
    assert MODULE.resolve_data_path(str(absolute), v2_root) == absolute


def test_classify_points_against_mask_uses_xy_pixel_order() -> None:
    mask = np.zeros((5, 6), dtype=bool)
    mask[2, 3] = True
    inside, valid = MODULE.classify_points_against_mask(
        mask, np.array([[3.0, 2.0], [0.0, 0.0], [7.0, 1.0]])
    )
    np.testing.assert_array_equal(inside, np.array([True, False, False]))
    np.testing.assert_array_equal(valid, np.array([True, True, False]))


def test_deterministic_point_downsampling_keeps_all_small_inputs() -> None:
    points = np.arange(12, dtype=float).reshape(6, 2)
    result = MODULE.deterministic_point_downsample(points, max_points=10)
    np.testing.assert_array_equal(result, points)


def test_section_record_reads_scalar_metadata_from_section_dataframe(tmp_path: Path) -> None:
    image = tmp_path / "he.tif"
    mask = tmp_path / "mask.png"
    image.touch()
    mask.touch()
    section = pd.DataFrame(
        {
            "raw_crop_path": [str(image), str(image)],
            "he_image_path": ["", ""],
            "he_path": ["", ""],
            "mask_path": [str(mask), str(mask)],
            "he_mask_path": ["", ""],
            "registration_transform_path": ["transform.npz", "transform.npz"],
            "registration_status": ["registered", "registered"],
            "registration_confidence": [0.8, 0.9],
            "forced_flip": [False, False],
            "preorientation_axis": ["none", "none"],
            "source_coordinate_frame": ["xenium", "xenium"],
            "x_he_local_px": [1.0, 2.0],
            "y_he_local_px": [3.0, 4.0],
        }
    )
    record, image_path, mask_path, points = MODULE._section_record(
        11, section, Path("/tmp/deepspatial_v2")
    )
    assert record["section_id"] == 11
    assert np.isclose(record["registration_confidence"], 0.85)
    assert image_path == image
    assert mask_path == mask
    np.testing.assert_allclose(points, [[1.0, 3.0], [2.0, 4.0]])


def test_discover_group_h5ads_excludes_merged_files(tmp_path: Path) -> None:
    for name in [
        "00029_g0.h5ad",
        "9589_g1.h5ad",
        "all_other_groups.h5ad",
        "all_other_groups_annotated_predicted.h5ad",
    ]:
        (tmp_path / name).touch()
    assert [p.name for p in MODULE.discover_group_h5ads(tmp_path)] == [
        "00029_g0.h5ad",
        "9589_g1.h5ad",
    ]
