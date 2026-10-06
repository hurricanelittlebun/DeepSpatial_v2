from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest

from scripts.render_9957_coordinate_repair_qc import (
    feature_grid_boundary,
    mask_pixels_to_global,
)



_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "deepspatial"
    / "data_utils"
    / "coordinate_repair.py"
)
_SPEC = spec_from_file_location("coordinate_repair_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
affine_for_feature_provenance = _MODULE.affine_for_feature_provenance
build_repaired_obsm = _MODULE.build_repaired_obsm
build_section_affines = _MODULE.build_section_affines
build_global_v4_affines = _MODULE.build_global_v4_affines
build_global_v4_coordinates = _MODULE.build_global_v4_coordinates
require_new_output = _MODULE.require_new_output
select_repaired_coordinates = _MODULE.select_repaired_coordinates


def test_select_repaired_coordinates_prefers_pre_manual_frame():
    pre_manual = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    v4 = pre_manual + 1000.0

    repaired = select_repaired_coordinates(
        {
            "spatial_st_corrected_pre_manual_v4": pre_manual,
            "spatial_st_corrected_v4": v4,
        }
    )

    np.testing.assert_array_equal(repaired, pre_manual)


def test_affine_for_unchanged_feature_section_is_identity():
    manual = np.asarray(
        [[0.8, -0.2, 100.0], [0.2, 0.8, -40.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )

    result = affine_for_feature_provenance(
        "copied_unchanged_pre_manual_v4", manual
    )

    np.testing.assert_allclose(result, np.eye(3))


def test_affine_for_manual_feature_section_is_inverse_of_v4_affine():
    manual = np.asarray(
        [[2.0, 0.0, 3.0], [0.0, 4.0, 5.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )

    result = affine_for_feature_provenance(
        "inverse_query_old_field_in_new_v4_frame", manual
    )

    np.testing.assert_allclose(result, np.linalg.inv(manual))


def test_require_new_output_rejects_existing_path(tmp_path: Path):
    existing = tmp_path / "already_exists"
    existing.mkdir()

    with pytest.raises(FileExistsError):
        require_new_output(existing)


def test_require_new_output_accepts_missing_path(tmp_path: Path):
    missing = tmp_path / "new_output"

    assert require_new_output(missing) == missing
    assert not missing.exists()


def test_build_repaired_obsm_replaces_training_coordinate_keys_only():
    pre_manual = np.asarray([[1.0, 2.0]], dtype=np.float32)
    v4 = np.asarray([[100.0, 200.0]], dtype=np.float32)
    other = np.asarray([[7.0, 8.0]], dtype=np.float32)

    repaired = build_repaired_obsm(
        {
            "spatial": v4,
            "spatial_registered": v4,
            "spatial_st_corrected": v4,
            "spatial_st_corrected_pre_manual_v4": pre_manual,
            "spatial_st_corrected_v4": v4,
            "spatial_he_local": other,
        }
    )

    for key in ("spatial", "spatial_registered", "spatial_st_corrected"):
        np.testing.assert_array_equal(repaired[key], pre_manual)
    np.testing.assert_array_equal(repaired["spatial_st_corrected_v4"], v4)
    np.testing.assert_array_equal(repaired["spatial_he_local"], other)


def test_build_section_affines_uses_operation_metadata_and_explicit_nucleus_ids():
    manual = np.asarray(
        [[2.0, 0.0, 3.0], [0.0, 4.0, 5.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    metadata = {
        "51": {"provenance": {"coordinate_operation": "copied_unchanged_pre_manual_v4"}},
        "55": {"provenance": {"coordinate_operation": "inverse_query_old_field_in_new_v4_frame"}},
        "61": {"provenance": {}},
    }

    affines = build_section_affines(metadata, manual, manual_section_ids={"61"})

    np.testing.assert_allclose(affines["51"], np.eye(3))
    np.testing.assert_allclose(affines["55"], np.linalg.inv(manual))
    np.testing.assert_allclose(affines["61"], np.linalg.inv(manual))


def test_build_global_v4_affines_moves_old_sections_into_confirmed_v4_frame():
    manual = np.asarray(
        [[2.0, 0.0, 3.0], [0.0, 4.0, 5.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    metadata = {
        "51": {"provenance": {"coordinate_operation": "copied_unchanged_pre_manual_v4"}},
        "55": {"provenance": {"coordinate_operation": "inverse_query_old_field_in_new_v4_frame"}},
    }

    affines = build_global_v4_affines(metadata, manual)

    np.testing.assert_allclose(affines["51"], manual)
    np.testing.assert_allclose(affines["55"], np.eye(3))


def test_build_global_v4_coordinates_transforms_old_and_keeps_v4_rows():
    manual = np.asarray(
        [[2.0, 0.0, 3.0], [0.0, 4.0, 5.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    pre = np.asarray([[1.0, 2.0], [10.0, 20.0]], dtype=np.float32)
    v4 = np.asarray([[100.0, 200.0], [23.0, 85.0]], dtype=np.float32)

    repaired = build_global_v4_coordinates(
        pre,
        v4,
        np.asarray([51, 61]),
        manual,
        manual_section_ids={"61"},
    )

    np.testing.assert_allclose(repaired[0], [5.0, 13.0])
    np.testing.assert_allclose(repaired[1], v4[1])


def test_mask_pixels_to_global_uses_xy_physical_scale_and_affine():
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:4, 1:4] = True
    affine = np.asarray(
        [[1.0, 0.0, 10.0], [0.0, 1.0, -4.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )

    points = mask_pixels_to_global(mask, (2.0, 3.0), affine)

    assert points.ndim == 2 and points.shape[1] == 2
    assert [12.0, -1.0] in points.tolist()


def test_feature_grid_boundary_uses_grid_centers_and_physical_spacing():
    valid = np.zeros((4, 5), dtype=bool)
    valid[1:3, 1:4] = True

    points = feature_grid_boundary(
        valid,
        origin_um=(100.0, 200.0),
        spacing_um=(10.0, 20.0),
    )

    assert points.ndim == 2 and points.shape[1] == 2
    assert [110.0, 220.0] in points.tolist()
    assert [130.0, 240.0] in points.tolist()
