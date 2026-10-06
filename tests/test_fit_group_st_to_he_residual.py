from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from PIL import Image


SCRIPT = Path(__file__).parents[1] / "scripts" / "fit_group_st_to_he_residual.py"
SPEC = importlib.util.spec_from_file_location("fit_group_st_to_he_residual", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_parse_group_stem_rejects_merged_h5ad_names() -> None:
    assert MODULE.parse_group_stem("9589_g1") == ("9589", "g1")
    with pytest.raises(ValueError):
        MODULE.parse_group_stem("all_other_groups")


def test_apply_current_to_fixed_uses_saved_residual_direction() -> None:
    points = np.array([[10.0, 20.0], [0.0, 0.0]])
    matrix = np.array(
        [[1.0, 0.0, 5.0], [0.0, 1.0, -3.0], [0.0, 0.0, 1.0]]
    )
    np.testing.assert_allclose(
        MODULE.apply_current_to_fixed(points, matrix),
        [[15.0, 17.0], [5.0, -3.0]],
    )


def test_residual_output_directory_is_group_specific(tmp_path: Path) -> None:
    output = MODULE.group_output_dir(tmp_path, "9589_g1")
    assert output == tmp_path / "9589_g1" / "registration_correction_he_v1"


def test_evaluation_exposes_current_to_fixed_matrix_for_materialization() -> None:
    matrix = np.eye(3)
    evaluation = {"matrix_current_g_to_fixed_he": matrix}
    np.testing.assert_array_equal(
        MODULE.current_to_fixed_matrix(evaluation), matrix
    )


def test_before_after_plot_receives_optimizer_parameters_explicitly(tmp_path: Path) -> None:
    image_path = tmp_path / "he.png"
    mask_path = tmp_path / "mask.png"
    Image.new("RGB", (20, 20), "white").save(image_path)
    Image.new("L", (20, 20), 255).save(mask_path)
    section = MODULE.SectionInput(
        sample_id="9589",
        group_id="g1",
        group_stem="9589_g1",
        section_id=1,
        registered=np.array([[5.0, 5.0], [10.0, 10.0]]),
        reported_local_px=np.array([[5.0, 5.0], [10.0, 10.0]]),
        image_path=image_path,
        mask_path=mask_path,
        transform_path=tmp_path / "transform.npz",
        mpp_x=1.0,
        mpp_y=1.0,
        registration_confidence=1.0,
        registration_status="ok",
        forced_flip=False,
        preorientation_axis="none",
        source_coordinate_frame="test",
    )
    baseline = {
        "local_coordinates_um": section.registered,
        "inside_fraction": 1.0,
        "inside_valid_fraction": 1.0,
        "median_signed_distance_um": 1.0,
    }
    candidate = dict(baseline)
    output_path = tmp_path / "overlay.png"
    MODULE._plot_before_after(
        section, baseline, candidate, np.zeros(5), output_path, max_plot_points=100
    )
    assert output_path.is_file()
