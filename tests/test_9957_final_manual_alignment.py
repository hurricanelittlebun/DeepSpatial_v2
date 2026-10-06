from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np


_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "deepspatial"
    / "data_utils"
    / "final_manual_alignment.py"
)
_SPEC = spec_from_file_location("final_manual_alignment_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

build_propagated_section_extras = _MODULE.build_propagated_section_extras
build_relative_feature_affine = _MODULE.build_relative_feature_affine


def test_manual_edge_residual_is_propagated_from_moving_section_forward():
    manual = np.asarray(
        [[0.0, -1.0, 100.0], [1.0, 0.0, -20.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )

    extras = build_propagated_section_extras(
        [54, 55, 61, 91], {55: manual}
    )

    np.testing.assert_allclose(extras["54"], np.eye(3))
    np.testing.assert_allclose(extras["55"], manual)
    np.testing.assert_allclose(extras["61"], manual)
    np.testing.assert_allclose(extras["91"], manual)


def test_relative_feature_affine_maps_old_grid_into_final_global_frame():
    old_transform = np.asarray(
        [[0.0, -1.0, 20.0], [1.0, 0.0, 30.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    final_transform = np.asarray(
        [[1.0, 0.0, 50.0], [0.0, 1.0, -10.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    global_frame = np.asarray(
        [[2.0, 0.0, 100.0], [0.0, 2.0, 200.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    extra = np.asarray(
        [[1.0, 0.0, 7.0], [0.0, 1.0, 9.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )

    result = build_relative_feature_affine(
        old_transform,
        final_transform,
        global_frame,
        extra,
    )
    expected = extra @ global_frame @ final_transform @ np.linalg.inv(old_transform) @ np.linalg.inv(global_frame)
    np.testing.assert_allclose(result, expected)
