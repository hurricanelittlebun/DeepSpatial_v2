from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


_MODULE_PATH = Path(__file__).parents[1] / "scripts" / "render_9957_final_st_he_3d.py"
if not _MODULE_PATH.exists():
    pytest.fail(f"production module is missing: {_MODULE_PATH}", pytrace=False)
_SPEC = importlib.util.spec_from_file_location("_render_9957_final_st_he_3d", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_z_display_preserves_section_order() -> None:
    displayed = _MODULE.z_display_values(np.array([0.0, 10.0, 450.0]), 5.0)
    np.testing.assert_allclose(displayed, [0.0, 50.0, 2250.0])
    assert np.all(np.diff(displayed) > 0)


def test_coordinate_frame_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="coordinate frame mismatch"):
        _MODULE.validate_coordinate_frames("he-frame", "st-frame")
