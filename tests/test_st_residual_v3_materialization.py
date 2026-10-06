from __future__ import annotations

import numpy as np
import pytest

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_MODULE_PATH = Path(__file__).parents[1] / "deepspatial" / "data_utils" / "st_residual_v3.py"
if not _MODULE_PATH.exists():
    pytest.fail(f"production module is missing: {_MODULE_PATH}", pytrace=False)
_SPEC = spec_from_file_location("_st_residual_v3_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
assemble_section_coordinates = _MODULE.assemble_section_coordinates


def test_assemble_section_coordinates_restores_original_row_order() -> None:
    section_ids = np.array([11, 1, 11, 1, 21], dtype=np.int64)
    candidate_by_section = {
        1: {
            "st_global_base": np.array([[10.0, 10.0], [20.0, 20.0]]),
            "st_global_candidate": np.array([[11.0, 10.0], [21.0, 20.0]]),
        },
        11: {
            "st_global_base": np.array([[110.0, 10.0], [120.0, 20.0]]),
            "st_global_candidate": np.array([[111.0, 10.0], [121.0, 20.0]]),
        },
        21: {
            "st_global_base": np.array([[210.0, 10.0]]),
            "st_global_candidate": np.array([[211.0, 10.0]]),
        },
    }

    base, candidate = assemble_section_coordinates(section_ids, candidate_by_section)

    np.testing.assert_allclose(
        base,
        [[110.0, 10.0], [10.0, 10.0], [120.0, 20.0], [20.0, 20.0], [210.0, 10.0]],
    )
    np.testing.assert_allclose(
        candidate,
        [[111.0, 10.0], [11.0, 10.0], [121.0, 20.0], [21.0, 20.0], [211.0, 10.0]],
    )


def test_assemble_section_coordinates_rejects_count_mismatch() -> None:
    section_ids = np.array([1, 1, 11], dtype=np.int64)
    candidate_by_section = {
        1: {
            "st_global_base": np.zeros((1, 2)),
            "st_global_candidate": np.zeros((1, 2)),
        },
        11: {
            "st_global_base": np.zeros((1, 2)),
            "st_global_candidate": np.zeros((1, 2)),
        },
    }

    with pytest.raises(ValueError, match="section 1.*2 rows.*1"):
        assemble_section_coordinates(section_ids, candidate_by_section)
