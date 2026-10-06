from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np


_MODULE_PATH = Path(__file__).resolve().parents[1] / "deepspatial" / "data_utils" / "section_filter.py"
_SPEC = spec_from_file_location("section_filter_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_drop_section_54_makes_53_and_55_adjacent():
    kept = _MODULE.filter_section_ids([51, 53, 54, 55, 56], {54})
    assert kept == ["51", "53", "55", "56"]
    assert _MODULE.adjacent_pairs(kept)[1] == ("53", "55")


def test_direct_affine_composes_surviving_global_sections():
    left = np.asarray(
        [[1.0, 0.0, 10.0], [0.0, 1.0, 20.0], [0.0, 0.0, 1.0]]
    )
    right = np.asarray(
        [[0.0, -1.0, 100.0], [1.0, 0.0, 200.0], [0.0, 0.0, 1.0]]
    )
    direct = _MODULE.compose_direct_affine(left, right)
    expected = np.linalg.inv(right) @ left
    np.testing.assert_allclose(direct, expected)
