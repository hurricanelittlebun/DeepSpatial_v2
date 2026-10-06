from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "materialize_group_st_to_he_residual.py"
SPEC = importlib.util.spec_from_file_location("materialize_group_st_to_he_residual", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_assemble_corrected_coordinates_preserves_section_order() -> None:
    original = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    section_ids = np.array([1, 1, 11, 11])
    candidates = {
        1: original[:2] + 10.0,
        11: original[2:] + 20.0,
    }
    result = MODULE.assemble_corrected_coordinates(original, section_ids, candidates)
    np.testing.assert_array_equal(
        result,
        [[11.0, 11.0], [12.0, 12.0], [23.0, 23.0], [24.0, 24.0]],
    )


def test_candidate_destination_is_group_specific(tmp_path: Path) -> None:
    path = MODULE.candidate_h5ad_path(tmp_path, "9589_g1")
    assert path == tmp_path / "9589_g1" / "registration_correction_he_v1" / "9589_g1__st_to_fixed_he_candidate.h5ad"


def test_section_summary_is_h5ad_serializable_json_text() -> None:
    summary = [{"section_id": 1, "status": "ok"}]
    encoded = MODULE.encode_section_summary(summary)
    assert isinstance(encoded, str)
    assert '"section_id": 1' in encoded
