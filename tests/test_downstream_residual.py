from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "deepspatial"
        / "data_utils"
        / "downstream_residual.py"
    )
    spec = importlib.util.spec_from_file_location("downstream_residual_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_downstream_residual_is_identity_before_edge_and_composes_after_edge():
    compose_downstream_matrix = _module().compose_downstream_matrix

    base = np.array(
        [[1.0, 0.0, 10.0], [0.0, 1.0, 20.0], [0.0, 0.0, 1.0]]
    )
    residual = np.array(
        [[0.0, -1.0, 100.0], [1.0, 0.0, -50.0], [0.0, 0.0, 1.0]]
    )

    np.testing.assert_allclose(
        compose_downstream_matrix(base, residual, section_id=53, start_section_id=55),
        base,
    )
    np.testing.assert_allclose(
        compose_downstream_matrix(base, residual, section_id=55, start_section_id=55),
        residual @ base,
    )
    np.testing.assert_allclose(
        compose_downstream_matrix(base, residual, section_id=91, start_section_id=55),
        residual @ base,
    )


def test_apply_downstream_residual_changes_only_downstream_points():
    apply_downstream_residual = _module().apply_downstream_residual

    points = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    sections = np.array([51, 53, 55])
    residual = np.array(
        [[1.0, 0.0, 100.0], [0.0, 1.0, -20.0], [0.0, 0.0, 1.0]]
    )

    result = apply_downstream_residual(
        points, sections, residual, start_section_id=55
    )
    np.testing.assert_allclose(result[:2], points[:2])
    np.testing.assert_allclose(result[2], [105.0, -14.0])
