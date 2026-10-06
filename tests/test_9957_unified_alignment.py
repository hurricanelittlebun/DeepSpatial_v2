from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
V2_SRC = Path(__file__).resolve().parents[2] / "DeepSpatial_v2" / "src"
if str(V2_SRC) not in sys.path:
    sys.path.insert(0, str(V2_SRC))

from deepspatial_v2.serial_registration.stalign_types import EdgeAlignment


def _module():
    return importlib.import_module("9957_unified_alignment")


class _AffineChain:
    def __init__(self, forward_matrix, inverse_matrix=None):
        self.forward_matrix = np.asarray(forward_matrix, dtype=float)
        self.inverse_matrix = (
            np.linalg.inv(self.forward_matrix)
            if inverse_matrix is None
            else np.asarray(inverse_matrix, dtype=float)
        )

    def forward_chunked(self, points, batch_size=16384):
        points = np.asarray(points, dtype=float)
        return _apply(points, self.forward_matrix)

    def inverse_chunked(self, points, batch_size=16384):
        points = np.asarray(points, dtype=float)
        return _apply(points, self.inverse_matrix)


def _apply(points, matrix):
    points = np.asarray(points, dtype=float)
    homogeneous = np.column_stack([points, np.ones(len(points))])
    return (homogeneous @ np.asarray(matrix, dtype=float).T)[:, :2]


def _edge(matrix):
    return EdgeAlignment(
        sample_id="9957",
        group_id="g0",
        fixed_section_id=51,
        moving_section_id=53,
        affine=np.asarray(matrix, dtype=float),
        forward_map=None,
        reverse_map=None,
        transformed_points_um=np.empty((0, 2), dtype=float),
        metrics={},
        status="synthetic",
    )


def test_section_51_is_unchanged_and_downstream_is_composed():
    module = _module()
    base = _AffineChain(
        np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 20.0], [0.0, 0.0, 1.0]])
    )
    edge = _edge(
        np.array([[2.0, 0.0, 100.0], [0.0, 2.0, 200.0], [0.0, 0.0, 1.0]])
    )
    chain = module.Unified9957Chain(base_chain=base, edge_51_to_53=edge)
    points = np.array([[1.0, 2.0], [3.0, 4.0]])

    np.testing.assert_allclose(chain.forward_points(points, 51), points)
    np.testing.assert_allclose(
        chain.forward_points(points, 53),
        _apply(_apply(points, base.forward_matrix), edge.affine),
    )
    np.testing.assert_allclose(
        chain.forward_points(points, 55),
        _apply(_apply(points, base.forward_matrix), edge.affine),
    )


def test_unified_chain_inverse_is_endpoint_consistent():
    module = _module()
    base_matrix = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, -5.0], [0.0, 0.0, 1.0]])
    edge_matrix = np.array([[0.0, -1.0, 100.0], [1.0, 0.0, 50.0], [0.0, 0.0, 1.0]])
    chain = module.Unified9957Chain(
        base_chain=_AffineChain(base_matrix),
        edge_51_to_53=_edge(edge_matrix),
    )
    points = np.array([[5.0, 7.0], [15.0, 11.0]])

    for section_id in (51, 53, 55):
        mapped = chain.forward_points(points, section_id)
        recovered = chain.inverse_points(mapped, section_id)
        np.testing.assert_allclose(recovered, points, atol=1e-8)


def test_raw_crop_local_to_v6_adds_bbox_offset_once():
    module = _module()
    row = {
        "bbox_x0": 10,
        "bbox_y0": 20,
        "analysis_pixel_size_x_um": 2.0,
        "analysis_pixel_size_y_um": 3.0,
    }
    crop_points = np.array([[4.0, 5.0], [0.0, 0.0]])
    expected = np.array([[24.0, 65.0], [20.0, 60.0]])
    np.testing.assert_allclose(module.raw_crop_local_to_v6(row, crop_points), expected)


def test_raw_crop_local_to_v6_rejects_invalid_shape():
    module = _module()
    row = {
        "bbox_x0": 0,
        "bbox_y0": 0,
        "analysis_pixel_size_x_um": 2.0,
        "analysis_pixel_size_y_um": 2.0,
    }
    with pytest.raises(ValueError, match="N x 2"):
        module.raw_crop_local_to_v6(row, np.array([1.0, 2.0]))
