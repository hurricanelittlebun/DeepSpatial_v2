"""Coordinate composition for the reviewed 9957/g0 51-to-53 repair.

The public point convention is physical x/y in the old v6/v9 registration
frame.  The new release keeps section 51 fixed, applies the existing v9
53->55 chain to section 53 and downstream sections, and then applies the new
51->53 edge.  This is the only composition that matches how the new edge was
fitted: its moving input is section 53 already rendered in the reviewed v9
post-processing frame.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


FIXED_SECTION = 51
NEW_EDGE_SECTION = 53
DOWNSTREAM_SECTION = 55


def _as_points(points: np.ndarray) -> np.ndarray:
    value = np.asarray(points, dtype=float)
    if value.ndim != 2 or value.shape[1] != 2:
        raise ValueError("points must have shape N x 2")
    if not np.isfinite(value).all():
        raise ValueError("points must be finite")
    return value


def _apply_affine(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    value = _as_points(points)
    transform = np.asarray(matrix, dtype=float)
    if transform.shape != (3, 3) or not np.isfinite(transform).all():
        raise ValueError("affine matrix must be a finite 3 x 3 array")
    if not np.allclose(transform[2], [0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError("affine matrix must have homogeneous last row [0, 0, 1]")
    homogeneous = np.column_stack([value, np.ones(len(value), dtype=float)])
    return (homogeneous @ transform.T)[:, :2]


def _value(row: Mapping[str, Any] | Any, key: str) -> Any:
    if isinstance(row, Mapping):
        return row[key]
    return getattr(row, key)


def raw_crop_local_to_v6(row: Mapping[str, Any] | Any, crop_xy_um: np.ndarray) -> np.ndarray:
    """Convert crop-local physical x/y to the v6 source physical frame.

    ``crop_xy_um`` is already in micrometers relative to the saved crop.  The
    crop bounding-box offset is therefore converted to micrometers exactly
    once and added; no section registration matrix is applied here.
    """

    points = _as_points(crop_xy_um)
    x0 = float(_value(row, "bbox_x0"))
    y0 = float(_value(row, "bbox_y0"))
    sx = float(_value(row, "analysis_pixel_size_x_um"))
    sy = float(_value(row, "analysis_pixel_size_y_um"))
    values = np.array([x0, y0, sx, sy], dtype=float)
    if not np.isfinite(values).all() or sx <= 0.0 or sy <= 0.0:
        raise ValueError("bbox and analysis pixel sizes must be finite; pixel sizes must be positive")
    return points + np.array([x0 * sx, y0 * sy], dtype=float)


def _edge_forward(points: np.ndarray, edge: Any) -> np.ndarray:
    try:
        from deepspatial_v2.serial_registration.stalign_backend import transform_points_with_edge
    except ImportError:
        transform_points_with_edge = None
    if transform_points_with_edge is not None:
        return np.asarray(transform_points_with_edge(points, edge), dtype=float)

    value = _apply_affine(points, np.asarray(edge.affine, dtype=float))
    residual = getattr(edge, "residual_affine", None)
    if residual is not None:
        value = _apply_affine(value, np.asarray(residual, dtype=float))
    return value


def _edge_inverse(points: np.ndarray, edge: Any) -> np.ndarray:
    try:
        from deepspatial_v2.serial_registration.stalign_backend import transform_points_target_to_source_with_edge
    except ImportError:
        transform_points_target_to_source_with_edge = None
    if transform_points_target_to_source_with_edge is not None:
        return np.asarray(transform_points_target_to_source_with_edge(points, edge), dtype=float)

    residual = getattr(edge, "residual_affine", None)
    value = _as_points(points)
    if residual is not None:
        value = _apply_affine(value, np.linalg.inv(np.asarray(residual, dtype=float)))
    return _apply_affine(value, np.linalg.inv(np.asarray(edge.affine, dtype=float)))


class Unified9957Chain:
    """Compose the existing v9 chain and the new 51->53 edge."""

    def __init__(
        self,
        *,
        base_chain: Any,
        edge_51_to_53: Any,
        new_edge_section: int = NEW_EDGE_SECTION,
        downstream_section: int = DOWNSTREAM_SECTION,
    ) -> None:
        self.base_chain = base_chain
        self.edge_51_to_53 = edge_51_to_53
        self.new_edge_section = int(new_edge_section)
        self.downstream_section = int(downstream_section)

    def forward_points(self, points: np.ndarray, section_id: int) -> np.ndarray:
        """Map old-frame points into the new unified frame."""

        value = _as_points(points)
        section = int(section_id)
        if section >= self.new_edge_section:
            value = np.asarray(self.base_chain.forward_chunked(value), dtype=float)
        if section >= self.new_edge_section:
            value = _edge_forward(value, self.edge_51_to_53)
        return value

    def inverse_points(self, points: np.ndarray, section_id: int) -> np.ndarray:
        """Map new-frame points back into the old v6/v9 frame."""

        value = _as_points(points)
        section = int(section_id)
        if section >= self.new_edge_section:
            value = _edge_inverse(value, self.edge_51_to_53)
        if section >= self.new_edge_section:
            value = np.asarray(self.base_chain.inverse_chunked(value), dtype=float)
        return value


__all__ = ["Unified9957Chain", "raw_crop_local_to_v6"]
