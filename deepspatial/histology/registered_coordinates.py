"""Apply existing H&E registration transforms to physical x/y coordinates."""

from __future__ import annotations

import numpy as np


def _validate_points(points: np.ndarray) -> np.ndarray:
    value = np.asarray(points, dtype=float)
    if value.ndim != 2 or value.shape[1] != 2 or not np.isfinite(value).all():
        raise ValueError("points must be a finite N x 2 array")
    return value


def _apply_homogeneous(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("matrix must be a finite 3 x 3 homogeneous transform")
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=float)])
    transformed = homogeneous @ matrix.T
    if np.any(np.isclose(transformed[:, 2], 0.0)):
        raise ValueError("homogeneous transform produced zero scale")
    return transformed[:, :2] / transformed[:, 2, None]


def source_to_registered_points(points, section, store, *, batch_size=32768):
    """Apply preorientation and an existing nonlinear edge chain."""

    value = _validate_points(points)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    result = np.empty_like(value)
    for start in range(0, len(value), int(batch_size)):
        stop = min(start + int(batch_size), len(value))
        batch = _apply_homogeneous(value[start:stop], section.preorientation_matrix)
        for edge_key, direction in section.edge_chain:
            batch = store.transform(batch, store.load_edge(edge_key), direction)
        result[start:stop] = batch
    if not np.isfinite(result).all():
        raise ValueError("registration produced non-finite coordinates")
    return result


def registered_to_source_points(points, section, store, *, batch_size=32768):
    """Invert a saved registration chain back to its source H&E frame."""

    value = _validate_points(points)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    inverse_preorientation = np.linalg.inv(
        np.asarray(section.preorientation_matrix, dtype=float)
    )
    result = np.empty_like(value)
    opposite = {"forward": "reverse", "reverse": "forward"}
    for start in range(0, len(value), int(batch_size)):
        stop = min(start + int(batch_size), len(value))
        batch = value[start:stop].copy()
        for edge_key, direction in reversed(section.edge_chain):
            if direction not in opposite:
                raise ValueError(f"unknown edge direction: {direction}")
            batch = store.transform(
                batch, store.load_edge(edge_key), opposite[direction]
            )
        result[start:stop] = _apply_homogeneous(batch, inverse_preorientation)
    if not np.isfinite(result).all():
        raise ValueError("inverse registration produced non-finite coordinates")
    return result


def source_um_to_crop_px(points, *, mpp, bbox_origin_px):
    """Convert source-image physical coordinates to masked-crop pixels."""

    value = _validate_points(points)
    scale = np.asarray(mpp, dtype=float)
    origin = np.asarray(bbox_origin_px, dtype=float)
    if (
        scale.shape != (2,)
        or origin.shape != (2,)
        or np.any(scale <= 0)
        or not np.isfinite(scale).all()
        or not np.isfinite(origin).all()
    ):
        raise ValueError("mpp and bbox_origin_px must be finite x/y pairs")
    return value / scale - origin


def crop_px_to_source_um(points, *, mpp, bbox_origin_px):
    """Convert masked-crop pixels to source-image physical coordinates."""

    value = _validate_points(points)
    scale = np.asarray(mpp, dtype=float)
    origin = np.asarray(bbox_origin_px, dtype=float)
    if (
        scale.shape != (2,)
        or origin.shape != (2,)
        or np.any(scale <= 0)
        or not np.isfinite(scale).all()
        or not np.isfinite(origin).all()
    ):
        raise ValueError("mpp and bbox_origin_px must be finite x/y pairs")
    return (value + origin) * scale
