"""Small, explicit helpers for propagating a downstream XY residual.

An edge correction is defined in the already materialized global frame.  The
fixed section and all sections before the moving section remain unchanged;
the moving section and later sections receive the same target-frame residual.
All matrices use the convention ``p_out = M @ p_in`` for homogeneous XY
coordinates.
"""

from __future__ import annotations

import numpy as np


_IDENTITY = np.eye(3, dtype=np.float64)


def _as_affine(value, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 3x3 matrix")
    if not np.allclose(matrix[2], [0.0, 0.0, 1.0], atol=1e-8, rtol=0.0):
        raise ValueError(f"{name} must have homogeneous last row [0,0,1]")
    if abs(float(np.linalg.det(matrix[:2, :2]))) <= 1e-12:
        raise ValueError(f"{name} must be invertible")
    return matrix


def compose_downstream_matrix(
    base_matrix,
    residual_affine,
    *,
    section_id: int,
    start_section_id: int,
) -> np.ndarray:
    """Compose ``residual_affine`` from ``start_section_id`` onward."""

    base = _as_affine(base_matrix, "base_matrix")
    residual = _as_affine(residual_affine, "residual_affine")
    if int(start_section_id) < 0:
        raise ValueError("start_section_id must be non-negative")
    return (residual @ base) if int(section_id) >= int(start_section_id) else base


def apply_downstream_residual(
    points,
    section_ids,
    residual_affine,
    *,
    start_section_id: int,
) -> np.ndarray:
    """Apply a target-frame residual only to downstream points.

    Parameters
    ----------
    points:
        Finite ``[N, 2]`` coordinates.
    section_ids:
        One section ID per point.
    residual_affine:
        Homogeneous XY matrix in the current global frame.
    start_section_id:
        First section receiving the residual.
    """

    value = np.asarray(points, dtype=np.float64)
    sections = np.asarray(section_ids)
    if value.ndim != 2 or value.shape[1] != 2 or not np.isfinite(value).all():
        raise ValueError("points must be finite with shape [N,2]")
    if sections.ndim != 1 or len(sections) != len(value):
        raise ValueError("section_ids must have shape [N]")
    residual = _as_affine(residual_affine, "residual_affine")
    output = value.copy()
    rows = sections.astype(np.int64, copy=False) >= int(start_section_id)
    if rows.any():
        homogeneous = np.column_stack([value[rows], np.ones(int(rows.sum()))])
        output[rows] = (homogeneous @ residual.T)[:, :2]
    return output


__all__ = [
    "apply_downstream_residual",
    "compose_downstream_matrix",
]
