"""Coordinate composition for the final 9957 manual-alignment release.

The registered feature grids were materialized from an older section transform
chain.  This module keeps the direction explicit when moving those grids into
the final global frame:

    p_old_grid -> global_frame -> final section registration -> manual residual

All matrices are physical XY homogeneous affines.  The functions are small and
dependency-light so the transform convention can be regression-tested without
opening any large H5AD or feature store.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

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


def build_propagated_section_extras(
    section_ids: Iterable[int | str],
    manual_edges: Mapping[int | str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Propagate an edge correction from its moving section downstream.

    An edge ``fixed -> moving`` changes the global pose beginning at the moving
    section.  Later sections inherit that pose so the serial chain does not
    introduce a new coordinate discontinuity.  ``manual_edges`` therefore maps
    the moving section ID to the target-frame residual affine.
    """

    ids = sorted({int(value) for value in section_ids})
    edges = {
        int(section): _as_affine(matrix, f"manual_edge_{section}")
        for section, matrix in manual_edges.items()
    }
    result: dict[str, np.ndarray] = {}
    cumulative = _IDENTITY.copy()
    for section_id in ids:
        if section_id in edges:
            cumulative = edges[section_id] @ cumulative
        result[str(section_id)] = cumulative.copy()
    return result


def build_relative_feature_affine(
    old_transform: np.ndarray,
    final_transform: np.ndarray,
    global_frame_affine: np.ndarray,
    extra_manual_affine: np.ndarray | None = None,
) -> np.ndarray:
    """Map an old registered feature grid into the final global frame.

    The existing feature grid is already expressed as ``global_frame_affine @
    p_old``.  The desired final point is ``extra @ global_frame_affine @
    final_transform @ inv(old_transform) @ p_old``.  Returning the relative
    matrix lets :func:`resample_feature_store` update the existing grid without
    rerunning UNI2.
    """

    old = _as_affine(old_transform, "old_transform")
    final = _as_affine(final_transform, "final_transform")
    global_frame = _as_affine(global_frame_affine, "global_frame_affine")
    extra = (
        _IDENTITY
        if extra_manual_affine is None
        else _as_affine(extra_manual_affine, "extra_manual_affine")
    )
    return extra @ global_frame @ final @ np.linalg.inv(old) @ np.linalg.inv(global_frame)


def compose_final_section_transform(
    registered_transform: np.ndarray,
    propagated_manual_affine: np.ndarray | None = None,
) -> np.ndarray:
    """Compose a propagated target-frame residual onto a section transform."""

    registered = _as_affine(registered_transform, "registered_transform")
    extra = (
        _IDENTITY
        if propagated_manual_affine is None
        else _as_affine(propagated_manual_affine, "propagated_manual_affine")
    )
    return extra @ registered


__all__ = [
    "build_propagated_section_extras",
    "build_relative_feature_affine",
    "compose_final_section_transform",
]
