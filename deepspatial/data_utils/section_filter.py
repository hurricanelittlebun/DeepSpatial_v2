"""Small, explicit helpers for releases with removed serial sections."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def filter_section_ids(
    section_ids: Iterable[int | str], removed_section_ids: Iterable[int | str]
) -> list[str]:
    """Return section IDs in input order after removing exact section IDs."""

    removed = {int(value) for value in removed_section_ids}
    result = [str(value) for value in section_ids if int(value) not in removed]
    if len(result) != len(set(result)):
        raise ValueError("section IDs must be unique")
    return result


def adjacent_pairs(section_ids: Iterable[int | str]) -> list[tuple[str, str]]:
    """Return consecutive section pairs in the supplied physical order."""

    values = [str(value) for value in section_ids]
    if len(values) != len(set(values)):
        raise ValueError("section IDs must be unique")
    return list(zip(values[:-1], values[1:]))


def compose_direct_affine(
    left_to_global: np.ndarray, right_to_global: np.ndarray
) -> np.ndarray:
    """Compose an affine from the left section frame directly to the right."""

    left = np.asarray(left_to_global, dtype=np.float64)
    right = np.asarray(right_to_global, dtype=np.float64)
    for name, matrix in (("left_to_global", left), ("right_to_global", right)):
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError(f"{name} must be a finite 3x3 matrix")
        if not np.allclose(matrix[2], [0.0, 0.0, 1.0], atol=1e-8, rtol=0.0):
            raise ValueError(f"{name} must have homogeneous last row [0,0,1]")
        if abs(float(np.linalg.det(matrix[:2, :2]))) <= 1e-12:
            raise ValueError(f"{name} must be invertible")
    # T_left and T_right both map their respective source frames into the
    # same global frame.  Therefore a point in the left source frame reaches
    # the right source frame via inv(T_right) @ T_left.
    return np.linalg.inv(right) @ left


__all__ = ["adjacent_pairs", "compose_direct_affine", "filter_section_ids"]
