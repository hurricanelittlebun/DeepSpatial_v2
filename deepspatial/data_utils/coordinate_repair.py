"""Small, dependency-light helpers for repairing a broken registered frame.

The 9957/g0 v4 manual-rotation materialization used two coordinate frames in
one series.  These helpers keep the frame decision explicit and fail closed
for provenance values that have not been audited.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np


PRE_MANUAL_COORDINATE_KEY = "spatial_st_corrected_pre_manual_v4"
MANUAL_COORDINATE_KEY = "spatial_st_corrected_v4"
UNCHANGED_FEATURE_PROVENANCE = "copied_unchanged_pre_manual_v4"
MANUAL_FEATURE_PROVENANCE = "inverse_query_old_field_in_new_v4_frame"


def select_repaired_coordinates(
    obsm: Mapping[str, object],
    *,
    preferred_key: str = PRE_MANUAL_COORDINATE_KEY,
) -> np.ndarray:
    """Return coordinates in the audited pre-manual registered frame.

    ``preferred_key`` is explicit so a future repair can name a different
    audited source.  The returned array is a float copy, preventing later
    in-place edits from mutating the source AnnData object.
    """

    if preferred_key not in obsm:
        raise KeyError(
            f"Required repaired coordinate key {preferred_key!r} is missing; "
            "refusing to guess between coordinate frames"
        )
    coordinates = np.asarray(obsm[preferred_key], dtype=np.float32)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(
            f"{preferred_key!r} must have shape [n, 2], got {coordinates.shape}"
        )
    if not np.isfinite(coordinates).all():
        raise ValueError(f"{preferred_key!r} contains non-finite coordinates")
    return coordinates.copy()


def build_repaired_obsm(
    obsm: Mapping[str, object],
    *,
    training_keys: tuple[str, ...] = (
        "spatial",
        "spatial_registered",
        "spatial_st_corrected",
    ),
) -> dict[str, np.ndarray]:
    """Copy ``obsm`` and put audited coordinates in training-facing keys."""

    repaired = {
        str(key): np.asarray(value).copy() for key, value in obsm.items()
    }
    coordinates = select_repaired_coordinates(repaired)
    for key in training_keys:
        repaired[key] = coordinates.copy()
    return repaired


def affine_for_feature_provenance(
    provenance: str,
    manual_affine: np.ndarray,
) -> np.ndarray:
    """Return the v4-source-to-repaired-frame affine for one feature section.

    Feature sections copied without the v4 manual rotation stay unchanged.
    Sections sampled into the v4 manual frame are mapped back with the inverse
    of the audited manual affine.  Unknown provenance is rejected instead of
    silently applying the wrong transform.
    """

    matrix = np.asarray(manual_affine, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"manual_affine must have shape (3, 3), got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("manual_affine contains non-finite values")
    if provenance == UNCHANGED_FEATURE_PROVENANCE:
        return np.eye(3, dtype=float)
    if provenance == MANUAL_FEATURE_PROVENANCE:
        return np.linalg.inv(matrix)
    raise ValueError(
        "Unknown feature provenance; refusing to guess a coordinate transform: "
        f"{provenance!r}"
    )


def build_section_affines(
    metadata_by_section: Mapping[str, Mapping[str, object]],
    manual_affine: np.ndarray,
    *,
    manual_section_ids: set[str] | frozenset[str] = frozenset(),
) -> dict[str, np.ndarray]:
    """Build a source-store-to-repaired-frame affine for every section.

    UNI2 metadata records ``coordinate_operation`` directly.  The nucleus
    store predates that field, so its audited update manifest can provide
    ``manual_section_ids`` explicitly.  Any remaining section defaults to the
    audited unchanged operation; no unlisted section receives the inverse
    affine accidentally.
    """

    manual_ids = {str(value) for value in manual_section_ids}
    result: dict[str, np.ndarray] = {}
    for section_id, metadata in metadata_by_section.items():
        sid = str(section_id)
        provenance = metadata.get("provenance", {})
        if not isinstance(provenance, Mapping):
            provenance = {}
        operation = provenance.get("coordinate_operation")
        if operation is None:
            operation = (
                MANUAL_FEATURE_PROVENANCE
                if sid in manual_ids
                else UNCHANGED_FEATURE_PROVENANCE
            )
        result[sid] = affine_for_feature_provenance(str(operation), manual_affine)
    return result


def build_global_v4_affines(
    metadata_by_section: Mapping[str, Mapping[str, object]],
    manual_affine: np.ndarray,
    *,
    manual_section_ids: set[str] | frozenset[str] = frozenset(),
) -> dict[str, np.ndarray]:
    """Build affines that place every section in the confirmed v4 frame.

    A section copied unchanged in the old v4 materialization is still in the
    pre-manual frame and therefore receives ``manual_affine``.  A section
    already resampled into the v4 frame receives identity.  This is the
    inverse direction of :func:`build_section_affines` and is intentionally a
    separate API to make the direction difficult to confuse.
    """

    manual_ids = {str(value) for value in manual_section_ids}
    result: dict[str, np.ndarray] = {}
    for section_id, metadata in metadata_by_section.items():
        sid = str(section_id)
        provenance = metadata.get("provenance", {})
        if not isinstance(provenance, Mapping):
            provenance = {}
        operation = provenance.get("coordinate_operation")
        if operation is None:
            operation = (
                MANUAL_FEATURE_PROVENANCE
                if sid in manual_ids
                else UNCHANGED_FEATURE_PROVENANCE
            )
        if operation == UNCHANGED_FEATURE_PROVENANCE:
            result[sid] = np.asarray(manual_affine, dtype=float).copy()
        elif operation == MANUAL_FEATURE_PROVENANCE:
            result[sid] = np.eye(3, dtype=float)
        else:
            raise ValueError(
                "Unknown feature provenance; refusing to guess a v4-frame transform: "
                f"{operation!r}"
            )
    return result


def build_global_v4_coordinates(
    pre_manual_coordinates: np.ndarray,
    v4_coordinates: np.ndarray,
    section_ids: np.ndarray,
    manual_affine: np.ndarray,
    *,
    manual_section_ids: set[str] | frozenset[str] = frozenset(),
) -> np.ndarray:
    """Put ST rows into the confirmed global v4 frame.

    Rows from sections already materialized in v4 keep their v4 coordinates.
    All other rows use the audited pre-manual coordinates followed by the
    same global affine that defines the confirmed H&E v4 frame.
    """

    pre = np.asarray(pre_manual_coordinates, dtype=np.float64)
    current = np.asarray(v4_coordinates, dtype=np.float64)
    sections = np.asarray(section_ids)
    if pre.ndim != 2 or pre.shape[1] != 2:
        raise ValueError("pre_manual_coordinates must have shape [N,2]")
    if current.shape != pre.shape:
        raise ValueError("v4_coordinates must have the same shape as pre_manual_coordinates")
    if sections.shape != (len(pre),):
        raise ValueError("section_ids must have shape [N]")
    matrix = np.asarray(manual_affine, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("manual_affine must have shape (3,3)")
    if not np.isfinite(pre).all() or not np.isfinite(current).all():
        raise ValueError("ST coordinates must be finite")
    homogeneous = np.concatenate(
        [pre, np.ones((len(pre), 1), dtype=np.float64)], axis=1
    )
    transformed = (homogeneous @ matrix.T)[:, :2]
    manual_ids = {str(value) for value in manual_section_ids}
    repaired = transformed
    for sid in manual_ids:
        rows = sections.astype(str) == sid
        repaired[rows] = current[rows]
    return repaired.astype(np.float32, copy=False)


def require_new_output(path: str | Path) -> Path:
    """Return a new output path, refusing to overwrite any existing path."""

    output = Path(path)
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing repair output: {output}"
        )
    return output
