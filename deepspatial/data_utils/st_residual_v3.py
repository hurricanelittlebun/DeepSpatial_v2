"""Small, dependency-light helpers for materializing the 9957 v3 ST release."""

from __future__ import annotations

from typing import Mapping

import numpy as np


def assemble_section_coordinates(
    section_ids: np.ndarray,
    candidate_by_section: Mapping[int, Mapping[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Restore per-section candidate coordinates to the original AnnData order.

    Candidate files are written section-by-section, whereas an AnnData object
    may interleave rows from different sections.  This function is deliberately
    strict: a missing section, wrong row count, wrong shape, or non-finite value
    aborts materialization instead of silently producing a coordinate mismatch.
    """

    ids = np.asarray(section_ids)
    if ids.ndim != 1:
        raise ValueError(f"section_ids must be one-dimensional, got {ids.shape}")

    base = np.empty((len(ids), 2), dtype=np.float64)
    candidate = np.empty((len(ids), 2), dtype=np.float64)
    for raw_sid in np.unique(ids):
        sid = int(raw_sid)
        rows = np.flatnonzero(ids == raw_sid)
        if sid not in candidate_by_section:
            raise ValueError(f"candidate coordinates are missing section {sid}")
        payload = candidate_by_section[sid]
        if "st_global_base" not in payload or "st_global_candidate" not in payload:
            raise ValueError(f"section {sid} candidate is missing required arrays")
        section_base = np.asarray(payload["st_global_base"], dtype=np.float64)
        section_candidate = np.asarray(payload["st_global_candidate"], dtype=np.float64)
        if section_base.shape != (len(rows), 2):
            raise ValueError(
                f"section {sid} has {len(rows)} rows in AnnData but "
                f"candidate base has shape {section_base.shape}"
            )
        if section_candidate.shape != (len(rows), 2):
            raise ValueError(
                f"section {sid} has {len(rows)} rows in AnnData but "
                f"candidate output has shape {section_candidate.shape}"
            )
        if not np.isfinite(section_base).all() or not np.isfinite(section_candidate).all():
            raise ValueError(f"section {sid} contains non-finite candidate coordinates")
        base[rows] = section_base
        candidate[rows] = section_candidate

    return base, candidate
