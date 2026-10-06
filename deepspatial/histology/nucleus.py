"""Nuclear-structure descriptors for optional anatomical path guidance.

This module deliberately describes local nuclear organization. It does not
assign identity to nuclei across sections and must not be interpreted as cell
lineage tracking.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from skimage.color import rgb2hed
from skimage.measure import regionprops


def _finite_xy(xy_um: np.ndarray) -> np.ndarray:
    value = np.asarray(xy_um, dtype=float)
    if value.ndim != 2 or value.shape[1] != 2 or not np.isfinite(value).all():
        raise ValueError("xy_um must be a finite [N,2] array")
    return value


def segment_h_and_e(rgb, mask, model, *, diameter_px: float):
    """Run an injected Cellpose model on a masked RGB H&E crop.

    Cellpose is imported by the caller's environment rather than at package
    import time. Pixels outside the reviewed tissue mask are made white so
    black raster padding cannot be interpreted as hematoxylin signal.
    """

    image = np.asarray(rgb)
    tissue = np.asarray(mask, dtype=bool)
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype.kind not in "ui":
        raise ValueError("rgb must be an integer RGB image")
    if tissue.shape != image.shape[:2]:
        raise ValueError("mask must match the RGB image height and width")
    if not np.isfinite(diameter_px) or diameter_px <= 0:
        raise ValueError("diameter_px must be positive")

    clean = image.astype(np.uint8, copy=True)
    clean[~tissue] = 255
    hematoxylin = np.maximum(rgb2hed(clean)[..., 0], 0).astype("float32")
    low, high = np.percentile(hematoxylin, [1, 99])
    stain = np.clip((hematoxylin - low) / max(float(high - low), 1e-8), 0, 1)
    masks, flows, _ = model.eval(
        stain,
        diameter=float(diameter_px),
        channels=[0, 0],
        normalize=False,
        cellprob_threshold=0,
        flow_threshold=0.4,
    )
    score = flows[2] if isinstance(flows, (list, tuple)) and len(flows) > 2 else None
    return np.asarray(masks, dtype=np.uint32), score


def nuclei_table_from_mask(
    instance_mask,
    score=None,
    *,
    mpp_um=(1.0, 1.0),
    section_id="section",
):
    """Return per-nucleus geometry and raw Cellpose score summaries."""

    labels = np.asarray(instance_mask)
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("instance_mask must be an integer [H,W] array")
    mpp = np.asarray(mpp_um, dtype=float)
    if mpp.shape != (2,) or not np.isfinite(mpp).all() or (mpp <= 0).any():
        raise ValueError("mpp_um must be two positive finite values")
    score_array = None if score is None else np.asarray(score, dtype=float)
    if score_array is not None and score_array.shape != labels.shape:
        raise ValueError("score must match instance_mask")

    height, width = labels.shape
    rows = []
    for region in regionprops(labels):
        y_px, x_px = region.centroid
        coords = region.coords
        raw_score = (
            float(np.nanmean(score_array[coords[:, 0], coords[:, 1]]))
            if score_array is not None
            else float("nan")
        )
        rows.append(
            {
                "cell_id": f"{section_id}_n{int(region.label)}",
                "nucleus_id": int(region.label),
                "x_px": float(x_px),
                "y_px": float(y_px),
                "area_um2": float(region.area * mpp[0] * mpp[1]),
                "perimeter_um": float(region.perimeter * np.mean(mpp)),
                "eccentricity": float(region.eccentricity),
                "cellpose_score_raw": raw_score,
                "touches_image_edge": bool(
                    region.bbox[0] == 0
                    or region.bbox[1] == 0
                    or region.bbox[2] == height
                    or region.bbox[3] == width
                ),
            }
        )
    import pandas as pd

    return pd.DataFrame(
        rows,
        columns=[
            "cell_id",
            "nucleus_id",
            "x_px",
            "y_px",
            "area_um2",
            "perimeter_um",
            "eccentricity",
            "cellpose_score_raw",
            "touches_image_edge",
        ],
    )


def descriptor_grid_from_nuclei(
    xy_um,
    areas_um2,
    eccentricities,
    grid_xy_um,
    *,
    radius_um: float,
    tissue_mask=None,
):
    """Build local nuclear descriptors on a physical XY grid.

    Channels are log count, density, mean log area, log-area spread, mean
    eccentricity, mean normalized nucleus distance, and tissue support. The
    last channel keeps valid tissue with genuinely low nuclear density
    distinguishable from unavailable/background grid points.
    """

    nuclei = _finite_xy(xy_um)
    grid = np.asarray(grid_xy_um, dtype=float)
    if grid.ndim != 3 or grid.shape[-1] != 2 or not np.isfinite(grid).all():
        raise ValueError("grid_xy_um must be a finite [H,W,2] array")
    areas = np.asarray(areas_um2, dtype=float).reshape(-1)
    eccentricity = np.asarray(eccentricities, dtype=float).reshape(-1)
    if len(nuclei) != len(areas) or len(nuclei) != len(eccentricity):
        raise ValueError("nuclear geometry arrays must match xy_um")
    if len(areas) and (not np.isfinite(areas).all() or (areas <= 0).any()):
        raise ValueError("areas_um2 must be positive and finite")
    if len(eccentricity) and (
        not np.isfinite(eccentricity).all()
        or (eccentricity < 0).any()
        or (eccentricity > 1).any()
    ):
        raise ValueError("eccentricities must lie in [0,1]")
    if not np.isfinite(radius_um) or radius_um <= 0:
        raise ValueError("radius_um must be positive")

    tissue = None if tissue_mask is None else np.asarray(tissue_mask, dtype=bool)
    if tissue is not None and tissue.shape != grid.shape[:2]:
        raise ValueError("tissue_mask must match grid_xy_um height and width")

    flat_grid = grid.reshape(-1, 2)
    descriptors = np.zeros((len(flat_grid), 7), dtype=np.float32)
    supported = np.zeros(len(flat_grid), dtype=bool)
    if len(nuclei):
        tree = cKDTree(nuclei)
        neighbours = tree.query_ball_point(flat_grid, r=float(radius_um))
        log_area = np.log1p(areas)
        window_area = np.pi * float(radius_um) ** 2
        for index, ids in enumerate(neighbours):
            if not ids:
                continue
            ids = np.asarray(ids, dtype=np.int64)
            delta = nuclei[ids] - flat_grid[index]
            distances = np.linalg.norm(delta, axis=1) / float(radius_um)
            count = len(ids)
            descriptors[index, 0] = np.log1p(count)
            descriptors[index, 1] = count / window_area
            descriptors[index, 2] = np.mean(log_area[ids])
            descriptors[index, 3] = np.std(log_area[ids])
            descriptors[index, 4] = np.mean(eccentricity[ids])
            descriptors[index, 5] = np.mean(distances)
            supported[index] = True
    if tissue is not None:
        descriptors[:, 6] = tissue.reshape(-1).astype(np.float32)
        # A tissue pixel without a nearby segmented nucleus is not evidence
        # for a nuclear descriptor.  Keep tissue support as an explicit
        # channel, but make validity require both conditions so downstream
        # path code records an auditable nuclear-support fallback.
        valid = tissue.reshape(-1).copy() & supported
    else:
        valid = supported
        descriptors[:, 6] = supported.astype(np.float32)
    return descriptors.reshape(*grid.shape[:2], -1), valid.reshape(grid.shape[:2])


__all__ = [
    "descriptor_grid_from_nuclei",
    "nuclei_table_from_mask",
    "segment_h_and_e",
]
