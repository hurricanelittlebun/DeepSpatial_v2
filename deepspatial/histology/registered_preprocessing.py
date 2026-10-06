"""Offline UNI2 grids addressed in an existing registered coordinate frame."""

from __future__ import annotations

import logging

import numpy as np
from scipy.ndimage import binary_erosion

from .registered_coordinates import (
    crop_px_to_source_um,
    registered_to_source_points,
    source_to_registered_points,
    source_um_to_crop_px,
)
from .sdpc import extract_sdpc_patches

log = logging.getLogger(__name__)


def _mask_boundary_points(mask, maximum=20000):
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        raise ValueError("source_mask must be a nonempty 2-D mask")
    boundary = mask & ~binary_erosion(mask)
    rows, columns = np.nonzero(boundary)
    if len(rows) > maximum:
        chosen = np.linspace(0, len(rows) - 1, maximum, dtype=int)
        rows, columns = rows[chosen], columns[chosen]
    return np.column_stack([columns, rows]).astype(float)


def precompute_registered_sdpc_section(
    reader,
    encoder,
    feature_store,
    section_id,
    *,
    section,
    transform_store,
    source_mask,
    source_mask_mpp,
    source_mask_bbox_origin_px,
    z_um,
    grid_spacing_um,
    patch_size_um,
    target_mpp,
    coordinate_frame,
    batch_size=16,
    source_image_offset_um=(0.0, 0.0),
    provenance=None,
):
    """Encode a regular registered grid by inverse-mapping centers to SDPC.

    The center follows the complete saved nonlinear transform. The patch pixels
    retain their source-slide orientation; no registration is re-estimated and
    no local image warp is performed in this first implementation.
    """

    spacing = float(grid_spacing_um)
    if spacing <= 0 or batch_size <= 0:
        raise ValueError("grid_spacing_um and batch_size must be positive")
    image_offset = np.asarray(source_image_offset_um, dtype=float)
    if image_offset.shape != (2,) or not np.isfinite(image_offset).all():
        raise ValueError("source_image_offset_um must be a finite x/y pair")
    mask = np.asarray(source_mask, dtype=bool)
    mask_mpp = np.asarray(source_mask_mpp, dtype=float)
    bbox = np.asarray(source_mask_bbox_origin_px, dtype=float)
    boundary_px = _mask_boundary_points(mask)
    boundary_um = crop_px_to_source_um(boundary_px, mpp=mask_mpp, bbox_origin_px=bbox)
    boundary_registered = source_to_registered_points(
        boundary_um, section, transform_store
    )
    low = np.floor(boundary_registered.min(axis=0) / spacing) * spacing
    high = np.ceil(boundary_registered.max(axis=0) / spacing) * spacing
    width, height = (np.rint((high - low) / spacing).astype(int) + 1)
    yy, xx = np.mgrid[:height, :width]
    registered_xy = np.stack([xx, yy], axis=-1) * spacing + low
    flat_registered = registered_xy.reshape(-1, 2)
    source_um = registered_to_source_points(
        flat_registered, section, transform_store
    )
    mask_px = source_um_to_crop_px(source_um, mpp=mask_mpp, bbox_origin_px=bbox)
    ix = np.rint(mask_px[:, 0]).astype(int)
    iy = np.rint(mask_px[:, 1]).astype(int)
    valid = (
        (ix >= 0) & (iy >= 0) & (ix < mask.shape[1]) & (iy < mask.shape[0])
    )
    valid[valid] &= mask[iy[valid], ix[valid]]
    ids = np.flatnonzero(valid)
    features = np.zeros((len(flat_registered), encoder.feature_dim), dtype=np.float32)
    for start in range(0, len(ids), int(batch_size)):
        chunk = ids[start : start + int(batch_size)]
        # Read one native SDPC tile per encoder batch, then crop individual
        # physical-FOV patches in memory.  This avoids hundreds of costly
        # decoder calls per section without changing the registered-to-source
        # mapping or the UNI2 patch geometry.
        patches = extract_sdpc_patches(
            reader,
            source_um[chunk] + image_offset,
            patch_size_um=patch_size_um,
            target_mpp=target_mpp,
        )
        features[chunk] = encoder.encode(patches, batch_size=batch_size).numpy()
        log.info(
            "section %s: encoded %d/%d valid patches",
            section_id,
            min(start + batch_size, len(ids)),
            len(ids),
        )
    details = dict(provenance or {})
    details.update(
        center_mapping="registered_to_source_saved_stalign_chain",
        patch_geometry="source_orientation_no_local_warp",
        target_mpp=float(target_mpp),
        source_image_offset_um=image_offset.tolist(),
        valid_grid_points=int(valid.sum()),
        total_grid_points=int(len(valid)),
    )
    feature_store.add_section(
        str(section_id),
        features.reshape(height, width, -1),
        z_um=float(z_um),
        origin_um=low,
        spacing_um=(spacing, spacing),
        patch_size_um=float(patch_size_um),
        mpp=(float(target_mpp), float(target_mpp)),
        coordinate_frame=str(coordinate_frame),
        valid_mask=valid.reshape(height, width),
        provenance=details,
    )
    return {
        "section_id": str(section_id),
        "grid_shape": [int(height), int(width)],
        "grid_origin_um": low.tolist(),
        "valid_grid_points": int(valid.sum()),
    }


__all__ = ["precompute_registered_sdpc_section"]
