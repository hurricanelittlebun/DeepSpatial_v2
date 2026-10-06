"""Offline physical-FOV extraction from ALREADY registered rasters/WSI.

No registration is performed. A reader must implement OpenSlide's read_region
and dimensions contract. For unresampled SDPC use an existing registered reader
that applies saved transforms, not the raw unregistered slide here.
"""

import logging
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


def extract_physical_patch(reader, xy_um, *, mpp, image_origin_um, patch_size_um):
    mpp = np.asarray(mpp, float)
    if (
        mpp.shape != (2,)
        or (mpp <= 0).any()
        or not np.isfinite(mpp).all()
        or patch_size_um <= 0
    ):
        raise ValueError("Positive mpp [x,y] and physical FOV required")
    center = (np.asarray(xy_um) - image_origin_um) / mpp
    size = np.maximum(1, np.rint(patch_size_um / mpp).astype(int))
    start = np.rint(center - size / 2).astype(int)
    patch = reader.read_region(tuple(start), 0, tuple(size)).convert("RGBA")
    # OpenSlide padding is transparent; composite white, never black tissue.
    background = Image.new("RGBA", patch.size, "white")
    background.alpha_composite(patch)
    # Anisotropic MPP: fixed physical square must become square encoder input.
    return background.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)


def precompute_section(
    reader,
    encoder,
    store,
    section_id,
    *,
    z_um,
    mpp,
    image_origin_um,
    grid_origin_um,
    grid_shape,
    grid_spacing_um,
    patch_size_um,
    coordinate_frame,
    valid_mask=None,
    batch_size=32,
    provenance=None,
):
    """One-section memory footprint [H,W,1536]; WSI pixels read only by patch.

    valid_mask should be an already-reviewed tissue/QC mask sampled on this grid.
    If omitted, all full-FOV grid locations are used (background is not detected).
    """
    h, w = map(int, grid_shape)
    if min(h, w) <= 0 or batch_size <= 0:
        raise ValueError("Positive grid dimensions/batch_size required")
    valid = (
        np.ones((h, w), bool)
        if valid_mask is None
        else np.asarray(valid_mask, bool).copy()
    )
    if valid.shape != (h, w):
        raise ValueError("Invalid grid mask shape")
    yy, xx = np.mgrid[:h, :w]
    xy = np.stack([xx, yy], -1) * np.asarray(grid_spacing_um) + np.asarray(
        grid_origin_um
    )
    extent = np.asarray(reader.dimensions) * np.asarray(mpp)
    local = xy - image_origin_um
    valid &= ((local >= patch_size_um / 2) & (local <= extent - patch_size_um / 2)).all(
        -1
    )
    ids = np.flatnonzero(valid.ravel())
    features = np.zeros((h * w, encoder.feature_dim), np.float32)
    for start in range(0, len(ids), batch_size):
        chunk = ids[start : start + batch_size]
        patches = [
            extract_physical_patch(
                reader,
                p,
                mpp=mpp,
                image_origin_um=image_origin_um,
                patch_size_um=patch_size_um,
            )
            for p in xy.reshape(-1, 2)[chunk]
        ]
        features[chunk] = encoder.encode(patches, batch_size=batch_size).numpy()
        log.info(
            "%s: encoded %d/%d patches",
            section_id,
            min(start + batch_size, len(ids)),
            len(ids),
        )
    store.add_section(
        section_id,
        features.reshape(h, w, -1),
        z_um=z_um,
        origin_um=grid_origin_um,
        spacing_um=grid_spacing_um,
        patch_size_um=patch_size_um,
        mpp=mpp,
        coordinate_frame=coordinate_frame,
        valid_mask=valid,
        provenance=provenance,
    )
