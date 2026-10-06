"""Minimal physical-coordinate access to vendor SDPC pyramids."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def choose_level_for_mpp(level_downsamples, mpp_level0, target_mpp):
    """Return the pyramid level whose physical pixel size is closest to target."""

    downsamples = np.asarray(level_downsamples, dtype=float).reshape(-1)
    if (
        len(downsamples) == 0
        or not np.isfinite(downsamples).all()
        or (downsamples <= 0).any()
        or not np.isfinite(mpp_level0)
        or mpp_level0 <= 0
        or not np.isfinite(target_mpp)
        or target_mpp <= 0
    ):
        raise ValueError("pyramid scales and MPP values must be positive and finite")
    level_mpp = float(mpp_level0) * downsamples
    return int(np.argmin(np.abs(np.log(level_mpp / float(target_mpp)))))


def level0_bbox_to_read_region(
    bbox_level0,
    *,
    downsample,
    level0_dimensions,
):
    """Convert an exclusive level-0 bbox to a level-aware read request.

    ``SdpcPyramid.read_region`` receives a level-0 location but a size in
    pixels of the requested pyramid level.  The native reader truncates the
    location after dividing by the level downsample, so this helper makes that
    effective origin explicit and returns the covered level-0 bbox as well.
    """

    bbox = np.asarray(bbox_level0, dtype=float).reshape(-1)
    dimensions = np.asarray(level0_dimensions, dtype=float).reshape(-1)
    if bbox.shape != (4,) or dimensions.shape != (2,):
        raise ValueError("bbox_level0 and level0_dimensions have invalid shapes")
    if (
        not np.isfinite(bbox).all()
        or not np.isfinite(dimensions).all()
        or (dimensions <= 0).any()
        or not np.isfinite(downsample)
        or downsample <= 0
    ):
        raise ValueError("bbox, dimensions and downsample must be finite and positive")
    x0, y0, x1, y1 = bbox
    width0, height0 = dimensions
    if not (0 <= x0 < x1 <= width0 and 0 <= y0 < y1 <= height0):
        raise ValueError("bbox_level0 must be an in-bounds exclusive bbox")

    start_x_level = int(np.floor(x0 / float(downsample)))
    start_y_level = int(np.floor(y0 / float(downsample)))
    end_x_level = int(np.ceil(x1 / float(downsample)))
    end_y_level = int(np.ceil(y1 / float(downsample)))
    size_x = end_x_level - start_x_level
    size_y = end_y_level - start_y_level
    if size_x < 1 or size_y < 1:
        raise ValueError("bbox produces an empty pyramid crop")

    origin_x = start_x_level * float(downsample)
    origin_y = start_y_level * float(downsample)
    covered_x = end_x_level * float(downsample)
    covered_y = end_y_level * float(downsample)
    # The native pyramid normally has dimensions divisible by the downsample.
    # If a vendor pyramid has a shorter final tile, do not silently request it.
    if covered_x > width0 + 1e-6 or covered_y > height0 + 1e-6:
        raise ValueError("bbox crop exceeds the level-0 pyramid boundary")
    return (
        (int(round(origin_x)), int(round(origin_y))),
        (int(size_x), int(size_y)),
        (
            int(round(origin_x)),
            int(round(origin_y)),
            int(round(covered_x)),
            int(round(covered_y)),
        ),
    )


def level_px_to_source_um(
    points,
    *,
    level0_origin,
    downsample,
    mpp_level0,
):
    """Map x/y coordinates in a pyramid crop to source physical micrometres."""

    value = np.asarray(points, dtype=float)
    origin = np.asarray(level0_origin, dtype=float)
    if value.ndim != 2 or value.shape[1] != 2 or not np.isfinite(value).all():
        raise ValueError("points must be a finite N x 2 array")
    if (
        origin.shape != (2,)
        or not np.isfinite(origin).all()
        or not np.isfinite(downsample)
        or downsample <= 0
        or not np.isfinite(mpp_level0)
        or mpp_level0 <= 0
    ):
        raise ValueError("origin, downsample and mpp_level0 must be valid")
    return (origin[None, :] + value * float(downsample)) * float(mpp_level0)


class SdpcPyramid:
    """Context-managed SDPC reader with level-0 location semantics."""

    def __init__(self, path):
        self.path = Path(path)
        self._backend = None

    def __enter__(self):
        try:
            from sdpc import Sdpc
        except Exception as error:  # pragma: no cover - native environment
            raise RuntimeError(
                "Cannot load sdpc-linux; include sdpc/so, so/ffmpeg and so/jpeg "
                "in LD_LIBRARY_PATH before starting Python"
            ) from error
        self._backend = Sdpc(str(self.path))
        return self

    def __exit__(self, *_):
        self.close()

    @property
    def level_dimensions(self):
        return tuple(tuple(map(int, item)) for item in self._backend.level_dimensions)

    @property
    def level_downsamples(self):
        return tuple(map(float, self._backend.level_downsample))

    @property
    def mpp_level0(self):
        return float(self._backend.sdpc.contents.picHead.contents.ruler)

    def read_region(self, location, level, size):
        return np.asarray(self._backend.read_region(location, level, size))

    def close(self):
        backend, self._backend = self._backend, None
        if backend is not None:
            backend.close()


def extract_sdpc_patch(
    reader,
    source_xy_um,
    *,
    patch_size_um,
    target_mpp=0.5,
    output_size=224,
):
    """Extract a fixed physical FOV, selecting the closest SDPC pyramid level."""

    if patch_size_um <= 0 or target_mpp <= 0 or output_size <= 0:
        raise ValueError("patch_size_um, target_mpp and output_size must be positive")
    center_um = np.asarray(source_xy_um, dtype=float)
    if center_um.shape != (2,) or not np.isfinite(center_um).all():
        raise ValueError("source_xy_um must be a finite x/y pair")
    downsamples = np.asarray(reader.level_downsamples, dtype=float)
    level_mpp = float(reader.mpp_level0) * downsamples
    level = int(np.argmin(np.abs(np.log(level_mpp / float(target_mpp)))))
    downsample = downsamples[level]
    size_level = max(1, int(round(float(patch_size_um) / level_mpp[level])))
    center_level = center_um / level_mpp[level]
    desired0 = np.rint(center_level - size_level / 2.0).astype(int)
    desired1 = desired0 + size_level
    width, height = map(int, reader.level_dimensions[level])
    valid0 = np.maximum(desired0, 0)
    valid1 = np.minimum(desired1, (width, height))
    canvas = Image.new("RGB", (size_level, size_level), "white")
    valid_size = valid1 - valid0
    if np.all(valid_size > 0):
        location_level0 = tuple(np.rint(valid0 * downsample).astype(int))
        rgb = reader.read_region(location_level0, level, tuple(valid_size.astype(int)))
        tile = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
        canvas.paste(tile, tuple((valid0 - desired0).astype(int)))
    return canvas.resize((int(output_size), int(output_size)), Image.Resampling.BILINEAR)


def extract_sdpc_patches(
    reader,
    source_xy_um,
    *,
    patch_size_um,
    target_mpp=0.5,
    output_size=224,
):
    """Extract a batch of fixed-FOV patches with one native SDPC read.

    The individual patch API above is intentionally simple, but calling the
    native decoder once per UNI2 patch is very expensive for a dense feature
    grid.  This function reads the smallest level-aware tile covering all
    requested centers and crops/resizes the patches in memory.  It preserves
    the same physical geometry and white boundary padding as
    :func:`extract_sdpc_patch`.
    """

    centers = np.asarray(source_xy_um, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 2 or not np.isfinite(centers).all():
        raise ValueError("source_xy_um must be finite [N,2]")
    if patch_size_um <= 0 or target_mpp <= 0 or output_size <= 0:
        raise ValueError("patch_size_um, target_mpp and output_size must be positive")
    if len(centers) == 0:
        return []

    downsamples = np.asarray(reader.level_downsamples, dtype=float)
    level_mpp = float(reader.mpp_level0) * downsamples
    level = int(np.argmin(np.abs(np.log(level_mpp / float(target_mpp)))))
    downsample = float(downsamples[level])
    size_level = max(1, int(round(float(patch_size_um) / level_mpp[level])))
    center_level = centers / float(level_mpp[level])
    desired0 = np.rint(center_level - size_level / 2.0).astype(int)
    desired1 = desired0 + size_level
    width, height = map(int, reader.level_dimensions[level])

    region0 = np.maximum(desired0.min(axis=0), 0)
    region1 = np.minimum(desired1.max(axis=0), (width, height))
    if np.any(region1 <= region0):
        return [Image.new("RGB", (int(output_size), int(output_size)), "white") for _ in centers]
    location_level0 = tuple(np.rint(region0 * downsample).astype(int))
    region_size = tuple((region1 - region0).astype(int))
    rgb = np.asarray(reader.read_region(location_level0, level, region_size), dtype=np.uint8)
    tile = Image.fromarray(rgb, mode="RGB")

    patches = []
    for start, end in zip(desired0, desired1):
        valid0 = np.maximum(start, region0)
        valid1 = np.minimum(end, region1)
        canvas = Image.new("RGB", (size_level, size_level), "white")
        if np.all(valid1 > valid0):
            crop = tile.crop(
                (
                    int(valid0[0] - region0[0]),
                    int(valid0[1] - region0[1]),
                    int(valid1[0] - region0[0]),
                    int(valid1[1] - region0[1]),
                )
            )
            canvas.paste(crop, (int(valid0[0] - start[0]), int(valid0[1] - start[1])))
        patches.append(canvas.resize((int(output_size), int(output_size)), Image.Resampling.BILINEAR))
    return patches


__all__ = [
    "SdpcPyramid",
    "choose_level_for_mpp",
    "extract_sdpc_patch",
    "extract_sdpc_patches",
    "level0_bbox_to_read_region",
    "level_px_to_source_um",
]
