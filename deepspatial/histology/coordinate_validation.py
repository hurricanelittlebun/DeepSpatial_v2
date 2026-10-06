"""Small geometry helpers for validating multi-resolution H&E coordinates."""

from __future__ import annotations

import math

import numpy as np


def load_candidate_coordinates(path, *, expected_n=None):
    """Load corrected ST coordinates from a residual-correction NPZ.

    The correction artifacts keep the original registered coordinates and
    store the candidate branch under ``spatial_st_corrected``.  This helper
    validates that branch before it is passed to a downstream coordinate
    conversion or plotting routine.
    """

    with np.load(path, allow_pickle=False) as source:
        if "spatial_st_corrected" not in source:
            raise KeyError(f"{path}: missing spatial_st_corrected")
        corrected = np.asarray(source["spatial_st_corrected"], dtype=np.float64)
    if corrected.ndim != 2 or corrected.shape[1] != 2:
        raise ValueError(
            f"{path}: spatial_st_corrected must have shape [N, 2], got {corrected.shape}"
        )
    if expected_n is not None and corrected.shape[0] != int(expected_n):
        raise ValueError(
            f"{path}: expected {int(expected_n)} coordinates, got {corrected.shape[0]}"
        )
    if not np.isfinite(corrected).all():
        raise ValueError(f"{path}: spatial_st_corrected contains non-finite values")
    return corrected


def _positive_pair(value, name):
    pair = np.asarray(value, dtype=float).reshape(-1)
    if pair.shape != (2,) or not np.isfinite(pair).all() or (pair <= 0).any():
        raise ValueError(f"{name} must contain two positive finite values")
    return pair


def _nonnegative_pair(value, name):
    pair = np.asarray(value, dtype=float).reshape(-1)
    if pair.shape != (2,) or not np.isfinite(pair).all() or (pair < 0).any():
        raise ValueError(f"{name} must contain two non-negative finite values")
    return pair


def effective_level0_origin(origin_level0, downsample):
    """Return the effective level-0 origin used by the native SDPC reader."""

    origin = np.asarray(origin_level0, dtype=float).reshape(-1)
    if origin.shape != (2,) or not np.isfinite(origin).all() or (origin < 0).any():
        raise ValueError("origin_level0 must contain two non-negative finite values")
    scale = float(downsample)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("downsample must be positive and finite")
    return tuple(int(math.floor(float(item) / scale) * scale) for item in origin)


def output_to_source_pixel_centers(
    output_shape,
    *,
    output_origin_level0,
    output_downsample,
    source_origin_level0,
    source_downsample,
):
    """Map output pixel centers to source-array pixel coordinates.

    The returned array has shape ``[height, width, 2]`` and stores ``x, y``
    source pixel coordinates.  Pixel centers use the convention ``index + 0.5``.
    """

    shape = tuple(int(item) for item in output_shape)
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError("output_shape must contain two positive integers")
    output_origin = _nonnegative_pair(output_origin_level0, "output_origin_level0")
    source_origin = _nonnegative_pair(source_origin_level0, "source_origin_level0")
    output_scale = float(output_downsample)
    source_scale = float(source_downsample)
    if (
        not math.isfinite(output_scale)
        or output_scale <= 0
        or not math.isfinite(source_scale)
        or source_scale <= 0
    ):
        raise ValueError("downsample values must be positive and finite")

    rows, cols = np.indices(shape, dtype=np.float32)
    physical_x = output_origin[0] + (cols + 0.5) * output_scale
    physical_y = output_origin[1] + (rows + 0.5) * output_scale
    source_x = (physical_x - source_origin[0]) / source_scale - 0.5
    source_y = (physical_y - source_origin[1]) / source_scale - 0.5
    return np.stack((source_x, source_y), axis=-1).astype(np.float32)


def local_um_to_output_pixel_centers(
    points_um,
    *,
    local_origin_level0,
    output_origin_level0,
    raw_mpp_um,
    output_downsample,
):
    """Map local physical coordinates to output-array pixel-center coordinates.

    ``points_um`` is measured from ``local_origin_level0`` in the same source
    slide frame.  The returned coordinates use the array convention
    ``index + 0.5`` and are ordered as ``x, y``.  This is useful when a crop
    was read from a pyramid level whose native reader floors the level-0
    origin, so the requested crop origin and the actual array origin differ
    by a few pixels.
    """

    value = np.asarray(points_um, dtype=float)
    if value.ndim != 2 or value.shape[1] != 2 or not np.isfinite(value).all():
        raise ValueError("points_um must be a finite N x 2 array")
    local_origin = _nonnegative_pair(local_origin_level0, "local_origin_level0")
    output_origin = _nonnegative_pair(output_origin_level0, "output_origin_level0")
    raw_mpp = float(raw_mpp_um)
    downsample = float(output_downsample)
    if not math.isfinite(raw_mpp) or raw_mpp <= 0:
        raise ValueError("raw_mpp_um must be positive and finite")
    if not math.isfinite(downsample) or downsample <= 0:
        raise ValueError("output_downsample must be positive and finite")
    source_level0 = local_origin[None, :] + value / raw_mpp
    return ((source_level0 - output_origin[None, :]) / downsample - 0.5).astype(
        np.float32
    )


__all__ = [
    "effective_level0_origin",
    "load_candidate_coordinates",
    "local_um_to_output_pixel_centers",
    "output_to_source_pixel_centers",
]
