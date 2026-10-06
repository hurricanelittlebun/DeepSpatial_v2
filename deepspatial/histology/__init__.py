"""Offline frozen UNI2 morphology and anatomical correspondence paths."""

from .config import HistologyConfig
from .feature_store import FeatureStore
from .path import MorphologyPathCache
from .registered_coordinates import (
    crop_px_to_source_um,
    registered_to_source_points,
    source_to_registered_points,
    source_um_to_crop_px,
)
from .sdpc import SdpcPyramid, extract_sdpc_patch
from .registered_preprocessing import precompute_registered_sdpc_section
from .uni2 import UNI2Encoder
from .nucleus import (
    descriptor_grid_from_nuclei,
    nuclei_table_from_mask,
    segment_h_and_e,
)
from .frame_resample import (
    load_section_affines,
    resample_feature_store,
    transform_points,
)

__all__ = [
    "HistologyConfig",
    "FeatureStore",
    "MorphologyPathCache",
    "crop_px_to_source_um",
    "registered_to_source_points",
    "source_to_registered_points",
    "source_um_to_crop_px",
    "SdpcPyramid",
    "extract_sdpc_patch",
    "precompute_registered_sdpc_section",
    "UNI2Encoder",
    "descriptor_grid_from_nuclei",
    "nuclei_table_from_mask",
    "segment_h_and_e",
    "load_section_affines",
    "resample_feature_store",
    "transform_points",
]
