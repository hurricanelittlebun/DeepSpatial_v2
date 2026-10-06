#!/usr/bin/env python3
"""Rebuild 00029 UNI2 grids with corrected companion-SDPC image offsets.

Serial H&E sections are copied from the existing feature store.  Only the ten
Xenium companion sections are re-encoded: their saved registration transforms
use local crop coordinates, so the raw SDPC crop origin is added when pixels
are read.  No H&E registration is re-estimated here.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import uuid
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
# Make direct ``python scripts/...py`` execution import the local package.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_DATASET = ROOT / "data/00029_g0"
DEFAULT_SOURCE = DEFAULT_DATASET / "uni2_features.h5"
DEFAULT_OUTPUT = DEFAULT_DATASET / "uni2_features_raw_corrected.h5"
DEFAULT_CHECKPOINT = ROOT / "weights/uni2/pytorch_model.bin"
DEFAULT_REGISTRATION = Path("/data/buyonggan/DeepSpatial_v2/outputs/registration")
DEFAULT_V2_SRC = Path("/data/buyonggan/DeepSpatial_v2/src")


def _copy_without_anchor_sections(source_path, output_path, omit):
    """Copy HDF5 groups without materializing 1536-dimensional grids in RAM."""

    omit = {str(value) for value in omit}
    from deepspatial.histology import FeatureStore

    source_store = FeatureStore(source_path)
    with h5py.File(source_path, "r") as source, h5py.File(output_path, "w") as output:
        for key, value in source.attrs.items():
            output.attrs[key] = value
        output.attrs["revision"] = str(uuid.uuid4())
        output.attrs["format"] = "deepspatial-uni2-grid-v1"
        target_sections = output.create_group("sections")
        for sid in source_store.sections:
            if sid in omit:
                continue
            source.copy(
                source["sections"][source_store._key(sid)],
                target_sections,
                name=source_store._key(sid),
            )


def _source_image_offset(row):
    if str(row.source_kind) != "xenium_companion_he":
        return (0.0, 0.0)
    return (
        float(row.raw_level0_x0) * float(row.raw_mpp_um),
        float(row.raw_level0_y0) * float(row.raw_mpp_um),
    )


def rebuild(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    source = Path(args.source).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    dataset = Path(args.dataset).resolve()
    rows = pd.read_parquet(dataset / "he_sections.parquet").sort_values("section_id")
    anchors = rows[rows.source_kind.astype(str).eq("xenium_companion_he")].copy()
    if anchors.empty:
        raise ValueError("No Xenium companion rows found in he_sections.parquet")

    _copy_without_anchor_sections(source, output, anchors.section_id.astype(str))

    from deepspatial.histology import FeatureStore, UNI2Encoder
    from deepspatial.histology.registered_preprocessing import (
        precompute_registered_sdpc_section,
    )
    from deepspatial.histology.sdpc import SdpcPyramid
    from deepspatial_v2.data_export.cell_level_alignment import (
        SavedTransformStore,
        load_section_transform,
    )

    store = FeatureStore(output, mode="a")
    encoder = UNI2Encoder(Path(args.checkpoint).resolve(), device=args.device)
    transforms = SavedTransformStore(Path(args.registration_root), device="cpu")
    source_path = str(anchors.iloc[0].raw_source_path)
    if anchors.raw_source_path.astype(str).nunique() != 1:
        raise ValueError("Expected all 00029 companion sections in one raw SDPC")
    results = []
    for row in anchors.itertuples(index=False):
        sid = str(int(row.section_id))
        mask = np.asarray(Image.open(row.source_mask_path).convert("L")) > 0
        section = load_section_transform(row.section_transform_path)
        offset = _source_image_offset(row)
        with SdpcPyramid(row.raw_source_path) as reader:
            if not np.isclose(reader.mpp_level0, row.raw_mpp_um, atol=1e-5):
                raise ValueError(
                    f"section {sid}: SDPC MPP {reader.mpp_level0} != manifest {row.raw_mpp_um}"
                )
            result = precompute_registered_sdpc_section(
                reader,
                encoder,
                store,
                sid,
                section=section,
                transform_store=transforms,
                source_mask=mask,
                source_mask_mpp=(
                    row.analysis_pixel_size_x_um,
                    row.analysis_pixel_size_y_um,
                ),
                source_mask_bbox_origin_px=(row.bbox_x0, row.bbox_y0),
                z_um=row.z_um,
                grid_spacing_um=args.grid_spacing_um,
                patch_size_um=args.patch_size_um,
                target_mpp=args.target_mpp,
                coordinate_frame=row.coordinate_frame,
                batch_size=args.batch_size,
                source_image_offset_um=offset,
                provenance={
                    "source_sdpc": row.raw_source_path,
                    "source_mask": row.source_mask_path,
                    "section_transform": row.section_transform_path,
                    "encoder": "UNI2-h_frozen",
                    "checkpoint": str(Path(args.checkpoint).resolve()),
                    "source_image_offset_um": list(offset),
                    "source_coordinate_mode": "companion_crop_local_transform_plus_raw_image_offset",
                    "rebuild_reason": "correct_companion_sdpc_global_image_coordinate",
                },
            )
            results.append(result)
        transforms.clear()

    final_store = FeatureStore(output)
    if set(final_store.sections) != set(rows.section_id.astype(str)):
        raise RuntimeError("Corrected feature store does not contain all H&E sections")
    manifest = {
        "format": "deepspatial-uni2-grid-v1-companion-corrected",
        "source_store": str(source),
        "output_store": str(output),
        "dataset": str(dataset),
        "coordinate_frame": str(final_store.metadata[final_store.sections[0]]["coordinate_frame"]),
        "n_sections": len(final_store.sections),
        "reencoded_sections": [int(value) for value in anchors.section_id],
        "copied_sections": [
            int(value) for value in rows.loc[~rows.section_id.isin(anchors.section_id), "section_id"]
        ],
        "target_mpp": float(args.target_mpp),
        "grid_spacing_um": float(args.grid_spacing_um),
        "patch_size_um": float(args.patch_size_um),
        "results": results,
        "note": "Registration transforms are unchanged; only companion raw SDPC patch lookup is corrected.",
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--registration-root", type=Path, default=DEFAULT_REGISTRATION)
    parser.add_argument("--v2-src", type=Path, default=DEFAULT_V2_SRC)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grid-spacing-um", type=float, default=56.0)
    parser.add_argument("--patch-size-um", type=float, default=112.0)
    parser.add_argument("--target-mpp", type=float, default=0.5)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.v2_src).resolve()))
    rebuild(args)


if __name__ == "__main__":
    main()
