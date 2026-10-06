#!/usr/bin/env python3
"""Re-map raw-SDPC nuclear tables into the saved 00029 registration frames.

The raw segmentation masks are intentionally not recomputed here.  Serial
00029 crops use the global source frame encoded by their reviewed manifest;
Xenium companion crops use the local crop frame used when their H&E/ST
registration transform was estimated.  The raw SDPC bbox is an extraction
locator for the latter, not an offset to add to the transform input.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HE_TABLE = ROOT / "data/00029_g0/he_sections.parquet"
DEFAULT_SOURCE = ROOT / "data/00029_g0/nucleus_path_raw_v1"
DEFAULT_OUTPUT = ROOT / "data/00029_g0/nucleus_path_raw_v1_coordinate_corrected"
DEFAULT_REGISTRATION = Path("/data/buyonggan/DeepSpatial_v2/outputs/registration")
DEFAULT_V2_SRC = Path("/data/buyonggan/DeepSpatial_v2/src")


def _load_registration(v2_src, registration_root):
    v2_src = str(Path(v2_src).resolve())
    if v2_src not in sys.path:
        sys.path.insert(0, v2_src)
    from deepspatial_v2.data_export.cell_level_alignment import (
        SavedTransformStore,
        load_section_transform,
        transform_section_points,
    )

    return (
        SavedTransformStore(Path(registration_root), device="cpu"),
        load_section_transform,
        transform_section_points,
    )


def _source_coordinates(table, row, crop_metadata):
    points = table[["x_px", "y_px"]].to_numpy(float)
    downsample = float(crop_metadata["level_downsample"])
    mpp_level0 = float(crop_metadata.get("raw_mpp_um", row.raw_mpp_um))
    if str(row.source_kind) == "xenium_companion_he":
        # The saved companion transform was fitted in the local H&E crop
        # frame, so the raw SDPC bbox must not be added here.
        return points * (downsample * mpp_level0)
    origin = np.asarray(crop_metadata["read_location_level0"], dtype=float)
    return (origin[None, :] + points * downsample) * mpp_level0


def remap(args):
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)
    (output / "segmentation").mkdir()

    he = pd.read_parquet(args.he_table).sort_values("section_id")
    store, load_section_transform, transform_section_points = _load_registration(
        args.v2_src, args.registration_root
    )
    summary_path = source / "segmentation_summary.csv"
    summary = pd.read_csv(summary_path) if summary_path.is_file() else None
    records = []

    for _, row in he.iterrows():
        sid = int(row.section_id)
        source_dir = source / "segmentation" / f"section-{sid:03d}"
        source_table_path = source_dir / "nuclei.csv"
        destination_dir = output / "segmentation" / f"section-{sid:03d}"
        destination_dir.mkdir(parents=True)
        if source_table_path.is_file():
            table = pd.read_csv(source_table_path)
        else:
            table = pd.DataFrame()

        metadata_path = source_dir / "raw_crop_metadata.json"
        if len(table):
            if not metadata_path.is_file():
                raise FileNotFoundError(f"Missing raw crop metadata: {metadata_path}")
            crop_metadata = json.loads(metadata_path.read_text())
            source_xy = _source_coordinates(table, row, crop_metadata)
            section = load_section_transform(row.section_transform_path)
            registered = transform_section_points(source_xy, section, store)
            table["x_source_um"], table["y_source_um"] = source_xy.T
            table["x_registered_um"], table["y_registered_um"] = registered.T

        table.to_csv(destination_dir / "nuclei.csv", index=False)
        table.to_csv(destination_dir / "nuclei_registered.csv", index=False)
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text())
            metadata["coordinate_correction"] = (
                "companion crop local source frame or serial SDPC global source frame"
            )
            (destination_dir / "raw_crop_metadata.json").write_text(
                json.dumps(metadata, indent=2) + "\n"
            )
        records.append(
            {
                "section_id": str(sid),
                "n_nuclei": int(len(table)),
                "source_coordinate_mode": (
                    "companion_crop_local_physical_um"
                    if str(row.source_kind) == "xenium_companion_he"
                    else "serial_sdpc_global_physical_um"
                ),
                "status": "coordinate_corrected",
            }
        )

    if summary is not None:
        corrections = {
            record["section_id"]: record["source_coordinate_mode"]
            for record in records
        }
        summary["coordinate_correction"] = summary["section_id"].astype(str).map(
            corrections
        ).fillna("")
        summary.to_csv(output / "segmentation_summary.csv", index=False)
    manifest = {
        "format": "deepspatial-nucleus-coordinate-correction-v1",
        "he_table": str(Path(args.he_table).resolve()),
        "source_segmentation": str(source),
        "registration_root": str(Path(args.registration_root).resolve()),
        "coordinate_rule": {
            "serial": "raw crop effective level-0 origin plus pixel*downsample, then raw mpp",
            "xenium_companion_he": "local crop pixel*selected level mpp; raw bbox is locator only",
        },
        "sections": records,
    }
    (output / "coordinate_correction_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    (output / "README.md").write_text(
        "# 00029 raw nuclear coordinate correction\n\n"
        "This directory contains corrected nuclear coordinate tables only. "
        "Instance masks and raw QC images remain in `source_segmentation`; "
        "the corrected tables use the same saved registration transforms and "
        "the `00029__g0_registered` frame.\n"
    )
    print(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--he-table", type=Path, default=DEFAULT_HE_TABLE)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--registration-root", type=Path, default=DEFAULT_REGISTRATION)
    parser.add_argument("--v2-src", type=Path, default=DEFAULT_V2_SRC)
    remap(parser.parse_args())


if __name__ == "__main__":
    main()
