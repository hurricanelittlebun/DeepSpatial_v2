"""Prepare all non-00029 registered tissue/g series for full reconstruction.

This entrypoint uses the reviewed ST-to-fixed-H&E candidate coordinates and the
validated H&E object/transform manifests.  It only materializes per-series
anchors and H&E metadata; it does not read WSI pixels or run a model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from deepspatial.data_utils.series_preparation import prepare_series


ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = Path("/data/buyonggan/DeepSpatial_v2")
SERIES = (
    "9589__g2",
    "9589__g3",
    "9589__g4",
    "9625__g0",
    "9934__g0",
    "9957__g0",
    "9985__g1",
    "9985__g2",
)


def _candidate_path(series_id: str) -> Path:
    tissue, group = series_id.split("__", 1)
    return ROOT / "data" / f"{tissue}_g{group[1:]}" / "registration_correction_he_v1" / f"{tissue}_g{group[1:]}__st_to_fixed_he_candidate.h5ad"


def prepare_one(
    series_id: str,
    *,
    output_root: Path,
    he_manifest: pd.DataFrame,
    transform_manifest: pd.DataFrame,
    xenium_crop_manifest: pd.DataFrame,
    section_thickness_um: float,
) -> dict:
    tissue, group = series_id.split("__", 1)
    candidate = _candidate_path(series_id)
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    output = output_root / f"{tissue}_{group}" / "reconstruction_prep_v1"
    result = prepare_series(
        candidate,
        he_manifest,
        transform_manifest,
        output,
        series_id=series_id,
        section_thickness_um=section_thickness_um,
        he_geometry_root=V2_ROOT / "outputs" / "segmentation",
        registration_root=V2_ROOT / "outputs" / "registration",
        raw_he_root=V2_ROOT / "data",
        xenium_crop_manifest=xenium_crop_manifest,
        spatial_key="spatial_st_corrected",
        validate_paths=True,
    )
    result["source_h5ad"] = str(candidate.resolve())
    result["output_dir"] = str(output.resolve())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / "data")
    parser.add_argument("--section-thickness-um", type=float, default=5.0)
    parser.add_argument("--series", nargs="*", default=list(SERIES))
    args = parser.parse_args()

    he_manifest = pd.read_parquet(
        V2_ROOT / "outputs/sequence/manifests/he_combined_object_manifest.parquet"
    )
    transform_manifest = pd.read_parquet(
        V2_ROOT / "outputs/registration/manifests/stalign_section_transforms.parquet"
    )
    crop_tables = [
        pd.read_parquet(V2_ROOT / f"outputs/xenium_he/{sample}/manifests/section_crops.parquet")
        for sample in ("0106542", "0106548")
    ]
    xenium_crop_manifest = pd.concat(crop_tables, ignore_index=True)

    records = []
    for series_id in args.series:
        print(f"preparing {series_id}", flush=True)
        result = prepare_one(
            series_id,
            output_root=args.output_root,
            he_manifest=he_manifest,
            transform_manifest=transform_manifest,
            xenium_crop_manifest=xenium_crop_manifest,
            section_thickness_um=args.section_thickness_um,
        )
        records.append(result)
        print(json.dumps(result, indent=2), flush=True)

    manifest = {
        "format": "deepspatial-full-reconstruction-preparation-v1",
        "section_thickness_um": float(args.section_thickness_um),
        "series": records,
    }
    path = args.output_root / "full_reconstruction_preparation_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
