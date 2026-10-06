"""Precompute registered UNI2 grids for the first 00029/g0 interval."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from deepspatial.histology import FeatureStore, UNI2Encoder
from deepspatial.histology.registered_preprocessing import (
    precompute_registered_sdpc_section,
)
from deepspatial.histology.sdpc import SdpcPyramid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/00029_g0")
    parser.add_argument("--through-section", type=int, default=11)
    parser.add_argument("--output", default=None)
    parser.add_argument("--checkpoint", default="weights/uni2/pytorch_model.bin")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--registration-root",
        required=True,
        help="registration root whose saved section/edge transforms define the registered frame",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grid-spacing-um", type=float, default=56.0)
    parser.add_argument("--patch-size-um", type=float, default=112.0)
    parser.add_argument("--target-mpp", type=float, default=0.5)
    args = parser.parse_args()

    from deepspatial_v2.data_export.cell_level_alignment import (
        SavedTransformStore,
        load_section_transform,
    )

    dataset = Path(args.dataset).resolve()
    output = Path(args.output) if args.output else dataset / "uni2_features_001_011.h5"
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(str(output) + ".log")],
    )
    rows = pd.read_parquet(dataset / "he_sections.parquet")
    rows = rows[rows.section_id <= args.through_section].sort_values("section_id")
    store = FeatureStore(output, mode="a")
    encoder = UNI2Encoder(Path(args.checkpoint).resolve(), device=args.device)
    transforms = SavedTransformStore(Path(args.registration_root).resolve(), device="cpu")
    results = []
    for row in rows.itertuples(index=False):
        sid = str(int(row.section_id))
        if sid in store.metadata:
            logging.info("Reusing completed section %s", sid)
            continue
        mask = np.asarray(Image.open(row.source_mask_path).convert("L")) > 0
        section = load_section_transform(row.section_transform_path)
        with SdpcPyramid(row.raw_source_path) as reader:
            if not np.isclose(reader.mpp_level0, row.raw_mpp_um, atol=1e-5):
                raise ValueError(
                    f"section {sid}: SDPC MPP {reader.mpp_level0} != manifest {row.raw_mpp_um}"
                )
            if str(row.source_kind) == "xenium_companion_he":
                # The saved transform is defined in the local companion H&E
                # crop frame.  Add the raw SDPC crop origin only when reading
                # pixels, not when applying the registration transform.
                source_image_offset_um = (
                    float(row.raw_level0_x0) * float(row.raw_mpp_um),
                    float(row.raw_level0_y0) * float(row.raw_mpp_um),
                )
            else:
                source_image_offset_um = (0.0, 0.0)
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
                source_image_offset_um=source_image_offset_um,
                provenance={
                    "source_sdpc": row.raw_source_path,
                    "source_mask": row.source_mask_path,
                    "section_transform": row.section_transform_path,
                    "registration_root": str(Path(args.registration_root).resolve()),
                    "encoder": "UNI2-h_frozen",
                    "checkpoint": str(Path(args.checkpoint).resolve()),
                    "source_image_offset_um": list(source_image_offset_um),
                },
            )
            results.append(result)
        transforms.clear()
    print(json.dumps({"output": str(output), "created": results}, indent=2))


if __name__ == "__main__":
    main()
