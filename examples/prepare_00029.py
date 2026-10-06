"""Prepare the first real DeepSpatial 2.0 series: 00029/g0."""

import argparse
from pathlib import Path

import pandas as pd

from deepspatial.data_utils.series_preparation import prepare_series


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        required=True,
        help="external DeepSpatial_v2 project root",
    )
    args = parser.parse_args()
    project = args.project_root.resolve()
    result = prepare_series(
        project / "data/cell_level_aligned/DeepSpatial_cell_level_aligned.h5ad",
        pd.read_parquet(project / "outputs/sequence/manifests/he_combined_object_manifest.parquet"),
        pd.read_parquet(project / "outputs/registration/manifests/stalign_section_transforms.parquet"),
        Path(__file__).resolve().parents[1] / "data/00029_g0",
        series_id="00029__g0",
        section_thickness_um=5.0,
        he_geometry_root=project / "outputs/segmentation",
        registration_root=project / "outputs/registration",
        raw_he_root=project / "data",
        xenium_crop_manifest=pd.read_parquet(
            project / "outputs/xenium_he/0106548/manifests/section_crops.parquet"
        ),
    )
    print(result)


if __name__ == "__main__":
    main()
