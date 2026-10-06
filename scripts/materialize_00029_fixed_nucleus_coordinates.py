#!/usr/bin/env python3
"""Materialize all 00029/g0 nuclear centroids in the fixed H&E frame.

The source Cellpose masks remain in their native high-resolution H&E crop
frames.  This script only transforms the per-nucleus physical coordinates
from the existing ``00029__g0_registered`` frame into the fixed H&E frame
used by ``spatial_st_corrected`` and the fixed morphology feature stores.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from deepspatial.histology.frame_resample import transform_points  # noqa: E402
from materialize_00029_fixed_he_feature_stores import (  # noqa: E402
    build_section_affines,
)


SOURCE_FRAME = "00029__g0_registered"
OUTPUT_FRAME = "00029__g0_fixed_he"
DEFAULT_HE_TABLE = ROOT / "data/00029_g0/he_sections.parquet"
DEFAULT_SOURCE = ROOT / "data/00029_g0/nucleus_path_raw_v1_coordinate_corrected_v2"
DEFAULT_RESIDUAL_DIR = ROOT / "data/00029_g0/registration_correction_v1"
DEFAULT_OUTPUT = ROOT / "data/00029_g0/fixed_he_single_cell_v1"


def add_fixed_coordinates(table: pd.DataFrame, matrix: np.ndarray) -> pd.DataFrame:
    """Add fixed-frame x/y columns while preserving every source column."""

    required = {"x_registered_um", "y_registered_um"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise KeyError(
            "Nucleus table lacks registered coordinates: " + ", ".join(missing)
        )
    result = table.copy()
    points = result[["x_registered_um", "y_registered_um"]].to_numpy(dtype=float)
    fixed = transform_points(points, matrix)
    result["x_fixed_he_um"] = fixed[:, 0]
    result["y_fixed_he_um"] = fixed[:, 1]
    return result


def _json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def materialize(
    *,
    he_table_path: Path,
    source_dir: Path,
    residual_dir: Path,
    output_dir: Path,
) -> dict:
    he_table_path = Path(he_table_path).resolve()
    source_dir = Path(source_dir).resolve()
    residual_dir = Path(residual_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    if not he_table_path.is_file():
        raise FileNotFoundError(he_table_path)
    if not source_dir.is_dir():
        raise NotADirectoryError(source_dir)
    if not residual_dir.is_dir():
        raise NotADirectoryError(residual_dir)

    he_table = pd.read_parquet(he_table_path).copy()
    matrices, transform_records = build_section_affines(
        he_table,
        residual_dir,
        interpolate_serial_affine=True,
    )
    transform_by_id = {
        str(record["section_id"]): record for record in transform_records
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    combined = []
    section_records = []
    for _, section in he_table.sort_values(["z_um", "section_id"]).iterrows():
        section_id = str(int(section["section_id"]))
        source_table = (
            source_dir
            / "segmentation"
            / f"section-{int(section_id):03d}"
            / "nuclei_registered.csv"
        )
        if not source_table.is_file():
            raise FileNotFoundError(source_table)
        table = pd.read_csv(source_table)
        if len(table):
            if not table["section_id"].astype(int).eq(int(section_id)).all():
                raise ValueError(f"Section ID mismatch in {source_table}")
            fixed = add_fixed_coordinates(table, matrices[section_id])
            fixed["coordinate_frame"] = OUTPUT_FRAME
            fixed["source_coordinate_frame"] = SOURCE_FRAME
            fixed["fixed_transform_type"] = transform_by_id[section_id][
                "transform_type"
            ]
            fixed["fixed_transform_sha256"] = transform_by_id[section_id][
                "matrix_sha256"
            ]
            combined.append(fixed)
        else:
            fixed = table.copy()
            fixed["x_fixed_he_um"] = pd.Series(dtype=float)
            fixed["y_fixed_he_um"] = pd.Series(dtype=float)
            fixed["coordinate_frame"] = pd.Series(dtype=str)
            fixed["source_coordinate_frame"] = pd.Series(dtype=str)
            fixed["fixed_transform_type"] = pd.Series(dtype=str)
            fixed["fixed_transform_sha256"] = pd.Series(dtype=str)

        section_output = output_dir / "sections" / f"section-{int(section_id):03d}"
        section_output.mkdir(parents=True, exist_ok=False)
        section_path = section_output / "nuclei_fixed.csv"
        fixed.to_csv(section_path, index=False)
        section_records.append(
            {
                "section_id": section_id,
                "z_um": float(section["z_um"]),
                "source_kind": str(section["source_kind"]),
                "n_nuclei": int(len(fixed)),
                "source_table": str(source_table),
                "fixed_table": str(section_path),
                "native_mask_directory": str(
                    source_dir / "segmentation" / f"section-{int(section_id):03d}"
                ),
                "transform": transform_by_id[section_id],
            }
        )

    if combined:
        all_nuclei = pd.concat(combined, ignore_index=True)
    else:
        all_nuclei = pd.DataFrame()
    combined_path = output_dir / "nuclei_fixed.parquet"
    all_nuclei.to_parquet(combined_path, index=False)

    manifest = {
        "format": "deepspatial-fixed-he-single-cell-v1",
        "status": "complete",
        "dataset": "00029/g0",
        "source_segmentation": str(source_dir),
        "source_coordinate_frame": SOURCE_FRAME,
        "output_coordinate_frame": OUTPUT_FRAME,
        "transform_rule": "p_fixed = M_section_source_to_fixed @ p_registered",
        "he_table": str(he_table_path),
        "n_sections": int(len(section_records)),
        "n_nuclei": int(len(all_nuclei)),
        "consolidated_table": str(combined_path),
        "candidate_st": {
            "directory": str(ROOT / "data/00029_g0/anchors_st_to_fixed_he_candidate"),
            "coordinate_key": "spatial_st_corrected",
            "coordinate_frame": OUTPUT_FRAME,
        },
        "fixed_feature_stores": {
            "directory": str(
                ROOT / "data/00029_g0/fixed_he_feature_frame_v2_interpolated"
            ),
            "coordinate_frame": OUTPUT_FRAME,
        },
        "mask_semantics": (
            "nuclei.tif remains in native high-resolution H&E crop pixels; "
            "nuclei_fixed.parquet provides its centroid coordinates in the fixed frame"
        ),
        "sections": section_records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_safe) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "# 00029/g0 fixed-H&E single-cell coordinates\n\n"
        "This is a non-destructive coordinate materialization. The source Cellpose "
        "instance masks remain in native high-resolution H&E crop coordinates. "
        "`nuclei_fixed.parquet` and `sections/*/nuclei_fixed.csv` add centroid "
        "coordinates in `00029__g0_fixed_he`, the same frame used by corrected ST "
        "(`obsm['spatial_st_corrected']`) and the fixed morphology feature stores.\n\n"
        "This represents anatomical correspondence, not cell tracking.\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--he-table", type=Path, default=DEFAULT_HE_TABLE)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--residual-dir", type=Path, default=DEFAULT_RESIDUAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = materialize(
        he_table_path=args.he_table,
        source_dir=args.source_dir,
        residual_dir=args.residual_dir,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "n_sections": manifest["n_sections"],
                "n_nuclei": manifest["n_nuclei"],
                "output_coordinate_frame": manifest["output_coordinate_frame"],
                "output_dir": str(Path(args.output_dir).resolve()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
