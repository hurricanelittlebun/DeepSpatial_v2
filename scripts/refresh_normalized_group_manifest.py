"""Refresh the manifest for a directory of per-series normalized H5AD files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import anndata as ad
import pandas as pd


GROUP_FILE = re.compile(r"^\d{4,5}_g\d+\.h5ad$")


def refresh_manifest(output_dir: Path | str, *, section_thickness_um: float = 5.0) -> dict:
    root = Path(output_dir).resolve()
    paths = sorted(path for path in root.glob("*.h5ad") if GROUP_FILE.match(path.name))
    if not paths:
        raise FileNotFoundError(f"No per-group H5AD files found in {root}")

    records = []
    section_frames = []
    total_obs = 0
    n_vars = None
    for path in paths:
        value = ad.read_h5ad(path, backed="r")
        try:
            if "series_id" not in value.obs:
                raise KeyError(f"{path} is missing obs['series_id']")
            series = value.obs["series_id"].astype(str).unique().tolist()
            if len(series) != 1:
                raise ValueError(f"{path} contains multiple series_id values")
            current_n_vars = int(value.n_vars)
            if n_vars is None:
                n_vars = current_n_vars
            elif current_n_vars != n_vars:
                raise ValueError(f"Gene count differs in {path}")
            sections = sorted(value.obs["section_id"].astype(int).unique().tolist())
            labels = [
                column
                for column in (
                    "cell_class",
                    "assigned_celltype",
                    "pred_domain",
                    "pred_niche",
                )
                if column in value.obs
            ]
            missing = {
                column: int(value.obs[column].isna().sum())
                for column in labels
            }
            records.append(
                {
                    "series_id": series[0],
                    "output": str(path),
                    "n_obs": int(value.n_obs),
                    "n_vars": int(value.n_vars),
                    "n_sections": len(sections),
                    "sections": sections,
                    "labels": labels,
                    "label_missing": missing,
                    "size_bytes": int(path.stat().st_size),
                }
            )
            section_frames.append(
                value.obs.groupby(
                    ["series_id", "section_id", "series_section_order", "z_um"],
                    observed=True,
                    dropna=False,
                )
                .size()
                .reset_index(name="n_cells")
            )
            total_obs += int(value.n_obs)
        finally:
            value.file.close()

    manifest = {
        "format": "deepspatial-normalized-groups-v1",
        "manifest_scope": "all per-tissue-g H5AD files in this directory",
        "coordinate_units": "micrometer",
        "section_thickness_um": float(section_thickness_um),
        "source_h5ads": [
            "/data/buyonggan/DeepSpatial_v2/data/cell_level_aligned/DeepSpatial_cell_level_aligned.h5ad",
            str((root / "all_other_groups_annotated_predicted.h5ad").resolve()),
        ],
        "group_count": len(records),
        "total_n_obs": total_obs,
        "n_vars": n_vars,
        "groups": records,
        "checks": {
            "grouping_key": "joint_series_id",
            "expression_matrix": "raw counts preserved in X",
            "obs_names": "joint_cell_id",
            "spatial": "registered XY in obsm['spatial']",
            "h_and_e_only_sections": "not included as cells",
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    if section_frames:
        section_summary = pd.concat(section_frames, ignore_index=True)
        section_summary.sort_values(
            ["series_id", "series_section_order", "section_id"]
        ).to_csv(root / "section_summary.csv", index=False)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--section-thickness-um", type=float, default=5.0)
    args = parser.parse_args()
    print(json.dumps(refresh_manifest(args.output_dir, section_thickness_um=args.section_thickness_um), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
