#!/usr/bin/env python3
"""Render high-contrast fixed-H&E versus original/corrected ST overlays."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import fit_00029_st_to_he_residual as residual


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data/00029_g0/st_only_residual_v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--registration-root", type=Path, default=Path("/data/buyonggan/DeepSpatial_v2/outputs/registration"))
    parser.add_argument("--anchor-dir", type=Path, default=DEFAULT_DATASET / "anchors")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATASET / "qc/residual_contrast")
    parser.add_argument("--sections", type=int, nargs="+", default=list(residual.ANCHORS))
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    anchor_dir = args.anchor_dir.resolve()
    manifest = pd.read_parquet(args.registration_root / "manifests/stalign_section_transforms.parquet")
    maps = residual.RegistrationMaps(args.registration_root, "00029", "g0")
    for section_id in args.sections:
        anchor_file = anchor_dir / f"section-{section_id:03d}.h5ad"
        if not anchor_file.is_file():
            raise FileNotFoundError(anchor_file)
        section_transform = args.registration_root / "transforms/00029/g0" / (
            f"section-{section_id}-00029__g0__section-{section_id}.npz"
        )
        with h5py.File(anchor_file, "r") as anchor:
            before = np.asarray(anchor["obsm"]["spatial_registered"], dtype=float)
            after = np.asarray(anchor["obsm"]["spatial_st_corrected"], dtype=float)
        with np.load(section_transform, allow_pickle=False) as source:
            chain = tuple(
                (str(key), str(direction))
                for key, direction in __import__("json").loads(str(source["edge_chain_json"].item()))
            )
        raster_row = manifest[
            manifest["sample_id"].astype(str).eq("00029")
            & manifest["group_id"].astype(str).eq("g0")
            & manifest["section_id"].eq(int(section_id))
        ].iloc[0]
        with np.load(args.registration_root / str(raster_row["raster_path"]), allow_pickle=False) as raster:
            grid = np.asarray(raster["grid_um"], dtype=float)
            mask = np.asarray(raster["mask"], dtype=bool)
        he = maps.apply_chain(grid[:, mask].T, chain, inverse=False)
        figure, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=180, facecolor="#101010")
        for axis, points, title, point_color in (
            (axes[0], before, "before: original ST", "#ff3030"),
            (axes[1], after, "after: ST residual-corrected", "#48a9ff"),
        ):
            axis.set_facecolor("#101010")
            axis.scatter(he[:, 0], he[:, 1], s=0.7, c="#00e676", alpha=0.48, linewidths=0, rasterized=True)
            axis.scatter(points[:, 0], points[:, 1], s=0.28, c=point_color, alpha=0.58, linewidths=0, rasterized=True)
            axis.set_aspect("equal")
            axis.set_title(title, color="white")
            axis.tick_params(colors="white")
            for spine in axis.spines.values():
                spine.set_color("white")
            axis.set_xlabel("fixed H&E / g x (µm)", color="white")
            axis.set_ylabel("fixed H&E / g y (µm)", color="white")
        legend = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#00e676", markersize=6, label="fixed H&E mask"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#ff3030", markersize=6, label="original ST"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#48a9ff", markersize=6, label="corrected ST"),
        ]
        figure.legend(handles=legend, loc="lower center", ncol=3, facecolor="#101010", labelcolor="white")
        figure.suptitle(f"00029/g0 section {section_id}: fixed H&E and ST residual correction", color="white")
        figure.tight_layout(rect=(0, 0.06, 1, 0.95))
        figure.savefig(output / f"section-{section_id:03d}__contrast_before_after.png", facecolor=figure.get_facecolor())
        plt.close(figure)


if __name__ == "__main__":
    main()
