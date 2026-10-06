"""Diagnose the 9957/g0 coordinate-frame break before reconstruction.

This compares the two coordinate columns preserved in the v4 ST candidate:
``spatial_st_corrected_pre_manual_v4`` and ``spatial_st_corrected_v4``.
It is intentionally diagnostic only and never overwrites an H5AD.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_H5AD = Path(
    "/data/buyonggan/DeepSpatial/data/9957_g0/"
    "registration_correction_he_v4_manual_rotation/"
    "9957_g0__st_to_fixed_he_candidate.h5ad"
)
DEFAULT_OUTPUT = Path(
    "/data/buyonggan/DeepSpatial/data/9957_g0/qc/coordinate_break_v4"
)


def _section_order(values: np.ndarray) -> list[str]:
    return sorted({str(v) for v in values}, key=lambda v: (int(v), v))


def render(h5ad_path: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    source = ad.read_h5ad(h5ad_path, backed="r")
    required = {
        "section_id",
        "spatial_st_corrected_pre_manual_v4",
        "spatial_st_corrected_v4",
    }
    missing = sorted(
        [f"obs:{key}" for key in ["section_id"] if key not in source.obs]
        + [f"obsm:{key}" for key in required - {"section_id"} if key not in source.obsm]
    )
    if missing:
        source.file.close()
        raise ValueError(f"Missing diagnostic fields: {missing}")

    section = source.obs["section_id"].astype(str).to_numpy()
    sections = _section_order(section)
    before = np.asarray(source.obsm["spatial_st_corrected_pre_manual_v4"])
    after = np.asarray(source.obsm["spatial_st_corrected_v4"])

    palette = {
        s: plt.get_cmap("tab20")(i % 20)
        for i, s in enumerate(sections)
    }
    centers = {"before": {}, "after": {}}
    rows = []
    for s in sections:
        mask = section == s
        b = before[mask]
        a = after[mask]
        centers["before"][s] = b.mean(axis=0)
        centers["after"][s] = a.mean(axis=0)
        rows.append(
            {
                "section_id": int(s),
                "n_cells": int(mask.sum()),
                "before_center_x_um": float(b[:, 0].mean()),
                "before_center_y_um": float(b[:, 1].mean()),
                "after_center_x_um": float(a[:, 0].mean()),
                "after_center_y_um": float(a[:, 1].mean()),
                "delta_center_x_um": float(a[:, 0].mean() - b[:, 0].mean()),
                "delta_center_y_um": float(a[:, 1].mean() - b[:, 1].mean()),
            }
        )

    # Use common limits so the displacement cannot be hidden by independent axes.
    all_xy = np.vstack([before, after])
    x_min, y_min = all_xy.min(axis=0) - 100
    x_max, y_max = all_xy.max(axis=0) + 100
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
    for ax, xy, title, key in [
        (axes[0], before, "Before manual v4 affine\n(expected continuous frame)", "before"),
        (axes[1], after, "Current v4 affine\n(used by reconstruction)", "after"),
    ]:
        for s in sections:
            mask = section == s
            ax.scatter(
                xy[mask, 0],
                xy[mask, 1],
                s=0.7,
                alpha=0.35,
                color=palette[s],
                label=s,
                rasterized=True,
            )
        center_xy = np.vstack([centers[key][s] for s in sections])
        ax.plot(center_xy[:, 0], center_xy[:, 1], color="black", linewidth=1.2, zorder=10)
        for s, point in zip(sections, center_xy):
            ax.text(point[0], point[1], s, fontsize=9, weight="bold", zorder=11)
        ax.set_title(title)
        ax.set_xlabel("registered X (µm)")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("registered Y (µm)")
    axes[1].legend(title="section", bbox_to_anchor=(1.03, 1), loc="upper left")
    fig.suptitle(
        "9957/g0 anchor coordinate-frame diagnostic\n"
        "The break after section 51 is introduced by the v4 manual affine",
        fontsize=14,
    )
    fig.tight_layout()
    figure_path = output / "anchor_coordinate_frame_comparison.png"
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    csv_path = output / "anchor_section_centers_before_after.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "source_h5ad": str(h5ad_path.resolve()),
        "before_coordinate_key": "spatial_st_corrected_pre_manual_v4",
        "after_coordinate_key": "spatial_st_corrected_v4",
        "sections": [int(s) for s in sections],
        "n_obs": int(source.n_obs),
        "n_updated_by_v4": int(np.any(np.abs(after - before) > 1e-8, axis=1).sum()),
        "diagnostic_plot": str(figure_path.resolve()),
        "section_centers_csv": str(csv_path.resolve()),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    source.file.close()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(render(args.h5ad.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
