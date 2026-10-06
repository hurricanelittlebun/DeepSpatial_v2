"""Render a 3-D opaque, high-contrast coordinate QC PNG for 9957/g0."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from render_9957_coordinate_repair_qc import (  # noqa: E402
    _downsample,
    _load_feature_records,
    feature_grid_boundary,
)


DEFAULT_GROUP = ROOT / "data" / "9957_g0"
DEFAULT_CANDIDATE = (
    DEFAULT_GROUP
    / "registration_correction_he_v5_coordinate_repair"
    / "9957_g0__st_to_global_v4_repaired_candidate.h5ad"
)
DEFAULT_FEATURE_STORE = (
    DEFAULT_GROUP
    / "reconstruction_prep_v5_coordinate_repair"
    / "uni2_features_global_v4_repaired.h5"
)
DEFAULT_OUTPUT = (
    DEFAULT_GROUP
    / "qc"
    / "coordinate_repair_v5_final_qc_v2"
    / "coordinate_repair_opaque_3d_high_contrast_v1.png"
)

COLORS = [
    "#e41a1c",  # red
    "#0057b7",  # blue
    "#00a651",  # green
    "#6a1b9a",  # purple
    "#ff7f00",  # orange
    "#008fb3",  # cyan
    "#b8860b",  # gold
    "#4d4d4d",  # dark gray
    "#e6007e",  # magenta
    "#006400",  # dark green
]


def _load_all_he_support(feature_store_path: Path, max_boundary_points: int = 1200):
    from deepspatial.histology.feature_store import FeatureStore

    store = FeatureStore(feature_store_path, cache_mb=256)
    support = []
    with h5py.File(feature_store_path, "r") as handle:
        for sid in store.sections:
            metadata = store.metadata[sid]
            key = store._key(sid)
            valid = handle["sections"][key]["valid"][:]
            boundary = feature_grid_boundary(
                valid,
                origin_um=metadata["origin_um"],
                spacing_um=metadata["spacing_um"],
                max_points=max_boundary_points,
            )
            if len(boundary):
                support.append(
                    {
                        "section_id": sid,
                        "z_um": float(metadata["z_um"]),
                        "points": boundary,
                    }
                )
    return support


def render(candidate: Path, feature_store: Path, output: Path) -> dict:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    records, frame = _load_feature_records(
        candidate, feature_store, max_boundary_points=8000
    )
    all_he = _load_all_he_support(feature_store)
    output.parent.mkdir(parents=True, exist_ok=True)

    z_values = np.array([record["z_um"] for record in records], dtype=float)
    z_min = float(z_values.min())
    z_max = float(z_values.max())
    z_span = max(z_max - z_min, 1.0)
    z_stretch = 5.0

    fig = plt.figure(figsize=(17, 13), dpi=220)
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    # All registered H&E/UNI2 feature support sections are black.  This gives
    # the continuous z-stack, while the ten ST anchor layers remain colored.
    for item in all_he:
        points = np.asarray(item["points"], dtype=float)
        z = np.full(len(points), (item["z_um"] - z_min) * z_stretch)
        ax.scatter(
            points[:, 0],
            points[:, 1],
            z,
            s=0.55,
            c="#111111",
            alpha=1.0,
            depthshade=False,
            linewidths=0,
            rasterized=True,
        )

    handles = []
    for index, record in enumerate(records):
        color = COLORS[index % len(COLORS)]
        points = _downsample(np.asarray(record["points"], dtype=float), 15000)
        z = np.full(len(points), (record["z_um"] - z_min) * z_stretch)
        ax.scatter(
            points[:, 0],
            points[:, 1],
            z,
            s=0.22,
            c=color,
            alpha=1.0,
            depthshade=False,
            linewidths=0,
            rasterized=True,
        )
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=color,
                markeredgecolor=color,
                markersize=7,
                label=f"ST section {record['section_id']}",
            )
        )

    ax.set_xlabel("global X (µm)", labelpad=10, fontsize=12)
    ax.set_ylabel("global Y (µm)", labelpad=10, fontsize=12)
    ax.set_zlabel("Z (µm; display stretched ×5)", labelpad=10, fontsize=12)
    ax.set_title(
        "9957/g0 repaired coordinate QC — opaque 3-D view\n"
        "colored points = ST anchors; black points = all registered H&E/UNI2 support\n"
        f"frame: {frame}",
        pad=20,
        fontsize=15,
    )
    ax.view_init(elev=25, azim=-58)
    ax.set_proj_type("ortho")
    ax.set_box_aspect((1.0, 1.0, max(0.45, z_stretch * z_span / 4000.0)))
    ax.grid(False)
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=9,
        title="ST sections",
    )
    fig.subplots_adjust(left=0.02, right=0.82, bottom=0.04, top=0.88)
    fig.savefig(output, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return {
        "status": "created",
        "output": str(output.resolve()),
        "coordinate_frame": frame,
        "st_section_count": len(records),
        "he_section_count": len(all_he),
        "total_st_cells": int(sum(record["n_cells"] for record in records)),
        "z_min_um": z_min,
        "z_max_um": z_max,
        "z_display_stretch": z_stretch,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--feature-store", type=Path, default=DEFAULT_FEATURE_STORE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing image: {args.output}")
    print(render(args.candidate, args.feature_store, args.output))


if __name__ == "__main__":
    main()
