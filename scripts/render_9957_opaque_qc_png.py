"""Render one opaque, high-contrast PNG for 9957/g0 coordinate QC."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from render_9957_coordinate_repair_qc import _load_feature_records  # noqa: E402


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
    / "coordinate_repair_opaque_high_contrast_v1.png"
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


def render(candidate: Path, feature_store: Path, output: Path) -> dict:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    records, frame = _load_feature_records(
        candidate, feature_store, max_boundary_points=8000
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 14), dpi=240, constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Draw registered H&E/UNI2 feature support first as opaque black points.
    # ST points are drawn afterwards so the section colors remain visible.
    for record in records:
        boundary = np.asarray(record["feature_boundary"], dtype=float)
        if len(boundary):
            ax.scatter(
                boundary[:, 0],
                boundary[:, 1],
                s=2.5,
                c="#111111",
                alpha=1.0,
                linewidths=0,
                rasterized=True,
            )

    handles = []
    for index, record in enumerate(records):
        color = COLORS[index % len(COLORS)]
        points = np.asarray(record["points"], dtype=float)
        scatter = ax.scatter(
            points[:, 0],
            points[:, 1],
            s=0.42,
            c=color,
            alpha=1.0,
            linewidths=0,
            rasterized=True,
            label=f"ST section {record['section_id']}",
        )
        handles.append(scatter)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("global X (µm)", fontsize=13)
    ax.set_ylabel("global Y (µm)", fontsize=13)
    ax.set_title(
        "9957/g0 repaired coordinate QC — opaque high-contrast view\n"
        "colored points = ST anchors; black points = registered H&E/UNI2 support\n"
        f"frame: {frame}",
        fontsize=15,
    )
    ax.grid(False)
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=10,
        markerscale=8,
        title="ST sections",
    )
    for spine in ax.spines.values():
        spine.set_color("#334155")
    fig.savefig(output, dpi=240, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return {
        "status": "created",
        "output": str(output.resolve()),
        "coordinate_frame": frame,
        "section_count": len(records),
        "total_st_cells": int(sum(record["n_cells"] for record in records)),
        "support_boundary_points": int(
            sum(len(record["feature_boundary"]) for record in records)
        ),
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
