#!/usr/bin/env python
"""Plot the section-31 cell-type comparison from the completed parity runs.

The three panels use the same registered XY frame, point size, axis limits, and
explicit cell-type color map.  The prediction panels are virtual cells; they
are not treated as one-to-one matches to the held-out ST cells.
"""

from __future__ import annotations

import argparse
import colorsys
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def _read_points(path: Path, label_key: str = "cell_class"):
    adata = ad.read_h5ad(path, backed="r")
    obs = adata.obs
    required = {"x_registered_g_um", "y_registered_g_um", label_key}
    missing = required.difference(obs.columns)
    if missing:
        raise KeyError(f"{path} is missing columns: {sorted(missing)}")

    x = np.asarray(obs["x_registered_g_um"], dtype=float)
    y = np.asarray(obs["y_registered_g_um"], dtype=float)
    labels = obs[label_key].astype(str).to_numpy()
    valid = np.isfinite(x) & np.isfinite(y) & (labels != "nan")
    return x[valid], y[valid], labels[valid]


def _make_palette(labels: list[str]) -> dict[str, tuple[float, float, float]]:
    # Deterministic explicit palette: the same sorted label always receives
    # the same color across all three panels and future reruns.
    n = len(labels)
    colors = {}
    for i, label in enumerate(labels):
        hue = i / max(n, 1)
        colors[label] = colorsys.hsv_to_rgb(hue, 0.68, 0.86)
    return colors


def _plot_panel(ax, points, title, palette, xlim, ylim, point_size, alpha):
    x, y, labels = points
    for label in palette:
        mask = labels == label
        if np.any(mask):
            ax.scatter(
                x[mask],
                y[mask],
                s=point_size,
                c=[palette[label]],
                alpha=alpha,
                linewidths=0,
                rasterized=True,
            )
    ax.set_title(f"{title}\nn={len(x):,}", fontsize=12, pad=10)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.set_xlabel("registered global X (µm)")
    ax.grid(True, color="#d9dde3", linewidth=0.45, alpha=0.6)
    ax.set_facecolor("#fbfcfe")
    for spine in ax.spines.values():
        spine.set_color("#9aa4b2")
        spine.set_linewidth(0.8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/00029_g0/visualizations/section31_celltype_comparison_v1"),
    )
    args = parser.parse_args()

    files = {
        "Ground truth ST31": Path(
            "data/00029_g0/st_only_residual_v1/celltype_domain_niche_anchors_v1/section-031.h5ad"
        ),
        "Original / celltype-only": Path(
            "data/00029_g0/experiments/odd_to_even_3d_control_v1/celltype_only/section-031_fixed_depth_prediction.h5ad"
        ),
        "H&E + celltype": Path(
            "data/00029_g0/experiments/odd_to_even_3d_control_v1/he_celltype/section-031_fixed_depth_prediction.h5ad"
        ),
    }
    points = {name: _read_points(path) for name, path in files.items()}
    labels = sorted({label for _, _, labs in points.values() for label in labs})
    palette = _make_palette(labels)

    all_x = np.concatenate([p[0] for p in points.values()])
    all_y = np.concatenate([p[1] for p in points.values()])
    x_span = float(np.max(all_x) - np.min(all_x))
    y_span = float(np.max(all_y) - np.min(all_y))
    margin_x = max(40.0, 0.02 * x_span)
    margin_y = max(40.0, 0.02 * y_span)
    xlim = (float(np.min(all_x) - margin_x), float(np.max(all_x) + margin_x))
    ylim = (float(np.min(all_y) - margin_y), float(np.max(all_y) + margin_y))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    point_size = 1.7
    alpha = 0.72

    # Main three-panel comparison.
    fig, axes = plt.subplots(1, 3, figsize=(22, 8.7), dpi=180)
    fig.subplots_adjust(left=0.035, right=0.78, bottom=0.12, top=0.86, wspace=0.08)
    for ax, (name, pts) in zip(axes, points.items()):
        _plot_panel(ax, pts, name, palette, xlim, ylim, point_size, alpha)
    axes[0].set_ylabel("registered global Y (µm)")
    axes[1].set_ylabel("")
    axes[2].set_ylabel("")

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=6,
            markerfacecolor=palette[label],
            markeredgecolor="none",
            label=label,
        )
        for label in labels
    ]
    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.795, 0.50),
        ncol=2,
        fontsize=8.2,
        title="cell_class (37 classes)",
        title_fontsize=10,
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.35,
    )
    fig.suptitle(
        "00029/g0 section 31 | held-out cell-type spatial comparison\n"
        "same registered XY frame and explicit cell-type colors; predictions are virtual cells",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.40,
        0.045,
        "Section 31 was not used as a training anchor in the odd→even experiment.",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    composite = args.output_dir / "section-031_celltype_three_panel_comparison.png"
    fig.savefig(composite, dpi=260, facecolor="white")
    plt.close(fig)

    # Also export the three panels separately, using exactly the same limits
    # and palette so they can be reviewed independently.
    for index, (name, pts) in enumerate(points.items(), start=1):
        fig, ax = plt.subplots(figsize=(8.2, 7.5), dpi=180)
        fig.subplots_adjust(left=0.11, right=0.98, bottom=0.10, top=0.88)
        _plot_panel(ax, pts, name, palette, xlim, ylim, point_size, alpha)
        ax.set_ylabel("registered global Y (µm)")
        fig.suptitle(
            f"00029/g0 section 31 | {name}\ncell-type spatial view",
            fontsize=14,
            fontweight="bold",
        )
        out = args.output_dir / f"section-031_celltype_panel_{index:02d}.png"
        fig.savefig(out, dpi=260, facecolor="white")
        plt.close(fig)

    print(f"WROTE {composite}")
    for path in sorted(args.output_dir.glob("*.png")):
        print(f"  {path} ({path.stat().st_size:,} bytes)")
    print(f"LABELS {len(labels)}")
    print(f"XLIM {xlim}")
    print(f"YLIM {ylim}")


if __name__ == "__main__":
    main()
