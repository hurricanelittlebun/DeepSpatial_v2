#!/usr/bin/env python3
"""Build an image-only H&E mask for the reviewed anchor sections.

This is deliberately a QC/derived-data step.  It does not modify any H5AD,
registration transform, ST coordinate, feature store, or original mask.  The
mask is based on visible stain in the native H&E crop, rather than on the
Xenium-support intersection used by the original exported mask.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image


SECTIONS = (1, 11, 21, 31, 41, 51, 61, 71, 81, 91)


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        return np.asarray(source.convert("RGB"), dtype=np.uint8)


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        return np.asarray(source.convert("L"), dtype=np.uint8) > 0


def make_visible_mask(
    rgb: np.ndarray,
    *,
    saturation_threshold: float,
    mean_value_threshold: float,
    min_component_area: int,
    open_radius: int,
    close_radius: int,
) -> np.ndarray:
    """Keep visibly stained H&E pixels and remove disconnected tiny debris."""

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    saturation = hsv[..., 1] / 255.0
    mean_value = rgb.astype(np.float32).mean(axis=2) / 255.0
    candidate = (saturation >= saturation_threshold) & (
        mean_value <= mean_value_threshold
    )

    if open_radius > 0:
        radius = int(open_radius)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )
        candidate = cv2.morphologyEx(
            candidate.astype(np.uint8), cv2.MORPH_OPEN, kernel
        ).astype(bool)
    if close_radius > 0:
        radius = int(close_radius)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )
        candidate = cv2.morphologyEx(
            candidate.astype(np.uint8), cv2.MORPH_CLOSE, kernel
        ).astype(bool)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate.astype(np.uint8), connectivity=8
    )
    retained = np.zeros_like(candidate, dtype=bool)
    for label in range(1, n_labels):
        if int(stats[label, cv2.CC_STAT_AREA]) >= int(min_component_area):
            retained[labels == label] = True
    return retained


def classify(mask: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(points).all(axis=1)
    x = np.rint(points[:, 0]).astype(np.int64)
    y = np.rint(points[:, 1]).astype(np.int64)
    valid = finite & (x >= 0) & (x < mask.shape[1]) & (y >= 0) & (y < mask.shape[0])
    inside = np.zeros(len(points), dtype=bool)
    inside[valid] = mask[y[valid], x[valid]]
    return inside, valid


def points_from_h5ad(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    import anndata as ad

    adata = ad.read_h5ad(path, backed="r")
    try:
        obs = adata.obs.copy()
    finally:
        adata.file.close()
    points = obs[["x_he_local_px", "y_he_local_px"]].to_numpy(dtype=float)
    return obs, points


def render_overlay(
    rgb: np.ndarray,
    mask: np.ndarray,
    points: np.ndarray,
    output: Path,
    section_id: int,
    inside_fraction: float,
    *,
    max_dimension: int,
    max_points: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    height, width = mask.shape
    scale = min(1.0, max_dimension / max(height, width))
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    image_small = np.asarray(
        Image.fromarray(rgb).resize(target, Image.Resampling.LANCZOS)
    )
    mask_small = np.asarray(
        Image.fromarray((mask.astype(np.uint8) * 255)).resize(
            target, Image.Resampling.NEAREST
        )
    ) > 0
    scaled_points = points * scale
    inside, valid = classify(mask, points)
    inside_points = scaled_points[inside & valid]
    outside_points = scaled_points[(~inside) & valid]
    if len(inside_points) > max_points:
        inside_points = inside_points[
            np.linspace(0, len(inside_points) - 1, max_points, dtype=np.int64)
        ]
    if len(outside_points) > max_points:
        outside_points = outside_points[
            np.linspace(0, len(outside_points) - 1, max_points, dtype=np.int64)
        ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=140)
    ax = axes[0]
    ax.imshow(image_small, origin="upper")
    ax.contour(mask_small.astype(float), levels=[0.5], colors="#ffe600", linewidths=0.9, origin="upper")
    ax.set_title("H&E + visible-stain mask boundary")

    ax = axes[1]
    rgba = image_small.astype(np.float32) / 255.0
    overlay = np.zeros_like(rgba)
    overlay[..., 1] = 1.0
    overlay[..., 0] = 0.05
    overlay[..., 2] = 0.25
    alpha = mask_small.astype(np.float32) * 0.28
    ax.imshow(rgba)
    ax.imshow(overlay, alpha=alpha, origin="upper")
    ax.contour(mask_small.astype(float), levels=[0.5], colors="#00ff55", linewidths=0.8, origin="upper")
    ax.set_title("Green fill = retained visible tissue")

    ax = axes[2]
    ax.imshow(image_small, origin="upper")
    if len(inside_points):
        ax.scatter(inside_points[:, 0], inside_points[:, 1], s=0.8, c="#00ff55", alpha=0.42, linewidths=0)
    if len(outside_points):
        ax.scatter(outside_points[:, 0], outside_points[:, 1], s=0.8, c="#ff2638", alpha=0.42, linewidths=0)
    ax.contour(mask_small.astype(float), levels=[0.5], colors="#00ffff", linewidths=0.8, origin="upper")
    ax.set_title(f"ST: green=in, red=out | in mask={inside_fraction:.3%}")

    for ax in axes:
        ax.set_xlim(0, target[0])
        ax.set_ylim(target[1], 0)
        ax.set_aspect("equal")
        ax.set_xlabel("H&E crop x (pixel)")
        ax.set_ylabel("H&E crop y (pixel)")
    fig.suptitle(f"9957/g0 section {section_id:03d} | visible-stain mask v2", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-h5ad", type=Path, required=True)
    parser.add_argument("--anchor-dir", type=Path)
    parser.add_argument("--v2-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--saturation", type=float, default=0.15)
    parser.add_argument("--mean-value", type=float, default=0.92)
    parser.add_argument("--min-area", type=int, default=500)
    parser.add_argument("--open-radius", type=int, default=1)
    parser.add_argument("--close-radius", type=int, default=2)
    parser.add_argument("--max-dimension", type=int, default=1800)
    parser.add_argument("--max-points", type=int, default=100000)
    args = parser.parse_args()

    output = args.output
    masks_dir = output / "masks"
    overlays_dir = output / "overlays"
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    section_metrics = []
    mask_metrics = []
    anchor_dir = args.anchor_dir or (args.group_h5ad.parent / "anchors")

    for section_id in SECTIONS:
        h5ad_path = anchor_dir / f"section-{section_id:03d}.h5ad"
        obs, points = points_from_h5ad(h5ad_path)
        raw_path = Path(str(obs["raw_crop_path"].iloc[0]))
        if not raw_path.is_absolute():
            raw_path = args.v2_root / raw_path
        rgb = load_rgb(raw_path)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"unexpected H&E shape for section {section_id}: {rgb.shape}")
        mask = make_visible_mask(
            rgb,
            saturation_threshold=args.saturation,
            mean_value_threshold=args.mean_value,
            min_component_area=args.min_area,
            open_radius=args.open_radius,
            close_radius=args.close_radius,
        )
        output_mask = masks_dir / f"section-{section_id:03d}__he_visible_stain_mask.png"
        Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(output_mask)
        inside, valid = classify(mask, points)
        valid_count = int(valid.sum())
        inside_count = int((inside & valid).sum())
        original_path = Path(str(obs["mask_path"].iloc[0]))
        if not original_path.is_absolute():
            original_path = args.v2_root / original_path
        original = load_mask(original_path)
        mask_metrics.append(
            {
                "section_id": section_id,
                "image_width_px": rgb.shape[1],
                "image_height_px": rgb.shape[0],
                "new_mask_fraction": float(mask.mean()),
                "original_mask_fraction": float(original.mean()),
                "new_minus_original_fraction": float(np.mean(mask & ~original)),
                "original_minus_new_fraction": float(np.mean(original & ~mask)),
                "new_mask_path": str(output_mask),
                "original_mask_path": str(original_path),
            }
        )
        overlay_path = overlays_dir / f"section-{section_id:03d}__he_st_visible_mask.png"
        render_overlay(
            rgb,
            mask,
            points,
            overlay_path,
            section_id,
            inside_count / valid_count if valid_count else float("nan"),
            max_dimension=args.max_dimension,
            max_points=args.max_points,
        )
        section_metrics.append(
            {
                "section_id": section_id,
                "n_st_cells": int(len(points)),
                "n_valid_points": valid_count,
                "n_inside_new_mask": inside_count,
                "n_outside_new_mask": int(valid_count - inside_count),
                "inside_new_mask_fraction": inside_count / valid_count if valid_count else np.nan,
                "image_path": str(raw_path),
                "mask_path": str(output_mask),
                "overlay_path": str(overlay_path),
            }
        )

    pd.DataFrame(section_metrics).to_csv(output / "section_metrics.csv", index=False)
    pd.DataFrame(mask_metrics).to_csv(output / "mask_metrics.csv", index=False)
    manifest = {
        "status": "complete",
        "sample_id": "9957",
        "group_id": "g0",
        "source_group_h5ad": str(args.group_h5ad),
        "source_coordinate_frame": "native H&E crop pixels",
        "changes": [
            "image-only visible-stain mask",
            "no ST coordinate modification",
            "no registration transform modification",
            "no H5AD modification",
        ],
        "parameters": {
            "saturation_threshold": args.saturation,
            "mean_value_threshold": args.mean_value,
            "min_component_area": args.min_area,
            "open_radius": args.open_radius,
            "close_radius": args.close_radius,
        },
        "sections": list(SECTIONS),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
