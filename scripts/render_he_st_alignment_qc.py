#!/usr/bin/env python3
"""Render per-section H&E/ST alignment quality-control overlays.

The overlays use the already exported H&E-local pixel coordinates from the
normalized group AnnData files.  They do not refit registration and they do
not modify any AnnData file.  The saved section transform is reported in the
figure metadata, while the actual H&E/ST overlay is drawn in the native H&E
crop coordinate frame:

    x_he_local_px, y_he_local_px -> raw H&E crop pixels

This is the appropriate frame for judging whether the ST cell centroids land
on the same tissue visible in the H&E crop.  Global registration is a separate
operation and is represented by the transform metadata and registered
coordinates already present in the input AnnData.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V2_ROOT = DEFAULT_REPO_ROOT.parent / "DeepSpatial_v2"
DEFAULT_GROUP_DIR = DEFAULT_REPO_ROOT / "data" / "normalized_groups_v1"


def discover_group_h5ads(group_dir: Path) -> list[Path]:
    """Find per-group H5ADs while excluding merged/all-groups files."""

    return sorted(
        path
        for path in group_dir.glob("*_g*.h5ad")
        if re.fullmatch(r".+_g\d+", path.stem)
    )


def resolve_data_path(value: Any, v2_root: Path) -> Path | None:
    """Resolve a path stored in AnnData metadata.

    The exported metadata contains both absolute paths and paths relative to
    DeepSpatial_v2.  Empty/NaN values are treated as missing.
    """

    if value is None:
        return None
    text = str(value)
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return None
    path = Path(text)
    return path if path.is_absolute() else v2_root / path


def classify_points_against_mask(
    mask: np.ndarray, points_xy: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Classify ``[x, y]`` points against a 2-D image mask.

    Pixel sampling follows the existing export convention: coordinates are
    rounded to the nearest pixel, then indexed as ``mask[row=y, col=x]``.
    ``valid`` distinguishes points outside the image rectangle from points
    that are inside the rectangle but outside tissue.
    """

    mask = np.asarray(mask).astype(bool)
    points = np.asarray(points_xy, dtype=float)
    if mask.ndim != 2:
        raise ValueError("mask must be a 2-D array")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points_xy must have shape [N, 2]")
    finite = np.isfinite(points).all(axis=1)
    columns = np.rint(points[:, 0]).astype(np.int64)
    rows = np.rint(points[:, 1]).astype(np.int64)
    valid = (
        finite
        & (rows >= 0)
        & (rows < mask.shape[0])
        & (columns >= 0)
        & (columns < mask.shape[1])
    )
    inside = np.zeros(len(points), dtype=bool)
    inside[valid] = mask[rows[valid], columns[valid]]
    return inside, valid


def deterministic_point_downsample(points: np.ndarray, max_points: int) -> np.ndarray:
    """Return an evenly distributed, deterministic point subset for plotting."""

    points = np.asarray(points)
    if max_points <= 0:
        raise ValueError("max_points must be positive")
    if len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
    return points[indices]


def _first_existing_path(values: list[Any], v2_root: Path) -> Path | None:
    for value in values:
        path = resolve_data_path(value, v2_root)
        if path is not None and path.exists():
            return path
    return None


def _safe_int(value: Any) -> int:
    return int(str(value).split(".", 1)[0])


def _format_optional(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as source:
        return source.convert("RGB").copy()


def _load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        array = np.asarray(source.convert("L"))
    return array > 0


def _resize_for_plot(
    image: Image.Image, mask: np.ndarray, max_dimension: int
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Resize image and mask together, returning x/y scale factors."""

    width, height = image.size
    scale = min(1.0, float(max_dimension) / float(max(width, height)))
    target_width = max(1, int(round(width * scale)))
    target_height = max(1, int(round(height * scale)))
    if target_width == width and target_height == height:
        return np.asarray(image), mask.astype(bool), 1.0, 1.0
    rgb = np.asarray(
        image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    )
    mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    mask_small = np.asarray(
        mask_image.resize((target_width, target_height), Image.Resampling.NEAREST)
    ) > 0
    return rgb, mask_small, target_width / width, target_height / height


def _draw_points(
    ax: Any,
    points: np.ndarray,
    color: str,
    max_points: int,
    size: float = 0.7,
    alpha: float = 0.42,
) -> None:
    points = deterministic_point_downsample(points, max_points)
    if len(points):
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=size,
            c=color,
            alpha=alpha,
            linewidths=0,
            rasterized=True,
        )


def _configure_image_axis(ax: Any, width: int, height: int) -> None:
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.set_xlabel("H&E crop x (pixel)")
    ax.set_ylabel("H&E crop y (pixel)")


def _draw_mask_boundary(ax: Any, mask: np.ndarray) -> None:
    # A contour on the resized array shares the same pixel frame as imshow.
    if mask.any() and (~mask).any():
        ax.contour(
            mask.astype(float),
            levels=[0.5],
            colors=["#ffe600"],
            linewidths=0.7,
            origin="upper",
        )


def _section_record(
    section_id: int,
    section: Any,
    v2_root: Path,
) -> tuple[dict[str, Any], Path | None, Path | None, np.ndarray]:
    """Extract one section's input paths and local ST points."""

    # ``section`` contains one row per ST cell.  Paths and registration
    # metadata are section-level values repeated on every row; use the first
    # row for those scalars and retain all rows for the point coordinates.
    if len(section) == 0:
        raise ValueError(f"section {section_id} has no cells")
    metadata = section.iloc[0]
    image_path = _first_existing_path(
        [
            metadata.get("raw_crop_path"),
            metadata.get("he_image_path"),
            metadata.get("he_path"),
        ],
        v2_root,
    )
    mask_path = _first_existing_path(
        [metadata.get("mask_path"), metadata.get("he_mask_path")], v2_root
    )
    x = np.asarray(section["x_he_local_px"], dtype=float)
    y = np.asarray(section["y_he_local_px"], dtype=float)
    points = np.column_stack([x, y])
    record: dict[str, Any] = {
        "section_id": int(section_id),
        "n_st_cells": int(len(points)),
        "image_path": str(image_path) if image_path else "",
        "mask_path": str(mask_path) if mask_path else "",
        "registration_transform_path": str(metadata.get("registration_transform_path", "")),
        "registration_status": str(metadata.get("registration_status", "")),
        "registration_confidence": float(section["registration_confidence"].median()),
        "forced_flip": bool(metadata.get("forced_flip", False)),
        "preorientation_axis": str(metadata.get("preorientation_axis", "none")),
        "source_coordinate_frame": str(metadata.get("source_coordinate_frame", "")),
        "coordinate_source": "x_he_local_px/y_he_local_px",
    }
    return record, image_path, mask_path, points


def render_section(
    *,
    record: dict[str, Any],
    image_path: Path,
    mask_path: Path,
    points: np.ndarray,
    output_path: Path,
    sample_id: str,
    group_id: str,
    max_dimension: int,
    max_plot_points: int,
) -> dict[str, Any]:
    """Render one four-panel H&E/ST alignment figure and return metrics."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    image = _load_rgb(image_path)
    mask = _load_mask(mask_path)
    if mask.shape != (image.height, image.width):
        raise ValueError(
            f"image/mask shape mismatch for section {record['section_id']}: "
            f"image={(image.height, image.width)} mask={mask.shape}"
        )
    inside, valid = classify_points_against_mask(mask, points)
    image_small, mask_small, sx, sy = _resize_for_plot(image, mask, max_dimension)
    points_small = points.copy()
    points_small[:, 0] *= sx
    points_small[:, 1] *= sy

    valid_count = int(valid.sum())
    inside_count = int((inside & valid).sum())
    outside_count = int((~inside & valid).sum())
    invalid_count = int((~valid).sum())
    inside_fraction = inside_count / valid_count if valid_count else float("nan")
    record.update(
        {
            "image_width_px": int(image.width),
            "image_height_px": int(image.height),
            "mask_width_px": int(mask.shape[1]),
            "mask_height_px": int(mask.shape[0]),
            "n_valid_local_points": valid_count,
            "n_inside_mask": inside_count,
            "n_outside_mask": outside_count,
            "n_invalid_local_points": invalid_count,
            "inside_mask_fraction": inside_fraction,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=130)
    ax_he, ax_inout, ax_mask, ax_info = axes.flat

    for ax in (ax_he, ax_inout):
        ax.imshow(image_small, origin="upper")
        _configure_image_axis(ax, image_small.shape[1], image_small.shape[0])
        _draw_mask_boundary(ax, mask_small)

    _draw_points(ax_he, points_small[valid], "#00e5ff", max_plot_points, size=0.8, alpha=0.43)
    ax_he.set_title("Native H&E crop + all valid ST centroids\ncyan = ST")

    _draw_points(
        ax_inout,
        points_small[inside & valid],
        "#00ff48",
        max_plot_points,
        size=0.9,
        alpha=0.48,
    )
    _draw_points(
        ax_inout,
        points_small[(~inside) & valid],
        "#ff2638",
        max_plot_points,
        size=0.9,
        alpha=0.48,
    )
    _draw_points(
        ax_inout,
        points_small[~valid],
        "#ff9d00",
        max_plot_points,
        size=1.4,
        alpha=0.7,
    )
    ax_inout.set_title("ST against H&E tissue mask\ngreen = in mask, red = outside, orange = invalid")

    mask_rgb = np.zeros((*mask_small.shape, 3), dtype=np.uint8)
    mask_rgb[mask_small] = np.array([40, 170, 80], dtype=np.uint8)
    ax_mask.imshow(mask_rgb, origin="upper")
    _configure_image_axis(ax_mask, mask_small.shape[1], mask_small.shape[0])
    _draw_points(ax_mask, points_small[inside & valid], "#00ffff", max_plot_points, size=0.8, alpha=0.45)
    _draw_points(ax_mask, points_small[(~inside) & valid], "#ff2035", max_plot_points, size=0.8, alpha=0.45)
    ax_mask.set_title("H&E mask frame\ncyan = in mask, red = outside mask")

    ax_info.axis("off")
    info = [
        f"sample / group: {sample_id} / {group_id}",
        f"section: {record['section_id']:03d}",
        f"ST cells: {record['n_st_cells']:,}",
        f"valid local points: {valid_count:,}",
        f"in H&E mask: {inside_count:,} ({inside_fraction:.3%})",
        f"outside H&E mask: {outside_count:,} ({outside_count / valid_count:.3%})"
        if valid_count
        else "outside H&E mask: NA",
        f"invalid/outside image: {invalid_count:,}",
        f"registration confidence: {_format_optional(record['registration_confidence'], 4)}",
        f"registration status: {record['registration_status'] or 'NA'}",
        f"forced flip: {record['forced_flip']}",
        f"preorientation: {record['preorientation_axis']}",
        "",
        "Coordinate frame used for overlay:",
        "x_he_local_px / y_he_local_px",
        "(native H&E crop pixels; no new registration fitted)",
        "",
        "Yellow line = exported H&E tissue mask boundary",
    ]
    ax_info.text(
        0.02,
        0.98,
        "\n".join(info),
        va="top",
        ha="left",
        fontsize=11,
        family="DejaVu Sans Mono",
        color="#202020",
    )
    ax_info.set_title("QC summary")

    fig.suptitle(
        f"{sample_id}/{group_id} section {record['section_id']:03d} | H&E–ST alignment",
        fontsize=16,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return record


def _render_overview(
    records: list[dict[str, Any]], output_path: Path, sample_id: str, group_id: str
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    if not records:
        return
    ncols = min(3, len(records))
    nrows = int(math.ceil(len(records) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.6 * ncols, 4.4 * nrows),
        squeeze=False,
        dpi=120,
    )
    axes_flat = axes.ravel()
    for ax, record in zip(axes_flat, records):
        path = Path(record["output_path"])
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.thumbnail((900, 700), Image.Resampling.LANCZOS)
            ax.imshow(image)
        ax.axis("off")
        ax.set_title(
            f"section {int(record['section_id']):03d} | "
            f"in mask={_format_optional(record['inside_mask_fraction'], 3)}",
            fontsize=10,
        )
    for ax in axes_flat[len(records) :]:
        ax.axis("off")
    fig.suptitle(f"{sample_id}/{group_id} | H&E–ST alignment overview", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def render_group(
    h5ad_path: Path,
    output_root: Path,
    v2_root: Path,
    *,
    max_dimension: int = 1600,
    max_plot_points: int = 80000,
) -> dict[str, Any]:
    """Render all ST-bearing sections from one normalized group H5AD."""

    import anndata as ad
    import pandas as pd

    stem = h5ad_path.stem
    if not re.fullmatch(r".+_g\d+", stem):
        raise ValueError(f"cannot infer sample/group from {h5ad_path.name}")
    sample_id, group_suffix = stem.rsplit("_g", 1)
    group_id = f"g{group_suffix}"
    output_dir = output_root / stem / "qc" / "he_st_alignment_v1"
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        needed = [
            "section_id",
            "raw_crop_path",
            "he_image_path",
            "he_path",
            "mask_path",
            "he_mask_path",
            "registration_transform_path",
            "registration_status",
            "registration_confidence",
            "forced_flip",
            "preorientation_axis",
            "source_coordinate_frame",
            "x_he_local_px",
            "y_he_local_px",
        ]
        missing = [column for column in needed if column not in adata.obs.columns]
        if missing:
            raise ValueError(f"{h5ad_path.name} missing required columns: {', '.join(missing)}")
        obs = adata.obs[needed].copy()
    finally:
        adata.file.close()

    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    section_ids = sorted(pd.to_numeric(obs["section_id"], errors="coerce").dropna().unique())
    for raw_section_id in section_ids:
        section_id = _safe_int(raw_section_id)
        section = obs.loc[obs["section_id"].astype(str) == str(raw_section_id)]
        if len(section) == 0:
            section = obs.loc[pd.to_numeric(obs["section_id"], errors="coerce") == raw_section_id]
        try:
            record, image_path, mask_path, points = _section_record(
                section_id, section, v2_root
            )
            if image_path is None:
                raise FileNotFoundError("no existing H&E crop path")
            if mask_path is None:
                raise FileNotFoundError("no existing H&E mask path")
            output_path = output_dir / f"section-{section_id:03d}__he_st_alignment.png"
            record["output_path"] = str(output_path)
            render_section(
                record=record,
                image_path=image_path,
                mask_path=mask_path,
                points=points,
                output_path=output_path,
                sample_id=sample_id,
                group_id=group_id,
                max_dimension=max_dimension,
                max_plot_points=max_plot_points,
            )
            records.append(record)
        except Exception as error:  # keep other sections renderable
            errors.append(
                {
                    "section_id": section_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )

    records_df = pd.DataFrame(records)
    records_df.to_csv(output_dir / "section_metrics.csv", index=False)
    _render_overview(
        records,
        output_dir / "overview_he_st_alignment.png",
        sample_id,
        group_id,
    )
    manifest = {
        "sample_id": sample_id,
        "group_id": group_id,
        "source_h5ad": str(h5ad_path),
        "coordinate_frame": "native H&E crop pixels",
        "coordinate_columns": ["x_he_local_px", "y_he_local_px"],
        "new_registration_fitted": False,
        "n_sections_expected": int(len(section_ids)),
        "n_sections_rendered": int(len(records)),
        "errors": errors,
        "sections": records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-dir", type=Path, default=DEFAULT_GROUP_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_REPO_ROOT / "data")
    parser.add_argument("--v2-root", type=Path, default=DEFAULT_V2_ROOT)
    parser.add_argument("--group", action="append", help="group H5AD stem, e.g. 00029_g0")
    parser.add_argument("--max-dimension", type=int, default=1600)
    parser.add_argument("--max-plot-points", type=int, default=80000)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.group:
        paths = [args.group_dir / f"{name}.h5ad" for name in args.group]
    else:
        paths = discover_group_h5ads(args.group_dir)
    if not paths:
        raise FileNotFoundError(f"no group H5AD files found under {args.group_dir}")
    summaries: list[dict[str, Any]] = []
    for path in paths:
        print(f"[render] {path.name}", flush=True)
        summary = render_group(
            path,
            args.output_root,
            args.v2_root,
            max_dimension=args.max_dimension,
            max_plot_points=args.max_plot_points,
        )
        summaries.append(
            {
                "sample_id": summary["sample_id"],
                "group_id": summary["group_id"],
                "n_sections_expected": summary["n_sections_expected"],
                "n_sections_rendered": summary["n_sections_rendered"],
                "errors": summary["errors"],
            }
        )
        print(
            f"  rendered {summary['n_sections_rendered']}/"
            f"{summary['n_sections_expected']} sections; errors={len(summary['errors'])}",
            flush=True,
        )
    summary_path = args.output_root / "he_st_alignment_qc_v1_summary.json"
    summary_path.write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[done] summary: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
