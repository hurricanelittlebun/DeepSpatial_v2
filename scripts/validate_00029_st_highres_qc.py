#!/usr/bin/env python3
"""Render high-resolution ST/H&E coordinate quality control for 00029/g0.

This is a diagnostic-only stage.  It reads the native SDPC level-3 crops and
does not re-register, rewrite, or replace any H&E/ST coordinates.

Two coordinate chains are deliberately shown separately:

* ``spatial_he_local`` -> native high-resolution companion H&E crop.  This
  tests whether the high-resolution image and the same-slice Xenium data use
  the same local physical frame.
* ``spatial_registered`` -> inverse of the saved STalign section transform ->
  native high-resolution companion H&E crop.  This tests the current
  registered-frame transform in the source image frame.

The distinction matters: a common registered XY frame is not itself the
pixel frame of an individual unwarped H&E crop.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import map_coordinates


ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = ROOT.parent / "DeepSpatial_v2"
V2_SRC = V2_ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from deepspatial.histology.coordinate_validation import (  # noqa: E402
    effective_level0_origin,
    load_candidate_coordinates,
    local_um_to_output_pixel_centers,
    output_to_source_pixel_centers,
)
from deepspatial.histology.sdpc import SdpcPyramid  # noqa: E402
from fit_00029_st_to_he_residual import (  # noqa: E402
    RegistrationMaps,
    read_section_transform,
    source_points_from_registered,
)


ANCHORS = (1, 11, 21, 31, 41, 51, 61, 71, 81, 91)
DEFAULT_DATASET = ROOT / "data" / "00029_g0"
DEFAULT_HE_TABLE = DEFAULT_DATASET / "he_sections.parquet"
DEFAULT_ANCHORS = DEFAULT_DATASET / "anchors"
DEFAULT_HIGH_ROOT = (
    DEFAULT_DATASET
    / "nucleus_path_raw_v1_coordinate_corrected_v2"
    / "segmentation"
)
DEFAULT_OUTPUT = DEFAULT_DATASET / "qc" / "st_highres_overlay_v1"
DEFAULT_REGISTRATION = V2_ROOT / "outputs" / "registration"
DEFAULT_SDPC = V2_ROOT / "data" / "0106548.sdpc"


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_h5ad_coordinates(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Read only coordinate arrays, avoiding a million-cell AnnData load."""

    with h5py.File(path, "r") as handle:
        local = np.asarray(handle["obsm"]["spatial_he_local"], dtype=np.float64)
        registered = np.asarray(handle["obsm"]["spatial_registered"], dtype=np.float64)
        stored_inside = None
        if "inside_he_mask" in handle["obs"]:
            value = handle["obs"]["inside_he_mask"]
            if isinstance(value, h5py.Dataset):
                stored_inside = np.asarray(value, dtype=bool)
    if local.ndim != 2 or local.shape[1] != 2:
        raise ValueError(f"{path}: spatial_he_local must have shape [N,2]")
    if registered.shape != local.shape:
        raise ValueError(f"{path}: registered/local coordinate shapes differ")
    if stored_inside is not None and stored_inside.shape != (len(local),):
        raise ValueError(f"{path}: inside_he_mask has an unexpected shape")
    return local, registered, stored_inside


def _rgb(value: np.ndarray) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"Expected an RGB image, got {image.shape}")
    return image[..., :3].astype(np.uint8, copy=False)


def _sample_mask(mask: np.ndarray, points_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample a boolean image at x/y points with nearest-neighbour semantics."""

    value = np.asarray(points_xy, dtype=np.float64)
    height, width = mask.shape
    valid = (
        np.isfinite(value).all(axis=1)
        & (value[:, 0] >= 0.0)
        & (value[:, 0] <= width - 1)
        & (value[:, 1] >= 0.0)
        & (value[:, 1] <= height - 1)
    )
    sampled = np.zeros(len(value), dtype=bool)
    if valid.any():
        sampled[valid] = map_coordinates(
            mask.astype(np.uint8),
            [value[valid, 1], value[valid, 0]],
            order=0,
            mode="constant",
            cval=0,
        ).astype(bool)
    return sampled, valid


def _high_mask_from_low(
    low_mask: np.ndarray,
    high_shape: tuple[int, int],
    *,
    low_origin_level0: tuple[int, int],
    low_downsample: float,
    high_origin_level0: tuple[int, int],
    high_downsample: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Map the reviewed low-resolution mask onto the native high crop."""

    low_xy = output_to_source_pixel_centers(
        high_shape,
        output_origin_level0=high_origin_level0,
        output_downsample=high_downsample,
        source_origin_level0=low_origin_level0,
        source_downsample=low_downsample,
    )
    high_mask, valid = _sample_mask(low_mask, low_xy.reshape(-1, 2))
    return high_mask.reshape(high_shape), valid.reshape(high_shape)


def _point_colors(inside: np.ndarray, valid: np.ndarray) -> np.ndarray:
    colors = np.empty((len(inside), 4), dtype=np.float32)
    colors[:] = [0.92, 0.10, 0.08, 0.45]  # red: outside crop or tissue mask
    colors[valid & inside] = [0.05, 0.85, 0.25, 0.55]  # green: inside mask
    return colors


def _plot_points(axis, points_xy, inside, valid, *, size=1.2, alpha=None):
    colors = _point_colors(inside, valid)
    if alpha is not None:
        colors[:, 3] = float(alpha)
    axis.scatter(
        points_xy[:, 0],
        points_xy[:, 1],
        s=size,
        c=colors,
        linewidths=0,
        rasterized=True,
    )


def _show_mask(axis, image: np.ndarray, mask: np.ndarray) -> None:
    axis.imshow(image)
    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    overlay[..., 1] = 1.0
    overlay[..., 3] = mask.astype(np.float32) * 0.22
    axis.imshow(overlay)


def _format_axes(axis, image_shape: tuple[int, int]) -> None:
    axis.set_xlim(-0.5, image_shape[1] - 0.5)
    axis.set_ylim(image_shape[0] - 0.5, -0.5)
    axis.set_axis_off()


def _save_section_figure(
    path: Path,
    *,
    section_id: int,
    image: np.ndarray,
    high_mask: np.ndarray,
    local_px: np.ndarray,
    local_inside: np.ndarray,
    local_valid: np.ndarray,
    comparison_px: np.ndarray,
    comparison_inside: np.ndarray,
    comparison_valid: np.ndarray,
    comparison_label: str,
    metrics: dict,
) -> None:
    shape = image.shape[:2]
    fig, axes = plt.subplots(3, 2, figsize=(15, 20), constrained_layout=True)

    axes[0, 0].imshow(image)
    _plot_points(axes[0, 0], local_px, local_inside, local_valid)
    axes[0, 0].set_title(
        "Same-slice local ST → high-res H&E\n"
        f"green=in tissue {metrics['local_inside_tissue_given_valid_fraction']:.3f}; "
        f"red=out {metrics['local_outside_tissue_fraction']:.3f}"
    )

    axes[0, 1].imshow(image)
    _plot_points(axes[0, 1], comparison_px, comparison_inside, comparison_valid)
    axes[0, 1].set_title(
        f"{comparison_label} → inverse saved STalign → high-res H&E\n"
        f"green=in tissue {metrics['comparison_inside_tissue_given_valid_fraction']:.3f}; "
        f"red=out {metrics['comparison_outside_tissue_fraction']:.3f}"
    )

    _show_mask(axes[1, 0], image, high_mask)
    _plot_points(axes[1, 0], local_px, local_inside, local_valid, size=1.0)
    axes[1, 0].set_title("H&E foreground mask + local ST coordinates")

    _show_mask(axes[1, 1], image, high_mask)
    _plot_points(axes[1, 1], comparison_px, comparison_inside, comparison_valid, size=1.0)
    axes[1, 1].set_title(f"H&E foreground mask + inverse-mapped {comparison_label}")

    # A direct comparison makes a registration-frame displacement visible.
    axes[2, 0].imshow(image)
    stride = max(1, int(np.ceil(len(local_px) / 6000)))
    local_subset = local_px[::stride]
    comparison_subset = comparison_px[::stride]
    axes[2, 0].scatter(
        local_subset[:, 0], local_subset[:, 1], s=1.2, c="#00d7ff", alpha=0.55,
        linewidths=0, label="spatial_he_local",
    )
    axes[2, 0].scatter(
        comparison_subset[:, 0], comparison_subset[:, 1], s=1.2, c="#ff00d0", alpha=0.45,
        linewidths=0, label=f"inverse {comparison_label}",
    )
    axes[2, 0].legend(loc="upper right", fontsize=8, framealpha=0.8)
    axes[2, 0].set_title("Cyan/local vs magenta/inverse-registered")

    axes[2, 1].imshow(image)
    valid_vector = local_valid & comparison_valid
    vector_indices = np.flatnonzero(valid_vector)
    if len(vector_indices) > 500:
        vector_indices = vector_indices[np.linspace(0, len(vector_indices) - 1, 500, dtype=int)]
    if len(vector_indices):
        start = local_px[vector_indices]
        delta = comparison_px[vector_indices] - start
        axes[2, 1].quiver(
            start[:, 0], start[:, 1], delta[:, 0], delta[:, 1],
            color="#ff8c00", angles="xy", scale_units="xy", scale=1,
            width=0.0015, alpha=0.65,
        )
    axes[2, 1].set_title(
        f"Orange arrows: local → inverse-{comparison_label} displacement\n"
        f"median={metrics['local_to_comparison_displacement_um_median']:.1f} µm; "
        f"p95={metrics['local_to_comparison_displacement_um_p95']:.1f} µm"
    )

    for axis in axes.flat:
        _format_axes(axis, shape)
    fig.suptitle(
        f"00029/g0 section {section_id:03d} | high-res ST/H&E QC | "
        f"registration={metrics['registration_status']} "
        f"confidence={metrics['registration_confidence']:.4f}",
        fontsize=14,
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_overview(
    path: Path,
    records: list[dict],
    *,
    coordinate_key: str,
    title: str,
) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(20, 9), constrained_layout=True)
    for axis, record in zip(axes.flat, records):
        axis.imshow(record["thumbnail"])
        points = record[coordinate_key]
        scale_x = record["thumbnail"].shape[1] / record["image_shape"][1]
        scale_y = record["thumbnail"].shape[0] / record["image_shape"][0]
        is_local = coordinate_key == "local_px"
        inside = record["local_inside"] if is_local else record["comparison_inside"]
        valid = record["local_valid"] if is_local else record["comparison_valid"]
        inside_fraction = (
            record["local_inside_fraction"]
            if is_local
            else record["comparison_inside_fraction"]
        )
        valid_fraction = (
            record["local_valid_fraction"]
            if is_local
            else record["comparison_valid_fraction"]
        )
        colors = _point_colors(inside, valid)
        axis.scatter(
            points[:, 0] * scale_x,
            points[:, 1] * scale_y,
            s=0.45,
            c=colors,
            linewidths=0,
            rasterized=True,
        )
        axis.set_title(
            f"s{record['section_id']:03d}\n"
            f"in={inside_fraction:.3f}, "
            f"valid={valid_fraction:.3f}",
            fontsize=9,
        )
        axis.set_axis_off()
    fig.suptitle(
        f"00029/g0 high-resolution ST/H&E QC | {title}",
        fontsize=14,
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_coverage(path: Path, rows: list[dict], *, comparison_label: str) -> None:
    frame = pd.DataFrame(rows).sort_values("section_id")
    x = np.arange(len(frame))
    width = 0.36
    fig, axis = plt.subplots(figsize=(14, 6), constrained_layout=True)
    axis.bar(
        x - width / 2,
        frame["local_inside_tissue_given_valid_fraction"],
        width,
        label="same-slice local ST",
        color="#11aa44",
    )
    axis.bar(
        x + width / 2,
        frame["comparison_inside_tissue_given_valid_fraction"],
        width,
        label=comparison_label,
        color="#d62f2f",
    )
    axis.set_ylim(0, 1.0)
    axis.set_xticks(x, [f"{int(value):03d}" for value in frame["section_id"]])
    axis.set_xlabel("anchor section")
    axis.set_ylabel("fraction inside high-res H&E foreground mask")
    axis.set_title("High-resolution ST/H&E foreground coverage comparison")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def run(args: argparse.Namespace) -> dict:
    dataset = Path(args.dataset_dir).resolve()
    he_table_path = Path(args.he_table).resolve()
    anchors_dir = Path(args.anchors_dir).resolve()
    high_root = Path(args.high_root).resolve()
    registration_root = Path(args.registration_root).resolve()
    sdpc_path = Path(args.sdpc_path).resolve()
    correction_dir = Path(args.correction_dir).resolve() if args.correction_dir else None
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing QC output: {output}")
    if not he_table_path.is_file():
        raise FileNotFoundError(he_table_path)
    if not sdpc_path.is_file():
        raise FileNotFoundError(sdpc_path)
    if correction_dir is not None and not correction_dir.is_dir():
        raise NotADirectoryError(correction_dir)
    output.mkdir(parents=True, exist_ok=False)
    sections = pd.read_parquet(he_table_path)
    sections["section_id"] = pd.to_numeric(sections["section_id"], errors="raise").astype(int)
    section_rows = sections.set_index("section_id", drop=False)
    maps = RegistrationMaps(registration_root, "00029", "g0")
    use_candidate = correction_dir is not None
    comparison_coordinate_key = "spatial_st_corrected" if use_candidate else "spatial_registered"
    comparison_label = "candidate corrected ST" if use_candidate else "registered ST"
    rows: list[dict] = []
    overview_records: list[dict] = []

    with SdpcPyramid(sdpc_path) as reader:
        raw_mpp = float(reader.mpp_level0)
        for section_id in ANCHORS:
            if section_id not in section_rows.index:
                raise KeyError(f"Missing H&E manifest row for section {section_id}")
            row = section_rows.loc[section_id]
            anchor_path = anchors_dir / f"section-{section_id:03d}.h5ad"
            high_meta_path = high_root / f"section-{section_id:03d}" / "raw_crop_metadata.json"
            if not anchor_path.is_file():
                raise FileNotFoundError(anchor_path)
            if not high_meta_path.is_file():
                raise FileNotFoundError(high_meta_path)

            local_um, original_registered_um, stored_inside = _read_h5ad_coordinates(anchor_path)
            if use_candidate:
                candidate_path = correction_dir / "candidate_coordinates" / (
                    f"section-{section_id:03d}__st_corrected_candidate.npz"
                )
                if not candidate_path.is_file():
                    raise FileNotFoundError(candidate_path)
                registered_um = load_candidate_coordinates(
                    candidate_path,
                    expected_n=len(original_registered_um),
                )
            else:
                registered_um = original_registered_um
            high_meta = _read_json(high_meta_path)
            source_image_path = Path(str(row["source_image_path"]))
            low_meta = _read_json(source_image_path.parent / "metadata.json")
            constrained_low_mask = np.asarray(
                Image.open(Path(str(row["source_mask_path"]))).convert("L")
            ) > 0
            foreground_mask_path = (
                source_image_path.parent / f"he_foreground_mask_level{int(low_meta['crop_level'])}.png"
            )
            if not foreground_mask_path.is_file():
                raise FileNotFoundError(f"Missing H&E foreground mask: {foreground_mask_path}")
            low_mask = np.asarray(Image.open(foreground_mask_path).convert("L")) > 0
            low_level = int(low_meta["crop_level"])
            low_downsample = float(reader.level_downsamples[low_level])
            high_level = int(high_meta["selected_level"])
            high_downsample = float(high_meta["level_downsample"])
            high_origin = tuple(int(value) for value in high_meta["read_location_level0"])
            high_size = tuple(int(value) for value in high_meta["read_size_level_px"])
            image = _rgb(reader.read_region(high_origin, high_level, high_size))
            if tuple(image.shape[:2]) != tuple(high_meta["highres_image_shape"]):
                raise ValueError(
                    f"section {section_id}: high-res image shape {image.shape[:2]} "
                    f"does not match metadata {high_meta['highres_image_shape']}"
                )
            requested_bbox = tuple(float(value) for value in low_meta["crop_level0_bbox"])
            low_origin = effective_level0_origin(requested_bbox[:2], low_downsample)
            high_mask, high_mask_valid = _high_mask_from_low(
                low_mask,
                image.shape[:2],
                low_origin_level0=low_origin,
                low_downsample=low_downsample,
                high_origin_level0=high_origin,
                high_downsample=high_downsample,
            )
            high_constrained_mask, constrained_mask_valid = _high_mask_from_low(
                constrained_low_mask,
                image.shape[:2],
                low_origin_level0=low_origin,
                low_downsample=low_downsample,
                high_origin_level0=high_origin,
                high_downsample=high_downsample,
            )

            local_px = local_um_to_output_pixel_centers(
                local_um,
                local_origin_level0=requested_bbox[:2],
                output_origin_level0=high_origin,
                raw_mpp_um=raw_mpp,
                output_downsample=high_downsample,
            )
            preorientation, chain = read_section_transform(Path(str(row["section_transform_path"])))
            inverse_source_um = source_points_from_registered(
                registered_um, preorientation, chain, maps
            )
            registered_px = local_um_to_output_pixel_centers(
                inverse_source_um,
                local_origin_level0=requested_bbox[:2],
                output_origin_level0=high_origin,
                raw_mpp_um=raw_mpp,
                output_downsample=high_downsample,
            )
            local_inside, local_valid = _sample_mask(high_mask, local_px)
            registered_inside, registered_valid = _sample_mask(high_mask, registered_px)
            local_inside_constrained, local_valid_constrained = _sample_mask(
                high_constrained_mask, local_px
            )
            registered_inside_constrained, registered_valid_constrained = _sample_mask(
                high_constrained_mask, registered_px
            )

            vector_distance_um = np.linalg.norm(inverse_source_um - local_um, axis=1)
            n = len(local_um)
            metrics = {
                "section_id": int(section_id),
                "z_um": float(row["z_um"]),
                "n_st_cells": int(n),
                "highres_image_shape_hw": [int(value) for value in image.shape[:2]],
                "highres_level": high_level,
                "highres_downsample_level0_px": high_downsample,
                "highres_pixel_size_um": float(high_meta["selected_level_mpp_um"]),
                "raw_mpp_um": raw_mpp,
                "requested_bbox_level0": [float(value) for value in requested_bbox],
                "highres_effective_origin_level0": [int(value) for value in high_origin],
                "lowres_effective_origin_level0": [int(value) for value in low_origin],
                "foreground_mask_highres_valid_fraction": float(high_mask_valid.mean()),
                "constrained_mask_highres_valid_fraction": float(constrained_mask_valid.mean()),
                "local_valid_fraction": float(local_valid.mean()),
                "local_inside_tissue_fraction": float((local_valid & local_inside).mean()),
                "local_inside_tissue_given_valid_fraction": float(
                    (local_inside[local_valid].mean()) if local_valid.any() else np.nan
                ),
                "local_outside_tissue_fraction": float(
                    (local_valid & ~local_inside).mean()
                    + (~local_valid).mean()
                ),
                "local_inside_constrained_fraction": float(
                    (local_valid_constrained & local_inside_constrained).mean()
                ),
                "local_inside_constrained_given_valid_fraction": float(
                    (local_inside_constrained[local_valid_constrained].mean())
                    if local_valid_constrained.any()
                    else np.nan
                ),
                "comparison_valid_fraction": float(registered_valid.mean()),
                "comparison_inside_tissue_fraction": float(
                    (registered_valid & registered_inside).mean()
                ),
                "comparison_inside_tissue_given_valid_fraction": float(
                    (registered_inside[registered_valid].mean())
                    if registered_valid.any()
                    else np.nan
                ),
                "comparison_outside_tissue_fraction": float(
                    (registered_valid & ~registered_inside).mean()
                    + (~registered_valid).mean()
                ),
                "comparison_inside_constrained_fraction": float(
                    (registered_valid_constrained & registered_inside_constrained).mean()
                ),
                "comparison_inside_constrained_given_valid_fraction": float(
                    (registered_inside_constrained[registered_valid_constrained].mean())
                    if registered_valid_constrained.any()
                    else np.nan
                ),
                "local_constrained_vs_stored_inside_mask_agreement": (
                    float((local_inside_constrained == stored_inside).mean())
                    if stored_inside is not None
                    else None
                ),
                "local_to_comparison_displacement_um_mean": float(vector_distance_um.mean()),
                "local_to_comparison_displacement_um_median": float(np.median(vector_distance_um)),
                "local_to_comparison_displacement_um_p95": float(np.percentile(vector_distance_um, 95)),
                "registration_status": str(row["registration_status"]),
                "registration_confidence": float(row["registration_confidence"]),
                "registration_transform_path": str(row["section_transform_path"]),
                "comparison_coordinate_key": comparison_coordinate_key,
                "comparison_source": (
                    str(correction_dir / "candidate_coordinates")
                    if use_candidate
                    else str(anchor_path)
                ),
                "coordinate_interpretation": {
                    "local": "spatial_he_local (companion-H&E crop micrometres)",
                    "comparison": (
                        f"{comparison_coordinate_key} inverse-mapped through complete saved STalign chain"
                    ),
                    "highres": "native SDPC level-3 crop; no image warp",
                },
            }
            section_output = output / f"section-{section_id:03d}"
            section_output.mkdir(parents=True, exist_ok=False)
            figure_path = section_output / "st_highres_overlay_qc.png"
            _save_section_figure(
                figure_path,
                section_id=section_id,
                image=image,
                high_mask=high_mask,
                local_px=local_px,
                local_inside=local_inside,
                local_valid=local_valid,
                comparison_px=registered_px,
                comparison_inside=registered_inside,
                comparison_valid=registered_valid,
                comparison_label=comparison_label,
                metrics=metrics,
            )
            point_table = pd.DataFrame(
                {
                    "row_index": np.arange(n, dtype=np.int64),
                    "local_x_um": local_um[:, 0],
                    "local_y_um": local_um[:, 1],
                    "local_x_highres_px": local_px[:, 0],
                    "local_y_highres_px": local_px[:, 1],
                    "local_valid": local_valid,
                    "local_inside_mask": local_inside,
                    "local_inside_tissue": local_inside,
                    "local_inside_constrained_mask": local_inside_constrained,
                    "comparison_source_x_um": inverse_source_um[:, 0],
                    "comparison_source_y_um": inverse_source_um[:, 1],
                    "comparison_x_highres_px": registered_px[:, 0],
                    "comparison_y_highres_px": registered_px[:, 1],
                    "comparison_valid": registered_valid,
                    "comparison_inside_mask": registered_inside,
                    "comparison_inside_tissue": registered_inside,
                    "comparison_inside_constrained_mask": registered_inside_constrained,
                    "local_to_comparison_distance_um": vector_distance_um,
                }
            )
            point_table.to_parquet(section_output / "st_highres_coordinates.parquet", index=False)
            metrics["overlay_path"] = str(figure_path)
            metrics["point_table_path"] = str(section_output / "st_highres_coordinates.parquet")
            rows.append(metrics)

            thumbnail = np.asarray(
                Image.fromarray(image).resize((min(480, image.shape[1]), int(round(image.shape[0] * min(480, image.shape[1]) / image.shape[1]))), Image.Resampling.BILINEAR)
            )
            overview_records.append(
                {
                    "section_id": int(section_id),
                    "thumbnail": thumbnail,
                    "image_shape": list(image.shape[:2]),
                    "local_px": local_px,
                    "registered_px": registered_px,
                    "local_inside": local_inside,
                    "comparison_inside": registered_inside,
                    "local_valid": local_valid,
                    "comparison_valid": registered_valid,
                    "local_inside_fraction": metrics["local_inside_tissue_given_valid_fraction"],
                    "comparison_inside_fraction": metrics["comparison_inside_tissue_given_valid_fraction"],
                    "local_valid_fraction": metrics["local_valid_fraction"],
                    "comparison_valid_fraction": metrics["comparison_valid_fraction"],
                    "inside": local_inside,
                    "valid": local_valid,
                }
            )

    section_metrics = pd.DataFrame(rows).sort_values("section_id")
    section_metrics.to_csv(output / "section_metrics.csv", index=False)
    _save_overview(
        output / "overview_local_st.png",
        overview_records,
        coordinate_key="local_px",
        title="same-slice local coordinates",
    )
    _save_overview(
        output / "overview_comparison_st.png",
        overview_records,
        coordinate_key="registered_px",
        title=f"{comparison_label} inverse-mapped to native H&E",
    )
    _save_coverage(
        output / "mask_coverage_comparison.png",
        rows,
        comparison_label=comparison_label,
    )

    report = {
        "format": "deepspatial-00029-st-highres-qc-v1",
        "status": "completed",
        "dataset": str(dataset),
        "coordinate_frame": "00029__g0_registered",
        "source_sdpc": str(sdpc_path),
        "anchor_sections": list(ANCHORS),
        "n_sections": len(rows),
        "total_st_cells": int(sum(row["n_st_cells"] for row in rows)),
        "local_coordinate_chain": (
            "H5AD obsm/spatial_he_local (um) -> requested SDPC bbox -> native level-3 pixels"
        ),
        "comparison_coordinate_chain": (
            f"{comparison_coordinate_key} (registered um) -> complete saved STalign inverse "
            "-> companion local um -> native level-3 pixels"
        ),
        "comparison_coordinate_key": comparison_coordinate_key,
        "comparison_source": str(correction_dir) if use_candidate else "anchor H5AD",
        "mask_chain": (
            "level-4 H&E foreground mask -> physical pixel-center mapping -> native level-3 grid; "
            "the Xenium-constrained mask is reported separately"
        ),
        "no_registration_refit": True,
        "no_input_overwrite": True,
        "sections": rows,
    }
    with (output / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=_json_safe)
        handle.write("\n")
    (output / "README.md").write_text(
        "# 00029/g0 high-resolution ST/H&E QC\n\n"
        "This directory is diagnostic-only. It does not re-register or overwrite "
        "the H&E, H5AD, STalign, or feature-store inputs.\n\n"
        "- `overview_local_st.png`: same-slice `spatial_he_local` over native SDPC level-3 H&E.\n"
        f"- `overview_comparison_st.png`: `{comparison_coordinate_key}` inverse-mapped through the complete saved STalign chain.\n"
        "- `section-*/st_highres_overlay_qc.png`: six-panel per-anchor diagnostic.\n"
        "- `section-*/st_highres_coordinates.parquet`: point-level pixel mapping and mask status.\n"
        "- `section_metrics.csv`: numeric coverage and displacement summary.\n\n"
        "Interpretation: local overlay tests high-res/low-res/native companion coordinate "
        "consistency. The comparison overlay additionally tests the selected "
        "serial-registration transform; it is not expected to coincide with an unwarped "
        "individual crop if the section has a non-identity STalign map.\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--he-table", type=Path, default=DEFAULT_HE_TABLE)
    parser.add_argument("--anchors-dir", type=Path, default=DEFAULT_ANCHORS)
    parser.add_argument("--high-root", type=Path, default=DEFAULT_HIGH_ROOT)
    parser.add_argument("--registration-root", type=Path, default=DEFAULT_REGISTRATION)
    parser.add_argument("--sdpc-path", type=Path, default=DEFAULT_SDPC)
    parser.add_argument(
        "--correction-dir",
        type=Path,
        default=None,
        help=(
            "Use per-section spatial_st_corrected NPZ coordinates from this "
            "registration_correction_v1 directory."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "n_sections": report["n_sections"],
                "total_st_cells": report["total_st_cells"],
                "output_dir": str(Path(args.output_dir).resolve()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
