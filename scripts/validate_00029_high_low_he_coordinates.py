"""Validate high-resolution SDPC crops against their level-4 H&E crops.

The validation keeps the level-4 H&E crop fixed and maps level-3 SDPC pixels
onto its pixel-center grid using the raw level-0 origins and pyramid scales.
It does not estimate a registration transform and does not alter any existing
H&E, Xenium, or feature-store files.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage import color, measure, morphology
from skimage.metrics import structural_similarity
from skimage.registration import phase_cross_correlation

_V2_SRC = Path(__file__).resolve().parents[2] / "DeepSpatial_v2" / "src"
if _V2_SRC.is_dir() and str(_V2_SRC) not in sys.path:
    sys.path.insert(0, str(_V2_SRC))
try:
    from deepspatial_v2.he_segmentation.foreground import (
        ForegroundConfig,
        detect_tissue_foreground,
    )
except Exception:  # pragma: no cover - fallback for standalone environments
    ForegroundConfig = None
    detect_tissue_foreground = None

from deepspatial.histology.coordinate_validation import (
    effective_level0_origin,
    output_to_source_pixel_centers,
)
from deepspatial.histology.sdpc import SdpcPyramid


ANCHORS = (1, 11, 21, 31, 41, 51, 61, 71, 81, 91)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _as_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _mask_iou(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    union = np.logical_or(a, b).sum()
    intersection = np.logical_and(a, b).sum()
    iou = float(intersection / union) if union else float("nan")
    denominator = a.sum() + b.sum()
    dice = float(2 * intersection / denominator) if denominator else float("nan")
    return iou, dice


def _foreground_from_rgb(rgb: np.ndarray) -> np.ndarray:
    """Re-run the reviewed foreground detector on the mapped high-res image.

    The implementation is kept local so this validation can run in the
    DeepSpatial environment even when OpenCV is only installed in the v2
    preprocessing environment.
    """

    rgb_float = np.clip(rgb.astype(np.float32), 0.0, 255.0)
    hsv = color.rgb2hsv(rgb_float / 255.0)
    saturation = hsv[..., 1]
    value = hsv[..., 2]
    dark = value < 0.14
    labels = measure.label(dark, connectivity=2)
    artifact = np.zeros_like(dark)
    height, width = dark.shape
    for region in measure.regionprops(labels):
        min_row, min_col, max_row, max_col = region.bbox
        touches_border = (
            min_row <= 2
            or min_col <= 2
            or max_row >= height - 2
            or max_col >= width - 2
        )
        if touches_border or region.area <= 100:
            artifact[labels == region.label] = True
    neutral = saturation < 0.04
    if int(neutral.sum()) < 256:
        neutral = np.ones(saturation.shape, dtype=bool)
    background_rgb = np.percentile(rgb_float[neutral], 90, axis=0)
    relative_od_rgb = np.log((background_rgb[None, None, :] + 1.0) / (rgb_float + 1.0))
    optical_density = np.clip(relative_od_rgb, 0.0, None).mean(axis=2)
    lab = color.rgb2lab(rgb_float / 255.0)
    background_ab = np.median(lab[neutral, 1:3], axis=0)
    chroma = np.sqrt(np.sum((lab[..., 1:3] - background_ab) ** 2, axis=2))
    stained = (optical_density >= 0.055) & (
        (saturation >= 0.10) | (chroma >= 6.0)
    )
    candidate = stained & ~artifact
    candidate = morphology.binary_closing(candidate, morphology.disk(2))
    candidate = morphology.remove_small_holes(candidate, area_threshold=100, connectivity=2)
    return morphology.remove_small_objects(candidate, min_size=200, connectivity=2)


def _normalize_for_plot(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = np.zeros(values.shape, dtype=np.float32)
    selected = values[valid]
    if selected.size == 0:
        return result
    lo, hi = np.percentile(selected, [1.0, 99.0])
    if hi <= lo:
        return result
    result[valid] = np.clip((values[valid] - lo) / (hi - lo), 0.0, 1.0)
    return result


def _gray(rgb: np.ndarray) -> np.ndarray:
    return np.dot(rgb[..., :3].astype(np.float32), [0.299, 0.587, 0.114])


def _sample_high_to_low(
    high: np.ndarray,
    *,
    low_shape: Tuple[int, int],
    low_origin_level0: Tuple[int, int],
    low_downsample: float,
    high_origin_level0: Tuple[int, int],
    high_downsample: float,
) -> Tuple[np.ndarray, np.ndarray]:
    source_xy = output_to_source_pixel_centers(
        low_shape,
        output_origin_level0=low_origin_level0,
        output_downsample=low_downsample,
        source_origin_level0=high_origin_level0,
        source_downsample=high_downsample,
    )
    source_x = source_xy[..., 0]
    source_y = source_xy[..., 1]
    height, width = high.shape[:2]
    valid = (
        (source_x >= 0.0)
        & (source_x <= width - 1)
        & (source_y >= 0.0)
        & (source_y <= height - 1)
    )
    sampled = np.full((*low_shape, 3), np.nan, dtype=np.float32)
    for channel in range(3):
        sampled[..., channel] = map_coordinates(
            high[..., channel].astype(np.float32),
            [source_y, source_x],
            order=1,
            mode="constant",
            cval=np.nan,
        )
    sampled[~valid] = np.nan
    return sampled, valid


def _phase_shift(low_gray: np.ndarray, high_gray: np.ndarray, valid: np.ndarray):
    if valid.sum() < 100:
        return [float("nan"), float("nan")], float("nan")
    # Suppress slow staining/background differences and retain structural
    # boundaries.  The returned shift is (row, column), in low-res pixels.
    a = low_gray.astype(np.float32)
    b = high_gray.astype(np.float32)
    a = gaussian_filter(a, 1.0) - gaussian_filter(a, 8.0)
    b = gaussian_filter(b, 1.0) - gaussian_filter(b, 8.0)
    a[~valid] = 0.0
    b[~valid] = 0.0
    shift, error, _ = phase_cross_correlation(
        a, b, upsample_factor=10, normalization=None
    )
    return [float(shift[0]), float(shift[1])], float(error)


def _red_green(low_gray: np.ndarray, high_gray: np.ndarray, valid: np.ndarray):
    low_norm = _normalize_for_plot(low_gray, valid)
    high_norm = _normalize_for_plot(high_gray, valid)
    image = np.zeros((*low_gray.shape, 3), dtype=np.float32)
    image[..., 0] = low_norm
    image[..., 1] = high_norm
    return image


def _save_overlay(
    output_path: Path,
    *,
    section_id: int,
    low: np.ndarray,
    high_mapped: np.ndarray,
    low_mask: np.ndarray,
    high_mask: np.ndarray,
    valid: np.ndarray,
    metrics: dict,
) -> None:
    high_display = np.nan_to_num(high_mapped, nan=255.0).clip(0, 255).astype(np.uint8)
    low_gray = _gray(low)
    high_gray = _gray(high_display)
    comparison_mask = valid & low_mask
    rgb_overlay = _red_green(low_gray, high_gray, comparison_mask)
    difference = np.abs(low_gray - high_gray)
    difference[~valid] = np.nan

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes[0, 0].imshow(low)
    axes[0, 0].set_title("Low-resolution H&E level 4")
    axes[0, 1].imshow(high_display)
    axes[0, 1].set_title("High-resolution level 3 mapped to level-4 grid")
    axes[1, 0].imshow(rgb_overlay)
    axes[1, 0].set_title("Red=low H&E, green=high H&E; yellow=overlap")
    axes[1, 1].imshow(difference, cmap="magma", vmin=0, vmax=60)
    axes[1, 1].contour(low_mask, levels=[0.5], colors="cyan", linewidths=0.4)
    axes[1, 1].contour(high_mask, levels=[0.5], colors="lime", linewidths=0.4)
    axes[1, 1].set_title("Absolute grayscale difference + masks")
    for axis in axes.flat:
        axis.set_axis_off()
    fig.suptitle(
        f"00029/g0 section {section_id:03d} | "
        f"corr={metrics['gray_pearson_tissue']:.4f}, "
        f"shift=({metrics['shift_y_px']:.2f},{metrics['shift_x_px']:.2f}) px, "
        f"mask IoU={metrics['foreground_mask_iou']:.4f}",
        fontsize=13,
    )
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_overview(output_path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    fig, axes = plt.subplots(2, 5, figsize=(18, 8), constrained_layout=True)
    for axis, row in zip(axes.flat, rows):
        image = np.asarray(Image.open(row["overlay_path"]).convert("RGB"))
        # The red/green panel is the lower-left quadrant of the 2x2 figure.
        height, width = image.shape[:2]
        panel = image[height // 2 :, : width // 2]
        axis.imshow(panel)
        axis.set_title(
            f"s{int(row['section_id']):03d}\n"
            f"r={row['gray_pearson_tissue']:.3f}, "
            f"shift={row['shift_norm_px']:.2f}px"
        )
        axis.set_axis_off()
    fig.suptitle("00029/g0 high-resolution vs low-resolution H&E coordinate check")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _process_section(row: pd.Series, high_root: Path, output_dir: Path, reader) -> dict:
    section_id = int(row["section_id"])
    masked_low_image_path = Path(row["source_image_path"])
    low_image_path = masked_low_image_path.parent / "he_level4.tif"
    if not low_image_path.is_file():
        low_image_path = masked_low_image_path
    low_meta = _read_json(masked_low_image_path.parent / "metadata.json")
    high_meta_path = high_root / f"section-{section_id:03d}" / "raw_crop_metadata.json"
    high_meta = _read_json(high_meta_path)

    low = _as_rgb(low_image_path)
    high_origin = tuple(int(value) for value in high_meta["read_location_level0"])
    high_size = tuple(int(value) for value in high_meta["read_size_level_px"])
    high_level = int(high_meta["selected_level"])
    high = reader.read_region(high_origin, high_level, high_size)

    low_level = int(low_meta["crop_level"])
    low_downsample = float(reader.level_downsamples[low_level])
    high_downsample = float(reader.level_downsamples[high_level])
    low_pixel_size_um = float(low_meta["ruler_raw_um_per_level0_px"]) * low_downsample
    requested_bbox = tuple(float(value) for value in low_meta["crop_level0_bbox"])
    low_origin = effective_level0_origin(requested_bbox[:2], low_downsample)
    expected_low_size = tuple(int(value) for value in low_meta["crop_level_dimensions"])
    if low.shape[:2][::-1] != expected_low_size:
        raise ValueError(
            f"section {section_id}: low image shape {low.shape[:2][::-1]} "
            f"does not match metadata {expected_low_size}"
        )
    low_direct = reader.read_region(low_origin, low_level, expected_low_size)
    if low_direct.shape != low.shape:
        raise ValueError(
            f"section {section_id}: direct low-res read {low_direct.shape} "
            f"does not match stored image {low.shape}"
        )

    high_mapped, valid = _sample_high_to_low(
        high,
        low_shape=low.shape[:2],
        low_origin_level0=low_origin,
        low_downsample=low_downsample,
        high_origin_level0=high_origin,
        high_downsample=high_downsample,
    )
    high_display = np.nan_to_num(high_mapped, nan=255.0).clip(0, 255).astype(np.uint8)
    low_mask_path = low_image_path.parent / f"he_foreground_mask_level{low_level}.png"
    low_mask = np.asarray(Image.open(low_mask_path)) > 0
    if low_mask.shape != low.shape[:2]:
        raise ValueError(f"section {section_id}: foreground mask shape mismatch")
    high_mask = _foreground_from_rgb(high_display)
    compare_mask = valid & low_mask
    low_gray = _gray(low)
    high_gray = _gray(high_display)
    tissue_low = low_gray[compare_mask]
    tissue_high = high_gray[compare_mask]
    shift, phase_error = _phase_shift(low_gray, high_gray, compare_mask)
    iou, dice = _mask_iou(low_mask & valid, high_mask & valid)
    output_section = output_dir / f"section-{section_id:03d}"
    output_section.mkdir(parents=True, exist_ok=True)
    mapped_path = output_section / "highres_mapped_to_low_grid.tif"
    Image.fromarray(high_display).save(mapped_path, compression="tiff_adobe_deflate")
    overlay_path = output_section / "high_low_coordinate_overlay.png"

    metrics = {
        "section_id": section_id,
        "z_um": float(row["z_um"]),
        "raw_source_same": str(low_meta["source_sdpc"]) == str(high_meta["raw_source_path"]),
        "requested_bbox_same": list(map(float, low_meta["crop_level0_bbox"]))
        == list(map(float, high_meta["requested_bbox_level0"])),
        "low_level": low_level,
        "high_level": high_level,
        "low_downsample_level0_px": low_downsample,
        "high_downsample_level0_px": high_downsample,
        "low_pixel_size_um": low_pixel_size_um,
        "high_pixel_size_um": float(high_meta["selected_level_mpp_um"]),
        "downsample_ratio_low_over_high": low_downsample / high_downsample,
        "low_shape_hw": list(map(int, low.shape[:2])),
        "high_shape_hw": list(map(int, high.shape[:2])),
        "low_requested_origin_level0": list(map(float, requested_bbox[:2])),
        "low_effective_origin_level0": list(map(int, low_origin)),
        "high_effective_origin_level0": list(map(int, high_origin)),
        "high_to_low_grid_valid_fraction": float(valid.mean()),
        "low_direct_read_max_abs_diff": int(np.abs(low.astype(np.int16) - low_direct.astype(np.int16)).max()),
        "low_direct_read_mae": float(np.abs(low.astype(np.float32) - low_direct.astype(np.float32)).mean()),
        "gray_pearson_tissue": _pearson(tissue_low, tissue_high),
        "rgb_pearson_r_tissue": _pearson(low[..., 0][compare_mask], high_display[..., 0][compare_mask]),
        "rgb_pearson_g_tissue": _pearson(low[..., 1][compare_mask], high_display[..., 1][compare_mask]),
        "rgb_pearson_b_tissue": _pearson(low[..., 2][compare_mask], high_display[..., 2][compare_mask]),
        "gray_mae_tissue": float(np.mean(np.abs(tissue_low - tissue_high))),
        "gray_rmse_tissue": float(np.sqrt(np.mean((tissue_low - tissue_high) ** 2))),
        "gray_ssim_full": float(structural_similarity(low_gray, high_gray, data_range=255.0)),
        "shift_y_px": shift[0],
        "shift_x_px": shift[1],
        "shift_norm_px": float(np.hypot(shift[0], shift[1])),
        "shift_norm_um": float(np.hypot(shift[0], shift[1]) * low_pixel_size_um),
        "phase_error": phase_error,
        "stored_low_foreground_fraction": float(low_mask.mean()),
        "derived_high_foreground_fraction": float(high_mask.mean()),
        "foreground_mask_iou": iou,
        "foreground_mask_dice": dice,
        "mapped_high_image": str(mapped_path),
        "masked_low_image": str(masked_low_image_path),
        "low_image_used_for_comparison": str(low_image_path),
        "overlay_path": str(overlay_path),
    }
    _save_overlay(
        overlay_path,
        section_id=section_id,
        low=low,
        high_mapped=high_mapped,
        low_mask=low_mask,
        high_mask=high_mask,
        valid=valid,
        metrics=metrics,
    )
    return metrics


def _status(metrics: dict) -> str:
    checks = [
        metrics["raw_source_same"],
        metrics["requested_bbox_same"],
        # A one-pixel edge loss is expected because the two pyramid levels
        # round the same level-0 crop to different integer dimensions.
        metrics["high_to_low_grid_valid_fraction"] >= 0.997,
        metrics["low_direct_read_max_abs_diff"] == 0,
        metrics["shift_norm_px"] <= 0.5,
        metrics["foreground_mask_iou"] >= 0.90,
    ]
    return "consistent" if all(checks) else "review_required"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/00029_g0"))
    parser.add_argument(
        "--high-root",
        type=Path,
        default=Path("data/00029_g0/nucleus_path_raw_v1_coordinate_corrected_v2/segmentation"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/00029_g0/qc/high_low_he_coordinate_validation_v1"),
    )
    parser.add_argument("--sections", nargs="+", type=int, default=list(ANCHORS))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    dataset_dir = args.dataset_dir if args.dataset_dir.is_absolute() else root / args.dataset_dir
    high_root = args.high_root if args.high_root.is_absolute() else root / args.high_root
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_parquet(dataset_dir / "he_sections.parquet")
    table["section_id"] = table["section_id"].astype(int)
    selected = table[table["section_id"].isin(args.sections)].sort_values("z_um")
    if len(selected) != len(args.sections):
        missing = sorted(set(args.sections) - set(selected["section_id"]))
        raise ValueError(f"Missing requested sections: {missing}")

    raw_groups: Dict[str, list] = {}
    for _, row in selected.iterrows():
        low_meta = _read_json(Path(row["source_image_path"]).parent / "metadata.json")
        raw_groups.setdefault(str(low_meta["source_sdpc"]), []).append(row)

    results = []
    for raw_source, rows in raw_groups.items():
        with SdpcPyramid(raw_source) as reader:
            for row in rows:
                print(f"CHECK section {int(row['section_id']):03d}", flush=True)
                result = _process_section(row, high_root, output_dir, reader)
                result["status"] = _status(result)
                results.append(result)
                print(
                    f"  {result['status']} corr={result['gray_pearson_tissue']:.4f} "
                    f"shift={result['shift_norm_px']:.3f}px "
                    f"mask_iou={result['foreground_mask_iou']:.4f}",
                    flush=True,
                )
    results.sort(key=lambda item: item["z_um"])
    frame_checks = {
        "n_sections": len(results),
        "n_consistent": sum(item["status"] == "consistent" for item in results),
        "n_review_required": sum(item["status"] != "consistent" for item in results),
        "all_same_raw_source": all(item["raw_source_same"] for item in results),
        "all_requested_bboxes_same": all(item["requested_bbox_same"] for item in results),
        "all_low_direct_reads_exact": all(item["low_direct_read_max_abs_diff"] == 0 for item in results),
        "min_grid_valid_fraction": float(min(item["high_to_low_grid_valid_fraction"] for item in results)),
        "min_gray_pearson_tissue": float(min(item["gray_pearson_tissue"] for item in results)),
        "max_shift_norm_px": float(max(item["shift_norm_px"] for item in results)),
        "min_foreground_mask_iou": float(min(item["foreground_mask_iou"] for item in results)),
        "conclusion": (
            "The high- and low-resolution crops are consistent representations of "
            "the same SDPC coordinate geometry under the recorded mapping."
            if all(item["status"] == "consistent" for item in results)
            else "At least one section needs manual review before treating the mapping as validated."
        ),
    }
    pd.DataFrame(results).to_csv(output_dir / "section_metrics.csv", index=False)
    (output_dir / "validation_report.json").write_text(
        json.dumps({"frame_checks": frame_checks, "sections": results}, indent=2) + "\n",
        encoding="utf-8",
    )
    _save_overview(output_dir / "overview_red_green.png", results)
    readme = f"""# 00029/g0 high-vs-low H&E coordinate validation

This is a read-only validation. The level-4 H&E crop is the fixed reference.
The level-3 SDPC crop is sampled onto the level-4 pixel-center grid using raw
level-0 crop origins and pyramid downsample factors. No registration transform
is estimated and no existing data is overwritten.

Sections checked: {', '.join(f'{int(x):03d}' for x in args.sections)}

## Result

{frame_checks['conclusion']}

- sections consistent: {frame_checks['n_consistent']}/{frame_checks['n_sections']}
- same SDPC source for every section: {frame_checks['all_same_raw_source']}
- identical requested crop boxes: {frame_checks['all_requested_bboxes_same']}
- stored level-4 image equals an independent SDPC level-4 read: {frame_checks['all_low_direct_reads_exact']}
- minimum tissue grayscale Pearson correlation: {frame_checks['min_gray_pearson_tissue']:.4f}
- maximum phase-correlation shift: {frame_checks['max_shift_norm_px']:.4f} low-resolution pixels
- minimum foreground-mask IoU: {frame_checks['min_foreground_mask_iou']:.4f}

The pixel coordinates are not numerically identical between levels. The result
tests whether they map to the same physical crop geometry after accounting for
MPP, pyramid level, and crop origin.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(frame_checks, indent=2), flush=True)


if __name__ == "__main__":
    main()
