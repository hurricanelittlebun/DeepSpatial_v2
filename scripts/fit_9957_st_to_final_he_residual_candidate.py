#!/usr/bin/env python3
"""Fit non-destructive ST-only residual affines in the final H&E frame.

The reviewed 9957/g0 H&E coordinate chain and its tissue masks are immutable
inputs.  This script starts from ``spatial_he_local`` in the final anchor H5AD,
maps those coordinates through the final H&E mapper, and fits only a small
per-anchor affine on the ST points.  It writes candidate coordinates and QC
figures; it never edits an H5AD, H&E image, H&E mask, feature store, or H&E
transform.

The candidate affine is expressed in the final global XY frame:

    p_st_candidate = M_st_base_to_fixed_he @ p_st_base

where ``p_st_base`` is ``spatial_he_local``.  The affine is fitted in the
fixed H&E crop's local physical frame, then the corrected local points are
mapped once through the fixed H&E chain.  This avoids using a nonlinear
global-to-local round trip as the fitting objective.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]
GROUP = ROOT / "data" / "9957_g0"
PREP = GROUP / "reconstruction_prep_final_he_53_55_v1_chain_v1"
REG = GROUP / "registration_correction_he_final_he_53_55_v1_chain_v1"
H5AD = REG / "9957_g0__st_to_final_he_53_55_v1_chain_v1.h5ad"
OUTPUT = GROUP / "st_to_final_he_residual_candidate_v1"
ANCHORS = (1, 11, 21, 31, 41, 51, 61, 71, 81, 91)
FRAME = "9957__g0_registered_final_he_53_55_v1_chain_v1"


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def apply_homogeneous(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    matrix = np.asarray(matrix, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape [N, 2]")
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("matrix must be finite with shape [3, 3]")
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=float)))
    output = (homogeneous @ matrix.T)[:, :2]
    if not np.isfinite(output).all():
        raise ValueError("affine transform produced non-finite points")
    return output


def residual_matrix(parameters: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Build a centered rotation/scale/translation residual affine."""

    theta, log_sx, log_sy, tx, ty = np.asarray(parameters, dtype=float)
    center = np.asarray(center, dtype=float)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("center must be a finite vector with shape [2]")
    cosine, sine = np.cos(theta), np.sin(theta)
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=float)
    linear = rotation @ np.diag([np.exp(log_sx), np.exp(log_sy)])
    matrix = np.eye(3, dtype=float)
    matrix[:2, :2] = linear
    matrix[:2, 2] = center + np.array([tx, ty]) - linear @ center
    return matrix


def _signed_distance_um(mask: np.ndarray, pixel_size_um: tuple[float, float]) -> np.ndarray:
    sy, sx = float(pixel_size_um[1]), float(pixel_size_um[0])
    inside = distance_transform_edt(mask, sampling=(sy, sx))
    outside = distance_transform_edt(~mask, sampling=(sy, sx))
    return inside - outside


def _sample_mask_distance(
    points: np.ndarray,
    mask: np.ndarray,
    signed_distance_um: np.ndarray,
    pixel_size_um: tuple[float, float],
    origin_um: tuple[float, float] = (0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=float)
    sx, sy = float(pixel_size_um[0]), float(pixel_size_um[1])
    origin_x, origin_y = float(origin_um[0]), float(origin_um[1])
    columns = (points[:, 0] - origin_x) / sx
    rows = (points[:, 1] - origin_y) / sy
    valid = (
        np.isfinite(points).all(axis=1)
        & (rows >= 0)
        & (rows <= mask.shape[0] - 1)
        & (columns >= 0)
        & (columns <= mask.shape[1] - 1)
    )
    inside = np.zeros(len(points), dtype=bool)
    signed = np.full(len(points), -float(np.max(np.abs(signed_distance_um))), dtype=float)
    if valid.any():
        valid_rows = rows[valid]
        valid_columns = columns[valid]
        inside[valid] = mask[
            np.rint(valid_rows).astype(np.int64),
            np.rint(valid_columns).astype(np.int64),
        ]
        signed[valid] = map_coordinates(
            signed_distance_um,
            [valid_rows, valid_columns],
            order=1,
            mode="nearest",
        )
    return inside, valid, signed


def mask_alignment_score(
    points: np.ndarray,
    mask: np.ndarray,
    *,
    pixel_size_um: tuple[float, float] = (1.0, 1.0),
    origin_um: tuple[float, float] = (0.0, 0.0),
    signed_distance_um: np.ndarray | None = None,
) -> dict[str, float]:
    """Evaluate points against a fixed mask without modifying the mask."""

    mask = np.asarray(mask, dtype=bool)
    if signed_distance_um is None:
        signed_distance_um = _signed_distance_um(mask, pixel_size_um)
    inside, valid, signed = _sample_mask_distance(
        points, mask, signed_distance_um, pixel_size_um, origin_um
    )
    valid_signed = signed[valid]
    inside_valid = float(inside[valid].mean()) if valid.any() else 0.0
    soft_inside = (
        float(np.tanh(np.clip(valid_signed / 40.0, -8.0, 8.0)).mean())
        if valid.any()
        else -1.0
    )
    # This score is only used to choose between identity and a candidate.  It
    # rewards mask coverage and distance from the fixed tissue boundary while
    # keeping both terms in a stable, interpretable range.
    quality = 0.70 * inside_valid + 0.30 * ((soft_inside + 1.0) / 2.0)
    return {
        "inside_fraction": float(inside.mean()),
        "valid_crop_fraction": float(valid.mean()),
        "inside_valid_fraction": inside_valid,
        "median_signed_distance_um": float(np.median(valid_signed)) if valid.any() else float("nan"),
        "mean_signed_distance_um": float(np.mean(valid_signed)) if valid.any() else float("nan"),
        "soft_inside_score": float((soft_inside + 1.0) / 2.0),
        "quality_score": float(quality),
    }


def rasterize_forward_mask(
    forward_points: np.ndarray,
    *,
    scale_um: float,
    padding_um: float = 0.0,
    dilation_iterations: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize a fixed mask from points already in the global frame.

    The returned origin is explicit and the raster contains no inverse
    coordinate query.  This is used for the v3 global-frame objective.
    """

    points = np.asarray(forward_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        raise ValueError("forward_points must be a non-empty [N, 2] array")
    if not np.isfinite(points).all() or scale_um <= 0 or padding_um < 0:
        raise ValueError("invalid forward points or raster parameters")
    origin = np.floor((points.min(axis=0) - padding_um) / scale_um) * scale_um
    extent = points.max(axis=0) - origin
    shape = np.maximum(1, np.floor(extent / scale_um).astype(np.int64) + 1)
    mask = np.zeros((int(shape[1]), int(shape[0])), dtype=bool)
    columns = np.floor((points[:, 0] - origin[0]) / scale_um).astype(np.int64)
    rows = np.floor((points[:, 1] - origin[1]) / scale_um).astype(np.int64)
    valid = (
        (rows >= 0)
        & (rows < mask.shape[0])
        & (columns >= 0)
        & (columns < mask.shape[1])
    )
    mask[rows[valid], columns[valid]] = True
    if dilation_iterations > 0:
        from scipy.ndimage import binary_dilation

        mask = binary_dilation(mask, iterations=int(dilation_iterations))
    return mask, origin


def evaluate_local_residual_candidate(
    parameters: np.ndarray,
    local_points: np.ndarray,
    center: np.ndarray,
    mask: np.ndarray,
    *,
    pixel_size_um: tuple[float, float] = (1.0, 1.0),
    origin_um: tuple[float, float] = (0.0, 0.0),
    signed_distance_um: np.ndarray | None = None,
) -> dict[str, Any]:
    """Apply and score an ST residual in the fixed H&E local frame."""

    matrix = residual_matrix(parameters, center)
    candidate_local = apply_homogeneous(local_points, matrix)
    metrics = mask_alignment_score(
        candidate_local,
        mask,
        pixel_size_um=pixel_size_um,
        origin_um=origin_um,
        signed_distance_um=signed_distance_um,
    )
    return {
        **metrics,
        "matrix_st_local_to_fixed_he_local": matrix,
        "candidate_local": candidate_local,
    }


def _load_renderer():
    path = ROOT / "scripts" / "render_9957_final_he_alignment.py"
    spec = importlib.util.spec_from_file_location("_final_he_renderer_for_st_residual", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load final H&E renderer from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The renderer has defaults for older releases; override them explicitly
    # so this script can never silently switch H&E coordinate versions.
    module.FINAL_TABLE = PREP / "he_sections.parquet"
    module.DIRECT_ROOT = GROUP / "stalign_realign_53_55_v1_coordinate_chain_candidate_v2_coarse_init"
    module.REMOVED_SECTION_IDS = {54}
    module.DIRECT_START_SECTION = 55
    return module


def _load_st_local_by_section(h5ad_path: Path) -> dict[int, np.ndarray]:
    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        if "section_id" not in adata.obs:
            raise KeyError("final H5AD lacks obs['section_id']")
        if "spatial_he_local" not in adata.obsm:
            raise KeyError("final H5AD lacks obsm['spatial_he_local']")
        section_ids = pd.to_numeric(adata.obs["section_id"], errors="raise").astype(int).to_numpy()
        local = np.asarray(adata.obsm["spatial_he_local"], dtype=float)
    finally:
        adata.file.close()
    if local.ndim != 2 or local.shape != (len(section_ids), 2):
        raise ValueError(f"invalid spatial_he_local shape: {local.shape}")
    return {
        sid: local[section_ids == sid]
        for sid in sorted(np.unique(section_ids).tolist())
    }


def _sample_indices(n: int, limit: int, seed: int) -> np.ndarray:
    if n <= limit:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=limit, replace=False))


def _evaluate(
    parameters: np.ndarray,
    base_local: np.ndarray,
    center: np.ndarray,
    mapper: Any,
    source: dict[str, Any],
    section_id: int,
    signed_distance_um: np.ndarray,
) -> dict[str, Any]:
    pixel_size = tuple(float(v) for v in source["pixel_size_um"])
    origin = tuple(float(v) for v in source["local_origin_um"])
    local_evaluation = evaluate_local_residual_candidate(
        parameters,
        base_local,
        center,
        source["mask"],
        pixel_size_um=pixel_size,
        origin_um=origin,
        signed_distance_um=signed_distance_um,
    )
    candidate_local = local_evaluation["candidate_local"]
    candidate_global = mapper.local_to_global(candidate_local, section_id)
    metrics = {key: value for key, value in local_evaluation.items() if key not in {"candidate_local"}}
    metrics.pop("matrix_st_local_to_fixed_he_local", None)
    return {
        **metrics,
        "matrix_st_local_to_fixed_he_local": local_evaluation["matrix_st_local_to_fixed_he_local"],
        "candidate_global": candidate_global,
        "candidate_local": candidate_local,
    }


def _plot_anchor(
    renderer: Any,
    mapper: Any,
    row: Any,
    source: dict[str, Any],
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    output_path: Path,
    max_plot_points: int,
    seed: int,
) -> None:
    sid = int(row.section_id)
    boundary_px = renderer.BASE._boundary_points(source["mask"], 6000)
    boundary_local = renderer.BASE._local_pixels_to_um(boundary_px, source)
    boundary_global = mapper.local_to_global(boundary_local, sid)
    low = boundary_global.min(axis=0) - 100.0
    high = boundary_global.max(axis=0) + 100.0
    scale_um = 6.0
    shape = (
        max(1, int(np.ceil((high[1] - low[1]) / scale_um))),
        max(1, int(np.ceil((high[0] - low[0]) / scale_um))),
    )
    rgb, _ = renderer.BASE._sample_source_on_global_grid(
        mapper, source, sid, low, shape, scale_um
    )
    extent = [
        float(low[0]),
        float(low[0] + shape[1] * scale_um),
        float(low[1] + shape[0] * scale_um),
        float(low[1]),
    ]
    rng = np.random.default_rng(seed)
    base = baseline["base_global"]
    after = candidate["candidate_global"]
    if len(base) > max_plot_points:
        indices = np.sort(rng.choice(len(base), size=max_plot_points, replace=False))
        base = base[indices]
        after = after[indices]

    fig, axes = plt.subplots(2, 2, figsize=(16, 13), dpi=160, constrained_layout=True)
    ax0, ax1, ax2, ax3 = axes.flat
    for axis in (ax0, ax1, ax2):
        axis.imshow(rgb, extent=extent, origin="upper", interpolation="nearest")
        axis.set_aspect("equal")
        axis.set_xlabel("final global X (µm)")
        axis.set_ylabel("final global Y (µm)")
    ax0.scatter(base[:, 0], base[:, 1], s=1.0, c="#00d9ff", alpha=0.42, linewidths=0, rasterized=True)
    ax0.set_title("Before: ST mapped through fixed H&E chain\ncyan = ST")
    ax1.scatter(after[:, 0], after[:, 1], s=1.0, c="#ffd400", alpha=0.48, linewidths=0, rasterized=True)
    ax1.set_title("After: ST-only residual affine\nyellow = corrected ST")
    ax2.scatter(base[:, 0], base[:, 1], s=1.0, c="#ff2638", alpha=0.35, linewidths=0, rasterized=True, label="before")
    ax2.scatter(after[:, 0], after[:, 1], s=1.0, c="#00d84a", alpha=0.35, linewidths=0, rasterized=True, label="after")
    ax2.legend(loc="upper right", markerscale=5, framealpha=0.9)
    ax2.set_title("Before / after in the same fixed H&E frame\nred = before; green = after")

    ax3.axis("off")
    p = np.asarray(candidate["parameters"], dtype=float)
    info = [
        f"frame: {FRAME}",
        f"section: {sid:03d} | z={float(row.z_um):.1f} µm",
        f"ST cells: {len(baseline['base_global']):,}",
        "",
        f"before inside mask: {baseline['inside_valid_fraction']:.4f}",
        f"after inside mask:  {candidate['inside_valid_fraction']:.4f}",
        f"change:             {candidate['inside_valid_fraction'] - baseline['inside_valid_fraction']:+.4f}",
        f"before median signed distance: {baseline['median_signed_distance_um']:.2f} µm",
        f"after median signed distance:  {candidate['median_signed_distance_um']:.2f} µm",
        f"before quality: {baseline['quality_score']:.5f}",
        f"after quality:  {candidate['quality_score']:.5f}",
        "",
        f"theta: {p[0]:+.5f} rad ({np.degrees(p[0]):+.2f}°)",
        f"log_sx/log_sy: {p[1]:+.5f} / {p[2]:+.5f}",
        f"translation: {p[3]:+.2f}, {p[4]:+.2f} µm",
        "",
        "H&E image, mask, and H&E transforms: FIXED",
        "Only ST candidate coordinates are changed.",
    ]
    ax3.text(0.02, 0.98, "\n".join(info), va="top", ha="left", fontsize=11, family="DejaVu Sans Mono")
    ax3.set_title("ST-only residual affine QC")
    fig.suptitle(f"9957/g0 section {sid:03d} | fixed H&E, ST fine adjustment", fontsize=17)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _contact_sheet(paths: list[Path], output_path: Path) -> None:
    thumbs = []
    for path in paths:
        with Image.open(path) as image:
            thumb = image.convert("RGB")
            thumb.thumbnail((760, 560), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (780, 610), "white")
            canvas.paste(thumb, ((780 - thumb.width) // 2, 30 + (560 - thumb.height) // 2))
            ImageDraw.Draw(canvas).text((10, 8), path.stem, fill="black")
            thumbs.append(canvas)
    columns = 2
    rows = int(np.ceil(len(thumbs) / columns))
    sheet = Image.new("RGB", (columns * 780, rows * 610), "white")
    for index, image in enumerate(thumbs):
        sheet.paste(image, ((index % columns) * 780, (index // columns) * 610))
    sheet.save(output_path)


def fit_anchor(
    sid: int,
    local_points: np.ndarray,
    row: Any,
    mapper: Any,
    source: dict[str, Any],
    output_dir: Path,
    *,
    max_opt_points: int,
    max_plot_points: int,
    seed: int,
    bounds: list[tuple[float, float]],
) -> dict[str, Any]:
    base_global = mapper.local_to_global(local_points, sid)
    center_local = np.mean(local_points, axis=0)
    signed_distance = _signed_distance_um(source["mask"], tuple(float(v) for v in source["pixel_size_um"]))
    opt_indices = _sample_indices(len(local_points), max_opt_points, seed + sid)
    opt_local = local_points[opt_indices]
    opt_center_local = np.mean(opt_local, axis=0)

    def objective(parameters: np.ndarray) -> float:
        evaluation = _evaluate(
            parameters,
            opt_local,
            opt_center_local,
            mapper,
            source,
            sid,
            signed_distance,
        )
        prior = (
            (parameters[0] / max(abs(bounds[0][0]), abs(bounds[0][1]))) ** 2
            + (parameters[1] / max(abs(bounds[1][0]), abs(bounds[1][1]))) ** 2
            + (parameters[2] / max(abs(bounds[2][0]), abs(bounds[2][1]))) ** 2
            + (parameters[3] / max(abs(bounds[3][0]), abs(bounds[3][1]))) ** 2
            + (parameters[4] / max(abs(bounds[4][0]), abs(bounds[4][1]))) ** 2
        )
        return -float(evaluation["quality_score"] - 0.005 * prior)

    result = minimize(
        objective,
        np.zeros(5, dtype=float),
        method="Powell",
        bounds=bounds,
        options={"maxiter": 70, "xtol": 1e-3, "ftol": 1e-5, "disp": False},
    )
    fitted = np.asarray(result.x, dtype=float)
    if not np.isfinite(fitted).all():
        fitted = np.zeros(5, dtype=float)

    baseline = _evaluate(
        np.zeros(5, dtype=float),
        local_points,
        center_local,
        mapper,
        source,
        sid,
        signed_distance,
    )
    candidate = _evaluate(
        fitted,
        local_points,
        center_local,
        mapper,
        source,
        sid,
        signed_distance,
    )
    # A mask-driven correction must not be accepted if it makes the full-cell
    # evidence worse.  In that case the candidate is explicitly identity,
    # rather than silently publishing an optimizer artifact.
    accepted = (
        candidate["quality_score"] >= baseline["quality_score"] + 1e-5
        and candidate["inside_valid_fraction"] >= baseline["inside_valid_fraction"] - 0.002
    )
    if not accepted:
        fitted = np.zeros(5, dtype=float)
        candidate = baseline.copy()
        candidate["parameters"] = fitted
        candidate["matrix_st_local_to_fixed_he_local"] = np.eye(3, dtype=float)
        candidate["candidate_global"] = base_global.copy()
        candidate["candidate_local"] = local_points.copy()
    baseline["base_global"] = base_global
    baseline["candidate_local"] = local_points
    baseline["parameters"] = np.zeros(5, dtype=float)
    candidate["base_global"] = base_global
    candidate["parameters"] = fitted
    at_bound = any(
        abs(fitted[index] - low) < 2e-3 or abs(fitted[index] - high) < 2e-3
        for index, (low, high) in enumerate(bounds)
    )
    status = "accepted_candidate" if accepted else "identity_retained"
    if at_bound and accepted:
        status = "accepted_candidate_bound_review"

    transform_dir = output_dir / "transforms"
    coordinate_dir = output_dir / "candidate_coordinates"
    overlay_dir = output_dir / "before_after_overlays"
    transform_dir.mkdir(parents=True, exist_ok=True)
    coordinate_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    transform_path = transform_dir / f"section-{sid:03d}__st_to_fixed_he_residual.npz"
    coordinate_path = coordinate_dir / f"section-{sid:03d}__st_corrected_candidate.npz"
    np.savez_compressed(
        transform_path,
        matrix_st_local_to_fixed_he_local=np.asarray(candidate["matrix_st_local_to_fixed_he_local"], dtype=np.float64),
        parameters=fitted,
        center_local=center_local,
        baseline_inside_valid_fraction=np.asarray([baseline["inside_valid_fraction"]]),
        candidate_inside_valid_fraction=np.asarray([candidate["inside_valid_fraction"]]),
        status=np.asarray(status),
        coordinate_frame=np.asarray(FRAME),
        he_is_fixed=np.asarray(True),
    )
    np.savez_compressed(
        coordinate_path,
        spatial_he_local=np.asarray(local_points, dtype=np.float64),
        st_global_base=np.asarray(base_global, dtype=np.float64),
        st_global_candidate=np.asarray(candidate["candidate_global"], dtype=np.float64),
        spatial_he_local_candidate=np.asarray(candidate["candidate_local"], dtype=np.float64),
        matrix_st_local_to_fixed_he_local=np.asarray(candidate["matrix_st_local_to_fixed_he_local"], dtype=np.float64),
        coordinate_frame=np.asarray(FRAME),
    )
    overlay_path = overlay_dir / f"section-{sid:03d}__st_before_after_fixed_he.png"
    _plot_anchor(
        mapper._renderer if hasattr(mapper, "_renderer") else _load_renderer(),
        mapper,
        row,
        source,
        baseline,
        candidate,
        overlay_path,
        max_plot_points,
        seed + sid,
    )
    metric_keys = (
        "inside_fraction",
        "valid_crop_fraction",
        "inside_valid_fraction",
        "median_signed_distance_um",
        "mean_signed_distance_um",
        "soft_inside_score",
        "quality_score",
    )
    return {
        "section_id": int(sid),
        "z_um": float(row.z_um),
        "n_cells": int(len(local_points)),
        "status": status,
        "accepted": bool(accepted),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_nfev": int(result.nfev),
        "parameters": fitted.tolist(),
        "matrix_st_local_to_fixed_he_local": np.asarray(candidate["matrix_st_local_to_fixed_he_local"]).tolist(),
        "baseline": {key: float(baseline[key]) for key in metric_keys},
        "candidate": {key: float(candidate[key]) for key in metric_keys},
        "improvement_inside_valid_fraction": float(
            candidate["inside_valid_fraction"] - baseline["inside_valid_fraction"]
        ),
        "improvement_quality_score": float(candidate["quality_score"] - baseline["quality_score"]),
        "transform_path": str(transform_path.resolve()),
        "coordinate_path": str(coordinate_path.resolve()),
        "overlay_path": str(overlay_path.resolve()),
        "h_e_image_path": str(Path(source["image_path"]).resolve()),
        "h_e_mask_path": str(Path(source["mask_path"]).resolve()),
        "h_e_is_fixed": True,
        "coordinate_frame": FRAME,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad", type=Path, default=H5AD)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--sections", type=int, nargs="+", default=list(ANCHORS))
    parser.add_argument("--max-opt-points", type=int, default=3000)
    parser.add_argument("--max-plot-points", type=int, default=80000)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing candidate directory: {output_dir}")
    if not args.h5ad.is_file():
        raise FileNotFoundError(args.h5ad)
    requested = tuple(sorted(set(int(v) for v in args.sections)))
    if not requested or any(v not in ANCHORS for v in requested):
        raise ValueError(f"sections must be a subset of {ANCHORS}")

    renderer = _load_renderer()
    table = pd.read_parquet(PREP / "he_sections.parquet").copy()
    table["section_id"] = pd.to_numeric(table["section_id"], errors="raise").astype(int)
    table = table[~table["section_id"].isin({54})].sort_values("section_id").reset_index(drop=True)
    mapper = renderer.FinalCoordinateMapper(table, device=args.device)
    # Attach the already-loaded renderer so plot generation cannot accidentally
    # instantiate a mapper with a different coordinate release.
    mapper._renderer = renderer
    local_by_section = _load_st_local_by_section(args.h5ad.resolve())
    rows = {int(row.section_id): row for row in table.itertuples(index=False)}
    missing = [sid for sid in requested if sid not in local_by_section or sid not in rows]
    if missing:
        raise ValueError(f"missing requested anchor sections: {missing}")

    bounds = [(-0.12, 0.12), (-0.08, 0.08), (-0.08, 0.08), (-180.0, 180.0), (-180.0, 180.0)]
    records = []
    overlay_paths = []
    mask_hashes: dict[str, str] = {}
    image_hashes: dict[str, str] = {}
    for sid in requested:
        print(f"fitting ST-only residual candidate for section {sid}", flush=True)
        source = renderer.BASE._load_source(rows[sid])
        mask_path = Path(source["mask_path"])
        image_path = Path(source["image_path"])
        mask_hashes[str(mask_path.resolve())] = sha256(mask_path)
        image_hashes[str(image_path.resolve())] = sha256(image_path)
        record = fit_anchor(
            sid,
            local_by_section[sid],
            rows[sid],
            mapper,
            source,
            output_dir,
            max_opt_points=int(args.max_opt_points),
            max_plot_points=int(args.max_plot_points),
            seed=int(args.seed),
            bounds=bounds,
        )
        records.append(record)
        overlay_paths.append(Path(record["overlay_path"]))
        print(json.dumps(record, ensure_ascii=False), flush=True)

    metrics = {
        "status": "st_only_residual_candidate_not_published",
        "coordinate_frame": FRAME,
        "h5ad_source": str(args.h5ad.resolve()),
        "h_e_table_source": str((PREP / "he_sections.parquet").resolve()),
        "anchor_sections": list(requested),
        "bounds": bounds,
        "he_is_fixed": True,
        "he_masks_sha256_before": mask_hashes,
        "he_images_sha256_before": image_hashes,
        "records": records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "residual_fit_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _contact_sheet(overlay_paths, output_dir / "st_before_after_contact_sheet.png")
    manifest = {
        "status": "candidate_only",
        "coordinate_frame": FRAME,
        "h5ad_source": str(args.h5ad.resolve()),
        "st_input_obsm": "spatial_he_local",
        "st_mapping": "spatial_he_local -> fixed final H&E mapper -> optional ST-only residual affine",
        "h_e_policy": "H&E images, masks, feature stores, and H&E transform chain were not modified",
        "published_to_h5ad": False,
        "output": str(output_dir.resolve()),
        "metrics": str((output_dir / "residual_fit_metrics.json").resolve()),
        "contact_sheet": str((output_dir / "st_before_after_contact_sheet.png").resolve()),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        "# 9957/g0 ST-to-fixed-H&E residual candidate\n\n"
        "This is a review-only candidate. The final H&E coordinate chain and masks are fixed. "
        "Only ST coordinates derived from `spatial_he_local` were given a bounded per-anchor "
        "residual affine. No H5AD or H&E artifact was overwritten.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
