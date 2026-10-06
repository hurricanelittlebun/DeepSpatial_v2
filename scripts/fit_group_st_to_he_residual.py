#!/usr/bin/env python3
"""Fit ST-only residual affines against fixed H&E masks for normalized groups.

This is the generalized version of the previous 00029/g0 correction.  H&E
images, masks, and serial-registration transforms are treated as fixed.  For
each ST anchor section, only a small affine is fitted in the registered ST/g
frame and materialized as:

    spatial_st_corrected = matrix_current_g_to_fixed_he @ spatial_registered

The original ``spatial_registered`` coordinates are never overwritten.  The
script writes transforms, before/after overlays, metrics, and compact NPZ
coordinate candidates.  Full candidate H5ADs can be materialized separately
after the overlays have been reviewed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.optimize import minimize


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import fit_00029_st_to_he_residual as residual  # noqa: E402


DEFAULT_ROOT = SCRIPT_DIR.parent
DEFAULT_V2_ROOT = DEFAULT_ROOT.parent / "DeepSpatial_v2"
DEFAULT_GROUP_DIR = DEFAULT_ROOT / "data/normalized_groups_v1"
DEFAULT_OUTPUT_ROOT = DEFAULT_ROOT / "data"


def parse_group_stem(stem: str) -> tuple[str, str]:
    match = re.fullmatch(r"(.+)_g(\d+)", str(stem))
    if match is None:
        raise ValueError(f"not a per-group stem: {stem!r}")
    return match.group(1), f"g{match.group(2)}"


def group_output_dir(output_root: Path | str, group_stem: str) -> Path:
    parse_group_stem(group_stem)
    return Path(output_root) / group_stem / "registration_correction_he_v1"


def apply_current_to_fixed(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a saved current-g -> fixed-H&E affine to XY points."""

    return residual.apply_homogeneous(np.asarray(points, dtype=float), matrix)


def current_to_fixed_matrix(evaluation: dict[str, Any]) -> np.ndarray:
    """Read the canonical current-g -> fixed-H&E matrix from an evaluation."""

    return np.asarray(evaluation["matrix_current_g_to_fixed_he"], dtype=float)


def _resolve_path(value: Any, v2_root: Path) -> Path | None:
    if value is None:
        return None
    text = str(value)
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return None
    path = Path(text)
    return path if path.is_absolute() else v2_root / path


def _first_existing(values: list[Any], v2_root: Path) -> Path | None:
    for value in values:
        path = _resolve_path(value, v2_root)
        if path is not None and path.is_file():
            return path
    return None


@dataclass
class SectionInput:
    sample_id: str
    group_id: str
    group_stem: str
    section_id: int
    registered: np.ndarray
    reported_local_px: np.ndarray
    image_path: Path
    mask_path: Path
    transform_path: Path
    mpp_x: float
    mpp_y: float
    registration_confidence: float
    registration_status: str
    forced_flip: bool
    preorientation_axis: str
    source_coordinate_frame: str


def load_group_sections(
    h5ad_path: Path | str,
    v2_root: Path | str,
) -> list[SectionInput]:
    """Load only obs/obsm metadata and ST coordinates from one group H5AD."""

    import anndata as ad
    import pandas as pd

    h5ad_path = Path(h5ad_path)
    v2_root = Path(v2_root)
    group_stem = h5ad_path.stem
    sample_id, group_id = parse_group_stem(group_stem)
    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        required = [
            "section_id",
            "raw_crop_path",
            "he_image_path",
            "he_path",
            "mask_path",
            "he_mask_path",
            "registration_transform_path",
            "crop_level_pixel_size_um",
            "registration_confidence",
            "registration_status",
            "forced_flip",
            "preorientation_axis",
            "source_coordinate_frame",
            "x_he_local_px",
            "y_he_local_px",
        ]
        missing = [column for column in required if column not in adata.obs.columns]
        if missing:
            raise ValueError(f"{h5ad_path.name} missing columns: {', '.join(missing)}")
        obs = adata.obs[required].copy()
        if "spatial_registered" not in adata.obsm:
            raise ValueError(f"{h5ad_path.name} lacks obsm['spatial_registered']")
        registered = np.asarray(adata.obsm["spatial_registered"], dtype=float)
    finally:
        adata.file.close()

    if registered.ndim != 2 or registered.shape != (len(obs), 2):
        raise ValueError(f"invalid spatial_registered shape in {h5ad_path}: {registered.shape}")
    section_values = pd.to_numeric(obs["section_id"], errors="coerce").to_numpy(float)
    if not np.isfinite(section_values).all():
        raise ValueError(f"non-finite section_id in {h5ad_path}")

    result: list[SectionInput] = []
    for section_id in sorted(np.unique(section_values).astype(int).tolist()):
        positions = np.flatnonzero(section_values.astype(int) == section_id)
        section = obs.iloc[positions]
        metadata = section.iloc[0]
        image_path = _first_existing(
            [metadata["raw_crop_path"], metadata["he_image_path"], metadata["he_path"]],
            v2_root,
        )
        mask_path = _first_existing(
            [metadata["mask_path"], metadata["he_mask_path"]], v2_root
        )
        transform_path = _resolve_path(metadata["registration_transform_path"], v2_root)
        if image_path is None or mask_path is None or transform_path is None:
            raise FileNotFoundError(
                f"{group_stem} section {section_id}: image={image_path}, "
                f"mask={mask_path}, transform={transform_path}"
            )
        mpp = float(metadata["crop_level_pixel_size_um"])
        if not np.isfinite(mpp) or mpp <= 0:
            raise ValueError(f"invalid crop_level_pixel_size_um for {group_stem} section {section_id}")
        reported_local_px = np.column_stack(
            [
                section["x_he_local_px"].to_numpy(dtype=float),
                section["y_he_local_px"].to_numpy(dtype=float),
            ]
        )
        result.append(
            SectionInput(
                sample_id=sample_id,
                group_id=group_id,
                group_stem=group_stem,
                section_id=section_id,
                registered=registered[positions],
                reported_local_px=reported_local_px,
                image_path=image_path,
                mask_path=mask_path,
                transform_path=transform_path,
                mpp_x=mpp,
                mpp_y=mpp,
                registration_confidence=float(
                    pd.to_numeric(section["registration_confidence"], errors="coerce").median()
                ),
                registration_status=str(metadata["registration_status"]),
                forced_flip=bool(metadata["forced_flip"]),
                preorientation_axis=str(metadata["preorientation_axis"]),
                source_coordinate_frame=str(metadata["source_coordinate_frame"]),
            )
        )
    return result


def _sample_indices(n: int, limit: int, seed: int) -> np.ndarray:
    if n <= limit:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=limit, replace=False))


def _evaluate(
    parameters: np.ndarray,
    registered: np.ndarray,
    center: np.ndarray,
    preorientation: np.ndarray,
    chain: tuple[tuple[str, str], ...],
    maps: residual.RegistrationMaps,
    mask: np.ndarray,
    signed_distance: np.ndarray,
    mpp_x: float,
    mpp_y: float,
) -> dict[str, Any]:
    matrix_he_to_current = residual.residual_matrix(parameters, center)
    matrix_current_to_fixed = np.linalg.inv(matrix_he_to_current)
    corrected = apply_current_to_fixed(registered, matrix_current_to_fixed)
    local = residual.source_points_from_registered(
        corrected, preorientation, chain, maps
    )
    inside, valid = residual.sample_mask(mask, local, mpp_x, mpp_y)
    signed = residual.sample_signed_distance(signed_distance, local, mpp_x, mpp_y)
    return {
        "inside_fraction": float(inside.mean()),
        "valid_crop_fraction": float(valid.mean()),
        "inside_valid_fraction": float(inside[valid].mean()) if valid.any() else 0.0,
        "median_signed_distance_um": float(np.median(signed[valid])) if valid.any() else float("nan"),
        "corrected_coordinates": corrected,
        "local_coordinates_um": local,
        "valid": valid,
        "inside": inside,
        "matrix_he_to_current_g": matrix_he_to_current,
        "matrix_current_g_to_fixed_he": matrix_current_to_fixed,
    }


def _plot_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    if len(points) <= max_points:
        return points
    indices = _sample_indices(len(points), max_points, seed)
    return points[indices]


def _plot_before_after(
    section: SectionInput,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    parameters: np.ndarray,
    output_path: Path,
    max_plot_points: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    with Image.open(section.image_path) as source:
        image = source.convert("RGB").copy()
    with Image.open(section.mask_path) as source:
        mask = np.asarray(source.convert("L")) > 0
    if mask.shape != (image.height, image.width):
        raise ValueError(
            f"image/mask shape mismatch in {section.group_stem} section {section.section_id}: "
            f"image={(image.height, image.width)}, mask={mask.shape}"
        )

    scale = min(1.0, 1600.0 / max(image.width, image.height))
    width = max(1, int(round(image.width * scale)))
    height = max(1, int(round(image.height * scale)))
    image_small = np.asarray(image.resize((width, height), Image.Resampling.LANCZOS))
    mask_small = np.asarray(
        Image.fromarray((mask.astype(np.uint8) * 255), mode="L").resize(
            (width, height), Image.Resampling.NEAREST
        )
    ) > 0

    before_px = baseline["local_coordinates_um"] / np.array([section.mpp_x, section.mpp_y])
    after_px = candidate["local_coordinates_um"] / np.array([section.mpp_x, section.mpp_y])
    before_px *= np.array([scale, scale])
    after_px *= np.array([scale, scale])
    reported_px = section.reported_local_px * np.array([scale, scale])

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=130)
    panels = [
        (axes[0, 0], before_px, "#ff2638", f"Before residual affine\nregistered ST inverse-mapped to H&E\ncoverage={baseline['inside_fraction']:.3f}"),
        (axes[0, 1], after_px, "#00d84a", f"After residual affine\nST corrected to fixed H&E\ncoverage={candidate['inside_fraction']:.3f}"),
    ]
    for ax, points, color, title in panels:
        ax.imshow(image_small, origin="upper")
        ax.contour(mask_small.astype(float), levels=[0.5], colors=["#ffe600"], linewidths=0.7, origin="upper")
        selected = _plot_points(points, max_plot_points, section.section_id + (1 if color == "#ff2638" else 2))
        ax.scatter(selected[:, 0], selected[:, 1], s=0.8, c=color, alpha=0.45, linewidths=0, rasterized=True)
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("H&E crop x (pixel)")
        ax.set_ylabel("H&E crop y (pixel)")

    ax = axes[1, 0]
    ax.imshow(image_small, origin="upper")
    ax.contour(mask_small.astype(float), levels=[0.5], colors=["#ffe600"], linewidths=0.7, origin="upper")
    before = _plot_points(before_px, max_plot_points, section.section_id + 3)
    after = _plot_points(after_px, max_plot_points, section.section_id + 4)
    ax.scatter(before[:, 0], before[:, 1], s=0.7, c="#ff2638", alpha=0.30, linewidths=0, rasterized=True, label="before")
    ax.scatter(after[:, 0], after[:, 1], s=0.7, c="#00d84a", alpha=0.30, linewidths=0, rasterized=True, label="after")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.set_title("Before / after overlay\nyellow = fixed H&E mask boundary")
    ax.legend(loc="upper right", markerscale=4)
    ax.set_xlabel("H&E crop x (pixel)")
    ax.set_ylabel("H&E crop y (pixel)")

    ax = axes[1, 1]
    ax.axis("off")
    p = np.asarray(parameters, dtype=float)
    info = [
        f"sample / group: {section.sample_id} / {section.group_id}",
        f"section: {section.section_id:03d}",
        f"ST cells: {len(section.registered):,}",
        f"native local ST in mask: {np.nanmean(residual.sample_mask(mask, section.reported_local_px * np.array([section.mpp_x, section.mpp_y]), section.mpp_x, section.mpp_y)[0]):.3f}",
        "",
        f"registered inverse baseline: {baseline['inside_fraction']:.4f}",
        f"residual candidate: {candidate['inside_fraction']:.4f}",
        f"improvement: {candidate['inside_fraction'] - baseline['inside_fraction']:+.4f}",
        f"baseline inside valid: {baseline['inside_valid_fraction']:.4f}",
        f"candidate inside valid: {candidate['inside_valid_fraction']:.4f}",
        f"baseline median signed distance (um): {baseline['median_signed_distance_um']:.2f}",
        f"candidate median signed distance (um): {candidate['median_signed_distance_um']:.2f}",
        "",
        f"theta (rad): {p[0]:+.5f}",
        f"log_sx / log_sy: {p[1]:+.5f} / {p[2]:+.5f}",
        f"translation (um): {p[3]:+.2f}, {p[4]:+.2f}",
        f"registration confidence: {section.registration_confidence:.4f}",
        f"forced flip: {section.forced_flip}",
        "",
        "H&E is fixed; only ST coordinates are changed.",
        "Original spatial_registered remains preserved.",
    ]
    ax.text(0.02, 0.98, "\n".join(info), va="top", ha="left", fontsize=10.5, family="DejaVu Sans Mono")
    ax.set_title("Residual-affine QC summary")

    fig.suptitle(f"{section.sample_id}/{section.group_id} section {section.section_id:03d} | ST → fixed H&E residual correction", fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fit_section(
    section: SectionInput,
    maps: residual.RegistrationMaps,
    output_dir: Path,
    *,
    max_opt_points: int,
    max_eval_points: int | None,
    max_plot_points: int,
    seed: int,
) -> dict[str, Any]:
    with Image.open(section.mask_path) as source:
        mask = np.asarray(source.convert("L")) > 0
    inside_distance = distance_transform_edt(mask)
    outside_distance = distance_transform_edt(~mask)
    signed_distance = inside_distance - outside_distance
    preorientation, chain = residual.read_section_transform(section.transform_path)

    opt_indices = _sample_indices(len(section.registered), max_opt_points, seed + section.section_id)
    opt_points = section.registered[opt_indices]
    center = np.mean(opt_points, axis=0)
    bounds = [
        (-0.25, 0.25),
        (-0.15, 0.15),
        (-0.15, 0.15),
        (-200.0, 200.0),
        (-200.0, 200.0),
    ]
    objective = lambda params: residual.residual_score(
        params,
        opt_points,
        center,
        preorientation,
        chain,
        maps,
        mask,
        signed_distance,
        section.mpp_x,
        section.mpp_y,
    )
    result = minimize(
        objective,
        np.zeros(5, dtype=float),
        method="Powell",
        bounds=bounds,
        options={"maxiter": 55, "xtol": 1e-3, "ftol": 1e-4, "disp": False},
    )
    parameters = np.asarray(result.x, dtype=float)
    if not np.isfinite(parameters).all():
        parameters = np.zeros(5, dtype=float)

    if max_eval_points is None or len(section.registered) <= max_eval_points:
        eval_indices = np.arange(len(section.registered), dtype=np.int64)
    else:
        eval_indices = _sample_indices(len(section.registered), max_eval_points, seed + 100000 + section.section_id)
    eval_registered = section.registered[eval_indices]
    baseline_eval = _evaluate(
        np.zeros(5, dtype=float),
        eval_registered,
        center,
        preorientation,
        chain,
        maps,
        mask,
        signed_distance,
        section.mpp_x,
        section.mpp_y,
    )
    candidate_eval = _evaluate(
        parameters,
        eval_registered,
        center,
        preorientation,
        chain,
        maps,
        mask,
        signed_distance,
        section.mpp_x,
        section.mpp_y,
    )
    baseline = _evaluate(
        np.zeros(5, dtype=float),
        section.registered,
        center,
        preorientation,
        chain,
        maps,
        mask,
        signed_distance,
        section.mpp_x,
        section.mpp_y,
    )
    candidate = _evaluate(
        parameters,
        section.registered,
        center,
        preorientation,
        chain,
        maps,
        mask,
        signed_distance,
        section.mpp_x,
        section.mpp_y,
    )
    improvement = candidate["inside_fraction"] - baseline["inside_fraction"]
    at_bound = any(
        abs(parameters[index] - lower) < 2e-3 or abs(parameters[index] - upper) < 2e-3
        for index, (lower, upper) in enumerate(bounds)
    )
    status = "provisional_residual_affine" if improvement >= 0.05 else "unresolved"
    if at_bound:
        status = "review_required_bound_hit" if improvement >= 0.05 else "unresolved_bound_hit"

    transform_dir = output_dir / "transforms"
    coordinate_dir = output_dir / "candidate_coordinates"
    overlay_dir = output_dir / "before_after_overlays"
    transform_dir.mkdir(parents=True, exist_ok=True)
    coordinate_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    matrix_he_to_current = candidate["matrix_he_to_current_g"]
    matrix_current_to_fixed = current_to_fixed_matrix(candidate)
    transform_path = transform_dir / f"section-{section.section_id:03d}__st_to_fixed_he_residual.npz"
    np.savez_compressed(
        transform_path,
        matrix_he_to_current_g=matrix_he_to_current,
        matrix_current_g_to_fixed_he=matrix_current_to_fixed,
        parameters=parameters,
        center=center,
        baseline_inside_fraction=np.asarray([baseline["inside_fraction"]]),
        candidate_inside_fraction=np.asarray([candidate["inside_fraction"]]),
        baseline_inside_valid_fraction=np.asarray([baseline["inside_valid_fraction"]]),
        candidate_inside_valid_fraction=np.asarray([candidate["inside_valid_fraction"]]),
        status=np.asarray(status),
    )
    coordinate_path = coordinate_dir / f"section-{section.section_id:03d}__st_corrected_candidate.npz"
    np.savez_compressed(
        coordinate_path,
        spatial_registered_original=section.registered,
        spatial_st_corrected=candidate["corrected_coordinates"],
        he_local_before_um=baseline["local_coordinates_um"],
        he_local_after_um=candidate["local_coordinates_um"],
        reported_he_local_px=section.reported_local_px,
    )
    overlay_path = overlay_dir / f"section-{section.section_id:03d}__before_after.png"
    _plot_before_after(
        section, baseline, candidate, parameters, overlay_path, max_plot_points
    )

    def metric_dict(value: dict[str, Any]) -> dict[str, Any]:
        return {
            key: item
            for key, item in value.items()
            if key
            in {
                "inside_fraction",
                "valid_crop_fraction",
                "inside_valid_fraction",
                "median_signed_distance_um",
            }
        }

    return {
        "sample_id": section.sample_id,
        "group_id": section.group_id,
        "group_stem": section.group_stem,
        "section_id": section.section_id,
        "n_cells": int(len(section.registered)),
        "baseline": metric_dict(baseline),
        "candidate": metric_dict(candidate),
        "optimization_eval_baseline": metric_dict(baseline_eval),
        "optimization_eval_candidate": metric_dict(candidate_eval),
        "native_local_inside_fraction": float(
            residual.sample_mask(
                mask,
                section.reported_local_px * np.array([section.mpp_x, section.mpp_y]),
                section.mpp_x,
                section.mpp_y,
            )[0].mean()
        ),
        "improvement": float(improvement),
        "status": status,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "parameters": parameters.tolist(),
        "center": center.tolist(),
        "matrix_he_to_current_g": matrix_he_to_current.tolist(),
        "matrix_current_g_to_fixed_he": matrix_current_to_fixed.tolist(),
        "mask_path": str(section.mask_path),
        "image_path": str(section.image_path),
        "registration_transform_path": str(section.transform_path),
        "registration_confidence": section.registration_confidence,
        "registration_status": section.registration_status,
        "forced_flip": section.forced_flip,
        "preorientation_axis": section.preorientation_axis,
        "source_coordinate_frame": section.source_coordinate_frame,
        "transform_output": str(transform_path),
        "coordinate_output": str(coordinate_path),
        "overlay_output": str(overlay_path),
        "h_and_e_residual_applied": False,
        "st_residual_applied": True,
    }


def discover_group_h5ads(group_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in Path(group_dir).glob("*_g*.h5ad")
        if re.fullmatch(r".+_g\d+", path.stem) and path.stem != "00029_g0"
    )


def fit_group(
    h5ad_path: Path,
    output_root: Path,
    registration_root: Path,
    v2_root: Path,
    *,
    max_opt_points: int,
    max_eval_points: int | None,
    max_plot_points: int,
    seed: int,
) -> dict[str, Any]:
    sections = load_group_sections(h5ad_path, v2_root)
    sample_id, group_id = parse_group_stem(h5ad_path.stem)
    maps = residual.RegistrationMaps(registration_root, sample_id, group_id)
    output_dir = group_output_dir(output_root, h5ad_path.stem)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for section in sections:
        print(f"  fitting {h5ad_path.stem} section {section.section_id}", flush=True)
        try:
            records.append(
                fit_section(
                    section,
                    maps,
                    output_dir,
                    max_opt_points=max_opt_points,
                    max_eval_points=max_eval_points,
                    max_plot_points=max_plot_points,
                    seed=seed + int(sample_id) + int(group_id[1:]) * 1000,
                )
            )
            record = records[-1]
            print(
                f"    baseline={record['baseline']['inside_fraction']:.4f} "
                f"candidate={record['candidate']['inside_fraction']:.4f} "
                f"delta={record['improvement']:+.4f} status={record['status']}",
                flush=True,
            )
        except Exception as error:
            errors.append(
                {
                    "section_id": section.section_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            print(f"    ERROR {type(error).__name__}: {error}", flush=True)

    metrics_path = output_dir / "residual_fit_metrics.json"
    metrics_path.write_text(json.dumps({"sections": records, "errors": errors}, indent=2, ensure_ascii=False) + "\n")
    summary = {
        "format": "deepspatial-st-to-fixed-he-residual-v1",
        "status": "candidate_generated_not_published",
        "sample_id": sample_id,
        "group_id": group_id,
        "group_h5ad": str(h5ad_path.resolve()),
        "fixed_reference": "native registered H&E crop and H&E tissue mask",
        "st_coordinate_key_original": "spatial_registered",
        "st_coordinate_key_candidate": "spatial_st_corrected",
        "h_and_e_residual_applied": False,
        "st_residual_applied": True,
        "n_sections_expected": len(sections),
        "n_sections_fitted": len(records),
        "errors": errors,
        "sections": [record["section_id"] for record in records],
        "note": "Review before materializing corrected group H5AD; original H5AD was not overwritten.",
    }
    (output_dir / "correction_manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-dir", type=Path, default=DEFAULT_GROUP_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--v2-root", type=Path, default=DEFAULT_V2_ROOT)
    parser.add_argument("--registration-root", type=Path, default=DEFAULT_V2_ROOT / "outputs/registration")
    parser.add_argument("--group", action="append", help="per-group H5AD stem; 00029_g0 is refused")
    parser.add_argument("--max-opt-points", type=int, default=2500)
    parser.add_argument("--max-eval-points", type=int, default=None)
    parser.add_argument("--max-plot-points", type=int, default=80000)
    parser.add_argument("--seed", type=int, default=20260922)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.group:
        paths = []
        for stem in args.group:
            if stem == "00029_g0":
                raise ValueError("00029_g0 is intentionally excluded from this correction")
            path = args.group_dir / f"{stem}.h5ad"
            if not path.is_file():
                raise FileNotFoundError(path)
            paths.append(path)
    else:
        paths = discover_group_h5ads(args.group_dir)
    if not paths:
        raise FileNotFoundError(f"no non-00029 group H5AD files found under {args.group_dir}")
    summaries = []
    for path in paths:
        print(f"[group] {path.name}", flush=True)
        summaries.append(
            fit_group(
                path,
                args.output_root,
                args.registration_root,
                args.v2_root,
                max_opt_points=args.max_opt_points,
                max_eval_points=args.max_eval_points,
                max_plot_points=args.max_plot_points,
                seed=args.seed,
            )
        )
    output = args.output_root / "st_to_he_residual_v1_summary.json"
    output.write_text(json.dumps(summaries, indent=2, ensure_ascii=False) + "\n")
    print(f"[done] {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
