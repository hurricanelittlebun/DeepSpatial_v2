#!/usr/bin/env python3
"""Generate a v3 ST-only residual candidate in the fixed global H&E frame.

v3 keeps H&E immutable and evaluates ST points directly against a global mask
created by *forward*-mapping the final H&E mask through the approved H&E chain.
It therefore preserves the visual intent of the earlier v1 candidate while
avoiding v1's global->local nonlinear round trip in the objective.
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
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]
GROUP = ROOT / "data" / "9957_g0"
PREP = GROUP / "reconstruction_prep_final_he_53_55_v1_chain_v1"
REG = GROUP / "registration_correction_he_final_he_53_55_v1_chain_v1"
H5AD = REG / "9957_g0__st_to_final_he_53_55_v1_chain_v1.h5ad"
V1_ROOT = GROUP / "st_to_final_he_residual_candidate_v1"
OUTPUT = GROUP / "st_to_final_he_residual_candidate_v3_global_frame"
ANCHORS = (1, 11, 21, 31, 41, 51, 61, 71, 81, 91)
FRAME = "9957__g0_registered_final_he_53_55_v1_chain_v1"


def _load_v2_module():
    path = ROOT / "scripts" / "fit_9957_st_to_final_he_residual_candidate.py"
    spec = importlib.util.spec_from_file_location("_st_residual_v2_helpers", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


H = _load_v2_module()
apply_homogeneous = H.apply_homogeneous
mask_alignment_score = H.mask_alignment_score
rasterize_forward_mask = H.rasterize_forward_mask
residual_matrix = H.residual_matrix


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _load_renderer():
    path = ROOT / "scripts" / "render_9957_final_he_alignment.py"
    spec = importlib.util.spec_from_file_location("_final_he_renderer_for_st_residual_v3", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load final H&E renderer from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.FINAL_TABLE = PREP / "he_sections.parquet"
    module.DIRECT_ROOT = GROUP / "stalign_realign_53_55_v1_coordinate_chain_candidate_v2_coarse_init"
    module.REMOVED_SECTION_IDS = {54}
    module.DIRECT_START_SECTION = 55
    return module


def _load_st_local_by_section(h5ad_path: Path) -> dict[int, np.ndarray]:
    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        section_ids = pd.to_numeric(adata.obs["section_id"], errors="raise").astype(int).to_numpy()
        local = np.asarray(adata.obsm["spatial_he_local"], dtype=float)
    finally:
        adata.file.close()
    if local.ndim != 2 or local.shape != (len(section_ids), 2):
        raise ValueError(f"invalid spatial_he_local shape: {local.shape}")
    return {sid: local[section_ids == sid] for sid in sorted(np.unique(section_ids).tolist())}


def _sample_indices(n: int, limit: int, seed: int) -> np.ndarray:
    if n <= limit:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=limit, replace=False))


def _chunked_local_to_global(mapper: Any, local: np.ndarray, sid: int, chunk_size: int = 65536) -> np.ndarray:
    result = np.empty_like(local, dtype=float)
    for start in range(0, len(local), chunk_size):
        stop = min(start + chunk_size, len(local))
        result[start:stop] = mapper.local_to_global(local[start:stop], sid)
    return result


def build_fixed_global_mask(
    mapper: Any,
    source: dict[str, Any],
    sid: int,
    *,
    raster_scale_um: float = 4.0,
    source_stride: int = 2,
    dilation_iterations: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward-rasterize the immutable H&E mask in final global XY."""

    mask = np.asarray(source["mask"], dtype=bool)
    stride = max(1, int(source_stride))
    sampled = mask[::stride, ::stride]
    rows, cols = np.nonzero(sampled)
    if len(rows) == 0:
        raise ValueError(f"empty H&E mask for section {sid}")
    sx, sy = (float(v) for v in source["pixel_size_um"])
    origin_x, origin_y = (float(v) for v in source["local_origin_um"])
    local = np.column_stack(
        [origin_x + cols * stride * sx, origin_y + rows * stride * sy]
    )
    global_points = _chunked_local_to_global(mapper, local, sid)
    global_mask, global_origin = rasterize_forward_mask(
        global_points,
        scale_um=float(raster_scale_um),
        padding_um=float(raster_scale_um) * 2.0,
        dilation_iterations=int(dilation_iterations),
    )
    signed_distance = distance_transform_edt(global_mask, sampling=(raster_scale_um, raster_scale_um)) - distance_transform_edt(
        ~global_mask, sampling=(raster_scale_um, raster_scale_um)
    )
    return global_mask, global_origin, signed_distance


def _parameters_from_matrix(matrix: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Convert the restricted residual matrix parameterization back to params."""

    matrix = np.asarray(matrix, dtype=float)
    center = np.asarray(center, dtype=float)
    linear = matrix[:2, :2]
    sx = float(np.linalg.norm(linear[:, 0]))
    sy = float(np.linalg.norm(linear[:, 1]))
    if sx <= 0 or sy <= 0:
        return np.zeros(5, dtype=float)
    theta = float(np.arctan2(linear[1, 0], linear[0, 0]))
    translation = matrix[:2, 2] - center + linear @ center
    return np.array([theta, np.log(sx), np.log(sy), translation[0], translation[1]], dtype=float)


def _evaluate(
    parameters: np.ndarray,
    base_global: np.ndarray,
    center: np.ndarray,
    global_mask: np.ndarray,
    global_origin: np.ndarray,
    signed_distance: np.ndarray,
    raster_scale_um: float,
) -> dict[str, Any]:
    matrix = residual_matrix(parameters, center)
    candidate_global = apply_homogeneous(base_global, matrix)
    metrics = mask_alignment_score(
        candidate_global,
        global_mask,
        pixel_size_um=(raster_scale_um, raster_scale_um),
        origin_um=tuple(float(v) for v in global_origin),
        signed_distance_um=signed_distance,
    )
    return {
        **metrics,
        "matrix_base_st_to_fixed_he": matrix,
        "candidate_global": candidate_global,
    }


def _load_v1_start(sid: int, center: np.ndarray) -> np.ndarray:
    path = V1_ROOT / "transforms" / f"section-{sid:03d}__st_to_fixed_he_residual.npz"
    if not path.is_file():
        return np.zeros(5, dtype=float)
    with np.load(path, allow_pickle=False) as payload:
        if "matrix_base_st_to_fixed_he" not in payload:
            return np.zeros(5, dtype=float)
        matrix = np.asarray(payload["matrix_base_st_to_fixed_he"], dtype=float)
    return _parameters_from_matrix(matrix, center)


def _fit_anchor(
    sid: int,
    local_points: np.ndarray,
    row: Any,
    mapper: Any,
    source: dict[str, Any],
    renderer: Any,
    output_dir: Path,
    *,
    max_opt_points: int,
    max_plot_points: int,
    seed: int,
    raster_scale_um: float,
    source_stride: int,
    dilation_iterations: int,
) -> dict[str, Any]:
    base_global = _chunked_local_to_global(mapper, local_points, sid)
    center = np.mean(base_global, axis=0)
    global_mask, global_origin, signed_distance = build_fixed_global_mask(
        mapper,
        source,
        sid,
        raster_scale_um=raster_scale_um,
        source_stride=source_stride,
        dilation_iterations=dilation_iterations,
    )
    opt_indices = _sample_indices(len(base_global), max_opt_points, seed + sid)
    opt_global = base_global[opt_indices]
    opt_center = np.mean(opt_global, axis=0)
    bounds = [(-0.15, 0.15), (-0.10, 0.10), (-0.10, 0.10), (-240.0, 240.0), (-240.0, 240.0)]

    def objective(parameters: np.ndarray) -> float:
        evaluation = _evaluate(
            parameters,
            opt_global,
            opt_center,
            global_mask,
            global_origin,
            signed_distance,
            raster_scale_um,
        )
        prior = (
            (parameters[0] / 0.15) ** 2
            + (parameters[1] / 0.10) ** 2
            + (parameters[2] / 0.10) ** 2
            + (parameters[3] / 240.0) ** 2
            + (parameters[4] / 240.0) ** 2
        )
        return -float(evaluation["quality_score"] - 0.005 * prior)

    starts = [np.zeros(5, dtype=float), _load_v1_start(sid, opt_center)]
    fits = []
    for start in starts:
        start = np.clip(start, [low for low, _ in bounds], [high for _, high in bounds])
        result = minimize(
            objective,
            start,
            method="Powell",
            bounds=bounds,
            options={"maxiter": 70, "xtol": 1e-3, "ftol": 1e-5, "disp": False},
        )
        fits.append((np.asarray(result.x, dtype=float), result))
    identity = _evaluate(
        np.zeros(5, dtype=float),
        base_global,
        center,
        global_mask,
        global_origin,
        signed_distance,
        raster_scale_um,
    )
    candidates = [
        (identity, np.zeros(5, dtype=float), None),
    ]
    for parameters, result in fits:
        evaluation = _evaluate(
            parameters,
            base_global,
            center,
            global_mask,
            global_origin,
            signed_distance,
            raster_scale_um,
        )
        candidates.append((evaluation, parameters, result))
    candidate, parameters, result = max(candidates, key=lambda item: item[0]["quality_score"])
    accepted = candidate["quality_score"] >= identity["quality_score"] + 1e-5 and candidate["inside_valid_fraction"] >= identity["inside_valid_fraction"] - 0.002
    if not accepted:
        candidate = identity
        parameters = np.zeros(5, dtype=float)
        result = None
    at_bound = any(
        abs(parameters[index] - low) < 2e-3 or abs(parameters[index] - high) < 2e-3
        for index, (low, high) in enumerate(bounds)
    )
    status = "accepted_candidate" if accepted else "identity_retained"
    if at_bound and accepted:
        status = "accepted_candidate_bound_review"

    transform_dir = output_dir / "transforms"
    coordinate_dir = output_dir / "candidate_coordinates"
    mask_dir = output_dir / "fixed_global_masks"
    overlay_dir = output_dir / "before_after_overlays"
    for directory in (transform_dir, coordinate_dir, mask_dir, overlay_dir):
        directory.mkdir(parents=True, exist_ok=True)
    transform_path = transform_dir / f"section-{sid:03d}__st_to_fixed_he_residual.npz"
    coordinate_path = coordinate_dir / f"section-{sid:03d}__st_corrected_candidate.npz"
    mask_path = mask_dir / f"section-{sid:03d}__fixed_global_he_mask.npz"
    np.savez_compressed(
        transform_path,
        matrix_base_st_to_fixed_he=np.asarray(candidate["matrix_base_st_to_fixed_he"], dtype=np.float64),
        parameters=parameters,
        center_global=center,
        fixed_global_mask_origin=global_origin,
        fixed_global_mask_scale_um=np.asarray([raster_scale_um]),
        status=np.asarray(status),
        coordinate_frame=np.asarray(FRAME),
        he_is_fixed=np.asarray(True),
    )
    np.savez_compressed(
        coordinate_path,
        spatial_he_local=np.asarray(local_points, dtype=np.float64),
        st_global_base=np.asarray(base_global, dtype=np.float64),
        st_global_candidate=np.asarray(candidate["candidate_global"], dtype=np.float64),
        matrix_base_st_to_fixed_he=np.asarray(candidate["matrix_base_st_to_fixed_he"], dtype=np.float64),
        coordinate_frame=np.asarray(FRAME),
    )
    np.savez_compressed(
        mask_path,
        fixed_global_he_mask=global_mask,
        origin_um=global_origin,
        scale_um=np.asarray([raster_scale_um]),
        coordinate_frame=np.asarray(FRAME),
        source_h_e_mask=np.asarray(source["mask_path"]),
    )

    baseline_for_plot = {
        "base_global": base_global,
        "candidate_global": base_global,
        "parameters": np.zeros(5, dtype=float),
        **identity,
    }
    candidate_for_plot = {"base_global": base_global, "parameters": parameters, **candidate}
    overlay_path = overlay_dir / f"section-{sid:03d}__st_before_after_fixed_he_global_mask.png"
    H._plot_anchor(
        renderer,
        mapper,
        row,
        source,
        baseline_for_plot,
        candidate_for_plot,
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
        "optimizer_success": bool(result.success) if result is not None else True,
        "optimizer_message": str(result.message) if result is not None else "identity retained",
        "parameters": parameters.tolist(),
        "matrix_base_st_to_fixed_he": np.asarray(candidate["matrix_base_st_to_fixed_he"]).tolist(),
        "baseline": {key: float(identity[key]) for key in metric_keys},
        "candidate": {key: float(candidate[key]) for key in metric_keys},
        "improvement_inside_valid_fraction": float(candidate["inside_valid_fraction"] - identity["inside_valid_fraction"]),
        "improvement_quality_score": float(candidate["quality_score"] - identity["quality_score"]),
        "transform_path": str(transform_path.resolve()),
        "coordinate_path": str(coordinate_path.resolve()),
        "fixed_global_mask_path": str(mask_path.resolve()),
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
    parser.add_argument("--raster-scale-um", type=float, default=4.0)
    parser.add_argument("--source-stride", type=int, default=2)
    parser.add_argument("--dilation-iterations", type=int, default=1)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing candidate directory: {output_dir}")
    requested = tuple(sorted(set(int(v) for v in args.sections)))
    if not requested or any(v not in ANCHORS for v in requested):
        raise ValueError(f"sections must be a subset of {ANCHORS}")

    renderer = _load_renderer()
    table = pd.read_parquet(PREP / "he_sections.parquet").copy()
    table["section_id"] = pd.to_numeric(table["section_id"], errors="raise").astype(int)
    table = table[~table["section_id"].isin({54})].sort_values("section_id").reset_index(drop=True)
    mapper = renderer.FinalCoordinateMapper(table, device=args.device)
    local_by_section = _load_st_local_by_section(args.h5ad.resolve())
    rows = {int(row.section_id): row for row in table.itertuples(index=False)}
    missing = [sid for sid in requested if sid not in local_by_section or sid not in rows]
    if missing:
        raise ValueError(f"missing requested anchor sections: {missing}")

    records = []
    mask_hashes: dict[str, str] = {}
    image_hashes: dict[str, str] = {}
    overlay_paths = []
    for sid in requested:
        print(f"fitting v3 fixed-global-mask ST candidate for section {sid}", flush=True)
        source = renderer.BASE._load_source(rows[sid])
        mask_path = Path(source["mask_path"])
        image_path = Path(source["image_path"])
        mask_hashes[str(mask_path.resolve())] = sha256(mask_path)
        image_hashes[str(image_path.resolve())] = sha256(image_path)
        record = _fit_anchor(
            sid,
            local_by_section[sid],
            rows[sid],
            mapper,
            source,
            renderer,
            output_dir,
            max_opt_points=int(args.max_opt_points),
            max_plot_points=int(args.max_plot_points),
            seed=int(args.seed),
            raster_scale_um=float(args.raster_scale_um),
            source_stride=int(args.source_stride),
            dilation_iterations=int(args.dilation_iterations),
        )
        records.append(record)
        overlay_paths.append(Path(record["overlay_path"]))
        print(json.dumps(record, ensure_ascii=False), flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "status": "st_only_residual_candidate_not_published",
        "coordinate_frame": FRAME,
        "h5ad_source": str(args.h5ad.resolve()),
        "st_input_obsm": "spatial_he_local",
        "st_mapping": "spatial_he_local -> fixed final H&E forward mapper -> ST-only global residual affine",
        "objective_frame": "fixed global H&E mask rasterized by forward mapping H&E mask pixels",
        "raster_scale_um": float(args.raster_scale_um),
        "source_stride": int(args.source_stride),
        "dilation_iterations": int(args.dilation_iterations),
        "h_e_policy": "H&E images, masks, feature stores, and H&E transform chain were not modified",
        "published_to_h5ad": False,
        "h_e_masks_sha256_before": mask_hashes,
        "h_e_images_sha256_before": image_hashes,
        "records": records,
    }
    (output_dir / "residual_fit_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    H._contact_sheet(overlay_paths, output_dir / "st_before_after_contact_sheet.png")
    manifest = {
        "status": "candidate_only",
        "coordinate_frame": FRAME,
        "h5ad_source": str(args.h5ad.resolve()),
        "st_input_obsm": "spatial_he_local",
        "objective": "fixed global H&E mask; only ST global coordinates are adjusted",
        "h_e_is_fixed": True,
        "published_to_h5ad": False,
        "output": str(output_dir.resolve()),
        "metrics": str((output_dir / "residual_fit_metrics.json").resolve()),
        "contact_sheet": str((output_dir / "st_before_after_contact_sheet.png").resolve()),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# 9957/g0 ST-to-fixed-H&E residual candidate v3\n\n"
        "H&E is fixed. The H&E mask is forward-rasterized once into the final global frame, "
        "then only ST anchor coordinates receive a bounded residual affine. This is a review-only "
        "candidate and is not embedded into the H5AD.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
