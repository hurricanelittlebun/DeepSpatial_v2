"""Render real H&E red/green overlays with the complete v10 transform chain.

The section table's ``matrix`` is only the coarse affine component.  For
9957/g0, sections >=55 also pass through the saved horizontal-flip -> STalign
non-rigid -> CCW90 chain and then the v10 feature-support residual affine.
This renderer samples the original H&E RGB image through the inverse of that
complete chain, so the output is an image overlay rather than a feature-grid
or boundary-point plot.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
V2_SRC = ROOT.parent / "DeepSpatial_v2" / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(V2_SRC) not in sys.path:
    sys.path.insert(0, str(V2_SRC))

GROUP = ROOT / "data" / "9957_g0"
PREP = GROUP / "reconstruction_prep_v10_feature_support_residual"
TABLE_PATH = PREP / "he_sections.parquet"
V9_ROOT = ROOT.parent / "DeepSpatial_v2" / "outputs" / "registration_9957_pairfix_v9_mask_from_v2_total_v7_affine"
V9_RASTER_DIR = V9_ROOT / "rasters" / "9957" / "g0"
V9_TRANSFORM_DIR = V9_ROOT / "transforms" / "9957" / "g0"
V6_TRANSFORM_DIR = GROUP / "reconstruction_prep_v6_final_manual_alignment" / "registration_transforms_final_manual_alignment_v6"
DIRECT_ROOT = PREP / "stalign_final_53_55" / "direct_v3"
RESIDUAL_PATH = GROUP / "feature_support_stalign_53_55_v3" / "pair-53-55" / "feature_support_residual_affine.npz"
OUTPUT_ROOT = GROUP / "qc" / "he_alignment_all_sections_v7_feature_support_residual" / "he_redgreen_overlays_full_chain_v2"


def _load_materializer():
    path = ROOT / "scripts" / "materialize_9957_latest_alignment_cutoff91.py"
    spec = importlib.util.spec_from_file_location("_deep_latest_alignment_materializer", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load full alignment chain from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _apply_affine(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(points, dtype=float)
    flat = value.reshape(-1, 2)
    homogeneous = np.column_stack([flat, np.ones(len(flat), dtype=float)])
    return (homogeneous @ np.asarray(matrix, dtype=float).T)[:, :2].reshape(value.shape)


def _load_matrix(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        matrix = np.asarray(payload["matrix"], dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"invalid affine matrix: {path}")
    return matrix


def _load_residual() -> np.ndarray:
    with np.load(RESIDUAL_PATH, allow_pickle=False) as payload:
        matrix = np.asarray(payload["affine"], dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"invalid residual affine: {RESIDUAL_PATH}")
    return matrix


def _load_chain(device: str):
    materializer = _load_materializer()
    edge, summary = materializer._load_edge(
        DIRECT_ROOT,
        summary_name="direct_stalign_summary.json",
        affine_name="direct_stalign_affine.npz",
        forward_name="direct_stalign_forward_map.npz",
        reverse_name="direct_stalign_reverse_map.npz",
        status="accepted_direct_stalign_v3_horizontal_flip",
        device=device,
    )
    if int(summary["fixed_section"]) != 53 or int(summary["moving_section"]) != 55:
        raise ValueError("direct STalign artifact is not the 53 -> 55 pair")
    return materializer.LatestAlignmentChain(edge, None, device=device)


def _load_source(row: Any) -> dict[str, Any]:
    image_path = Path(str(row.source_image_path))
    mask_path = Path(str(row.source_mask_path))
    if not image_path.is_file() or not mask_path.is_file():
        raise FileNotFoundError(f"missing source H&E or mask for section {row.section_id}")
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    with Image.open(mask_path) as mask_image:
        mask = np.asarray(mask_image.convert("L"), dtype=np.uint8) > 0
    if rgb.shape[:2] != mask.shape:
        raise ValueError(f"H&E/mask shape mismatch for section {row.section_id}")
    sx = float(row.analysis_pixel_size_x_um)
    sy = float(row.analysis_pixel_size_y_um)
    if sx <= 0 or sy <= 0:
        raise ValueError(f"invalid source pixel size for section {row.section_id}")
    local_origin = np.array([float(row.bbox_x0) * sx, float(row.bbox_y0) * sy], dtype=float)
    v9_raster_paths = sorted(V9_RASTER_DIR.glob(f"section-{int(row.section_id)}-*.npz"))
    if len(v9_raster_paths) != 1:
        raise FileNotFoundError(f"expected one v9 raster for section {row.section_id}")
    with np.load(v9_raster_paths[0], allow_pickle=False) as payload:
        preorientation = np.asarray(payload["preorientation_matrix"], dtype=float)
        forced_flip = bool(np.asarray(payload.get("forced_flip", [0])).reshape(-1)[0])
    v9_path = next(V9_TRANSFORM_DIR.glob(f"section-{int(row.section_id)}-*.npz"), None)
    if v9_path is None:
        raise FileNotFoundError(f"missing v9 transform for section {row.section_id}")
    v6_path = V6_TRANSFORM_DIR / (
        f"section-{int(row.section_id)}-9957__g0__section-{int(row.section_id)}.npz"
    )
    if not v6_path.is_file():
        raise FileNotFoundError(v6_path)
    v9_matrix = _load_matrix(v9_path)
    v6_matrix = _load_matrix(v6_path)
    # The source raster is expressed in the v9 pre-registration frame.  The
    # direct STalign experiment first bridged that frame into v6; using the
    # v6 matrix directly here would double-apply the serial-registration pose.
    bridge_v9_to_v6 = v6_matrix @ np.linalg.inv(v9_matrix)
    return {
        "section_id": int(row.section_id),
        "z_um": float(row.z_um),
        "rgb": rgb,
        "mask": mask,
        "pixel_size_um": np.array([sx, sy], dtype=float),
        "local_origin_um": local_origin,
        "preorientation_matrix": preorientation,
        "forced_flip": forced_flip,
        "v9_matrix": v9_matrix,
        "v6_matrix": v6_matrix,
        "bridge_v9_to_v6": bridge_v9_to_v6,
        "v9_raster_path": str(v9_raster_paths[0].resolve()),
        "v9_transform_path": str(v9_path.resolve()),
        "source_image_path": str(image_path.resolve()),
        "source_mask_path": str(mask_path.resolve()),
    }


def _global_from_local(points: np.ndarray, section: dict[str, Any], chain, residual: np.ndarray) -> np.ndarray:
    value = _apply_affine(points, section["preorientation_matrix"])
    value = _apply_affine(value, section["bridge_v9_to_v6"])
    if section["section_id"] >= 55:
        value = chain.forward_chunked(value, batch_size=32768)
        value = _apply_affine(value, residual)
    return value


def _local_from_global(points: np.ndarray, section: dict[str, Any], chain, residual: np.ndarray) -> np.ndarray:
    value = np.asarray(points, dtype=float)
    if section["section_id"] >= 55:
        value = _apply_affine(value, np.linalg.inv(residual))
        value = chain.inverse_chunked(value, batch_size=32768)
    value = _apply_affine(value, np.linalg.inv(section["bridge_v9_to_v6"]))
    return _apply_affine(value, np.linalg.inv(section["preorientation_matrix"]))


def _foreground_bbox(section: dict[str, Any], chain, residual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = section["mask"]
    from scipy.ndimage import binary_erosion

    boundary = mask & ~binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)
    rows, cols = np.nonzero(boundary)
    if len(rows) == 0:
        rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        height, width = mask.shape
        rows = np.array([0, 0, height - 1, height - 1], dtype=int)
        cols = np.array([0, width - 1, 0, width - 1], dtype=int)
    step = max(1, len(rows) // 20000)
    rows = rows[::step]
    cols = cols[::step]
    sx, sy = section["pixel_size_um"]
    local = np.column_stack(
        [section["local_origin_um"][0] + cols * sx, section["local_origin_um"][1] + rows * sy]
    )
    mapped = _global_from_local(local, section, chain, residual)
    return mapped.min(axis=0), mapped.max(axis=0)


def _sample_density(
    section: dict[str, Any],
    origin: np.ndarray,
    scale_um: float,
    shape: tuple[int, int],
    chain,
    residual: np.ndarray,
    chunk_size: int = 131072,
) -> np.ndarray:
    from scipy.ndimage import map_coordinates

    height, width = shape
    rgb = section["rgb"].astype(np.float32) / 255.0
    mask = section["mask"].astype(np.float32)
    sx, sy = section["pixel_size_um"]
    output = np.zeros(height * width, dtype=np.float32)
    for start in range(0, height * width, chunk_size):
        stop = min(start + chunk_size, height * width)
        indices = np.arange(start, stop, dtype=np.int64)
        rows = indices // width
        cols = indices % width
        global_points = np.column_stack(
            [origin[0] + cols * scale_um, origin[1] + rows * scale_um]
        )
        local = _local_from_global(global_points, section, chain, residual)
        source_cols = (local[:, 0] - section["local_origin_um"][0]) / sx
        source_rows = (local[:, 1] - section["local_origin_um"][1]) / sy
        inside = (
            (source_cols >= 0.0)
            & (source_cols <= rgb.shape[1] - 1)
            & (source_rows >= 0.0)
            & (source_rows <= rgb.shape[0] - 1)
        )
        rows_clip = np.clip(source_rows, 0.0, rgb.shape[0] - 1)
        cols_clip = np.clip(source_cols, 0.0, rgb.shape[1] - 1)
        sampled_rgb = np.stack(
            [
                map_coordinates(
                    rgb[..., channel],
                    [rows_clip, cols_clip],
                    order=1,
                    mode="constant",
                    cval=1.0,
                    prefilter=False,
                )
                for channel in range(3)
            ],
            axis=1,
        )
        sampled_mask = map_coordinates(
            mask,
            [rows_clip, cols_clip],
            order=0,
            mode="constant",
            cval=0.0,
            prefilter=False,
        ) >= 0.5
        density = np.clip(1.0 - sampled_rgb.mean(axis=1), 0.0, 1.0)
        density *= (inside & sampled_mask)
        output[start:stop] = density.astype(np.float32, copy=False)
    return output.reshape(height, width)


def _enhance_density(value: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = np.asarray(value)[np.asarray(mask, dtype=bool)]
    valid = valid[np.isfinite(valid) & (valid > 0)]
    if len(valid) < 10:
        return np.asarray(value, dtype=np.float32)
    lo, hi = np.percentile(valid, [2.0, 99.5])
    if hi <= lo + 1e-8:
        return np.asarray(value, dtype=np.float32)
    return np.clip((np.asarray(value, dtype=np.float32) - lo) / (hi - lo), 0.0, 1.0) ** 0.65


def _pair_overlay(first: dict[str, Any], second: dict[str, Any], chain, residual: np.ndarray, scale_um: float) -> tuple[np.ndarray, dict]:
    first_min, first_max = _foreground_bbox(first, chain, residual)
    second_min, second_max = _foreground_bbox(second, chain, residual)
    origin = np.floor(np.minimum(first_min, second_min) / scale_um) * scale_um
    maximum = np.ceil(np.maximum(first_max, second_max) / scale_um) * scale_um
    width = int(round((maximum[0] - origin[0]) / scale_um)) + 1
    height = int(round((maximum[1] - origin[1]) / scale_um)) + 1
    if max(height, width) > 6000:
        raise ValueError(f"full-chain H&E canvas too large at {scale_um} um/px: {(height, width)}")
    first_density = _sample_density(first, origin, scale_um, (height, width), chain, residual)
    second_density = _sample_density(second, origin, scale_um, (height, width), chain, residual)
    first_density = _enhance_density(first_density, first_density > 0)
    second_density = _enhance_density(second_density, second_density > 0)
    composite = np.zeros((height, width, 3), dtype=np.float32)
    composite[..., 0] = np.clip(first_density * 1.25, 0.0, 1.0)
    composite[..., 1] = np.clip(second_density * 1.25, 0.0, 1.0)
    composite[..., 2] = 0.015 * np.maximum(first_density, second_density)
    image = np.clip(np.rint(composite * 255.0), 0, 255).astype(np.uint8)
    metadata = {
        "first_section": int(first["section_id"]),
        "second_section": int(second["section_id"]),
        "first_z_um": float(first["z_um"]),
        "second_z_um": float(second["z_um"]),
        "origin_um": origin.tolist(),
        "scale_um_per_pixel": float(scale_um),
        "shape_yx": [height, width],
        "transform_semantics": "raw H&E source -> v6 affine -> full v9 STalign chain (sections >=55) -> v10 residual affine (sections >=55)",
    }
    return image, metadata


def _save_pair(first: dict[str, Any], second: dict[str, Any], chain, residual: np.ndarray, path: Path, scale_um: float) -> dict:
    image, metadata = _pair_overlay(first, second, chain, residual, scale_um)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(path)
    metadata["path"] = str(path.resolve())
    return metadata


def render(pair_ids: list[tuple[int, int]], *, device: str, scale_um: float) -> dict:
    output = OUTPUT_ROOT / "key_pairs"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    table = pd.read_parquet(TABLE_PATH).sort_values("z_um", kind="stable").reset_index(drop=True)
    rows = {int(row.section_id): row for row in table.itertuples(index=False)}
    missing = sorted({sid for pair in pair_ids for sid in pair if sid not in rows})
    if missing:
        raise KeyError(f"sections missing from v10 table: {missing}")
    chain = _load_chain(device)
    residual = _load_residual()
    sections = {sid: _load_source(rows[sid]) for sid in sorted({sid for pair in pair_ids for sid in pair})}
    records = []
    for first_sid, second_sid in pair_ids:
        path = output / f"pair-{first_sid:03d}-{second_sid:03d}__he_redgreen_full_chain.png"
        records.append(_save_pair(sections[first_sid], sections[second_sid], chain, residual, path, scale_um))
    manifest = {
        "status": "complete",
        "coordinate_frame": "9957__g0_registered_feature_support_residual_v10",
        "source_table": str(TABLE_PATH.resolve()),
        "device": device,
        "overlay_semantics": "Real H&E RGB source images sampled through the complete v10 transform chain; red=first section optical density, green=second section optical density, yellow=overlap.",
        "pairs": records,
        "residual_affine": residual.tolist(),
        "full_chain": "v6 affine -> horizontal canvas flip -> direct STalign nonlinear map -> CCW90 -> v10 residual for section >=55",
    }
    manifest_path = OUTPUT_ROOT / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scale-um", type=float, default=2.0)
    args = parser.parse_args()
    result = render([(51, 53), (53, 55)], device=args.device, scale_um=float(args.scale_um))
    print(json.dumps({"status": result["status"], "pairs": result["pairs"], "manifest": str((OUTPUT_ROOT / 'manifest.json').resolve())}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
