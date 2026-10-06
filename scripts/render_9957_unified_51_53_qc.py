"""Render raw H&E red/green QC for the unified 9957/g0 v11 frame."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import binary_erosion, map_coordinates

ROOT = Path(__file__).resolve().parents[1]
GROUP = ROOT / "data" / "9957_g0"
OUTPUT_ROOT = GROUP / "qc" / "he_alignment_all_sections_v11_unified_51_53"
V11_PREP = GROUP / "reconstruction_prep_v11_unified_51_53"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_old_renderer():
    return _load_module(
        ROOT / "scripts" / "render_9957_v10_he_redgreen_full_chain.py",
        "_old_9957_raw_he_renderer",
    )


def _load_materializer():
    return _load_module(
        ROOT / "scripts" / "materialize_9957_unified_51_53_v11.py",
        "_unified_9957_materializer_for_qc",
    )


def _apply_affine(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(points, dtype=float)
    homogeneous = np.column_stack([value, np.ones(len(value), dtype=float)])
    return (homogeneous @ np.asarray(matrix, dtype=float).T)[:, :2]


def _forward(chain, points: np.ndarray, section: int) -> np.ndarray:
    value = np.asarray(points, dtype=float)
    output = np.empty_like(value)
    for start in range(0, len(value), 32768):
        stop = min(start + 32768, len(value))
        output[start:stop] = chain.forward_points(value[start:stop], int(section))
    return output


def _inverse(chain, points: np.ndarray, section: int) -> np.ndarray:
    value = np.asarray(points, dtype=float)
    output = np.empty_like(value)
    for start in range(0, len(value), 32768):
        stop = min(start + 32768, len(value))
        output[start:stop] = chain.inverse_points(value[start:stop], int(section))
    return output


def _global_from_local(points: np.ndarray, section: dict, chain) -> np.ndarray:
    value = _apply_affine(points, section["preorientation_matrix"])
    value = _apply_affine(value, section["bridge_v9_to_v6"])
    if int(section["section_id"]) >= 53:
        value = _forward(chain, value, int(section["section_id"]))
    return value


def _local_from_global(points: np.ndarray, section: dict, chain) -> np.ndarray:
    value = np.asarray(points, dtype=float)
    if int(section["section_id"]) >= 53:
        value = _inverse(chain, value, int(section["section_id"]))
    value = _apply_affine(value, np.linalg.inv(section["bridge_v9_to_v6"]))
    return _apply_affine(value, np.linalg.inv(section["preorientation_matrix"]))


def _foreground_bbox(section: dict, chain) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(section["mask"], dtype=bool)
    boundary = mask & ~binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)
    rows, cols = np.nonzero(boundary)
    if len(rows) == 0:
        rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        raise ValueError(f"empty H&E mask for section {section['section_id']}")
    step = max(1, len(rows) // 20000)
    rows = rows[::step]
    cols = cols[::step]
    sx, sy = section["pixel_size_um"]
    local = np.column_stack(
        [
            section["local_origin_um"][0] + cols * sx,
            section["local_origin_um"][1] + rows * sy,
        ]
    )
    mapped = _global_from_local(local, section, chain)
    return mapped.min(axis=0), mapped.max(axis=0)


def _sample_density(
    section: dict,
    origin: np.ndarray,
    scale_um: float,
    shape: tuple[int, int],
    chain,
    chunk_size: int = 131072,
) -> np.ndarray:
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
        local = _local_from_global(global_points, section, chain)
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
        density *= inside & sampled_mask
        output[start:stop] = density.astype(np.float32, copy=False)
    return output.reshape(height, width)


def _enhance(value: np.ndarray) -> np.ndarray:
    valid = np.asarray(value, dtype=float)
    valid = valid[np.isfinite(valid) & (valid > 0)]
    if len(valid) < 10:
        return np.asarray(value, dtype=np.float32)
    lo, hi = np.percentile(valid, [2.0, 99.5])
    if hi <= lo + 1e-8:
        return np.asarray(value, dtype=np.float32)
    return np.clip((np.asarray(value, dtype=np.float32) - lo) / (hi - lo), 0.0, 1.0) ** 0.65


def _pair_overlay(first: dict, second: dict, chain, scale_um: float) -> tuple[np.ndarray, dict]:
    first_min, first_max = _foreground_bbox(first, chain)
    second_min, second_max = _foreground_bbox(second, chain)
    origin = np.floor(np.minimum(first_min, second_min) / scale_um) * scale_um
    maximum = np.ceil(np.maximum(first_max, second_max) / scale_um) * scale_um
    width = int(round((maximum[0] - origin[0]) / scale_um)) + 1
    height = int(round((maximum[1] - origin[1]) / scale_um)) + 1
    if max(height, width) > 6000:
        raise ValueError(f"QC canvas too large: {(height, width)}")
    first_density = _enhance(_sample_density(first, origin, scale_um, (height, width), chain))
    second_density = _enhance(_sample_density(second, origin, scale_um, (height, width), chain))
    composite = np.zeros((height, width, 3), dtype=np.float32)
    composite[..., 0] = np.clip(first_density * 1.25, 0.0, 1.0)
    composite[..., 1] = np.clip(second_density * 1.25, 0.0, 1.0)
    composite[..., 2] = 0.015 * np.maximum(first_density, second_density)
    image = np.clip(np.rint(composite * 255.0), 0, 255).astype(np.uint8)
    return image, {
        "first_section": int(first["section_id"]),
        "second_section": int(second["section_id"]),
        "first_z_um": float(first["z_um"]),
        "second_z_um": float(second["z_um"]),
        "origin_um": origin.tolist(),
        "scale_um_per_pixel": float(scale_um),
        "shape_yx": [height, width],
        "transform_semantics": "raw H&E -> v9 preorientation -> v9-to-v6 bridge -> v9 postrotate chain for section>=53 -> new 51-to-53 STalign edge",
    }


def render(*, device: str, scale_um: float) -> dict:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT_ROOT}")
    old_renderer = _load_old_renderer()
    materializer = _load_materializer()
    chain, chain_inputs, _, _ = materializer._load_chain(device)
    table_path = V11_PREP / "he_sections.parquet"
    table = pd.read_parquet(table_path).sort_values("z_um", kind="stable").reset_index(drop=True)
    rows = {int(row.section_id): row for row in table.itertuples(index=False)}
    pairs = [(51, 53), (53, 55)]
    sections = {sid: old_renderer._load_source(rows[sid]) for pair in pairs for sid in pair}
    output_dir = OUTPUT_ROOT / "he_redgreen_overlays" / "key_pairs"
    records = []
    for first_sid, second_sid in pairs:
        image, metadata = _pair_overlay(
            sections[first_sid], sections[second_sid], chain, float(scale_um)
        )
        path = output_dir / f"pair-{first_sid:03d}-{second_sid:03d}__he_redgreen_unified_v11.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image, mode="RGB").save(path)
        metadata["path"] = str(path.resolve())
        records.append(metadata)
    manifest = {
        "status": "complete",
        "release": "9957_unified_51_53_v11_he_redgreen_qc",
        "coordinate_frame": materializer.OUTPUT_FRAME,
        "source_table": str(table_path.resolve()),
        "pairs": records,
        "chain_inputs": chain_inputs,
        "overlay_semantics": "red=first raw H&E optical density; green=second raw H&E optical density; yellow=overlap",
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scale-um", type=float, default=2.0)
    args = parser.parse_args()
    result = render(device=args.device, scale_um=float(args.scale_um))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
