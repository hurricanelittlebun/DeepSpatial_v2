"""Render actual registered H&E red/green overlays for 9957/g0 v10."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
GROUP = ROOT / "data" / "9957_g0"
PREP = GROUP / "reconstruction_prep_v10_feature_support_residual"
TABLE_PATH = PREP / "he_sections.parquet"
OUTPUT_ROOT = GROUP / "qc" / "he_alignment_all_sections_v7_feature_support_residual" / "he_redgreen_overlays_v2"
DEFAULT_SCALE_UM = 4.0
KEY_SCALE_UM = 2.0


def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(points, dtype=float)
    homogeneous = np.column_stack([value, np.ones(len(value), dtype=float)])
    return (homogeneous @ np.asarray(matrix, dtype=float).T)[:, :2]


def _load_rgb_and_mask(row) -> tuple[np.ndarray, np.ndarray]:
    image_path = Path(str(row.source_image_path))
    mask_path = Path(str(row.source_mask_path))
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    with Image.open(mask_path) as mask_image:
        mask = np.asarray(mask_image.convert("L"), dtype=np.uint8) > 0
    if rgb.shape[:2] != mask.shape:
        raise ValueError(f"H&E/mask shape mismatch for section {row.section_id}: {rgb.shape} vs {mask.shape}")
    return rgb, mask


def _load_section(row) -> dict:
    rgb, mask = _load_rgb_and_mask(row)
    transform_path = Path(str(row.section_transform_path))
    with np.load(transform_path, allow_pickle=False) as payload:
        matrix = np.asarray(payload["matrix"], dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"invalid matrix for section {row.section_id}: {transform_path}")
    sx = float(row.analysis_pixel_size_x_um)
    sy = float(row.analysis_pixel_size_y_um)
    rows, cols = np.nonzero(mask)
    if len(rows):
        local = np.column_stack([cols * sx, rows * sy])
        global_points = _transform_points(local, matrix)
        bbox_min = global_points.min(axis=0)
        bbox_max = global_points.max(axis=0)
    else:
        corners = np.array([[0, 0], [(rgb.shape[1] - 1) * sx, 0], [0, (rgb.shape[0] - 1) * sy], [(rgb.shape[1] - 1) * sx, (rgb.shape[0] - 1) * sy]], dtype=float)
        global_points = _transform_points(corners, matrix)
        bbox_min = global_points.min(axis=0)
        bbox_max = global_points.max(axis=0)
    return {
        "section_id": int(row.section_id),
        "z_um": float(row.z_um),
        "rgb": rgb,
        "mask": mask,
        "matrix": matrix,
        "pixel_size_um": (sx, sy),
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
    }


def _sample_optical_density(section: dict, origin: np.ndarray, scale_um: float, shape: tuple[int, int]) -> np.ndarray:
    from scipy.ndimage import map_coordinates

    height, width = shape
    yy, xx = np.mgrid[:height, :width]
    global_points = np.column_stack([
        origin[0] + xx.reshape(-1) * scale_um,
        origin[1] + yy.reshape(-1) * scale_um,
    ])
    local = _transform_points(global_points, np.linalg.inv(section["matrix"]))
    sx, sy = section["pixel_size_um"]
    columns = local[:, 0] / sx
    rows = local[:, 1] / sy
    rgb = section["rgb"].astype(np.float32) / 255.0
    tissue = section["mask"].astype(np.float32)
    inside = (
        (columns >= 0)
        & (columns <= rgb.shape[1] - 1)
        & (rows >= 0)
        & (rows <= rgb.shape[0] - 1)
    )
    rows_clip = np.clip(rows, 0, rgb.shape[0] - 1)
    cols_clip = np.clip(columns, 0, rgb.shape[1] - 1)
    # Optical density retains H&E texture while providing an unambiguous
    # red/green comparison.  Dark nuclei and tissue are stronger signals.
    sampled = np.stack([
        map_coordinates(rgb[..., channel], [rows_clip, cols_clip], order=1, mode="constant", cval=1.0, prefilter=False)
        for channel in range(3)
    ], axis=1)
    sampled_mask = map_coordinates(tissue, [rows_clip, cols_clip], order=0, mode="constant", cval=0.0, prefilter=False) >= 0.5
    density = np.clip(1.0 - sampled.mean(axis=1), 0.0, 1.0)
    density *= (inside & sampled_mask)
    return density.reshape(height, width)


def _pair_overlay(first: dict, second: dict, scale_um: float) -> tuple[np.ndarray, dict]:
    origin = np.floor(np.minimum(first["bbox_min"], second["bbox_min"]) / scale_um) * scale_um
    maximum = np.ceil(np.maximum(first["bbox_max"], second["bbox_max"]) / scale_um) * scale_um
    width = int(round((maximum[0] - origin[0]) / scale_um)) + 1
    height = int(round((maximum[1] - origin[1]) / scale_um)) + 1
    if max(height, width) > 5000:
        raise ValueError(f"overlay canvas is too large at {scale_um} um/px: {(height, width)}")
    first_density = _sample_optical_density(first, origin, scale_um, (height, width))
    second_density = _sample_optical_density(second, origin, scale_um, (height, width))
    # Black background; red=first H&E, green=second H&E, yellow=overlap.
    composite = np.zeros((height, width, 3), dtype=np.float32)
    composite[..., 0] = np.clip(first_density * 1.35, 0.0, 1.0)
    composite[..., 1] = np.clip(second_density * 1.35, 0.0, 1.0)
    composite[..., 2] = 0.02 * np.maximum(first_density, second_density)
    image = np.clip(np.rint(composite * 255.0), 0, 255).astype(np.uint8)
    return image, {
        "origin_um": origin.tolist(),
        "scale_um_per_pixel": float(scale_um),
        "shape_yx": [height, width],
        "first_section": first["section_id"],
        "second_section": second["section_id"],
        "first_z_um": first["z_um"],
        "second_z_um": second["z_um"],
    }


def _save_pair(first: dict, second: dict, path: Path, scale_um: float) -> dict:
    image, metadata = _pair_overlay(first, second, scale_um)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    Image.fromarray(image, mode="RGB").save(path)
    metadata["path"] = str(path.resolve())
    return metadata


def _contact_sheet(paths: list[Path], output: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    if output.exists():
        raise FileExistsError(output)
    ncols = 5
    nrows = int(np.ceil(len(paths) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, max(3.5 * nrows, 4.0)), dpi=150, squeeze=False)
    for ax, path in zip(axes.flat, paths):
        ax.imshow(np.asarray(Image.open(path)))
        ax.set_title(path.stem.replace("__he_redgreen", ""), fontsize=8)
        ax.axis("off")
    for ax in axes.flat[len(paths):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def render() -> dict:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite existing overlay directory: {OUTPUT_ROOT}")
    table = pd.read_parquet(TABLE_PATH).sort_values("z_um", kind="stable").reset_index(drop=True)
    sections = [_load_section(row) for row in table.itertuples(index=False)]
    by_id = {item["section_id"]: item for item in sections}
    pair_dir = OUTPUT_ROOT / "adjacent_pairs"
    pair_records = []
    pair_paths = []
    for first, second in zip(sections[:-1], sections[1:]):
        path = pair_dir / f"pair-{first['section_id']:03d}-{second['section_id']:03d}__he_redgreen.png"
        pair_records.append(_save_pair(first, second, path, DEFAULT_SCALE_UM))
        pair_paths.append(path)
    key_dir = OUTPUT_ROOT / "key_pairs"
    key_records = []
    for left, right in ((51, 53), (53, 55)):
        path = key_dir / f"pair-{left:03d}-{right:03d}__he_redgreen_highres.png"
        key_records.append(_save_pair(by_id[left], by_id[right], path, KEY_SCALE_UM))
    _contact_sheet(pair_paths, OUTPUT_ROOT / "adjacent_pairs_contact_sheet.png", "9957/g0 v10 actual H&E red/green adjacent-section overlays")
    manifest = {
        "status": "complete",
        "coordinate_frame": "9957__g0_registered_feature_support_residual_v10",
        "source_table": str(TABLE_PATH.resolve()),
        "overlay_semantics": "Actual H&E RGB images sampled into a common registered physical canvas; red=first section optical density, green=second section optical density, yellow=overlap.",
        "default_scale_um_per_pixel": DEFAULT_SCALE_UM,
        "key_scale_um_per_pixel": KEY_SCALE_UM,
        "adjacent_pairs": pair_records,
        "key_pairs": key_records,
        "contact_sheet": str((OUTPUT_ROOT / "adjacent_pairs_contact_sheet.png").resolve()),
    }
    (OUTPUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    result = render()
    print(json.dumps({
        "status": result["status"],
        "key_pairs": [item["path"] for item in result["key_pairs"]],
        "contact_sheet": result["contact_sheet"],
        "adjacent_pair_count": len(result["adjacent_pairs"]),
    }, indent=2, ensure_ascii=False))
