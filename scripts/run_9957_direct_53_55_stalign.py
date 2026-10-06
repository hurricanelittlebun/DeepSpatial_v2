"""Run a genuine direct H&E STalign edge from 9957/g0 section 53 to 55.

Section 54 is intentionally not used.  The v6 section placement is used only
as the coarse initialization: raw H&E texture is sampled into the v6 physical
frame for sections 53 and 55, and official STalign LDDMM is run on those two
surviving sections directly.  The resulting edge is saved separately and can
be promoted by the section-54 removal materializer.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import map_coordinates

PROJECT_ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = PROJECT_ROOT.parent / "DeepSpatial_v2"
V2_SRC = V2_ROOT / "src"
V2_SCRIPTS = V2_ROOT / "scripts"
for path in (PROJECT_ROOT, V2_SRC, V2_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from deepspatial_v2.serial_registration.he_image_raster import morphology_channels  # noqa: E402
from deepspatial_v2.serial_registration.partial_overlap_qc import (  # noqa: E402
    compute_partial_overlap_metrics,
)
from deepspatial_v2.serial_registration.stalign_backend import (  # noqa: E402
    run_stalign_edge,
    transform_image_source_to_target,
    transform_mask_source_to_target,
)
from deepspatial_v2.serial_registration.stalign_types import (  # noqa: E402
    RasterizedSection,
    StalignConfig,
)


DEFAULT_GROUP = PROJECT_ROOT / "data" / "9957_g0"
DEFAULT_V9_ROOT = V2_ROOT / "outputs" / "registration_9957_pairfix_v9_mask_from_v2_total_v7_affine"
DEFAULT_HE_MANIFEST = V2_ROOT / "outputs/sequence/manifests/he_combined_object_manifest.parquet"
DEFAULT_V6_TRANSFORMS = DEFAULT_GROUP / "reconstruction_prep_v6_final_manual_alignment/registration_transforms_final_manual_alignment_v6"
DEFAULT_OUTPUT = DEFAULT_GROUP / "stalign_direct_53_55_v1"


def _apply_affine(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(points, dtype=float)
    flat = value.reshape(-1, 2)
    homogeneous = np.concatenate([flat, np.ones((len(flat), 1))], axis=1)
    return (homogeneous @ np.asarray(matrix, dtype=float).T)[:, :2].reshape(value.shape)


def _section_raster_path(v9_root: Path, section: int) -> Path:
    paths = sorted((v9_root / "rasters/9957/g0").glob(f"section-{section}-*.npz"))
    if len(paths) != 1:
        raise FileNotFoundError(f"expected one v9 raster for section {section}, got {paths}")
    return paths[0]


def _load_v9_source_raster(v9_root: Path, section: int) -> dict[str, Any]:
    path = _section_raster_path(v9_root, section)
    with np.load(path, allow_pickle=True) as data:
        grid = np.asarray(data["grid_um"], dtype=float)
        mask = np.asarray(data["mask"], dtype=bool)
        dx = float(np.asarray(data["dx_um"]).reshape(-1)[0])
        forced_flip = bool(np.asarray(data.get("forced_flip", [0])).reshape(-1)[0])
    if grid.shape[0] != 2 or grid.shape[1:] != mask.shape:
        raise ValueError(f"invalid raster {path}")
    return {"path": path, "grid": grid, "mask": mask, "dx": dx, "forced_flip": forced_flip}


def _load_source_h_and_e(
    v9_root: Path, section: int, he_manifest: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Sample the raw segmented H&E crop on its v9 physical raster."""

    # Reuse the established source sampler so crop origin, MPP and reviewed
    # horizontal flips have exactly the same semantics as prior v9 runs.
    candidate = importlib.import_module("run_9957_partial_overlap_stalign_he_candidate")
    source = candidate._load_source_he(v9_root, section, he_manifest)
    grid = np.asarray(source.grid_um, dtype=float)
    return (
        np.asarray(source.image, dtype=np.float32),
        np.asarray(source.mask, dtype=bool),
        {
            "source_raster_path": str(_section_raster_path(v9_root, section).resolve()),
            "source_grid_shape": list(grid.shape),
            "source_grid_origin_um": grid[:, 0, 0].tolist(),
            "source_grid_max_um": grid[:, -1, -1].tolist(),
            "source_dx_um": float(source.dx_um),
            "forced_flip": bool(source.forced_flip),
        },
    )


def _load_matrix(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        value = np.asarray(data["matrix"], dtype=float)
    if value.shape != (3, 3) or not np.isfinite(value).all():
        raise ValueError(f"invalid transform matrix: {path}")
    return value


def _v9_transform_path(v9_root: Path, section: int) -> Path:
    paths = sorted((v9_root / "transforms/9957/g0").glob(f"section-{section}-*.npz"))
    if len(paths) != 1:
        raise FileNotFoundError(f"expected one v9 section transform for {section}, got {paths}")
    return paths[0]


def _v6_transform_path(v6_transform_dir: Path, section: int) -> Path:
    paths = sorted(v6_transform_dir.glob(f"section-{section}-*.npz"))
    if len(paths) != 1:
        raise FileNotFoundError(f"expected one v6 section transform for {section}, got {paths}")
    return paths[0]


def _v9_to_v6_bridge(v9_root: Path, v6_transform_dir: Path, section: int) -> dict[str, Any]:
    """Map v9 registered physical coordinates into the v6 final frame."""

    v9_path = _v9_transform_path(v9_root, section)
    v6_path = _v6_transform_path(v6_transform_dir, section)
    v9_matrix = _load_matrix(v9_path)
    v6_matrix = _load_matrix(v6_path)
    bridge = v6_matrix @ np.linalg.inv(v9_matrix)
    return {
        "v9_transform_path": str(v9_path.resolve()),
        "v6_transform_path": str(v6_path.resolve()),
        "v9_matrix": v9_matrix,
        "v6_matrix": v6_matrix,
        "bridge_v9_to_v6": bridge,
    }


def _canvas_geometry(
    sources: dict[int, dict[str, Any]], bridges: dict[int, np.ndarray], dx_um: float
) -> tuple[tuple[float, float], tuple[int, int], np.ndarray]:
    transformed = []
    for section, source in sources.items():
        transformed.append(_apply_affine(source["grid"].reshape(2, -1).T, bridges[section]))
    points = np.concatenate(transformed, axis=0)
    origin = np.floor(np.min(points, axis=0) / dx_um - 2.0) * dx_um
    maximum = np.ceil(np.max(points, axis=0) / dx_um + 2.0) * dx_um
    shape = tuple((np.rint((maximum - origin) / dx_um).astype(int) + 1).tolist())
    height, width = int(shape[1]), int(shape[0])
    x = origin[0] + dx_um * np.arange(width, dtype=float)
    y = origin[1] + dx_um * np.arange(height, dtype=float)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    grid = np.stack([xx, yy], axis=0)
    return (float(origin[0]), float(origin[1])), (height, width), grid


def _resample_to_canvas(
    image: np.ndarray,
    mask: np.ndarray,
    source_grid: np.ndarray,
    bridge: np.ndarray,
    canvas_grid: np.ndarray,
    dx_um: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a v9 raster into an axis-aligned v6 physical canvas."""

    inverse = np.linalg.inv(np.asarray(bridge, dtype=float))
    source_points = _apply_affine(canvas_grid.reshape(2, -1).T, inverse)
    source_x = (source_grid[0, 0, 0], source_grid[1, 0, 0])
    columns = (source_points[:, 0] - source_x[0]) / dx_um
    rows = (source_points[:, 1] - source_x[1]) / dx_um
    source_shape = mask.shape
    valid = (
        (columns >= 0.0)
        & (columns <= source_shape[1] - 1)
        & (rows >= 0.0)
        & (rows <= source_shape[0] - 1)
    )
    rows_clip = np.clip(rows, 0.0, source_shape[0] - 1)
    cols_clip = np.clip(columns, 0.0, source_shape[1] - 1)
    target_shape = canvas_grid.shape[1:]
    output = np.zeros((image.shape[0], *target_shape), dtype=np.float32)
    for channel in range(image.shape[0]):
        sampled = map_coordinates(
            image[channel], [rows_clip, cols_clip], order=1, mode="constant", cval=0.0
        )
        output[channel] = sampled.reshape(target_shape)
    sampled_mask = map_coordinates(
        mask.astype(np.float32), [rows_clip, cols_clip], order=0, mode="constant", cval=0.0
    ) >= 0.5
    valid_mask = (valid & sampled_mask).reshape(target_shape)
    output[:, ~valid_mask] = 0.0
    return output, valid_mask


def _morphology_to_rgb(channels: np.ndarray) -> np.ndarray:
    value = np.asarray(channels, dtype=float)
    rgb = np.moveaxis(1.0 - np.clip(value, 0.0, 1.0), 0, -1)
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)


def _apply_additional_moving_flip(
    image: np.ndarray, mask: np.ndarray, flip: str
) -> tuple[np.ndarray, np.ndarray]:
    """Apply an explicit additional orientation correction to section 55.

    The source loader already applies the reviewed ``forced_flip`` recorded in
    the v9 raster metadata.  This helper is deliberately separate so a new
    reviewed correction can be audited as an additional operation rather than
    silently changing the source orientation semantics.
    """

    image_axes = {
        "none": None,
        "horizontal": 2,
        "vertical": 1,
        "horizontal_vertical": (1, 2),
    }
    mask_axes = {
        "none": None,
        "horizontal": 1,
        "vertical": 0,
        "horizontal_vertical": (0, 1),
    }
    if flip not in image_axes:
        raise ValueError(f"unsupported moving flip: {flip}")
    image_axis = image_axes[flip]
    mask_axis = mask_axes[flip]
    if image_axis is None:
        return np.ascontiguousarray(image), np.ascontiguousarray(mask)
    return (
        np.ascontiguousarray(np.flip(image, axis=image_axis)),
        np.ascontiguousarray(np.flip(mask, axis=mask_axis)),
    )


def _mask_overlay(fixed: np.ndarray, moving: np.ndarray) -> np.ndarray:
    output = np.zeros((*fixed.shape, 3), dtype=np.uint8)
    output[..., 0] = np.where(fixed, 230, 0).astype(np.uint8)
    output[..., 1] = np.where(moving, 220, 0).astype(np.uint8)
    output[..., 2] = np.where(fixed & moving, 120, 0).astype(np.uint8)
    return output


def _save_map(value: Any, path: Path) -> None:
    arrays: dict[str, np.ndarray] = {}
    if isinstance(value, dict):
        for key in ("A", "v", "WM"):
            if key in value and value[key] is not None:
                item = value[key]
                arrays[key] = np.asarray(item.detach().cpu() if hasattr(item, "detach") else item)
        xv = value.get("xv")
        if isinstance(xv, (list, tuple)):
            for index, item in enumerate(xv):
                arrays[f"xv_{index}"] = np.asarray(item.detach().cpu() if hasattr(item, "detach") else item)
    if not arrays:
        raise ValueError("STalign map contained no serializable arrays")
    np.savez_compressed(path, **arrays)


def run_direct(
    *,
    v9_root: Path,
    v6_transform_dir: Path,
    he_manifest_path: Path,
    output_root: Path,
    device: str,
    niter: int,
    moving_additional_flip: str,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite direct STalign output: {output_root}")
    output_root.mkdir(parents=True)
    he_manifest = pd.read_parquet(he_manifest_path)
    source: dict[int, dict[str, Any]] = {}
    bridges: dict[int, np.ndarray] = {}
    provenance: dict[str, Any] = {}
    for section in (53, 55):
        raster = _load_v9_source_raster(v9_root, section)
        image, mask, he_meta = _load_source_h_and_e(v9_root, section, he_manifest)
        if section == 55:
            he_meta["additional_flip"] = moving_additional_flip
            he_meta["additional_flip_stage"] = "v6_canvas_after_bridge"
        if image.shape[1:] != mask.shape:
            raise ValueError(f"H&E/mask shape mismatch for section {section}")
        source[section] = {**raster, "image": image, "mask": mask}
        bridge_meta = _v9_to_v6_bridge(v9_root, v6_transform_dir, section)
        bridges[section] = bridge_meta["bridge_v9_to_v6"]
        provenance[str(section)] = {
            **he_meta,
            "v9_transform_path": bridge_meta["v9_transform_path"],
            "v6_transform_path": bridge_meta["v6_transform_path"],
            "v9_matrix": bridge_meta["v9_matrix"].tolist(),
            "v6_matrix": bridge_meta["v6_matrix"].tolist(),
            "bridge_v9_to_v6": bridge_meta["bridge_v9_to_v6"].tolist(),
        }
    dx_values = {float(source[s]["dx"]) for s in (53, 55)}
    if len(dx_values) != 1:
        raise ValueError(f"sections do not share one STalign raster spacing: {dx_values}")
    dx_um = float(next(iter(dx_values)))
    origin, shape, canvas_grid = _canvas_geometry(source, bridges, dx_um)
    images: dict[int, np.ndarray] = {}
    masks: dict[int, np.ndarray] = {}
    for section in (53, 55):
        images[section], masks[section] = _resample_to_canvas(
            source[section]["image"],
            source[section]["mask"],
            source[section]["grid"],
            bridges[section],
            canvas_grid,
            dx_um,
        )
    # The reviewed correction is a flip in the already registered v6 canvas,
    # not a flip of the local crop before its v9->v6 bridge.  Keeping this
    # operation here makes the mirror operation explicit and reproducible.
    if moving_additional_flip != "none":
        images[55], masks[55] = _apply_additional_moving_flip(
            images[55], masks[55], moving_additional_flip
        )
    fixed = RasterizedSection(
        sample_id="9957",
        group_id="g0",
        section_id=53,
        section_uid="9957__g0__section-53",
        image=images[53],
        grid_um=canvas_grid,
        mask=masks[53],
        origin_um=origin,
        dx_um=dx_um,
    )
    moving = RasterizedSection(
        sample_id="9957",
        group_id="g0",
        section_id=55,
        section_uid="9957__g0__section-55",
        image=images[55],
        grid_um=canvas_grid,
        mask=masks[55],
        origin_um=origin,
        dx_um=dx_um,
    )
    baseline = compute_partial_overlap_metrics(
        fixed.mask, moving.mask, dx_um, trim_fraction=0.2, tolerance_um=20.0
    )
    config = StalignConfig(
        dx_um=dx_um,
        padding_um=0.0,
        noflip=True,
        lddmm_niter=int(niter),
        lddmm_a_um=500.0,
        lddmm_expand=2.0,
        lddmm_diffeo_start=0,
        lddmm_device=device,
        max_deformation_p95_um=1000.0,
    )
    edge = run_stalign_edge(fixed, moving, config, np.eye(3, dtype=float))
    warped_mask = transform_mask_source_to_target(moving, fixed, edge)
    warped_image = transform_image_source_to_target(moving, fixed, edge)
    candidate = compute_partial_overlap_metrics(
        fixed.mask, warped_mask, dx_um, trim_fraction=0.2, tolerance_um=20.0
    )
    pair_dir = output_root / "pair-53-55"
    pair_dir.mkdir(parents=True)
    Image.fromarray(_morphology_to_rgb(fixed.image)).save(pair_dir / "fixed_section53_v6_frame.png")
    Image.fromarray(_morphology_to_rgb(moving.image)).save(pair_dir / "moving_section55_v6_frame_before_stalign.png")
    Image.fromarray(_morphology_to_rgb(warped_image)).save(pair_dir / "moving_section55_after_direct_stalign.png")
    Image.fromarray(_mask_overlay(fixed.mask, moving.mask)).save(pair_dir / "overlay_before_red53_green55.png")
    Image.fromarray(_mask_overlay(fixed.mask, warped_mask)).save(pair_dir / "overlay_after_red53_green55.png")
    np.savez_compressed(
        pair_dir / "direct_stalign_affine.npz",
        affine=np.asarray(edge.affine, dtype=float),
        initial_affine=np.eye(3, dtype=float),
        origin_um=np.asarray(origin, dtype=float),
        dx_um=np.asarray([dx_um], dtype=float),
        shape_rc=np.asarray(shape, dtype=np.int64),
    )
    _save_map(edge.forward_map, pair_dir / "direct_stalign_forward_map.npz")
    _save_map(edge.reverse_map, pair_dir / "direct_stalign_reverse_map.npz")
    np.savez_compressed(
        pair_dir / "direct_stalign_masks.npz",
        fixed_mask=fixed.mask,
        moving_mask_before=moving.mask,
        moving_mask_after=warped_mask,
    )
    summary = {
        "status": "direct_stalign_completed",
        "sample_id": "9957",
        "group_id": "g0",
        "fixed_section": 53,
        "moving_section": 55,
        "removed_intermediate_section": 54,
        "algorithm": "official STalign LDDMM",
        "stalign_version": importlib.metadata.version("STalign"),
        "device": device,
        "niter": int(niter),
        "noflip": True,
        "moving_additional_flip": moving_additional_flip,
        "coordinate_frame": "9957__g0_registered_final_manual_alignment_v6",
        "canvas_origin_um": list(origin),
        "canvas_shape_rc": list(shape),
        "dx_um": dx_um,
        "initial_affine": np.eye(3, dtype=float).tolist(),
        "direct_stalign_affine": np.asarray(edge.affine, dtype=float).tolist(),
        "stalign_metrics": {key: float(value) for key, value in edge.metrics.items() if np.isscalar(value)},
        "baseline_metrics": baseline,
        "candidate_metrics": candidate,
        "source_provenance": provenance,
        "pair_output": str(pair_dir.resolve()),
        "feature_store_note": "This artifact is the direct STalign edge; v7 feature/nucleus stores are materialized only from this edge.",
    }
    (output_root / "direct_stalign_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v9-root", type=Path, default=DEFAULT_V9_ROOT)
    parser.add_argument("--v6-transform-dir", type=Path, default=DEFAULT_V6_TRANSFORMS)
    parser.add_argument("--he-manifest", type=Path, default=DEFAULT_HE_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--niter", type=int, default=250)
    parser.add_argument(
        "--moving-additional-flip",
        choices=("none", "horizontal", "vertical", "horizontal_vertical"),
        default="none",
        help="Additional reviewed flip applied to moving section 55 after the source raster's recorded orientation.",
    )
    args = parser.parse_args()
    summary = run_direct(
        v9_root=args.v9_root.resolve(),
        v6_transform_dir=args.v6_transform_dir.resolve(),
        he_manifest_path=args.he_manifest.resolve(),
        output_root=args.output_root.resolve(),
        device=args.device,
        niter=args.niter,
        moving_additional_flip=args.moving_additional_flip,
    )
    print(json.dumps({key: summary[key] for key in ("status", "fixed_section", "moving_section", "stalign_version", "device", "moving_additional_flip", "baseline_metrics", "candidate_metrics", "direct_stalign_affine")}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
