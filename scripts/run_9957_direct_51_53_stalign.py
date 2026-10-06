"""Run the unified 9957/g0 section-51 -> section-53 STalign edge.

This is deliberately different from the historical direct 53 -> 55 run.
Section 51 is kept in the v6 frame.  Section 53 is first rendered through
the reviewed v9 53 -> 55 post-processing chain (horizontal flip, direct
STalign, CCW90), and a new STalign edge maps that old post-processed frame
back to the fixed section-51 frame.  The resulting edge can therefore be
composed *after* the old v9 chain for section 53 and all downstream sections.

The old v9/v10 artifacts are never overwritten.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable

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
DEFAULT_V9_DIRECT_ROOT = (
    DEFAULT_GROUP / "reconstruction_prep_v9_postrotate_ccw90_cutoff91" / "stalign_final_53_55" / "direct_v3"
)
DEFAULT_HE_MANIFEST = V2_ROOT / "outputs/sequence/manifests/he_combined_object_manifest.parquet"
DEFAULT_V6_TRANSFORMS = DEFAULT_GROUP / "reconstruction_prep_v6_final_manual_alignment/registration_transforms_final_manual_alignment_v6"
DEFAULT_OUTPUT = DEFAULT_GROUP / "stalign_direct_51_53_unified_v1"

FIXED_SECTION = 51
MOVING_SECTION = 53


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_existing_runner():
    return _load_module(
        PROJECT_ROOT / "scripts" / "run_9957_direct_53_55_stalign.py",
        "_deep_direct_53_55_helpers",
    )


def _load_latest_materializer():
    return _load_module(
        PROJECT_ROOT / "scripts" / "materialize_9957_latest_alignment_cutoff91.py",
        "_deep_v9_chain_for_51_53",
    )


def _apply_affine(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(points, dtype=float)
    if value.ndim != 2 or value.shape[1] != 2:
        raise ValueError("points must have shape [N, 2]")
    transform = np.asarray(matrix, dtype=float)
    if transform.shape != (3, 3) or not np.isfinite(transform).all():
        raise ValueError("matrix must be a finite 3x3 array")
    homogeneous = np.column_stack([value, np.ones(len(value), dtype=float)])
    return (homogeneous @ transform.T)[:, :2]


def _load_matrix(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        matrix = np.asarray(payload["matrix"], dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"invalid transform matrix: {path}")
    return matrix


def _v6_bridge(v9_root: Path, v6_transform_dir: Path, section: int) -> np.ndarray:
    helper = _load_existing_runner()
    metadata = helper._v9_to_v6_bridge(v9_root, v6_transform_dir, section)
    return np.asarray(metadata["bridge_v9_to_v6"], dtype=float)


def _load_old_v9_chain(v9_direct_root: Path, device: str):
    materializer = _load_latest_materializer()
    edge, summary = materializer._load_edge(
        v9_direct_root,
        summary_name="direct_stalign_summary.json",
        affine_name="direct_stalign_affine.npz",
        forward_name="direct_stalign_forward_map.npz",
        reverse_name="direct_stalign_reverse_map.npz",
        status="accepted_v9_direct_53_to_55",
        device=device,
    )
    if int(summary["fixed_section"]) != 53 or int(summary["moving_section"]) != 55:
        raise ValueError("the supplied old chain is not the reviewed 53 -> 55 edge")
    return materializer.LatestAlignmentChain(edge, None, device=device), summary


def _target_bbox(
    source: dict[int, dict[str, Any]],
    bridges: dict[int, np.ndarray],
    old_chain: Any,
    dx_um: float,
) -> tuple[tuple[float, float], tuple[int, int], np.ndarray]:
    """Return a common physical canvas for fixed-v6 and moving-old-v9 images."""

    points = []
    for section in (FIXED_SECTION, MOVING_SECTION):
        grid = np.asarray(source[section]["grid"], dtype=float)
        local_points = grid.reshape(2, -1).T
        value = _apply_affine(local_points, bridges[section])
        if section == MOVING_SECTION:
            value = old_chain.forward_chunked(value, batch_size=32768)
        points.append(value)
    value = np.concatenate(points, axis=0)
    origin_array = np.floor(np.min(value, axis=0) / dx_um - 2.0) * dx_um
    maximum = np.ceil(np.max(value, axis=0) / dx_um + 2.0) * dx_um
    width = int(np.rint((maximum[0] - origin_array[0]) / dx_um)) + 1
    height = int(np.rint((maximum[1] - origin_array[1]) / dx_um)) + 1
    if width <= 0 or height <= 0 or width * height > 30_000_000:
        raise ValueError(f"invalid or excessive direct 51->53 canvas: {(height, width)}")
    x = origin_array[0] + dx_um * np.arange(width, dtype=float)
    y = origin_array[1] + dx_um * np.arange(height, dtype=float)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    grid = np.stack([xx, yy], axis=0)
    return (float(origin_array[0]), float(origin_array[1])), (height, width), grid


def _resample_to_canvas(
    image: np.ndarray,
    mask: np.ndarray,
    source_grid: np.ndarray,
    canvas_grid: np.ndarray,
    target_to_source: Callable[[np.ndarray], np.ndarray],
    dx_um: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample one already-rasterized H&E source into the common canvas."""

    target_points = canvas_grid.reshape(2, -1).T
    source_points = np.asarray(target_to_source(target_points), dtype=float)
    source_origin = np.asarray(source_grid[:, 0, 0], dtype=float)
    columns = (source_points[:, 0] - source_origin[0]) / dx_um
    rows = (source_points[:, 1] - source_origin[1]) / dx_um
    source_shape = tuple(int(value) for value in mask.shape)
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
            np.asarray(image[channel], dtype=np.float32),
            [rows_clip, cols_clip],
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        output[channel] = sampled.reshape(target_shape)
    sampled_mask = map_coordinates(
        np.asarray(mask, dtype=np.float32),
        [rows_clip, cols_clip],
        order=0,
        mode="constant",
        cval=0.0,
        prefilter=False,
    ) >= 0.5
    valid_mask = (valid & sampled_mask).reshape(target_shape)
    output[:, ~valid_mask] = 0.0
    return output, valid_mask


def _morphology_to_rgb(channels: np.ndarray) -> np.ndarray:
    value = np.asarray(channels, dtype=float)
    if value.ndim != 3 or value.shape[0] != 3:
        raise ValueError("morphology channels must have shape [3, H, W]")
    rgb = np.moveaxis(1.0 - np.clip(value, 0.0, 1.0), 0, -1)
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)


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
                arrays[f"xv_{index}"] = np.asarray(
                    item.detach().cpu() if hasattr(item, "detach") else item
                )
    if not arrays:
        raise ValueError("STalign map contained no serializable arrays")
    np.savez_compressed(path, **arrays)


def run_direct(
    *,
    v9_root: Path,
    v9_direct_root: Path,
    v6_transform_dir: Path,
    he_manifest_path: Path,
    output_root: Path,
    device: str,
    niter: int,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite direct STalign output: {output_root}")
    output_root.mkdir(parents=True)

    helper = _load_existing_runner()
    materializer = _load_latest_materializer()
    he_manifest = pd.read_parquet(he_manifest_path)
    old_chain, old_summary = _load_old_v9_chain(v9_direct_root, device)

    source: dict[int, dict[str, Any]] = {}
    bridges: dict[int, np.ndarray] = {}
    provenance: dict[str, Any] = {}
    for section in (FIXED_SECTION, MOVING_SECTION):
        raster = helper._load_v9_source_raster(v9_root, section)
        image, mask, he_meta = helper._load_source_h_and_e(v9_root, section, he_manifest)
        source[section] = {**raster, "image": image, "mask": mask}
        bridges[section] = _v6_bridge(v9_root, v6_transform_dir, section)
        provenance[str(section)] = {
            **he_meta,
            "v9_to_v6_bridge": bridges[section].tolist(),
        }
        if image.shape[1:] != mask.shape:
            raise ValueError(f"H&E/mask shape mismatch for section {section}")

    dx_values = {float(source[s]["dx"]) for s in (FIXED_SECTION, MOVING_SECTION)}
    if len(dx_values) != 1:
        raise ValueError(f"sections do not share one source raster spacing: {dx_values}")
    dx_um = float(next(iter(dx_values)))
    origin, shape, canvas_grid = _target_bbox(source, bridges, old_chain, dx_um)

    inverse_bridges = {section: np.linalg.inv(bridges[section]) for section in (FIXED_SECTION, MOVING_SECTION)}

    def fixed_target_to_source(points: np.ndarray) -> np.ndarray:
        return _apply_affine(points, inverse_bridges[FIXED_SECTION])

    def moving_target_to_source(points: np.ndarray) -> np.ndarray:
        value = old_chain.inverse_chunked(points, batch_size=32768)
        return _apply_affine(value, inverse_bridges[MOVING_SECTION])

    fixed_image, fixed_mask = _resample_to_canvas(
        source[FIXED_SECTION]["image"],
        source[FIXED_SECTION]["mask"],
        source[FIXED_SECTION]["grid"],
        canvas_grid,
        fixed_target_to_source,
        dx_um,
    )
    moving_image, moving_mask = _resample_to_canvas(
        source[MOVING_SECTION]["image"],
        source[MOVING_SECTION]["mask"],
        source[MOVING_SECTION]["grid"],
        canvas_grid,
        moving_target_to_source,
        dx_um,
    )

    fixed = RasterizedSection(
        sample_id="9957",
        group_id="g0",
        section_id=FIXED_SECTION,
        section_uid="9957__g0__section-51",
        image=fixed_image,
        grid_um=canvas_grid,
        mask=fixed_mask,
        origin_um=origin,
        dx_um=dx_um,
    )
    moving = RasterizedSection(
        sample_id="9957",
        group_id="g0",
        section_id=MOVING_SECTION,
        section_uid="9957__g0__section-53",
        image=moving_image,
        grid_um=canvas_grid,
        mask=moving_mask,
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

    pair_dir = output_root / "pair-51-53"
    pair_dir.mkdir(parents=True)
    Image.fromarray(_morphology_to_rgb(fixed.image)).save(pair_dir / "fixed_section51_v6_frame.png")
    Image.fromarray(_morphology_to_rgb(moving.image)).save(
        pair_dir / "moving_section53_v9_postrotate_frame_before_stalign.png"
    )
    Image.fromarray(_morphology_to_rgb(warped_image)).save(
        pair_dir / "moving_section53_after_unified_stalign.png"
    )
    Image.fromarray(_mask_overlay(fixed.mask, moving.mask)).save(
        pair_dir / "overlay_before_red51_green53.png"
    )
    Image.fromarray(_mask_overlay(fixed.mask, warped_mask)).save(
        pair_dir / "overlay_after_red51_green53.png"
    )
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
    source_frame = "v9_postrotate_53_to_55_frame"
    target_frame = "v6_section51_fixed_frame"
    summary = {
        "status": "direct_stalign_completed",
        "sample_id": "9957",
        "group_id": "g0",
        "fixed_section": FIXED_SECTION,
        "moving_section": MOVING_SECTION,
        "algorithm": "official STalign LDDMM",
        "stalign_version": importlib.metadata.version("STalign"),
        "device": device,
        "niter": int(niter),
        "noflip": True,
        "coordinate_frame": target_frame,
        "moving_input_frame": source_frame,
        "composition_semantics": "new_edge(old_v9_chain(v6_points)) -> fixed_section51_v6_frame",
        "old_v9_chain_summary": old_summary,
        "old_v9_direct_root": str(v9_direct_root.resolve()),
        "canvas_origin_um": list(origin),
        "canvas_shape_rc": list(shape),
        "dx_um": dx_um,
        "initial_affine": np.eye(3, dtype=float).tolist(),
        "direct_stalign_affine": np.asarray(edge.affine, dtype=float).tolist(),
        "stalign_metrics": {
            key: float(value) for key, value in edge.metrics.items() if np.isscalar(value)
        },
        "baseline_metrics": baseline,
        "candidate_metrics": candidate,
        "source_provenance": provenance,
        "pair_output": str(pair_dir.resolve()),
        "v6_transform_dir": str(v6_transform_dir.resolve()),
        "he_manifest": str(he_manifest_path.resolve()),
        "warning": "Review raw H&E red/green output before materialization; metrics are diagnostic only.",
    }
    (output_root / "direct_stalign_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v9-root", type=Path, default=DEFAULT_V9_ROOT)
    parser.add_argument("--v9-direct-root", type=Path, default=DEFAULT_V9_DIRECT_ROOT)
    parser.add_argument("--v6-transform-dir", type=Path, default=DEFAULT_V6_TRANSFORMS)
    parser.add_argument("--he-manifest", type=Path, default=DEFAULT_HE_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--niter", type=int, default=250)
    args = parser.parse_args()
    summary = run_direct(
        v9_root=args.v9_root.resolve(),
        v9_direct_root=args.v9_direct_root.resolve(),
        v6_transform_dir=args.v6_transform_dir.resolve(),
        he_manifest_path=args.he_manifest.resolve(),
        output_root=args.output_root.resolve(),
        device=args.device,
        niter=args.niter,
    )
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "status",
                    "fixed_section",
                    "moving_section",
                    "stalign_version",
                    "device",
                    "baseline_metrics",
                    "candidate_metrics",
                    "direct_stalign_affine",
                    "pair_output",
                )
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
