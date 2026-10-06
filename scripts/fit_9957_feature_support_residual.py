"""Fit a 53 -> 55 residual on the actual v9 feature-support grids.

The older 53 -> 55 STalign edge was fitted on a raw H&E raster.  Downstream
UNI2/nucleus stores, however, are queried on their own registered support
grids.  This script fits one small, auditable STalign affine residual on
those exact grids and saves it without modifying any v9 artifact.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
V2_SRC = ROOT.parent / "DeepSpatial_v2" / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(V2_SRC) not in sys.path:
    sys.path.insert(0, str(V2_SRC))

from deepspatial_v2.serial_registration.partial_overlap_qc import (  # noqa: E402
    compute_partial_overlap_metrics,
)
from deepspatial_v2.serial_registration.stalign_backend import (  # noqa: E402
    estimate_initial_affine,
    run_stalign_edge,
)
from deepspatial_v2.serial_registration.stalign_types import (  # noqa: E402
    RasterizedSection,
    StalignConfig,
)


GROUP = ROOT / "data" / "9957_g0"
DEFAULT_STORE = GROUP / "reconstruction_prep_v9_postrotate_ccw90_cutoff91" / (
    "uni2_features_latest_alignment_cutoff91_v9.h5"
)
DEFAULT_OUTPUT = GROUP / "feature_support_stalign_53_55_v1"


def _load_feature_store_class():
    path = ROOT / "deepspatial" / "histology" / "feature_store.py"
    spec = importlib.util.spec_from_file_location("_feature_store_for_residual", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load FeatureStore from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FeatureStore


FeatureStore = _load_feature_store_class()


def _grid(metadata: dict) -> np.ndarray:
    height, width, _ = metadata["shape"]
    yy, xx = np.mgrid[: int(height), : int(width)]
    return np.stack(
        [
            float(metadata["origin_um"][0]) + xx * float(metadata["spacing_um"][0]),
            float(metadata["origin_um"][1]) + yy * float(metadata["spacing_um"][1]),
        ],
        axis=0,
    )


def _load_section(store: FeatureStore, section_id: int) -> RasterizedSection:
    sid = str(int(section_id))
    metadata = store.metadata[sid]
    with h5py.File(store.path, "r") as handle:
        mask = np.asarray(handle["sections"][store._key(sid)]["valid"][:], dtype=bool)
    grid = _grid(metadata)
    return RasterizedSection(
        sample_id="9957",
        group_id="g0",
        section_id=int(section_id),
        section_uid=f"9957__g0__section-{int(section_id)}",
        image=mask.astype(np.float32, copy=False)[None, ...],
        grid_um=grid,
        mask=mask,
        origin_um=tuple(float(v) for v in metadata["origin_um"]),
        dx_um=float(metadata["spacing_um"][0]),
    )


def _save_stalign_map(value, path: Path) -> None:
    arrays = {}
    if isinstance(value, dict):
        for key in ("A", "v", "WM"):
            if value.get(key) is not None:
                item = value[key]
                arrays[key] = np.asarray(
                    item.detach().cpu() if hasattr(item, "detach") else item
                )
        xv = value.get("xv")
        if isinstance(xv, (list, tuple)):
            for index, item in enumerate(xv):
                arrays[f"xv_{index}"] = np.asarray(
                    item.detach().cpu() if hasattr(item, "detach") else item
                )
    if not arrays:
        raise ValueError("STalign map did not contain serializable arrays")
    np.savez_compressed(path, **arrays)


def _warp_mask_to_fixed(moving: RasterizedSection, fixed: RasterizedSection, affine: np.ndarray) -> np.ndarray:
    from scipy.ndimage import map_coordinates

    target = fixed.grid_um.reshape(2, -1).T
    homogeneous = np.column_stack([target, np.ones(len(target))])
    source = (homogeneous @ np.linalg.inv(affine).T)[:, :2]
    source_origin = np.asarray(moving.origin_um, dtype=float)
    spacing = float(moving.dx_um)
    columns = (source[:, 0] - source_origin[0]) / spacing
    rows = (source[:, 1] - source_origin[1]) / spacing
    warped = map_coordinates(
        moving.mask.astype(np.float32),
        [rows, columns],
        order=0,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    return (warped >= 0.5).reshape(fixed.mask.shape)


def _common_canvas_masks(first: RasterizedSection, second: RasterizedSection) -> tuple[np.ndarray, np.ndarray]:
    """Put two same-spacing support masks on one physical comparison canvas."""

    spacing = float(first.dx_um)
    points = []
    for section in (first, second):
        rows, cols = np.nonzero(section.mask)
        points.append(
            np.column_stack(
                [
                    float(section.origin_um[0]) + cols * spacing,
                    float(section.origin_um[1]) + rows * spacing,
                ]
            )
        )
    all_points = np.concatenate(points, axis=0)
    origin = np.floor(np.min(all_points, axis=0) / spacing) * spacing
    maximum = np.ceil(np.max(all_points, axis=0) / spacing) * spacing
    width = int(round((maximum[0] - origin[0]) / spacing)) + 1
    height = int(round((maximum[1] - origin[1]) / spacing)) + 1

    def place(section: RasterizedSection) -> np.ndarray:
        rows, cols = np.nonzero(section.mask)
        x = float(section.origin_um[0]) + cols * spacing
        y = float(section.origin_um[1]) + rows * spacing
        out = np.zeros((height, width), dtype=bool)
        out[
            np.rint((y - origin[1]) / spacing).astype(int),
            np.rint((x - origin[0]) / spacing).astype(int),
        ] = True
        return out

    return place(first), place(second)


def _overlay(fixed: np.ndarray, moving: np.ndarray) -> np.ndarray:
    output = np.zeros((*fixed.shape, 3), dtype=np.uint8)
    output[..., 0] = np.where(fixed, 235, 0).astype(np.uint8)
    output[..., 1] = np.where(moving, 235, 0).astype(np.uint8)
    output[..., 2] = np.where(fixed & moving, 155, 0).astype(np.uint8)
    return output


def fit(*, store_path: Path, output_root: Path, device: str, niter: int) -> dict:
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite residual artifact: {output_root}")
    output_root.mkdir(parents=True)
    np.random.seed(20260930)
    try:
        import torch

        torch.manual_seed(20260930)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(20260930)
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("STalign residual fitting requires PyTorch") from error

    store = FeatureStore(store_path, cache_mb=512)
    fixed = _load_section(store, 53)
    moving = _load_section(store, 55)
    dx = float(fixed.dx_um)
    if not np.isclose(dx, moving.dx_um):
        raise ValueError("53 and 55 feature grids do not share spacing")
    config = StalignConfig(
        dx_um=dx,
        padding_um=0.0,
        noflip=True,
        lddmm_niter=int(niter),
        lddmm_a_um=500.0,
        lddmm_expand=2.0,
        lddmm_diffeo_start=0,
        lddmm_device=device,
        coarse_angle_step_deg=15.0,
        coarse_sample_size=2000,
        max_deformation_p95_um=1000.0,
    )
    initial = estimate_initial_affine(fixed, moving, config)
    edge = run_stalign_edge(fixed, moving, config, initial.matrix)
    residual = np.asarray(edge.affine, dtype=float)
    warped = _warp_mask_to_fixed(moving, fixed, residual)
    fixed_common, moving_common = _common_canvas_masks(fixed, moving)
    baseline = compute_partial_overlap_metrics(
        fixed_common, moving_common, dx, trim_fraction=0.2, tolerance_um=20.0
    )
    candidate = compute_partial_overlap_metrics(
        fixed.mask, warped, dx, trim_fraction=0.2, tolerance_um=20.0
    )

    pair = output_root / "pair-53-55"
    pair.mkdir()
    Image.fromarray(_overlay(fixed_common, moving_common)).save(pair / "overlay_before_red53_green55.png")
    Image.fromarray(_overlay(fixed.mask, warped)).save(pair / "overlay_after_red53_green55.png")
    np.savez_compressed(
        pair / "feature_support_residual_affine.npz",
        affine=residual,
        initial_affine=np.asarray(initial.matrix, dtype=float),
        spacing_um=np.asarray([dx], dtype=float),
        fixed_origin_um=np.asarray(fixed.origin_um, dtype=float),
        moving_origin_um=np.asarray(moving.origin_um, dtype=float),
        fixed_shape_rc=np.asarray(fixed.mask.shape, dtype=np.int64),
        moving_shape_rc=np.asarray(moving.mask.shape, dtype=np.int64),
    )
    _save_stalign_map(edge.forward_map, pair / "stalign_forward_map.npz")
    _save_stalign_map(edge.reverse_map, pair / "stalign_reverse_map.npz")
    np.savez_compressed(
        pair / "feature_support_masks.npz",
        fixed_mask=fixed.mask,
        moving_mask_before=moving.mask,
        moving_mask_after=warped,
    )
    summary = {
        "status": "feature_support_residual_fitted",
        "algorithm": "official STalign LDDMM on v9 UNI2 valid-support masks",
        "sample_id": "9957",
        "group_id": "g0",
        "fixed_section": 53,
        "moving_section": 55,
        "source_feature_store": str(store_path.resolve()),
        "output_root": str(output_root.resolve()),
        "coordinate_frame_before_residual": store.metadata["53"]["coordinate_frame"],
        "residual_semantics": "p_new = residual_affine @ p_v9 for section 55 and all downstream sections",
        "device": device,
        "niter": int(niter),
        "noflip": True,
        "initial_angle_deg": float(initial.angle_deg),
        "initial_affine": np.asarray(initial.matrix, dtype=float).tolist(),
        "residual_affine": residual.tolist(),
        "stalign_metrics": {key: float(value) for key, value in edge.metrics.items() if np.isscalar(value)},
        "baseline_metrics": baseline,
        "candidate_metrics": candidate,
        "fixed_mask_shape_rc": list(fixed.mask.shape),
        "moving_mask_shape_rc": list(moving.mask.shape),
        "feature_spacing_um": dx,
        "pair_output": str(pair.resolve()),
    }
    (output_root / "feature_support_residual_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--niter", type=int, default=120)
    args = parser.parse_args()
    result = fit(
        store_path=args.store.resolve(),
        output_root=args.output_root.resolve(),
        device=args.device,
        niter=args.niter,
    )
    print(json.dumps({key: result[key] for key in (
        "status", "initial_angle_deg", "residual_affine", "baseline_metrics", "candidate_metrics", "pair_output"
    )}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
