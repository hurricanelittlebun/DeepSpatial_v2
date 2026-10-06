"""Compute a 53->55 STalign candidate in the original v1/v6 H&E frame.

This is deliberately a candidate-only operation.  It uses the raw H&E crops
and the reviewed v1 -> v5 -> v6 coordinate lineage, runs STalign once on
sections 53 and 55, and writes only an isolated candidate directory.  It does
not modify section transform tables, H5AD files, UNI2 features, nucleus
features, or downstream sections.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = PROJECT_ROOT.parent / "DeepSpatial_v2"
V2_SRC = V2_ROOT / "src"
for path in (PROJECT_ROOT, V2_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from deepspatial_v2.serial_registration.he_image_raster import morphology_channels  # noqa: E402
from deepspatial_v2.serial_registration.partial_overlap_qc import (  # noqa: E402
    compute_partial_overlap_metrics,
)
from deepspatial_v2.serial_registration.stalign_backend import (  # noqa: E402
    estimate_initial_affine,
    run_stalign_edge,
    transform_image_source_to_target,
    transform_mask_source_to_target,
)
from deepspatial_v2.serial_registration.stalign_types import (  # noqa: E402
    RasterizedSection,
    StalignConfig,
)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_module(
    PROJECT_ROOT / "scripts" / "render_9957_v8_true_coordinate_he_sections.py",
    "_base_9957_v1_coordinate_renderer",
)

GROUP = PROJECT_ROOT / "data" / "9957_g0"
ARCHIVE_TOP = (
    PROJECT_ROOT.parent
    / ".deepSpatial_archives"
    / "9957_g0_v5_rollback_20261001_132456"
    / "top_level"
)
SOURCE_PREP = ARCHIVE_TOP / "reconstruction_prep_v6_final_manual_alignment"
SOURCE_TABLE = SOURCE_PREP / "he_sections.parquet"
SOURCE_TRANSFORM_DIR = SOURCE_PREP / "registration_transforms_final_manual_alignment_v6"
BASE_REGISTRATION_ROOT = PROJECT_ROOT.parent / "DeepSpatial_v2" / "outputs" / "registration"
OUTPUT_ROOT = GROUP / "stalign_realign_53_55_v1_coordinate_chain_candidate_v2_coarse_init"
FIXED_SECTION = 53
MOVING_SECTION = 55
DX_UM = 16.0


def _apply_affine(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(points, dtype=float)
    flat = value.reshape(-1, 2)
    homogeneous = np.column_stack([flat, np.ones(len(flat), dtype=float)])
    return (homogeneous @ np.asarray(matrix, dtype=float).T)[:, :2].reshape(value.shape)


def _matrix_sha256(matrix: np.ndarray) -> str:
    value = np.asarray(matrix, dtype="<f8")
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


class V1CoordinateMapper:
    """Map raw H&E local physical XY to the reviewed v6 global frame.

    No 53->55 edge is present here.  This is intentionally the coordinate
    lineage used by the diagonal_v1 QC before the candidate edge is added.
    """

    def __init__(self, table: pd.DataFrame, *, device: str = "cpu") -> None:
        self.table = table
        self.device = device
        self.store = BASE.SavedTransformStore(BASE_REGISTRATION_ROOT, device=device)
        self.sections: dict[int, dict[str, Any]] = {}
        self.final_applied: dict[int, bool] = {}
        for row in table.sort_values("section_id", kind="stable").itertuples(index=False):
            sid = int(row.section_id)
            source_transform = SOURCE_TRANSFORM_DIR / Path(str(row.section_transform_path)).name
            if not source_transform.is_file():
                raise FileNotFoundError(source_transform)
            section = BASE.load_section_transform(source_transform)
            v1_to_v5, v5_to_v6, affine_meta = BASE._load_frame_affines(sid)
            self.sections[sid] = {
                "row": row,
                "section": section,
                "source_transform": source_transform,
                "v1_to_v5": v1_to_v5,
                "v5_to_v6": v5_to_v6,
                "affine_meta": affine_meta,
            }
            self.final_applied[sid] = False

    def local_to_global(self, points: np.ndarray, section_id: int) -> np.ndarray:
        record = self.sections[int(section_id)]
        value = BASE.transform_section_points(
            np.asarray(points, dtype=float), record["section"], self.store
        )
        value = _apply_affine(value, record["v1_to_v5"])
        return _apply_affine(value, record["v5_to_v6"])

    def global_to_local(self, points: np.ndarray, section_id: int) -> np.ndarray:
        record = self.sections[int(section_id)]
        value = _apply_affine(
            np.asarray(points, dtype=float), np.linalg.inv(record["v5_to_v6"])
        )
        value = _apply_affine(value, np.linalg.inv(record["v1_to_v5"]))
        return BASE._registered_to_source_points(value, record["section"], self.store)

    def roundtrip_error_um(self, points: np.ndarray, section_id: int) -> np.ndarray:
        recovered = self.global_to_local(self.local_to_global(points, section_id), section_id)
        return np.linalg.norm(recovered - np.asarray(points, dtype=float), axis=1)

    def metadata(self) -> dict[str, Any]:
        return {
            "coordinate_frame": "9957__g0_registered_final_manual_alignment_v6",
            "mapping_order": [
                "raw H&E crop local physical XY",
                "base saved preorientation + STalign edge chain",
                "v1 -> v5 frame affine",
                "v5 -> v6 frame affine",
            ],
            "base_registration_root": str(BASE_REGISTRATION_ROOT.resolve()),
            "source_table": str(SOURCE_TABLE.resolve()),
            "source_transform_dir": str(SOURCE_TRANSFORM_DIR.resolve()),
            "new_edge_present": False,
        }


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


def _morphology_to_rgb(channels: np.ndarray) -> np.ndarray:
    value = np.asarray(channels, dtype=float)
    rgb = np.moveaxis(1.0 - np.clip(value, 0.0, 1.0), 0, -1)
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)


def select_initial_affine(
    fixed: RasterizedSection, moving: RasterizedSection
) -> tuple[np.ndarray, float]:
    """Select a positive rotation initializer in the v1 global frame."""

    config = StalignConfig(
        dx_um=float(fixed.dx_um),
        noflip=True,
        coarse_angle_step_deg=5.0,
        coarse_sample_size=4000,
    )
    candidate = estimate_initial_affine(fixed, moving, config)
    return np.asarray(candidate.matrix, dtype=float), float(candidate.angle_deg)


def _mask_overlay(fixed: np.ndarray, moving: np.ndarray) -> np.ndarray:
    output = np.zeros((*fixed.shape, 3), dtype=np.uint8)
    output[..., 0] = np.where(fixed, 230, 0).astype(np.uint8)
    output[..., 1] = np.where(moving, 220, 0).astype(np.uint8)
    output[..., 2] = np.where(fixed & moving, 120, 0).astype(np.uint8)
    return output


def _redgreen_overlay(
    fixed_image: np.ndarray,
    fixed_mask: np.ndarray,
    moving_image: np.ndarray,
    moving_mask: np.ndarray,
) -> np.ndarray:
    fixed_density = np.clip(np.mean(fixed_image, axis=0), 0.0, 1.0)
    moving_density = np.clip(np.mean(moving_image, axis=0), 0.0, 1.0)
    output = np.zeros((*fixed_mask.shape, 3), dtype=np.uint8)
    output[..., 0] = np.clip(fixed_density * 255.0, 0, 255).astype(np.uint8) * fixed_mask
    output[..., 1] = np.clip(moving_density * 255.0, 0, 255).astype(np.uint8) * moving_mask
    return output


def _crop_nonzero(image: np.ndarray, *, margin_px: int = 12) -> np.ndarray:
    signal = np.any(image > 0, axis=2)
    rows, cols = np.nonzero(signal)
    if len(rows) == 0:
        return image
    r0 = max(0, int(rows.min()) - margin_px)
    r1 = min(image.shape[0], int(rows.max()) + margin_px + 1)
    c0 = max(0, int(cols.min()) - margin_px)
    c1 = min(image.shape[1], int(cols.max()) + margin_px + 1)
    return image[r0:r1, c0:c1]


def _global_bbox(mapper: V1CoordinateMapper, source: dict[str, Any], sid: int) -> tuple[np.ndarray, np.ndarray]:
    boundary_px = BASE._boundary_points(source["mask"], 6000)
    local = BASE._local_pixels_to_um(boundary_px, source)
    global_points = mapper.local_to_global(local, sid)
    return global_points.min(axis=0), global_points.max(axis=0)


def _make_canvas(
    mapper: V1CoordinateMapper,
    sources: dict[int, dict[str, Any]],
    *,
    dx_um: float,
    margin_um: float,
) -> tuple[np.ndarray, tuple[int, int], np.ndarray]:
    bboxes = [_global_bbox(mapper, sources[sid], sid) for sid in (FIXED_SECTION, MOVING_SECTION)]
    low = np.min(np.stack([item[0] for item in bboxes], axis=0), axis=0) - margin_um
    high = np.max(np.stack([item[1] for item in bboxes], axis=0), axis=0) + margin_um
    origin = np.floor(low / dx_um - 2.0) * dx_um
    maximum = np.ceil(high / dx_um + 2.0) * dx_um
    width = int(np.rint((maximum[0] - origin[0]) / dx_um)) + 1
    height = int(np.rint((maximum[1] - origin[1]) / dx_um)) + 1
    x = origin[0] + dx_um * np.arange(width, dtype=float)
    y = origin[1] + dx_um * np.arange(height, dtype=float)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return origin, (height, width), np.stack([xx, yy], axis=0)


def _sample_source(
    mapper: V1CoordinateMapper,
    source: dict[str, Any],
    sid: int,
    origin: np.ndarray,
    shape: tuple[int, int],
    dx_um: float,
) -> tuple[np.ndarray, np.ndarray]:
    rgb, valid = BASE._sample_source_on_global_grid(
        mapper, source, sid, origin, shape, dx_um
    )
    channels = morphology_channels(rgb)
    channels[:, ~valid] = 0.0
    return channels, valid


def run_candidate(*, device: str, niter: int, dx_um: float, margin_um: float) -> dict[str, Any]:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"refusing to overwrite candidate output: {OUTPUT_ROOT}")
    table = pd.read_parquet(SOURCE_TABLE).copy()
    table["section_id"] = pd.to_numeric(table["section_id"], errors="raise").astype(int)
    rows = {int(row.section_id): row for row in table.itertuples(index=False)}
    if FIXED_SECTION not in rows or MOVING_SECTION not in rows:
        raise KeyError("v1 source table must contain sections 53 and 55")
    mapper = V1CoordinateMapper(table, device=device)
    sources = {sid: BASE._load_source(rows[sid]) for sid in (FIXED_SECTION, MOVING_SECTION)}
    origin, shape, grid = _make_canvas(mapper, sources, dx_um=dx_um, margin_um=margin_um)
    images: dict[int, np.ndarray] = {}
    masks: dict[int, np.ndarray] = {}
    for sid in (FIXED_SECTION, MOVING_SECTION):
        images[sid], masks[sid] = _sample_source(
            mapper, sources[sid], sid, origin, shape, dx_um
        )

    fixed = RasterizedSection(
        sample_id="9957",
        group_id="g0",
        section_id=FIXED_SECTION,
        section_uid="9957__g0__section-53",
        image=images[FIXED_SECTION],
        grid_um=grid,
        mask=masks[FIXED_SECTION],
        origin_um=(float(origin[0]), float(origin[1])),
        dx_um=dx_um,
    )
    moving = RasterizedSection(
        sample_id="9957",
        group_id="g0",
        section_id=MOVING_SECTION,
        section_uid="9957__g0__section-55",
        image=images[MOVING_SECTION],
        grid_um=grid,
        mask=masks[MOVING_SECTION],
        origin_um=(float(origin[0]), float(origin[1])),
        dx_um=dx_um,
    )

    baseline = compute_partial_overlap_metrics(
        fixed.mask, moving.mask, dx_um, trim_fraction=0.2, tolerance_um=20.0
    )
    initial_affine, initial_angle_deg = select_initial_affine(fixed, moving)
    config = StalignConfig(
        dx_um=dx_um,
        padding_um=0.0,
        noflip=True,
        coarse_angle_step_deg=5.0,
        coarse_sample_size=4000,
        lddmm_niter=int(niter),
        lddmm_a_um=500.0,
        lddmm_expand=2.0,
        lddmm_diffeo_start=0,
        lddmm_device=device,
        max_deformation_p95_um=1000.0,
    )
    edge = run_stalign_edge(fixed, moving, config, initial_affine)
    warped_mask = transform_mask_source_to_target(moving, fixed, edge)
    warped_image = transform_image_source_to_target(moving, fixed, edge)
    candidate = compute_partial_overlap_metrics(
        fixed.mask, warped_mask, dx_um, trim_fraction=0.2, tolerance_um=20.0
    )

    OUTPUT_ROOT.mkdir(parents=True)
    pair_dir = OUTPUT_ROOT / "pair-53-55"
    pair_dir.mkdir()
    fixed_rgb = _morphology_to_rgb(fixed.image)
    moving_before_rgb = _morphology_to_rgb(moving.image)
    moving_after_rgb = _morphology_to_rgb(warped_image)
    Image.fromarray(fixed_rgb).save(pair_dir / "fixed_section53_v1_frame.png")
    Image.fromarray(moving_before_rgb).save(pair_dir / "moving_section55_v1_frame_before_stalign.png")
    Image.fromarray(moving_after_rgb).save(pair_dir / "moving_section55_after_v1_chain_stalign.png")
    Image.fromarray(_mask_overlay(fixed.mask, moving.mask)).save(
        pair_dir / "overlay_before_red53_green55.png"
    )
    Image.fromarray(_mask_overlay(fixed.mask, warped_mask)).save(
        pair_dir / "overlay_after_red53_green55.png"
    )
    redgreen = _redgreen_overlay(fixed.image, fixed.mask, warped_image, warped_mask)
    Image.fromarray(redgreen).save(pair_dir / "overlay_after_he_red53_green55.png")
    Image.fromarray(_crop_nonzero(redgreen)).save(
        pair_dir / "overlay_after_he_red53_green55_cropped.png"
    )
    np.savez_compressed(
        pair_dir / "direct_stalign_affine.npz",
        affine=np.asarray(edge.affine, dtype=float),
        initial_affine=np.asarray(initial_affine, dtype=float),
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

    provenance = {}
    for sid in (FIXED_SECTION, MOVING_SECTION):
        record = mapper.sections[sid]
        provenance[str(sid)] = {
            "source_image_path": sources[sid]["image_path"],
            "source_mask_path": sources[sid]["mask_path"],
            "v6_transform_path": str(record["source_transform"].resolve()),
            "v1_to_v5": np.asarray(record["v1_to_v5"]).tolist(),
            "v5_to_v6": np.asarray(record["v5_to_v6"]).tolist(),
            "v1_to_v5_sha256": _matrix_sha256(record["v1_to_v5"]),
            "v5_to_v6_sha256": _matrix_sha256(record["v5_to_v6"]),
            "roundtrip_max_um": float(
                mapper.roundtrip_error_um(
                    BASE._local_pixels_to_um(
                        BASE._boundary_points(sources[sid]["mask"], 500), sources[sid]
                    ),
                    sid,
                ).max()
            ),
        }
    summary = {
        "status": "v1_coordinate_chain_stalign_candidate_only",
        "candidate_only": True,
        "embedded_downstream": False,
        "sample_id": "9957",
        "group_id": "g0",
        "fixed_section": FIXED_SECTION,
        "moving_section": MOVING_SECTION,
        "removed_intermediate_section": 54,
        "algorithm": "official STalign LDDMM",
        "stalign_version": importlib.metadata.version("STalign"),
        "device": device,
        "niter": int(niter),
        "noflip": True,
        "initial_angle_deg": initial_angle_deg,
        "moving_additional_reflection": "none",
        "coordinate_frame": "9957__g0_registered_final_manual_alignment_v6",
        "input_coordinate_chain": mapper.metadata(),
        "canvas_origin_um": origin.tolist(),
        "canvas_shape_rc": list(shape),
        "dx_um": float(dx_um),
        "initial_affine": np.asarray(initial_affine, dtype=float).tolist(),
        "direct_stalign_affine": np.asarray(edge.affine, dtype=float).tolist(),
        "stalign_metrics": {
            key: float(value) for key, value in edge.metrics.items() if np.isscalar(value)
        },
        "baseline_metrics": baseline,
        "candidate_metrics": candidate,
        "source_provenance": provenance,
        "pair_output": str(pair_dir.resolve()),
        "next_step": "review overlay; do not apply to v1 tables or downstream data yet",
    }
    (OUTPUT_ROOT / "direct_stalign_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    (OUTPUT_ROOT / "README.md").write_text(
        "# 9957/g0 53->55 STalign candidate in the diagonal_v1 coordinate chain\n\n"
        "This directory is candidate-only. It uses the raw H&E crops and the reviewed "
        "v1/v5/v6 coordinate lineage. No section transform table, H5AD, UNI2 store, "
        "nucleus store, or section 55+ downstream data was modified.\n\n"
        "Review `pair-53-55/overlay_after_he_red53_green55_cropped.png` before any "
        "promotion.\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--niter", type=int, default=250)
    parser.add_argument("--dx-um", type=float, default=DX_UM)
    parser.add_argument("--margin-um", type=float, default=80.0)
    args = parser.parse_args()
    result = run_candidate(
        device=args.device,
        niter=args.niter,
        dx_um=args.dx_um,
        margin_um=args.margin_um,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "candidate_only": result["candidate_only"],
                "embedded_downstream": result["embedded_downstream"],
                "pair_output": result["pair_output"],
                "baseline_metrics": result["baseline_metrics"],
                "candidate_metrics": result["candidate_metrics"],
                "direct_stalign_affine": result["direct_stalign_affine"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
