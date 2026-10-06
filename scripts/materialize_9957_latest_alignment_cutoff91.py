"""Materialize the latest reviewed 9957/g0 coordinate release.

This release is intentionally independent of the older v6/v7 directories.
It keeps section 91, removes section 54, excludes section IDs above 91, and
applies the complete reviewed 53 -> 55 chain to the downstream registered
sections:

    v6 coordinate --horizontal canvas flip--> direct STalign
                  --counter-clockwise 90 deg--> fine-tune STalign
                  --> latest coordinate frame

The nonlinear maps are kept as first-class audit artifacts.  ``matrix`` in a
section transform is only the corresponding coarse affine composition; the
H5AD coordinates and feature stores are materialized with the full saved maps.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Callable

import anndata as ad
import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V2_SRC = ROOT.parent / "DeepSpatial_v2" / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(V2_SRC) not in sys.path:
    sys.path.insert(0, str(V2_SRC))

from deepspatial_v2.data_export.cell_level_alignment import (  # noqa: E402
    load_stalign_map,
)
from deepspatial_v2.serial_registration.stalign_backend import (  # noqa: E402
    transform_points_target_to_source_with_edge,
    transform_points_with_edge,
)
from deepspatial_v2.serial_registration.stalign_types import EdgeAlignment  # noqa: E402


def _load_feature_store_class():
    """Load FeatureStore without requiring the training package imports."""

    # The top-level deepspatial package imports the training stack.  This
    # materializer only needs the standalone HDF5 feature-store module.
    path = ROOT / "deepspatial" / "histology" / "feature_store.py"
    spec = importlib.util.spec_from_file_location("_deep_feature_store", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load FeatureStore from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FeatureStore


FeatureStore = _load_feature_store_class()


DEFAULT_GROUP = ROOT / "data" / "9957_g0"
DEFAULT_SOURCE_PREP = DEFAULT_GROUP / "reconstruction_prep_v6_final_manual_alignment"
DEFAULT_SOURCE_REG = DEFAULT_GROUP / "registration_correction_he_v6_final_manual_alignment"
DEFAULT_OUTPUT_PREP = DEFAULT_GROUP / "reconstruction_prep_v8_latest_alignment_cutoff91"
DEFAULT_OUTPUT_REG = DEFAULT_GROUP / "registration_correction_he_v8_latest_alignment_cutoff91"
DEFAULT_OUTPUT_QC = DEFAULT_GROUP / "qc" / "he_alignment_all_sections_v5_latest_alignment_cutoff91"
DEFAULT_OUTPUT_3D = (
    DEFAULT_GROUP
    / "qc"
    / "3d_latest_alignment_cutoff91"
    / "9957_g0_latest_alignment_cutoff91_opaque_3d.png"
)
DEFAULT_V3_ROOT = DEFAULT_GROUP / "stalign_direct_53_55_horizontal_canvas_flip_v3"
DEFAULT_POST_ROOT = DEFAULT_GROUP / "stalign_direct_53_55_postrotate_ccw90_v1"
DEFAULT_FINETUNE_ROOT = (
    DEFAULT_GROUP / "stalign_direct_53_55_postrotate_ccw90_finetune_v1"
)

RELEASE = "9957_latest_53_55_ccw90_finetune_cutoff91_v8"
OUTPUT_FRAME = "9957__g0_registered_latest_53_55_ccw90_finetune_v8_cutoff91"
VERSION_TAG = "v8"
REMOVED_SECTION_IDS = (54,)
CUTOFF_MAX_SECTION_ID = 91
DIRECT_FIXED_SECTION = 53
DIRECT_MOVING_SECTION = 55
TRANSFORM_BATCH_SIZE = 16384


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def _load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _apply_affine(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    matrix = np.asarray(matrix, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape [N, 2]")
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("matrix must be a finite 3x3 matrix")
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=float)])
    result = (homogeneous @ matrix.T)[:, :2]
    if not np.isfinite(result).all():
        raise ValueError("affine transformation produced non-finite coordinates")
    return result


def _map_from_three_points(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.shape != (3, 2) or target.shape != (3, 2):
        raise ValueError("three source and target points are required")
    design = np.column_stack([source, np.ones(3, dtype=float)])
    if abs(float(np.linalg.det(design))) < 1e-12:
        raise ValueError("source points are collinear")
    # design @ matrix.T = target
    matrix = np.linalg.solve(design, target).T
    result = np.eye(3, dtype=float)
    result[:2] = matrix
    return result


def horizontal_canvas_flip_affine(origin_um: tuple[float, float], shape_rc: tuple[int, int], dx_um: float) -> np.ndarray:
    """Affine matching np.flip(array, axis=1) on a physical canvas."""

    origin = np.asarray(origin_um, dtype=float)
    height, width = map(int, shape_rc)
    dx = float(dx_um)
    source = np.array(
        [
            [origin[0], origin[1]],
            [origin[0] + dx, origin[1]],
            [origin[0], origin[1] + dx],
        ],
        dtype=float,
    )
    target = source.copy()
    target[:, 0] = 2.0 * origin[0] + (width - 1) * dx - source[:, 0]
    return _map_from_three_points(source, target)


def canvas_ccw90_affine(origin_um: tuple[float, float], shape_rc: tuple[int, int], dx_um: float) -> np.ndarray:
    """Affine matching the saved square-pad/np.rot90/center-crop operation."""

    origin = np.asarray(origin_um, dtype=float)
    height, width = map(int, shape_rc)
    dx = float(dx_um)
    side = max(height, width)
    pad_row = (side - height) // 2
    pad_col = (side - width) // 2
    crop_row = (side - height) // 2
    crop_col = (side - width) // 2

    def transform_one(point: np.ndarray) -> np.ndarray:
        col = (point[0] - origin[0]) / dx
        row = (point[1] - origin[1]) / dx
        new_row = (side - 1) - (col + pad_col) - crop_row
        new_col = (row + pad_row) - crop_col
        return np.array([origin[0] + new_col * dx, origin[1] + new_row * dx])

    source = np.array(
        [
            [origin[0], origin[1]],
            [origin[0] + dx, origin[1]],
            [origin[0], origin[1] + dx],
        ],
        dtype=float,
    )
    target = np.vstack([transform_one(point) for point in source])
    return _map_from_three_points(source, target)


def _load_edge(
    root: Path,
    *,
    summary_name: str,
    affine_name: str,
    forward_name: str,
    reverse_name: str,
    status: str,
    device: str,
) -> tuple[EdgeAlignment, dict]:
    summary = _read_json(root / summary_name)
    pair = root / "pair-53-55"
    paths = [pair / affine_name, pair / forward_name, pair / reverse_name]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    with np.load(paths[0], allow_pickle=False) as payload:
        affine = np.asarray(payload["affine"], dtype=float)
    edge = EdgeAlignment(
        sample_id="9957",
        group_id="g0",
        fixed_section_id=DIRECT_FIXED_SECTION,
        moving_section_id=DIRECT_MOVING_SECTION,
        affine=affine,
        forward_map=load_stalign_map(paths[1], device),
        reverse_map=load_stalign_map(paths[2], device),
        transformed_points_um=np.empty((0, 2), dtype=float),
        metrics={},
        status=status,
    )
    return edge, summary


class LatestAlignmentChain:
    """Full point transform from the v6 frame to the latest frame."""

    def __init__(
        self,
        direct_edge: EdgeAlignment,
        finetune_edge: EdgeAlignment | None,
        *,
        device: str,
    ):
        self.direct_edge = direct_edge
        self.finetune_edge = finetune_edge
        self.device = device
        self.canvas_origin_um = (640.0, -352.0)
        self.canvas_shape_rc = (270, 266)
        self.canvas_dx_um = 16.0
        self.horizontal_flip = horizontal_canvas_flip_affine(
            self.canvas_origin_um, self.canvas_shape_rc, self.canvas_dx_um
        )
        self.ccw90 = canvas_ccw90_affine(
            self.canvas_origin_um, self.canvas_shape_rc, self.canvas_dx_um
        )
        self.coarse_affine = self.ccw90 @ np.asarray(direct_edge.affine, dtype=float) @ self.horizontal_flip
        if finetune_edge is not None:
            self.coarse_affine = (
                np.asarray(finetune_edge.affine, dtype=float) @ self.coarse_affine
            )

    def forward(self, points: np.ndarray) -> np.ndarray:
        value = _apply_affine(points, self.horizontal_flip)
        value = transform_points_with_edge(value, self.direct_edge)
        value = _apply_affine(value, self.ccw90)
        if self.finetune_edge is not None:
            value = transform_points_with_edge(value, self.finetune_edge)
        return value

    def inverse(self, points: np.ndarray) -> np.ndarray:
        value = points
        if self.finetune_edge is not None:
            value = transform_points_target_to_source_with_edge(value, self.finetune_edge)
        value = _apply_affine(value, np.linalg.inv(self.ccw90))
        value = transform_points_target_to_source_with_edge(value, self.direct_edge)
        return _apply_affine(value, np.linalg.inv(self.horizontal_flip))

    def forward_chunked(self, points: np.ndarray, batch_size: int = TRANSFORM_BATCH_SIZE) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        output = np.empty_like(points, dtype=float)
        for start in range(0, len(points), int(batch_size)):
            stop = min(start + int(batch_size), len(points))
            output[start:stop] = self.forward(points[start:stop])
        return output

    def inverse_chunked(self, points: np.ndarray, batch_size: int = TRANSFORM_BATCH_SIZE) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        output = np.empty_like(points, dtype=float)
        for start in range(0, len(points), int(batch_size)):
            stop = min(start + int(batch_size), len(points))
            output[start:stop] = self.inverse(points[start:stop])
        return output

    def metadata(self) -> dict:
        return {
            "coordinate_semantics": "v6 registered H&E/ST XY -> latest registered H&E/ST XY",
            "chain_order": [
                "horizontal_canvas_flip_v3",
                "direct_stalign_53_to_55_v3",
                "counterclockwise_90_canvas_rotation_v1",
            ],
            "fixed_section": DIRECT_FIXED_SECTION,
            "moving_section": DIRECT_MOVING_SECTION,
            "canvas_origin_um": list(self.canvas_origin_um),
            "canvas_shape_rc": list(self.canvas_shape_rc),
            "canvas_dx_um": self.canvas_dx_um,
            "horizontal_flip_affine": self.horizontal_flip.tolist(),
            "ccw90_affine": self.ccw90.tolist(),
            "coarse_affine": self.coarse_affine.tolist(),
            "nonlinear_maps": {
                "direct_forward": "stalign_final_53_55/direct_v3/pair-53-55/direct_stalign_forward_map.npz",
                "direct_reverse": "stalign_final_53_55/direct_v3/pair-53-55/direct_stalign_reverse_map.npz",
            },
        }


def _grid_centers(metadata: dict, *, origin=None, shape=None) -> np.ndarray:
    if origin is None:
        origin = np.asarray(metadata["origin_um"], dtype=float)
    else:
        origin = np.asarray(origin, dtype=float)
    if shape is None:
        height, width = (int(value) for value in metadata["shape"][:2])
    else:
        height, width = (int(value) for value in shape)
    spacing = np.asarray(metadata["spacing_um"], dtype=float)
    yy, xx = np.mgrid[:height, :width]
    return np.column_stack(
        [
            origin[0] + xx.reshape(-1) * spacing[0],
            origin[1] + yy.reshape(-1) * spacing[1],
        ]
    )


def _target_grid_metadata(source_metadata: dict, transformed_source_centers: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    spacing = np.asarray(source_metadata["spacing_um"], dtype=float)
    minimum = np.floor(np.min(transformed_source_centers, axis=0) / spacing) * spacing
    # One-cell margin prevents a transformed edge from being clipped by the
    # regular target grid.  Invalid margin cells are retained in ``valid``.
    origin = minimum - spacing
    maximum = np.max(transformed_source_centers, axis=0) + spacing
    shape = np.ceil((maximum - origin) / spacing).astype(int) + 1
    return origin, (int(shape[1]), int(shape[0]))


def _resample_section(
    source: FeatureStore,
    section_id: str,
    chain: LatestAlignmentChain,
) -> tuple[np.ndarray, np.ndarray, dict]:
    metadata = source.metadata[section_id]
    source_centers = _grid_centers(metadata)
    transformed_centers = chain.forward_chunked(source_centers)
    target_origin, target_shape = _target_grid_metadata(metadata, transformed_centers)
    target_points = _grid_centers(metadata, origin=target_origin, shape=target_shape)
    source_points = chain.inverse_chunked(target_points)
    queried, valid = source.get_feature(
        section_id,
        source_points,
        return_valid=True,
        nearest_max_distance_um=0.0,
    )
    features = queried.detach().cpu().numpy().astype("float32", copy=False)
    valid_array = valid.detach().cpu().numpy().astype(bool, copy=False)
    height, width = target_shape
    return (
        features.reshape(height, width, -1),
        valid_array.reshape(height, width),
        {
            "transform_type": "full_latest_chain_inverse_query",
            "source_section_id": section_id,
            "target_grid_origin_um": target_origin.tolist(),
            "target_grid_shape": [height, width],
            "output_valid_fraction": float(valid_array.mean()),
            "source_center_forward_bbox_um": [
                np.min(transformed_centers, axis=0).tolist(),
                np.max(transformed_centers, axis=0).tolist(),
            ],
        },
    )


def _copy_feature_store(
    source_path: Path,
    output_path: Path,
    kept_sections: list[str],
    chain: LatestAlignmentChain,
    *,
    store_name: str,
) -> dict:
    source = FeatureStore(source_path, cache_mb=1024)
    if set(kept_sections) - set(source.sections):
        raise KeyError(f"{store_name} lacks sections: {sorted(set(kept_sections) - set(source.sections))}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = FeatureStore(output_path, mode="a")
    written = []
    valid_fractions = {}
    with h5py.File(source_path, "r") as handle:
        for section_id in kept_sections:
            metadata = source.metadata[section_id]
            if int(section_id) >= DIRECT_MOVING_SECTION:
                features, valid, transform_record = _resample_section(source, section_id, chain)
                target_origin = transform_record["target_grid_origin_um"]
                target_shape = transform_record["target_grid_shape"]
            else:
                group = handle["sections"][source._key(section_id)]
                features = group["features"][:]
                valid = group["valid"][:]
                target_origin = metadata["origin_um"]
                target_shape = list(features.shape[:2])
                transform_record = {
                    "transform_type": "identity_before_53_to_55_chain",
                    "source_section_id": section_id,
                    "target_grid_origin_um": target_origin,
                    "target_grid_shape": target_shape,
                    "output_valid_fraction": float(np.asarray(valid, dtype=bool).mean()),
                }
            provenance = dict(metadata.get("provenance", {}))
            provenance.update(
                {
                    "release": RELEASE,
                    "source_store": str(source_path.resolve()),
                    "source_coordinate_frame": metadata["coordinate_frame"],
                    "latest_coordinate_frame": OUTPUT_FRAME,
                    "latest_chain_applied": int(section_id) >= DIRECT_MOVING_SECTION,
                    "latest_chain": chain.metadata(),
                    "section_transform": transform_record,
                }
            )
            output.add_section(
                section_id,
                features,
                z_um=float(metadata["z_um"]),
                origin_um=target_origin,
                spacing_um=metadata["spacing_um"],
                patch_size_um=float(metadata["patch_size_um"]),
                mpp=metadata["mpp"],
                coordinate_frame=OUTPUT_FRAME,
                valid_mask=valid,
                provenance=provenance,
            )
            written.append(section_id)
            valid_fractions[section_id] = float(np.asarray(valid, dtype=bool).mean())
    output.refresh()
    return {
        "store_name": store_name,
        "source_store": str(source_path.resolve()),
        "output_store": str(output_path.resolve()),
        "source_sections": len(source.sections),
        "output_sections": written,
        "feature_dim": int(output.feature_dim),
        "coordinate_frame": OUTPUT_FRAME,
        "valid_fraction_by_section": valid_fractions,
    }


def _copy_transforms_and_table(
    source_table_path: Path,
    output_table_path: Path,
    source_transform_dir: Path,
    output_transform_dir: Path,
    kept_sections: list[str],
    chain: LatestAlignmentChain,
) -> tuple[dict, dict[str, np.ndarray]]:
    source_table = pd.read_parquet(source_table_path).copy()
    source_table["section_id"] = pd.to_numeric(source_table["section_id"], errors="raise").astype(int)
    source_table = source_table.sort_values("z_um", kind="stable").reset_index(drop=True)
    requested = {int(value) for value in kept_sections}
    kept = source_table[source_table["section_id"].isin(requested)].copy()
    if set(kept["section_id"].astype(int)) != requested:
        raise KeyError("H&E table and requested section IDs differ")
    matrices = {}
    output_transform_dir.mkdir(parents=True, exist_ok=True)
    chain_json = json.dumps(chain.metadata(), ensure_ascii=False, separators=(",", ":"))
    for row in kept.itertuples(index=False):
        sid = str(int(row.section_id))
        source_path = source_transform_dir / Path(str(row.section_transform_path)).name
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        arrays = _load_npz_arrays(source_path)
        old_matrix = np.asarray(arrays["matrix"], dtype=float)
        apply_chain = int(row.section_id) >= DIRECT_MOVING_SECTION
        final_matrix = chain.coarse_affine @ old_matrix if apply_chain else old_matrix
        arrays["source_coordinate_frame"] = np.asarray(str(row.coordinate_frame))
        arrays[f"matrix_before_latest_alignment_{VERSION_TAG}"] = old_matrix
        arrays["matrix"] = final_matrix
        arrays["latest_alignment_release"] = np.asarray(RELEASE)
        arrays["latest_alignment_coordinate_frame"] = np.asarray(OUTPUT_FRAME)
        arrays["latest_alignment_applied"] = np.asarray([int(apply_chain)], dtype=np.uint8)
        arrays["latest_alignment_chain_json"] = np.asarray(chain_json)
        arrays["latest_alignment_coarse_affine"] = np.asarray(chain.coarse_affine, dtype=float)
        output_path = output_transform_dir / source_path.name
        np.savez_compressed(output_path, **arrays)
        matrices[sid] = final_matrix
        kept.loc[kept["section_id"] == int(row.section_id), "section_transform_path"] = str(output_path.resolve())

    kept["coordinate_frame"] = OUTPUT_FRAME
    kept["latest_alignment_release"] = RELEASE
    kept["latest_alignment_applied"] = kept["section_id"] >= DIRECT_MOVING_SECTION
    kept["latest_alignment_chain"] = chain_json
    kept["latest_alignment_cutoff_max_section_id"] = CUTOFF_MAX_SECTION_ID
    kept["removed_section_ids"] = json.dumps(list(REMOVED_SECTION_IDS))
    kept.to_parquet(output_table_path, index=False)
    return (
        {
            "source_table": str(source_table_path.resolve()),
            "output_table": str(output_table_path.resolve()),
            "source_section_count": int(len(source_table)),
            "output_section_count": int(len(kept)),
            "output_section_ids": [int(value) for value in kept.section_id],
            "excluded_above_cutoff": [
                int(value)
                for value in source_table.loc[source_table.section_id > CUTOFF_MAX_SECTION_ID, "section_id"]
            ],
        },
        matrices,
    )


def _copy_audit_artifacts(
    output_prep: Path,
    v3_root: Path,
    post_root: Path,
    finetune_root: Path | None,
) -> dict:
    destination = output_prep / "stalign_final_53_55"
    destination.mkdir(parents=True, exist_ok=False)
    mapping = {
        "direct_v3": v3_root,
        "postrotate_ccw90_v1": post_root,
    }
    if finetune_root is not None:
        mapping["finetune_v1"] = finetune_root
    copied = {}
    for name, source in mapping.items():
        if not source.is_dir():
            raise FileNotFoundError(source)
        target = destination / name
        shutil.copytree(source, target)
        copied[name] = str(target.resolve())
    return copied


def _materialize_h5ad(
    source_path: Path,
    output_path: Path,
    anchors_dir: Path,
    transforms_dir: Path,
    chain: LatestAlignmentChain,
    chain_metadata: dict,
) -> dict:
    source = ad.read_h5ad(source_path)
    if "section_id" not in source.obs or "spatial" not in source.obsm:
        raise KeyError("Source H5AD requires obs['section_id'] and obsm['spatial']")
    all_section_ids = pd.to_numeric(source.obs["section_id"], errors="raise").astype(int).to_numpy()
    keep = (~np.isin(all_section_ids, np.asarray(REMOVED_SECTION_IDS, dtype=int))) & (
        all_section_ids <= CUTOFF_MAX_SECTION_ID
    )
    source = source[keep].copy()
    section_ids = all_section_ids[keep]
    current = np.asarray(source.obsm["spatial"], dtype=np.float64)
    source.obsm[f"spatial_before_latest_alignment_{VERSION_TAG}"] = current.astype(np.float32)
    final = current.copy()
    for sid in sorted(np.unique(section_ids)):
        rows = section_ids == int(sid)
        if int(sid) >= DIRECT_MOVING_SECTION:
            final[rows] = chain.forward_chunked(current[rows])
    source.obsm[f"spatial_latest_alignment_{VERSION_TAG}"] = final.astype(np.float32)
    # These are the active keys used by the reconstruction code and by the
    # existing visualization utilities. Historical keys retain their original
    # values and remain available for audit comparisons.
    for key in ("spatial", "spatial_registered", "spatial_st_corrected"):
        if key in source.obsm:
            source.obsm[key] = final.astype(np.float32)
    source.obs["coordinate_frame"] = OUTPUT_FRAME
    source.obs["latest_alignment_release"] = RELEASE
    source.obs["latest_alignment_applied"] = section_ids >= DIRECT_MOVING_SECTION
    source.obs["latest_alignment_cutoff_max_section_id"] = CUTOFF_MAX_SECTION_ID
    source.obs["removed_section_ids"] = json.dumps(list(REMOVED_SECTION_IDS))
    source.obs["x_latest_alignment_um"] = final[:, 0]
    source.obs["y_latest_alignment_um"] = final[:, 1]
    transform_path_by_sid = {
        int(sid): str((transforms_dir / f"section-{int(sid)}-9957__g0__section-{int(sid)}.npz").resolve())
        for sid in sorted(set(section_ids.tolist()))
    }
    source.obs["registration_transform_path"] = [transform_path_by_sid[int(sid)] for sid in section_ids]
    source.uns[f"deepspatial_latest_alignment_{VERSION_TAG}"] = {
        "release": RELEASE,
        "coordinate_frame": OUTPUT_FRAME,
        "source_h5ad": str(source_path.resolve()),
        "removed_section_ids": list(REMOVED_SECTION_IDS),
        "cutoff_max_section_id": CUTOFF_MAX_SECTION_ID,
        "kept_section_ids": sorted(set(int(value) for value in section_ids)),
        "active_coordinate_key": "spatial_st_corrected",
        "latest_coordinate_key": f"spatial_latest_alignment_{VERSION_TAG}",
        "chain": chain_metadata,
    }
    source.uns[f"deepspatial_section_filter_{VERSION_TAG}"] = {
        "release": RELEASE,
        "removed_section_ids": list(REMOVED_SECTION_IDS),
        "cutoff_max_section_id": CUTOFF_MAX_SECTION_ID,
        "section_91_kept": bool(91 in set(section_ids.tolist())),
        "source_section_ids": sorted(set(int(value) for value in all_section_ids)),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source.write_h5ad(output_path, compression="gzip")

    anchors_dir.mkdir(parents=True, exist_ok=True)
    anchor_records = []
    for sid in sorted(set(int(value) for value in section_ids)):
        anchor = source[section_ids == sid].copy()
        anchor.uns["deepspatial_series"] = {
            "series_id": "9957__g0",
            "section_id": sid,
            "section_thickness_um": 5.0,
            "coordinate_frame": OUTPUT_FRAME,
            "release": RELEASE,
            "latest_alignment_applied": bool(sid >= DIRECT_MOVING_SECTION),
        }
        anchor_path = anchors_dir / f"section-{sid:03d}.h5ad"
        anchor.write_h5ad(anchor_path, compression="gzip")
        anchor_records.append({"section_id": sid, "path": str(anchor_path.resolve()), "n_obs": int(anchor.n_obs)})
    return {
        "source": str(source_path.resolve()),
        "output": str(output_path.resolve()),
        "n_obs": int(source.n_obs),
        "n_sections": len(anchor_records),
        "anchor_records": anchor_records,
        "section_ids": sorted(set(int(value) for value in section_ids)),
    }


def _assert_chain_artifacts(chain: LatestAlignmentChain) -> dict:
    points = np.array(
        [
            [640.0, -352.0],
            [1200.0, 500.0],
            [3000.0, 1800.0],
            [4500.0, 3800.0],
        ],
        dtype=float,
    )
    forward = chain.forward_chunked(points)
    inverse = chain.inverse_chunked(forward)
    error = np.linalg.norm(inverse - points, axis=1)
    if not np.isfinite(error).all() or float(error.max()) > 5.0:
        raise AssertionError(f"latest 53->55 chain inverse round-trip failed: {error}")
    # Confirm the saved CCW operation exactly matches the prior postrotation
    # mask artifact. This catches row/column and physical x/y convention bugs.
    mask_path = DEFAULT_POST_ROOT / "pair-53-55" / "postrotation_masks.npz"
    with np.load(mask_path, allow_pickle=False) as payload:
        before = np.asarray(payload["moving_mask_before"], dtype=bool)
        expected = np.asarray(payload["moving_mask_after_ccw90"], dtype=bool)
    rows, cols = np.nonzero(before)
    coords = np.column_stack(
        [
            chain.canvas_origin_um[0] + cols * chain.canvas_dx_um,
            chain.canvas_origin_um[1] + rows * chain.canvas_dx_um,
        ]
    )
    mapped = _apply_affine(coords, chain.ccw90)
    out_cols = np.rint((mapped[:, 0] - chain.canvas_origin_um[0]) / chain.canvas_dx_um).astype(int)
    out_rows = np.rint((mapped[:, 1] - chain.canvas_origin_um[1]) / chain.canvas_dx_um).astype(int)
    reconstructed = np.zeros_like(expected)
    valid = (
        (out_rows >= 0)
        & (out_rows < reconstructed.shape[0])
        & (out_cols >= 0)
        & (out_cols < reconstructed.shape[1])
    )
    reconstructed[out_rows[valid], out_cols[valid]] = True
    if not np.array_equal(reconstructed, expected):
        raise AssertionError("CCW90 affine does not reproduce the saved postrotation mask")
    return {
        "chain_roundtrip_max_error_um": float(error.max()),
        "ccw90_mask_reproduction_exact": True,
    }


def materialize(
    *,
    source_prep: Path,
    source_reg: Path,
    output_prep: Path,
    output_reg: Path,
    output_qc: Path,
    direct_root: Path,
    finetune_root: Path | None,
    device: str,
) -> dict:
    for path in (output_prep, output_reg, output_qc):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    if not output_reg.parent.exists():
        output_reg.parent.mkdir(parents=True, exist_ok=True)

    direct_edge, direct_summary = _load_edge(
        direct_root,
        summary_name="direct_stalign_summary.json",
        affine_name="direct_stalign_affine.npz",
        forward_name="direct_stalign_forward_map.npz",
        reverse_name="direct_stalign_reverse_map.npz",
        status="accepted_direct_stalign_v3_horizontal_flip",
        device=device,
    )
    if int(direct_summary["fixed_section"]) != 53 or int(direct_summary["moving_section"]) != 55:
        raise ValueError("direct artifact is not the 53 -> 55 pair")
    finetune_edge = None
    finetune_summary = None
    if finetune_root is not None:
        finetune_edge, finetune_summary = _load_edge(
            finetune_root,
            summary_name="postrotate_finetune_summary.json",
            affine_name="finetune_affine.npz",
            forward_name="finetune_forward_map.npz",
            reverse_name="finetune_reverse_map.npz",
            status="accepted_postrotate_ccw90_finetune",
            device=device,
        )
        if int(finetune_summary["fixed_section"]) != 53 or int(finetune_summary["moving_section"]) != 55:
            raise ValueError("finetune artifact is not the 53 -> 55 pair")
    chain = LatestAlignmentChain(direct_edge, finetune_edge, device=device)
    chain_check = _assert_chain_artifacts(chain)
    chain_metadata = {
        **chain.metadata(),
        "direct_summary": direct_summary,
        "finetune_summary": finetune_summary,
        "chain_checks": chain_check,
    }

    source_table = pd.read_parquet(source_prep / "he_sections.parquet")
    source_table["section_id"] = pd.to_numeric(source_table["section_id"], errors="raise").astype(int)
    source_ids = sorted(set(int(value) for value in source_table.section_id))
    excluded_above_cutoff = [sid for sid in source_ids if sid > CUTOFF_MAX_SECTION_ID]
    kept_ids = [sid for sid in source_ids if sid not in REMOVED_SECTION_IDS and sid <= CUTOFF_MAX_SECTION_ID]
    if 91 not in kept_ids:
        raise AssertionError("Section 91 must be retained")
    if 54 in kept_ids or any(sid > CUTOFF_MAX_SECTION_ID for sid in kept_ids):
        raise AssertionError("section filter is incorrect")
    kept = [str(sid) for sid in kept_ids]

    output_prep.mkdir(parents=True, exist_ok=False)
    output_reg.mkdir(parents=True, exist_ok=False)
    output_transform_dir = output_prep / f"registration_transforms_latest_alignment_{VERSION_TAG}"
    table_result, matrices = _copy_transforms_and_table(
        source_prep / "he_sections.parquet",
        output_prep / "he_sections.parquet",
        source_prep / "registration_transforms_final_manual_alignment_v6",
        output_transform_dir,
        kept,
        chain,
    )
    audit_paths = _copy_audit_artifacts(
        output_prep,
        DEFAULT_V3_ROOT,
        DEFAULT_POST_ROOT,
        finetune_root,
    )

    feature_result = _copy_feature_store(
        source_prep / "uni2_features_final_manual_alignment_v6.h5",
        output_prep / f"uni2_features_latest_alignment_cutoff91_{VERSION_TAG}.h5",
        kept,
        chain,
        store_name="uni2",
    )
    nucleus_result = _copy_feature_store(
        source_prep / "nucleus_path_final_manual_alignment_v6" / "nucleus_features.h5",
        output_prep / f"nucleus_path_latest_alignment_cutoff91_{VERSION_TAG}" / "nucleus_features.h5",
        kept,
        chain,
        store_name="nucleus",
    )
    candidate_result = _materialize_h5ad(
        source_reg / "9957_g0__st_to_final_manual_alignment_v6.h5ad",
        output_reg / f"9957_g0__st_to_latest_alignment_cutoff91_{VERSION_TAG}.h5ad",
        output_prep / "anchors",
        output_transform_dir,
        chain,
        chain_metadata,
    )

    # 35-36 was the latest manually reviewed local correction.  It must not
    # be changed by the 53-55 downstream release.
    source_35 = source_prep / "registration_transforms_final_manual_alignment_v6" / "section-35-9957__g0__section-35.npz"
    source_36 = source_prep / "registration_transforms_final_manual_alignment_v6" / "section-36-9957__g0__section-36.npz"
    preserved_35_36 = {}
    for sid, source_path in ((35, source_35), (36, source_36)):
        with np.load(source_path, allow_pickle=False) as source:
            with np.load(output_transform_dir / source_path.name, allow_pickle=False) as output:
                preserved_35_36[str(sid)] = bool(np.allclose(source["matrix"], output["matrix"]))
    if not all(preserved_35_36.values()):
        raise AssertionError(f"35-36 transform changed unexpectedly: {preserved_35_36}")

    manifest = {
        "status": "materialized",
        "release": RELEASE,
        "coordinate_frame": OUTPUT_FRAME,
        "source_prep": str(source_prep.resolve()),
        "source_reg": str(source_reg.resolve()),
        "removed_section_ids": list(REMOVED_SECTION_IDS),
        "cutoff_max_section_id": CUTOFF_MAX_SECTION_ID,
        "source_section_ids": source_ids,
        "excluded_above_cutoff": excluded_above_cutoff,
        "output_section_ids": kept_ids,
        "section_91_kept": True,
        "section_54_removed": True,
        "latest_35_36_preserved": preserved_35_36,
        "chain": chain_metadata,
        "audit_artifacts": audit_paths,
        "he_table": table_result,
        "feature_store": feature_result,
        "nucleus_store": nucleus_result,
        "candidate": candidate_result,
        "output_qc_directory": str(output_qc.resolve()),
        "cache_policy": "rebuild UOT/path caches because 54 is removed and the 53->55 coordinate chain changed",
    }
    _write_json(output_prep / "latest_alignment_manifest.json", manifest)
    _write_json(output_reg / "latest_alignment_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-prep", type=Path, default=DEFAULT_SOURCE_PREP)
    parser.add_argument("--source-reg", type=Path, default=DEFAULT_SOURCE_REG)
    parser.add_argument("--output-prep", type=Path, default=DEFAULT_OUTPUT_PREP)
    parser.add_argument("--output-reg", type=Path, default=DEFAULT_OUTPUT_REG)
    parser.add_argument("--output-qc", type=Path, default=DEFAULT_OUTPUT_QC)
    parser.add_argument("--direct-root", type=Path, default=DEFAULT_V3_ROOT)
    parser.add_argument("--finetune-root", type=Path, default=DEFAULT_FINETUNE_ROOT)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = materialize(
        source_prep=args.source_prep.resolve(),
        source_reg=args.source_reg.resolve(),
        output_prep=args.output_prep.resolve(),
        output_reg=args.output_reg.resolve(),
        output_qc=args.output_qc.resolve(),
        direct_root=args.direct_root.resolve(),
        finetune_root=args.finetune_root.resolve(),
        device=args.device,
    )
    print(json.dumps({
        "status": result["status"],
        "release": result["release"],
        "output_section_count": len(result["output_section_ids"]),
        "section_91_kept": result["section_91_kept"],
        "section_54_removed": result["section_54_removed"],
        "latest_35_36_preserved": result["latest_35_36_preserved"],
        "h5ad": result["candidate"]["output"],
        "uni2": result["feature_store"]["output_store"],
        "nucleus": result["nucleus_store"]["output_store"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
