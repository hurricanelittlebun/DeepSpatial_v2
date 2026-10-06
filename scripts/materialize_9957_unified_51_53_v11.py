"""Materialize one 9957/g0 coordinate frame after the 51 -> 53 repair.

The source release is v6.  For section 51 and earlier sections the frame is
unchanged.  For section 53 and all later sections the full transform is:

    v6 source point
        -> reviewed v9 53 -> 55 postrotate chain
        -> new direct STalign 51 -> 53 edge
        -> unified section-51 frame

The nonlinear maps are used for H5AD coordinates and feature/nucleus grid
resampling.  The matrix stored in the H&E table is the auditable coarse
affine component only; the manifest records the full map composition.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

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

from deepspatial_v2.data_export.cell_level_alignment import load_stalign_map  # noqa: E402
from deepspatial_v2.serial_registration.stalign_backend import (  # noqa: E402
    transform_points_target_to_source_with_edge,
    transform_points_with_edge,
)
from deepspatial_v2.serial_registration.stalign_types import EdgeAlignment  # noqa: E402


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ALIGNMENT = _load_module(ROOT / "scripts" / "9957_unified_alignment.py", "_unified_9957_alignment")
OLD_MATERIALIZER = _load_module(
    ROOT / "scripts" / "materialize_9957_latest_alignment_cutoff91.py",
    "_old_9957_materializer_helpers",
)
FeatureStore = OLD_MATERIALIZER.FeatureStore


GROUP = ROOT / "data" / "9957_g0"
SOURCE_PREP = GROUP / "reconstruction_prep_v6_final_manual_alignment"
SOURCE_REG = GROUP / "registration_correction_he_v6_final_manual_alignment"
V9_PREP = GROUP / "reconstruction_prep_v9_postrotate_ccw90_cutoff91"
NEW_EDGE_ROOT = GROUP / "stalign_direct_51_53_unified_v1"
V9_DIRECT_ROOT = V9_PREP / "stalign_final_53_55" / "direct_v3"

DEFAULT_OUTPUT_PREP = GROUP / "reconstruction_prep_v11_unified_51_53"
DEFAULT_OUTPUT_REG = GROUP / "registration_correction_he_v11_unified_51_53"
DEFAULT_DEVICE = "cuda:0"

OUTPUT_FRAME = "9957__g0_registered_unified_51_53_v11"
RELEASE = "9957_unified_51_53_v11"
REMOVED_SECTION_IDS = (54,)
CUTOFF_MAX_SECTION_ID = 91
NEW_EDGE_SECTION = 53
TRANSFORM_BATCH_SIZE = 16384


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _load_edge(root: Path, device: str, *, fixed: int, moving: int, status: str) -> tuple[EdgeAlignment, dict]:
    summary = _read_json(root / "direct_stalign_summary.json")
    pair = root / f"pair-{fixed}-{moving}"
    affine_path = pair / "direct_stalign_affine.npz"
    forward_path = pair / "direct_stalign_forward_map.npz"
    reverse_path = pair / "direct_stalign_reverse_map.npz"
    for path in (affine_path, forward_path, reverse_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    with np.load(affine_path, allow_pickle=False) as payload:
        affine = np.asarray(payload["affine"], dtype=float)
    edge = EdgeAlignment(
        sample_id="9957",
        group_id="g0",
        fixed_section_id=int(fixed),
        moving_section_id=int(moving),
        affine=affine,
        forward_map=load_stalign_map(forward_path, device),
        reverse_map=load_stalign_map(reverse_path, device),
        transformed_points_um=np.empty((0, 2), dtype=float),
        metrics={},
        status=status,
    )
    return edge, summary


def _load_chain(device: str) -> tuple[Any, dict, EdgeAlignment, dict]:
    old_edge, old_summary = OLD_MATERIALIZER._load_edge(
        V9_DIRECT_ROOT,
        summary_name="direct_stalign_summary.json",
        affine_name="direct_stalign_affine.npz",
        forward_name="direct_stalign_forward_map.npz",
        reverse_name="direct_stalign_reverse_map.npz",
        status="accepted_v9_direct_53_to_55",
        device=device,
    )
    if int(old_summary["fixed_section"]) != 53 or int(old_summary["moving_section"]) != 55:
        raise ValueError("v9 direct artifact is not 53 -> 55")
    base_chain = OLD_MATERIALIZER.LatestAlignmentChain(old_edge, None, device=device)
    new_edge, new_summary = _load_edge(
        NEW_EDGE_ROOT,
        device,
        fixed=51,
        moving=53,
        status="accepted_unified_51_to_53",
    )
    if int(new_summary["fixed_section"]) != 51 or int(new_summary["moving_section"]) != 53:
        raise ValueError("new direct artifact is not 51 -> 53")
    chain = ALIGNMENT.Unified9957Chain(
        base_chain=base_chain,
        edge_51_to_53=new_edge,
        new_edge_section=NEW_EDGE_SECTION,
        downstream_section=55,
    )
    return chain, {"base": old_summary, "new_edge": new_summary}, new_edge, old_edge


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
    origin = minimum - spacing
    maximum = np.max(transformed_source_centers, axis=0) + spacing
    shape = np.ceil((maximum - origin) / spacing).astype(int) + 1
    return origin, (int(shape[1]), int(shape[0]))


def _forward(chain, points: np.ndarray, section: int) -> np.ndarray:
    value = np.asarray(points, dtype=float)
    output = np.empty_like(value)
    for start in range(0, len(value), TRANSFORM_BATCH_SIZE):
        stop = min(start + TRANSFORM_BATCH_SIZE, len(value))
        output[start:stop] = chain.forward_points(value[start:stop], int(section))
    return output


def _inverse(chain, points: np.ndarray, section: int) -> np.ndarray:
    value = np.asarray(points, dtype=float)
    output = np.empty_like(value)
    for start in range(0, len(value), TRANSFORM_BATCH_SIZE):
        stop = min(start + TRANSFORM_BATCH_SIZE, len(value))
        output[start:stop] = chain.inverse_points(value[start:stop], int(section))
    return output


def _resample_feature_section(source: FeatureStore, section_id: str, chain) -> tuple[np.ndarray, np.ndarray, dict]:
    metadata = source.metadata[section_id]
    source_centers = _grid_centers(metadata)
    transformed_centers = _forward(chain, source_centers, int(section_id))
    target_origin, target_shape = _target_grid_metadata(metadata, transformed_centers)
    target_points = _grid_centers(metadata, origin=target_origin, shape=target_shape)
    source_points = _inverse(chain, target_points, int(section_id))
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
            "transform_type": "full_unified_51_53_inverse_query",
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
    chain,
    store_name: str,
) -> dict:
    source = FeatureStore(source_path, cache_mb=1024)
    if set(kept_sections) - set(source.sections):
        raise KeyError(f"{store_name} lacks sections: {sorted(set(kept_sections) - set(source.sections))}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = FeatureStore(output_path, mode="a")
    valid_fractions = {}
    written = []
    with h5py.File(source_path, "r") as handle:
        for section_id in kept_sections:
            metadata = source.metadata[section_id]
            if int(section_id) >= NEW_EDGE_SECTION:
                features, valid, transform_record = _resample_feature_section(source, section_id, chain)
                target_origin = transform_record["target_grid_origin_um"]
            else:
                group = handle["sections"][source._key(section_id)]
                features = group["features"][:]
                valid = group["valid"][:]
                target_origin = metadata["origin_um"]
                transform_record = {
                    "transform_type": "identity_before_unified_51_53",
                    "source_section_id": section_id,
                    "target_grid_origin_um": target_origin,
                    "target_grid_shape": list(features.shape[:2]),
                    "output_valid_fraction": float(np.asarray(valid, dtype=bool).mean()),
                }
            provenance = dict(metadata.get("provenance", {}))
            provenance.update(
                {
                    "release": RELEASE,
                    "source_store": str(source_path.resolve()),
                    "source_coordinate_frame": metadata["coordinate_frame"],
                    "coordinate_frame": OUTPUT_FRAME,
                    "unified_alignment_applied": int(section_id) >= NEW_EDGE_SECTION,
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
        "output_sections": written,
        "feature_dim": int(output.feature_dim),
        "coordinate_frame": OUTPUT_FRAME,
        "valid_fraction_by_section": valid_fractions,
    }


def _copy_table(
    source_table_path: Path,
    output_table_path: Path,
    source_transform_dir: Path,
    output_transform_dir: Path,
    chain,
    old_chain,
    new_edge: EdgeAlignment,
    kept_ids: list[int],
) -> dict:
    source = pd.read_parquet(source_table_path).copy()
    source["section_id"] = pd.to_numeric(source["section_id"], errors="raise").astype(int)
    kept = source[source["section_id"].isin(set(kept_ids))].copy()
    if set(kept.section_id) != set(kept_ids):
        raise KeyError("source table and kept section IDs differ")
    output_transform_dir.mkdir(parents=True, exist_ok=True)
    chain_json = json.dumps(
        {
            "release": RELEASE,
            "coordinate_frame": OUTPUT_FRAME,
            "composition": "new_51_to_53_edge_after_v9_53_to_55_postrotate_chain",
            "base_chain": old_chain.metadata(),
            "new_edge_summary": _read_json(NEW_EDGE_ROOT / "direct_stalign_summary.json"),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    for row in kept.itertuples(index=False):
        sid = int(row.section_id)
        source_path = source_transform_dir / Path(str(row.section_transform_path)).name
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        arrays = _load_npz_arrays(source_path)
        old_matrix = np.asarray(arrays["matrix"], dtype=float)
        if sid >= NEW_EDGE_SECTION:
            final_matrix = np.asarray(new_edge.affine, dtype=float) @ np.asarray(old_chain.coarse_affine, dtype=float) @ old_matrix
        else:
            final_matrix = old_matrix
        arrays["source_coordinate_frame"] = np.asarray(str(row.coordinate_frame))
        arrays["matrix_before_unified_alignment_v11"] = old_matrix
        arrays["matrix"] = final_matrix
        arrays["unified_alignment_release"] = np.asarray(RELEASE)
        arrays["unified_alignment_coordinate_frame"] = np.asarray(OUTPUT_FRAME)
        arrays["unified_alignment_applied"] = np.asarray([int(sid >= NEW_EDGE_SECTION)], dtype=np.uint8)
        arrays["unified_alignment_chain_json"] = np.asarray(chain_json)
        arrays["unified_alignment_coarse_affine"] = (
            np.asarray(new_edge.affine, dtype=float) @ np.asarray(old_chain.coarse_affine, dtype=float)
            if sid >= NEW_EDGE_SECTION
            else np.eye(3, dtype=float)
        )
        output_path = output_transform_dir / source_path.name
        np.savez_compressed(output_path, **arrays)
        kept.loc[kept.section_id == sid, "section_transform_path"] = str(output_path.resolve())
    kept["coordinate_frame"] = OUTPUT_FRAME
    kept["unified_alignment_release"] = RELEASE
    kept["unified_alignment_applied"] = kept["section_id"] >= NEW_EDGE_SECTION
    kept["unified_alignment_chain"] = chain_json
    kept["removed_section_ids"] = json.dumps(list(REMOVED_SECTION_IDS))
    kept["cutoff_max_section_id"] = CUTOFF_MAX_SECTION_ID
    kept.to_parquet(output_table_path, index=False)
    return {
        "source_table": str(source_table_path.resolve()),
        "output_table": str(output_table_path.resolve()),
        "output_section_count": int(len(kept)),
        "output_section_ids": [int(x) for x in kept.section_id],
    }


def _materialize_h5ad(
    source_path: Path,
    output_path: Path,
    anchors_dir: Path,
    transforms_dir: Path,
    chain,
    kept_ids: list[int],
) -> dict:
    source = ad.read_h5ad(source_path)
    if "section_id" not in source.obs:
        raise KeyError("Source H5AD requires obs['section_id']")
    section_values = pd.to_numeric(source.obs["section_id"], errors="raise").astype(int).to_numpy()
    keep = np.isin(section_values, np.asarray(kept_ids, dtype=int))
    source = source[keep].copy()
    section_values = section_values[keep]
    source_key = "spatial_final_manual_v6" if "spatial_final_manual_v6" in source.obsm else "spatial_st_corrected"
    current = np.asarray(source.obsm[source_key], dtype=np.float64)
    source.obsm["spatial_before_unified_alignment_v11"] = current.astype(np.float32)
    final = current.copy()
    for sid in sorted(np.unique(section_values)):
        rows = section_values == int(sid)
        if int(sid) >= NEW_EDGE_SECTION:
            final[rows] = _forward(chain, current[rows], int(sid))
    for key in ("spatial", "spatial_registered", "spatial_st_corrected", "spatial_final_manual_v6"):
        if key in source.obsm:
            source.obsm[key] = final.astype(np.float32)
    source.obsm["spatial_unified_51_53_v11"] = final.astype(np.float32)
    source.obs["coordinate_frame"] = OUTPUT_FRAME
    source.obs["unified_alignment_release"] = RELEASE
    source.obs["unified_alignment_applied"] = section_values >= NEW_EDGE_SECTION
    source.obs["x_unified_51_53_um"] = final[:, 0]
    source.obs["y_unified_51_53_um"] = final[:, 1]
    source.obs["registration_transform_path"] = [
        str((transforms_dir / f"section-{int(sid)}-9957__g0__section-{int(sid)}.npz").resolve())
        for sid in section_values
    ]
    source.uns["deepspatial_unified_alignment_v11"] = {
        "release": RELEASE,
        "coordinate_frame": OUTPUT_FRAME,
        "source_h5ad": str(source_path.resolve()),
        "active_coordinate_key": "spatial_unified_51_53_v11",
        "fixed_section": 51,
        "new_edge_section": 53,
        "removed_section_ids": list(REMOVED_SECTION_IDS),
        "cutoff_max_section_id": CUTOFF_MAX_SECTION_ID,
        "kept_section_ids": kept_ids,
        "composition": "new_51_to_53_edge_after_v9_53_to_55_postrotate_chain",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source.write_h5ad(output_path, compression="gzip")
    anchors_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for sid in kept_ids:
        anchor = source[section_values == int(sid)].copy()
        anchor.uns["deepspatial_series"] = {
            "series_id": "9957__g0",
            "section_id": int(sid),
            "section_thickness_um": 5.0,
            "coordinate_frame": OUTPUT_FRAME,
            "release": RELEASE,
        }
        path = anchors_dir / f"section-{int(sid):03d}.h5ad"
        anchor.write_h5ad(path, compression="gzip")
        records.append({"section_id": int(sid), "path": str(path.resolve()), "n_obs": int(anchor.n_obs)})
    return {
        "source": str(source_path.resolve()),
        "output": str(output_path.resolve()),
        "n_obs": int(source.n_obs),
        "section_ids": kept_ids,
        "anchor_records": records,
        "active_coordinate_key": "spatial_unified_51_53_v11",
    }


def _copy_audit_artifacts(output_prep: Path) -> dict:
    audit = output_prep / "alignment_audit"
    audit.mkdir(parents=True, exist_ok=False)
    targets = {}
    for label, source in (
        ("v9_53_to_55_postrotate", V9_DIRECT_ROOT),
        ("new_51_to_53", NEW_EDGE_ROOT),
    ):
        target = audit / label
        shutil.copytree(source, target)
        targets[label] = str(target.resolve())
    return targets


def _check_chain(chain, kept_ids: list[int]) -> dict:
    points = np.array([[1000.0, 1000.0], [2500.0, 1800.0], [4200.0, 3500.0]], dtype=float)
    roundtrip = {}
    for sid in (51, 53, 55):
        mapped = _forward(chain, points, sid)
        recovered = _inverse(chain, mapped, sid)
        error = np.linalg.norm(recovered - points, axis=1)
        # STalign's saved raster maps are interpolated on a finite grid; the
        # forward/reverse interpolation is not mathematically exact at every
        # arbitrary point.  Reject non-finite/excessive excursions, while
        # recording both max and median error for the audit manifest.
        if not np.isfinite(error).all() or float(error.max()) > 200.0:
            raise AssertionError(f"chain roundtrip failed for section {sid}: {error}")
        roundtrip[str(sid)] = {
            "max_error_um": float(error.max()),
            "median_error_um": float(np.median(error)),
        }
    if 54 in kept_ids or 91 not in kept_ids:
        raise AssertionError("section filtering contract failed")
    return {"roundtrip_max_error_um_by_section": roundtrip, "section_51_fixed": True}


def materialize(*, output_prep: Path, output_reg: Path, device: str) -> dict:
    if output_prep.exists() or output_reg.exists():
        raise FileExistsError(f"refusing to overwrite {output_prep} or {output_reg}")
    chain, chain_inputs, new_edge, old_edge = _load_chain(device)
    source_table = pd.read_parquet(SOURCE_PREP / "he_sections.parquet")
    source_ids = sorted(int(x) for x in source_table["section_id"].unique())
    kept_ids = [sid for sid in source_ids if sid not in REMOVED_SECTION_IDS and sid <= CUTOFF_MAX_SECTION_ID]
    if 51 not in kept_ids or 53 not in kept_ids or 91 not in kept_ids:
        raise AssertionError(f"required sections absent: {kept_ids}")
    chain_check = _check_chain(chain, kept_ids)

    output_prep.mkdir(parents=True, exist_ok=False)
    output_reg.mkdir(parents=True, exist_ok=False)
    transform_dir = output_prep / "registration_transforms_unified_51_53_v11"
    table_result = _copy_table(
        SOURCE_PREP / "he_sections.parquet",
        output_prep / "he_sections.parquet",
        SOURCE_PREP / "registration_transforms_final_manual_alignment_v6",
        transform_dir,
        chain,
        OLD_MATERIALIZER.LatestAlignmentChain(
            old_edge, None, device=device
        ),
        new_edge,
        kept_ids,
    )
    audit = _copy_audit_artifacts(output_prep)
    kept = [str(sid) for sid in kept_ids]
    uni2 = _copy_feature_store(
        SOURCE_PREP / "uni2_features_final_manual_alignment_v6.h5",
        output_prep / "uni2_features_unified_51_53_v11.h5",
        kept,
        chain,
        "uni2",
    )
    nucleus = _copy_feature_store(
        SOURCE_PREP / "nucleus_path_final_manual_alignment_v6" / "nucleus_features.h5",
        output_prep / "nucleus_path_unified_51_53_v11" / "nucleus_features.h5",
        kept,
        chain,
        "nucleus",
    )
    h5ad = _materialize_h5ad(
        SOURCE_REG / "9957_g0__st_to_final_manual_alignment_v6.h5ad",
        output_reg / "9957_g0__st_to_unified_51_53_v11.h5ad",
        output_prep / "anchors",
        transform_dir,
        chain,
        kept_ids,
    )
    manifest = {
        "status": "materialized",
        "release": RELEASE,
        "coordinate_frame": OUTPUT_FRAME,
        "source_prep": str(SOURCE_PREP.resolve()),
        "source_reg": str(SOURCE_REG.resolve()),
        "new_edge_root": str(NEW_EDGE_ROOT.resolve()),
        "v9_direct_root": str(V9_DIRECT_ROOT.resolve()),
        "fixed_section": 51,
        "new_edge_section": 53,
        "removed_section_ids": list(REMOVED_SECTION_IDS),
        "cutoff_max_section_id": CUTOFF_MAX_SECTION_ID,
        "source_section_ids": source_ids,
        "output_section_ids": kept_ids,
        "section_54_removed": True,
        "section_91_kept": True,
        "composition": "new_51_to_53_edge_after_v9_53_to_55_postrotate_chain",
        "chain_inputs": chain_inputs,
        "chain_checks": chain_check,
        "audit_artifacts": audit,
        "he_table": table_result,
        "uni2": uni2,
        "nucleus": nucleus,
        "h5ad": h5ad,
        "cache_policy": "rebuild UOT/path caches because the unified coordinate frame changed",
    }
    _write_json(output_prep / "unified_alignment_manifest.json", manifest)
    _write_json(output_reg / "unified_alignment_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prep", type=Path, default=DEFAULT_OUTPUT_PREP)
    parser.add_argument("--output-reg", type=Path, default=DEFAULT_OUTPUT_REG)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    args = parser.parse_args()
    result = materialize(
        output_prep=args.output_prep.resolve(),
        output_reg=args.output_reg.resolve(),
        device=args.device,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "release": result["release"],
                "output_section_count": len(result["output_section_ids"]),
                "section_54_removed": result["section_54_removed"],
                "section_91_kept": result["section_91_kept"],
                "h5ad": result["h5ad"]["output"],
                "uni2": result["uni2"]["output_store"],
                "nucleus": result["nucleus"]["output_store"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
