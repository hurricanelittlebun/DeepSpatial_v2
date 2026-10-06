#!/usr/bin/env python3
"""Materialize the reviewed 9957/g0 v3 ST residual into a new H5AD.

The final H&E release is immutable.  This script only changes the ST-side
coordinate representation in a new copy of the source H5AD.  The source file
is never overwritten, and the original coordinate arrays are retained under
explicit ``*_before_st_residual_v3`` keys for auditability.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
GROUP = ROOT / "data" / "9957_g0"
FRAME = "9957__g0_registered_final_he_53_55_v1_chain_v1"
ANCHORS = (1, 11, 21, 31, 41, 51, 61, 71, 81, 91)
SOURCE_H5AD = (
    GROUP
    / "registration_correction_he_final_he_53_55_v1_chain_v1"
    / "9957_g0__st_to_final_he_53_55_v1_chain_v1.h5ad"
)
CANDIDATE_ROOT = GROUP / "st_to_final_he_residual_candidate_v3_global_frame"
HE_TABLE = GROUP / "reconstruction_prep_final_he_53_55_v1_chain_v1" / "he_sections.parquet"
OUTPUT_DIR = GROUP / "registration_final_st_residual_v3_global_frame"
OUTPUT_H5AD = OUTPUT_DIR / "9957_g0__st_to_final_he_53_55_v1_chain_v1__st_residual_v3.h5ad"


def _load_coordinate_helper():
    path = ROOT / "deepspatial" / "data_utils" / "st_residual_v3.py"
    spec = importlib.util.spec_from_file_location("_st_residual_v3_materialization_helpers", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import coordinate helper from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _scalar_text(value: Any) -> str:
    array = np.asarray(value)
    if array.ndim == 0:
        return str(array.item())
    if array.size == 1:
        return str(array.reshape(-1)[0])
    raise ValueError(f"expected a scalar text value, got shape {array.shape}")


def load_candidate_payloads(candidate_root: Path, section_ids: np.ndarray) -> dict[int, dict[str, Any]]:
    payloads: dict[int, dict[str, Any]] = {}
    for raw_sid in np.unique(section_ids):
        sid = int(raw_sid)
        path = candidate_root / "candidate_coordinates" / f"section-{sid:03d}__st_corrected_candidate.npz"
        if not path.is_file():
            raise FileNotFoundError(f"missing v3 candidate for section {sid}: {path}")
        with np.load(path, allow_pickle=False) as payload:
            frame = _scalar_text(payload["coordinate_frame"])
            if frame != FRAME:
                raise ValueError(f"section {sid} candidate frame is {frame!r}, expected {FRAME!r}")
            payloads[sid] = {
                "st_global_base": np.asarray(payload["st_global_base"], dtype=np.float64),
                "st_global_candidate": np.asarray(payload["st_global_candidate"], dtype=np.float64),
                "matrix_base_st_to_fixed_he": np.asarray(
                    payload["matrix_base_st_to_fixed_he"], dtype=np.float64
                ),
                "candidate_path": str(path.resolve()),
                "transform_path": str(
                    (candidate_root / "transforms" / f"section-{sid:03d}__st_to_fixed_he_residual.npz").resolve()
                ),
            }
    return payloads


def _load_section_metrics(candidate_root: Path) -> dict[int, dict[str, Any]]:
    path = candidate_root / "residual_fit_metrics.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(record["section_id"]): record for record in data.get("records", [])}


def _add_before_coordinate_keys(adata: ad.AnnData) -> dict[str, str]:
    copied: dict[str, str] = {}
    for key in ("spatial", "spatial_registered", "spatial_st_corrected"):
        if key in adata.obsm:
            backup_key = f"{key}_before_st_residual_v3"
            if backup_key not in adata.obsm:
                adata.obsm[backup_key] = np.asarray(adata.obsm[key]).copy()
            copied[key] = backup_key
    return copied


def materialize(
    source_h5ad: Path,
    candidate_root: Path,
    output_h5ad: Path,
    he_table: Path,
) -> dict[str, Any]:
    source_h5ad = source_h5ad.resolve()
    candidate_root = candidate_root.resolve()
    output_h5ad = output_h5ad.resolve()
    he_table = he_table.resolve()
    if not source_h5ad.is_file():
        raise FileNotFoundError(source_h5ad)
    if not he_table.is_file():
        raise FileNotFoundError(he_table)
    if output_h5ad.exists():
        raise FileExistsError(f"refusing to overwrite final H5AD: {output_h5ad}")

    source_hash_before = sha256(source_h5ad)
    he_table_hash = sha256(he_table)
    adata = ad.read_h5ad(source_h5ad)
    if "section_id" not in adata.obs:
        raise KeyError("source H5AD is missing obs['section_id']")
    section_ids = pd.to_numeric(adata.obs["section_id"], errors="raise").astype(int).to_numpy()
    observed_sections = tuple(sorted(int(value) for value in np.unique(section_ids)))
    if observed_sections != ANCHORS:
        raise ValueError(f"source anchor sections are {observed_sections}, expected {ANCHORS}")

    payloads = load_candidate_payloads(candidate_root, section_ids)
    helper = _load_coordinate_helper()
    base, candidate = helper.assemble_section_coordinates(section_ids, payloads)
    metrics = _load_section_metrics(candidate_root)

    backup_keys = _add_before_coordinate_keys(adata)
    adata.obsm["spatial_st_v3_base"] = base.astype(np.float32)
    adata.obsm["spatial_st_residual_v3"] = candidate.astype(np.float32)
    adata.obsm["spatial_st_final_he_v3"] = candidate.astype(np.float32)
    # These are the canonical coordinate keys used by existing DeepSpatial
    # consumers.  The old values remain available under explicit backup keys.
    for key in ("spatial", "spatial_registered", "spatial_st_corrected"):
        adata.obsm[key] = candidate.astype(np.float32)

    old_frame = adata.obs["coordinate_frame"].astype(str).to_numpy() if "coordinate_frame" in adata.obs else None
    if old_frame is not None:
        adata.obs["coordinate_frame_before_st_residual_v3"] = old_frame
    adata.obs["coordinate_frame"] = pd.Categorical(
        np.full(adata.n_obs, FRAME, dtype=object), categories=[FRAME]
    )
    adata.obs["st_residual_v3_applied"] = True
    adata.obs["st_residual_v3_section"] = section_ids.astype(np.int64)
    adata.obs["st_residual_v3_transform_path"] = np.asarray(
        [payloads[int(sid)]["transform_path"] for sid in section_ids], dtype=object
    )
    adata.obs["x_st_residual_v3_um"] = candidate[:, 0]
    adata.obs["y_st_residual_v3_um"] = candidate[:, 1]
    adata.obs["x_st_v3_base_um"] = base[:, 0]
    adata.obs["y_st_v3_base_um"] = base[:, 1]
    adata.obs["st_residual_v3_status"] = np.asarray(
        [str(metrics[int(sid)].get("status", "unknown")) for sid in section_ids], dtype=object
    )
    adata.obs["st_residual_v3_inside_valid_fraction"] = np.asarray(
        [float(metrics[int(sid)]["candidate"]["inside_valid_fraction"]) for sid in section_ids],
        dtype=np.float32,
    )
    adata.obs["st_residual_v3_improvement"] = np.asarray(
        [float(metrics[int(sid)]["improvement_inside_valid_fraction"]) for sid in section_ids],
        dtype=np.float32,
    )

    adata.uns["deepspatial_st_residual_v3"] = {
        "release": "9957_g0_st_to_fixed_he_residual_v3_global_frame",
        "coordinate_frame": FRAME,
        "source_h5ad": str(source_h5ad),
        "candidate_root": str(candidate_root),
        "he_table": str(he_table),
        "st_input_obsm": "spatial_he_local",
        "mapping": "spatial_he_local -> immutable final H&E global mapper -> v3 bounded ST residual affine",
        "he_is_fixed": True,
        "h_e_images_masks_and_transforms_modified": False,
        "section_ids": list(ANCHORS),
        "removed_he_section_ids": [54],
        "canonical_obsm": "spatial_st_residual_v3",
        "base_obsm": "spatial_st_v3_base",
        "original_coordinate_backups": backup_keys,
        "transform_paths": {
            str(sid): payloads[sid]["transform_path"] for sid in ANCHORS
        },
    }
    adata.uns["deepspatial_coordinate_contract"] = {
        **dict(adata.uns.get("deepspatial_coordinate_contract", {})),
        "active_st_xy_obsm": "spatial_st_residual_v3",
        "active_coordinate_frame": FRAME,
        "he_coordinate_frame": FRAME,
    }

    output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_h5ad, compression="gzip")
    source_hash_after = sha256(source_h5ad)
    if source_hash_after != source_hash_before:
        raise RuntimeError("source H5AD changed during materialization")
    output_hash = sha256(output_h5ad)
    manifest = {
        "status": "complete",
        "release": "9957_g0_st_to_fixed_he_residual_v3_global_frame",
        "output_h5ad": str(output_h5ad),
        "output_h5ad_sha256": output_hash,
        "source_h5ad": str(source_h5ad),
        "source_h5ad_sha256_before": source_hash_before,
        "source_h5ad_sha256_after": source_hash_after,
        "source_h5ad_unchanged": source_hash_before == source_hash_after,
        "candidate_root": str(candidate_root),
        "he_table": str(he_table),
        "he_table_sha256": he_table_hash,
        "coordinate_frame": FRAME,
        "he_is_fixed": True,
        "h_e_images_masks_and_transforms_modified": False,
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "sections": list(ANCHORS),
        "removed_he_sections": [54],
        "active_obsm": "spatial_st_residual_v3",
        "base_obsm": "spatial_st_v3_base",
        "original_obsm_backups": backup_keys,
        "candidate_section_files": {
            str(sid): payloads[sid]["candidate_path"] for sid in ANCHORS
        },
    }
    (output_h5ad.parent / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-h5ad", type=Path, default=SOURCE_H5AD)
    parser.add_argument("--candidate-root", type=Path, default=CANDIDATE_ROOT)
    parser.add_argument("--he-table", type=Path, default=HE_TABLE)
    parser.add_argument("--output-h5ad", type=Path, default=OUTPUT_H5AD)
    args = parser.parse_args()
    print(json.dumps(materialize(args.source_h5ad, args.candidate_root, args.output_h5ad, args.he_table), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
