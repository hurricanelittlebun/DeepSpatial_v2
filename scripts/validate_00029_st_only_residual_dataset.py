#!/usr/bin/env python3
"""Validate and describe the canonical 00029/g0 ST-only data package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import anndata as ad
import h5py
import numpy as np
import pandas as pd

# Make direct ``python scripts/...py`` execution independent of the current
# working directory.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepspatial.data_utils.uot_cache import load_sparse_coupling
from deepspatial.histology import FeatureStore


CORRECTED_KEY = "spatial_st_corrected"
ORIGINAL_KEY = "spatial_registered"


def make_final_config(
    *,
    anchor_dir: str,
    feature_path: str,
    nucleus_feature_path: str,
    path_cache: str,
    uot_dir: str,
    coordinate_frame: str,
) -> dict:
    """Return the explicit DeepSpatial configuration for the corrected run."""

    if not coordinate_frame:
        raise ValueError("coordinate_frame must be non-empty")
    return {
        "dataset": "00029/g0",
        "coordinate_frame": str(coordinate_frame),
        "spatial_key": CORRECTED_KEY,
        "original_spatial_key": ORIGINAL_KEY,
        "z_key": "z_um",
        "he_policy": "unchanged_registered_he",
        "serial_he_transform_policy": "identity_residual",
        "apply_external_residual_at_runtime": False,
        "use_histology": True,
        "use_morphology_uot": True,
        "use_morphology_path": True,
        "use_nucleus_path": True,
        "anchor_dir": str(anchor_dir),
        "feature_path": str(feature_path),
        "nucleus_feature_path": str(nucleus_feature_path),
        "uot_dir": str(uot_dir),
        "path_cache": str(path_cache),
        "note": "Residual affine is materialized only in ST anchor H5ADs; H&E is fixed.",
    }


def validate_he_table(table: pd.DataFrame, coordinate_frame: str) -> dict:
    """Validate that the H&E table has no residual correction applied."""

    required = {"section_id", "coordinate_frame", "residual_applied_to_he"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise KeyError(f"H&E table missing columns: {missing}")
    frames = set(table["coordinate_frame"].astype(str))
    if frames != {str(coordinate_frame)}:
        raise ValueError(
            f"H&E table coordinate frame {sorted(frames)} does not match {coordinate_frame!r}"
        )
    if table["residual_applied_to_he"].astype(bool).any():
        raise ValueError("H&E residual affine was applied to at least one section")
    policies = set(table["he_coordinate_policy"].astype(str))
    if policies != {"original_registered_only"}:
        raise ValueError(f"Unexpected H&E coordinate policies: {sorted(policies)}")
    return {
        "n_sections": int(len(table)),
        "coordinate_frame": str(coordinate_frame),
        "residual_applied_to_he": False,
        "policies": sorted(policies),
    }


def _validate_nuclei_table(table: pd.DataFrame, coordinate_frame: str) -> dict:
    required = {
        "section_id",
        "x_registered_um",
        "y_registered_um",
        "coordinate_frame",
        "residual_applied_to_he",
        "he_coordinate_policy",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise KeyError(f"Nuclei table missing columns: {missing}")
    if set(table["coordinate_frame"].astype(str)) != {str(coordinate_frame)}:
        raise ValueError("Nuclei table coordinate frame does not match canonical frame")
    if table["residual_applied_to_he"].astype(bool).any():
        raise ValueError("Residual affine was applied to H&E nuclear coordinates")
    if {"x_fixed_he_um", "y_fixed_he_um"}.intersection(table.columns):
        raise ValueError("Fixed-frame nuclear coordinate columns leaked into ST-only table")
    return {
        "n_nuclei": int(len(table)),
        "n_sections": int(table["section_id"].nunique()),
        "coordinate_frame": str(coordinate_frame),
        "residual_applied_to_he": False,
    }


def _section_id(path: Path) -> int:
    match = re.search(r"section-(\d+)", path.name)
    if match is None:
        raise ValueError(f"Cannot parse section ID from {path}")
    return int(match.group(1))


def _validate_anchors(anchor_dir: Path, coordinate_frame: str) -> dict:
    rows = []
    for path in sorted(anchor_dir.glob("section-*.h5ad")):
        value = ad.read_h5ad(path, backed="r")
        try:
            if ORIGINAL_KEY not in value.obsm or CORRECTED_KEY not in value.obsm:
                raise KeyError(f"{path} must contain both ST coordinate keys")
            original = np.asarray(value.obsm[ORIGINAL_KEY], dtype=np.float64)
            corrected = np.asarray(value.obsm[CORRECTED_KEY], dtype=np.float64)
            if original.shape != (value.n_obs, 2) or corrected.shape != (value.n_obs, 2):
                raise ValueError(f"{path} has invalid ST coordinate shapes")
            if not np.isfinite(original).all() or not np.isfinite(corrected).all():
                raise ValueError(f"{path} has non-finite ST coordinates")
            contract = value.uns.get("deepspatial_coordinate_contract", {})
            if str(contract.get("coordinate_frame")) != str(coordinate_frame):
                raise ValueError(f"{path} has an unexpected coordinate frame contract")
            if not bool(contract.get("st_residual_applied")):
                raise ValueError(f"{path} does not record ST residual application")
            if bool(contract.get("h_and_e_residual_applied")):
                raise ValueError(f"{path} records residual application to H&E")
            rows.append(
                {
                    "section_id": _section_id(path),
                    "n_obs": int(value.n_obs),
                    "corrected_coordinate_changed": bool(
                        not np.array_equal(original, corrected)
                    ),
                }
            )
        finally:
            value.file.close()
    if len(rows) < 2:
        raise ValueError("At least two corrected anchors are required")
    return {"n_anchors": len(rows), "sections": rows}


def _validate_uot(
    uot_dir: Path,
    anchor_dir: Path,
    coordinate_frame: str,
    feature_revision: str,
) -> dict:
    manifest_path = uot_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("coordinate_frame") != str(coordinate_frame):
        raise ValueError("UOT manifest coordinate frame does not match canonical frame")
    if manifest.get("spatial_key") != CORRECTED_KEY:
        raise ValueError("UOT manifest does not use spatial_st_corrected")
    if manifest.get("feature_store_revision") != str(feature_revision):
        raise ValueError("UOT feature revision does not match UNI2 feature store")
    anchor_sizes = {}
    for path in sorted(anchor_dir.glob("section-*.h5ad")):
        value = ad.read_h5ad(path, backed="r")
        try:
            anchor_sizes[_section_id(path)] = int(value.n_obs)
        finally:
            value.file.close()
    rows = []
    for path in sorted(uot_dir.glob("pair-*.npz")):
        match = re.match(r"pair-(\d+)-(\d+)\.npz$", path.name)
        if match is None:
            raise ValueError(f"Invalid UOT pair filename {path.name}")
        s0, s1 = int(match.group(1)), int(match.group(2))
        coupling, metadata = load_sparse_coupling(path)
        if metadata.get("coordinate_frame") != str(coordinate_frame):
            raise ValueError(f"{path} has an unexpected coordinate frame")
        if metadata.get("spatial_key") != CORRECTED_KEY:
            raise ValueError(f"{path} has an unexpected spatial key")
        if tuple(coupling.shape) != (anchor_sizes[s0], anchor_sizes[s1]):
            raise ValueError(f"{path} shape does not match corrected anchor sizes")
        rows.append({"pair": path.stem, "edge_count": int(len(coupling))})
    expected = list(zip(sorted(anchor_sizes)[:-1], sorted(anchor_sizes)[1:]))
    observed = [(int(row["pair"].split("-")[1]), int(row["pair"].split("-")[2])) for row in rows]
    if observed != expected:
        raise ValueError(f"UOT pair sequence {observed} does not match anchors {expected}")
    return {
        "n_pairs": len(rows),
        "total_edges": int(sum(row["edge_count"] for row in rows)),
        "manifest": str(manifest_path.resolve()),
    }


def _validate_feature_store(path: Path, coordinate_frame: str, expected_sections: list[str]) -> dict:
    store = FeatureStore(path)
    if store.sections != expected_sections:
        raise ValueError(
            f"Feature store sections {store.sections} do not match {expected_sections}"
        )
    frames = {str(store.metadata[sid]["coordinate_frame"]) for sid in store.sections}
    if frames != {str(coordinate_frame)}:
        raise ValueError(f"Feature store frame {sorted(frames)} does not match canonical frame")
    return {
        "path": str(path.resolve()),
        "revision": store.revision,
        "n_sections": len(store.sections),
        "feature_dim": int(store.feature_dim),
        "coordinate_frame": str(coordinate_frame),
    }


def _validate_path_cache(path: Path, coordinate_frame: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        path_count = len(handle.get("paths", {}))
        fallback_count = 0
        for group in handle.get("paths", {}).values():
            metadata = json.loads(group.attrs["metadata"])
            if metadata.get("coordinate_frame") != str(coordinate_frame):
                raise ValueError("Path cache contains a non-canonical coordinate frame")
            fallback_count += bool(json.loads(group.attrs.get("morphology_fallback_sections", "[]")))
    return {
        "path": str(path.resolve()),
        "n_paths": int(path_count),
        "fallback_paths": int(fallback_count),
        "coordinate_frame": str(coordinate_frame),
    }


def validate_dataset(
    dataset_dir: Path | str,
    *,
    feature_path: Path | str,
    nucleus_feature_path: Path | str,
    uot_dir: Path | str,
    path_cache: Path | str,
    coordinate_frame: str,
) -> dict:
    dataset_dir = Path(dataset_dir)
    anchor_dir = dataset_dir / "anchors"
    he_path = dataset_dir / "he_sections.parquet"
    nuclei_path = dataset_dir / "nuclei_registered.parquet"
    if not he_path.is_file():
        raise FileNotFoundError(he_path)
    if not nuclei_path.is_file():
        raise FileNotFoundError(nuclei_path)
    anchor_result = _validate_anchors(anchor_dir, coordinate_frame)
    he_result = validate_he_table(pd.read_parquet(he_path), coordinate_frame)
    nuclei_result = _validate_nuclei_table(pd.read_parquet(nuclei_path), coordinate_frame)
    he_store = FeatureStore(feature_path)
    expected_sections = he_store.sections
    he_store_result = _validate_feature_store(Path(feature_path), coordinate_frame, expected_sections)
    nucleus_store_result = _validate_feature_store(
        Path(nucleus_feature_path), coordinate_frame, expected_sections
    )
    uot_result = _validate_uot(
        Path(uot_dir), anchor_dir, coordinate_frame, he_store.revision
    )
    path_result = _validate_path_cache(Path(path_cache), coordinate_frame)
    return {
        "status": "pass",
        "dataset": str(dataset_dir.resolve()),
        "coordinate_frame": str(coordinate_frame),
        "st_coordinate_key": CORRECTED_KEY,
        "h_and_e_residual_applied": False,
        "anchor": anchor_result,
        "he": he_result,
        "nuclei": nuclei_result,
        "uni2": he_store_result,
        "nucleus_features": nucleus_store_result,
        "uot": uot_result,
        "path_cache": path_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--feature-store", type=Path, required=True)
    parser.add_argument("--nucleus-feature-store", type=Path, required=True)
    parser.add_argument("--uot-dir", type=Path, required=True)
    parser.add_argument("--path-cache", type=Path, required=True)
    parser.add_argument("--coordinate-frame", default="00029__g0_registered")
    args = parser.parse_args()
    result = validate_dataset(
        args.dataset_dir,
        feature_path=args.feature_store,
        nucleus_feature_path=args.nucleus_feature_store,
        uot_dir=args.uot_dir,
        path_cache=args.path_cache,
        coordinate_frame=args.coordinate_frame,
    )
    config = make_final_config(
        anchor_dir=str((args.dataset_dir / "anchors").resolve()),
        feature_path=str(args.feature_store.resolve()),
        nucleus_feature_path=str(args.nucleus_feature_store.resolve()),
        path_cache=str(args.path_cache.resolve()),
        uot_dir=str(args.uot_dir.resolve()),
        coordinate_frame=args.coordinate_frame,
    )
    output_dir = args.dataset_dir
    (output_dir / "deepspatial_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n"
    )
    (output_dir / "validation_report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
