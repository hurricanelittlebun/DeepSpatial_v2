#!/usr/bin/env python3
"""Materialize non-destructive ST-to-fixed-H&E candidate group H5ADs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GROUP_DIR = DEFAULT_ROOT / "data/normalized_groups_v1"
DEFAULT_OUTPUT_ROOT = DEFAULT_ROOT / "data"


def parse_group_stem(stem: str) -> tuple[str, str]:
    match = re.fullmatch(r"(.+)_g(\d+)", str(stem))
    if match is None:
        raise ValueError(f"not a per-group stem: {stem!r}")
    return match.group(1), f"g{match.group(2)}"


def candidate_h5ad_path(output_root: Path | str, group_stem: str) -> Path:
    parse_group_stem(group_stem)
    return (
        Path(output_root)
        / group_stem
        / "registration_correction_he_v1"
        / f"{group_stem}__st_to_fixed_he_candidate.h5ad"
    )


def assemble_corrected_coordinates(
    original: np.ndarray,
    section_ids: np.ndarray,
    candidates: dict[int, np.ndarray],
) -> np.ndarray:
    """Reassemble section-wise candidate coordinates in original row order."""

    original = np.asarray(original, dtype=np.float64)
    section_ids = np.asarray(section_ids)
    if original.ndim != 2 or original.shape[1] != 2:
        raise ValueError("original coordinates must have shape [N, 2]")
    if len(section_ids) != len(original):
        raise ValueError("section_ids and original coordinates have different lengths")
    if not np.isfinite(original).all():
        raise ValueError("original coordinates contain non-finite values")
    corrected = np.empty_like(original)
    seen: set[int] = set()
    for raw_section_id in np.unique(section_ids):
        section_id = int(raw_section_id)
        if section_id not in candidates:
            raise KeyError(f"missing candidate coordinates for section {section_id}")
        indices = np.flatnonzero(section_ids == raw_section_id)
        value = np.asarray(candidates[section_id], dtype=np.float64)
        if value.shape != (len(indices), 2):
            raise ValueError(
                f"candidate shape for section {section_id} is {value.shape}; "
                f"expected {(len(indices), 2)}"
            )
        if not np.isfinite(value).all():
            raise ValueError(f"candidate coordinates for section {section_id} are non-finite")
        corrected[indices] = value
        seen.add(section_id)
    if seen != {int(value) for value in np.unique(section_ids)}:
        raise ValueError("candidate section IDs do not match source section IDs")
    return corrected


def encode_section_summary(summary: list[dict[str, Any]]) -> str:
    """Encode structured section records as an AnnData/ HDF5-safe string."""

    return json.dumps(summary, ensure_ascii=False, sort_keys=True)


def _discover_groups(group_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in group_dir.glob("*_g*.h5ad")
        if re.fullmatch(r".+_g\d+", path.stem) and path.stem != "00029_g0"
    )


def materialize_group(
    source_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    import anndata as ad
    import pandas as pd

    group_stem = source_path.stem
    sample_id, group_id = parse_group_stem(group_stem)
    correction_dir = output_root / group_stem / "registration_correction_he_v1"
    metrics_path = correction_dir / "residual_fit_metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    destination = candidate_h5ad_path(output_root, group_stem)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")

    metrics = json.loads(metrics_path.read_text())
    if metrics.get("errors"):
        raise ValueError(f"cannot materialize {group_stem} with fit errors: {metrics['errors']}")
    records = metrics.get("sections", [])
    candidates: dict[int, np.ndarray] = {}
    record_by_section: dict[int, dict[str, Any]] = {}
    for record in records:
        section_id = int(record["section_id"])
        candidate_path = Path(record["coordinate_output"])
        if not candidate_path.is_file():
            raise FileNotFoundError(candidate_path)
        with np.load(candidate_path, allow_pickle=False) as value:
            candidates[section_id] = np.asarray(value["spatial_st_corrected"], dtype=np.float64)
        record_by_section[section_id] = record

    value = ad.read_h5ad(source_path)
    if "spatial_registered" not in value.obsm:
        raise KeyError(f"{source_path} lacks obsm['spatial_registered']")
    original = np.asarray(value.obsm["spatial_registered"], dtype=np.float64)
    if "section_id" not in value.obs.columns:
        raise KeyError(f"{source_path} lacks obs['section_id']")
    section_ids = pd.to_numeric(value.obs["section_id"], errors="coerce").to_numpy()
    if not np.isfinite(section_ids).all():
        raise ValueError(f"{source_path} has non-finite section_id")
    section_ids = section_ids.astype(int)
    corrected = assemble_corrected_coordinates(original, section_ids, candidates)
    expected_sections = set(np.unique(section_ids).tolist())
    if expected_sections != set(record_by_section):
        raise ValueError(
            f"section mismatch for {group_stem}: source={sorted(expected_sections)}, "
            f"fit={sorted(record_by_section)}"
        )

    value.obsm["spatial_st_corrected"] = corrected
    status = pd.Series(section_ids, index=value.obs.index).map(
        {section_id: record["status"] for section_id, record in record_by_section.items()}
    )
    improvement = pd.Series(section_ids, index=value.obs.index).map(
        {section_id: float(record["improvement"]) for section_id, record in record_by_section.items()}
    )
    transform_path = pd.Series(section_ids, index=value.obs.index).map(
        {section_id: str(record["transform_output"]) for section_id, record in record_by_section.items()}
    )
    value.obs["st_residual_applied"] = True
    value.obs["st_residual_status"] = status.astype(str).to_numpy()
    value.obs["st_residual_improvement"] = improvement.to_numpy(dtype=float)
    value.obs["st_residual_transform_path"] = transform_path.astype(str).to_numpy()
    section_manifest = []
    for section_id in sorted(record_by_section):
        record = record_by_section[section_id]
        section_manifest.append(
            {
                "section_id": section_id,
                "status": record["status"],
                "improvement": record["improvement"],
                "baseline_inside_fraction": record["baseline"]["inside_fraction"],
                "candidate_inside_fraction": record["candidate"]["inside_fraction"],
                "transform_path": record["transform_output"],
            }
        )
    contract = {
        "format": "deepspatial-st-to-fixed-he-residual-v1",
        "sample_id": sample_id,
        "group_id": group_id,
        "coordinate_units": "micrometer",
        "coordinate_frame": f"{sample_id}__{group_id}_registered",
        "h_and_e_coordinate_frame": f"{sample_id}__{group_id}_registered",
        "original_coordinate_key": "spatial_registered",
        "corrected_coordinate_key": "spatial_st_corrected",
        "st_residual_applied": True,
        "h_and_e_residual_applied": False,
        "transform_direction": "spatial_st_corrected = matrix_current_g_to_fixed_he @ spatial_registered",
        "h_and_e_policy": "fixed; no H&E coordinates or images were modified",
        "source_h5ad": str(source_path.resolve()),
        "fit_metrics": str(metrics_path.resolve()),
        "section_summary_json": encode_section_summary(section_manifest),
        "note": "Candidate package; use spatial_st_corrected explicitly after QC review.",
    }
    value.uns["deepspatial_residual_correction"] = contract
    value.uns["deepspatial_coordinate_contract"] = contract.copy()
    destination.parent.mkdir(parents=True, exist_ok=True)
    value.write_h5ad(destination, compression="gzip")
    manifest = {
        "status": "candidate_generated_not_published",
        "source_h5ad": str(source_path.resolve()),
        "candidate_h5ad": str(destination.resolve()),
        "n_obs": int(value.n_obs),
        "n_vars": int(value.n_vars),
        "coordinate_key": "spatial_st_corrected",
        "original_coordinate_key": "spatial_registered",
        "h_and_e_residual_applied": False,
        "sections": section_manifest,
    }
    (correction_dir / "candidate_h5ad_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-dir", type=Path, default=DEFAULT_GROUP_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--group", action="append")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.group:
        paths = [args.group_dir / f"{group}.h5ad" for group in args.group]
    else:
        paths = _discover_groups(args.group_dir)
    summaries = []
    for source_path in paths:
        if source_path.stem == "00029_g0":
            raise ValueError("00029_g0 is intentionally excluded")
        print(f"[materialize] {source_path.name}", flush=True)
        manifest = materialize_group(source_path, args.output_root)
        summaries.append(manifest)
        print(f"  wrote {manifest['candidate_h5ad']}", flush=True)
    summary_path = args.output_root / "st_to_he_residual_v1_candidate_h5ad_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False) + "\n")
    print(f"[done] {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
