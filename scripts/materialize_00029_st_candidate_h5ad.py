#!/usr/bin/env python3
"""Write non-destructive per-anchor H5AD candidates with corrected ST XY."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np


def main() -> None:
    root = Path("/data/buyonggan/DeepSpatial")
    series = root / "data/00029_g0"
    correction = series / "registration_correction_v1"
    output = series / "anchors_st_to_fixed_he_candidate"
    output.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((correction / "residual_fit_metrics.json").read_text())
    manifest = {
        "status": "candidate_not_published",
        "coordinate_key": "spatial_st_corrected",
        "original_coordinate_key_preserved": "spatial_registered",
        "sections": [],
        "note": (
            "Use spatial_key='spatial_st_corrected' explicitly. The default spatial and "
            "spatial_registered keys remain the original coordinates."
        ),
    }
    for record in metrics["sections"]:
        section_id = int(record["section_id"])
        source_path = series / "anchors" / f"section-{section_id:03d}.h5ad"
        candidate_path = correction / "candidate_coordinates" / (
            f"section-{section_id:03d}__st_corrected_candidate.npz"
        )
        destination = output / f"section-{section_id:03d}__st_to_fixed_he_candidate.h5ad"
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite {destination}")
        candidate = np.load(candidate_path, allow_pickle=False)
        corrected = np.asarray(candidate["spatial_st_corrected"], dtype=np.float64)
        value = ad.read_h5ad(source_path)
        original = np.asarray(value.obsm["spatial_registered"], dtype=np.float64)
        saved_original = np.asarray(candidate["spatial_registered_original"], dtype=np.float64)
        if original.shape != saved_original.shape or not np.array_equal(original, saved_original):
            raise ValueError(f"original coordinate mismatch for section {section_id}")
        if corrected.shape != (value.n_obs, 2) or not np.isfinite(corrected).all():
            raise ValueError(f"invalid corrected coordinates for section {section_id}")
        value.obsm["spatial_st_corrected"] = corrected
        value.uns["deepspatial_residual_correction"] = {
            "status": "candidate_not_published",
            "coordinate_units": "micrometer",
            "fixed_reference": "registered H&E mask / g frame",
            "coordinate_key": "spatial_st_corrected",
            "original_coordinate_key": "spatial_registered",
            "transform_path": str(
                correction
                / "transforms"
                / f"section-{section_id:03d}__st_to_fixed_he_residual.npz"
            ),
            "baseline_inside_fraction": float(record["baseline"]["inside_fraction"]),
            "candidate_inside_fraction": float(record["candidate"]["inside_fraction"]),
            "holdout_validation_available": True,
        }
        value.write_h5ad(destination, compression="gzip")
        manifest["sections"].append(
            {
                "section_id": section_id,
                "source": str(source_path),
                "candidate_h5ad": str(destination),
                "n_obs": int(value.n_obs),
                "n_vars": int(value.n_vars),
            }
        )
        print(destination, flush=True)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
