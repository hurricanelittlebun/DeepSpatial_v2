"""Compare old/candidate ST coordinates against the fixed UNI2 feature grids.

This is deliberately non-destructive.  The H&E feature store is already in the
fixed registered-H&E frame; only the queried ST coordinates change.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deepspatial.histology import FeatureStore


def _section_id(path: Path) -> int:
    return int(path.stem.split("section-")[-1].split("__", 1)[0])


def _query(store: FeatureStore, section_id: int, xy: np.ndarray) -> dict:
    features, valid, quality = store.get_feature(
        str(section_id),
        xy,
        return_valid=True,
        return_quality=True,
    )
    del features
    valid = valid.detach().cpu().numpy().astype(bool)
    quality = quality.detach().cpu().numpy().astype(np.uint8)
    return {
        "feature_valid_fraction": float(valid.mean()),
        "quality_full_bilinear_fraction": float(np.mean(quality == 1)),
        "quality_partial_bilinear_fraction": float(np.mean(quality == 2)),
        "quality_nearest_fallback_fraction": float(np.mean(quality == 3)),
        "quality_invalid_fraction": float(np.mean(quality == 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", default="data/00029_g0")
    parser.add_argument("--feature-store", default="data/00029_g0/uni2_features.h5")
    parser.add_argument(
        "--candidate-dir",
        default="data/00029_g0/anchors_st_to_fixed_he_candidate",
    )
    parser.add_argument(
        "--output",
        default="data/00029_g0/registration_correction_v1/feature_coverage_comparison",
    )
    args = parser.parse_args()

    series = Path(args.series)
    candidate_dir = Path(args.candidate_dir)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    store = FeatureStore(args.feature_store)

    rows = []
    source_paths = sorted(
        (series / "anchors").glob("section-*.h5ad"), key=_section_id
    )
    if not source_paths:
        raise FileNotFoundError(f"No source anchors under {series / 'anchors'}")

    for source_path in source_paths:
        section_id = _section_id(source_path)
        candidate_path = candidate_dir / (
            f"section-{section_id:03d}__st_to_fixed_he_candidate.h5ad"
        )
        if not candidate_path.is_file():
            raise FileNotFoundError(candidate_path)

        source = ad.read_h5ad(source_path)
        candidate = ad.read_h5ad(candidate_path)
        if source.n_obs != candidate.n_obs:
            raise ValueError(f"Observation count changed for section {section_id}")
        old = np.asarray(source.obsm["spatial_registered"], dtype=np.float64)
        new = np.asarray(candidate.obsm["spatial_st_corrected"], dtype=np.float64)
        if old.shape != new.shape or old.shape[1] != 2:
            raise ValueError(f"Invalid coordinate shape for section {section_id}")

        before = _query(store, section_id, old)
        after = _query(store, section_id, new)
        displacement = np.linalg.norm(new - old, axis=1)
        row = {
            "section_id": section_id,
            "n_cells": int(len(old)),
            "median_st_displacement_um": float(np.median(displacement)),
            "p95_st_displacement_um": float(np.quantile(displacement, 0.95)),
            "max_st_displacement_um": float(displacement.max()),
        }
        row.update({f"before_{key}": value for key, value in before.items()})
        row.update({f"after_{key}": value for key, value in after.items()})
        row["delta_feature_valid_fraction"] = (
            row["after_feature_valid_fraction"]
            - row["before_feature_valid_fraction"]
        )
        rows.append(row)

    frame = pd.DataFrame(rows).sort_values("section_id")
    frame.to_csv(output / "coverage_comparison.csv", index=False)
    (output / "coverage_comparison.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    x = np.arange(len(frame))
    width = 0.36
    axes[0].bar(
        x - width / 2,
        frame["before_feature_valid_fraction"],
        width,
        label="original spatial_registered",
        color="#d95f02",
    )
    axes[0].bar(
        x + width / 2,
        frame["after_feature_valid_fraction"],
        width,
        label="candidate spatial_st_corrected",
        color="#1b9e77",
    )
    axes[0].set(
        title="UNI2 feature query coverage",
        ylabel="valid query fraction",
        ylim=(0, 1.05),
    )
    axes[0].set_xticks(x, frame["section_id"].astype(str))
    axes[0].legend(fontsize=8)
    axes[1].plot(
        frame["section_id"],
        frame["before_quality_nearest_fallback_fraction"],
        "o-",
        label="original nearest fallback",
        color="#d95f02",
    )
    axes[1].plot(
        frame["section_id"],
        frame["after_quality_nearest_fallback_fraction"],
        "o-",
        label="candidate nearest fallback",
        color="#1b9e77",
    )
    axes[1].set(
        title="Feature query quality",
        xlabel="section",
        ylabel="nearest-valid fallback fraction",
        ylim=(0, 1.05),
    )
    axes[1].legend(fontsize=8)
    fig.savefig(output / "coverage_comparison.png", dpi=180)
    plt.close(fig)

    print(output / "coverage_comparison.csv")
    print(output / "coverage_comparison.png")


if __name__ == "__main__":
    main()
