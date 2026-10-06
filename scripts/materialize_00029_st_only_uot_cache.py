#!/usr/bin/env python3
"""Materialize a verified corrected-ST UOT cache in the canonical H&E frame.

The sparse couplings are not recomputed here.  Existing corrected couplings
are accepted only after their endpoint shapes, coordinate key, and UNI2
feature revision agree with the new materialized anchor H5ADs.  Row, column,
and mass arrays are copied byte-for-value through the repository cache API;
only auditable metadata is enriched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import anndata as ad
import numpy as np

# Make direct ``python scripts/...py`` execution independent of the current
# working directory.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepspatial.data_utils.uot_cache import load_sparse_coupling, save_sparse_coupling


PAIR_RE = re.compile(r"^pair-(\d+)-(\d+)\.npz$")
CORRECTED_KEY = "spatial_st_corrected"


def _sha256_array(value: np.ndarray) -> str:
    array = np.asarray(value, dtype="<f8", order="C")
    return hashlib.sha256(array.tobytes()).hexdigest()


def _section_id(path: Path) -> int:
    match = re.search(r"section-(\d+)", path.name)
    if match is None:
        raise ValueError(f"Cannot parse section ID from {path}")
    return int(match.group(1))


def _anchor_metadata(anchor_dir: Path) -> dict[int, dict]:
    result = {}
    for path in sorted(anchor_dir.glob("section-*.h5ad")):
        section_id = _section_id(path)
        value = ad.read_h5ad(path, backed="r")
        try:
            if CORRECTED_KEY not in value.obsm:
                raise KeyError(f"{path} lacks obsm[{CORRECTED_KEY!r}]")
            coordinates = np.asarray(value.obsm[CORRECTED_KEY], dtype=np.float64)
            if coordinates.shape != (value.n_obs, 2) or not np.isfinite(coordinates).all():
                raise ValueError(
                    f"{path} has invalid {CORRECTED_KEY!r}; expected finite [n_obs, 2]"
                )
            result[section_id] = {
                "n_obs": int(value.n_obs),
                "coordinate_sha256": _sha256_array(coordinates),
                "path": str(path.resolve()),
            }
        finally:
            value.file.close()
    if len(result) < 2:
        raise ValueError("At least two corrected anchor H5ADs are required")
    return result


def _pair_ids(path: Path) -> tuple[int, int]:
    match = PAIR_RE.match(path.name)
    if match is None:
        raise ValueError(f"Invalid UOT pair filename: {path.name}")
    return int(match.group(1)), int(match.group(2))


def materialize_verified_uot_cache(
    source_dir: Path | str,
    *,
    anchor_dir: Path | str,
    output_dir: Path | str,
    coordinate_frame: str,
    feature_revision: str,
) -> dict:
    """Copy corrected sparse couplings after validating the new endpoints."""

    source_dir = Path(source_dir)
    anchor_dir = Path(anchor_dir)
    output_dir = Path(output_dir)
    if not source_dir.is_dir():
        raise NotADirectoryError(source_dir)
    if not anchor_dir.is_dir():
        raise NotADirectoryError(anchor_dir)
    if not coordinate_frame or not feature_revision:
        raise ValueError("coordinate_frame and feature_revision are required")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to write into non-empty output directory: {output_dir}"
        )

    anchors = _anchor_metadata(anchor_dir)
    source_pairs = sorted(source_dir.glob("pair-*.npz"), key=lambda p: _pair_ids(p))
    if not source_pairs:
        raise FileNotFoundError(f"No pair-*.npz files found under {source_dir}")
    expected_pairs = list(zip(sorted(anchors)[:-1], sorted(anchors)[1:]))
    observed_pairs = [_pair_ids(path) for path in source_pairs]
    if observed_pairs != expected_pairs:
        raise ValueError(
            "UOT pair sequence does not match corrected anchors: "
            f"observed={observed_pairs}, expected={expected_pairs}"
        )

    validated = []
    for source_path in source_pairs:
        source_section, target_section = _pair_ids(source_path)
        coupling, metadata = load_sparse_coupling(source_path)
        if metadata.get("spatial_key") != CORRECTED_KEY:
            raise ValueError(
                f"{source_path} must use {CORRECTED_KEY!r}; "
                f"got {metadata.get('spatial_key')!r}"
            )
        source_meta = str(metadata.get("source_section", ""))
        target_meta = str(metadata.get("target_section", ""))
        if int(source_meta) != source_section or int(target_meta) != target_section:
            raise ValueError(f"{source_path} metadata section IDs do not match filename")
        source_revision = str(metadata.get("feature_store_revision", ""))
        if source_revision != str(feature_revision):
            raise ValueError(
                f"{source_path} feature revision {source_revision!r} does not match "
                f"{feature_revision!r}"
            )
        existing_frame = metadata.get("coordinate_frame")
        if existing_frame is not None and str(existing_frame) != str(coordinate_frame):
            raise ValueError(
                f"{source_path} has coordinate frame {existing_frame!r}; "
                f"expected {coordinate_frame!r}"
            )
        expected_shape = (anchors[source_section]["n_obs"], anchors[target_section]["n_obs"])
        if tuple(coupling.shape) != expected_shape:
            raise ValueError(
                f"{source_path} shape {coupling.shape} does not match corrected "
                f"anchor shape {expected_shape}"
            )
        validated.append(
            (
                source_path,
                coupling,
                metadata,
                source_section,
                target_section,
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for source_path, coupling, metadata, source_section, target_section in validated:
        output_path = output_dir / source_path.name
        enriched = dict(metadata)
        enriched.update(
            {
                "coordinate_frame": str(coordinate_frame),
                "spatial_key": CORRECTED_KEY,
                "h_and_e_residual_applied": False,
                "st_residual_materialized": True,
                "source_cache": str(source_path.resolve()),
                "source_coordinate_sha256": anchors[source_section]["coordinate_sha256"],
                "target_coordinate_sha256": anchors[target_section]["coordinate_sha256"],
                "cache_materialization": "verified_corrected_sparse_uot_copy",
            }
        )
        save_sparse_coupling(output_path, coupling, enriched)
        records.append(
            {
                "pair": source_path.stem,
                "source": str(source_path.resolve()),
                "output": str(output_path.resolve()),
                "edge_count": int(len(coupling)),
                "shape": [int(coupling.shape[0]), int(coupling.shape[1])],
                "source_coordinate_sha256": enriched["source_coordinate_sha256"],
                "target_coordinate_sha256": enriched["target_coordinate_sha256"],
            }
        )

    manifest = {
        "format": "deepspatial-verified-st-only-uot-v1",
        "status": "complete",
        "coordinate_frame": str(coordinate_frame),
        "spatial_key": CORRECTED_KEY,
        "feature_store_revision": str(feature_revision),
        "source_cache": str(source_dir.resolve()),
        "anchor_dir": str(anchor_dir.resolve()),
        "recomputed": False,
        "h_and_e_residual_applied": False,
        "pairs": records,
        "total_edges": int(sum(row["edge_count"] for row in records)),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--anchor-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--coordinate-frame", required=True)
    parser.add_argument("--feature-revision", default=None)
    parser.add_argument("--feature-store", type=Path, default=None)
    args = parser.parse_args()

    feature_revision = args.feature_revision
    if feature_revision is None and args.feature_store is not None:
        from deepspatial.histology import FeatureStore

        feature_revision = FeatureStore(args.feature_store).revision
    if feature_revision is None:
        parser.error("provide --feature-revision or --feature-store")

    manifest = materialize_verified_uot_cache(
        args.source_dir,
        anchor_dir=args.anchor_dir,
        output_dir=args.output_dir,
        coordinate_frame=args.coordinate_frame,
        feature_revision=feature_revision,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
