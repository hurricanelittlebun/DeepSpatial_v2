"""Build a corrected 00029/g0 morphology path cache from sparse UOT caches.

The endpoint coordinates come from ``spatial_st_corrected`` and are already in
the fixed registered-H&E physical frame.  This script does not read WSI data,
run UNI2, or alter the original path cache.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepspatial.data_utils.uot_cache import load_sparse_coupling
from deepspatial.histology import FeatureStore, MorphologyPathCache


def _section_number(path: Path) -> int:
    return int(path.stem.split("section-")[-1].split("__", 1)[0])


def _load_anchor(path: Path) -> tuple[str, np.ndarray, int, list[str]]:
    obj = ad.read_h5ad(path, backed="r")
    try:
        if "spatial_st_corrected" not in obj.obsm:
            raise KeyError(f"{path} lacks obsm['spatial_st_corrected']")
        if "section_id" not in obj.obs or "z_um" not in obj.obs:
            raise KeyError(f"{path} needs obs['section_id'] and obs['z_um']")
        section_id = str(obj.obs["section_id"].iloc[0])
        z_um = float(obj.obs["z_um"].iloc[0])
        xy = np.asarray(obj.obsm["spatial_st_corrected"], dtype=np.float64)
        names = [str(value) for value in obj.obs_names]
        if xy.shape != (obj.n_obs, 2) or not np.isfinite(xy).all():
            raise ValueError(f"Invalid corrected coordinates in {path}")
        return section_id, xy, int(obj.n_obs), names
    finally:
        obj.file.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--anchor-dir",
        default="data/00029_g0/anchors_st_to_fixed_he_candidate",
    )
    parser.add_argument(
        "--uot-dir",
        default="data/00029_g0/uot_cache_topk64_st_corrected_v1",
    )
    parser.add_argument(
        "--feature-store",
        default="data/00029_g0/uni2_features_raw_corrected.h5",
    )
    parser.add_argument(
        "--nucleus-feature-store",
        default="data/00029_g0/nucleus_path_raw_final/nucleus_features.h5",
    )
    parser.add_argument(
        "--output",
        default="data/00029_g0/train_full_00029_st_corrected_v1/paths.h5",
    )
    parser.add_argument("--pairs", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--candidate-radius-um", type=float, default=100.0)
    parser.add_argument("--candidate-spacing-um", type=float, default=25.0)
    parser.add_argument("--move-weight", type=float, default=0.001)
    parser.add_argument("--morphology-weight", type=float, default=1.0)
    parser.add_argument("--nucleus-weight", type=float, default=1.0)
    parser.add_argument(
        "--coordinate-frame",
        default=None,
        help="Require this frame for morphology and nuclear stores and UOT caches.",
    )
    parser.add_argument("--allow-linear-fallback", action="store_true")
    args = parser.parse_args()

    anchor_dir = Path(args.anchor_dir)
    uot_dir = Path(args.uot_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing path cache: {output}; choose a new path"
        )

    anchor_paths = sorted(anchor_dir.glob("section-*.h5ad"), key=_section_number)
    if len(anchor_paths) < 2:
        raise ValueError(f"Expected at least two candidate anchors under {anchor_dir}")
    anchors = [_load_anchor(path) for path in anchor_paths]
    sections = [item[0] for item in anchors]
    coordinates = [item[1] for item in anchors]
    counts = [item[2] for item in anchors]
    names = [item[3] for item in anchors]
    pair_sizes = np.asarray(counts[:-1], dtype=np.int64) * np.asarray(
        counts[1:], dtype=np.int64
    )
    requested_counts = np.floor(args.pairs * pair_sizes / pair_sizes.sum()).astype(int)

    store = FeatureStore(args.feature_store)
    nucleus_store = FeatureStore(args.nucleus_feature_store)
    store_frames = {
        str(store.metadata[sid]["coordinate_frame"]) for sid in store.sections
    }
    nucleus_frames = {
        str(nucleus_store.metadata[sid]["coordinate_frame"])
        for sid in nucleus_store.sections
    }
    if len(store_frames) != 1 or len(nucleus_frames) != 1:
        raise ValueError(
            "Feature stores must each use one coordinate frame: "
            f"morphology={sorted(store_frames)}, nucleus={sorted(nucleus_frames)}"
        )
    coordinate_frame = next(iter(store_frames))
    if nucleus_frames != {coordinate_frame}:
        raise ValueError(
            "Morphology and nucleus stores use different coordinate frames: "
            f"{sorted(store_frames)} vs {sorted(nucleus_frames)}"
        )
    if args.coordinate_frame is not None and coordinate_frame != str(args.coordinate_frame):
        raise ValueError(
            "Feature store coordinate frame does not match requested frame: "
            f"store={coordinate_frame!r}, requested={args.coordinate_frame!r}"
        )
    cache = MorphologyPathCache(
        output,
        store,
        coordinate_frame=coordinate_frame,
        candidate_radius_um=args.candidate_radius_um,
        candidate_spacing_um=args.candidate_spacing_um,
        move_weight=args.move_weight,
        morphology_weight=args.morphology_weight,
        nucleus_store=nucleus_store,
        nucleus_weight=args.nucleus_weight,
        allow_linear_fallback=args.allow_linear_fallback,
    )
    rng = np.random.default_rng(args.seed)
    pair_rows = []

    for index, n_sample in enumerate(requested_counts):
        s0, x0 = sections[index], coordinates[index]
        s1, x1 = sections[index + 1], coordinates[index + 1]
        uot_path = uot_dir / f"pair-{int(s0):03d}-{int(s1):03d}.npz"
        if not uot_path.is_file():
            raise FileNotFoundError(uot_path)
        coupling, metadata = load_sparse_coupling(uot_path)
        if tuple(coupling.shape) != (len(x0), len(x1)):
            raise ValueError(
                f"UOT shape {coupling.shape} does not match {s0}->{s1} "
                f"coordinates {(len(x0), len(x1))}"
            )
        if metadata.get("spatial_key") != "spatial_st_corrected":
            raise ValueError(f"UOT cache is not corrected-coordinate cache: {uot_path}")
        if (
            args.coordinate_frame is not None
            and metadata.get("coordinate_frame") != coordinate_frame
        ):
            raise ValueError(
                f"UOT cache frame does not match feature stores: {uot_path}; "
                f"expected {coordinate_frame!r}, got {metadata.get('coordinate_frame')!r}"
            )
        idx0, idx1 = coupling.sample(int(n_sample), rng=rng)
        pair_ids = [
            f"{s0}:{names[index][i]}->{s1}:{names[index + 1][j]}"
            for i, j in zip(idx0, idx1)
        ]
        ids = cache.prepare(
            s0,
            s1,
            x0[idx0],
            x1[idx1],
            pair_ids,
        )
        pair_rows.append(
            {
                "source_section": int(s0),
                "target_section": int(s1),
                "requested_samples": int(n_sample),
                "unique_paths_added": int(len(set(ids))),
                "path_id_count_returned": int(len(ids)),
                "uot_cache": str(uot_path),
                "uot_feature_revision": metadata.get("feature_store_revision"),
            }
        )
        print(
            f"{s0}->{s1}: requested={n_sample:,}, "
            f"returned={len(ids):,}, unique_in_batch={len(set(ids)):,}",
            flush=True,
        )

    with h5py.File(output, "r") as handle:
        path_count = len(handle["paths"])
        fallback_count = 0
        for group in handle["paths"].values():
            fallback_count += bool(json.loads(group.attrs["morphology_fallback_sections"]))

    summary = {
        "status": "complete",
        "coordinate_key": "spatial_st_corrected",
        "coordinate_frame": coordinate_frame,
        "anchor_dir": str(anchor_dir),
        "uot_dir": str(uot_dir),
        "feature_store": str(args.feature_store),
        "feature_store_revision": store.revision,
        "nucleus_feature_store": str(args.nucleus_feature_store),
        "nucleus_feature_store_revision": nucleus_store.revision,
        "output": str(output),
        "requested_total_pairs": int(requested_counts.sum()),
        "unique_cached_paths": int(path_count),
        "fallback_path_count": int(fallback_count),
        "fallback_path_fraction": float(fallback_count / max(path_count, 1)),
        "parameters": {
            "pairs": int(args.pairs),
            "seed": int(args.seed),
            "candidate_radius_um": float(args.candidate_radius_um),
            "candidate_spacing_um": float(args.candidate_spacing_um),
            "move_weight": float(args.move_weight),
            "morphology_weight": float(args.morphology_weight),
            "nucleus_weight": float(args.nucleus_weight),
            "allow_linear_fallback": bool(args.allow_linear_fallback),
        },
        "pairs": pair_rows,
        "note": "Anatomical morphology paths, not cell lineage tracking.",
    }
    summary_path = output.parent / "path_cache_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(summary_path)
    print(f"cached_paths={path_count:,} fallback_paths={fallback_count:,}")


if __name__ == "__main__":
    main()
