"""Build sparse UOT caches and QC plots for the 00029/g0 anchor series."""

import argparse
import gc
import json
import re
import sys
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from matplotlib.collections import LineCollection
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepspatial.data_utils.uot_cache import (
    load_sparse_coupling,
    save_sparse_coupling,
    summarize_sparse_coupling,
)
from deepspatial.data_utils.uot_solver import compute_uot_coupling
from deepspatial.histology import FeatureStore


def _section_number(path):
    match = re.search(r"section-(\d+)", Path(path).stem)
    if match is None:
        raise ValueError(f"Cannot parse section number from {path}")
    return int(match.group(1))


def _anchor_paths(dataset, anchor_dir=None):
    root = Path(anchor_dir) if anchor_dir is not None else Path(dataset) / "anchors"
    paths = sorted(
        root.glob("section-*.h5ad"), key=_section_number
    )
    if len(paths) < 2:
        raise ValueError(f"Expected at least two anchor H5ADs under {root}")
    return paths


def _load_section(path, spatial_key):
    obj = ad.read_h5ad(path)
    if spatial_key not in obj.obsm:
        raise KeyError(f"{path} lacks obsm[{spatial_key!r}]")
    if "section_id" not in obj.obs or "z_um" not in obj.obs:
        raise KeyError(f"{path} needs obs['section_id'] and obs['z_um']")
    xy = np.asarray(obj.obsm[spatial_key], dtype=np.float32)
    expression = obj.X.toarray() if sp.issparse(obj.X) else np.asarray(obj.X)
    expression = expression.astype(np.float32, copy=False)
    section_id = str(obj.obs["section_id"].iloc[0])
    z_um = float(obj.obs["z_um"].iloc[0])
    return obj, xy, expression, section_id, z_um


def _validate_feature_store_frame(feature_store, expected_frame=None):
    """Return the single frame used by a feature store, optionally requiring one."""

    frames = {
        str(feature_store.metadata[section_id]["coordinate_frame"])
        for section_id in feature_store.sections
    }
    if len(frames) != 1:
        raise ValueError(f"Feature store must use one coordinate frame, got {sorted(frames)}")
    frame = next(iter(frames))
    if expected_frame is not None and frame != str(expected_frame):
        raise ValueError(
            "Feature store coordinate frame does not match requested coordinate frame: "
            f"store={frame!r}, requested={expected_frame!r}"
        )
    return frame


def _pair_cache_path(cache_dir, source_id, target_id):
    return Path(cache_dir) / f"pair-{int(source_id):03d}-{int(target_id):03d}.npz"


def _compute_or_load_pair(
    source_path,
    target_path,
    feature_store,
    cache_dir,
    *,
    spatial_key,
    top_k,
    bidirectional,
    alpha_spatial,
    uot_reg,
    uot_tau,
    histology_weight,
    coordinate_frame,
    overwrite=False,
):
    source, x0, g0, source_id, z0 = _load_section(source_path, spatial_key)
    target, x1, g1, target_id, z1 = _load_section(target_path, spatial_key)
    if source.var_names.tolist() != target.var_names.tolist():
        raise ValueError(f"Gene order differs between {source_path} and {target_path}")

    cache_path = _pair_cache_path(cache_dir, source_id, target_id)
    metadata = {
        "source_section": source_id,
        "target_section": target_id,
        "source_z_um": z0,
        "target_z_um": z1,
        "top_k": int(top_k),
        "bidirectional": bool(bidirectional),
        "alpha_spatial": float(alpha_spatial),
        "uot_reg": float(uot_reg),
        "uot_tau": float(uot_tau),
        "histology_weight": float(histology_weight),
        "solver": "sparse_topk",
        "feature_store_revision": feature_store.revision,
        "coordinate_frame": str(coordinate_frame),
        "spatial_key": spatial_key,
    }

    coupling = None
    cached_metadata = None
    if cache_path.exists() and not overwrite:
        candidate, cached_metadata = load_sparse_coupling(cache_path)
        if all(cached_metadata.get(key) == value for key, value in metadata.items()):
            coupling = candidate

    h0_valid_fraction = None
    h1_valid_fraction = None
    if coupling is None:
        h0, h0_valid = feature_store.get_feature(
            source_id, x0, return_valid=True
        )
        h1, h1_valid = feature_store.get_feature(
            target_id, x1, return_valid=True
        )
        h0 = h0.numpy()
        h1 = h1.numpy()
        h0_valid = h0_valid.numpy()
        h1_valid = h1_valid.numpy()
        h0_valid_fraction = float(h0_valid.mean())
        h1_valid_fraction = float(h1_valid.mean())
        c0 = np.zeros((len(x0), 1), dtype=np.float32)
        c1 = np.zeros((len(x1), 1), dtype=np.float32)
        coupling = compute_uot_coupling(
            x0,
            g0,
            c0,
            x1,
            g1,
            c1,
            alpha_spatial=alpha_spatial,
            uot_reg=uot_reg,
            uot_tau=uot_tau,
            solver="sparse_topk",
            top_k=top_k,
            bidirectional=bidirectional,
            candidate_x0=x0,
            candidate_x1=x1,
            use_celltype=False,
            histology_weight=histology_weight,
            h0=h0,
            h1=h1,
            h0_valid=h0_valid,
            h1_valid=h1_valid,
        )
        metadata["source_feature_valid_fraction"] = h0_valid_fraction
        metadata["target_feature_valid_fraction"] = h1_valid_fraction
        save_sparse_coupling(cache_path, coupling, metadata)
    else:
        h0_valid_fraction = cached_metadata.get("source_feature_valid_fraction")
        h1_valid_fraction = cached_metadata.get("target_feature_valid_fraction")
        metadata = cached_metadata

    summary = summarize_sparse_coupling(coupling, x0, x1)
    summary.update(
        {
            "source_section": source_id,
            "target_section": target_id,
            "source_z_um": z0,
            "target_z_um": z1,
            "z_gap_um": z1 - z0,
            "top_k": int(top_k),
            "bidirectional": bool(bidirectional),
            "source_feature_valid_fraction": h0_valid_fraction,
            "target_feature_valid_fraction": h1_valid_fraction,
            "cache_path": str(cache_path),
        }
    )
    return summary, coupling, x0, x1


def _plot_pair(path, source_xy, target_xy, coupling, summary, max_edges):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    edge_count = len(coupling)
    keep = min(int(max_edges), edge_count)
    if keep:
        selected = np.argpartition(coupling.mass, -keep)[-keep:]
        selected = selected[np.argsort(coupling.mass[selected])[::-1]]
        segments = np.stack(
            [source_xy[coupling.row[selected]], target_xy[coupling.col[selected]]],
            axis=1,
        )
        weights = coupling.mass[selected]
        weights = weights / max(weights.max(), np.finfo(float).eps)
    else:
        segments = np.empty((0, 2, 2), dtype=float)
        weights = np.empty(0, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    ax.scatter(
        source_xy[:, 0], source_xy[:, 1], s=1, c="#1976d2", alpha=0.22, label="source"
    )
    ax.scatter(
        target_xy[:, 0], target_xy[:, 1], s=1, c="#f57c00", alpha=0.22, label="target"
    )
    if len(segments):
        line_collection = LineCollection(
            segments,
            colors=(0.25, 0.25, 0.25, 0.18),
            linewidths=0.25 + 0.9 * weights,
        )
        ax.add_collection(line_collection)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("registered X (um)")
    ax.set_ylabel("registered Y (um)")
    ax.set_title(
        f"{summary['source_section']} → {summary['target_section']} | "
        f"k={summary['top_k']} | edges={summary['edge_count']:,}\n"
        f"support={summary['source_support_fraction']:.3f}/"
        f"{summary['target_support_fraction']:.3f}, "
        f"weighted p95={summary['weighted_distance_p95_um']:.1f} um"
    )
    ax.legend(loc="best", markerscale=5)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_overview(path, summary_frame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    labels = [f"{a}-{b}" for a, b in zip(summary_frame.source_section, summary_frame.target_section)]
    axes[0].bar(labels, summary_frame.edge_fraction_of_dense)
    axes[0].set_ylabel("sparse edges / dense pairs")
    axes[0].tick_params(axis="x", rotation=60)
    axes[1].bar(labels, summary_frame.weighted_distance_p95_um)
    axes[1].set_ylabel("weighted distance p95 (um)")
    axes[1].tick_params(axis="x", rotation=60)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _parse_pair(value):
    left, right = value.replace("section-", "").split("-")
    return int(left), int(right)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/00029_g0")
    parser.add_argument(
        "--anchor-dir",
        default=None,
        help="Directory containing section-*.h5ad; defaults to DATASET/anchors.",
    )
    parser.add_argument("--spatial-key", default="spatial_registered")
    parser.add_argument("--feature-store", default="data/00029_g0/uni2_features.h5")
    parser.add_argument("--output", default="data/00029_g0/uot_cache_topk64")
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--sweep-k", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--sweep-pairs", nargs="*", default=None)
    parser.add_argument("--max-edges-plot", type=int, default=2000)
    parser.add_argument("--alpha-spatial", type=float, default=0.5)
    parser.add_argument("--uot-reg", type=float, default=0.8)
    parser.add_argument("--uot-tau", type=float, default=0.05)
    parser.add_argument("--histology-weight", type=float, default=1.0)
    parser.add_argument(
        "--coordinate-frame",
        default=None,
        help="Require this frame for the morphology feature store; recorded in cache metadata.",
    )
    parser.add_argument("--no-bidirectional", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    qc_dir = output / "qc_overlays"
    qc_dir.mkdir(parents=True, exist_ok=True)
    anchor_paths = _anchor_paths(dataset, args.anchor_dir)
    pairs = list(zip(anchor_paths[:-1], anchor_paths[1:]))
    bidirectional = not args.no_bidirectional
    feature_store = FeatureStore(args.feature_store)
    coordinate_frame = _validate_feature_store_frame(
        feature_store, args.coordinate_frame
    )

    summaries = []
    for source_path, target_path in tqdm(pairs, desc="00029 sparse UOT", unit="pair"):
        summary, coupling, x0, x1 = _compute_or_load_pair(
            source_path,
            target_path,
            feature_store,
            output,
            spatial_key=args.spatial_key,
            top_k=args.top_k,
            bidirectional=bidirectional,
            alpha_spatial=args.alpha_spatial,
            uot_reg=args.uot_reg,
            uot_tau=args.uot_tau,
            histology_weight=args.histology_weight,
            coordinate_frame=coordinate_frame,
            overwrite=args.overwrite,
        )
        summaries.append(summary)
        _plot_pair(
            qc_dir / f"pair-{summary['source_section']}-{summary['target_section']}.png",
            x0,
            x1,
            coupling,
            summary,
            args.max_edges_plot,
        )
        del coupling, x0, x1
        gc.collect()

    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(output / "summary.csv", index=False)
    _plot_overview(output / "summary_overview.png", summary_frame)

    if args.sweep_pairs:
        wanted = {_parse_pair(item) for item in args.sweep_pairs}
        sweep_pairs = [
            pair for pair in pairs
            if (_section_number(pair[0]), _section_number(pair[1])) in wanted
        ]
    else:
        sweep_indices = sorted({0, len(pairs) // 2, len(pairs) - 1})
        sweep_pairs = [pairs[index] for index in sweep_indices]

    sweep_rows = []
    for source_path, target_path in tqdm(sweep_pairs, desc="top-k sweep", unit="pair"):
        for top_k in args.sweep_k:
            sweep_cache_dir = output if top_k == args.top_k else output / f"sweep_k{top_k}"
            summary, coupling, x0, x1 = _compute_or_load_pair(
                source_path,
                target_path,
                feature_store,
                sweep_cache_dir,
                spatial_key=args.spatial_key,
                top_k=top_k,
                bidirectional=bidirectional,
                alpha_spatial=args.alpha_spatial,
                uot_reg=args.uot_reg,
                uot_tau=args.uot_tau,
                histology_weight=args.histology_weight,
                coordinate_frame=coordinate_frame,
                overwrite=args.overwrite,
            )
            sweep_rows.append(summary)
            del coupling, x0, x1
            gc.collect()
    if sweep_rows:
        pd.DataFrame(sweep_rows).to_csv(output / "top_k_sweep.csv", index=False)

    run_metadata = {
        "dataset": str(dataset),
        "anchor_dir": str(args.anchor_dir or dataset / "anchors"),
        "spatial_key": args.spatial_key,
        "feature_store": str(args.feature_store),
        "coordinate_frame": coordinate_frame,
        "anchor_count": len(anchor_paths),
        "pair_count": len(pairs),
        "main_top_k": args.top_k,
        "sweep_k": args.sweep_k,
        "sweep_pairs": [
            [int(_section_number(a)), int(_section_number(b))]
            for a, b in sweep_pairs
        ],
        "bidirectional": bidirectional,
        "alpha_spatial": args.alpha_spatial,
        "uot_reg": args.uot_reg,
        "uot_tau": args.uot_tau,
        "histology_weight": args.histology_weight,
    }
    (output / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2) + "\n"
    )
    print(output / "summary.csv")
    print(output / "top_k_sweep.csv")


if __name__ == "__main__":
    main()
