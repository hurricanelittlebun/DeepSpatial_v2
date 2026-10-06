"""Compare direct Cellpose-nucleus queries with and without H&E guidance.

This is deliberately different from the fixed-depth 3-D comparison.  The
query cells are the Cellpose nucleus centroids detected on section 031.  The
same query coordinates are passed to two already-trained holdout models:

* original DeepSpatial with the cell-type branch and no histology;
* H&E + morphology/nucleus-path model with the cell-type branch.

The figure compares predicted cell-type labels at exactly the same H&E
nucleus locations.  It is an anatomical correspondence visualization, not a
cell-lineage or one-to-one ST-cell tracking result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import scipy.sparse as sp

from deepspatial import DeepSpatial


DEFAULT_QUERY = ROOT / "data/00029_g0/nucleus_path_highres_rerun_v3/segmentation/section-031/nuclei_registered.csv"
DEFAULT_NOHE_CHECKPOINT = ROOT / "data/00029_g0/original_celltype_v1/holdout_model/deepspatial-epoch=08-loss=0.0182.ckpt"
DEFAULT_HE_CHECKPOINT = ROOT / "data/00029_g0/experiments/st_only_residual_nucleus_uot_path_celltype_v1/model/deepspatial-epoch=09-loss=0.0185.ckpt"
DEFAULT_NOHE_ANCHORS = ROOT / "data/00029_g0/st_only_residual_v1/celltype_domain_niche_anchors_v1"
DEFAULT_HE_ANCHORS = ROOT / "data/00029_g0/st_only_residual_v1/celltype_anchors_v1"
DEFAULT_OUTPUT = ROOT / "data/00029_g0/visualizations/direct_nucleus_query_he_comparison_v1"

LEFT_SECTION = 21
RIGHT_SECTION = 41
QUERY_SECTION = 31


def _anchor(anchor_dir: Path, section: int) -> ad.AnnData:
    path = Path(anchor_dir) / f"section-{section:03d}.h5ad"
    if not path.is_file():
        raise FileNotFoundError(path)
    return ad.read_h5ad(path)


def _load_query(path: Path) -> ad.AnnData:
    table = pd.read_csv(path)
    required = {"cell_id", "section_id", "z_um", "x_registered_um", "y_registered_um"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Cellpose table is missing columns: {missing}")
    if table.empty:
        raise ValueError(f"Cellpose table is empty: {path}")
    if not table["section_id"].astype(int).eq(QUERY_SECTION).all():
        raise ValueError("The query table is not section 031")
    coords = table[["x_registered_um", "y_registered_um"]].to_numpy(dtype=np.float64)
    z = table["z_um"].to_numpy(dtype=np.float64)
    if not np.isfinite(coords).all() or not np.isfinite(z).all():
        raise ValueError("Cellpose query contains non-finite coordinates")
    if not np.allclose(z, z[0]):
        raise ValueError("All section-031 query nuclei must have one z_um")

    obs = table.copy()
    obs.index = obs["cell_id"].astype(str)
    obs.index.name = "nucleus_query_id"
    # The predictor only needs z_um and the spatial key, but retaining the
    # Cellpose metadata makes the output traceable and useful downstream.
    query = ad.AnnData(
        X=sp.csr_matrix((len(table), 1), dtype=np.float32),
        obs=obs,
        var=pd.DataFrame(index=["coordinate_query"]),
        obsm={"spatial_st_corrected": coords},
    )
    query.uns["query_provenance"] = {
        "source": str(path.resolve()),
        "query_semantics": "Cellpose nucleus centroids on H&E section 031",
        "coordinate_frame": "00029__g0_registered",
        "coordinate_columns": ["x_registered_um", "y_registered_um"],
        "z_um": float(z[0]),
    }
    return query


def _predict(
    checkpoint: Path,
    anchor_dir: Path,
    query: ad.AnnData,
    *,
    steps: int,
    chunk_size: int,
    device: str,
) -> ad.AnnData:
    model = DeepSpatial()
    model.load_checkpoint(str(checkpoint), sampling_method="dopri5")
    if not model.use_celltype:
        raise RuntimeError(f"Checkpoint does not contain the cell-type branch: {checkpoint}")
    left = _anchor(anchor_dir, LEFT_SECTION)
    right = _anchor(anchor_dir, RIGHT_SECTION)
    result = model.predict_on_he_cells(
        query.copy(),
        left,
        right,
        steps=steps,
        chunk_size=chunk_size,
        device=device,
    )
    result.obs["query_cell_id"] = query.obs_names.to_numpy()
    result.obs["query_section_id"] = QUERY_SECTION
    result.obs["query_coordinate_frame"] = "00029__g0_registered"
    result.uns["direct_nucleus_query"] = {
        "checkpoint": str(checkpoint.resolve()),
        "anchor_left": LEFT_SECTION,
        "anchor_right": RIGHT_SECTION,
        "query_section": QUERY_SECTION,
        "query_semantics": "Cellpose nucleus centroids; anatomical correspondence, not cell tracking",
        "steps": int(steps),
        "chunk_size": int(chunk_size),
    }
    return result


def _labels(prediction: ad.AnnData) -> np.ndarray:
    if "cell_class" not in prediction.obs:
        raise KeyError("Prediction lacks obs['cell_class']")
    return prediction.obs["cell_class"].astype(str).to_numpy()


def _plot(
    query: ad.AnnData,
    nohe: ad.AnnData,
    he: ad.AnnData,
    output: Path,
    categories: list[str],
) -> dict:
    xy = np.asarray(query.obsm["spatial_st_corrected"], dtype=np.float64)
    nohe_labels = _labels(nohe)
    he_labels = _labels(he)
    if len(nohe_labels) != len(he_labels) or len(nohe_labels) != len(xy):
        raise ValueError("Query and prediction lengths do not agree")

    # A declared color map is shared by all panels, so color means the same
    # cell type in the two arms.
    cmap = plt.get_cmap("turbo")
    color_map = {
        category: cmap(0.08 + 0.84 * i / max(1, len(categories) - 1))
        for i, category in enumerate(categories)
    }
    nohe_colors = np.asarray([color_map.get(x, "#777777") for x in nohe_labels])
    he_colors = np.asarray([color_map.get(x, "#777777") for x in he_labels])
    same = nohe_labels == he_labels

    xmin, ymin = np.min(xy, axis=0)
    xmax, ymax = np.max(xy, axis=0)
    padx = max(50.0, 0.04 * (xmax - xmin))
    pady = max(50.0, 0.04 * (ymax - ymin))
    limits = (xmin - padx, xmax + padx, ymin - pady, ymax + pady)

    fig, axes = plt.subplots(1, 3, figsize=(22, 8), constrained_layout=False)
    panels = [
        (axes[0], nohe_colors, "Original DeepSpatial\nwithout H&E", None),
        (axes[1], he_colors, "H&E + nucleus path", None),
        (
            axes[2],
            np.where(same[:, None], np.array([[0.80, 0.80, 0.80, 1.0]]), np.array([[0.88, 0.15, 0.15, 1.0]])),
            f"Prediction change\nagreement = {same.mean():.1%}",
            same,
        ),
    ]
    for ax, colors, title, _ in panels:
        ax.scatter(xy[:, 0], xy[:, 1], s=7, c=colors, linewidths=0, alpha=0.82, rasterized=True)
        ax.set_title(title, fontsize=14, pad=10)
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("registered X (µm)", fontsize=10)
        ax.set_ylabel("registered Y (µm)", fontsize=10)
        ax.grid(True, color="#dddddd", linewidth=0.5, alpha=0.55)
        ax.tick_params(labelsize=9)

    axes[2].legend(
        handles=[
            Line2D([0], [0], marker="o", color="w", label="same predicted label", markerfacecolor="#cccccc", markersize=7),
            Line2D([0], [0], marker="o", color="w", label="label changed", markerfacecolor="#e02525", markersize=7),
        ],
        loc="upper right",
        fontsize=9,
        frameon=True,
    )

    handles = [
        Line2D([0], [0], marker="o", color="w", label=category, markerfacecolor=color_map[category], markersize=6)
        for category in categories
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.075),
    )
    fig.suptitle(
        "00029/g0 section-031 direct prediction at Cellpose nucleus centers",
        fontsize=17,
        y=0.985,
    )
    fig.text(
        0.5,
        0.945,
        f"same {len(xy):,} queries | z=150 µm | endpoints section 021 → 041 | same registered XY frame",
        ha="center",
        fontsize=10,
        color="#444444",
    )
    output.mkdir(parents=True, exist_ok=True)
    figure_path = output / "section-031_direct_nucleus_nohe_vs_he_celltype.png"
    fig.savefig(figure_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    summary = {
        "n_query_nuclei": int(len(xy)),
        "query_section": QUERY_SECTION,
        "query_z_um": float(query.obs["z_um"].iloc[0]),
        "coordinate_frame": "00029__g0_registered",
        "nohe_vs_he_label_agreement": float(same.mean()),
        "changed_label_fraction": float((~same).mean()),
        "nohe_label_counts": pd.Series(nohe_labels).value_counts().sort_index().to_dict(),
        "he_label_counts": pd.Series(he_labels).value_counts().sort_index().to_dict(),
        "prediction_distance_um": {
            "nohe_median": float(np.median(nohe.obs["prediction_distance_um"])),
            "nohe_p95": float(np.quantile(nohe.obs["prediction_distance_um"], 0.95)),
            "he_median": float(np.median(he.obs["prediction_distance_um"])),
            "he_p95": float(np.quantile(he.obs["prediction_distance_um"], 0.95)),
        },
        "figure": str(figure_path.resolve()),
        "interpretation": "Agreement is model-to-model agreement on the same H&E nuclei, not ground-truth accuracy.",
    }
    (output / "comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", type=Path, default=DEFAULT_QUERY)
    parser.add_argument("--nohe-checkpoint", type=Path, default=DEFAULT_NOHE_CHECKPOINT)
    parser.add_argument("--he-checkpoint", type=Path, default=DEFAULT_HE_CHECKPOINT)
    parser.add_argument("--nohe-anchor-dir", type=Path, default=DEFAULT_NOHE_ANCHORS)
    parser.add_argument("--he-anchor-dir", type=Path, default=DEFAULT_HE_ANCHORS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    paths = [
        args.query,
        args.nohe_checkpoint,
        args.he_checkpoint,
        args.nohe_anchor_dir,
        args.he_anchor_dir,
    ]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)

    query = _load_query(args.query)
    nohe = _predict(
        args.nohe_checkpoint,
        args.nohe_anchor_dir,
        query,
        steps=args.steps,
        chunk_size=args.chunk_size,
        device=args.device,
    )
    he = _predict(
        args.he_checkpoint,
        args.he_anchor_dir,
        query,
        steps=args.steps,
        chunk_size=args.chunk_size,
        device=args.device,
    )

    # The two checkpoints were trained with the same 37-class vocabulary.
    categories = [str(x) for x in nohe.uns.get("direct_nucleus_query", {}).get("categories", [])]
    if not categories:
        categories = sorted(set(_labels(nohe)) | set(_labels(he)))
    # Keep all provenance in the saved outputs.
    nohe.uns["query_provenance"] = query.uns["query_provenance"]
    he.uns["query_provenance"] = query.uns["query_provenance"]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    nohe_path = output / "section-031_nucleus_direct_prediction_nohe.h5ad"
    he_path = output / "section-031_nucleus_direct_prediction_he_nucleus_path.h5ad"
    nohe.write_h5ad(nohe_path, compression="gzip")
    he.write_h5ad(he_path, compression="gzip")
    summary = _plot(query, nohe, he, output, categories)
    manifest = {
        "status": "complete",
        "method": "direct H&E Cellpose nucleus-center query",
        "query_source": str(args.query.resolve()),
        "nohe_checkpoint": str(args.nohe_checkpoint.resolve()),
        "he_checkpoint": str(args.he_checkpoint.resolve()),
        "nohe_anchor_dir": str(args.nohe_anchor_dir.resolve()),
        "he_anchor_dir": str(args.he_anchor_dir.resolve()),
        "output_h5ad": {
            "nohe": str(nohe_path),
            "he_nucleus_path": str(he_path),
        },
        "summary": summary,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
