"""Run the original DeepSpatial flow with the cell-type branch on 00029/g0.

This experiment deliberately disables every histology component:

* no H&E/UNI2 feature input;
* no morphology-aware UOT term;
* no morphology- or nucleus-guided path;
* no histology spatial head.

The original DeepSpatial cell-type state and loss remain enabled.  The
current repository's sparse top-k UOT backend is retained so that the
original model can run on the 00029 cell counts without materializing an
all-pairs dense transport matrix.

Two outputs are produced in one experiment directory:

1. a full reconstruction trained on all ten 00029 anchors;
2. an independent section-031 holdout run, trained without sections 031 and
   051, for metrics comparable to the earlier 00029 ablations.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import anndata as ad
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scipy.sparse as sp
import torch
from sklearn.metrics import accuracy_score, f1_score

from deepspatial import DeepSpatial


SECTIONS = (1, 11, 21, 31, 41, 51, 61, 71, 81, 91)
HOLDOUT_TRAINING_SECTIONS = (1, 11, 21, 41, 61, 71, 81, 91)
HELDOUT_SECTION = 31
LEFT_ENDPOINT = 21
RIGHT_ENDPOINT = 41
DEFAULT_ANCHOR_DIR = ROOT / "data/00029_g0/st_only_residual_v1/celltype_domain_niche_anchors_v1"
DEFAULT_OUTPUT = ROOT / "data/00029_g0/original_celltype_v1"
COORDINATE_FRAME = "00029__g0_registered"


def _anchor_path(anchor_dir: Path, section: int) -> Path:
    return anchor_dir / f"section-{int(section):03d}.h5ad"


def _load_anchor(path: Path) -> ad.AnnData:
    value = ad.read_h5ad(path)
    required = {"cell_class", "section_id", "z_um"}
    missing = sorted(required - set(value.obs.columns))
    if missing:
        raise ValueError(f"{path} is missing obs columns: {missing}")
    if "spatial_st_corrected" not in value.obsm:
        raise ValueError(f"{path} lacks obsm['spatial_st_corrected']")
    if value.obs["cell_class"].isna().any():
        raise ValueError(f"{path} contains missing cell_class labels")
    if value.obs["z_um"].nunique() != 1:
        raise ValueError(f"{path} must contain one z_um value")
    return value


def _load_anchors(anchor_dir: Path, sections: tuple[int, ...]) -> list[ad.AnnData]:
    anchors = [_load_anchor(_anchor_path(anchor_dir, section)) for section in sections]
    genes = anchors[0].var_names.tolist()
    z_values = []
    for anchor in anchors:
        if anchor.var_names.tolist() != genes:
            raise ValueError("Anchor gene order differs across sections")
        z_values.append(float(anchor.obs["z_um"].iloc[0]))
    if not np.all(np.diff(z_values) > 0):
        raise ValueError(f"Anchors are not in increasing z order: {z_values}")
    return anchors


def _anchor_only(anchors: list[ad.AnnData]) -> ad.AnnData:
    """Keep anchor metadata and the canonical coordinate keys in the output."""

    pieces = []
    for anchor in anchors:
        obsm = {
            key: np.asarray(anchor.obsm[key]).copy()
            for key in ("spatial", "spatial_registered", "spatial_st_corrected")
            if key in anchor.obsm
        }
        piece = ad.AnnData(
            X=anchor.X.copy(),
            obs=anchor.obs.copy(),
            var=anchor.var.copy(),
            obsm=obsm,
        )
        if "spatial" not in piece.obsm:
            piece.obsm["spatial"] = piece.obsm["spatial_st_corrected"].copy()
        if "spatial_registered" not in piece.obsm:
            piece.obsm["spatial_registered"] = piece.obsm["spatial_st_corrected"].copy()
        piece.obs["is_reconstructed"] = False
        piece.obs["is_xenium_anchor"] = True
        pieces.append(piece)
    return ad.concat(pieces, join="outer", merge="first", uns_merge="first", index_unique=None)


def _query(target: ad.AnnData) -> ad.AnnData:
    return ad.AnnData(
        X=sp.csr_matrix((target.n_obs, 1), dtype=np.float32),
        obs=target.obs[["z_um"]].copy(),
        var=pd.DataFrame(index=["coordinate_query"]),
        obsm={"spatial_st_corrected": np.asarray(target.obsm["spatial_st_corrected"]).copy()},
    )


def _train_model(
    anchors: list[ad.AnnData],
    *,
    model_dir: Path,
    seed: int,
    epochs: int,
    pair_count: int,
    lambda_c: float,
    force: bool,
) -> tuple[DeepSpatial, Path]:
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted(p for p in model_dir.glob("*.ckpt") if p.name != "last.ckpt")
    if force:
        checkpoints = []
    if not checkpoints:
        pl.seed_everything(seed, workers=True)
        model = DeepSpatial()
        model.setup_data(
            anchors,
            spatial_key="spatial_st_corrected",
            z_key="z_um",
            label_key="cell_class",
            n_samples_base=pair_count,
            batch_size=128,
            num_workers=0,
            uot_solver="sparse_topk",
            uot_top_k=64,
            uot_bidirectional=True,
            use_celltype=True,
            histology=None,
        )
        model.build_model(
            patch_size=8,
            hidden_size=256,
            depth=6,
            num_heads=8,
            mlp_ratio=4.0,
            lr=2e-4,
            weight_decay=1e-5,
            lambda_g=0.1,
            lambda_c=lambda_c,
            sampling_method="dopri5",
            atol=1e-5,
            rtol=1e-5,
        )
        model.fit(
            max_epochs=epochs,
            save_dir=str(model_dir),
            accelerator="gpu",
            devices=1,
            save_ckpt=True,
        )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        checkpoints = sorted(p for p in model_dir.glob("*.ckpt") if p.name != "last.ckpt")
    if not checkpoints:
        raise RuntimeError(f"No checkpoint produced under {model_dir}")
    checkpoint = checkpoints[-1]
    trained = DeepSpatial()
    trained.load_checkpoint(str(checkpoint), sampling_method="dopri5")
    if not trained.use_celltype:
        raise RuntimeError("Checkpoint unexpectedly has use_celltype=False")
    return trained, checkpoint


def _dense_log_cpm(matrix) -> np.ndarray:
    values = matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)
    values = values.astype(np.float32, copy=False)
    library = values.sum(axis=1, keepdims=True)
    return np.log1p(values * (1e4 / np.maximum(library, 1.0)))


def _macro_gene_pearson(a: np.ndarray, b: np.ndarray) -> float:
    am = a.mean(axis=0, dtype=np.float64)
    bm = b.mean(axis=0, dtype=np.float64)
    n = len(a)
    a64 = a.astype(np.float64, copy=False)
    b64 = b.astype(np.float64, copy=False)
    numerator = np.sum(a64 * b64, axis=0) - n * am * bm
    avar = np.sum(a64 * a64, axis=0) - n * am * am
    bvar = np.sum(b64 * b64, axis=0) - n * bm * bm
    denominator = np.sqrt(np.maximum(avar, 0) * np.maximum(bvar, 0))
    valid = denominator > 1e-12
    return float(np.mean(numerator[valid] / denominator[valid])) if valid.any() else float("nan")


def _expression_metrics(predicted, truth, distances) -> dict:
    pred = _dense_log_cpm(predicted)
    target = _dense_log_cpm(truth)
    difference = pred - target
    return {
        "n_cells": int(pred.shape[0]),
        "n_genes": int(pred.shape[1]),
        "macro_gene_pearson": _macro_gene_pearson(pred, target),
        "mean_profile_pearson": float(np.corrcoef(pred.mean(axis=0), target.mean(axis=0))[0, 1]),
        "mae_log1p_cpm": float(np.mean(np.abs(difference), dtype=np.float64)),
        "rmse_log1p_cpm": float(np.sqrt(np.mean(difference * difference, dtype=np.float64))),
        "prediction_distance_median_um": float(np.median(distances)),
        "prediction_distance_p95_um": float(np.quantile(distances, 0.95)),
    }


def _celltype_metrics(prediction: ad.AnnData, target: ad.AnnData, categories) -> dict:
    truth = target.obs["cell_class"].astype(str).to_numpy()
    predicted = prediction.obs["cell_class"].astype(str).to_numpy()
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(
            f1_score(truth, predicted, labels=list(categories), average="macro", zero_division=0)
        ),
        "n_cells": int(len(truth)),
        "n_categories_evaluated": int(len(set(truth))),
        "unassigned_truth_fraction": float(np.mean(truth == "unassigned")),
        "unassigned_prediction_fraction": float(np.mean(predicted == "unassigned")),
    }


def run(args) -> dict:
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    anchor_dir = Path(args.anchor_dir).resolve()
    sections = tuple(int(x) for x in args.sections)
    anchors = _load_anchors(anchor_dir, sections)
    categories = sorted({str(x) for anchor in anchors for x in anchor.obs["cell_class"].astype(str)})

    full_model, full_checkpoint = _train_model(
        anchors,
        model_dir=output / "full_model",
        seed=args.seed,
        epochs=args.epochs,
        pair_count=args.pairs,
        lambda_c=args.lambda_c,
        force=args.force,
    )
    full_virtual = full_model.reconstruct_full_volume(
        anchors,
        thickness=args.thickness_um,
        steps=args.reconstruction_steps,
        chunk_size=args.reconstruction_chunk_size,
        device=args.device,
    )
    full_virtual.obs["is_reconstructed"] = True
    full_virtual.obs["is_xenium_anchor"] = False
    if "spatial" not in full_virtual.obsm:
        full_virtual.obsm["spatial"] = np.asarray(full_virtual.obsm["spatial_st_corrected"]).copy()
    if "spatial_registered" not in full_virtual.obsm:
        full_virtual.obsm["spatial_registered"] = np.asarray(full_virtual.obsm["spatial_st_corrected"]).copy()
    anchor_output = _anchor_only(anchors)
    full = ad.concat(
        [anchor_output, full_virtual],
        join="outer",
        merge="first",
        uns_merge="first",
        index_unique=None,
    )
    anchor_cell_count = int(anchor_output.n_obs)
    virtual_cell_count = int(full_virtual.n_obs)
    full_cell_count = int(full.n_obs)
    full.obs_names = [f"00029__g0__cell_{i:08d}" for i in range(full.n_obs)]
    full.uns["deepspatial_reconstruction"] = {
        "series_id": "00029__g0",
        "coordinate_frame": COORDINATE_FRAME,
        "spatial_key": "spatial_st_corrected",
        "label_key": "cell_class",
        "checkpoint": str(full_checkpoint),
        "method": "original DeepSpatial sparse-topk UOT + linear flow + cell-type branch",
        "histology": False,
        "use_morphology_uot": False,
        "use_morphology_path": False,
        "use_nucleus_path": False,
        "use_celltype": True,
        "lambda_c": float(args.lambda_c),
        "label_semantics": {
            "cell_class": "model cell-type flow prediction",
            "pred_domain": "forward inherits low-Z anchor / backward inherits high-Z anchor",
            "pred_niche": "forward inherits low-Z anchor / backward inherits high-Z anchor",
        },
    }
    final_path = output / (
        f"adata_00029_full_thickness{int(args.thickness_um)}_original_celltype.h5ad"
    )
    if args.force or not final_path.exists():
        full.write_h5ad(final_path, compression="gzip")

    del full_virtual, full, anchor_output, full_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Independent section-031 evaluation with the same holdout protocol used
    # for the previously reported 00029 metrics.
    holdout_anchors = _load_anchors(anchor_dir, HOLDOUT_TRAINING_SECTIONS)
    target = _load_anchor(_anchor_path(anchor_dir, HELDOUT_SECTION))
    train_categories = sorted(
        {str(x) for anchor in holdout_anchors for x in anchor.obs["cell_class"].astype(str)}
    )
    if set(target.obs["cell_class"].astype(str)) - set(train_categories):
        raise ValueError("Held-out section contains a cell type absent from training")
    holdout_model, holdout_checkpoint = _train_model(
        holdout_anchors,
        model_dir=output / "holdout_model",
        seed=args.seed,
        epochs=args.epochs,
        pair_count=args.pairs,
        lambda_c=args.lambda_c,
        force=args.force,
    )
    query = _query(target)
    prediction = holdout_model.predict_on_he_cells(
        query,
        _load_anchor(_anchor_path(anchor_dir, LEFT_ENDPOINT)),
        _load_anchor(_anchor_path(anchor_dir, RIGHT_ENDPOINT)),
        steps=40,
        chunk_size=args.inference_chunk_size,
        device=args.device,
    )
    expression = _expression_metrics(
        prediction.X,
        target.X,
        prediction.obs["prediction_distance_um"].to_numpy(),
    )
    celltype = _celltype_metrics(prediction, target, train_categories)
    holdout_record = {
        "experiment": "00029_g0_original_celltype_v1",
        "checkpoint": str(holdout_checkpoint),
        "anchor_dir": str(anchor_dir),
        "training_sections": list(HOLDOUT_TRAINING_SECTIONS),
        "heldout_section": HELDOUT_SECTION,
        "endpoint_left": LEFT_ENDPOINT,
        "endpoint_right": RIGHT_ENDPOINT,
        "heldout_expression_used_for_training": False,
        "seed": args.seed,
        "epochs": args.epochs,
        "sampled_uot_pairs_total": args.pairs,
        "uot_solver": "sparse_topk",
        "uot_top_k": 64,
        "use_celltype": True,
        "lambda_c": float(args.lambda_c),
        "num_classes": int(holdout_model.num_classes),
        "components": {
            "use_histology": False,
            "use_morphology_uot": False,
            "use_morphology_path": False,
            "use_nucleus_path": False,
            "use_celltype": True,
        },
        "expression_metrics": expression,
        "celltype_metrics": celltype,
    }
    (output / "holdout_metrics.json").write_text(json.dumps(holdout_record, indent=2) + "\n")
    (output / "summary_metrics.json").write_text(
        json.dumps({"expression": expression, "celltype": celltype}, indent=2) + "\n"
    )

    manifest = {
        "status": "complete",
        "experiment": "00029_g0_original_celltype_v1",
        "series_id": "00029__g0",
        "anchor_dir": str(anchor_dir),
        "sections": list(sections),
        "thickness_um": float(args.thickness_um),
        "reconstruction_steps": int(args.reconstruction_steps),
        "reconstruction_chunk_size": int(args.reconstruction_chunk_size),
        "coordinate_frame": COORDINATE_FRAME,
        "spatial_key": "spatial_st_corrected",
        "label_key": "cell_class",
        "full_checkpoint": str(full_checkpoint),
        "holdout_checkpoint": str(holdout_checkpoint),
        "anchor_cell_count": anchor_cell_count,
        "virtual_cell_count": virtual_cell_count,
        "total_cell_count": full_cell_count,
        "full_output": str(final_path),
        "holdout_metrics": holdout_record,
        "components": holdout_record["components"],
        "note": "No H&E/UNI2/nucleus features were used; sparse top-k is retained for million-cell scalability.",
    }
    (output / "reconstruction_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    del prediction, target, query, holdout_model, holdout_anchors, anchors
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-dir", type=Path, default=DEFAULT_ANCHOR_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sections", nargs="+", type=int, default=list(SECTIONS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--pairs", type=int, default=20_000)
    parser.add_argument("--lambda-c", type=float, default=10.0)
    parser.add_argument("--thickness-um", type=float, default=10.0)
    parser.add_argument("--reconstruction-steps", type=int, default=100)
    parser.add_argument("--reconstruction-chunk-size", type=int, default=1024)
    parser.add_argument("--inference-chunk-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    start = time.time()
    result = run(args)
    result["elapsed_min"] = (time.time() - start) / 60.0
    (Path(args.output).resolve() / "reconstruction_manifest.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
