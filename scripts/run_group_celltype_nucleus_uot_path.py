"""Train and reconstruct one registered Xenium/H&E tissue series.

This is the group-level equivalent of the validated 00029/g0 protocol:
sparse-top-k morphology-aware UOT, UNI2 morphology/nucleus-guided spatial
paths, and the original cell-type branch with ``lambda_c=10``.  It deliberately
does not use H&E as a direct gene-expression regressor.
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
from deepspatial.histology import HistologyConfig


def _anchor_path(anchor_dir: Path, section: int) -> Path:
    return anchor_dir / f"section-{int(section):03d}.h5ad"


def _load_anchor(path: Path) -> ad.AnnData:
    obj = ad.read_h5ad(path)
    required = {"cell_class", "section_id", "z_um"}
    missing = sorted(required - set(obj.obs.columns))
    if missing:
        raise ValueError(f"{path} is missing obs columns: {missing}")
    if "spatial_st_corrected" not in obj.obsm:
        raise ValueError(f"{path} lacks obsm['spatial_st_corrected']")
    if obj.obs["cell_class"].isna().any():
        raise ValueError(f"{path} contains missing cell_class labels")
    if obj.obs["z_um"].nunique() != 1:
        raise ValueError(f"{path} must contain one z_um value")
    for key in ("pred_domain", "pred_niche"):
        if key not in obj.obs or obj.obs[key].isna().any():
            raise ValueError(f"{path} must contain complete {key!r} labels")
    return obj


def _load_anchors(anchor_dir: Path, sections: list[int]) -> list[ad.AnnData]:
    anchors = [_load_anchor(_anchor_path(anchor_dir, section)) for section in sections]
    if not anchors:
        raise ValueError("No anchors requested")
    genes = anchors[0].var_names.tolist()
    z = []
    for anchor in anchors:
        if anchor.var_names.tolist() != genes:
            raise ValueError("Anchor gene order differs")
        z.append(float(anchor.obs["z_um"].iloc[0]))
    if not np.all(np.diff(z) > 0):
        raise ValueError(f"Anchors are not in increasing z order: {z}")
    return anchors


def _config(
    *,
    feature_path: Path,
    nucleus_feature_path: Path,
    path_cache: Path,
    coordinate_frame: str,
    seed: int,
    nucleus_weight: float,
) -> HistologyConfig:
    return HistologyConfig(
        use_histology=True,
        use_morphology_uot=True,
        use_morphology_path=True,
        use_nucleus_path=True,
        feature_path=str(feature_path),
        nucleus_feature_path=str(nucleus_feature_path),
        path_cache=str(path_cache),
        section_key="section_id",
        coordinate_frame=coordinate_frame,
        nucleus_weight=float(nucleus_weight),
        allow_linear_fallback=True,
        seed=int(seed),
    )


def _dense_log_cpm(matrix) -> np.ndarray:
    values = matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)
    values = values.astype(np.float32, copy=False)
    library = values.sum(axis=1, keepdims=True)
    return np.log1p(values * (1e4 / np.maximum(library, 1.0)))


def _macro_gene_pearson(a: np.ndarray, b: np.ndarray) -> float:
    am = a.mean(axis=0, dtype=np.float64)
    bm = b.mean(axis=0, dtype=np.float64)
    n = len(a)
    numerator = np.sum(a.astype(np.float64) * b.astype(np.float64), axis=0) - n * am * bm
    avar = np.sum(a.astype(np.float64) ** 2, axis=0) - n * am * am
    bvar = np.sum(b.astype(np.float64) ** 2, axis=0) - n * bm * bm
    denominator = np.sqrt(np.maximum(avar, 0) * np.maximum(bvar, 0))
    valid = denominator > 1e-12
    return float(np.mean(numerator[valid] / denominator[valid])) if valid.any() else float("nan")


def _expression_metrics(predicted, truth, distances) -> dict:
    pred = _dense_log_cpm(predicted)
    target = _dense_log_cpm(truth)
    diff = pred - target
    return {
        "n_cells": int(pred.shape[0]),
        "n_genes": int(pred.shape[1]),
        "macro_gene_pearson": _macro_gene_pearson(pred, target),
        "mean_profile_pearson": float(np.corrcoef(pred.mean(0), target.mean(0))[0, 1]),
        "mae_log1p_cpm": float(np.mean(np.abs(diff), dtype=np.float64)),
        "rmse_log1p_cpm": float(np.sqrt(np.mean(diff * diff, dtype=np.float64))),
        "prediction_distance_median_um": float(np.median(distances)),
        "prediction_distance_p95_um": float(np.quantile(distances, 0.95)),
    }


def _celltype_metrics(prediction: ad.AnnData, target: ad.AnnData, categories) -> dict:
    true = target.obs["cell_class"].astype(str).to_numpy()
    pred = prediction.obs["cell_class"].astype(str).to_numpy()
    return {
        "accuracy": float(accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, labels=list(categories), average="macro", zero_division=0)),
        "n_cells": int(len(true)),
        "n_categories_evaluated": int(len(set(true))),
    }


def _query(target: ad.AnnData) -> ad.AnnData:
    return ad.AnnData(
        X=sp.csr_matrix((target.n_obs, 1), dtype=np.float32),
        obs=target.obs[["z_um"]].copy(),
        var=pd.DataFrame(index=["coordinate_query"]),
        obsm={"spatial_st_corrected": np.asarray(target.obsm["spatial_st_corrected"]).copy()},
    )


def _choose_checkpoint(model_dir: Path) -> Path:
    checkpoints = sorted(p for p in model_dir.glob("*.ckpt") if p.name != "last.ckpt")
    if not checkpoints:
        checkpoints = sorted(model_dir.glob("*.ckpt"))
    if not checkpoints:
        raise RuntimeError(f"No checkpoint under {model_dir}")
    return checkpoints[-1]


def _anchor_only(anchors: list[ad.AnnData]) -> ad.AnnData:
    """Keep a consistent compact set of obsm keys for final concatenation."""

    pieces = []
    for anchor in anchors:
        piece = ad.AnnData(
            X=anchor.X.copy(),
            obs=anchor.obs.copy(),
            var=anchor.var.copy(),
            obsm={"spatial_st_corrected": np.asarray(anchor.obsm["spatial_st_corrected"]).copy()},
        )
        piece.obsm["spatial"] = piece.obsm["spatial_st_corrected"].copy()
        piece.obsm["spatial_registered"] = piece.obsm["spatial_st_corrected"].copy()
        piece.obs["is_reconstructed"] = False
        piece.obs["is_xenium_anchor"] = True
        pieces.append(piece)
    return ad.concat(pieces, join="outer", merge="first", uns_merge="first", index_unique=None)


def train_and_reconstruct(args) -> dict:
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    anchor_dir = Path(args.anchor_dir).resolve()
    feature_path = Path(args.feature_store).resolve()
    nucleus_path = Path(args.nucleus_feature_store).resolve()
    path_cache = Path(args.path_cache).resolve()
    sections = [int(x) for x in args.sections]
    train_sections = (
        sections
        if args.full
        else [int(x) for x in args.training_sections]
    )
    all_anchors = _load_anchors(anchor_dir, sections)
    train_anchors = _load_anchors(anchor_dir, train_sections)
    categories = sorted({str(x) for a in train_anchors for x in a.obs["cell_class"].astype(str)})
    target = None
    if not args.full:
        if args.heldout_section is None or args.endpoint_left is None or args.endpoint_right is None:
            raise ValueError(
                "heldout-section, endpoint-left, and endpoint-right are required "
                "unless --full is enabled"
            )
        target = _load_anchor(_anchor_path(anchor_dir, args.heldout_section))
        if set(target.obs["cell_class"].astype(str)) - set(categories):
            raise ValueError("Held-out cell type absent from training vocabulary")

    model_dir = output / "model"
    pl.seed_everything(args.seed, workers=True)
    if not list(model_dir.glob("*.ckpt")) or args.force:
        model = DeepSpatial()
        model.setup_data(
            train_anchors,
            spatial_key="spatial_st_corrected",
            z_key="z_um",
            label_key="cell_class",
            n_samples_base=args.pairs,
            batch_size=128,
            num_workers=0,
            uot_solver="sparse_topk",
            uot_top_k=64,
            uot_bidirectional=True,
            use_celltype=True,
            histology=_config(
                feature_path=feature_path,
                nucleus_feature_path=nucleus_path,
                path_cache=path_cache,
                coordinate_frame=args.coordinate_frame,
                seed=args.seed,
                nucleus_weight=args.nucleus_weight,
            ),
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
            lambda_c=args.lambda_c,
            sampling_method="dopri5",
            atol=1e-5,
            rtol=1e-5,
        )
        model.fit(max_epochs=args.epochs, save_dir=str(model_dir), accelerator="gpu", devices=1, save_ckpt=True)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    checkpoint = _choose_checkpoint(model_dir)
    trained = DeepSpatial()
    trained.load_checkpoint(str(checkpoint), sampling_method="dopri5")
    prediction = None
    holdout = None
    if not args.full:
        query = _query(target)
        prediction = trained.predict_on_he_cells(
            query,
            _load_anchor(_anchor_path(anchor_dir, args.endpoint_left)),
            _load_anchor(_anchor_path(anchor_dir, args.endpoint_right)),
            steps=40,
            chunk_size=args.inference_chunk_size,
            device=args.device,
        )
        holdout = {
            "expression": _expression_metrics(
                prediction.X, target.X, prediction.obs["prediction_distance_um"].to_numpy()
            ),
            "celltype": _celltype_metrics(prediction, target, categories),
        }
        (output / "holdout_metrics.json").write_text(json.dumps(holdout, indent=2) + "\n")

    virtual = trained.reconstruct_full_volume(
        all_anchors,
        thickness=args.thickness_um,
        steps=args.reconstruction_steps,
        chunk_size=args.reconstruction_chunk_size,
        device=args.device,
    )
    virtual.obs["is_reconstructed"] = True
    virtual.obs["is_xenium_anchor"] = False
    if "spatial" not in virtual.obsm:
        virtual.obsm["spatial"] = np.asarray(virtual.obsm["spatial_st_corrected"]).copy()
    if "spatial_registered" not in virtual.obsm:
        virtual.obsm["spatial_registered"] = np.asarray(virtual.obsm["spatial_st_corrected"]).copy()
    anchors_out = _anchor_only(all_anchors)
    full = ad.concat([anchors_out, virtual], join="outer", merge="first", uns_merge="first", index_unique=None)
    full.obs_names = [f"{args.series_id}_cell_{i:08d}" for i in range(full.n_obs)]
    full.uns["deepspatial_reconstruction"] = {
        "series_id": args.series_id,
        "coordinate_frame": args.coordinate_frame,
        "spatial_key": "spatial_st_corrected",
        "checkpoint": str(checkpoint),
        "method": "morphology-aware sparse-topk UOT + UNI2/nucleus morphology-guided path + cell-type flow",
        "evaluation_mode": "full_no_holdout" if args.full else "holdout",
        "label_semantics": {
            "cell_class": "model cell-type branch",
            "pred_domain": "forward inherits low-Z anchor / backward inherits high-Z anchor",
            "pred_niche": "forward inherits low-Z anchor / backward inherits high-Z anchor",
        },
    }
    final_path = output / f"adata_{args.series_id.replace('__', '_')}_full_thickness{int(args.thickness_um)}_celltype_domain_niche.h5ad"
    full.write_h5ad(final_path, compression="gzip")
    manifest = {
        "status": "complete",
        "series_id": args.series_id,
        "checkpoint": str(checkpoint),
        "anchor_dir": str(anchor_dir),
        "sections": sections,
        "training_sections": train_sections,
        "heldout_section": None if args.full else args.heldout_section,
        "thickness_um": args.thickness_um,
        "reconstruction_steps": args.reconstruction_steps,
        "reconstruction_chunk_size": args.reconstruction_chunk_size,
        "coordinate_frame": args.coordinate_frame,
        "spatial_key": "spatial_st_corrected",
        "anchor_cell_count": int(anchors_out.n_obs),
        "virtual_cell_count": int(virtual.n_obs),
        "total_cell_count": int(full.n_obs),
        "output": str(final_path),
        "holdout_metrics": holdout,
        "evaluation_mode": "full_no_holdout" if args.full else "holdout",
        "components": {
            "use_histology": True,
            "use_morphology_uot": True,
            "use_morphology_path": True,
            "use_nucleus_path": True,
            "use_celltype": True,
            "lambda_c": args.lambda_c,
            "uot_solver": "sparse_topk",
            "uot_top_k": 64,
        },
        "feature_store": str(feature_path),
        "nucleus_feature_store": str(nucleus_path),
        "path_cache": str(path_cache),
    }
    (output / "reconstruction_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    del prediction, virtual, full, anchors_out, trained, all_anchors, train_anchors, target
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series-id", default="9589__g1")
    parser.add_argument("--anchor-dir", required=True, type=Path)
    parser.add_argument("--feature-store", required=True, type=Path)
    parser.add_argument("--nucleus-feature-store", required=True, type=Path)
    parser.add_argument("--path-cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--coordinate-frame", required=True)
    parser.add_argument("--sections", nargs="+", type=int, required=True)
    parser.add_argument("--training-sections", nargs="+", type=int)
    parser.add_argument("--heldout-section", type=int)
    parser.add_argument("--endpoint-left", type=int)
    parser.add_argument("--endpoint-right", type=int)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--pairs", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nucleus-weight", type=float, default=1.0)
    parser.add_argument("--lambda-c", type=float, default=10.0)
    parser.add_argument("--thickness-um", type=float, default=10.0)
    parser.add_argument("--reconstruction-steps", type=int, default=100)
    parser.add_argument("--reconstruction-chunk-size", type=int, default=512)
    parser.add_argument("--inference-chunk-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Train on every supplied anchor and reconstruct without a holdout evaluation.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not args.full and args.training_sections is None:
        parser.error("--training-sections is required unless --full is enabled")
    if args.full:
        args.training_sections = list(args.sections)
    start = time.time()
    record = train_and_reconstruct(args)
    record["elapsed_min"] = (time.time() - start) / 60.0
    (Path(args.output) / "reconstruction_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
