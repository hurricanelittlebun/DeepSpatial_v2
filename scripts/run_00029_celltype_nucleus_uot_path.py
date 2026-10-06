"""Retrain and evaluate the 00029 nucleus-guided model with cell types.

This entrypoint intentionally uses a new experiment directory.  It consumes
the validated per-section anchors produced by
``prepare_00029_celltype_anchors.py`` and keeps the existing morphology UOT,
UNI2 path, and nucleus path configuration unchanged.
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


SECTIONS = (1, 11, 21, 31, 41, 51, 61, 71, 81, 91)
TRAINING_SECTIONS = (1, 11, 21, 41, 61, 71, 81, 91)
HELDOUT_SECTION = 31
EXPERIMENT_NAME = "st_only_residual_nucleus_uot_path_celltype_v1"

DEFAULT_ANCHOR_DIR = ROOT / "data/00029_g0/st_only_residual_v1/celltype_anchors_v1"
DEFAULT_OUTPUT = ROOT / f"data/00029_g0/experiments/{EXPERIMENT_NAME}"
FEATURES = ROOT / "data/00029_g0/uni2_features.h5"
NUCLEUS_FEATURES = ROOT / "data/00029_g0/st_only_residual_v1/nucleus_features_registered/nucleus_features.h5"
PATH_CACHE = ROOT / "data/00029_g0/st_only_residual_v1/paths.h5"
FRAME = "00029__g0_registered"


def build_training_spec(*, lambda_c: float = 10.0) -> dict:
    """Return the immutable compatibility-critical training settings."""

    return {
        "output_name": EXPERIMENT_NAME,
        "training_sections": list(TRAINING_SECTIONS),
        "heldout_section": HELDOUT_SECTION,
        "label_key": "cell_class",
        "use_celltype": True,
        "uot_solver": "sparse_topk",
        "uot_top_k": 64,
        "uot_bidirectional": True,
        "use_nucleus_path": True,
        "use_histology": True,
        "use_morphology_uot": True,
        "use_morphology_path": True,
        "coordinate_frame": FRAME,
        "lambda_c": float(lambda_c),
    }


def _anchor_path(anchor_dir: Path, section: int) -> Path:
    return anchor_dir / f"section-{section:03d}.h5ad"


def _load_anchors(anchor_dir: Path, sections: tuple[int, ...]) -> list[ad.AnnData]:
    anchors = []
    for section in sections:
        path = _anchor_path(anchor_dir, section)
        if not path.is_file():
            raise FileNotFoundError(path)
        anchor = ad.read_h5ad(path)
        if "cell_class" not in anchor.obs:
            raise ValueError(f"{path} has no cell_class label")
        if anchor.obs["cell_class"].isna().any():
            raise ValueError(f"{path} contains missing cell_class labels")
        if anchor.obs["z_um"].nunique() != 1:
            raise ValueError(f"{path} must contain one physical z_um value")
        if "spatial_st_corrected" not in anchor.obsm:
            raise ValueError(f"{path} has no registered spatial_st_corrected coordinates")
        anchors.append(anchor)
    genes = anchors[0].var_names.tolist()
    for anchor in anchors[1:]:
        if anchor.var_names.tolist() != genes:
            raise ValueError("Anchor gene order differs across sections")
    labels = sorted({str(x) for anchor in anchors for x in anchor.obs["cell_class"].astype(str)})
    if len(labels) < 2:
        raise ValueError("Cell-type training requires at least two observed labels")
    return anchors


def _histology_config(
    *,
    seed: int,
    feature_path: Path,
    nucleus_feature_path: Path,
    path_cache: Path,
    coordinate_frame: str,
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
        nucleus_weight=nucleus_weight,
        allow_linear_fallback=True,
        seed=seed,
    )


def _dense_log_cpm(matrix) -> np.ndarray:
    values = matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)
    values = values.astype(np.float32, copy=False)
    library = values.sum(axis=1, keepdims=True)
    return np.log1p(values * (1e4 / np.maximum(library, 1.0)))


def _correlation_columns(a: np.ndarray, b: np.ndarray) -> float:
    am = a.mean(axis=0, dtype=np.float64)
    bm = b.mean(axis=0, dtype=np.float64)
    an = a.astype(np.float64, copy=False)
    bn = b.astype(np.float64, copy=False)
    numerator = np.sum(an * bn, axis=0) - len(a) * am * bm
    avar = np.sum(an * an, axis=0) - len(a) * am * am
    bvar = np.sum(bn * bn, axis=0) - len(b) * bm * bm
    denominator = np.sqrt(np.maximum(avar, 0) * np.maximum(bvar, 0))
    valid = denominator > 1e-12
    if not valid.any():
        return float("nan")
    return float(np.mean(numerator[valid] / denominator[valid]))


def _expression_metrics(predicted, truth, distances) -> dict:
    pred = _dense_log_cpm(predicted)
    target = _dense_log_cpm(truth)
    difference = pred - target
    result = {
        "n_cells": int(pred.shape[0]),
        "n_genes": int(pred.shape[1]),
        "macro_gene_pearson": _correlation_columns(pred, target),
        "mean_profile_pearson": float(np.corrcoef(pred.mean(axis=0), target.mean(axis=0))[0, 1]),
        "mae_log1p_cpm": float(np.mean(np.abs(difference), dtype=np.float64)),
        "rmse_log1p_cpm": float(np.sqrt(np.mean(difference * difference, dtype=np.float64))),
        "prediction_distance_median_um": float(np.median(distances)),
        "prediction_distance_p95_um": float(np.quantile(distances, 0.95)),
    }
    del pred, target, difference
    return result


def _celltype_metrics(prediction: ad.AnnData, target: ad.AnnData, categories: list[str]) -> dict:
    y_true = target.obs["cell_class"].astype(str).to_numpy()
    y_pred = prediction.obs["cell_class"].astype(str).to_numpy()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=categories, average="macro", zero_division=0)
        ),
        "n_cells": int(len(y_true)),
        "n_categories_evaluated": int(len(set(y_true))),
        "unassigned_truth_fraction": float(np.mean(y_true == "unassigned")),
        "unassigned_prediction_fraction": float(np.mean(y_pred == "unassigned")),
    }


def _query_for_target(target: ad.AnnData, spatial_key: str, z_key: str) -> ad.AnnData:
    return ad.AnnData(
        X=sp.csr_matrix((target.n_obs, 1), dtype=np.float32),
        obs=target.obs[[z_key]].copy(),
        var=pd.DataFrame(index=["coordinate_query"]),
        obsm={spatial_key: np.asarray(target.obsm[spatial_key]).copy()},
    )


def run_experiment(
    *,
    output: Path,
    anchor_dir: Path,
    feature_path: Path,
    nucleus_feature_path: Path,
    path_cache: Path,
    coordinate_frame: str,
    seed: int,
    epochs: int,
    pair_count: int,
    nucleus_weight: float,
    lambda_c: float,
    gpu_device: str,
    force: bool = False,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "holdout_metrics.json"
    if result_path.exists() and not force:
        return json.loads(result_path.read_text())
    for path in (feature_path, nucleus_feature_path, path_cache):
        if not path.is_file():
            raise FileNotFoundError(path)

    train_anchors = _load_anchors(anchor_dir, TRAINING_SECTIONS)
    target = ad.read_h5ad(_anchor_path(anchor_dir, HELDOUT_SECTION))
    all_categories = sorted(
        {str(x) for anchor in train_anchors for x in anchor.obs["cell_class"].astype(str)}
    )
    if set(target.obs["cell_class"].astype(str)) - set(all_categories):
        raise ValueError("Held-out section contains a cell type absent from training")

    pl.seed_everything(seed, workers=True)
    model = DeepSpatial()
    model.setup_data(
        train_anchors,
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
        histology=_histology_config(
            seed=seed,
            feature_path=feature_path,
            nucleus_feature_path=nucleus_feature_path,
            path_cache=path_cache,
            coordinate_frame=coordinate_frame,
            nucleus_weight=nucleus_weight,
        ),
    )
    if model.num_classes != len(model.categories):
        raise RuntimeError("Model class dimension and category vocabulary disagree")
    if model.num_classes != len(all_categories):
        raise RuntimeError("Training category vocabulary changed unexpectedly")
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
        save_dir=str(output / "model"),
        accelerator="gpu",
        devices=1,
        save_ckpt=True,
    )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    checkpoints = sorted((output / "model").glob("*.ckpt"))
    if not checkpoints:
        raise RuntimeError("Training produced no checkpoint")
    checkpoint = checkpoints[-1]
    trained = DeepSpatial()
    trained.load_checkpoint(str(checkpoint), sampling_method="dopri5")
    if not trained.use_celltype or trained.num_classes != len(all_categories):
        raise RuntimeError("Saved checkpoint is not compatible with the cell-type branch")
    query = _query_for_target(target, "spatial_st_corrected", "z_um")
    prediction = trained.predict_on_he_cells(
        query,
        ad.read_h5ad(_anchor_path(anchor_dir, 21)),
        ad.read_h5ad(_anchor_path(anchor_dir, 41)),
        steps=40,
        chunk_size=512,
        device=gpu_device,
    )
    expression = _expression_metrics(
        prediction.X,
        target.X,
        prediction.obs["prediction_distance_um"].to_numpy(),
    )
    celltype = _celltype_metrics(prediction, target, all_categories)
    record = {
        "experiment": output.name,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "anchor_dir": str(anchor_dir.relative_to(ROOT)),
        "annotation_label": "assigned_celltype -> cell_class",
        "join_key": "source_cell_id",
        "training_sections": list(TRAINING_SECTIONS),
        "heldout_section": HELDOUT_SECTION,
        "heldout_expression_used_for_training": False,
        "seed": seed,
        "epochs": epochs,
        "sampled_uot_pairs_total": pair_count,
        "uot_solver": "sparse_topk",
        "uot_top_k": 64,
        "use_celltype": True,
        "lambda_c": float(lambda_c),
        "num_classes": int(trained.num_classes),
        "categories": all_categories,
        "components": build_training_spec(lambda_c=lambda_c),
        "histology": {
            "feature_store": str(feature_path.relative_to(ROOT)),
            "nucleus_feature_store": str(nucleus_feature_path.relative_to(ROOT)),
            "path_cache": str(path_cache.relative_to(ROOT)),
            "coordinate_frame": coordinate_frame,
            "nucleus_weight": nucleus_weight,
            "nucleus_semantics": "local nuclear anatomy; not cell lineage",
        },
        "expression_metrics": expression,
        "celltype_metrics": celltype,
    }
    result_path.write_text(json.dumps(record, indent=2) + "\n")
    (output / "summary_metrics.json").write_text(
        json.dumps({"expression": expression, "celltype": celltype}, indent=2) + "\n"
    )
    del prediction, target, trained, train_anchors
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--anchor-dir", type=Path, default=DEFAULT_ANCHOR_DIR)
    parser.add_argument("--feature-store", type=Path, default=FEATURES)
    parser.add_argument("--nucleus-feature-store", type=Path, default=NUCLEUS_FEATURES)
    parser.add_argument("--path-cache", type=Path, default=PATH_CACHE)
    parser.add_argument("--coordinate-frame", default=FRAME)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--pairs", type=int, default=20_000)
    parser.add_argument("--nucleus-weight", type=float, default=1.0)
    parser.add_argument("--lambda-c", type=float, default=10.0)
    parser.add_argument("--gpu-device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else ROOT / path

    output = resolve(args.output)
    start = time.time()
    record = run_experiment(
        output=output,
        anchor_dir=resolve(args.anchor_dir),
        feature_path=resolve(args.feature_store),
        nucleus_feature_path=resolve(args.nucleus_feature_store),
        path_cache=resolve(args.path_cache),
        coordinate_frame=args.coordinate_frame,
        seed=args.seed,
        epochs=args.epochs,
        pair_count=args.pairs,
        nucleus_weight=args.nucleus_weight,
        lambda_c=args.lambda_c,
        gpu_device=args.gpu_device,
        force=args.force,
    )
    print(f"DONE in {(time.time() - start) / 60:.1f} min", flush=True)
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
