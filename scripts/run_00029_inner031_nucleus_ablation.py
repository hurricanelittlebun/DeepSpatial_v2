"""Run the nucleus-guided anatomical-path ablation on 00029/g0.

This is an independent, path-only nuclear ablation.  It keeps the existing
00029/g0 holdout protocol and enables morphology-aware UOT plus the
UNI2/nuclear morphology path, while leaving the spatial head disabled.  The
nuclear descriptor is local anatomy, not cell tracking or lineage.
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

from deepspatial import DeepSpatial
from deepspatial.histology import HistologyConfig


DEFAULT_ANCHOR_DIR = ROOT / "data/00029_g0/st_only_residual_v1/anchors"
FEATURES = ROOT / "data/00029_g0/uni2_features.h5"
NUCLEUS_FEATURES = ROOT / "data/00029_g0/st_only_residual_v1/nucleus_features_registered/nucleus_features.h5"
PATH_CACHE = ROOT / "data/00029_g0/st_only_residual_v1/paths.h5"
FRAME = "00029__g0_registered"
SECTIONS = (1, 11, 21, 31, 41, 51, 61, 71, 81, 91)


def _anchor_path(anchor_dir: Path, section: int) -> Path:
    """Return the selected package's canonical per-anchor H5AD."""

    return Path(anchor_dir) / f"section-{section:03d}.h5ad"


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


def _score(predicted, truth, distances) -> dict:
    pred = _dense_log_cpm(predicted)
    target = _dense_log_cpm(truth)
    difference = pred - target
    pmean = pred.mean(axis=0)
    tmean = target.mean(axis=0)
    profile_corr = float(np.corrcoef(pmean, tmean)[0, 1])
    result = {
        "n_cells": int(pred.shape[0]),
        "n_genes": int(pred.shape[1]),
        "macro_gene_pearson": _correlation_columns(pred, target),
        "mean_profile_pearson": profile_corr,
        "mae_log1p_cpm": float(np.mean(np.abs(difference), dtype=np.float64)),
        "rmse_log1p_cpm": float(
            np.sqrt(np.mean(difference * difference, dtype=np.float64))
        ),
        "prediction_distance_median_um": float(np.median(distances)),
        "prediction_distance_p95_um": float(np.quantile(distances, 0.95)),
    }
    del pred, target, difference
    return result


def _config(
    output: Path,
    seed: int,
    nucleus_weight: float,
    *,
    feature_path: Path,
    nucleus_feature_path: Path,
    path_cache: Path,
    coordinate_frame: str,
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


def _run(
    output: Path,
    seed: int,
    epochs: int,
    pair_count: int,
    nucleus_weight: float,
    *,
    anchor_dir: Path,
    feature_path: Path,
    nucleus_feature_path: Path,
    path_cache: Path,
    coordinate_frame: str,
):
    arm = "nucleus_path"
    arm_dir = output / arm
    model_dir = arm_dir / "model"
    arm_dir.mkdir(parents=True, exist_ok=True)
    result_path = arm_dir / "holdout_metrics.json"
    if result_path.exists():
        return json.loads(result_path.read_text())

    ckpts = sorted(model_dir.glob("*.ckpt"))
    if not ckpts:
        pl.seed_everything(seed, workers=True)
        training_sections = [s for s in SECTIONS if s not in (31, 51)]
        anchors = [
            ad.read_h5ad(_anchor_path(anchor_dir, s)) for s in training_sections
        ]
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
            use_celltype=False,
            histology=_config(
                arm_dir,
                seed,
                nucleus_weight,
                feature_path=feature_path,
                nucleus_feature_path=nucleus_feature_path,
                path_cache=path_cache,
                coordinate_frame=coordinate_frame,
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
            lambda_c=10.0,
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
        del model, anchors
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        ckpts = sorted(model_dir.glob("*.ckpt"))
    if not ckpts:
        raise RuntimeError("Training produced no checkpoint for nucleus_path")

    checkpoint = ckpts[0]
    trained = DeepSpatial()
    trained.load_checkpoint(str(checkpoint), sampling_method="dopri5")

    target_path = _anchor_path(anchor_dir, 31)
    target = ad.read_h5ad(target_path)
    query = ad.AnnData(
        X=sp.csr_matrix((target.n_obs, 1), dtype=np.float32),
        obs=target.obs[["z_um"]].copy(),
        var=pd.DataFrame(index=["coordinate_query"]),
        obsm={
            "spatial_st_corrected": np.asarray(
                target.obsm["spatial_st_corrected"]
            ).copy()
        },
    )
    prediction = trained.predict_on_he_cells(
        query,
        ad.read_h5ad(_anchor_path(anchor_dir, 21)),
        ad.read_h5ad(_anchor_path(anchor_dir, 41)),
        steps=40,
        chunk_size=512,
        device="cuda",
    )
    metrics = _score(
        prediction.X,
        target.X,
        prediction.obs["prediction_distance_um"].to_numpy(),
    )
    record = {
        "arm": arm,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "anchor_dir": str(anchor_dir.relative_to(ROOT)),
        "heldout_section": 31,
        "heldout_expression_used_for_training": False,
        "query_coordinates_source": "section-031 spatial_st_corrected in the declared fixed H&E frame; expression not passed to predictor",
        "training_sections": [s for s in SECTIONS if s not in (31, 51)],
        "seed": seed,
        "epochs": epochs,
        "sampled_uot_pairs_total": pair_count,
        "uot_solver": "sparse_topk",
        "uot_top_k": 64,
        "use_celltype": False,
        "components": {
            "use_histology": True,
            "use_morphology_uot": True,
            "use_morphology_path": True,
            "use_nucleus_path": True,
        },
        "histology": {
            "feature_store": str(feature_path.relative_to(ROOT)),
            "nucleus_feature_store": str(nucleus_feature_path.relative_to(ROOT)),
            "path_cache": str(path_cache.relative_to(ROOT)),
            "coordinate_frame": coordinate_frame,
            "nucleus_weight": nucleus_weight,
            "nucleus_semantics": "local nuclear anatomy; not cell lineage",
        },
        "metrics": metrics,
    }
    result_path.write_text(json.dumps(record, indent=2) + "\n")
    del prediction, target, query, trained
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/00029_g0/experiments/st_only_residual_nucleus_uot_path_v1",
    )
    parser.add_argument("--anchor-dir", type=Path, default=DEFAULT_ANCHOR_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--pairs", type=int, default=20_000)
    parser.add_argument("--nucleus-weight", type=float, default=1.0)
    parser.add_argument("--feature-store", type=Path, default=FEATURES)
    parser.add_argument("--nucleus-feature-store", type=Path, default=NUCLEUS_FEATURES)
    parser.add_argument("--path-cache", type=Path, default=PATH_CACHE)
    parser.add_argument("--coordinate-frame", default=FRAME)
    parser.add_argument("--experiment-name", default=None)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    anchor_dir = args.anchor_dir if args.anchor_dir.is_absolute() else ROOT / args.anchor_dir
    feature_path = args.feature_store if args.feature_store.is_absolute() else ROOT / args.feature_store
    nucleus_feature_path = (
        args.nucleus_feature_store
        if args.nucleus_feature_store.is_absolute()
        else ROOT / args.nucleus_feature_store
    )
    path_cache = args.path_cache if args.path_cache.is_absolute() else ROOT / args.path_cache
    output.mkdir(parents=True, exist_ok=True)
    if not feature_path.is_file():
        raise FileNotFoundError(feature_path)
    if not nucleus_feature_path.is_file():
        raise FileNotFoundError(nucleus_feature_path)
    if not path_cache.is_file():
        raise FileNotFoundError(path_cache)
    for section_id in SECTIONS:
        anchor_path = _anchor_path(anchor_dir, section_id)
        if not anchor_path.is_file():
            raise FileNotFoundError(anchor_path)
    manifest = {
        "experiment": args.experiment_name or output.name,
        "seed": args.seed,
        "epochs": args.epochs,
        "sampled_uot_pairs_total": args.pairs,
        "heldout_section": 31,
        "final_test_section_051_used": False,
        "training_sections": [s for s in SECTIONS if s not in (31, 51)],
        "anchor_dir": str(anchor_dir.relative_to(ROOT)),
        "feature_store": str(feature_path.relative_to(ROOT)),
        "nucleus_feature_store": str(nucleus_feature_path.relative_to(ROOT)),
        "path_cache": str(path_cache.relative_to(ROOT)),
        "coordinate_frame": args.coordinate_frame,
        "nucleus_weight": args.nucleus_weight,
        "components": {
            "use_histology": True,
            "use_morphology_uot": True,
            "use_morphology_path": True,
            "use_nucleus_path": True,
            "use_celltype": False,
        },
        "evaluation": "per-cell prediction at held-out corrected XY in the declared feature frame; evaluate CP10k-log expression after prediction",
    }
    manifest_path = output / "experiment_manifest.json"
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    start = time.time()
    print("START nucleus_path", flush=True)
    record = _run(
        output,
        args.seed,
        args.epochs,
        args.pairs,
        args.nucleus_weight,
        anchor_dir=anchor_dir,
        feature_path=feature_path,
        nucleus_feature_path=nucleus_feature_path,
        path_cache=path_cache,
        coordinate_frame=args.coordinate_frame,
    )
    print(
        f"DONE nucleus_path in {(time.time() - start) / 60:.1f} min: "
        f"gene_r={record['metrics']['macro_gene_pearson']:.5f}, "
        f"profile_r={record['metrics']['mean_profile_pearson']:.5f}, "
        f"MAE={record['metrics']['mae_log1p_cpm']:.5f}",
        flush=True,
    )
    (output / "summary_metrics.json").write_text(
        json.dumps({"nucleus_path": record["metrics"]}, indent=2) + "\n"
    )
    print(json.dumps(record["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
