"""Run a reproducible component ablation on 00029/g0 section 031.

Section 031 expression is excluded from model setup/training. Its XY coordinates
are used only after fitting as query locations; expression is loaded for scoring
after predictions have been generated.
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
FRAME = "00029__g0_registered"
SECTIONS = (1, 11, 21, 31, 41, 51, 61, 71, 81, 91)

ARMS = {
    "baseline_original": (False, False, False),
    "baseline_physical": (True, False, False),
    "uot_only": (True, True, False),
    "path_only": (True, False, True),
    "full": (True, True, True),
}


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
        "rmse_log1p_cpm": float(np.sqrt(np.mean(difference * difference, dtype=np.float64))),
        "prediction_distance_median_um": float(np.median(distances)),
        "prediction_distance_p95_um": float(np.quantile(distances, 0.95)),
    }
    del pred, target, difference
    return result


def _config(name: str, output: Path, seed: int) -> HistologyConfig:
    use, uot, path = ARMS[name]
    return HistologyConfig(
        use_histology=use,
        use_morphology_uot=uot,
        use_morphology_path=path,
        feature_path=str(FEATURES),
        path_cache=str(output / "path_cache.h5") if path else None,
        section_key="section_id",
        coordinate_frame=FRAME,
        allow_linear_fallback=True,
        seed=seed,
    )


def _run_arm(
    name: str,
    output: Path,
    seed: int,
    epochs: int,
    pair_count: int,
    anchor_dir: Path,
) -> dict:
    arm_dir = output / name
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
            histology=_config(name, arm_dir, seed),
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
        torch.cuda.empty_cache()
        ckpts = sorted(model_dir.glob("*.ckpt"))
    if not ckpts:
        raise RuntimeError(f"Training produced no checkpoint for {name}")

    checkpoint = ckpts[0]
    trained = DeepSpatial()
    trained.load_checkpoint(str(checkpoint), sampling_method="dopri5")

    # Load section 031 only after fitting. Prediction receives a coordinate-only
    # AnnData; the held-out gene matrix is not passed into the model.
    target_path = _anchor_path(anchor_dir, 31)
    target = ad.read_h5ad(target_path)
    query = ad.AnnData(
        X=sp.csr_matrix((target.n_obs, 1), dtype=np.float32),
        obs=target.obs[["z_um"]].copy(),
        var=pd.DataFrame(index=["coordinate_query"]),
        obsm={"spatial_st_corrected": np.asarray(target.obsm["spatial_st_corrected"]).copy()},
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
        "arm": name,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "anchor_dir": str(anchor_dir.relative_to(ROOT)),
        "heldout_section": 31,
        "heldout_expression_used_for_training": False,
        "query_coordinates_source": "section-031 spatial_st_corrected; expression not passed to predictor",
        "training_sections": [s for s in SECTIONS if s not in (31, 51)],
        "seed": seed,
        "epochs": epochs,
        "sampled_uot_pairs_total": pair_count,
        "uot_solver": "sparse_topk",
        "uot_top_k": 64,
        "use_celltype": False,
        "components": {
            "use_histology": ARMS[name][0],
            "use_morphology_uot": ARMS[name][1],
            "use_morphology_path": ARMS[name][2],
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
        default=ROOT / "data/00029_g0/experiments/st_only_residual_baseline_v1",
    )
    parser.add_argument("--anchor-dir", type=Path, default=DEFAULT_ANCHOR_DIR)
    parser.add_argument("--arms", nargs="+", choices=tuple(ARMS), default=list(ARMS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--pairs", type=int, default=20_000)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    anchor_dir = args.anchor_dir if args.anchor_dir.is_absolute() else ROOT / args.anchor_dir
    output.mkdir(parents=True, exist_ok=True)
    if not FEATURES.is_file():
        raise FileNotFoundError(FEATURES)
    for section_id in SECTIONS:
        anchor_path = _anchor_path(anchor_dir, section_id)
        if not anchor_path.is_file():
            raise FileNotFoundError(anchor_path)
    manifest = {
        "experiment": "00029_g0_internal_ablation_holdout031_v1",
        "seed": args.seed,
        "epochs": args.epochs,
        "sampled_uot_pairs_total": args.pairs,
        "heldout_section": 31,
        "final_test_section_051_used": False,
        "training_sections": [s for s in SECTIONS if s not in (31, 51)],
        "anchor_dir": str(anchor_dir.relative_to(ROOT)),
        "feature_store": str(FEATURES.relative_to(ROOT)),
        "arms": {
            k: dict(zip(("use_histology", "use_morphology_uot", "use_morphology_path"), v))
            for k, v in ARMS.items()
        },
        "evaluation": "per-cell prediction at held-out registered XY; evaluate CP10k-log expression after prediction",
    }
    manifest_path = output / "experiment_manifest.json"
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    results = []
    for name in args.arms:
        start = time.time()
        print(f"START {name}", flush=True)
        record = _run_arm(
            name, output, args.seed, args.epochs, args.pairs, anchor_dir
        )
        results.append(record)
        print(
            f"DONE {name} in {(time.time()-start)/60:.1f} min: "
            f"gene_r={record['metrics']['macro_gene_pearson']:.5f}, "
            f"profile_r={record['metrics']['mean_profile_pearson']:.5f}, "
            f"MAE={record['metrics']['mae_log1p_cpm']:.5f}",
            flush=True,
        )
    summary = {item["arm"]: item["metrics"] for item in results}
    (output / "summary_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
