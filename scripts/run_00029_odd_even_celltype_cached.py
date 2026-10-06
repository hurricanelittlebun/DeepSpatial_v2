"""Odd/even anchor holdout with one-pass cell-type trajectory caching.

The ten 00029 anchors are split by alternating order, not by their numeric
section ID (all numeric IDs are odd):

* odd-order anchors: 1, 21, 41, 61, 81
* even-order anchors: 11, 31, 51, 71, 91

Only targets strictly bracketed by the opposite training group are scored.
The boundary targets 1 and 91 are recorded but excluded because they would
require extrapolation.  Each target is inferred once; the continuous c-flow
returned by that same ODE pass is written to a compressed cache and then used
to materialize cell-type labels and confidence.
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
import torch

from deepspatial import DeepSpatial
from scripts.run_00029_celltype_nucleus_uot_path import (
    _celltype_metrics,
    _expression_metrics,
    _histology_config,
    _load_anchors,
    _query_for_target,
)


ALL_SECTIONS = (1, 11, 21, 31, 41, 51, 61, 71, 81, 91)
ORDER_ODD = (1, 21, 41, 61, 81)
ORDER_EVEN = (11, 31, 51, 71, 91)
SCORED_ODD_TO_EVEN = (11, 31, 51, 71)
SCORED_EVEN_TO_ODD = (21, 41, 61, 81)
DEFAULT_ANCHOR_DIR = ROOT / "data/00029_g0/st_only_residual_v1/celltype_anchors_v1"
DEFAULT_FEATURES = ROOT / "data/00029_g0/uni2_features.h5"
DEFAULT_NUCLEUS_FEATURES = ROOT / "data/00029_g0/st_only_residual_v1/nucleus_features_registered/nucleus_features.h5"
DEFAULT_PATH_CACHE = ROOT / "data/00029_g0/st_only_residual_v1/paths.h5"
DEFAULT_OUTPUT = ROOT / "data/00029_g0/experiments/odd_even_celltype_cached_v1"
COORDINATE_FRAME = "00029__g0_registered"


def bracketing_pair(training_sections: list[int] | tuple[int, ...], target_section: int):
    """Return strict lower/upper training anchors or ``None`` at a boundary."""

    lower = [section for section in training_sections if section < target_section]
    upper = [section for section in training_sections if section > target_section]
    if not lower or not upper:
        return None
    return max(lower), min(upper)


def labels_from_cached_trajectory(
    cache: dict,
    *,
    n_query: int,
    categories,
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize labels from cached continuous c trajectories, without ODE."""

    labels = np.full(n_query, None, dtype=object)
    confidence = np.full(n_query, np.nan, dtype=np.float32)
    steps = int(cache["steps"])
    categories = np.asarray(categories, dtype=object)
    for plane in cache["planes"].values():
        query_ids = np.asarray(plane["query_ids"], dtype=np.int64)
        t_values = np.asarray(plane["t_values"], dtype=np.float32)
        trajectory = np.asarray(plane["c_traj_cont"], dtype=np.float32)
        if trajectory.shape[:2] != (steps, len(query_ids)):
            raise ValueError("Cached c trajectory shape does not match query IDs")
        time_indices = np.clip(
            np.rint(t_values * (steps - 1)).astype(np.int64),
            0,
            steps - 1,
        )
        row_indices = np.arange(len(query_ids), dtype=np.int64)
        state = trajectory[time_indices, row_indices]
        state = state - state.max(axis=1, keepdims=True)
        probabilities = np.exp(state)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        class_indices = np.argmax(probabilities, axis=1)
        labels[query_ids] = categories[class_indices]
        confidence[query_ids] = probabilities[np.arange(len(query_ids)), class_indices]

    if any(value is None for value in labels) or not np.isfinite(confidence).all():
        raise ValueError("Cell-type cache does not cover every query cell")
    return labels.astype(str), confidence


def _save_plane_cache(
    cache: dict,
    path: Path,
    *,
    labels: np.ndarray,
    confidence: np.ndarray,
    direction: str,
    target_section: int,
    lower_section: int,
    upper_section: int,
) -> None:
    """Persist the one-pass trajectory and its derived labels for one target."""

    if len(cache["planes"]) != 1:
        raise ValueError("A target section must produce exactly one cached plane")
    plane = next(iter(cache["planes"].values()))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        version=np.array([int(cache["version"])], dtype=np.int64),
        steps=np.array([int(cache["steps"])], dtype=np.int64),
        z_um=np.array([float(plane["z_um"])], dtype=np.float32),
        query_ids=np.asarray(plane["query_ids"], dtype=np.int64),
        t_values=np.asarray(plane["t_values"], dtype=np.float32),
        source_indices=np.asarray(plane["source_indices"], dtype=np.int64),
        c_traj_cont=np.asarray(plane["c_traj_cont"], dtype=np.float32),
        predicted_labels=np.asarray(labels, dtype="U64"),
        predicted_confidence=np.asarray(confidence, dtype=np.float32),
        direction=np.array([direction]),
        target_section=np.array([int(target_section)], dtype=np.int64),
        lower_section=np.array([int(lower_section)], dtype=np.int64),
        upper_section=np.array([int(upper_section)], dtype=np.int64),
    )


def _mean_metrics(records: list[dict], metric_group: str) -> dict:
    values = [record[metric_group] for record in records]
    n_total = sum(int(value["n_cells"]) for value in values)
    result = {
        "n_sections": len(values),
        "n_cells_total": n_total,
    }
    for key in values[0]:
        if key in {"n_cells", "n_genes", "n_categories_evaluated"}:
            continue
        result[f"mean_{key}"] = float(np.mean([value[key] for value in values]))
    if metric_group == "celltype":
        result["weighted_accuracy"] = float(
            sum(value["accuracy"] * value["n_cells"] for value in values) / n_total
        )
    return result


def _train_or_load(
    *,
    output: Path,
    training_sections: tuple[int, ...],
    anchor_dir: Path,
    feature_path: Path,
    nucleus_feature_path: Path,
    path_cache: Path,
    epochs: int,
    pairs: int,
    seed: int,
    lambda_c: float,
    nucleus_weight: float,
    force: bool,
) -> tuple[DeepSpatial, Path]:
    model_dir = output / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted(p for p in model_dir.glob("*.ckpt") if p.name != "last.ckpt")
    if force or not checkpoints:
        anchors = _load_anchors(anchor_dir, training_sections)
        pl.seed_everything(seed, workers=True)
        model = DeepSpatial()
        model.setup_data(
            anchors,
            spatial_key="spatial_st_corrected",
            z_key="z_um",
            label_key="cell_class",
            n_samples_base=pairs,
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
                coordinate_frame=COORDINATE_FRAME,
                nucleus_weight=nucleus_weight,
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
        del model, anchors
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        checkpoints = sorted(p for p in model_dir.glob("*.ckpt") if p.name != "last.ckpt")
    if not checkpoints:
        raise RuntimeError(f"No checkpoint produced under {model_dir}")
    checkpoint = checkpoints[-1]
    trained = DeepSpatial()
    trained.load_checkpoint(str(checkpoint), sampling_method="dopri5")
    return trained, checkpoint


def run_direction(args, *, name: str, training_sections: tuple[int, ...], targets: tuple[int, ...]) -> dict:
    output = Path(args.output).resolve() / name
    output.mkdir(parents=True, exist_ok=True)
    all_anchors = {
        section: _load_anchors(Path(args.anchor_dir), (section,))[0]
        for section in ALL_SECTIONS
    }
    categories = sorted(
        {
            str(value)
            for section in training_sections
            for value in all_anchors[section].obs["cell_class"].astype(str)
        }
    )
    trained, checkpoint = _train_or_load(
        output=output,
        training_sections=training_sections,
        anchor_dir=Path(args.anchor_dir),
        feature_path=Path(args.feature_store),
        nucleus_feature_path=Path(args.nucleus_feature_store),
        path_cache=Path(args.path_cache),
        epochs=args.epochs,
        pairs=args.pairs,
        seed=args.seed,
        lambda_c=args.lambda_c,
        nucleus_weight=args.nucleus_weight,
        force=args.force,
    )
    cache_dir = output / "celltype_trajectory_cache"
    records = []
    for target_section in targets:
        pair = bracketing_pair(training_sections, target_section)
        if pair is None:
            continue
        lower_section, upper_section = pair
        target = all_anchors[target_section]
        query = _query_for_target(target, "spatial_st_corrected", "z_um")
        trajectory_cache = {}
        prediction = trained.predict_on_he_cells(
            query,
            all_anchors[lower_section],
            all_anchors[upper_section],
            steps=args.prediction_steps,
            chunk_size=args.inference_chunk_size,
            device=args.device,
            trajectory_cache=trajectory_cache,
        )
        labels, confidence = labels_from_cached_trajectory(
            trajectory_cache,
            n_query=target.n_obs,
            categories=np.asarray(categories, dtype=object),
        )
        prediction.obs["cell_class"] = pd.Categorical(labels, categories=categories)
        prediction.obs["celltype_confidence"] = confidence
        cache_path = cache_dir / f"section-{target_section:03d}_trajectory.npz"
        _save_plane_cache(
            trajectory_cache,
            cache_path,
            labels=labels,
            confidence=confidence,
            direction=name,
            target_section=target_section,
            lower_section=lower_section,
            upper_section=upper_section,
        )
        record = {
            "target_section": target_section,
            "lower_section": lower_section,
            "upper_section": upper_section,
            "expression": _expression_metrics(
                prediction.X,
                target.X,
                prediction.obs["prediction_distance_um"].to_numpy(),
            ),
            "celltype": _celltype_metrics(prediction, target, categories),
            "trajectory_cache": str(cache_path),
            "trajectory_cache_used_for_labels": True,
        }
        records.append(record)

    boundary_targets = sorted(set(ALL_SECTIONS) - set(targets))
    summary = {
        "direction": name,
        "training_sections": list(training_sections),
        "scored_target_sections": [record["target_section"] for record in records],
        "boundary_not_scored": boundary_targets,
        "checkpoint": str(checkpoint),
        "use_celltype": True,
        "lambda_c": args.lambda_c,
        "prediction_steps": args.prediction_steps,
        "trajectory_cache_used": True,
        "per_section": records,
        "expression": _mean_metrics(records, "expression"),
        "celltype": _mean_metrics(records, "celltype"),
    }
    (output / "holdout_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    del trained, all_anchors
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-dir", type=Path, default=DEFAULT_ANCHOR_DIR)
    parser.add_argument("--feature-store", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--nucleus-feature-store", type=Path, default=DEFAULT_NUCLEUS_FEATURES)
    parser.add_argument("--path-cache", type=Path, default=DEFAULT_PATH_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--pairs", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lambda-c", type=float, default=10.0)
    parser.add_argument("--nucleus-weight", type=float, default=1.0)
    parser.add_argument("--prediction-steps", type=int, default=40)
    parser.add_argument("--inference-chunk-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    start = time.time()
    odd_to_even = run_direction(
        args,
        name="odd_to_even",
        training_sections=ORDER_ODD,
        targets=SCORED_ODD_TO_EVEN,
    )
    even_to_odd = run_direction(
        args,
        name="even_to_odd",
        training_sections=ORDER_EVEN,
        targets=SCORED_EVEN_TO_ODD,
    )
    manifest = {
        "status": "complete",
        "series_id": "00029__g0",
        "split_semantics": "alternating_anchor_order",
        "odd_order_training_sections": list(ORDER_ODD),
        "even_order_training_sections": list(ORDER_EVEN),
        "odd_to_even_scored_targets": list(SCORED_ODD_TO_EVEN),
        "even_to_odd_scored_targets": list(SCORED_EVEN_TO_ODD),
        "boundary_not_scored": [1, 91],
        "trajectory_cache": {
            "enabled": True,
            "one_pass_per_target": True,
            "format": "compressed npz",
        },
        "odd_to_even": odd_to_even,
        "even_to_odd": even_to_odd,
        "elapsed_min": (time.time() - start) / 60.0,
    }
    Path(args.output).mkdir(parents=True, exist_ok=True)
    (Path(args.output) / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
