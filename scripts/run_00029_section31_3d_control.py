"""3-D reconstruction control experiment for held-out section 031.

This is a deployment-style evaluation.  Section 031 is excluded from both
training runs and is never used to generate the virtual volume.  Each arm
first reconstructs the complete volume covered by its training anchors, then
extracts the generated cells near the physical depth of section 031.  Only
after that are generated cells matched to the real section for scoring.

Arms:
    he_celltype: cell-type branch + morphology/nucleus H&E components.
    celltype_only: cell-type branch without all histology components.
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
from scipy.spatial import cKDTree
from sklearn.metrics import accuracy_score, f1_score

from deepspatial import DeepSpatial
from deepspatial.histology import HistologyConfig
from scripts.run_00029_celltype_nucleus_uot_path import (
    _expression_metrics,
    _histology_config,
    _load_anchors,
)


TRAINING_SECTIONS = (1, 11, 21, 41, 61, 71, 81, 91)
HELDOUT_SECTION = 31
DEFAULT_ANCHOR_DIR = ROOT / "data/00029_g0/st_only_residual_v1/celltype_anchors_v1"
DEFAULT_FEATURES = ROOT / "data/00029_g0/uni2_features.h5"
DEFAULT_NUCLEUS_FEATURES = ROOT / (
    "data/00029_g0/st_only_residual_v1/"
    "nucleus_features_registered/nucleus_features.h5"
)
DEFAULT_PATH_CACHE = ROOT / "data/00029_g0/st_only_residual_v1/paths.h5"
DEFAULT_OUTPUT = ROOT / "data/00029_g0/experiments/section31_3d_control_v1"
COORDINATE_FRAME = "00029__g0_registered"


def _anchor_path(anchor_dir: Path, section: int) -> Path:
    return anchor_dir / f"section-{section:03d}.h5ad"


def _materialize_labels_from_flow(
    flow_state: np.ndarray, categories: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Convert cached continuous cell-type states into labels and confidence."""

    state = np.asarray(flow_state, dtype=np.float32)
    if state.ndim != 2 or state.shape[1] != len(categories):
        raise ValueError("celltype_flow_state has an incompatible shape")
    shifted = state - np.max(state, axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
    indices = np.argmax(probabilities, axis=1)
    return (
        np.asarray(categories, dtype=object)[indices].astype(str),
        probabilities[np.arange(len(indices)), indices].astype(np.float32),
    )


def _unique_spatial_matches(
    generated_xy: np.ndarray,
    target_xy: np.ndarray,
    *,
    max_distance_um: float,
    candidate_k: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedily make one-to-one sparse spatial matches.

    The target coordinates are used only after reconstruction, for evaluation.
    They are never passed into the model or used to generate virtual cells.
    """

    generated_xy = np.asarray(generated_xy, dtype=np.float64)
    target_xy = np.asarray(target_xy, dtype=np.float64)
    if len(generated_xy) == 0 or len(target_xy) == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
        )
    tree = cKDTree(generated_xy)
    k = min(max(1, int(candidate_k)), len(generated_xy))
    distances, indices = tree.query(
        target_xy,
        k=k,
        distance_upper_bound=float(max_distance_um),
    )
    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    pairs = []
    for target_index in range(len(target_xy)):
        for rank in range(k):
            distance = float(distances[target_index, rank])
            generated_index = int(indices[target_index, rank])
            if generated_index < len(generated_xy) and np.isfinite(distance):
                pairs.append((distance, target_index, generated_index))
    pairs.sort(key=lambda value: value[0])

    used_targets = np.zeros(len(target_xy), dtype=bool)
    used_generated = np.zeros(len(generated_xy), dtype=bool)
    matched_targets = []
    matched_generated = []
    matched_distances = []
    for distance, target_index, generated_index in pairs:
        if used_targets[target_index] or used_generated[generated_index]:
            continue
        used_targets[target_index] = True
        used_generated[generated_index] = True
        matched_targets.append(target_index)
        matched_generated.append(generated_index)
        matched_distances.append(distance)

    # Keep target order deterministic for reproducible metric rows.
    order = np.argsort(np.asarray(matched_targets, dtype=np.int64))
    return (
        np.asarray(matched_targets, dtype=np.int64)[order],
        np.asarray(matched_generated, dtype=np.int64)[order],
        np.asarray(matched_distances, dtype=np.float32)[order],
    )


def match_fixed_depth(
    reconstructed: ad.AnnData,
    target: ad.AnnData,
    *,
    target_z: float,
    categories: list[str],
    tolerance_um: float = 5.0,
    max_distance_um: float = 50.0,
    spatial_key: str = "spatial_st_corrected",
    z_key: str = "z_um",
) -> tuple[ad.AnnData, dict]:
    """Extract a generated Z layer and match it to target cells for scoring."""

    if spatial_key not in reconstructed.obsm or spatial_key not in target.obsm:
        raise KeyError(f"Both objects must contain obsm[{spatial_key!r}]")
    if z_key not in reconstructed.obs or z_key not in target.obs:
        raise KeyError(f"Both objects must contain obs[{z_key!r}]")
    if "celltype_flow_state" not in reconstructed.obsm:
        raise KeyError("Reconstruction lacks cached celltype_flow_state")

    z_values = np.asarray(reconstructed.obs[z_key], dtype=np.float64)
    layer_mask = np.abs(z_values - float(target_z)) <= float(tolerance_um)
    layer_indices = np.flatnonzero(layer_mask)
    if not len(layer_indices):
        raise ValueError("No generated cells fall inside the requested Z layer")

    target_indices, generated_indices, distances = _unique_spatial_matches(
        np.asarray(reconstructed.obsm[spatial_key])[layer_indices],
        np.asarray(target.obsm[spatial_key]),
        max_distance_um=max_distance_um,
    )
    selected_generated = layer_indices[generated_indices]
    selected_target_names = target.obs_names.to_numpy()[target_indices]
    if not len(selected_generated):
        raise ValueError("No generated cells could be matched to the target section")

    obs = reconstructed.obs.iloc[selected_generated].copy()
    obs.index = selected_target_names
    obs["target_obs_name"] = selected_target_names
    obs["prediction_distance_um"] = distances
    obs["fixed_depth_target_z_um"] = float(target_z)
    labels, confidence = _materialize_labels_from_flow(
        np.asarray(reconstructed.obsm["celltype_flow_state"])[selected_generated],
        categories,
    )
    obs["cell_class"] = pd.Categorical(labels, categories=categories)
    obs["celltype_confidence"] = confidence

    prediction = ad.AnnData(
        X=reconstructed.X[selected_generated].copy(),
        obs=obs,
        var=reconstructed.var.copy(),
        obsm={
            spatial_key: np.asarray(reconstructed.obsm[spatial_key])[selected_generated],
            "target_spatial": np.asarray(target.obsm[spatial_key])[target_indices],
        },
    )
    summary = {
        "target_z_um": float(target_z),
        "z_tolerance_um": float(tolerance_um),
        "max_match_distance_um": float(max_distance_um),
        "generated_layer_cells": int(len(layer_indices)),
        "target_cells": int(target.n_obs),
        "matched_cells": int(len(selected_generated)),
        "coverage": float(len(selected_generated) / target.n_obs),
        "distance_median_um": float(np.median(distances)),
        "distance_p95_um": float(np.quantile(distances, 0.95)),
        "distance_within_25um": float(np.mean(distances <= 25.0)),
        "distance_within_50um": float(np.mean(distances <= 50.0)),
    }
    return prediction, summary


def _celltype_metrics(
    prediction: ad.AnnData, target: ad.AnnData, categories: list[str]
) -> dict:
    target_subset = target[prediction.obs_names].copy()
    y_true = target_subset.obs["cell_class"].astype(str).to_numpy()
    y_pred = prediction.obs["cell_class"].astype(str).to_numpy()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=categories, average="macro", zero_division=0)
        ),
        "n_cells": int(len(y_true)),
        "n_categories_evaluated": int(len(set(y_true))),
    }


def _build_model(
    *,
    output: Path,
    train_anchors: list[ad.AnnData],
    arm: str,
    feature_path: Path,
    nucleus_feature_path: Path,
    path_cache: Path,
    seed: int,
    epochs: int,
    pairs: int,
    lambda_c: float,
    nucleus_weight: float,
    force: bool,
) -> tuple[DeepSpatial, Path, dict]:
    model_dir = output / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted(model_dir.glob("*.ckpt"))
    if force or not checkpoints:
        pl.seed_everything(seed, workers=True)
        if arm == "he_celltype":
            histology = _histology_config(
                seed=seed,
                feature_path=feature_path,
                nucleus_feature_path=nucleus_feature_path,
                path_cache=path_cache,
                coordinate_frame=COORDINATE_FRAME,
                nucleus_weight=nucleus_weight,
            )
        elif arm == "celltype_only":
            histology = HistologyConfig()
        else:
            raise ValueError(f"Unknown arm: {arm}")

        model = DeepSpatial()
        model.setup_data(
            train_anchors,
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
            histology=histology,
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
        checkpoints = sorted(model_dir.glob("*.ckpt"))

    if not checkpoints:
        raise RuntimeError(f"No checkpoint produced under {model_dir}")
    checkpoint = checkpoints[-1]
    trained = DeepSpatial()
    trained.load_checkpoint(str(checkpoint), sampling_method="dopri5")
    spec = {
        "arm": arm,
        "use_celltype": True,
        "use_histology": arm == "he_celltype",
        "use_morphology_uot": arm == "he_celltype",
        "use_morphology_path": arm == "he_celltype",
        "use_nucleus_path": arm == "he_celltype",
        "lambda_c": float(lambda_c),
        "nucleus_weight": float(nucleus_weight),
        "training_sections": list(TRAINING_SECTIONS),
        "heldout_section": HELDOUT_SECTION,
        "checkpoint": str(checkpoint),
    }
    return trained, checkpoint, spec


def run_arm(args, arm: str) -> dict:
    output = Path(args.output).resolve() / arm
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "section31_3d_metrics.json"
    if result_path.exists() and not args.force:
        return json.loads(result_path.read_text())

    anchor_dir = Path(args.anchor_dir).resolve()
    train_anchors = _load_anchors(anchor_dir, TRAINING_SECTIONS)
    target = _load_anchors(anchor_dir, (HELDOUT_SECTION,))[0]
    categories = sorted(
        {str(value) for anchor in train_anchors for value in anchor.obs["cell_class"].astype(str)}
    )
    if set(target.obs["cell_class"].astype(str)) - set(categories):
        raise ValueError("Held-out section contains a cell type absent from training")

    trained, checkpoint, spec = _build_model(
        output=output,
        train_anchors=train_anchors,
        arm=arm,
        feature_path=Path(args.feature_store).resolve(),
        nucleus_feature_path=Path(args.nucleus_feature_store).resolve(),
        path_cache=Path(args.path_cache).resolve(),
        seed=args.seed,
        epochs=args.epochs,
        pairs=args.pairs,
        lambda_c=args.lambda_c,
        nucleus_weight=args.nucleus_weight,
        force=args.force,
    )

    # The target section is not passed to reconstruction.  All virtual cells
    # are generated from the eight observed training anchors only.
    pl.seed_everything(args.seed, workers=True)
    virtual_volume = trained.reconstruct_full_volume(
        train_anchors,
        thickness=args.thickness_um,
        steps=args.reconstruction_steps,
        chunk_size=args.reconstruction_chunk_size,
        device=args.device,
        cache_celltype_trajectory=True,
    )
    virtual_volume.obs["is_reconstructed"] = True
    virtual_volume_path = output / "virtual_volume_training_anchors_only.h5ad"
    virtual_volume.write_h5ad(virtual_volume_path, compression="gzip")

    target_z = float(target.obs["z_um"].iloc[0])
    prediction, spatial_summary = match_fixed_depth(
        virtual_volume,
        target,
        target_z=target_z,
        categories=categories,
        tolerance_um=args.depth_tolerance_um,
        max_distance_um=args.max_match_distance_um,
    )
    target_matched = target[prediction.obs_names].copy()
    expression = _expression_metrics(
        prediction.X,
        target_matched.X,
        prediction.obs["prediction_distance_um"].to_numpy(),
    )
    celltype = _celltype_metrics(prediction, target, categories)
    prediction_path = output / "section-031_fixed_depth_matched_prediction.h5ad"
    prediction.write_h5ad(prediction_path, compression="gzip")

    record = {
        **spec,
        "coordinate_frame": COORDINATE_FRAME,
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "sampled_uot_pairs_per_gap": int(args.pairs),
        "thickness_um": float(args.thickness_um),
        "reconstruction_steps": int(args.reconstruction_steps),
        "reconstruction_chunk_size": int(args.reconstruction_chunk_size),
        "fixed_depth": {
            "section": HELDOUT_SECTION,
            "z_um": target_z,
            "tolerance_um": float(args.depth_tolerance_um),
            "generation_used_target_coordinates": False,
        },
        "expression_metrics": expression,
        "celltype_metrics": celltype,
        "spatial_metrics": spatial_summary,
        "virtual_volume": str(virtual_volume_path),
        "matched_prediction": str(prediction_path),
        "celltype_cache": virtual_volume.uns.get(
            "deepspatial_celltype_trajectory_cache", {}
        ),
    }
    result_path.write_text(json.dumps(record, indent=2) + "\n")
    del prediction, target_matched, virtual_volume, trained, train_anchors, target
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("he_celltype", "celltype_only"), required=True)
    parser.add_argument("--anchor-dir", type=Path, default=DEFAULT_ANCHOR_DIR)
    parser.add_argument("--feature-store", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--nucleus-feature-store", type=Path, default=DEFAULT_NUCLEUS_FEATURES)
    parser.add_argument("--path-cache", type=Path, default=DEFAULT_PATH_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--pairs", type=int, default=20_000)
    parser.add_argument("--lambda-c", type=float, default=10.0)
    parser.add_argument("--nucleus-weight", type=float, default=1.0)
    parser.add_argument("--thickness-um", type=float, default=10.0)
    parser.add_argument("--reconstruction-steps", type=int, default=100)
    parser.add_argument("--reconstruction-chunk-size", type=int, default=512)
    parser.add_argument("--depth-tolerance-um", type=float, default=5.0)
    parser.add_argument("--max-match-distance-um", type=float, default=50.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for required in (
        args.anchor_dir,
        args.feature_store if args.arm == "he_celltype" else None,
        args.nucleus_feature_store if args.arm == "he_celltype" else None,
        args.path_cache if args.arm == "he_celltype" else None,
    ):
        if required is not None:
            path = required if required.is_absolute() else ROOT / required
            if not path.exists():
                raise FileNotFoundError(path)

    start = time.time()
    result = run_arm(args, args.arm)
    result["elapsed_min"] = (time.time() - start) / 60.0
    output = Path(args.output).resolve() / args.arm
    output.mkdir(parents=True, exist_ok=True)
    (output / "section31_3d_metrics.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
