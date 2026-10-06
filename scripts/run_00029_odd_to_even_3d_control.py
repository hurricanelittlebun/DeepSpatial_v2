"""Alternating-anchor 3-D control experiments for 00029/g0.

The two supported directions train on one parity of the observed anchors and
evaluate only the strictly bracketed sections of the opposite parity.  The
out-of-range boundary section is deliberately excluded from the primary
evaluation rather than scored as an extrapolation.

Both arms use the cell-type branch.  ``he_celltype`` additionally enables the
registered UNI2 morphology and nucleus-guided H&E components; ``celltype_only``
disables all histology components.  Each arm generates its full virtual volume
from the five training anchors before any target section is used for scoring.
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

from scripts.run_00029_celltype_nucleus_uot_path import (
    _expression_metrics,
    _histology_config,
    _load_anchors,
)
from scripts.run_00029_section31_3d_control import (
    DEFAULT_ANCHOR_DIR,
    DEFAULT_FEATURES,
    DEFAULT_NUCLEUS_FEATURES,
    DEFAULT_PATH_CACHE,
    _unique_spatial_matches,
    _build_model,
    _celltype_metrics,
    match_fixed_depth,
)
from deepspatial import DeepSpatial
from deepspatial.histology import HistologyConfig


DEFAULT_OUTPUT = ROOT / "data/00029_g0/experiments/odd_to_even_3d_control_v1"
DEFAULT_NO_CELLTYPE_OUTPUT = ROOT / (
    "data/00029_g0/experiments/odd_to_even_3d_control_no_celltype_v1"
)


_DIRECTION_SPECS = {
    "odd_to_even": {
        "training_sections": (1, 21, 41, 61, 81),
        "target_sections": (11, 31, 51, 71),
        "boundary_not_scored": (91,),
    },
    "even_to_odd": {
        "training_sections": (11, 31, 51, 71, 91),
        "target_sections": (21, 41, 61, 81),
        "boundary_not_scored": (1,),
    },
}


def direction_spec(direction: str) -> dict:
    """Return an immutable section split for an alternating-anchor direction."""

    try:
        spec = _DIRECTION_SPECS[direction]
    except KeyError as exc:
        raise ValueError(
            f"Unknown direction {direction!r}; choose from {sorted(_DIRECTION_SPECS)}"
        ) from exc
    return {key: tuple(value) for key, value in spec.items()}


def inherit_source_annotations(
    reconstruction: ad.AnnData,
    source_anchors: list[ad.AnnData],
    *,
    source_id_key: str = "source_cell_id",
    label_columns: tuple[str, ...] = (
        "cell_class",
        "pred_domain",
        "pred_niche",
        "pred_domain_conf",
        "pred_niche_conf",
    ),
) -> ad.AnnData:
    """Restore source-anchor annotations after a no-celltype reconstruction.

    ``use_celltype=False`` intentionally removes cell-type information from the
    model state and UOT cost.  The reconstruction still copies source-anchor
    metadata, but the core assigns a placeholder ``cell_class`` because the
    c-state has one dummy class.  This helper replaces that placeholder with
    the annotation belonging to each generated cell's ``source_cell_id``.
    It is metadata inheritance, not a cell-type prediction.
    """

    if source_id_key not in reconstruction.obs:
        raise KeyError(f"Reconstruction lacks obs[{source_id_key!r}]")

    source_frames = []
    for anchor in source_anchors:
        if source_id_key not in anchor.obs:
            raise KeyError(f"Source anchor lacks obs[{source_id_key!r}]")
        source_frames.append(anchor.obs.copy())
    source_obs = pd.concat(source_frames, axis=0)
    source_ids = source_obs[source_id_key].astype(str)
    if source_ids.duplicated().any():
        raise ValueError("source_cell_id is not unique across source anchors")
    source_obs.index = source_ids.to_numpy()

    result = reconstruction.copy()
    query_ids = result.obs[source_id_key].astype(str)
    missing = sorted(set(query_ids) - set(source_obs.index))
    if missing:
        raise ValueError(
            f"{len(missing)} generated cells have no source annotation; "
            f"first IDs: {missing[:5]}"
        )

    copied = []
    for column in label_columns:
        if column not in source_obs.columns:
            continue
        values = source_obs[column].reindex(query_ids.to_numpy())
        if values.isna().any():
            raise ValueError(f"Missing inherited annotation values in {column!r}")
        if isinstance(source_obs[column].dtype, pd.CategoricalDtype):
            result.obs[column] = pd.Categorical(
                values.to_numpy(), categories=source_obs[column].cat.categories
            )
        else:
            result.obs[column] = values.to_numpy()
        copied.append(column)

    if "cell_class" not in copied:
        raise KeyError("Source anchors must contain obs['cell_class']")

    result.obs["cell_class_inherited"] = pd.Categorical(
        result.obs["cell_class"].astype(str),
        categories=source_obs["cell_class"].astype("category").cat.categories,
    )
    result.obs["celltype_label_source"] = pd.Categorical(
        np.full(result.n_obs, "inherited_from_source_anchor", dtype=object)
    )
    result.obs["celltype_model_used"] = False
    result.obs["celltype_label_source_cell_id"] = query_ids.to_numpy()
    result.uns.setdefault("deepspatial_celltype_annotation", {})
    result.uns["deepspatial_celltype_annotation"].update(
        {
            "method": "source-anchor metadata inheritance after use_celltype=False reconstruction",
            "model_celltype_branch_used": False,
            "copied_columns": copied,
            "source_id_key": source_id_key,
            "label_semantics": "not a cell-type prediction; inherited from the generated cell source anchor",
        }
    )
    return result


def match_fixed_depth_inherited(
    reconstructed: ad.AnnData,
    target: ad.AnnData,
    *,
    target_z: float,
    tolerance_um: float = 5.0,
    max_distance_um: float = 50.0,
    spatial_key: str = "spatial_st_corrected",
    z_key: str = "z_um",
) -> tuple[ad.AnnData, dict]:
    """Match a generated layer without requiring a cached c-flow state."""

    z_values = np.asarray(reconstructed.obs[z_key], dtype=np.float64)
    layer_indices = np.flatnonzero(
        np.abs(z_values - float(target_z)) <= float(tolerance_um)
    )
    if not len(layer_indices):
        raise ValueError("No generated cells fall inside the requested Z layer")

    target_indices, generated_indices, distances = _unique_spatial_matches(
        np.asarray(reconstructed.obsm[spatial_key])[layer_indices],
        np.asarray(target.obsm[spatial_key]),
        max_distance_um=max_distance_um,
    )
    if not len(generated_indices):
        raise ValueError("No generated cells could be matched to the target section")

    selected_generated = layer_indices[generated_indices]
    target_names = target.obs_names.to_numpy()[target_indices]
    obs = reconstructed.obs.iloc[selected_generated].copy()
    obs.index = target_names
    obs["target_obs_name"] = target_names
    obs["prediction_distance_um"] = distances
    obs["fixed_depth_target_z_um"] = float(target_z)
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


def _build_no_celltype_model(
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
    nucleus_weight: float,
    force: bool,
) -> tuple[DeepSpatial, Path, dict]:
    """Train/load a morphology model with the cell-type branch disabled."""

    model_dir = output / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted(p for p in model_dir.glob("*.ckpt") if p.name != "last.ckpt")
    use_histology = arm == "he_inherit_labels"
    if arm not in {"he_inherit_labels", "nohe_inherit_labels"}:
        raise ValueError(f"Unknown no-celltype arm: {arm}")

    if force or not checkpoints:
        pl.seed_everything(seed, workers=True)
        histology = (
            _histology_config(
                seed=seed,
                feature_path=feature_path,
                nucleus_feature_path=nucleus_feature_path,
                path_cache=path_cache,
                coordinate_frame="00029__g0_registered",
                nucleus_weight=nucleus_weight,
            )
            if use_histology
            else HistologyConfig()
        )
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
            use_celltype=False,
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
            lambda_c=0.0,
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
        del model, histology
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        checkpoints = sorted(p for p in model_dir.glob("*.ckpt") if p.name != "last.ckpt")

    if not checkpoints:
        raise RuntimeError(f"No checkpoint produced under {model_dir}")
    checkpoint = checkpoints[-1]
    trained = DeepSpatial()
    trained.load_checkpoint(str(checkpoint), sampling_method="dopri5")
    spec = {
        "arm": arm,
        "use_celltype": False,
        "celltype_annotations": "inherited_from_source_anchor",
        "use_histology": use_histology,
        "use_morphology_uot": use_histology,
        "use_morphology_path": use_histology,
        "use_nucleus_path": use_histology,
        "lambda_c": 0.0,
        "nucleus_weight": float(nucleus_weight),
        "training_sections": list(direction_spec("odd_to_even")["training_sections"]),
        "checkpoint": str(checkpoint),
    }
    return trained, checkpoint, spec


def run_no_celltype_arm(args, arm: str) -> dict:
    """Run one odd-to-even arm with post-hoc source annotation inheritance."""

    split = direction_spec(args.direction)
    training_sections = split["training_sections"]
    target_sections = split["target_sections"]
    output = Path(args.output).resolve() / arm
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / f"{args.direction}_metrics.json"
    if result_path.exists() and not args.force:
        return json.loads(result_path.read_text())

    anchor_dir = Path(args.anchor_dir).resolve()
    train_anchors = _load_anchors(anchor_dir, training_sections)
    target_anchors = {
        section: _load_anchors(anchor_dir, (section,))[0]
        for section in target_sections
    }
    categories = sorted(
        {
            str(value)
            for anchor in train_anchors
            for value in anchor.obs["cell_class"].astype(str)
        }
    )

    feature_path = Path(args.feature_store).resolve()
    nucleus_feature_path = Path(args.nucleus_feature_store).resolve()
    path_cache = Path(args.path_cache).resolve()
    if arm == "he_inherit_labels":
        for required in (feature_path, nucleus_feature_path, path_cache):
            if not required.exists():
                raise FileNotFoundError(required)

    trained, checkpoint, model_spec = _build_no_celltype_model(
        output=output,
        train_anchors=train_anchors,
        arm=arm,
        feature_path=feature_path,
        nucleus_feature_path=nucleus_feature_path,
        path_cache=path_cache,
        seed=args.seed,
        epochs=args.epochs,
        pairs=args.pairs,
        nucleus_weight=args.nucleus_weight,
        force=args.force,
    )

    virtual_volume_path = output / (
        f"virtual_volume_{args.direction.split('_to_')[0]}_anchors_only.h5ad"
    )
    if virtual_volume_path.exists() and not args.force:
        virtual_volume = ad.read_h5ad(virtual_volume_path)
    else:
        pl.seed_everything(args.seed, workers=True)
        virtual_volume = trained.reconstruct_full_volume(
            train_anchors,
            thickness=args.thickness_um,
            steps=args.reconstruction_steps,
            chunk_size=args.reconstruction_chunk_size,
            device=args.device,
            cache_celltype_trajectory=False,
        )
        virtual_volume.obs["is_reconstructed"] = True
        virtual_volume = inherit_source_annotations(virtual_volume, train_anchors)
        virtual_volume.write_h5ad(virtual_volume_path, compression="gzip")

    records = []
    target_paths = {}
    for section in target_sections:
        target = target_anchors[section]
        target_z = float(target.obs["z_um"].iloc[0])
        prediction, spatial = match_fixed_depth_inherited(
            virtual_volume,
            target,
            target_z=target_z,
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
        prediction_path = output / f"section-{section:03d}_fixed_depth_prediction.h5ad"
        prediction.write_h5ad(prediction_path, compression="gzip")
        target_paths[str(section)] = str(prediction_path)
        records.append(
            {
                "target_section": section,
                "target_z_um": target_z,
                "expression": expression,
                "celltype": celltype,
                "spatial": spatial,
            }
        )
        del prediction, target_matched

    result = {
        "experiment": f"00029_g0_{args.direction}_3d_control_no_celltype_v1",
        "arm": arm,
        "direction": args.direction,
        "training_sections": list(training_sections),
        "scored_target_sections": list(target_sections),
        "boundary_not_scored": list(split["boundary_not_scored"]),
        "checkpoint": str(checkpoint),
        "model_spec": {
            **model_spec,
            "training_sections": list(training_sections),
            "target_sections": list(target_sections),
        },
        "protocol": {
            "evaluation": "full_3d_reconstruction_then_fixed_depth_matching",
            "target_coordinates_used_for_generation": False,
            "target_expression_used_for_generation": False,
            "annotation_method": "source-anchor inheritance after use_celltype=False",
            "thickness_um": float(args.thickness_um),
            "reconstruction_steps": int(args.reconstruction_steps),
            "reconstruction_chunk_size": int(args.reconstruction_chunk_size),
            "depth_tolerance_um": float(args.depth_tolerance_um),
            "max_match_distance_um": float(args.max_match_distance_um),
        },
        "per_section": records,
        "expression": _mean_metric([record["expression"] for record in records]),
        "celltype": _mean_metric([record["celltype"] for record in records]),
        "spatial": _mean_metric([record["spatial"] for record in records]),
        "virtual_volume": str(virtual_volume_path),
        "fixed_depth_predictions": target_paths,
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    del virtual_volume, trained, train_anchors, target_anchors
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _mean_metric(values: list[dict]) -> dict:
    result = {"n_sections": len(values)}
    numeric_keys = set(values[0])
    for metric_key in sorted(numeric_keys):
        if metric_key in {"n_cells", "n_categories_evaluated"}:
            continue
        result[f"mean_{metric_key}"] = float(
            np.mean([float(value[metric_key]) for value in values])
        )
    result["total_n_cells"] = int(
        sum(int(value.get("n_cells", value.get("matched_cells", 0))) for value in values)
    )
    return result


def run_arm(args, arm: str) -> dict:
    split = direction_spec(args.direction)
    training_sections = split["training_sections"]
    target_sections = split["target_sections"]
    boundary_not_scored = split["boundary_not_scored"]
    output = Path(args.output).resolve() / arm
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / f"{args.direction}_metrics.json"
    if result_path.exists() and not args.force:
        return json.loads(result_path.read_text())

    anchor_dir = Path(args.anchor_dir).resolve()
    train_anchors = _load_anchors(anchor_dir, training_sections)
    target_anchors = {
        section: _load_anchors(anchor_dir, (section,))[0]
        for section in target_sections
    }
    categories = sorted(
        {
            str(value)
            for anchor in train_anchors
            for value in anchor.obs["cell_class"].astype(str)
        }
    )
    for section, target in target_anchors.items():
        missing = set(target.obs["cell_class"].astype(str)) - set(categories)
        if missing:
            raise ValueError(f"section {section} has unseen cell types: {sorted(missing)}")

    trained, checkpoint, model_spec = _build_model(
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

    virtual_volume_path = output / (
        f"virtual_volume_{args.direction.split('_to_')[0]}_anchors_only.h5ad"
    )
    if virtual_volume_path.exists() and not args.force:
        virtual_volume = ad.read_h5ad(virtual_volume_path)
    else:
        # Reconstruct only from the selected parity anchors.  The target anchors are not
        # passed to this call and cannot affect virtual coordinates or expression.
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
        virtual_volume.write_h5ad(virtual_volume_path, compression="gzip")

    records = []
    target_paths = {}
    for section in target_sections:
        target = target_anchors[section]
        target_z = float(target.obs["z_um"].iloc[0])
        prediction, spatial = match_fixed_depth(
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
        prediction_path = output / f"section-{section:03d}_fixed_depth_prediction.h5ad"
        prediction.write_h5ad(prediction_path, compression="gzip")
        target_paths[str(section)] = str(prediction_path)
        records.append(
            {
                "target_section": section,
                "target_z_um": target_z,
                "expression": expression,
                "celltype": celltype,
                "spatial": spatial,
            }
        )
        del prediction, target_matched

    result = {
        "experiment": f"00029_g0_{args.direction}_3d_control_v1",
        "arm": arm,
        "direction": args.direction,
        "training_sections": list(training_sections),
        "scored_target_sections": list(target_sections),
        "boundary_not_scored": list(boundary_not_scored),
        "checkpoint": str(checkpoint),
        "model_spec": {
            **model_spec,
            "training_sections": list(training_sections),
            "target_sections": list(target_sections),
        },
        "protocol": {
            "evaluation": "full_3d_reconstruction_then_fixed_depth_matching",
            "target_coordinates_used_for_generation": False,
            "target_expression_used_for_generation": False,
            "direction": args.direction,
            "thickness_um": float(args.thickness_um),
            "reconstruction_steps": int(args.reconstruction_steps),
            "reconstruction_chunk_size": int(args.reconstruction_chunk_size),
            "depth_tolerance_um": float(args.depth_tolerance_um),
            "max_match_distance_um": float(args.max_match_distance_um),
            "celltype_trajectory_cached_during_reconstruction": True,
        },
        "per_section": records,
        "expression": _mean_metric([record["expression"] for record in records]),
        "celltype": _mean_metric([record["celltype"] for record in records]),
        "spatial": _mean_metric([record["spatial"] for record in records]),
        "virtual_volume": str(virtual_volume_path),
        "fixed_depth_predictions": target_paths,
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    del virtual_volume, trained, train_anchors, target_anchors
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--direction",
        choices=tuple(_DIRECTION_SPECS),
        default="odd_to_even",
        help="Which parity supplies training anchors and which supplies held-out targets.",
    )
    parser.add_argument(
        "--arm",
        choices=(
            "he_celltype",
            "celltype_only",
            "he_inherit_labels",
            "nohe_inherit_labels",
        ),
        required=True,
    )
    parser.add_argument("--anchor-dir", type=Path, default=DEFAULT_ANCHOR_DIR)
    parser.add_argument("--feature-store", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--nucleus-feature-store", type=Path, default=DEFAULT_NUCLEUS_FEATURES)
    parser.add_argument("--path-cache", type=Path, default=DEFAULT_PATH_CACHE)
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT
    )
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

    required = [args.anchor_dir]
    if args.arm in {"he_celltype", "he_inherit_labels"}:
        required.extend([args.feature_store, args.nucleus_feature_store, args.path_cache])
    for path in required:
        resolved = path if path.is_absolute() else ROOT / path
        if not resolved.exists():
            raise FileNotFoundError(resolved)

    start = time.time()
    if args.arm in {"he_inherit_labels", "nohe_inherit_labels"}:
        result = run_no_celltype_arm(args, args.arm)
    else:
        result = run_arm(args, args.arm)
    result["elapsed_min"] = (time.time() - start) / 60.0
    output = Path(args.output).resolve() / args.arm
    (output / f"{args.direction}_metrics.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
