"""Add cell-type annotations to an existing reconstruction with a trained c-branch.

This is a post-processing step.  It does not change the reconstructed gene
matrix or spatial coordinates.  The supplied checkpoint must have been trained
with ``use_celltype=True``; its cell-type flow is evaluated along the existing
reconstruction source trajectories and the resulting labels are written to a
new H5AD.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepspatial import DeepSpatial


DEFAULT_RECONSTRUCTION = (
    ROOT
    / "data/00029_g0/experiments/st_only_residual_nucleus_uot_path_inherit_labels_v1"
    / "adata_00029_g0_no_celltype_full_thickness10_celltype_inherited.h5ad"
)
DEFAULT_CHECKPOINT = (
    ROOT
    / "data/00029_g0/experiments/st_only_residual_nucleus_uot_path_celltype_v1"
    / "model/deepspatial-epoch=09-loss=0.0185.ckpt"
)
DEFAULT_ANCHOR_DIR = ROOT / "data/00029_g0/st_only_residual_v1/celltype_domain_niche_anchors_v1"
DEFAULT_OUTPUT = (
    ROOT
    / "data/00029_g0/experiments/st_only_residual_nucleus_uot_path_inherit_labels_v1"
    / "adata_00029_g0_no_celltype_full_thickness10_celltype_model_annotated.h5ad"
)
SECTIONS = (1, 11, 21, 31, 41, 51, 61, 71, 81, 91)


def attach_celltype_predictions(
    reconstruction: ad.AnnData,
    predicted_labels,
    confidence,
    *,
    label_key: str = "cell_class",
) -> ad.AnnData:
    """Write model labels while preserving the labels already in the input.

    Anchor rows retain their observed label.  Virtual rows receive the labels
    from the celltype-enabled checkpoint.  The previous column is copied to
    ``<label_key>_inherited`` so source-label inheritance remains auditable.
    """

    n_obs = reconstruction.n_obs
    labels = np.asarray(predicted_labels, dtype=object)
    scores = np.asarray(confidence, dtype=np.float32)
    if labels.shape != (n_obs,) or scores.shape != (n_obs,):
        raise ValueError(
            "prediction length must match reconstruction n_obs "
            f"({n_obs}); got labels={labels.shape}, confidence={scores.shape}"
        )
    if label_key not in reconstruction.obs:
        raise KeyError(f"Reconstruction must contain obs[{label_key!r}]")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("celltype confidence must be finite and lie in [0, 1]")

    result = reconstruction.copy()
    inherited = result.obs[label_key].astype(str).to_numpy(dtype=object)
    result.obs[f"{label_key}_inherited"] = pd.Categorical(inherited)

    if "is_xenium_anchor" in result.obs:
        anchor_mask = result.obs["is_xenium_anchor"].to_numpy(dtype=bool)
    else:
        anchor_mask = np.zeros(n_obs, dtype=bool)

    final_labels = labels.copy()
    final_labels[anchor_mask] = inherited[anchor_mask]
    if pd.isna(final_labels).any():
        raise ValueError("celltype predictions contain missing labels")
    categories = sorted({str(value) for value in final_labels})
    result.obs[label_key] = pd.Categorical(final_labels.astype(str), categories=categories)

    scores = scores.copy()
    scores[anchor_mask] = 1.0
    result.obs["celltype_confidence"] = scores
    source = np.full(n_obs, "celltype_true_checkpoint", dtype=object)
    source[anchor_mask] = "observed_anchor"
    result.obs["celltype_label_source"] = pd.Categorical(source)
    result.obs["celltype_model_used"] = ~anchor_mask
    return result


def _anchor_path(anchor_dir: Path, section: int) -> Path:
    return anchor_dir / f"section-{int(section):03d}.h5ad"


def _load_anchors(anchor_dir: Path, sections: tuple[int, ...]) -> list[ad.AnnData]:
    anchors = []
    genes = None
    for section in sections:
        path = _anchor_path(anchor_dir, section)
        if not path.is_file():
            raise FileNotFoundError(path)
        anchor = ad.read_h5ad(path)
        required_obs = {"cell_class", "z_um", "source_cell_id"}
        missing = sorted(required_obs - set(anchor.obs.columns))
        if missing:
            raise ValueError(f"{path} is missing obs columns: {missing}")
        if "spatial_st_corrected" not in anchor.obsm:
            raise ValueError(f"{path} lacks obsm['spatial_st_corrected']")
        if anchor.obs["cell_class"].isna().any():
            raise ValueError(f"{path} contains missing cell_class labels")
        if anchor.obs["z_um"].nunique() != 1:
            raise ValueError(f"{path} must contain one z_um value")
        if genes is None:
            genes = anchor.var_names.tolist()
        elif anchor.var_names.tolist() != genes:
            raise ValueError("Anchor gene order differs across sections")
        anchors.append(anchor)

    z_values = np.array([float(anchor.obs["z_um"].iloc[0]) for anchor in anchors])
    if not np.all(np.diff(z_values) > 0):
        raise ValueError(f"Anchors are not in increasing z order: {z_values.tolist()}")
    return anchors


def _section_uid(anchor: ad.AnnData, section: int) -> str:
    if "source_section_uid" in anchor.obs:
        return str(anchor.obs["source_section_uid"].iloc[0])
    return f"section-{int(section):03d}"


def _extract_source_batch(model: DeepSpatial, anchor: ad.AnnData, indices, device):
    x = torch.as_tensor(anchor.obsm["spatial_norm"][indices], dtype=torch.float32, device=device)
    matrix = anchor.X[indices]
    g_array = matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)
    g = torch.as_tensor(g_array, dtype=torch.float32, device=device)
    codes = pd.Categorical(
        anchor.obs[model.label_key].astype(str), categories=model.categories
    ).codes.astype(np.int64)
    selected_codes = codes[np.asarray(indices, dtype=np.int64)]
    if (selected_codes < 0).any():
        raise ValueError("Source anchor contains a missing or unknown cell type")
    c = torch.nn.functional.one_hot(
        torch.as_tensor(selected_codes, dtype=torch.long, device=device),
        num_classes=model.num_classes,
    ).float()
    z_norm = float(anchor.obs["z_norm"].iloc[0])
    return x, g, c, z_norm


def _predict_branch(
    model: DeepSpatial,
    lower: ad.AnnData,
    upper: ad.AnnData,
    source: ad.AnnData,
    query_rows: np.ndarray,
    query_t: np.ndarray,
    reconstruction: ad.AnnData,
    *,
    reverse: bool,
    steps: int,
    chunk_size: int,
    device: str,
    labels: np.ndarray,
    confidence: np.ndarray,
) -> None:
    """Evaluate the c-flow on source parents for one adjacent anchor gap."""

    model._prepare_reconstruction_inputs(lower, upper)
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
    model.module.to(dev)
    model.module.eval()

    source_ids = source.obs["source_cell_id"].astype(str).to_numpy()
    parent_ids = reconstruction.obs.iloc[query_rows]["source_cell_id"].astype(str).to_numpy()
    source_lookup = {cell_id: index for index, cell_id in enumerate(source_ids)}
    missing = sorted(set(parent_ids) - set(source_lookup))
    if missing:
        raise ValueError(
            f"{len(missing)} reconstruction source_cell_id values are absent from its anchor"
        )
    parent_indices = np.array([source_lookup[value] for value in parent_ids], dtype=np.int64)
    unique_parents = np.unique(parent_indices)
    local_positions = np.searchsorted(unique_parents, parent_indices)

    source_z = float(source.obs["z_norm"].iloc[0])
    target_z = float((upper if reverse else lower).obs["z_norm"].iloc[0])
    # For a forward branch the source is lower and target is upper.  For a
    # reverse branch the source is upper and target is lower.
    if reverse:
        source_z = float(upper.obs["z_norm"].iloc[0])
        target_z = float(lower.obs["z_norm"].iloc[0])
    delta_z = target_z - source_z

    for start in range(0, len(unique_parents), chunk_size):
        stop = min(start + chunk_size, len(unique_parents))
        source_indices = unique_parents[start:stop]
        x0, g0, c0, _ = _extract_source_batch(model, source, source_indices, dev)
        batch = {
            "x0": x0,
            "g0": g0,
            "c0": c0,
            "z0": torch.full((len(source_indices), 1), source_z, device=dev),
            "z1": torch.full((len(source_indices), 1), target_z, device=dev),
            "delta_z": torch.full((len(source_indices), 1), delta_z, device=dev),
        }
        if reverse:
            batch["reverse"] = True
        sampled = model.module.sample(batch, mode="ODE", steps=steps)
        local_query = np.flatnonzero(
            (local_positions >= start) & (local_positions < stop)
        )
        if len(local_query) == 0:
            continue
        time_index = np.clip(
            np.rint(query_t[local_query] * (steps - 1)).astype(np.int64),
            0,
            steps - 1,
        )
        parent_position = local_positions[local_query] - start
        c_cont = sampled["c_traj_cont"]
        c_state = c_cont[time_index, parent_position]
        c_prob = torch.softmax(c_state, dim=-1)
        c_index = torch.argmax(c_prob, dim=-1).detach().cpu().numpy()
        c_conf = torch.max(c_prob, dim=-1).values.detach().cpu().numpy()
        label_values = np.asarray(model.categories, dtype=object)[c_index]
        output_rows = query_rows[local_query]
        labels[output_rows] = label_values
        confidence[output_rows] = c_conf.astype(np.float32)


def predict_celltype_for_reconstruction(
    model: DeepSpatial,
    reconstruction: ad.AnnData,
    anchors: list[ad.AnnData],
    *,
    steps: int = 40,
    chunk_size: int = 512,
    device: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Predict labels for virtual cells using the existing celltype flow."""

    required = {"z_um", "source_cell_id", "source_section_uid"}
    missing = sorted(required - set(reconstruction.obs.columns))
    if missing:
        raise ValueError(f"Reconstruction is missing required columns: {missing}")
    if "spatial_st_corrected" not in reconstruction.obsm:
        raise ValueError("Reconstruction lacks obsm['spatial_st_corrected']")
    if not model.use_celltype:
        raise ValueError("The checkpoint must be trained with use_celltype=True")

    labels = np.full(reconstruction.n_obs, None, dtype=object)
    confidence = np.full(reconstruction.n_obs, np.nan, dtype=np.float32)
    anchor_mask = reconstruction.obs.get(
        "is_xenium_anchor", pd.Series(False, index=reconstruction.obs.index)
    ).to_numpy(dtype=bool)
    labels[anchor_mask] = reconstruction.obs.loc[anchor_mask, model.label_key].astype(str).to_numpy()
    confidence[anchor_mask] = 1.0

    anchor_by_uid = {
        _section_uid(anchor, section): (section, anchor)
        for section, anchor in zip(SECTIONS[: len(anchors)], anchors)
    }
    sorted_anchors = sorted(
        enumerate(anchors), key=lambda item: float(item[1].obs["z_um"].iloc[0])
    )
    z_values = np.array([float(anchor.obs["z_um"].iloc[0]) for _, anchor in sorted_anchors])
    source_uid = reconstruction.obs["source_section_uid"].astype(str).to_numpy()
    query_z = reconstruction.obs["z_um"].to_numpy(dtype=float)
    virtual_rows = np.flatnonzero(~anchor_mask)

    branch_groups: dict[tuple[str, bool], list[int]] = {}
    branch_t: dict[tuple[str, bool], list[float]] = {}
    for row in virtual_rows:
        uid = source_uid[row]
        if uid not in anchor_by_uid:
            raise ValueError(f"Unknown reconstruction source_section_uid: {uid}")
        source_position = next(
            i for i, (_, anchor) in enumerate(sorted_anchors) if _section_uid(anchor, i + 1) == uid
        )
        source_z = z_values[source_position]
        z = query_z[row]
        if z > source_z + 1e-7:
            if source_position >= len(sorted_anchors) - 1:
                raise ValueError(f"Virtual cell {row} lies above the last anchor")
            reverse = False
            target_z = z_values[source_position + 1]
        elif z < source_z - 1e-7:
            if source_position == 0:
                raise ValueError(f"Virtual cell {row} lies below the first anchor")
            reverse = True
            target_z = z_values[source_position - 1]
        else:
            # Exact-anchor virtual points are rare; use the source annotation
            # rather than inventing a nonzero flow interval.
            source_anchor = sorted_anchors[source_position][1]
            source_lookup = dict(
                zip(
                    source_anchor.obs["source_cell_id"].astype(str),
                    source_anchor.obs[model.label_key].astype(str),
                )
            )
            source_id = str(reconstruction.obs.iloc[row]["source_cell_id"])
            labels[row] = source_lookup.get(source_id, "unknown")
            confidence[row] = 1.0
            continue
        key = (uid, reverse)
        branch_groups.setdefault(key, []).append(row)
        branch_t.setdefault(key, []).append(
            abs(z - source_z) / max(abs(target_z - source_z), 1e-12)
        )

    for (uid, reverse), rows in branch_groups.items():
        source_section, source = anchor_by_uid[uid]
        source_position = next(
            i for i, (_, anchor) in enumerate(sorted_anchors) if _section_uid(anchor, i + 1) == uid
        )
        lower = sorted_anchors[source_position - 1][1] if reverse else source
        upper = source if reverse else sorted_anchors[source_position + 1][1]
        _predict_branch(
            model,
            lower,
            upper,
            source,
            np.asarray(rows, dtype=np.int64),
            np.asarray(branch_t[(uid, reverse)], dtype=np.float32),
            reconstruction,
            reverse=reverse,
            steps=steps,
            chunk_size=chunk_size,
            device=device,
            labels=labels,
            confidence=confidence,
        )

    if any(value is None for value in labels) or not np.isfinite(confidence).all():
        raise RuntimeError("Celltype prediction did not cover every reconstruction row")
    return labels.astype(str), confidence


def annotate_reconstruction(
    input_path: Path,
    output_path: Path,
    checkpoint: Path,
    anchor_dir: Path,
    *,
    steps: int = 40,
    chunk_size: int = 512,
    device: str = "auto",
    overwrite: bool = False,
) -> Path:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
    reconstruction = ad.read_h5ad(input_path)
    anchors = _load_anchors(anchor_dir, SECTIONS)
    model = DeepSpatial()
    model.load_checkpoint(str(checkpoint), sampling_method="dopri5")
    labels, confidence = predict_celltype_for_reconstruction(
        model,
        reconstruction,
        anchors,
        steps=steps,
        chunk_size=chunk_size,
        device=device,
    )
    result = attach_celltype_predictions(reconstruction, labels, confidence)
    result.uns.setdefault("deepspatial_celltype_annotation", {})
    result.uns["deepspatial_celltype_annotation"].update(
        {
            "method": "existing celltype-enabled DeepSpatial c-flow postprocessing",
            "checkpoint": str(checkpoint),
            "anchor_dir": str(anchor_dir),
            "steps": int(steps),
            "chunk_size": int(chunk_size),
            "device": device,
            "expression_and_spatial_unchanged": True,
            "inherited_labels_preserved_as": "cell_class_inherited",
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.write_h5ad(output_path, compression="gzip")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_RECONSTRUCTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--anchor-dir", type=Path, default=DEFAULT_ANCHOR_DIR)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    annotate_reconstruction(
        args.input,
        args.output,
        args.checkpoint,
        args.anchor_dir,
        steps=args.steps,
        chunk_size=args.chunk_size,
        device=args.device,
        overwrite=args.overwrite,
    )
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
