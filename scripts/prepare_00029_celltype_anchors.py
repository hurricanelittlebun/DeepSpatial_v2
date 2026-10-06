"""Materialize cell-type annotations onto the canonical 00029 anchors.

The annotation H5AD and the canonical DeepSpatial anchors may have different
row order or AnnData indices.  This module therefore joins only on the stable
Xenium ``source_cell_id`` and validates genes and registered coordinates before
writing a new labeled anchor package.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


SECTIONS = (1, 11, 21, 31, 41, 51, 61, 71, 81, 91)
DEFAULT_ANNOTATED = Path("data/00029_g0/adata_00029_niche_domain.h5ad")
DEFAULT_ANCHOR_DIR = Path("data/00029_g0/st_only_residual_v1/anchors")
DEFAULT_OUTPUT_DIR = Path("data/00029_g0/st_only_residual_v1/celltype_anchors_v1")
DEFAULT_PROPAGATED_LABEL_COLUMNS = ("pred_domain", "pred_niche")
DEFAULT_PROPAGATED_SCORE_COLUMNS = ("pred_domain_conf", "pred_niche_conf")


def _string_ids(adata: ad.AnnData, key: str) -> pd.Index:
    if key not in adata.obs:
        raise ValueError(f"Missing required obs column {key!r}")
    values = pd.Index(adata.obs[key].astype(str))
    if values.isna().any() or (values == "nan").any():
        raise ValueError(f"Missing values in {key}")
    if values.has_duplicates:
        raise ValueError(f"duplicate {key} values are ambiguous")
    return values


def _label_categories(annotation: ad.AnnData, label_column: str) -> list[str]:
    values = annotation.obs[label_column]
    if values.isna().any():
        raise ValueError("missing annotation labels")
    if isinstance(values.dtype, pd.CategoricalDtype):
        categories = [str(x) for x in values.cat.categories]
    else:
        categories = sorted(str(x) for x in values.astype(str).unique())
    if not categories:
        raise ValueError("annotation label vocabulary is empty")
    return categories


def _aligned_annotation_values(
    annotation: ad.AnnData,
    annotation_ids: pd.Index,
    anchor_ids: pd.Index,
    column: str,
) -> np.ndarray | pd.Categorical:
    """Return one annotation column in the exact order of the anchor rows."""

    if column not in annotation.obs:
        raise ValueError(f"Missing annotation column {column!r}")
    values = annotation.obs[column]
    if values.isna().any():
        raise ValueError(f"missing annotation values in {column!r}")

    by_id = pd.Series(values.to_numpy(), index=annotation_ids)
    aligned = by_id.loc[anchor_ids].to_numpy()
    if isinstance(values.dtype, pd.CategoricalDtype):
        return pd.Categorical(
            aligned,
            categories=values.cat.categories,
            ordered=values.cat.ordered,
        )
    return aligned


def label_anchor_from_annotations(
    annotation: ad.AnnData,
    anchor: ad.AnnData,
    *,
    source_id_key: str = "source_cell_id",
    label_column: str = "assigned_celltype",
    output_label_key: str = "cell_class",
    spatial_key: str = "spatial_st_corrected",
    coordinate_atol: float = 1e-6,
    propagated_label_columns: tuple[str, ...] = DEFAULT_PROPAGATED_LABEL_COLUMNS,
    propagated_score_columns: tuple[str, ...] = DEFAULT_PROPAGATED_SCORE_COLUMNS,
) -> ad.AnnData:
    """Return a labeled copy of ``anchor`` after strict compatibility checks.

    In addition to the model's ``cell_class`` label, preserve categorical
    section metadata (currently ``pred_domain`` and ``pred_niche``) on every
    anchor row.  These columns are later inherited by reconstruction's
    forward/backward branches; they are not model predictions.
    """

    if label_column not in annotation.obs:
        raise ValueError(f"Missing annotation column {label_column!r}")
    annotation_ids = _string_ids(annotation, source_id_key)
    anchor_ids = _string_ids(anchor, source_id_key)
    categories = _label_categories(annotation, label_column)

    annotation_set = set(annotation_ids)
    anchor_set = set(anchor_ids)
    missing = sorted(anchor_set - annotation_set)
    if missing:
        raise ValueError(
            f"missing annotation labels for {len(missing)} anchor source_cell_id values"
        )

    if annotation.var_names.tolist() != anchor.var_names.tolist():
        raise ValueError("gene order differs between annotation and anchor")

    if spatial_key not in annotation.obsm or spatial_key not in anchor.obsm:
        raise ValueError(f"Both objects must contain obsm[{spatial_key!r}]")
    annotation_positions = annotation_ids.get_indexer(anchor_ids)
    annotation_xy = np.asarray(annotation.obsm[spatial_key])[annotation_positions]
    anchor_xy = np.asarray(anchor.obsm[spatial_key])
    if annotation_xy.shape != anchor_xy.shape or not np.allclose(
        annotation_xy, anchor_xy, rtol=0.0, atol=coordinate_atol
    ):
        raise ValueError(f"{spatial_key} coordinates differ between annotation and anchor")

    labels_by_id = pd.Series(
        annotation.obs[label_column].astype(str).to_numpy(),
        index=annotation_ids,
    )
    labels = labels_by_id.loc[anchor_ids].to_numpy()
    if pd.isna(labels).any():
        raise ValueError("missing annotation labels")

    result = anchor.copy()
    result.obs[output_label_key] = pd.Categorical(labels, categories=categories)
    for column in propagated_label_columns:
        result.obs[column] = _aligned_annotation_values(
            annotation, annotation_ids, anchor_ids, column
        )
    copied_score_columns = []
    for column in propagated_score_columns:
        if column in annotation.obs:
            result.obs[column] = _aligned_annotation_values(
                annotation, annotation_ids, anchor_ids, column
            )
            copied_score_columns.append(column)
    result.uns.setdefault("deepspatial_celltype_annotation", {})
    result.uns["deepspatial_celltype_annotation"].update(
        {
            "source_id_key": source_id_key,
            "source_label_column": label_column,
            "output_label_key": output_label_key,
            "categories": categories,
            "join": "source_cell_id",
            "validated_spatial_key": spatial_key,
            "propagated_label_columns": list(propagated_label_columns),
            "propagated_score_columns": copied_score_columns,
            "label_semantics": (
                "cell_class is used by the model; propagated labels are "
                "inherited from the forward/backward source anchor"
            ),
        }
    )
    return result


def materialize_labeled_anchors(
    annotated_path: Path | str,
    anchor_dir: Path | str,
    output_dir: Path | str,
    *,
    sections: tuple[int, ...] = SECTIONS,
    label_column: str = "assigned_celltype",
) -> dict:
    """Create one labeled H5AD per canonical anchor section."""

    annotated_path = Path(annotated_path)
    anchor_dir = Path(anchor_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    annotation = ad.read_h5ad(annotated_path)
    if label_column not in annotation.obs:
        raise ValueError(f"Missing annotation column {label_column!r}")

    records = []
    categories = _label_categories(annotation, label_column)
    for section in sections:
        anchor_path = anchor_dir / f"section-{section:03d}.h5ad"
        if not anchor_path.is_file():
            raise FileNotFoundError(anchor_path)
        anchor = ad.read_h5ad(anchor_path)
        labeled = label_anchor_from_annotations(
            annotation, anchor, label_column=label_column
        )
        output_path = output_dir / f"section-{section:03d}.h5ad"
        labeled.write_h5ad(output_path)
        records.append(
            {
                "section_id": int(section),
                "n_cells": int(labeled.n_obs),
                "output": str(output_path),
                "label_column": "cell_class",
                "n_categories": len(categories),
                "propagated_label_columns": list(DEFAULT_PROPAGATED_LABEL_COLUMNS),
                "propagated_score_columns": [
                    column
                    for column in DEFAULT_PROPAGATED_SCORE_COLUMNS
                    if column in annotation.obs
                ],
            }
        )

    manifest = {
        "format": "deepspatial-00029-celltype-anchors-v1",
        "annotation_source": str(annotated_path.resolve()),
        "canonical_anchor_dir": str(anchor_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "label_column": label_column,
        "output_label_key": "cell_class",
        "join_key": "source_cell_id",
        "propagated_label_columns": list(DEFAULT_PROPAGATED_LABEL_COLUMNS),
        "propagated_score_columns": [
            column
            for column in DEFAULT_PROPAGATED_SCORE_COLUMNS
            if column in annotation.obs
        ],
        "categories": categories,
        "sections": records,
        "n_cells_total": int(sum(x["n_cells"] for x in records)),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotated", type=Path, default=DEFAULT_ANNOTATED)
    parser.add_argument("--anchor-dir", type=Path, default=DEFAULT_ANCHOR_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label-column", default="assigned_celltype")
    args = parser.parse_args()
    manifest = materialize_labeled_anchors(
        args.annotated,
        args.anchor_dir,
        args.output_dir,
        label_column=args.label_column,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
