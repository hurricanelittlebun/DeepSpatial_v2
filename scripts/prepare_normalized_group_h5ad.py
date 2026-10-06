"""Split the validated cell-level Xenium table into one H5AD per tissue/g.

The input H5AD is already a joined, registered anchor table.  This script does
not infer tissue membership from filenames and does not read the transcript
table.  It uses the validated ``joint_series_id`` column as the only grouping
key, then materializes a small, reproducible coordinate contract for each
series.

The generated H5AD files contain ST anchor cells only.  H&E-only sections are
represented by the accompanying section metadata in the source project and
are deliberately not invented as observations here.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import anndata as ad
import numpy as np
import pandas as pd


DEFAULT_THICKNESS_UM = 5.0
CANONICAL_SPATIAL_KEY = "spatial_registered"
LABEL_COLUMNS = (
    "assigned_celltype",
    "major_cell_type",
    "pred_domain",
    "pred_domain_conf",
    "pred_niche",
    "pred_niche_conf",
)


def _as_string(values: pd.Series) -> pd.Series:
    """Return a non-null string series without turning missing values into IDs."""

    result = values.astype("string")
    if result.isna().any():
        raise ValueError(f"Column {values.name!r} contains missing identifiers")
    return result


def _require_source(source: ad.AnnData) -> None:
    required_obs = {
        "joint_series_id",
        "joint_cell_id",
        "source_cell_id",
        "sample_id",
        "tissue_id",
        "section_id",
    }
    missing_obs = sorted(required_obs - set(source.obs.columns))
    if missing_obs:
        raise KeyError("Source H5AD is missing obs columns: " + ", ".join(missing_obs))
    if CANONICAL_SPATIAL_KEY not in source.obsm:
        raise KeyError(f"Source H5AD is missing obsm[{CANONICAL_SPATIAL_KEY!r}]")


def _clean_categories(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in frame.columns:
        if isinstance(frame[column].dtype, pd.CategoricalDtype):
            frame[column] = frame[column].cat.remove_unused_categories()
    return frame


def _join_annotation_columns(
    group: ad.AnnData,
    annotation: ad.AnnData,
    *,
    source_id_key: str = "source_cell_id",
) -> dict[str, object]:
    """Copy available 00029 labels by stable source_cell_id.

    Labels are optional.  This function never joins by row order and never
    changes the expression matrix or spatial coordinates.
    """

    if source_id_key not in annotation.obs:
        raise KeyError(f"Annotation H5AD is missing obs[{source_id_key!r}]")
    annotation_ids = _as_string(annotation.obs[source_id_key])
    if annotation_ids.duplicated().any():
        raise ValueError("Annotation source_cell_id values are not unique")

    group_ids = _as_string(group.obs[source_id_key])
    lookup = annotation.obs.copy()
    lookup.index = annotation_ids.to_numpy()
    missing = group_ids[~group_ids.isin(lookup.index)]
    if not missing.empty:
        raise ValueError(
            "Annotation is missing "
            f"{missing.nunique()} source_cell_id values for {group.obs['series_id'].iloc[0]}"
        )

    copied: list[str] = []
    for column in LABEL_COLUMNS:
        if column not in lookup.columns:
            continue
        values = lookup.loc[group_ids.to_numpy(), column].to_numpy()
        # Preserve categorical vocabularies when available.  This is useful
        # for downstream DeepSpatial cell-type encoding.
        if isinstance(lookup[column].dtype, pd.CategoricalDtype):
            group.obs[column] = pd.Categorical(
                values,
                categories=lookup[column].cat.categories,
                ordered=lookup[column].cat.ordered,
            )
        else:
            group.obs[column] = values
        copied.append(column)

    if "assigned_celltype" in copied and "cell_class" not in group.obs:
        assigned = group.obs["assigned_celltype"]
        if isinstance(assigned.dtype, pd.CategoricalDtype):
            group.obs["cell_class"] = pd.Categorical(
                assigned,
                categories=assigned.cat.categories,
                ordered=assigned.cat.ordered,
            )
        else:
            group.obs["cell_class"] = assigned.to_numpy()
        copied.append("cell_class")
    return {
        "status": "available",
        "source": "data/00029_g0/adata_00029_niche_domain.h5ad",
        "join_key": source_id_key,
        "columns": copied,
    }


def _ensure_cell_class_alias(group: ad.AnnData) -> list[str]:
    """Expose the standard DeepSpatial label key without deleting source labels."""

    if "cell_class" in group.obs:
        return ["cell_class"]
    if "assigned_celltype" not in group.obs:
        return []
    assigned = group.obs["assigned_celltype"]
    if isinstance(assigned.dtype, pd.CategoricalDtype):
        group.obs["cell_class"] = pd.Categorical(
            assigned,
            categories=assigned.cat.categories,
            ordered=assigned.cat.ordered,
        )
    else:
        group.obs["cell_class"] = assigned.to_numpy()
    return ["cell_class"]


def materialize_group(
    source: ad.AnnData,
    series_id: str,
    *,
    section_thickness_um: float = DEFAULT_THICKNESS_UM,
    annotation: ad.AnnData | None = None,
) -> ad.AnnData:
    """Materialize one canonical tissue/g H5AD object from a backed source."""

    _require_source(source)
    source_series = _as_string(source.obs["joint_series_id"])
    mask = source_series.eq(str(series_id)).to_numpy()
    if not mask.any():
        raise KeyError(f"No observations found for joint_series_id={series_id!r}")

    group = source[mask].to_memory()
    group.obs = group.obs.copy()
    group.obs["series_id"] = str(series_id)
    group.obs["g_id"] = str(series_id).split("__", 1)[1]
    group.obs["core_id"] = _as_string(group.obs["tissue_id"]).to_numpy()

    section_ids = pd.to_numeric(group.obs["section_id"], errors="raise").astype(int)
    if section_ids.empty:
        raise ValueError(f"Series {series_id} has no section IDs")
    section_values = sorted(section_ids.unique().tolist())
    section_order = {section: order for order, section in enumerate(section_values)}
    group.obs["series_section_order"] = section_ids.map(section_order).astype("int32").to_numpy()
    group.obs["z_um"] = (
        (section_ids - section_values[0]).astype(float) * float(section_thickness_um)
    ).to_numpy(dtype=np.float32)
    group.obs["z_global"] = group.obs["z_um"].to_numpy(dtype=np.float32)

    registered = np.asarray(group.obsm[CANONICAL_SPATIAL_KEY], dtype=np.float64)
    if registered.ndim != 2 or registered.shape != (group.n_obs, 2):
        raise ValueError(
            f"{CANONICAL_SPATIAL_KEY} must have shape (n_obs, 2), got {registered.shape}"
        )
    if not np.isfinite(registered).all():
        raise ValueError(f"Non-finite {CANONICAL_SPATIAL_KEY} coordinates in {series_id}")

    # AnnData's conventional spatial key is made explicit and always means
    # registered physical XY in this output package.  The source-specific
    # spatial_registered/spatial_raw/spatial_he_local keys remain available.
    group.obsm["spatial"] = registered.copy()
    group.obs["x_global"] = registered[:, 0]
    group.obs["y_global"] = registered[:, 1]
    group.obs["is_xenium_anchor"] = True
    group.obs["is_reconstructed"] = False

    joint_ids = _as_string(group.obs["joint_cell_id"])
    if joint_ids.duplicated().any():
        raise ValueError(f"Duplicate joint_cell_id values in {series_id}")
    group.obs_names = pd.Index(joint_ids.to_numpy(), name="joint_cell_id")
    group.obs = _clean_categories(group.obs)

    label_info: dict[str, object] = {
        "status": "not_available",
        "join_key": "source_cell_id",
        "columns": [],
    }
    aliases = _ensure_cell_class_alias(group)
    source_label_columns = [
        column
        for column in (*LABEL_COLUMNS, *aliases)
        if column in group.obs
    ]
    if source_label_columns:
        label_info = {
            "status": "available",
            "source": "source_h5ad",
            "join_key": "source_cell_id",
            "columns": source_label_columns,
        }
    if annotation is not None and str(series_id) == "00029__g0":
        label_info = _join_annotation_columns(group, annotation)
        _ensure_cell_class_alias(group)
        group.obs = _clean_categories(group.obs)

    # Keep metadata JSON-compatible so it can be inspected without loading X.
    group.uns["deepspatial_group_h5ad"] = {
        "format": "deepspatial-normalized-group-v1",
        "series_id": str(series_id),
        "tissue_id": str(group.obs["tissue_id"].astype(str).iloc[0]),
        "group_id": str(group.obs["g_id"].iloc[0]),
        "coordinate_frame": f"{series_id}_registered",
        "coordinate_units": "micrometer",
        "spatial_key": "spatial",
        "registered_spatial_key": CANONICAL_SPATIAL_KEY,
        "z_definition": "(section_id - min(section_id in series)) * section_thickness_um",
        "section_thickness_um": float(section_thickness_um),
        "anchor_sections": section_values,
        "n_obs": int(group.n_obs),
        "n_vars": int(group.n_vars),
        "counts_semantics": "raw Xenium cell-by-gene counts",
        "h_and_e_only_sections_included": False,
        "labels": label_info,
    }
    return group


def _section_summary(group: ad.AnnData) -> pd.DataFrame:
    return (
        group.obs.groupby(
            ["series_id", "section_id", "series_section_order", "z_um"],
            observed=True,
            dropna=False,
        )
        .size()
        .reset_index(name="n_cells")
        .sort_values(["series_section_order", "section_id"])
    )


def write_normalized_groups(
    source_h5ad: Path | str,
    output_dir: Path | str,
    *,
    annotation_h5ad: Path | str | None = None,
    series_ids: Iterable[str] | None = None,
    section_thickness_um: float = DEFAULT_THICKNESS_UM,
    compression: str = "gzip",
    overwrite: bool = False,
) -> dict[str, object]:
    """Write one H5AD plus a global manifest/QA table per tissue/g series."""

    source_path = Path(source_h5ad).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = ad.read_h5ad(source_path, backed="r")
    annotation = None
    try:
        _require_source(source)
        available = sorted(_as_string(source.obs["joint_series_id"]).unique().tolist())
        requested = available if series_ids is None else [str(x) for x in series_ids]
        unknown = sorted(set(requested) - set(available))
        if unknown:
            raise KeyError("Unknown joint_series_id values: " + ", ".join(unknown))
        if len(set(requested)) != len(requested):
            raise ValueError("series_ids contains duplicates")

        if annotation_h5ad is not None:
            annotation = ad.read_h5ad(Path(annotation_h5ad).resolve(), backed="r")

        records: list[dict[str, object]] = []
        section_frames: list[pd.DataFrame] = []
        for series_id in requested:
            filename = f"{series_id.replace('__', '_')}.h5ad"
            target = output / filename
            if target.exists() and not overwrite:
                raise FileExistsError(
                    f"Refusing to overwrite existing output: {target}. Use --overwrite explicitly."
                )
            group = materialize_group(
                source,
                series_id,
                section_thickness_um=section_thickness_um,
                annotation=annotation,
            )
            temporary = target.with_name(target.name + ".tmp")
            if temporary.exists():
                temporary.unlink()
            group.write_h5ad(temporary, compression=compression)
            os.replace(temporary, target)

            section = _section_summary(group)
            section_frames.append(section)
            records.append(
                {
                    "series_id": series_id,
                    "output": str(target),
                    "n_obs": int(group.n_obs),
                    "n_vars": int(group.n_vars),
                    "n_sections": int(section["section_id"].nunique()),
                    "sections": [int(x) for x in section["section_id"].tolist()],
                    "labels_status": group.uns["deepspatial_group_h5ad"]["labels"]["status"],
                    "size_bytes": int(target.stat().st_size),
                }
            )
            del group

        manifest = {
            "format": "deepspatial-normalized-groups-v1",
            "source_h5ad": str(source_path),
            "source_shape": [int(source.n_obs), int(source.n_vars)],
            "source_spatial_key": CANONICAL_SPATIAL_KEY,
            "coordinate_units": "micrometer",
            "section_thickness_um": float(section_thickness_um),
            "group_count": len(records),
            "groups": records,
            "checks": {
                "grouping_key": "joint_series_id",
                "expression_matrix": "raw counts preserved in X",
                "obs_names": "joint_cell_id",
                "spatial": "registered XY copied to obsm['spatial']",
                "h_and_e_only_sections": "not included as cells",
                "transcripts_parquet_read": False,
            },
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        )
        if section_frames:
            pd.concat(section_frames, ignore_index=True).to_csv(
                output / "section_summary.csv", index=False
            )
        return manifest
    finally:
        if annotation is not None and annotation.isbacked:
            annotation.file.close()
        if source.isbacked:
            source.file.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-h5ad", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--annotation-h5ad", type=Path, default=None)
    parser.add_argument("--section-thickness-um", type=float, default=DEFAULT_THICKNESS_UM)
    parser.add_argument("--compression", default="gzip", choices=("gzip", "lzf", "none"))
    parser.add_argument("--series-id", action="append", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    compression = None if args.compression == "none" else args.compression
    manifest = write_normalized_groups(
        args.source_h5ad,
        args.output_dir,
        annotation_h5ad=args.annotation_h5ad,
        series_ids=args.series_id,
        section_thickness_um=args.section_thickness_um,
        compression=compression,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
