"""Prepare one independently registered tissue/g series for DeepSpatial."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def _resolve_path(value: object, root: Path) -> str:
    path = Path(str(value))
    return str(path if path.is_absolute() else (root / path).resolve())


def _series_he_rows(frame: pd.DataFrame, tissue_id: str, group_id: str) -> pd.DataFrame:
    sample_match = frame["sample_id"].fillna("").astype(str).str.zfill(5).eq(tissue_id)
    tissue_match = frame.get("tissue_id", pd.Series("", index=frame.index)).fillna("").astype(str).str.zfill(5).eq(tissue_id)
    result = frame[(sample_match | tissue_match) & frame["group_id"].astype(str).eq(group_id)].copy()
    result["section_id"] = pd.to_numeric(result["section_id"], errors="raise").astype(int)
    if result["section_id"].duplicated().any():
        duplicate = sorted(result.loc[result["section_id"].duplicated(False), "section_id"].unique())
        raise ValueError(f"One tissue/g section must resolve to one H&E object; duplicates: {duplicate}")
    return result.sort_values("section_id", kind="stable").reset_index(drop=True)


def prepare_series(
    source_h5ad: Path | str,
    he_manifest: pd.DataFrame,
    transform_manifest: pd.DataFrame,
    output_dir: Path | str,
    *,
    series_id: str,
    section_thickness_um: float,
    he_geometry_root: Path | str,
    registration_root: Path | str,
    raw_he_root: Path | str | None = None,
    xenium_crop_manifest: pd.DataFrame | None = None,
    spatial_key: str = "spatial_registered",
    validate_paths: bool = True,
) -> dict:
    """Split ST anchors and write the H&E/transform manifest for one tissue/g.

    ``section_id`` is a physical ordinal and is never renumbered. The first ST
    anchor center is defined as z=0; therefore absolute z offset remains arbitrary
    while all within-series physical distances are correct.
    """

    if not np.isfinite(section_thickness_um) or section_thickness_um <= 0:
        raise ValueError("section_thickness_um must be positive")
    parts = str(series_id).split("__", 1)
    if len(parts) != 2 or not all(parts):
        raise ValueError("series_id must have the exact tissue__group form")
    tissue_id, group_id = parts[0].zfill(5), parts[1]
    output = Path(output_dir)
    anchors_dir = output / "anchors"
    anchors_dir.mkdir(parents=True, exist_ok=True)

    source = ad.read_h5ad(source_h5ad, backed="r")
    try:
        if "joint_series_id" not in source.obs or "section_id" not in source.obs:
            raise KeyError("Source H5AD needs obs[joint_series_id, section_id]")
        if spatial_key not in source.obsm:
            raise KeyError(f"Source H5AD needs obsm[{spatial_key!r}]")
        selected = source.obs["joint_series_id"].astype(str).eq(series_id)
        anchor_sections = [int(value) for value in sorted(pd.to_numeric(source.obs.loc[selected, "section_id"], errors="raise").astype(int).unique())]
        if len(anchor_sections) < 2:
            raise ValueError("A series needs at least two ST anchors")
        z_origin_section = anchor_sections[0]
        n_anchor_cells = 0
        for section_id in anchor_sections:
            mask = selected & pd.to_numeric(source.obs["section_id"], errors="coerce").eq(section_id)
            anchor = source[mask.to_numpy()].to_memory()
            n_anchor_cells += int(anchor.n_obs)
            anchor.obs["section_id"] = int(section_id)
            anchor.obs["section_id_str"] = str(section_id)
            anchor.obs["z_um"] = float((section_id - z_origin_section) * section_thickness_um)
            anchor.obsm["spatial"] = np.asarray(anchor.obsm[spatial_key], dtype=np.float64).copy()
            anchor.uns["deepspatial_series"] = {
                "series_id": series_id,
                "tissue_id": tissue_id,
                "group_id": group_id,
                "section_thickness_um": float(section_thickness_um),
                "z_origin_section": int(z_origin_section),
                "z_definition": "(section_id-z_origin_section)*section_thickness_um",
                "coordinate_frame": f"{series_id}_registered",
            }
            anchor.write_h5ad(anchors_dir / f"section-{section_id:03d}.h5ad", compression="gzip")
    finally:
        source.file.close()

    geometry_root = Path(he_geometry_root)
    registration_root = Path(registration_root)
    he = _series_he_rows(he_manifest, tissue_id, group_id)
    start, stop = anchor_sections[0], anchor_sections[-1]
    he = he[he["section_id"].between(start, stop)].copy()
    if he.empty:
        raise ValueError("No H&E sections found in the ST anchor range")

    transforms = transform_manifest.copy()
    transforms["sample_id"] = transforms["sample_id"].astype(str).str.zfill(5)
    transforms["section_id"] = pd.to_numeric(transforms["section_id"], errors="raise").astype(int)
    transforms = transforms[(transforms["sample_id"].eq(tissue_id)) & (transforms["group_id"].astype(str).eq(group_id))]
    keep = ["section_id", "affine_path", "registration_status", "registration_confidence"]
    missing_columns = sorted(set(keep) - set(transforms.columns))
    if missing_columns:
        raise KeyError("Transform manifest missing: " + ", ".join(missing_columns))
    transforms = transforms[keep].rename(columns={"affine_path": "section_transform_path"})
    he = he.merge(transforms, on="section_id", how="left", validate="one_to_one")
    if he["section_transform_path"].isna().any():
        raise ValueError("Every H&E section needs a saved registration transform")

    he["z_um"] = (he["section_id"] - z_origin_section) * float(section_thickness_um)
    he["coordinate_frame"] = f"{series_id}_registered"
    he["source_image_path"] = [
        _resolve_path(value, geometry_root) for value in he["masked_crop_path"]
    ]
    he["source_mask_path"] = [_resolve_path(value, geometry_root) for value in he["mask_path"]]
    he["section_transform_path"] = [
        _resolve_path(value, registration_root) for value in he["section_transform_path"]
    ]
    path_columns = ["source_image_path", "source_mask_path", "section_transform_path"]

    # Preserve level-0 SDPC coordinates for high-resolution pathology feature
    # extraction. Registration still uses the reviewed low-resolution masks.
    if raw_he_root is not None:
        raw_root = Path(raw_he_root)
        # Some local Xenium folders use the displayed tissue number without
        # left padding (for example ``9589``), while the tabular manifests
        # normalize sample/tissue identifiers to five characters
        # (``09589``).  Resolve the physical directory explicitly instead of
        # silently constructing a path that cannot exist.
        raw_tissue_dir = raw_root / tissue_id
        if not raw_tissue_dir.is_dir():
            try:
                raw_tissue_dir = raw_root / str(int(tissue_id))
            except ValueError:
                pass
        raw_paths = []
        for row in he.itertuples(index=False):
            if str(row.source_kind) == "xenium_companion_he":
                raw_paths.append(raw_root / f"{row.source_image_id}.sdpc")
            else:
                raw_paths.append(raw_tissue_dir / str(row.source_filename))
        he["raw_source_path"] = [str(path.resolve()) for path in raw_paths]
        path_columns.append("raw_source_path")

        for column in ("level0_x0", "level0_y0", "level0_x1", "level0_y1"):
            he[f"raw_{column}"] = pd.to_numeric(he.get(column), errors="coerce")
        if xenium_crop_manifest is not None:
            crops = xenium_crop_manifest.copy()
            required = {
                "section_uid", "sdpc_x0_level0", "sdpc_y0_level0",
                "sdpc_x1_level0", "sdpc_y1_level0",
            }
            missing = sorted(required - set(crops.columns))
            if missing:
                raise KeyError("Xenium crop manifest missing: " + ", ".join(missing))
            crops = crops[list(required)].rename(
                columns={
                    "section_uid": "source_section_uid",
                    "sdpc_x0_level0": "anchor_level0_x0",
                    "sdpc_y0_level0": "anchor_level0_y0",
                    "sdpc_x1_level0": "anchor_level0_x1",
                    "sdpc_y1_level0": "anchor_level0_y1",
                }
            )
            he = he.merge(crops, on="source_section_uid", how="left", validate="many_to_one")
            anchor = he["source_kind"].astype(str).eq("xenium_companion_he")
            for axis in ("x0", "y0", "x1", "y1"):
                he.loc[anchor, f"raw_level0_{axis}"] = he.loc[
                    anchor, f"anchor_level0_{axis}"
                ]
                he = he.drop(columns=f"anchor_level0_{axis}")
        required_raw = [
            "raw_level0_x0", "raw_level0_y0", "raw_level0_x1", "raw_level0_y1"
        ]
        if he[required_raw].isna().any().any():
            bad = he.loc[he[required_raw].isna().any(axis=1), "section_id"].tolist()
            raise ValueError(f"Missing level-0 SDPC crop coordinates for sections: {bad}")
        he[required_raw] = he[required_raw].astype(np.int64)
        if "analysis_level" not in he:
            raise KeyError("H&E manifest needs analysis_level for SDPC level-0 MPP")
        levels = pd.to_numeric(he["analysis_level"], errors="raise").astype(int)
        if (levels < 0).any():
            raise ValueError("analysis_level must be non-negative")
        # SDPC scans in this dataset use a 2x pyramid. This also corrects the
        # companion-H&E rows where legacy ruler_raw contains crop-level MPP.
        he["raw_mpp_um"] = pd.to_numeric(
            he["analysis_pixel_size_x_um"], errors="raise"
        ) / np.power(2.0, levels)
    if validate_paths:
        absent = [path for column in path_columns for path in he[column] if not Path(path).is_file()]
        if absent:
            raise FileNotFoundError(f"Missing {len(absent)} series inputs; first: {absent[0]}")

    output_columns = [
        "section_id", "z_um", "coordinate_frame", "source_kind", "source_image_path",
        "source_mask_path", "section_transform_path", "analysis_pixel_size_x_um",
        "analysis_pixel_size_y_um", "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1",
        "registration_status", "registration_confidence",
    ]
    output_columns += [
        column
        for column in (
            "raw_source_path", "raw_level0_x0", "raw_level0_y0",
            "raw_level0_x1", "raw_level0_y1", "raw_mpp_um",
        )
        if column in he
    ]
    he[output_columns].to_parquet(output / "he_sections.parquet", index=False)
    observed = set(he["section_id"]) | set(anchor_sections)
    missing_sections = [section for section in range(start, stop + 1) if section not in observed]
    result = {
        "series_id": series_id,
        "anchor_sections": anchor_sections,
        "he_sections": [int(value) for value in he["section_id"]],
        "missing_sections": missing_sections,
        "section_thickness_um": float(section_thickness_um),
        "z_origin_section": int(z_origin_section),
        "z_range_um": [0.0, float((stop - start) * section_thickness_um)],
        "n_anchor_cells": n_anchor_cells,
        "use_celltype": False,
        "celltype_note": "No cell_class column was asserted by this preparation step.",
    }
    (output / "dataset_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
