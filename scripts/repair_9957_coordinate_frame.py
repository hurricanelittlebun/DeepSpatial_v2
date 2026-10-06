"""Materialize 9957/g0 in one global, user-approved v4 coordinate frame.

The old v4 integration put H&E sections 55+ into a manual frame while the
earlier sections remained in the pre-manual frame.  This script applies the
same audited global affine to the earlier sections, keeps already-v4 sections
unchanged, and rebuilds all training-facing artifacts in new directories.
It never overwrites an existing result.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepspatial.data_utils.coordinate_repair import (  # noqa: E402
    build_global_v4_affines,
    build_global_v4_coordinates,
    build_repaired_obsm,
    require_new_output,
    select_repaired_coordinates,
)
from deepspatial.histology import FeatureStore  # noqa: E402
from deepspatial.histology.frame_resample import (  # noqa: E402
    resample_feature_store,
    transform_points,
)


DEFAULT_GROUP = ROOT / "data" / "9957_g0"
DEFAULT_CANDIDATE = (
    DEFAULT_GROUP
    / "registration_correction_he_v4_manual_rotation"
    / "9957_g0__st_to_fixed_he_candidate.h5ad"
)
DEFAULT_CANDIDATE_MANIFEST = (
    DEFAULT_GROUP
    / "registration_correction_he_v4_manual_rotation"
    / "candidate_h5ad_manifest.json"
)
DEFAULT_FEATURE_STORE = (
    DEFAULT_GROUP
    / "reconstruction_prep_v4_manual_rotation"
    / "uni2_features_v4_registered_v2.h5"
)
DEFAULT_NUCLEUS_STORE = (
    DEFAULT_GROUP
    / "reconstruction_prep_v4_manual_rotation"
    / "nucleus_path_v4_manual_rotation"
    / "nucleus_features.h5"
)
DEFAULT_NUCLEUS_UPDATE_MANIFEST = (
    DEFAULT_GROUP
    / "reconstruction_prep_v4_manual_rotation"
    / "nucleus_path_v4_manual_rotation"
    / "coordinate_update_manifest.json"
)
DEFAULT_HE_TABLE = (
    DEFAULT_GROUP / "reconstruction_prep_v4_manual_rotation" / "he_sections.parquet"
)

OUTPUT_FRAME = "9957__g0_registered_manual_rotation_v4_global_repaired"


def _read_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def _manual_affine(path: Path) -> np.ndarray:
    manifest = _read_json(path)
    matrix = np.asarray(manifest["matrix"], dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 manual affine in {path}")
    return matrix


def _manual_nucleus_sections(path: Path) -> set[str]:
    manifest = _read_json(path)
    return {str(row["section_id"]) for row in manifest["updated_sections"]}


def _store_metadata(path: Path) -> dict[str, dict]:
    store = FeatureStore(path)
    return {str(section_id): dict(metadata) for section_id, metadata in store.metadata.items()}


def _ensure_no_existing_outputs(paths: list[Path]) -> None:
    for path in paths:
        require_new_output(path)


def _write_global_transform(
    source_path: Path,
    output_path: Path,
    affine: np.ndarray,
    *,
    source_section: int,
    target_frame: str,
) -> None:
    """Copy one transform and compose a global frame affine if needed."""

    with np.load(source_path, allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}
    matrix = np.asarray(arrays["matrix"], dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"Invalid registration matrix in {source_path}")
    arrays["matrix"] = affine @ matrix
    arrays["global_frame_affine"] = np.asarray(affine, dtype=np.float64)
    arrays["global_frame_section"] = np.asarray([int(source_section)], dtype=np.int64)
    arrays["global_coordinate_frame"] = np.asarray(target_frame)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)


def _materialize_he_table(
    source_table: Path,
    output_table: Path,
    *,
    affine_by_section: dict[str, np.ndarray],
    transform_output_dir: Path,
    target_frame: str,
) -> dict:
    table = pd.read_parquet(source_table).copy()
    if "section_id" not in table or "section_transform_path" not in table:
        raise KeyError("H&E table must contain section_id and section_transform_path")
    table["section_id"] = pd.to_numeric(table["section_id"], errors="raise").astype(int)
    if table["section_id"].duplicated().any():
        raise ValueError("H&E table has duplicate section IDs")

    output_paths = []
    transform_rows = []
    for row in table.itertuples(index=False):
        sid = str(int(row.section_id))
        source_transform = Path(str(row.section_transform_path))
        if not source_transform.is_file():
            raise FileNotFoundError(source_transform)
        output_transform = transform_output_dir / source_transform.name
        _write_global_transform(
            source_transform,
            output_transform,
            affine_by_section[sid],
            source_section=int(row.section_id),
            target_frame=target_frame,
        )
        output_paths.append(str(output_transform.resolve()))
        transform_rows.append(
            {
                "section_id": int(row.section_id),
                "source_transform": str(source_transform.resolve()),
                "output_transform": str(output_transform.resolve()),
                "matrix_source_to_global": affine_by_section[sid].tolist(),
            }
        )
    table["section_transform_path"] = output_paths
    table["coordinate_frame"] = str(target_frame)
    output_table.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output_table, index=False)
    return {
        "source_table": str(source_table.resolve()),
        "output_table": str(output_table.resolve()),
        "n_sections": int(len(table)),
        "transforms": transform_rows,
    }


def _materialize_candidate(
    source_path: Path,
    output_path: Path,
    anchors_dir: Path,
    *,
    manual_affine: np.ndarray,
    target_frame: str,
    manual_anchor_sections: set[str],
) -> dict:
    source = ad.read_h5ad(source_path)
    if "section_id" not in source.obs:
        raise KeyError("Candidate H5AD needs obs['section_id']")
    required = {
        "spatial_st_corrected_pre_manual_v4",
        "spatial_st_corrected_v4",
    }
    missing = sorted(required - set(source.obsm.keys()))
    if missing:
        raise KeyError(f"Candidate H5AD lacks repaired coordinate keys: {missing}")

    section_ids = pd.to_numeric(source.obs["section_id"], errors="raise").astype(int).to_numpy()
    pre = select_repaired_coordinates(source.obsm)
    current_v4 = np.asarray(source.obsm["spatial_st_corrected_v4"], dtype=np.float64)
    repaired = build_global_v4_coordinates(
        pre,
        current_v4,
        section_ids,
        manual_affine,
        manual_section_ids=manual_anchor_sections,
    )
    if not np.isfinite(repaired).all():
        raise ValueError("Repaired ST coordinates contain non-finite values")

    for key, value in build_repaired_obsm(source.obsm).items():
        # Preserve both audited source coordinate columns below; training keys
        # are assigned explicitly to the unified global v4 coordinates.
        source.obsm[key] = value
    source.obsm["spatial_st_corrected_repaired_v5"] = repaired.copy()
    for key in ("spatial", "spatial_registered", "spatial_st_corrected"):
        source.obsm[key] = repaired.copy()

    source.obs["coordinate_frame"] = str(target_frame)
    source.obs["x_st_corrected_repaired_v5"] = repaired[:, 0]
    source.obs["y_st_corrected_repaired_v5"] = repaired[:, 1]
    source.obs["coordinate_repair_v5_applied"] = True
    source.uns["deepspatial_coordinate_repair_v5"] = {
        "source": str(source_path.resolve()),
        "coordinate_frame": target_frame,
        "coordinate_source_pre_manual": "obsm/spatial_st_corrected_pre_manual_v4",
        "coordinate_source_already_v4": "obsm/spatial_st_corrected_v4",
        "manual_anchor_sections_kept_in_v4": sorted(map(int, manual_anchor_sections)),
        "global_affine": manual_affine.tolist(),
        "semantics": "global registered H&E/ST frame; not a new local registration",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source.write_h5ad(output_path, compression="gzip")

    anchors_dir.mkdir(parents=True, exist_ok=True)
    anchor_rows = []
    for section_id in sorted(np.unique(section_ids)):
        mask = section_ids == int(section_id)
        anchor = source[mask].copy()
        anchor.obs["section_id"] = int(section_id)
        anchor.obs["z_um"] = float(anchor.obs["z_um"].iloc[0])
        anchor.uns["deepspatial_series"] = {
            "series_id": "9957__g0",
            "coordinate_frame": target_frame,
            "section_id": int(section_id),
            "section_thickness_um": 5.0,
            "coordinate_repair": "global_v4_frame_v5",
        }
        anchor_path = anchors_dir / f"section-{int(section_id):03d}.h5ad"
        if anchor_path.exists():
            raise FileExistsError(anchor_path)
        anchor.write_h5ad(anchor_path, compression="gzip")
        anchor_rows.append(
            {
                "section_id": int(section_id),
                "path": str(anchor_path.resolve()),
                "n_obs": int(anchor.n_obs),
                "coordinate_frame": target_frame,
                "manual_v4_kept": str(int(section_id)) in manual_anchor_sections,
            }
        )
    return {
        "source": str(source_path.resolve()),
        "output": str(output_path.resolve()),
        "n_obs": int(source.n_obs),
        "n_vars": int(source.n_vars),
        "manual_anchor_sections": sorted(map(int, manual_anchor_sections)),
        "anchors": anchor_rows,
    }


def repair_9957(
    *,
    candidate: Path,
    candidate_manifest: Path,
    feature_store: Path,
    nucleus_store: Path,
    nucleus_update_manifest: Path,
    he_table: Path,
    candidate_output_dir: Path,
    prep_output_dir: Path,
    target_frame: str,
) -> dict:
    candidate_output_dir = require_new_output(candidate_output_dir)
    prep_output_dir = require_new_output(prep_output_dir)
    _ensure_no_existing_outputs([candidate_output_dir, prep_output_dir])

    manual_affine = _manual_affine(candidate_manifest)
    nucleus_manual_sections = _manual_nucleus_sections(nucleus_update_manifest)

    # Derive ST sections already in v4 from the two audited coordinate columns,
    # rather than assuming a particular anchor numbering pattern.
    source = ad.read_h5ad(candidate, backed="r")
    try:
        pre = np.asarray(source.obsm["spatial_st_corrected_pre_manual_v4"])
        current = np.asarray(source.obsm["spatial_st_corrected_v4"])
        section_ids = pd.to_numeric(source.obs["section_id"], errors="raise").astype(int).to_numpy()
        changed = np.max(np.abs(current - pre), axis=1) > 1e-5
        manual_anchor_sections = {str(int(x)) for x in np.unique(section_ids[changed])}
    finally:
        source.file.close()

    candidate_path = candidate_output_dir / "9957_g0__st_to_global_v4_repaired_candidate.h5ad"
    anchors_dir = prep_output_dir / "anchors"
    candidate_manifest_out = _materialize_candidate(
        candidate,
        candidate_path,
        anchors_dir,
        manual_affine=manual_affine,
        target_frame=target_frame,
        manual_anchor_sections=manual_anchor_sections,
    )

    feature_metadata = _store_metadata(feature_store)
    feature_affines = build_global_v4_affines(feature_metadata, manual_affine)
    feature_path = prep_output_dir / "uni2_features_global_v4_repaired.h5"
    feature_resample = resample_feature_store(
        feature_store,
        feature_path,
        feature_affines,
        output_frame=target_frame,
        provenance={
            "repair": "9957_global_v4_frame_v5",
            "manual_affine": manual_affine.tolist(),
            "source_store": str(feature_store.resolve()),
        },
    )

    nucleus_metadata = _store_metadata(nucleus_store)
    nucleus_affines = build_global_v4_affines(
        nucleus_metadata,
        manual_affine,
        manual_section_ids=nucleus_manual_sections,
    )
    nucleus_path = prep_output_dir / "nucleus_path_global_v4_repaired" / "nucleus_features.h5"
    nucleus_resample = resample_feature_store(
        nucleus_store,
        nucleus_path,
        nucleus_affines,
        output_frame=target_frame,
        provenance={
            "repair": "9957_global_v4_frame_v5",
            "manual_affine": manual_affine.tolist(),
            "source_store": str(nucleus_store.resolve()),
            "manual_sections_from_manifest": sorted(map(int, nucleus_manual_sections)),
        },
    )

    # The v4 H&E table contains the accepted manual transforms.  Compose the
    # same global affine into the old-frame sections so image/mask metadata,
    # feature grids, nuclei, and ST all advertise the same frame.
    he_output = prep_output_dir / "he_sections.parquet"
    transform_output_dir = prep_output_dir / "registration_transforms_global_v4_repaired"
    he_table_manifest = _materialize_he_table(
        he_table,
        he_output,
        affine_by_section=feature_affines,
        transform_output_dir=transform_output_dir,
        target_frame=target_frame,
    )

    dataset_manifest = {
        "series_id": "9957__g0",
        "coordinate_frame": target_frame,
        "anchor_sections": candidate_manifest_out["anchors"],
        "section_thickness_um": 5.0,
        "z_range_um": [0.0, 450.0],
        "source_candidate": str(candidate.resolve()),
        "source_candidate_manifest": str(candidate_manifest.resolve()),
        "source_feature_store": str(feature_store.resolve()),
        "source_nucleus_store": str(nucleus_store.resolve()),
        "manual_affine": manual_affine.tolist(),
        "manual_anchor_sections_kept_in_v4": sorted(map(int, manual_anchor_sections)),
        "nucleus_manual_sections_from_manifest": sorted(map(int, nucleus_manual_sections)),
        "feature_store": str(feature_path.resolve()),
        "nucleus_feature_store": str(nucleus_path.resolve()),
        "he_table": str(he_output.resolve()),
        "coordinate_semantics": (
            "old-frame sections receive manual_affine; sections already in the "
            "confirmed manual v4 frame receive identity"
        ),
    }
    (prep_output_dir / "dataset_manifest.json").write_text(
        json.dumps(dataset_manifest, indent=2) + "\n"
    )
    repair_manifest = {
        "status": "complete",
        "coordinate_frame": target_frame,
        "candidate": candidate_manifest_out,
        "feature_store": feature_resample,
        "nucleus_store": nucleus_resample,
        "he_table": he_table_manifest,
        "manual_affine": manual_affine.tolist(),
        "manual_anchor_sections": sorted(map(int, manual_anchor_sections)),
        "nucleus_manual_sections": sorted(map(int, nucleus_manual_sections)),
    }
    (prep_output_dir / "coordinate_repair_manifest.json").write_text(
        json.dumps(repair_manifest, indent=2) + "\n"
    )
    (candidate_output_dir / "coordinate_repair_manifest.json").write_text(
        json.dumps(repair_manifest, indent=2) + "\n"
    )
    return repair_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--candidate-manifest", type=Path, default=DEFAULT_CANDIDATE_MANIFEST)
    parser.add_argument("--feature-store", type=Path, default=DEFAULT_FEATURE_STORE)
    parser.add_argument("--nucleus-store", type=Path, default=DEFAULT_NUCLEUS_STORE)
    parser.add_argument("--nucleus-update-manifest", type=Path, default=DEFAULT_NUCLEUS_UPDATE_MANIFEST)
    parser.add_argument("--he-table", type=Path, default=DEFAULT_HE_TABLE)
    parser.add_argument(
        "--candidate-output-dir",
        type=Path,
        default=DEFAULT_GROUP / "registration_correction_he_v5_coordinate_repair",
    )
    parser.add_argument(
        "--prep-output-dir",
        type=Path,
        default=DEFAULT_GROUP / "reconstruction_prep_v5_coordinate_repair",
    )
    parser.add_argument("--coordinate-frame", default=OUTPUT_FRAME)
    args = parser.parse_args()
    result = repair_9957(
        candidate=args.candidate,
        candidate_manifest=args.candidate_manifest,
        feature_store=args.feature_store,
        nucleus_store=args.nucleus_store,
        nucleus_update_manifest=args.nucleus_update_manifest,
        he_table=args.he_table,
        candidate_output_dir=args.candidate_output_dir,
        prep_output_dir=args.prep_output_dir,
        target_frame=args.coordinate_frame,
    )
    print(json.dumps({
        "status": result["status"],
        "coordinate_frame": result["coordinate_frame"],
        "candidate_output": result["candidate"]["output"],
        "prep_output": str(args.prep_output_dir.resolve()),
        "manual_anchor_sections": result["manual_anchor_sections"],
    }, indent=2))


if __name__ == "__main__":
    main()
