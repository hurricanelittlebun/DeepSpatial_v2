"""Materialize the final 9957 manual-alignment coordinate release.

This is deliberately a new release.  It does not overwrite v5.  The source
feature and nucleus grids were extracted with an older section transform chain;
they are resampled with the relative old->v9->final affine so corrections such
as 35->36 are present in the actual grids, not only in metadata.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepspatial.data_utils.final_manual_alignment import (  # noqa: E402
    build_propagated_section_extras,
    build_relative_feature_affine,
    compose_final_section_transform,
)
from deepspatial.histology import FeatureStore  # noqa: E402
from deepspatial.histology.frame_resample import resample_feature_store  # noqa: E402


DEFAULT_GROUP = ROOT / "data" / "9957_g0"
DEFAULT_SOURCE_PREP = DEFAULT_GROUP / "reconstruction_prep_v5_coordinate_repair"
DEFAULT_SOURCE_REG = (
    DEFAULT_GROUP
    / "registration_correction_he_v5_coordinate_repair"
)
DEFAULT_OLD_FEATURE = DEFAULT_GROUP / "reconstruction_prep_v1" / "uni2_features.h5"
DEFAULT_V9_ROOT = (
    ROOT.parent
    / "DeepSpatial_v2"
    / "outputs"
    / "registration_9957_pairfix_v9_mask_from_v2_total_v7_affine"
)
DEFAULT_INTEGRATED_ROOT = (
    ROOT.parent
    / "DeepSpatial_v2"
    / "outputs"
    / "registration_9957_manual_rotation_v4_integrated_v2"
)
DEFAULT_OUTPUT_PREP = DEFAULT_GROUP / "reconstruction_prep_v6_final_manual_alignment"
DEFAULT_OUTPUT_REG = DEFAULT_GROUP / "registration_correction_he_v6_final_manual_alignment"
DEFAULT_OUTPUT_QC = DEFAULT_GROUP / "qc" / "he_alignment_all_sections_v3_final_manual"

FINAL_FRAME = "9957__g0_registered_final_manual_alignment_v6"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _key(section_id: str | int) -> str:
    import hashlib

    return hashlib.sha256(str(section_id).encode()).hexdigest()


def _load_matrix(path: Path, key: str = "matrix") -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        if key not in payload.files:
            raise KeyError(f"{path} lacks {key}")
        matrix = np.asarray(payload[key], dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"Invalid affine in {path}")
    if not np.allclose(matrix[2], [0.0, 0.0, 1.0], atol=1e-8, rtol=0.0):
        raise ValueError(f"Invalid homogeneous affine in {path}")
    if abs(float(np.linalg.det(matrix[:2, :2]))) <= 1e-12:
        raise ValueError(f"Singular affine in {path}")
    return matrix


def _apply_affine(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(points, dtype=np.float64)
    homogeneous = np.concatenate(
        [value, np.ones((len(value), 1), dtype=np.float64)], axis=1
    )
    return (homogeneous @ matrix.T)[:, :2]


def _load_manual_edges(integrated_root: Path) -> dict[int, np.ndarray]:
    path = integrated_root / "manual_rotation_v4_integration_manifest.json"
    manifest = _read_json(path)
    result: dict[int, np.ndarray] = {}
    for record in manifest.get("edges", []):
        edge_key = str(record["edge_key"])
        moving_section = int(edge_key.rsplit("__", 1)[-1])
        result[moving_section] = np.asarray(
            record["manual_target_affine"], dtype=np.float64
        )
    if not result:
        raise ValueError(f"No final manual residuals found in {path}")
    return result


def _load_old_transforms(old_feature_store: Path) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    with h5py.File(old_feature_store, "r") as handle:
        for key in handle["sections"]:
            group = handle["sections"][key]
            section_id = str(group.attrs["section_id"])
            metadata = json.loads(group.attrs["metadata"])
            source = metadata.get("provenance", {}).get("section_transform")
            if not source:
                raise KeyError(f"Section {section_id} has no old section_transform")
            result[section_id] = _load_matrix(Path(source).resolve())
    return result


def _load_v9_transforms(v9_root: Path, section_ids: list[str]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    directory = v9_root / "transforms" / "9957" / "g0"
    for section_id in section_ids:
        path = directory / (
            f"section-{int(section_id)}-9957__g0__section-{int(section_id)}.npz"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        result[section_id] = _load_matrix(path)
    return result


def _load_source_sections(feature_store: Path) -> list[str]:
    store = FeatureStore(feature_store)
    return [str(section_id) for section_id in store.sections]


def _write_transform(
    source_path: Path,
    output_path: Path,
    final_matrix: np.ndarray,
    extra: np.ndarray,
    *,
    section_id: int,
    output_frame: str,
) -> None:
    with np.load(source_path, allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}
    arrays["matrix"] = final_matrix
    arrays["final_manual_residual"] = extra
    arrays["final_alignment_section"] = np.asarray([section_id], dtype=np.int64)
    arrays["final_alignment_coordinate_frame"] = np.asarray(output_frame)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)


def _materialize_he_table(
    source_table: Path,
    source_transform_dir: Path,
    output_table: Path,
    output_transform_dir: Path,
    final_matrices: dict[str, np.ndarray],
    extras: dict[str, np.ndarray],
) -> dict:
    table = pd.read_parquet(source_table).copy()
    table["section_id"] = pd.to_numeric(table["section_id"], errors="raise").astype(int)
    output_paths = []
    records = []
    for row in table.itertuples(index=False):
        sid = str(int(row.section_id))
        source_path = source_transform_dir / Path(str(row.section_transform_path)).name
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        output_path = output_transform_dir / source_path.name
        _write_transform(
            source_path,
            output_path,
            final_matrices[sid],
            extras[sid],
            section_id=int(row.section_id),
            output_frame=FINAL_FRAME,
        )
        output_paths.append(str(output_path.resolve()))
        records.append(
            {
                "section_id": int(row.section_id),
                "source_transform": str(source_path.resolve()),
                "output_transform": str(output_path.resolve()),
                "propagated_manual_residual": extras[sid].tolist(),
                "final_matrix": final_matrices[sid].tolist(),
            }
        )
    table["section_transform_path"] = output_paths
    table["coordinate_frame"] = FINAL_FRAME
    table["final_manual_alignment_v6"] = True
    table["final_manual_residual_applied"] = [
        json.dumps(extras[str(int(sid))].tolist()) for sid in table["section_id"]
    ]
    output_table.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output_table, index=False)
    return {"source_table": str(source_table), "output_table": str(output_table), "sections": records}


def _materialize_candidate(
    source_candidate: Path,
    output_candidate_dir: Path,
    extras: dict[str, np.ndarray],
) -> dict:
    source = ad.read_h5ad(source_candidate)
    section_ids = pd.to_numeric(source.obs["section_id"], errors="raise").astype(int).to_numpy()
    current = np.asarray(source.obsm["spatial"], dtype=np.float64)
    source.obsm["spatial_before_final_manual_v6"] = current.astype(np.float32)
    final = np.empty_like(current)
    for section_id in np.unique(section_ids):
        rows = section_ids == section_id
        final[rows] = _apply_affine(current[rows], extras[str(int(section_id))])
    source.obsm["spatial_final_manual_v6"] = final.astype(np.float32)
    for key in ("spatial", "spatial_registered", "spatial_st_corrected"):
        source.obsm[key] = final.astype(np.float32)
    source.obs["coordinate_frame"] = FINAL_FRAME
    source.obs["final_manual_alignment_v6"] = True
    source.obs["final_manual_residual_applied"] = [
        json.dumps(extras[str(int(sid))].tolist()) for sid in section_ids
    ]
    source.uns["deepspatial_final_manual_alignment_v6"] = {
        "coordinate_frame": FINAL_FRAME,
        "source_candidate": str(source_candidate.resolve()),
        "semantics": "v9 registered frame plus final reviewed target-frame manual residuals",
    }
    output_candidate_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_candidate_dir / "9957_g0__st_to_final_manual_alignment_v6.h5ad"
    source.write_h5ad(output_path, compression="gzip")

    # ``output_candidate_dir`` is the release directory inside the group
    # directory (``.../9957_g0/registration_correction_...``).  The anchor
    # files belong next to it under the sibling preparation release, not under
    # ``data/``.  Keeping this explicit prevents a silent second copy of the
    # anchors from being written one directory too high.
    anchors_dir = output_candidate_dir.parent / "reconstruction_prep_v6_final_manual_alignment" / "anchors"
    anchors_dir.mkdir(parents=True, exist_ok=True)
    anchors = []
    for section_id in sorted(np.unique(section_ids)):
        mask = section_ids == int(section_id)
        anchor = source[mask].copy()
        anchor.obs["section_id"] = int(section_id)
        anchor.uns["deepspatial_series"] = {
            "series_id": "9957__g0",
            "coordinate_frame": FINAL_FRAME,
            "section_id": int(section_id),
            "section_thickness_um": 5.0,
            "coordinate_repair": "final_manual_alignment_v6",
        }
        anchor_path = anchors_dir / f"section-{int(section_id):03d}.h5ad"
        anchor.write_h5ad(anchor_path, compression="gzip")
        anchors.append({"section_id": int(section_id), "path": str(anchor_path.resolve()), "n_obs": int(anchor.n_obs)})
    return {"source": str(source_candidate.resolve()), "output": str(output_path.resolve()), "n_obs": int(source.n_obs), "anchors": anchors}


def materialize(
    *,
    source_prep: Path,
    source_reg: Path,
    old_feature_store: Path,
    v9_root: Path,
    integrated_root: Path,
    output_prep: Path,
    output_reg: Path,
    output_qc: Path,
) -> dict:
    for path in (output_prep, output_reg, output_qc):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")

    source_feature = source_prep / "uni2_features_global_v4_repaired.h5"
    source_nucleus = source_prep / "nucleus_path_global_v4_repaired" / "nucleus_features.h5"
    source_table = source_prep / "he_sections.parquet"
    source_transform_dir = source_prep / "registration_transforms_global_v4_repaired"
    source_candidate = source_reg / "9957_g0__st_to_global_v4_repaired_candidate.h5ad"
    source_manifest = _read_json(source_prep / "dataset_manifest.json")
    global_frame_affine = np.asarray(source_manifest["manual_affine"], dtype=np.float64)

    section_ids = _load_source_sections(source_feature)
    old_transforms = _load_old_transforms(old_feature_store)
    v9_transforms = _load_v9_transforms(v9_root, section_ids)
    missing_old = sorted(set(section_ids) - set(old_transforms))
    if missing_old:
        raise KeyError(f"Old transform missing for sections: {missing_old}")
    manual_edges = _load_manual_edges(integrated_root)
    extras = build_propagated_section_extras(section_ids, manual_edges)

    final_matrices = {
        sid: compose_final_section_transform(
            global_frame_affine @ v9_transforms[sid], extras[sid]
        )
        for sid in section_ids
    }
    feature_affines = {
        sid: build_relative_feature_affine(
            old_transforms[sid],
            v9_transforms[sid],
            global_frame_affine,
            extras[sid],
        )
        for sid in section_ids
    }

    output_prep.mkdir(parents=True, exist_ok=False)
    output_reg.mkdir(parents=True, exist_ok=False)
    output_qc.mkdir(parents=True, exist_ok=False)
    feature_output = output_prep / "uni2_features_final_manual_alignment_v6.h5"
    nucleus_output = output_prep / "nucleus_path_final_manual_alignment_v6" / "nucleus_features.h5"
    feature_result = resample_feature_store(
        source_feature,
        feature_output,
        feature_affines,
        output_frame=FINAL_FRAME,
        provenance={
            "release": "9957_final_manual_alignment_v6",
            "source_store": str(source_feature.resolve()),
            "old_feature_store": str(old_feature_store.resolve()),
            "v9_registration_root": str(v9_root.resolve()),
            "manual_integration_root": str(integrated_root.resolve()),
            "direction": "old_grid -> global_frame -> v9_section -> propagated_final_manual_residual",
        },
    )
    nucleus_result = resample_feature_store(
        source_nucleus,
        nucleus_output,
        feature_affines,
        output_frame=FINAL_FRAME,
        provenance={
            "release": "9957_final_manual_alignment_v6",
            "source_store": str(source_nucleus.resolve()),
            "old_feature_store": str(old_feature_store.resolve()),
            "v9_registration_root": str(v9_root.resolve()),
            "manual_integration_root": str(integrated_root.resolve()),
            "direction": "old_nucleus_grid -> global_frame -> v9_section -> propagated_final_manual_residual",
        },
    )

    output_table = output_prep / "he_sections.parquet"
    he_result = _materialize_he_table(
        source_table,
        source_transform_dir,
        output_table,
        output_prep / "registration_transforms_final_manual_alignment_v6",
        final_matrices,
        extras,
    )
    candidate_result = _materialize_candidate(source_candidate, output_reg, extras)

    manifest = {
        "status": "materialized",
        "release": "9957_final_manual_alignment_v6",
        "coordinate_frame": FINAL_FRAME,
        "n_sections": len(section_ids),
        "section_ids": [int(sid) for sid in section_ids],
        "source_prep": str(source_prep.resolve()),
        "source_reg": str(source_reg.resolve()),
        "old_feature_store": str(old_feature_store.resolve()),
        "v9_root": str(v9_root.resolve()),
        "integrated_root": str(integrated_root.resolve()),
        "global_frame_affine": global_frame_affine.tolist(),
        "manual_edges_by_moving_section": {str(k): v.tolist() for k, v in manual_edges.items()},
        "propagation_rule": "manual edge residual begins at moving section and is inherited downstream",
        "feature_affines": {sid: feature_affines[sid].tolist() for sid in section_ids},
        "he_table": he_result,
        "feature_store": feature_result,
        "nucleus_store": nucleus_result,
        "candidate": candidate_result,
        "qc_output": str(output_qc.resolve()),
        "note": "UOT/path caches are not copied; they must be rebuilt from this release before training.",
    }
    (output_prep / "final_manual_alignment_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_reg / "final_manual_alignment_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-prep", type=Path, default=DEFAULT_SOURCE_PREP)
    parser.add_argument("--source-reg", type=Path, default=DEFAULT_SOURCE_REG)
    parser.add_argument("--old-feature-store", type=Path, default=DEFAULT_OLD_FEATURE)
    parser.add_argument("--v9-root", type=Path, default=DEFAULT_V9_ROOT)
    parser.add_argument("--integrated-root", type=Path, default=DEFAULT_INTEGRATED_ROOT)
    parser.add_argument("--output-prep", type=Path, default=DEFAULT_OUTPUT_PREP)
    parser.add_argument("--output-reg", type=Path, default=DEFAULT_OUTPUT_REG)
    parser.add_argument("--output-qc", type=Path, default=DEFAULT_OUTPUT_QC)
    args = parser.parse_args()
    result = materialize(
        source_prep=args.source_prep.resolve(),
        source_reg=args.source_reg.resolve(),
        old_feature_store=args.old_feature_store.resolve(),
        v9_root=args.v9_root.resolve(),
        integrated_root=args.integrated_root.resolve(),
        output_prep=args.output_prep.resolve(),
        output_reg=args.output_reg.resolve(),
        output_qc=args.output_qc.resolve(),
    )
    print(json.dumps({k: result[k] for k in ["status", "release", "coordinate_frame", "n_sections", "section_ids", "qc_output"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
