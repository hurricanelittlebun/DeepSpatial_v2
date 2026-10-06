"""Create a new 9957/g0 release with section 54 removed.

The source release is kept immutable.  Section 53 is the fixed endpoint and
section 55 is directly registered to it with a separately run official
STalign edge.  The direct edge is then applied to the surviving downstream
sections in the registered frame; this script never substitutes affine
composition for the direct STalign registration.
"""

from __future__ import annotations

import argparse
import json
import sys
import shutil
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
V2_SRC = ROOT.parent / "DeepSpatial_v2" / "src"
if str(V2_SRC) not in sys.path:
    sys.path.insert(0, str(V2_SRC))

from deepspatial.data_utils.section_filter import adjacent_pairs, filter_section_ids  # noqa: E402
from deepspatial_v2.data_export.cell_level_alignment import (  # noqa: E402
    load_stalign_map,
)
from deepspatial_v2.serial_registration.stalign_backend import (  # noqa: E402
    transform_points_target_to_source_with_edge,
    transform_points_with_edge,
)
from deepspatial_v2.serial_registration.stalign_types import EdgeAlignment  # noqa: E402
from deepspatial.histology import FeatureStore  # noqa: E402


DEFAULT_GROUP = ROOT / "data" / "9957_g0"
DEFAULT_SOURCE_PREP = DEFAULT_GROUP / "reconstruction_prep_v6_final_manual_alignment"
DEFAULT_SOURCE_REG = DEFAULT_GROUP / "registration_correction_he_v6_final_manual_alignment"
DEFAULT_OUTPUT_PREP = DEFAULT_GROUP / "reconstruction_prep_v7_without_section54"
DEFAULT_OUTPUT_REG = DEFAULT_GROUP / "registration_correction_he_v7_without_section54"
DEFAULT_OUTPUT_QC = DEFAULT_GROUP / "qc" / "he_alignment_all_sections_v4_without_section54"
DEFAULT_DIRECT_STALIGN = DEFAULT_GROUP / "stalign_direct_53_55_v1"
OUTPUT_FRAME = "9957__g0_registered_direct_stalign_53_55_v7_without_section54"
REMOVED_SECTION_IDS = (54,)
DIRECT_FIXED_SECTION = 53
DIRECT_MOVING_SECTION = 55


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _load_direct_stalign_edge(direct_root: Path, *, device: str) -> tuple[EdgeAlignment, dict]:
    """Load the separately executed direct 53->55 STalign edge."""

    summary_path = direct_root / "direct_stalign_summary.json"
    pair_dir = direct_root / "pair-53-55"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = _read_json(summary_path)
    required = {
        "direct_stalign_completed",
        "fixed_section",
        "moving_section",
    }
    if summary.get("status") != "direct_stalign_completed":
        raise ValueError(f"Direct STalign artifact is not complete: {summary.get('status')}")
    if int(summary["fixed_section"]) != DIRECT_FIXED_SECTION or int(summary["moving_section"]) != DIRECT_MOVING_SECTION:
        raise ValueError("Direct STalign artifact is not the required 53->55 edge")
    affine_path = pair_dir / "direct_stalign_affine.npz"
    forward_path = pair_dir / "direct_stalign_forward_map.npz"
    reverse_path = pair_dir / "direct_stalign_reverse_map.npz"
    for path in (affine_path, forward_path, reverse_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    with np.load(affine_path, allow_pickle=False) as payload:
        affine = np.asarray(payload["affine"], dtype=float)
    if affine.shape != (3, 3) or not np.isfinite(affine).all():
        raise ValueError(f"Invalid direct STalign affine: {affine_path}")
    forward = load_stalign_map(forward_path, device)
    reverse = load_stalign_map(reverse_path, device)
    edge = EdgeAlignment(
        sample_id="9957",
        group_id="g0",
        fixed_section_id=DIRECT_FIXED_SECTION,
        moving_section_id=DIRECT_MOVING_SECTION,
        affine=affine,
        forward_map=forward,
        reverse_map=reverse,
        transformed_points_um=np.empty((0, 2), dtype=float),
        metrics={
            str(key): float(value)
            for key, value in summary.get("stalign_metrics", {}).items()
            if np.isscalar(value)
        },
        status="accepted_direct_stalign",
    )
    return edge, summary


def _grid_centers(metadata: dict) -> np.ndarray:
    height, width, _ = (int(value) for value in metadata["shape"])
    origin = np.asarray(metadata["origin_um"], dtype=float)
    spacing = np.asarray(metadata["spacing_um"], dtype=float)
    yy, xx = np.mgrid[:height, :width]
    return np.column_stack(
        [origin[0] + xx.reshape(-1) * spacing[0], origin[1] + yy.reshape(-1) * spacing[1]]
    )


def _resample_section_with_direct_edge(
    source: FeatureStore,
    section_id: str,
    edge: EdgeAlignment,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Query a regular target grid through the full saved STalign inverse map."""

    metadata = source.metadata[section_id]
    target_points = _grid_centers(metadata)
    source_points = transform_points_target_to_source_with_edge(target_points, edge)
    queried, valid = source.get_feature(
        section_id,
        source_points,
        return_valid=True,
        nearest_max_distance_um=0.0,
    )
    features = queried.detach().cpu().numpy().astype("float32", copy=False)
    valid_array = valid.detach().cpu().numpy().astype(bool, copy=False)
    shape = tuple(int(value) for value in metadata["shape"])
    return (
        features.reshape(shape[0], shape[1], -1),
        valid_array.reshape(shape[:2]),
        {
            "transform_type": "full_stalign_inverse_query",
            "direct_stalign_edge": "9957__g0__53__55",
            "source_section_id": section_id,
            "target_grid_origin_um": list(map(float, metadata["origin_um"])),
            "target_grid_shape": list(map(int, shape[:2])),
            "output_valid_fraction": float(valid_array.mean()),
        },
    )


def _copy_feature_store(
    source_path: Path,
    output_path: Path,
    kept_sections: list[str],
    *,
    output_frame: str,
    removed_section_ids: list[int],
    direct_edge: EdgeAlignment,
    direct_start_section: int,
    direct_edge_summary: dict,
) -> dict:
    source = FeatureStore(source_path)
    if source.sections != sorted(source.sections, key=lambda sid: source.metadata[sid]["z_um"]):
        raise ValueError(f"Source store is not z ordered: {source_path}")
    if set(kept_sections) - set(source.sections):
        raise KeyError(f"Store {source_path} lacks sections: {sorted(set(kept_sections) - set(source.sections))}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = FeatureStore(output_path, mode="a")
    written = []
    with h5py.File(source_path, "r") as handle:
        for sid in kept_sections:
            metadata = source.metadata[sid]
            group = handle["sections"][source._key(sid)]
            if int(sid) >= int(direct_start_section):
                features, valid, transform_record = _resample_section_with_direct_edge(
                    source, sid, direct_edge
                )
            else:
                features = group["features"][:]
                valid = group["valid"][:]
                transform_record = {
                    "transform_type": "identity",
                    "direct_stalign_edge": "not_applied_before_fixed_section",
                    "source_section_id": sid,
                    "output_valid_fraction": float(np.asarray(valid, dtype=bool).mean()),
                }
            provenance = dict(metadata.get("provenance", {}))
            provenance["section_filter_release"] = "9957_without_section54_v7_direct_stalign"
            provenance["removed_section_ids"] = removed_section_ids
            provenance["source_store"] = str(source_path.resolve())
            provenance["direct_stalign_53_55"] = {
                "summary": direct_edge_summary,
                "section_transform": transform_record,
            }
            output.add_section(
                sid,
                features,
                z_um=float(metadata["z_um"]),
                origin_um=metadata["origin_um"],
                spacing_um=metadata["spacing_um"],
                patch_size_um=float(metadata["patch_size_um"]),
                mpp=metadata["mpp"],
                coordinate_frame=output_frame,
                valid_mask=valid,
                provenance=provenance,
            )
            written.append(sid)
    output.refresh()
    return {
        "source_store": str(source_path.resolve()),
        "output_store": str(output_path.resolve()),
        "source_sections": len(source.sections),
        "output_sections": written,
        "feature_dim": int(output.feature_dim),
        "coordinate_frame": output_frame,
        "direct_stalign_edge": "9957__g0__53__55",
    }


def _copy_transforms_and_table(
    source_table_path: Path,
    output_table_path: Path,
    source_transform_dir: Path,
    output_transform_dir: Path,
    kept_sections: list[str],
    *,
    output_frame: str,
    removed_section_ids: list[int],
    direct_edge: EdgeAlignment,
    direct_edge_summary: dict,
) -> tuple[dict, dict[str, np.ndarray]]:
    source_table = pd.read_parquet(source_table_path).copy()
    source_table["section_id"] = pd.to_numeric(source_table["section_id"], errors="raise").astype(int)
    if source_table["section_id"].duplicated().any():
        raise ValueError("H&E table has duplicate section IDs")
    kept = source_table[source_table["section_id"].astype(str).isin(kept_sections)].copy()
    if len(kept) != len(kept_sections):
        missing = sorted(set(kept_sections) - set(kept["section_id"].astype(str)))
        raise KeyError(f"H&E table missing kept sections: {missing}")
    kept = kept.sort_values("z_um", kind="stable").reset_index(drop=True)
    matrices: dict[str, np.ndarray] = {}
    for row in kept.itertuples(index=False):
        sid = str(int(row.section_id))
        source_path = source_transform_dir / Path(str(row.section_transform_path)).name
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        arrays = _load_npz_arrays(source_path)
        matrix = np.asarray(arrays["matrix"], dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError(f"Invalid section matrix in {source_path}")
        direct_applied = int(row.section_id) >= DIRECT_MOVING_SECTION
        final_matrix = (
            np.asarray(direct_edge.affine, dtype=float) @ matrix
            if direct_applied
            else matrix
        )
        arrays["matrix_before_direct_stalign"] = matrix
        arrays["matrix"] = final_matrix
        arrays["section_filter_release"] = np.asarray(
            "9957_without_section54_v7_direct_stalign"
        )
        arrays["removed_section_ids"] = np.asarray(removed_section_ids, dtype=np.int64)
        arrays["direct_stalign_edge_key"] = np.asarray("9957__g0__53__55")
        arrays["direct_stalign_applied"] = np.asarray([int(direct_applied)], dtype=np.uint8)
        arrays["direct_stalign_affine"] = np.asarray(direct_edge.affine, dtype=float)
        output_path = output_transform_dir / source_path.name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **arrays)
        matrices[sid] = final_matrix
        kept.loc[kept["section_id"] == int(row.section_id), "section_transform_path"] = str(output_path.resolve())

    kept["coordinate_frame"] = output_frame
    kept["section_filter_release"] = "9957_without_section54_v7_direct_stalign"
    kept["removed_section_ids"] = json.dumps(removed_section_ids)
    kept["direct_stalign_edge_key"] = "9957__g0__53__55"
    kept["direct_stalign_applied"] = kept["section_id"] >= DIRECT_MOVING_SECTION
    kept["direct_predecessor_section"] = ""
    kept["direct_pair_key"] = ""
    kept["direct_stalign_summary_path"] = str(
        (direct_edge_summary.get("pair_output", ""))
    )
    ordered_ids = kept["section_id"].astype(int).tolist()
    for index, sid_value in enumerate(ordered_ids):
        if index:
            kept.loc[kept["section_id"] == sid_value, "direct_predecessor_section"] = str(
                ordered_ids[index - 1]
            )
        if index and ordered_ids[index - 1] == DIRECT_FIXED_SECTION and sid_value == DIRECT_MOVING_SECTION:
            kept.loc[kept["section_id"] == sid_value, "direct_pair_key"] = "9957__g0__53__55"
    kept.to_parquet(output_table_path, index=False)
    return {
        "source_table": str(source_table_path.resolve()),
        "output_table": str(output_table_path.resolve()),
        "source_section_count": int(len(source_table)),
        "output_section_count": int(len(kept)),
        "output_sections": [int(x) for x in kept.section_id],
        "direct_stalign_edge": "9957__g0__53__55",
        "direct_stalign_applied_sections": [
            int(x) for x in kept.loc[kept["direct_stalign_applied"], "section_id"]
        ],
    }, matrices


def _write_direct_pair_metadata(
    output_prep: Path,
    matrices: dict[str, np.ndarray],
    *,
    output_frame: str,
    left_section: int,
    right_section: int,
    direct_edge: EdgeAlignment,
    direct_edge_summary: dict,
) -> dict:
    left = str(left_section)
    right = str(right_section)
    value = {
        "pair_key": f"9957__g0__{left_section}__{right_section}",
        "fixed_section_id": left_section,
        "moving_section_id": right_section,
        "source_semantics": "direct H&E STalign edge after section 54 removal",
        "registration_algorithm": "official STalign LDDMM",
        "coordinate_frame": output_frame,
        "direct_stalign_affine": np.asarray(direct_edge.affine, dtype=float).tolist(),
        "source_section_matrices": {
            left: matrices[left].tolist(),
            right: matrices[right].tolist(),
        },
        "removed_intermediate_sections": [54],
        "registration_rerun": True,
        "direct_stalign_summary": direct_edge_summary,
        "direct_stalign_status": direct_edge.status,
        "direct_stalign_maps": {
            "forward": "pair-53-55/direct_stalign_forward_map.npz",
            "reverse": "pair-53-55/direct_stalign_reverse_map.npz",
        },
    }
    _write_json(output_prep / "direct_pair_53_55.json", value)
    return value


def _materialize_candidate(
    source_path: Path,
    output_path: Path,
    anchors_dir: Path,
    *,
    output_frame: str,
    removed_section_ids: list[int],
    direct_edge: EdgeAlignment,
    direct_edge_summary: dict,
) -> dict:
    source = ad.read_h5ad(source_path)
    if "section_id" not in source.obs:
        raise KeyError("Candidate H5AD lacks obs['section_id']")
    section_ids = pd.to_numeric(source.obs["section_id"], errors="raise").astype(int)
    if section_ids.isin(removed_section_ids).any():
        source = source[~section_ids.isin(removed_section_ids)].copy()
    if "spatial" not in source.obsm:
        raise KeyError("Candidate H5AD lacks obsm['spatial']")
    current_spatial = np.asarray(source.obsm["spatial"], dtype=np.float64)
    source.obsm["spatial_before_direct_stalign_v7"] = current_spatial.astype(np.float32)
    final_spatial = current_spatial.copy()
    for section_id in np.unique(section_ids):
        rows = section_ids == int(section_id)
        if int(section_id) >= DIRECT_MOVING_SECTION:
            final_spatial[rows] = transform_points_with_edge(current_spatial[rows], direct_edge)
    for key in ("spatial", "spatial_registered", "spatial_st_corrected"):
        if key in source.obsm:
            source.obsm[key] = final_spatial.astype(np.float32)
    source.obs["coordinate_frame"] = output_frame
    source.obs["section_filter_release"] = "9957_without_section54_v7_direct_stalign"
    source.obs["direct_stalign_53_55_applied"] = section_ids >= DIRECT_MOVING_SECTION
    source.uns["deepspatial_section_filter_v7"] = {
        "coordinate_frame": output_frame,
        "removed_section_ids": removed_section_ids,
        "direct_pair": [53, 55],
        "source_candidate": str(source_path.resolve()),
        "direct_stalign_summary": direct_edge_summary,
        "spatial_transform": "full saved STalign forward map applied to sections >=55; section 53 is fixed",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source.write_h5ad(output_path, compression="gzip")

    anchors_dir.mkdir(parents=True, exist_ok=True)
    anchors = []
    ids = pd.to_numeric(source.obs["section_id"], errors="raise").astype(int).to_numpy()
    for sid in sorted(np.unique(ids)):
        anchor = source[ids == int(sid)].copy()
        anchor.uns["deepspatial_series"] = {
            "series_id": "9957__g0",
            "coordinate_frame": output_frame,
            "section_id": int(sid),
            "section_thickness_um": 5.0,
            "section_filter_release": "9957_without_section54_v7_direct_stalign",
            "direct_stalign_53_55": int(sid) >= DIRECT_MOVING_SECTION,
        }
        path = anchors_dir / f"section-{int(sid):03d}.h5ad"
        anchor.write_h5ad(path, compression="gzip")
        anchors.append({"section_id": int(sid), "path": str(path.resolve()), "n_obs": int(anchor.n_obs)})
    return {
        "source": str(source_path.resolve()),
        "output": str(output_path.resolve()),
        "n_obs": int(source.n_obs),
        "n_sections": int(len(anchors)),
        "anchors": anchors,
    }


def materialize(
    *,
    source_prep: Path,
    source_reg: Path,
    output_prep: Path,
    output_reg: Path,
    output_qc: Path,
    direct_stalign_root: Path,
    direct_stalign_device: str,
) -> dict:
    for path in (output_prep, output_reg, output_qc):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")

    removed = [int(value) for value in REMOVED_SECTION_IDS]
    source_table = pd.read_parquet(source_prep / "he_sections.parquet")
    source_ids = [str(int(value)) for value in source_table.sort_values("z_um")["section_id"]]
    if "54" not in source_ids:
        raise ValueError("Source release does not contain section 54")
    kept = filter_section_ids(source_ids, removed)
    pairs = adjacent_pairs(kept)
    if ("53", "55") not in pairs:
        raise AssertionError(f"Expected direct pair (53,55), got nearby pairs: {pairs}")
    direct_edge, direct_edge_summary = _load_direct_stalign_edge(
        direct_stalign_root, device=direct_stalign_device
    )

    output_prep.mkdir(parents=True, exist_ok=False)
    output_reg.mkdir(parents=True, exist_ok=False)
    # Leave the QC path absent; the renderer creates it atomically and refuses
    # to overwrite an existing QC release.

    table_result, matrices = _copy_transforms_and_table(
        source_prep / "he_sections.parquet",
        output_prep / "he_sections.parquet",
        source_prep / "registration_transforms_final_manual_alignment_v6",
        output_prep / "registration_transforms_without_section54_v7",
        kept,
        output_frame=OUTPUT_FRAME,
        removed_section_ids=removed,
        direct_edge=direct_edge,
        direct_edge_summary=direct_edge_summary,
    )
    direct_pair = _write_direct_pair_metadata(
        output_prep,
        matrices,
        output_frame=OUTPUT_FRAME,
        left_section=53,
        right_section=55,
        direct_edge=direct_edge,
        direct_edge_summary=direct_edge_summary,
    )
    direct_copy = output_prep / "direct_stalign_53_55"
    shutil.copytree(direct_stalign_root, direct_copy)

    feature_result = _copy_feature_store(
        source_prep / "uni2_features_final_manual_alignment_v6.h5",
        output_prep / "uni2_features_without_section54_v7.h5",
        kept,
        output_frame=OUTPUT_FRAME,
        removed_section_ids=removed,
        direct_edge=direct_edge,
        direct_start_section=DIRECT_MOVING_SECTION,
        direct_edge_summary=direct_edge_summary,
    )
    nucleus_result = _copy_feature_store(
        source_prep / "nucleus_path_final_manual_alignment_v6" / "nucleus_features.h5",
        output_prep / "nucleus_path_without_section54_v7" / "nucleus_features.h5",
        kept,
        output_frame=OUTPUT_FRAME,
        removed_section_ids=removed,
        direct_edge=direct_edge,
        direct_start_section=DIRECT_MOVING_SECTION,
        direct_edge_summary=direct_edge_summary,
    )
    candidate_result = _materialize_candidate(
        source_reg / "9957_g0__st_to_final_manual_alignment_v6.h5ad",
        output_reg / "9957_g0__st_to_without_section54_v7.h5ad",
        output_prep / "anchors",
        output_frame=OUTPUT_FRAME,
        removed_section_ids=removed,
        direct_edge=direct_edge,
        direct_edge_summary=direct_edge_summary,
    )
    manifest = {
        "status": "materialized",
        "release": "9957_without_section54_v7",
        "coordinate_frame": OUTPUT_FRAME,
        "source_prep": str(source_prep.resolve()),
        "source_reg": str(source_reg.resolve()),
        "removed_section_ids": removed,
        "source_section_count": len(source_ids),
        "output_section_count": len(kept),
        "output_section_ids": [int(value) for value in kept],
        "adjacent_pairs": [[int(a), int(b)] for a, b in pairs],
        "direct_pair": direct_pair,
        "direct_stalign_root": str(direct_copy.resolve()),
        "direct_stalign_device": direct_stalign_device,
        "direct_stalign_summary": direct_edge_summary,
        "he_table": table_result,
        "feature_store": feature_result,
        "nucleus_store": nucleus_result,
        "candidate": candidate_result,
        "qc_output": str(output_qc.resolve()),
        "cache_policy": "UOT and morphology path caches must be rebuilt because section adjacency changed.",
    }
    _write_json(output_prep / "section_filter_manifest.json", manifest)
    _write_json(output_reg / "section_filter_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-prep", type=Path, default=DEFAULT_SOURCE_PREP)
    parser.add_argument("--source-reg", type=Path, default=DEFAULT_SOURCE_REG)
    parser.add_argument("--output-prep", type=Path, default=DEFAULT_OUTPUT_PREP)
    parser.add_argument("--output-reg", type=Path, default=DEFAULT_OUTPUT_REG)
    parser.add_argument("--output-qc", type=Path, default=DEFAULT_OUTPUT_QC)
    parser.add_argument("--direct-stalign-root", type=Path, default=DEFAULT_DIRECT_STALIGN)
    parser.add_argument("--direct-stalign-device", default="cuda:0")
    args = parser.parse_args()
    result = materialize(
        source_prep=args.source_prep.resolve(),
        source_reg=args.source_reg.resolve(),
        output_prep=args.output_prep.resolve(),
        output_reg=args.output_reg.resolve(),
        output_qc=args.output_qc.resolve(),
        direct_stalign_root=args.direct_stalign_root.resolve(),
        direct_stalign_device=args.direct_stalign_device,
    )
    print(json.dumps({key: result[key] for key in ("status", "release", "removed_section_ids", "source_section_count", "output_section_count", "direct_pair")}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
