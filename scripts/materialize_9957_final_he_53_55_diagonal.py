"""Materialize the final 9957/g0 H&E coordinate release.

The release starts from the reviewed v6 H&E frame and replaces the old
53->54->55 continuation with the user-approved direct 53->55 STalign map.  The
map already contains the target-frame upper-left/lower-right diagonal
reflection.  All downstream H&E coordinates and feature grids from section
55 onward are queried through this saved map; no separate runtime residual is
needed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V2_SRC = ROOT.parent / "DeepSpatial_v2" / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(V2_SRC) not in sys.path:
    sys.path.insert(0, str(V2_SRC))

from deepspatial_v2.data_export.cell_level_alignment import load_stalign_map  # noqa: E402
from deepspatial_v2.serial_registration.stalign_backend import (  # noqa: E402
    transform_points_target_to_source_with_edge,
    transform_points_with_edge,
)
from deepspatial_v2.serial_registration.stalign_types import EdgeAlignment  # noqa: E402


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M8 = _load_module(
    ROOT / "scripts" / "materialize_9957_latest_alignment_cutoff91.py",
    "_materialize_9957_latest_helpers",
)


GROUP = ROOT / "data" / "9957_g0"
ARCHIVE_TOP = (
    ROOT.parent
    / ".deepSpatial_archives"
    / "9957_g0_v5_rollback_20261001_132456"
    / "top_level"
)
SOURCE_PREP = ARCHIVE_TOP / "reconstruction_prep_v6_final_manual_alignment"
SOURCE_REG = ARCHIVE_TOP / "registration_correction_he_v6_final_manual_alignment"
DIRECT_ROOT = GROUP / "stalign_realign_53_55_v8_preview_v2_diagonal_reflection"

RELEASE = "9957_final_he_53_55_diagonal_v1"
OUTPUT_FRAME = "9957__g0_registered_final_he_53_55_diagonal_v1"
VERSION_TAG = "final_he_53_55_diagonal_v1"
OUTPUT_PREP = GROUP / "reconstruction_prep_final_he_53_55_diagonal_v1"
OUTPUT_REG = GROUP / "registration_correction_he_final_he_53_55_diagonal_v1"
OUTPUT_QC = GROUP / "qc" / "he_alignment_final_he_53_55_diagonal_v1"
REMOVED_SECTION_IDS = (54,)
CUTOFF_MAX_SECTION_ID = 91
DIRECT_MOVING_SECTION = 55


class DirectFinalChain:
    """Full v6->final map: identity before 55, direct STalign from 55 on."""

    def __init__(self, edge: EdgeAlignment, summary: dict, *, device: str):
        self.direct_edge = edge
        self.summary = summary
        self.device = device
        self.coarse_affine = np.asarray(edge.affine, dtype=float)

    def forward(self, points: np.ndarray) -> np.ndarray:
        return transform_points_with_edge(np.asarray(points, dtype=float), self.direct_edge)

    def inverse(self, points: np.ndarray) -> np.ndarray:
        return transform_points_target_to_source_with_edge(
            np.asarray(points, dtype=float), self.direct_edge
        )

    def forward_chunked(self, points: np.ndarray, batch_size: int = 16384) -> np.ndarray:
        value = np.asarray(points, dtype=float)
        output = np.empty_like(value)
        for start in range(0, len(value), int(batch_size)):
            stop = min(start + int(batch_size), len(value))
            output[start:stop] = self.forward(value[start:stop])
        return output

    def inverse_chunked(self, points: np.ndarray, batch_size: int = 16384) -> np.ndarray:
        value = np.asarray(points, dtype=float)
        output = np.empty_like(value)
        for start in range(0, len(value), int(batch_size)):
            stop = min(start + int(batch_size), len(value))
            output[start:stop] = self.inverse(value[start:stop])
        return output

    def metadata(self) -> dict:
        return {
            "coordinate_semantics": "v6 registered H&E XY -> final registered H&E XY",
            "chain_order": ["direct_stalign_53_to_55_with_baked_diagonal_reflection"],
            "fixed_section": 53,
            "moving_section": 55,
            "apply_to_sections": "section_id >= 55",
            "coarse_affine_is_baked_active_value": True,
            "direct_stalign_summary": self.summary,
            "direct_stalign_maps": {
                "forward": "pair-53-55/direct_stalign_forward_map.npz",
                "reverse": "pair-53-55/direct_stalign_reverse_map.npz",
            },
        }


def _load_direct_edge(root: Path, device: str) -> tuple[EdgeAlignment, dict]:
    summary = json.loads((root / "direct_stalign_summary.json").read_text(encoding="utf-8"))
    if not str(summary.get("status", "")).startswith("direct_stalign_completed"):
        raise ValueError(f"unexpected direct-map status: {summary.get('status')}")
    if int(summary["fixed_section"]) != 53 or int(summary["moving_section"]) != 55:
        raise ValueError("direct artifact is not section 53 -> 55")
    pair = root / "pair-53-55"
    affine_path = pair / "direct_stalign_affine.npz"
    with np.load(affine_path, allow_pickle=False) as payload:
        affine = np.asarray(payload["affine"], dtype=float)
    edge = EdgeAlignment(
        sample_id="9957",
        group_id="g0",
        fixed_section_id=53,
        moving_section_id=55,
        affine=affine,
        forward_map=load_stalign_map(pair / "direct_stalign_forward_map.npz", device),
        reverse_map=load_stalign_map(pair / "direct_stalign_reverse_map.npz", device),
        transformed_points_um=np.empty((0, 2), dtype=float),
        metrics={
            str(key): float(value)
            for key, value in summary.get("stalign_metrics", {}).items()
            if np.isscalar(value)
        },
        status="accepted_final_direct_stalign_diagonal_baked",
    )
    return edge, summary


def _set_helper_release_constants() -> None:
    # Reuse the tested feature/H5AD materialization routines, but make all
    # generated provenance and active coordinate-frame names refer to this
    # final release rather than the old v8 release.
    M8.RELEASE = RELEASE
    M8.OUTPUT_FRAME = OUTPUT_FRAME
    M8.VERSION_TAG = VERSION_TAG
    M8.REMOVED_SECTION_IDS = REMOVED_SECTION_IDS
    M8.CUTOFF_MAX_SECTION_ID = CUTOFF_MAX_SECTION_ID
    M8.DIRECT_MOVING_SECTION = DIRECT_MOVING_SECTION


def _copy_transform_table(
    source_table_path: Path,
    output_table_path: Path,
    source_transform_dir: Path,
    output_transform_dir: Path,
    kept_ids: list[int],
    chain: DirectFinalChain,
) -> dict:
    table = pd.read_parquet(source_table_path).copy()
    table["section_id"] = pd.to_numeric(table["section_id"], errors="raise").astype(int)
    kept = table[table["section_id"].isin(kept_ids)].copy().sort_values("z_um", kind="stable")
    if len(kept) != len(kept_ids):
        raise ValueError("source H&E table does not contain all requested sections")
    output_transform_dir.mkdir(parents=True, exist_ok=True)
    matrices = {}
    chain_json = json.dumps(chain.metadata(), ensure_ascii=False, separators=(",", ":"))
    for row in kept.itertuples(index=False):
        sid = int(row.section_id)
        source_path = source_transform_dir / Path(str(row.section_transform_path)).name
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        arrays = M8._load_npz_arrays(source_path)
        old_matrix = np.asarray(arrays["matrix"], dtype=float)
        applied = sid >= DIRECT_MOVING_SECTION
        final_matrix = chain.coarse_affine @ old_matrix if applied else old_matrix
        # ``matrix`` is the active coarse coordinate transform.  It is baked
        # in-place; the old value is retained only for audit, never required
        # to reconstruct the final coordinates.
        arrays["matrix_before_final_he_alignment"] = old_matrix
        arrays["matrix"] = final_matrix
        arrays["final_he_alignment_release"] = np.asarray(RELEASE)
        arrays["final_he_alignment_coordinate_frame"] = np.asarray(OUTPUT_FRAME)
        arrays["final_he_alignment_applied"] = np.asarray([int(applied)], dtype=np.uint8)
        arrays["final_he_alignment_chain_json"] = np.asarray(chain_json)
        arrays["final_he_alignment_coarse_affine"] = np.asarray(chain.coarse_affine, dtype=float)
        output_path = output_transform_dir / source_path.name
        np.savez_compressed(output_path, **arrays)
        matrices[str(sid)] = final_matrix
        kept.loc[kept["section_id"] == sid, "section_transform_path"] = str(output_path.resolve())

    kept["coordinate_frame"] = OUTPUT_FRAME
    kept["final_he_alignment_release"] = RELEASE
    kept["final_he_alignment_applied"] = kept["section_id"] >= DIRECT_MOVING_SECTION
    kept["final_he_alignment_chain"] = chain_json
    kept["removed_section_ids"] = json.dumps(list(REMOVED_SECTION_IDS))
    kept.to_parquet(output_table_path, index=False)
    return {
        "source_table": str(source_table_path.resolve()),
        "output_table": str(output_table_path.resolve()),
        "source_section_count": int(len(table)),
        "output_section_count": int(len(kept)),
        "output_section_ids": [int(x) for x in kept["section_id"]],
        "applied_sections": [int(x) for x in kept.loc[kept["final_he_alignment_applied"], "section_id"]],
        "active_matrix_baked": True,
        "matrices": matrices,
    }


def materialize(device: str) -> dict:
    for path in (OUTPUT_PREP, OUTPUT_REG, OUTPUT_QC):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing path: {path}")
    _set_helper_release_constants()
    edge, edge_summary = _load_direct_edge(DIRECT_ROOT, device)
    chain = DirectFinalChain(edge, edge_summary, device=device)

    source_table = pd.read_parquet(SOURCE_PREP / "he_sections.parquet")
    source_table["section_id"] = pd.to_numeric(source_table["section_id"], errors="raise").astype(int)
    source_ids = sorted(int(x) for x in source_table["section_id"])
    kept_ids = [x for x in source_ids if x not in REMOVED_SECTION_IDS and x <= CUTOFF_MAX_SECTION_ID]
    if 53 not in kept_ids or 55 not in kept_ids or 54 in kept_ids:
        raise AssertionError(f"invalid final section set: {kept_ids}")

    OUTPUT_PREP.mkdir(parents=True, exist_ok=False)
    OUTPUT_REG.mkdir(parents=True, exist_ok=False)
    transform_result = _copy_transform_table(
        SOURCE_PREP / "he_sections.parquet",
        OUTPUT_PREP / "he_sections.parquet",
        SOURCE_PREP / "registration_transforms_final_manual_alignment_v6",
        OUTPUT_PREP / f"registration_transforms_{VERSION_TAG}",
        kept_ids,
        chain,
    )

    direct_copy = OUTPUT_PREP / "direct_stalign_53_55_final"
    shutil.copytree(DIRECT_ROOT, direct_copy)

    feature_result = M8._copy_feature_store(
        SOURCE_PREP / "uni2_features_final_manual_alignment_v6.h5",
        OUTPUT_PREP / f"uni2_features_{VERSION_TAG}.h5",
        [str(x) for x in kept_ids],
        chain,
        store_name="uni2",
    )
    nucleus_result = M8._copy_feature_store(
        SOURCE_PREP / "nucleus_path_final_manual_alignment_v6" / "nucleus_features.h5",
        OUTPUT_PREP / f"nucleus_path_{VERSION_TAG}" / "nucleus_features.h5",
        [str(x) for x in kept_ids],
        chain,
        store_name="nucleus",
    )
    h5ad_result = M8._materialize_h5ad(
        SOURCE_REG / "9957_g0__st_to_final_manual_alignment_v6.h5ad",
        OUTPUT_REG / f"9957_g0__st_to_{VERSION_TAG}.h5ad",
        OUTPUT_PREP / "anchors",
        OUTPUT_PREP / f"registration_transforms_{VERSION_TAG}",
        chain,
        chain.metadata(),
    )

    manifest = {
        "status": "materialized_final_he_coordinate_release",
        "release": RELEASE,
        "coordinate_frame": OUTPUT_FRAME,
        "source_prep": str(SOURCE_PREP.resolve()),
        "source_reg": str(SOURCE_REG.resolve()),
        "direct_stalign_root": str(DIRECT_ROOT.resolve()),
        "direct_stalign_status": edge.status,
        "removed_section_ids": list(REMOVED_SECTION_IDS),
        "source_section_ids": source_ids,
        "output_section_ids": kept_ids,
        "section_54_removed": True,
        "direct_pair": {
            "fixed_section": 53,
            "moving_section": 55,
            "applied_to_sections": ">=55",
            "active_map_baked": True,
            "affine": np.asarray(edge.affine, dtype=float).tolist(),
            "summary": edge_summary,
        },
        "transform_table": transform_result,
        "feature_store": feature_result,
        "nucleus_store": nucleus_result,
        "h5ad": h5ad_result,
        "st_alignment_note": "No new ST-to-H&E residual fit was performed; ST coordinates in the H5AD follow the same global frame propagation for sections >=55.",
        "cache_policy": "Rebuild UOT and morphology-path caches from the final feature stores and final H5AD.",
        "qc_output": str(OUTPUT_QC.resolve()),
    }
    for path in (OUTPUT_PREP / "final_he_alignment_manifest.json", OUTPUT_REG / "final_he_alignment_manifest.json"):
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = materialize(args.device)
    print(json.dumps({
        "status": result["status"],
        "release": result["release"],
        "output_section_count": len(result["output_section_ids"]),
        "section_54_removed": result["section_54_removed"],
        "prep": str(OUTPUT_PREP),
        "reg": str(OUTPUT_REG),
        "h5ad": result["h5ad"]["output"],
        "uni2": result["feature_store"]["output_store"],
        "nucleus": result["nucleus_store"]["output_store"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
