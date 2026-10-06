"""Promote the reviewed v1-coordinate-chain 53->55 candidate.

The candidate is promoted into a new release directory only after it has been
reviewed.  Existing releases are never overwritten.  Sections 1-53 retain the
v6 active coordinates, section 54 is excluded, and the candidate edge is
applied to sections 55-91.  The edge is baked into transform tables, H5AD,
UNI2 features, and nucleus features through the tested materializer helpers.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V2_SRC = ROOT.parent / "DeepSpatial_v2" / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(V2_SRC) not in sys.path:
    sys.path.insert(0, str(V2_SRC))

from deepspatial_v2.data_export.cell_level_alignment import load_stalign_map  # noqa: E402
from deepspatial_v2.serial_registration.stalign_types import EdgeAlignment  # noqa: E402


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MATERIALIZER = _load_module(
    ROOT / "scripts" / "materialize_9957_final_he_53_55_diagonal.py",
    "_materialize_9957_v1_chain_helpers",
)

GROUP = ROOT / "data" / "9957_g0"
CANDIDATE_ROOT = GROUP / "stalign_realign_53_55_v1_coordinate_chain_candidate_v2_coarse_init"
RELEASE = "9957_final_he_53_55_v1_chain_v1"
OUTPUT_FRAME = "9957__g0_registered_final_he_53_55_v1_chain_v1"
OUTPUT_PREP = GROUP / "reconstruction_prep_final_he_53_55_v1_chain_v1"
OUTPUT_REG = GROUP / "registration_correction_he_final_he_53_55_v1_chain_v1"
OUTPUT_QC = GROUP / "qc" / "he_alignment_final_he_53_55_v1_chain_v1"
VERSION_TAG = "final_he_53_55_v1_chain_v1"


def _load_candidate_edge(root: Path, device: str):
    summary = json.loads((root / "direct_stalign_summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "v1_coordinate_chain_stalign_candidate_only":
        raise ValueError(f"unexpected candidate status: {summary.get('status')}")
    if not summary.get("candidate_only") or summary.get("embedded_downstream"):
        raise ValueError("candidate is not an unembedded review candidate")
    if int(summary["fixed_section"]) != 53 or int(summary["moving_section"]) != 55:
        raise ValueError("candidate is not section 53 -> 55")
    pair = root / "pair-53-55"
    with np.load(pair / "direct_stalign_affine.npz", allow_pickle=False) as payload:
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
        status="accepted_final_v1_coordinate_chain_stalign",
    )
    return edge, summary


def materialize(device: str) -> dict:
    if not CANDIDATE_ROOT.is_dir():
        raise FileNotFoundError(CANDIDATE_ROOT)
    for path in (OUTPUT_PREP, OUTPUT_REG, OUTPUT_QC):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing path: {path}")

    MATERIALIZER.RELEASE = RELEASE
    MATERIALIZER.OUTPUT_FRAME = OUTPUT_FRAME
    MATERIALIZER.VERSION_TAG = VERSION_TAG
    MATERIALIZER.OUTPUT_PREP = OUTPUT_PREP
    MATERIALIZER.OUTPUT_REG = OUTPUT_REG
    MATERIALIZER.OUTPUT_QC = OUTPUT_QC
    MATERIALIZER.DIRECT_ROOT = CANDIDATE_ROOT
    MATERIALIZER._load_direct_edge = _load_candidate_edge

    result = MATERIALIZER.materialize(device)
    result["promotion_source"] = str(CANDIDATE_ROOT.resolve())
    result["promotion_status"] = "embedded_v1_coordinate_chain_candidate"
    for path in (
        OUTPUT_PREP / "final_he_alignment_manifest.json",
        OUTPUT_REG / "final_he_alignment_manifest.json",
    ):
        path.write_text(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
                default=lambda value: value.tolist() if isinstance(value, np.ndarray) else str(value),
            )
            + "\n",
            encoding="utf-8",
        )
    return result


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    result = materialize(args.device)
    print(
        json.dumps(
            {
                "status": result["status"],
                "promotion_status": result["promotion_status"],
                "release": result["release"],
                "output_section_count": len(result["output_section_ids"]),
                "section_54_removed": result["section_54_removed"],
                "prep": str(OUTPUT_PREP),
                "reg": str(OUTPUT_REG),
                "h5ad": result["h5ad"]["output"],
                "uni2": result["feature_store"]["output_store"],
                "nucleus": result["nucleus_store"]["output_store"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
