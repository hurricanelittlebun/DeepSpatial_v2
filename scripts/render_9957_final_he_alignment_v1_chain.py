"""Render per-section H&E QC for the embedded v1-chain release.

This is a thin, read-only wrapper around the validated final H&E renderer.  It
changes only the input/output release constants so that the figures are made
from the embedded v1-chain table and the approved 53->55 STalign edge.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "scripts" / "render_9957_final_he_alignment.py"


def _load_base():
    spec = importlib.util.spec_from_file_location("_render_9957_final_he_alignment", BASE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load renderer from {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    renderer = _load_base()
    group = ROOT / "data" / "9957_g0"
    prep = group / "reconstruction_prep_final_he_53_55_v1_chain_v1"
    candidate = group / "stalign_realign_53_55_v1_coordinate_chain_candidate_v2_coarse_init"

    renderer.FINAL_PREP = prep
    renderer.FINAL_TABLE = prep / "he_sections.parquet"
    renderer.FINAL_TRANSFORM_DIR = prep / "registration_transforms_final_he_53_55_v1_chain_v1"
    renderer.DIRECT_ROOT = candidate
    renderer.OUTPUT_ROOT = group / "qc" / "he_alignment_final_he_53_55_v1_chain_v1"
    renderer.REMOVED_SECTION_IDS = {54}
    renderer.DIRECT_START_SECTION = 55

    # Replace the old release-specific metadata so the QC manifest cannot be
    # mistaken for the earlier diagonal_v1 release.
    def metadata(self):
        return {
            "coordinate_frame": "9957__g0_registered_final_he_53_55_v1_chain_v1",
            "mapping_order": [
                "raw H&E crop local physical XY",
                "base saved preorientation + STalign edge chain",
                "v1 -> v5 frame affine",
                "v5 -> v6 frame affine",
                "approved direct 53 -> 55 STalign map for section >=55",
            ],
            "base_registration_root": str(renderer.BASE_REGISTRATION_ROOT.resolve()),
            "direct_summary": self.direct_summary,
        }

    renderer.FinalCoordinateMapper.metadata = metadata
    result = renderer.render(scale_um=6.0, margin_um=80.0, device="cpu")
    print(json.dumps({
        "status": result["status"],
        "coordinate_frame": result["coordinate_frame"],
        "section_count": result["section_count"],
        "adjacent_pair_count": result["adjacent_pair_count"],
        "output_root": result["outputs"]["sections"].rsplit("/sections", 1)[0],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
