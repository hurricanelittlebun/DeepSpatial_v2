"""Materialize 9957/g0 using exactly the reviewed postrotate_ccw90 alignment.

This is the release requested after review of
``stalign_direct_53_55_postrotate_ccw90_v1``.  It deliberately does not load
or apply the later finetune artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

import materialize_9957_latest_alignment_cutoff91 as implementation


ROOT = Path(__file__).resolve().parents[1]
GROUP = ROOT / "data" / "9957_g0"

implementation.RELEASE = "9957_postrotate_ccw90_cutoff91_v9"
implementation.OUTPUT_FRAME = (
    "9957__g0_registered_postrotate_ccw90_v9_cutoff91"
)
implementation.VERSION_TAG = "v9"


def main() -> int:
    output_prep = GROUP / "reconstruction_prep_v9_postrotate_ccw90_cutoff91"
    output_reg = GROUP / "registration_correction_he_v9_postrotate_ccw90_cutoff91"
    output_qc = GROUP / "qc" / "he_alignment_all_sections_v6_postrotate_ccw90_cutoff91"
    result = implementation.materialize(
        source_prep=(GROUP / "reconstruction_prep_v6_final_manual_alignment").resolve(),
        source_reg=(GROUP / "registration_correction_he_v6_final_manual_alignment").resolve(),
        output_prep=output_prep.resolve(),
        output_reg=output_reg.resolve(),
        output_qc=output_qc.resolve(),
        direct_root=(GROUP / "stalign_direct_53_55_horizontal_canvas_flip_v3").resolve(),
        finetune_root=None,
        device="cuda:0",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "release": result["release"],
                "chain_order": result["chain"]["chain_order"],
                "output_section_count": len(result["output_section_ids"]),
                "section_91_kept": result["section_91_kept"],
                "section_54_removed": result["section_54_removed"],
                "latest_35_36_preserved": result["latest_35_36_preserved"],
                "h5ad": result["candidate"]["output"],
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
