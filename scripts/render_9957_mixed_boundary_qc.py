"""Render the boundary QC for the mixed v11 9957/g0 release.

The two boundary pairs intentionally use the reviewed, pair-specific
rendering semantics:

* 51 -> 53: the already reviewed v10 authoritative full-chain overlay;
* 53 -> 55: the already reviewed v9 post-rotation authoritative overlay.

The images are copied from the reviewed releases so that this boundary QC
cannot change because of a renderer implementation detail.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
GROUP = ROOT / "data" / "9957_g0"
MIXED_PREP = GROUP / "reconstruction_prep_v11_mixed_v10_pre53_v9_post55"
V9_PREP = GROUP / "reconstruction_prep_v9_postrotate_ccw90_cutoff91"
V10_AUTHORITATIVE = (
    GROUP
    / "qc"
    / "he_alignment_all_sections_v7_feature_support_residual"
    / "he_redgreen_overlays_full_chain_v1"
    / "key_pairs"
    / "pair-051-053__he_redgreen_full_chain.png"
)
V9_AUTHORITATIVE = (
    GROUP
    / "qc"
    / "he_alignment_all_sections_v6_postrotate_ccw90_cutoff91"
    / "he_redgreen_overlays_authoritative_v1"
    / "key_pairs"
    / "pair-053-055__he_redgreen_postrotate_authoritative.png"
)
OUTPUT = GROUP / "qc" / "he_alignment_all_sections_v8_mixed_v10_pre53_v9_post55"


def render() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    table_path = MIXED_PREP / "he_sections.parquet"
    table = pd.read_parquet(table_path).sort_values("z_um", kind="stable")
    rows = {int(row.section_id): row for row in table.itertuples(index=False)}
    if 51 not in rows or 53 not in rows or 55 not in rows:
        raise KeyError("mixed table must contain sections 51, 53 and 55")

    if not V10_AUTHORITATIVE.is_file():
        raise FileNotFoundError(V10_AUTHORITATIVE)
    first_path = OUTPUT / "key_pairs" / "pair-051-053__he_redgreen_mixed_v10_authoritative.png"
    first_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(V10_AUTHORITATIVE, first_path)
    with Image.open(first_path) as image_check:
        first_shape = [int(image_check.height), int(image_check.width)]
    first_record = {
        "first_section": 51,
        "second_section": 53,
        "first_z_um": float(rows[51].z_um),
        "second_z_um": float(rows[53].z_um),
        "shape_yx": first_shape,
        "path": str(first_path.resolve()),
        "pair": "51-53",
        "source_release": "v10",
        "source_overlay": str(V10_AUTHORITATIVE.resolve()),
        "render_semantics": "reviewed v10 full-chain authoritative overlay; red=51, green=53, yellow=overlap",
    }

    if not V9_AUTHORITATIVE.is_file():
        raise FileNotFoundError(V9_AUTHORITATIVE)
    second_path = OUTPUT / "key_pairs" / "pair-053-055__he_redgreen_mixed_v9_authoritative.png"
    shutil.copy2(V9_AUTHORITATIVE, second_path)
    with Image.open(second_path) as image_check:
        second_shape = [int(image_check.height), int(image_check.width)]
    second_record = {
        "first_section": 53,
        "second_section": 55,
        "first_z_um": float(rows[53].z_um),
        "second_z_um": float(rows[55].z_um),
        "shape_yx": second_shape,
        "path": str(second_path.resolve()),
        "source_release": "v9",
        "source_overlay": str(V9_AUTHORITATIVE.resolve()),
        "render_semantics": "reviewed v9 authoritative full chain; red=53, green=55, yellow=overlap",
    }

    manifest = {
        "status": "complete",
        "release": "9957_mixed_v10_pre53_v9_post55_v11",
        "coordinate_frame": "9957__g0_registered_mixed_v10_pre53_v9_post55_v11",
        "source_table": str(table_path.resolve()),
        "pairs": [first_record, second_record],
        "policy": {
            "pair_51_53": "v10 reviewed full-chain authoritative overlay",
            "pair_53_55": "v9 reviewed authoritative overlay",
            "sections_ge_55": "v9 coordinate branch",
        },
        "note": "The pair images are the already reviewed v10/v9 authoritative overlays, copied into the mixed v11 QC directory so the data release and visual evidence use the same pair policy.",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    print(json.dumps(render(), indent=2, ensure_ascii=False))
