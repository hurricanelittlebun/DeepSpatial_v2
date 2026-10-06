"""Render H&E red/green overlays in the reviewed 9957 postrotate-v9 frame.

This deliberately uses the reviewed chain from the v9 materialization:

    raw H&E local coordinates
      -> v9 preorientation
      -> v9-to-v6 bridge
      -> horizontal canvas flip
      -> direct STalign 53->55 map
      -> counter-clockwise 90-degree canvas rotation

No feature-support residual affine is applied.  This is the image-space
counterpart of ``reconstruction_prep_v9_postrotate_ccw90_cutoff91`` and is
kept separate from the later v10 residual release.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
GROUP = ROOT / "data" / "9957_g0"
V9_PREP = GROUP / "reconstruction_prep_v9_postrotate_ccw90_cutoff91"
OUTPUT_ROOT = (
    GROUP
    / "qc"
    / "he_alignment_all_sections_v6_postrotate_ccw90_cutoff91"
    / "he_redgreen_overlays_authoritative_v1"
)


def _load_renderer():
    path = ROOT / "scripts" / "render_9957_v10_he_redgreen_full_chain.py"
    spec = importlib.util.spec_from_file_location("_he_redgreen_renderer", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load renderer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    renderer = _load_renderer()
    renderer.PREP = V9_PREP
    renderer.TABLE_PATH = V9_PREP / "he_sections.parquet"
    renderer.DIRECT_ROOT = V9_PREP / "stalign_final_53_55" / "direct_v3"
    renderer.OUTPUT_ROOT = OUTPUT_ROOT

    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUTPUT_ROOT}")

    table = pd.read_parquet(renderer.TABLE_PATH).sort_values("z_um", kind="stable")
    rows = {int(row.section_id): row for row in table.itertuples(index=False)}
    pair_ids = [(51, 53), (53, 55)]
    missing = sorted({sid for pair in pair_ids for sid in pair if sid not in rows})
    if missing:
        raise KeyError(f"sections missing from v9 table: {missing}")

    chain = renderer._load_chain("cuda:0")
    identity_residual = np.eye(3, dtype=float)
    sections = {
        sid: renderer._load_source(rows[sid])
        for sid in sorted({sid for pair in pair_ids for sid in pair})
    }

    records = []
    output_dir = OUTPUT_ROOT / "key_pairs"
    for first_sid, second_sid in pair_ids:
        path = output_dir / (
            f"pair-{first_sid:03d}-{second_sid:03d}__he_redgreen_postrotate_authoritative.png"
        )
        record = renderer._save_pair(
            sections[first_sid],
            sections[second_sid],
            chain,
            identity_residual,
            path,
            2.0,
        )
        record["transform_semantics"] = (
            "raw H&E -> v9 preorientation -> v9-to-v6 bridge -> "
            "horizontal flip -> direct STalign -> CCW90; no v10 residual"
        )
        records.append(record)

    materializer = renderer._load_materializer()
    chain_checks = materializer._assert_chain_artifacts(chain)
    manifest = {
        "status": "complete",
        "release": "9957_postrotate_ccw90_cutoff91_v9_authoritative_redgreen_v1",
        "coordinate_frame": "9957__g0_registered_postrotate_ccw90_v9_cutoff91",
        "source_prep": str(V9_PREP.resolve()),
        "source_table": str(renderer.TABLE_PATH.resolve()),
        "residual_affine_applied": False,
        "postprocessing": [
            "horizontal_canvas_flip_v3",
            "direct_stalign_53_to_55_v3",
            "counterclockwise_90_canvas_rotation_v1",
        ],
        "chain_checks": chain_checks,
        "overlay_semantics": "red=first H&E optical density; green=second H&E optical density; yellow=overlap",
        "pairs": records,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
