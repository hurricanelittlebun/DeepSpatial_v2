"""Render raw H&E red/green overlays for every adjacent mixed-v11 section pair.

This is the same optical-density overlay style used by the reviewed key-pair
figures.  It is not a feature-grid scatter or contour plot.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
GROUP = ROOT / "data" / "9957_g0"
MIXED_PREP = GROUP / "reconstruction_prep_v11_mixed_v10_pre53_v9_post55"
V9_PREP = GROUP / "reconstruction_prep_v9_postrotate_ccw90_cutoff91"
OUTPUT = GROUP / "qc" / "he_alignment_all_sections_v8_mixed_v10_pre53_v9_post55" / "he_redgreen_raw_keypair_style"


def _load_renderer():
    path = ROOT / "scripts" / "render_9957_v10_he_redgreen_full_chain.py"
    spec = importlib.util.spec_from_file_location("_mixed_v11_raw_renderer", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load raw H&E renderer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def render(output: Path, *, device: str, scale_um: float) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing directory: {output}")
    renderer = _load_renderer()

    # Use the mixed v11 H&E table, but use the already reviewed v9 optical
    # transform chain for section >=55 and no v10 residual.  Sections <=53
    # are unchanged between the v9 and v10 pre-53 frames.
    renderer.PREP = MIXED_PREP
    renderer.TABLE_PATH = MIXED_PREP / "he_sections.parquet"
    renderer.DIRECT_ROOT = V9_PREP / "stalign_final_53_55" / "direct_v3"
    renderer.OUTPUT_ROOT = output

    table = pd.read_parquet(renderer.TABLE_PATH).sort_values("z_um", kind="stable").reset_index(drop=True)
    rows = {int(row.section_id): row for row in table.itertuples(index=False)}
    section_ids = [int(value) for value in table["section_id"]]
    pairs = list(zip(section_ids[:-1], section_ids[1:]))
    chain = renderer._load_chain(device)
    identity_residual = np.eye(3, dtype=float)

    # Load each raw section once.  The key-pair style overlay samples these
    # source H&E images through the complete reviewed v9 chain.
    sections = {sid: renderer._load_source(rows[sid]) for sid in section_ids}
    pair_dir = output / "key_pairs"
    pair_dir.mkdir(parents=True, exist_ok=False)
    records = []
    for index, (first_sid, second_sid) in enumerate(pairs, start=1):
        path = pair_dir / f"pair-{first_sid:03d}-{second_sid:03d}__he_redgreen_raw.png"
        record = renderer._save_pair(
            sections[first_sid],
            sections[second_sid],
            chain,
            identity_residual,
            path,
            scale_um,
        )
        record.update(
            {
                "pair_index": index,
                "first_source_release": str(rows[first_sid].mixed_source_release),
                "second_source_release": str(rows[second_sid].mixed_source_release),
                "transform_semantics": (
                    "raw H&E optical density -> v9 preorientation -> v9-to-v6 bridge -> "
                    "horizontal canvas flip -> direct STalign 53-to-55 map -> CCW90; "
                    "no v10 residual affine"
                ),
            }
        )
        records.append(record)
        if index % 10 == 0 or index == len(pairs):
            print(f"rendered {index}/{len(pairs)} adjacent raw H&E overlays", flush=True)

    materializer = renderer._load_materializer()
    chain_checks = materializer._assert_chain_artifacts(chain)
    manifest = {
        "status": "complete",
        "release": "9957_mixed_v10_pre53_v9_post55_v11_raw_he_redgreen_v1",
        "coordinate_frame": "9957__g0_registered_mixed_v10_pre53_v9_post55_v11",
        "source_prep": str(MIXED_PREP.resolve()),
        "source_table": str(renderer.TABLE_PATH.resolve()),
        "raw_overlay_style": "same optical-density H&E red/green/yellow-overlap style as reviewed key-pair figures",
        "scale_um_per_pixel": float(scale_um),
        "device": device,
        "section_count": len(section_ids),
        "adjacent_pair_count": len(records),
        "section_ids": section_ids,
        "removed_or_absent_section_ids_1_to_99": [sid for sid in range(1, 100) if sid not in set(section_ids)],
        "residual_affine_applied": False,
        "postprocessing": [
            "horizontal_canvas_flip_v3",
            "direct_stalign_53_to_55_v3",
            "counterclockwise_90_canvas_rotation_v1",
        ],
        "chain_checks": chain_checks,
        "overlay_semantics": "red=first/lower-Z H&E optical density; green=second/higher-Z H&E optical density; yellow=overlap",
        "pairs": records,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scale-um", type=float, default=2.0)
    args = parser.parse_args()
    result = render(args.output, device=args.device, scale_um=float(args.scale_um))
    print(json.dumps({
        "status": result["status"],
        "output": str(args.output.resolve()),
        "section_count": result["section_count"],
        "adjacent_pair_count": result["adjacent_pair_count"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
