"""Render red/green adjacent H&E alignment overlays for mixed 9957/g0 v11."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pandas as pd

from render_9957_all_he_alignment_qc import (
    _bounds,
    _configure,
    _draw_boundary,
    _load_supports,
)


ROOT = Path(__file__).resolve().parents[1]
GROUP = ROOT / "data" / "9957_g0"
PREP = GROUP / "reconstruction_prep_v11_mixed_v10_pre53_v9_post55"
STORE = PREP / "uni2_features_mixed_v11.h5"
TABLE = PREP / "he_sections.parquet"
OUTPUT = GROUP / "qc" / "he_alignment_all_sections_v8_mixed_v10_pre53_v9_post55" / "he_redgreen_pairs"

RED = "#e41a1c"
GREEN = "#00a651"


def _render_pairs(records: list[dict], output: Path, bounds: tuple[float, float, float, float]) -> list[Path]:
    pair_dir = output / "pairs"
    pair_dir.mkdir(parents=True, exist_ok=False)
    outputs = []
    for left, right in zip(records[:-1], records[1:]):
        fig, ax = plt.subplots(figsize=(8, 7), dpi=160, constrained_layout=True)
        _draw_boundary(ax, left, RED, size=2.8, alpha=1.0)
        _draw_boundary(ax, right, GREEN, size=2.8, alpha=1.0)
        _configure(ax, bounds)
        ax.set_title(
            f"9957/g0 H&E red-green alignment | sections {left['section_id']:03d} → {right['section_id']:03d}\n"
            f"red=lower Z ({left['z_um']:.1f} µm); green=higher Z ({right['z_um']:.1f} µm)",
            fontsize=12,
        )
        output_path = pair_dir / (
            f"pair-{left['section_id']:03d}-{right['section_id']:03d}__he_redgreen.png"
        )
        fig.savefig(output_path, dpi=160, facecolor="white")
        plt.close(fig)
        outputs.append(output_path)
    return outputs


def _render_contacts(paths: list[Path], output: Path, page_size: int = 10) -> list[Path]:
    contact_dir = output / "contact_sheets"
    contact_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for start in range(0, len(paths), page_size):
        batch = paths[start : start + page_size]
        fig, axes = plt.subplots(2, 5, figsize=(18, 7.5), dpi=150, squeeze=False)
        for ax, path in zip(axes.flat, batch):
            ax.imshow(plt.imread(path))
            ax.axis("off")
            ax.set_title(path.stem.replace("__he_redgreen", ""), fontsize=8)
        for ax in axes.flat[len(batch) :]:
            ax.axis("off")
        page = start // page_size + 1
        fig.suptitle(f"9957/g0 mixed v11 H&E red-green adjacent alignment | page {page}", fontsize=15)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        path = contact_dir / f"he_redgreen_contact_sheet_page-{page:02d}.png"
        fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        outputs.append(path)
    return outputs


def render(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing directory: {output}")
    output.mkdir(parents=True)
    records, frame = _load_supports(STORE, TABLE)
    bounds = _bounds(records)
    pair_paths = _render_pairs(records, output, bounds)
    contact_paths = _render_contacts(pair_paths, output)
    manifest = {
        "status": "complete",
        "coordinate_frame": frame,
        "feature_store": str(STORE.resolve()),
        "he_table": str(TABLE.resolve()),
        "actual_section_count": len(records),
        "adjacent_pair_count": len(pair_paths),
        "actual_sections": [int(record["section_id"]) for record in records],
        "missing_section_ids_1_to_99": [
            sid for sid in range(1, 100) if sid not in {int(record["section_id"]) for record in records}
        ],
        "overlay_semantics": "red=lower-Z section, green=higher-Z section; point boundaries from the registered H&E/UNI2 feature-grid valid masks",
        "pair_images": [str(path.resolve()) for path in pair_paths],
        "contact_sheets": [str(path.resolve()) for path in contact_paths],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    pd.DataFrame(
        [{"section_id": record["section_id"], "z_um": record["z_um"]} for record in records]
    ).to_csv(output / "sections_present.csv", index=False)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = render(args.output)
    print(json.dumps({
        "status": result["status"],
        "output": str(args.output.resolve()),
        "actual_section_count": result["actual_section_count"],
        "adjacent_pair_count": result["adjacent_pair_count"],
        "contact_sheets": result["contact_sheets"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
