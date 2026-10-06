"""Render continuous red/green H&E support contours for mixed 9957/g0 v11."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from deepspatial.histology.feature_store import FeatureStore


ROOT = Path(__file__).resolve().parents[1]
GROUP = ROOT / "data" / "9957_g0"
PREP = GROUP / "reconstruction_prep_v11_mixed_v10_pre53_v9_post55"
STORE = PREP / "uni2_features_mixed_v11.h5"
TABLE = PREP / "he_sections.parquet"
OUTPUT = GROUP / "qc" / "he_alignment_all_sections_v8_mixed_v10_pre53_v9_post55" / "he_redgreen_contours"

RED = "#e41a1c"
GREEN = "#00a651"


def _load_records() -> tuple[list[dict], str]:
    table = pd.read_parquet(TABLE).copy()
    table["section_id"] = pd.to_numeric(table["section_id"], errors="raise").astype(int)
    table_by_id = {int(row.section_id): row for row in table.itertuples(index=False)}
    store = FeatureStore(STORE, cache_mb=128)
    records = []
    with h5py.File(STORE, "r") as handle:
        for sid_text in store.sections:
            metadata = store.metadata[sid_text]
            key = store._key(sid_text)
            valid = np.asarray(handle["sections"][key]["valid"][:], dtype=bool)
            sid = int(sid_text)
            row = table_by_id[sid]
            records.append(
                {
                    "section_id": sid,
                    "z_um": float(metadata["z_um"]),
                    "origin_um": np.asarray(metadata["origin_um"], dtype=float),
                    "spacing_um": np.asarray(metadata["spacing_um"], dtype=float),
                    "valid": valid,
                    "source_release": str(getattr(row, "mixed_source_release", "unknown")),
                }
            )
    frames = {str(metadata["coordinate_frame"]) for metadata in store.metadata.values()}
    if len(frames) != 1:
        raise ValueError(f"Feature store has multiple coordinate frames: {sorted(frames)}")
    records.sort(key=lambda item: item["z_um"])
    return records, next(iter(frames))


def _grid_extent(record: dict) -> tuple[float, float, float, float]:
    rows, cols = np.nonzero(record["valid"])
    if len(rows) == 0:
        raise ValueError(f"section {record['section_id']} has no valid support")
    origin = record["origin_um"]
    spacing = record["spacing_um"]
    x = origin[0] + cols * spacing[0]
    y = origin[1] + rows * spacing[1]
    return float(x.min()), float(x.max()), float(y.min()), float(y.max())


def _bounds(records: list[dict]) -> tuple[float, float, float, float]:
    extents = np.asarray([_grid_extent(record) for record in records], dtype=float)
    xmin, xmax = float(extents[:, 0].min()), float(extents[:, 1].max())
    ymin, ymax = float(extents[:, 2].min()), float(extents[:, 3].max())
    pad = 0.04 * max(xmax - xmin, ymax - ymin, 1.0)
    return xmin - pad, xmax + pad, ymin - pad, ymax + pad


def _configure(ax, bounds):
    xmin, xmax, ymin, ymax = bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("global X (µm)")
    ax.set_ylabel("global Y (µm)")
    ax.grid(False)


def _draw_contour(ax, record: dict, color: str, linewidth: float = 1.4):
    valid = record["valid"].astype(np.float32)
    height, width = valid.shape
    origin = record["origin_um"]
    spacing = record["spacing_um"]
    x = origin[0] + np.arange(width, dtype=float) * spacing[0]
    y = origin[1] + np.arange(height, dtype=float) * spacing[1]
    ax.contour(x, y, valid, levels=[0.5], colors=[color], linewidths=linewidth)


def _render_pairs(records: list[dict], output: Path, bounds: tuple[float, float, float, float]) -> list[Path]:
    pair_dir = output / "pairs"
    pair_dir.mkdir(parents=True, exist_ok=False)
    outputs = []
    for left, right in zip(records[:-1], records[1:]):
        fig, ax = plt.subplots(figsize=(8, 7), dpi=170, constrained_layout=True)
        _draw_contour(ax, left, RED)
        _draw_contour(ax, right, GREEN)
        _configure(ax, bounds)
        ax.set_title(
            f"9957/g0 H&E red-green contour alignment | {left['section_id']:03d} → {right['section_id']:03d}\n"
            f"red=lower Z ({left['z_um']:.1f} µm, {left['source_release']}); "
            f"green=higher Z ({right['z_um']:.1f} µm, {right['source_release']})",
            fontsize=11,
        )
        path = pair_dir / f"pair-{left['section_id']:03d}-{right['section_id']:03d}__he_redgreen_contours.png"
        fig.savefig(path, dpi=170, facecolor="white")
        plt.close(fig)
        outputs.append(path)
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
            ax.set_title(path.stem.replace("__he_redgreen_contours", ""), fontsize=8)
        for ax in axes.flat[len(batch) :]:
            ax.axis("off")
        page = start // page_size + 1
        fig.suptitle(f"9957/g0 mixed v11 H&E red-green contour alignment | page {page}", fontsize=15)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        path = contact_dir / f"he_redgreen_contours_contact_sheet_page-{page:02d}.png"
        fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        outputs.append(path)
    return outputs


def render(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing directory: {output}")
    output.mkdir(parents=True)
    records, frame = _load_records()
    bounds = _bounds(records)
    pair_paths = _render_pairs(records, output, bounds)
    contact_paths = _render_contacts(pair_paths, output)
    section_ids = [int(record["section_id"]) for record in records]
    manifest = {
        "status": "complete",
        "coordinate_frame": frame,
        "feature_store": str(STORE.resolve()),
        "he_table": str(TABLE.resolve()),
        "actual_section_count": len(records),
        "adjacent_pair_count": len(pair_paths),
        "actual_sections": section_ids,
        "removed_or_absent_section_ids_1_to_99": [sid for sid in range(1, 100) if sid not in section_ids],
        "overlay_semantics": "continuous contour lines; red=lower-Z section, green=higher-Z section; no per-pixel or point scatter rendering",
        "pair_images": [str(path.resolve()) for path in pair_paths],
        "contact_sheets": [str(path.resolve()) for path in contact_paths],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
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
