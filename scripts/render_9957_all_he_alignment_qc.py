"""Render every registered H&E section and every adjacent H&E overlay for 9957/g0."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepspatial.histology.feature_store import FeatureStore  # noqa: E402
from render_9957_coordinate_repair_qc import feature_grid_boundary  # noqa: E402


DEFAULT_GROUP = ROOT / "data" / "9957_g0"
DEFAULT_HE_TABLE = (
    DEFAULT_GROUP / "reconstruction_prep_v5_coordinate_repair" / "he_sections.parquet"
)
DEFAULT_FEATURE_STORE = (
    DEFAULT_GROUP
    / "reconstruction_prep_v5_coordinate_repair"
    / "uni2_features_global_v4_repaired.h5"
)
DEFAULT_OUTPUT = DEFAULT_GROUP / "qc" / "he_alignment_all_sections_v2"

RED = "#e41a1c"
BLUE = "#0057b7"
GREEN = "#00a651"
GRAY = "#9ca3af"


def _load_supports(feature_store_path: Path, he_table_path: Path) -> tuple[list[dict], str]:
    table = pd.read_parquet(he_table_path).copy()
    table["section_id"] = pd.to_numeric(table["section_id"], errors="raise").astype(int)
    table_by_id = {int(row.section_id): row for row in table.itertuples(index=False)}
    store = FeatureStore(feature_store_path, cache_mb=256)
    records = []
    with h5py.File(feature_store_path, "r") as handle:
        for sid_text in store.sections:
            sid = int(sid_text)
            metadata = store.metadata[sid_text]
            valid = handle["sections"][store._key(sid_text)]["valid"][:]
            boundary = feature_grid_boundary(
                valid,
                origin_um=metadata["origin_um"],
                spacing_um=metadata["spacing_um"],
                max_points=6000,
            )
            row = table_by_id.get(sid)
            records.append(
                {
                    "section_id": sid,
                    "z_um": float(metadata["z_um"]),
                    "boundary": boundary,
                    "registration_status": str(getattr(row, "registration_status", "NA")),
                    "registration_confidence": float(
                        getattr(row, "registration_confidence", np.nan)
                    ),
                }
            )
    frame_values = {str(metadata["coordinate_frame"]) for metadata in store.metadata.values()}
    if len(frame_values) != 1:
        raise ValueError(f"Feature store has multiple coordinate frames: {sorted(frame_values)}")
    return sorted(records, key=lambda item: item["z_um"]), next(iter(frame_values))


def _bounds(records: list[dict]) -> tuple[float, float, float, float]:
    points = [record["boundary"] for record in records if len(record["boundary"])]
    if not points:
        raise ValueError("No valid H&E/UNI2 support boundaries found")
    merged = np.concatenate(points, axis=0)
    xmin, ymin = merged.min(axis=0)
    xmax, ymax = merged.max(axis=0)
    pad = 0.04 * max(float(xmax - xmin), float(ymax - ymin), 1.0)
    return float(xmin - pad), float(xmax + pad), float(ymin - pad), float(ymax + pad)


def _configure(ax, bounds, *, compact=False):
    xmin, xmax, ymin, ymax = bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)
    ax.set_aspect("equal", adjustable="box")
    if not compact:
        ax.set_xlabel("global X (µm)")
        ax.set_ylabel("global Y (µm)")
        ax.grid(False)
    else:
        ax.set_xticks([])
        ax.set_yticks([])


def _draw_boundary(ax, record, color, *, size=2.3, alpha=1.0):
    points = record["boundary"]
    if len(points):
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=size,
            color=color,
            alpha=alpha,
            linewidths=0,
            rasterized=True,
        )


def _section_title(record, previous, following):
    conf = record["registration_confidence"]
    conf_text = "NA" if not np.isfinite(conf) else f"{conf:.3f}"
    prev_text = "—" if previous is None else str(previous["section_id"])
    next_text = "—" if following is None else str(following["section_id"])
    return (
        f"H&E section {record['section_id']:03d} | z={record['z_um']:.1f} µm | "
        f"prev={prev_text}, next={next_text}\n"
        f"red=current; blue=previous; green=next | confidence={conf_text}"
    )


def _render_section_images(records, output_dir, bounds):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    section_dir = output_dir / "sections"
    section_dir.mkdir(parents=True, exist_ok=False)
    outputs = []
    for index, record in enumerate(records):
        previous = records[index - 1] if index else None
        following = records[index + 1] if index + 1 < len(records) else None
        fig, ax = plt.subplots(figsize=(8, 7), dpi=150, constrained_layout=True)
        if previous is not None:
            _draw_boundary(ax, previous, BLUE, size=2.0)
        if following is not None:
            _draw_boundary(ax, following, GREEN, size=2.0)
        _draw_boundary(ax, record, RED, size=2.8)
        _configure(ax, bounds)
        ax.set_title(_section_title(record, previous, following), fontsize=12)
        output = section_dir / f"section-{record['section_id']:03d}__he_global_alignment.png"
        fig.savefig(output, dpi=150, facecolor="white")
        plt.close(fig)
        outputs.append(output)
    return outputs


def _render_pair_images(records, output_dir, bounds):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    pair_dir = output_dir / "adjacent_pairs"
    pair_dir.mkdir(parents=True, exist_ok=False)
    outputs = []
    for left, right in zip(records[:-1], records[1:]):
        fig, ax = plt.subplots(figsize=(8, 7), dpi=150, constrained_layout=True)
        _draw_boundary(ax, left, RED, size=2.8)
        _draw_boundary(ax, right, BLUE, size=2.8)
        _configure(ax, bounds)
        ax.set_title(
            f"Adjacent H&E alignment: {left['section_id']:03d} (red, z={left['z_um']:.1f}) "
            f"vs {right['section_id']:03d} (blue, z={right['z_um']:.1f})",
            fontsize=12,
        )
        output = pair_dir / (
            f"pair-{left['section_id']:03d}-{right['section_id']:03d}__he_alignment.png"
        )
        fig.savefig(output, dpi=150, facecolor="white")
        plt.close(fig)
        outputs.append(output)
    return outputs


def _render_contact_pages(records, output_dir, bounds, *, kind, page_size=10):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    source_dir = output_dir / ("sections" if kind == "sections" else "adjacent_pairs")
    paths = sorted(source_dir.glob("*.png"))
    contact_dir = output_dir / "contact_sheets"
    contact_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for page_start in range(0, len(paths), page_size):
        batch = paths[page_start : page_start + page_size]
        ncols = 5
        nrows = 2
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(18, 7.5),
            dpi=150,
            squeeze=False,
        )
        for ax, path in zip(axes.flat, batch):
            with plt.rc_context({"figure.max_open_warning": 0}):
                image = plt.imread(path)
            ax.imshow(image)
            ax.axis("off")
            ax.set_title(path.stem.replace("__he_global_alignment", "").replace("__he_alignment", ""), fontsize=8)
        for ax in axes.flat[len(batch) :]:
            ax.axis("off")
        page = page_start // page_size + 1
        title = "Registered H&E sections" if kind == "sections" else "Adjacent registered H&E pairs"
        fig.suptitle(f"9957/g0 | {title} | page {page}", fontsize=15)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        output = contact_dir / f"{kind}_contact_sheet_page-{page:02d}.png"
        fig.savefig(output, dpi=150, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        outputs.append(output)
    return outputs


def _render_all_overview(records, output_dir, bounds):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 11), dpi=180, constrained_layout=True)
    colors = plt.get_cmap("turbo")(np.linspace(0.02, 0.98, len(records)))
    for color, record in zip(colors, records):
        _draw_boundary(ax, record, color, size=1.2)
    _configure(ax, bounds)
    ax.set_title(
        "9957/g0 all registered H&E/UNI2 supports\n"
        "color progresses from low to high Z; points are support boundaries",
        fontsize=14,
    )
    output = output_dir / "all_registered_he_sections_overview.png"
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)
    return output


def render(he_table: Path, feature_store: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing directory: {output_dir}")
    output_dir.mkdir(parents=True)
    records, frame = _load_supports(feature_store, he_table)
    bounds = _bounds(records)
    section_paths = _render_section_images(records, output_dir, bounds)
    pair_paths = _render_pair_images(records, output_dir, bounds)
    section_contacts = _render_contact_pages(records, output_dir, bounds, kind="sections")
    pair_contacts = _render_contact_pages(records, output_dir, bounds, kind="pairs")
    overview = _render_all_overview(records, output_dir, bounds)
    present = [int(record["section_id"]) for record in records]
    missing = [section_id for section_id in range(1, 100) if section_id not in present]
    pd.DataFrame(records)[
        ["section_id", "z_um", "registration_status", "registration_confidence"]
    ].to_csv(output_dir / "sections_present.csv", index=False)
    (output_dir / "sections_missing_1_to_99.txt").write_text(
        " ".join(map(str, missing)) + "\n", encoding="utf-8"
    )
    manifest = {
        "status": "complete",
        "coordinate_frame": frame,
        "he_table": str(he_table.resolve()),
        "feature_store": str(feature_store.resolve()),
        "actual_section_count": len(records),
        "actual_sections": present,
        "missing_section_ids_1_to_99": missing,
        "global_bounds_um": bounds,
        "section_images": [str(path.resolve()) for path in section_paths],
        "adjacent_pair_images": [str(path.resolve()) for path in pair_paths],
        "section_contact_sheets": [str(path.resolve()) for path in section_contacts],
        "pair_contact_sheets": [str(path.resolve()) for path in pair_contacts],
        "all_sections_overview": str(overview.resolve()),
        "semantics": (
            "Each plot uses the valid registered H&E/UNI2 feature-grid support "
            "in the same repaired global frame used by reconstruction; it does "
            "not refit registration or transform a raw mask through an incomplete chain."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--he-table", type=Path, default=DEFAULT_HE_TABLE)
    parser.add_argument("--feature-store", type=Path, default=DEFAULT_FEATURE_STORE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = render(args.he_table, args.feature_store, args.output)
    print(json.dumps({
        "status": result["status"],
        "output": str(args.output.resolve()),
        "actual_section_count": result["actual_section_count"],
        "actual_sections": result["actual_sections"],
        "missing_section_ids_1_to_99": result["missing_section_ids_1_to_99"],
        "section_images": len(result["section_images"]),
        "adjacent_pair_images": len(result["adjacent_pair_images"]),
        "section_contact_sheets": result["section_contact_sheets"],
        "pair_contact_sheets": result["pair_contact_sheets"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
