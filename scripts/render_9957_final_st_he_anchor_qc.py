"""Render ST-versus-final-H&E overlays for the ten Xenium anchor sections.

This is diagnostic only.  It reads the final embedded H&E coordinate chain and
the final ST coordinates from the H5AD, but does not fit or write any new ST
transform.  Therefore red/cyan displacement in these figures is observable
alignment error, not an automatically hidden display fit.
"""

from __future__ import annotations

import importlib.util
import json
import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = ROOT / "scripts" / "render_9957_final_he_alignment.py"
GROUP = ROOT / "data" / "9957_g0"
PREP = GROUP / "reconstruction_prep_final_he_53_55_v1_chain_v1"
REG = GROUP / "registration_correction_he_final_he_53_55_v1_chain_v1"
CANDIDATE = GROUP / "stalign_realign_53_55_v1_coordinate_chain_candidate_v2_coarse_init"
OUTPUT = GROUP / "qc" / "st_he_alignment_final_he_53_55_v1_chain_v1"
ANCHORS = [1, 11, 21, 31, 41, 51, 61, 71, 81, 91]
SCALE_UM = 6.0
MARGIN_UM = 80.0
BOUNDARY_POINTS = 6000
MAX_PLOT_POINTS = 120000


def _load_renderer():
    spec = importlib.util.spec_from_file_location("_final_he_renderer_for_st_qc", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load renderer from {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.FINAL_TABLE = PREP / "he_sections.parquet"
    module.DIRECT_ROOT = CANDIDATE
    module.REMOVED_SECTION_IDS = {54}
    module.DIRECT_START_SECTION = 55
    return module


def _sample_points(points: np.ndarray, maximum: int) -> np.ndarray:
    if len(points) <= maximum:
        return points
    idx = np.linspace(0, len(points) - 1, maximum, dtype=np.int64)
    return points[idx]


def _classify_points(renderer, mapper, source, sid: int, points: np.ndarray):
    local = mapper.global_to_local(points, sid)
    sx, sy = source["pixel_size_um"]
    cols = np.rint((local[:, 0] - source["local_origin_um"][0]) / sx).astype(np.int64)
    rows = np.rint((local[:, 1] - source["local_origin_um"][1]) / sy).astype(np.int64)
    valid = (
        np.isfinite(local).all(axis=1)
        & (rows >= 0)
        & (rows < source["mask"].shape[0])
        & (cols >= 0)
        & (cols < source["mask"].shape[1])
    )
    inside = np.zeros(len(points), dtype=bool)
    inside[valid] = source["mask"][rows[valid], cols[valid]]
    return inside, valid


def _plot_anchor(renderer, mapper, row, source, st_points, output_path, st_coordinate_label):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    sid = int(row.section_id)
    boundary_px = renderer.BASE._boundary_points(source["mask"], BOUNDARY_POINTS)
    local_boundary = renderer.BASE._local_pixels_to_um(boundary_px, source)
    global_boundary = mapper.local_to_global(local_boundary, sid)
    low = global_boundary.min(axis=0) - MARGIN_UM
    high = global_boundary.max(axis=0) + MARGIN_UM
    shape = (
        max(1, int(np.ceil((high[1] - low[1]) / SCALE_UM))),
        max(1, int(np.ceil((high[0] - low[0]) / SCALE_UM))),
    )
    rgb, valid_he = renderer.BASE._sample_source_on_global_grid(
        mapper, source, sid, low, shape, SCALE_UM
    )
    inside, valid = _classify_points(renderer, mapper, source, sid, st_points)
    outside = valid & ~inside
    invalid = ~valid
    valid_count = int(valid.sum())
    inside_count = int((inside & valid).sum())
    inside_fraction = inside_count / valid_count if valid_count else float("nan")

    extent = [
        float(low[0]),
        float(low[0] + shape[1] * SCALE_UM),
        float(low[1] + shape[0] * SCALE_UM),
        float(low[1]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(17, 13), dpi=150, constrained_layout=True)
    ax0, ax1, ax2, ax3 = axes.flat

    for ax in (ax0, ax1, ax2):
        ax.imshow(rgb, extent=extent, origin="upper", interpolation="nearest")
        ax.set_aspect("equal")
        ax.set_xlabel("final global X (µm)")
        ax.set_ylabel("final global Y (µm)")

    shown = _sample_points(st_points, MAX_PLOT_POINTS)
    ax0.scatter(shown[:, 0], shown[:, 1], s=0.9, c="#00e5ff", alpha=0.42, linewidths=0, rasterized=True)
    ax0.set_title("Final H&E global frame + all ST centroids\ncyan = ST")

    if inside.any():
        ax1.scatter(st_points[inside, 0], st_points[inside, 1], s=0.9, c="#00f04b", alpha=0.42, linewidths=0, rasterized=True, label="ST in H&E mask")
    if outside.any():
        ax1.scatter(st_points[outside, 0], st_points[outside, 1], s=1.2, c="#ff1f35", alpha=0.60, linewidths=0, rasterized=True, label="ST outside mask")
    if invalid.any():
        ax1.scatter(st_points[invalid, 0], st_points[invalid, 1], s=1.4, c="#ff9d00", alpha=0.75, linewidths=0, rasterized=True, label="outside crop")
    ax1.legend(loc="upper right", framealpha=0.9, markerscale=5)
    ax1.set_title("ST against final H&E tissue mask\ngreen = in mask; red = outside")

    mask_rgb = np.zeros((*valid_he.shape, 3), dtype=np.uint8)
    mask_rgb[valid_he] = np.array([50, 165, 80], dtype=np.uint8)
    ax2.imshow(mask_rgb, extent=extent, origin="upper", interpolation="nearest")
    if inside.any():
        ax2.scatter(st_points[inside, 0], st_points[inside, 1], s=0.9, c="#00e5ff", alpha=0.42, linewidths=0, rasterized=True)
    if outside.any():
        ax2.scatter(st_points[outside, 0], st_points[outside, 1], s=1.2, c="#ff2035", alpha=0.65, linewidths=0, rasterized=True)
    ax2.set_aspect("equal")
    ax2.set_xlabel("final global X (µm)")
    ax2.set_ylabel("final global Y (µm)")
    ax2.set_title("Final H&E mask frame\ncyan = in mask; red = outside")

    ax3.axis("off")
    info = [
        "Coordinate frame: 9957__g0_registered_final_he_53_55_v1_chain_v1",
        f"section: {sid:03d} | z = {float(row.z_um):.1f} µm",
        f"ST cells: {len(st_points):,}",
        f"inside H&E mask: {inside_count:,}/{valid_count:,} ({inside_fraction:.3%})",
        f"outside H&E mask: {int(outside.sum()):,}",
        f"outside H&E crop: {int(invalid.sum()):,}",
        "",
        "ST coordinate source:",
        st_coordinate_label,
        "",
        "No new ST transform was fitted.",
        "The overlay uses the already embedded final global coordinates.",
    ]
    ax3.text(0.03, 0.97, "\n".join(info), va="top", ha="left", fontsize=12, family="DejaVu Sans Mono")
    ax3.set_title("QC summary")
    fig.suptitle(f"9957/g0 section {sid:03d} | ST ↔ final H&E", fontsize=17)
    fig.savefig(output_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        "section_id": sid,
        "z_um": float(row.z_um),
        "n_st_cells": int(len(st_points)),
        "n_valid_st": valid_count,
        "n_inside_he_mask": inside_count,
        "n_outside_he_mask": int(outside.sum()),
        "n_outside_he_crop": int(invalid.sum()),
        "inside_he_mask_fraction": float(inside_fraction),
        "output_path": str(output_path.resolve()),
    }


def _contact_sheet(paths, output_path):
    thumbs = []
    for path in paths:
        with Image.open(path) as image:
            thumb = image.convert("RGB")
            thumb.thumbnail((760, 560), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (780, 610), "white")
            canvas.paste(thumb, ((780 - thumb.width) // 2, 30 + (560 - thumb.height) // 2))
            ImageDraw.Draw(canvas).text((10, 8), path.stem, fill="black")
            thumbs.append(canvas)
    sheet = Image.new("RGB", (780 * 2, 610 * 5), "white")
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((i % 2) * 780, (i // 2) * 610))
    sheet.save(output_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--st-source",
        choices=["registered_global", "he_local"],
        default="registered_global",
        help="Use existing final global ST coordinates or H&E-local coordinates mapped through the final H&E chain.",
    )
    args = parser.parse_args()
    output = OUTPUT if args.st_source == "registered_global" else GROUP / "qc" / "st_he_alignment_final_he_53_55_v1_chain_v1_he_local_mapped"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    renderer = _load_renderer()
    table = pd.read_parquet(PREP / "he_sections.parquet").copy()
    table["section_id"] = pd.to_numeric(table["section_id"], errors="raise").astype(int)
    table = table[table["section_id"].isin(ANCHORS)].sort_values("section_id").reset_index(drop=True)
    if table["section_id"].tolist() != ANCHORS:
        raise ValueError(f"Expected anchors {ANCHORS}, got {table['section_id'].tolist()}")

    mapper = renderer.FinalCoordinateMapper(
        pd.read_parquet(PREP / "he_sections.parquet").assign(
            section_id=lambda frame: pd.to_numeric(frame.section_id, errors="raise").astype(int)
        ),
        device="cpu",
    )
    h5ad_path = REG / "9957_g0__st_to_final_he_53_55_v1_chain_v1.h5ad"
    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        section_values = pd.to_numeric(adata.obs["section_id"], errors="raise").astype(int).to_numpy()
        if args.st_source == "he_local":
            coord_key = "spatial_he_local"
            local_coords = np.asarray(adata.obsm[coord_key], dtype=float)
            st_by_sid = {
                sid: mapper.local_to_global(local_coords[section_values == sid], sid)
                for sid in ANCHORS
            }
            coord_label = "spatial_he_local → final H&E mapper"
        else:
            coord_key = "spatial_latest_alignment_final_he_53_55_v1_chain_v1"
            if coord_key not in adata.obsm:
                coord_key = "spatial_registered"
            coords = np.asarray(adata.obsm[coord_key], dtype=float)
            st_by_sid = {sid: coords[section_values == sid] for sid in ANCHORS}
            coord_label = coord_key
    finally:
        adata.file.close()

    anchor_dir = output / "anchors"
    anchor_dir.mkdir(parents=True, exist_ok=False)
    source_table = pd.read_parquet(PREP / "he_sections.parquet")
    source_table["section_id"] = pd.to_numeric(source_table["section_id"], errors="raise").astype(int)
    rows = {int(row.section_id): row for row in source_table.itertuples(index=False)}
    records = []
    paths = []
    for sid in ANCHORS:
        source = renderer.BASE._load_source(rows[sid])
        image_output = anchor_dir / f"section-{sid:03d}__st_he_final_alignment.png"
        record = _plot_anchor(renderer, mapper, rows[sid], source, st_by_sid[sid], image_output, coord_label)
        records.append(record)
        paths.append(image_output)

    _contact_sheet(paths, output / "st_he_anchor_contact_sheet.png")
    summary = {
        "status": "rendered_st_he_anchor_qc",
        "coordinate_frame": "9957__g0_registered_final_he_53_55_v1_chain_v1",
        "h5ad": str(h5ad_path.resolve()),
        "st_coordinate_key": coord_key,
        "st_coordinate_mapping": coord_label,
        "anchor_sections": ANCHORS,
        "new_st_transform_fitted": False,
        "outputs": {
            "anchors": str(anchor_dir.resolve()),
            "contact_sheet": str((output / "st_he_anchor_contact_sheet.png").resolve()),
        },
        "records": records,
    }
    (output / "manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# 9957/g0 ST ↔ final H&E anchor QC\n\n"
        "These figures use the final embedded v1-chain H&E coordinate frame. "
        "The ST points are read from the final H5AD coordinate source shown in the manifest. "
        "For `he_local`, they are mapped through the final H&E chain; no new ST residual fit was written.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": summary["status"],
        "coordinate_frame": summary["coordinate_frame"],
        "st_coordinate_key": coord_key,
        "st_coordinate_mapping": coord_label,
        "anchor_count": len(ANCHORS),
        "output": str(output.resolve()),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
