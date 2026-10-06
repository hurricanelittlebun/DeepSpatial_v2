"""Render final 9957/g0 H&E sections and adjacent red/green overlays.

This renderer uses the same raw-H&E -> v6 frame -> final direct 53->55 map
lineage used by the final materialized feature stores.  It never uses the old
v8 horizontal-flip/CCW90 chain and never applies a display-only fit.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[1]
V2_SRC = ROOT.parent / "DeepSpatial_v2" / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(V2_SRC) not in sys.path:
    sys.path.insert(0, str(V2_SRC))

from deepspatial_v2.data_export.cell_level_alignment import (  # noqa: E402
    SavedTransformStore,
    load_section_transform,
    load_stalign_map,
)
from deepspatial_v2.serial_registration.stalign_backend import (  # noqa: E402
    transform_points_target_to_source_with_edge,
    transform_points_with_edge,
)
from deepspatial_v2.serial_registration.stalign_types import EdgeAlignment  # noqa: E402


def _load_base_renderer():
    path = ROOT / "scripts" / "render_9957_v8_true_coordinate_he_sections.py"
    spec = importlib.util.spec_from_file_location("_base_he_renderer", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load renderer helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_base_renderer()
GROUP = ROOT / "data" / "9957_g0"
ARCHIVE_TOP = (
    ROOT.parent
    / ".deepSpatial_archives"
    / "9957_g0_v5_rollback_20261001_132456"
    / "top_level"
)
SOURCE_PREP = ARCHIVE_TOP / "reconstruction_prep_v6_final_manual_alignment"
FINAL_PREP = GROUP / "reconstruction_prep_final_he_53_55_diagonal_v1"
FINAL_TABLE = FINAL_PREP / "he_sections.parquet"
FINAL_TRANSFORM_DIR = FINAL_PREP / "registration_transforms_final_he_53_55_diagonal_v1"
DIRECT_ROOT = FINAL_PREP / "direct_stalign_53_55_final"
OUTPUT_ROOT = GROUP / "qc" / "he_alignment_final_he_53_55_diagonal_v1"
BASE_REGISTRATION_ROOT = ROOT.parent / "DeepSpatial_v2" / "outputs" / "registration"
V5_FEATURE_STORE = ARCHIVE_TOP / "reconstruction_prep_v5_coordinate_repair" / "uni2_features_global_v4_repaired.h5"
V6_FEATURE_STORE = SOURCE_PREP / "uni2_features_final_manual_alignment_v6.h5"
REMOVED_SECTION_IDS = {54}
DIRECT_START_SECTION = 55
DEFAULT_SCALE_UM = 6.0
DEFAULT_MARGIN_UM = 80.0
BOUNDARY_POINTS = 6000
AUDIT_POINTS = 500


def _load_edge(root: Path, device: str) -> tuple[EdgeAlignment, dict[str, Any]]:
    summary = json.loads((root / "direct_stalign_summary.json").read_text(encoding="utf-8"))
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
        metrics={},
        status="final_direct_stalign_diagonal_baked",
    )
    return edge, summary


def _apply_affine(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(points, dtype=float)
    homogeneous = np.column_stack([value, np.ones(len(value), dtype=float)])
    return (homogeneous @ np.asarray(matrix, dtype=float).T)[:, :2]


def _registered_to_source_points(points: np.ndarray, section: Any, store: SavedTransformStore) -> np.ndarray:
    value = np.asarray(points, dtype=float)
    for edge_key, direction in reversed(section.edge_chain):
        inverse_direction = "reverse" if direction == "forward" else "forward"
        value = store.transform(value, store.load_edge(edge_key), inverse_direction)
    return _apply_affine(value, np.linalg.inv(section.preorientation_matrix))


def _read_frame_affines(section_id: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    # Reuse the exact archived v1->v5 and v5->v6 provenance parser.
    return BASE._load_frame_affines(section_id)


class FinalCoordinateMapper:
    def __init__(self, table: pd.DataFrame, *, device: str):
        self.table = table
        self.device = device
        self.store = SavedTransformStore(BASE_REGISTRATION_ROOT, device=device)
        self.direct_edge, self.direct_summary = _load_edge(DIRECT_ROOT, device)
        self.sections: dict[int, dict[str, Any]] = {}
        v6_transform_dir = SOURCE_PREP / "registration_transforms_final_manual_alignment_v6"
        for row in table.sort_values("section_id").itertuples(index=False):
            sid = int(row.section_id)
            source_transform = v6_transform_dir / Path(str(row.section_transform_path)).name
            section = load_section_transform(source_transform)
            v1_to_v5, v5_to_v6, affine_meta = _read_frame_affines(sid)
            self.sections[sid] = {
                "row": row,
                "section": section,
                "source_transform": source_transform,
                "v1_to_v5": v1_to_v5,
                "v5_to_v6": v5_to_v6,
                "final_applied": sid >= DIRECT_START_SECTION,
                "affine_meta": affine_meta,
            }

    def local_to_global(self, points: np.ndarray, section_id: int) -> np.ndarray:
        record = self.sections[int(section_id)]
        value = BASE.transform_section_points(np.asarray(points, dtype=float), record["section"], self.store)
        value = _apply_affine(value, record["v1_to_v5"])
        value = _apply_affine(value, record["v5_to_v6"])
        if record["final_applied"]:
            value = transform_points_with_edge(value, self.direct_edge)
        return value

    def global_to_local(self, points: np.ndarray, section_id: int) -> np.ndarray:
        record = self.sections[int(section_id)]
        value = np.asarray(points, dtype=float)
        if record["final_applied"]:
            value = transform_points_target_to_source_with_edge(value, self.direct_edge)
        value = _apply_affine(value, np.linalg.inv(record["v5_to_v6"]))
        value = _apply_affine(value, np.linalg.inv(record["v1_to_v5"]))
        return _registered_to_source_points(value, record["section"], self.store)

    def metadata(self) -> dict[str, Any]:
        return {
            "coordinate_frame": "9957__g0_registered_final_he_53_55_diagonal_v1",
            "mapping_order": [
                "raw H&E crop local physical XY",
                "base saved preorientation + STalign edge chain",
                "v1 -> v5 frame affine",
                "v5 -> v6 frame affine",
                "direct 53 -> 55 STalign map with baked diagonal reflection for section >=55",
            ],
            "base_registration_root": str(BASE_REGISTRATION_ROOT.resolve()),
            "direct_summary": self.direct_summary,
        }


def _contact_sheet(paths: list[Path], output: Path, title: str, columns: int = 2) -> None:
    thumbs = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((700, 430), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (720, 470), "white")
        x = (720 - image.width) // 2
        y = 28 + (430 - image.height) // 2
        canvas.paste(image, (x, y))
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 8), path.stem, fill="black")
        thumbs.append(canvas)
    rows = int(np.ceil(len(thumbs) / columns))
    sheet = Image.new("RGB", (columns * 720, rows * 470), "white")
    for i, image in enumerate(thumbs):
        sheet.paste(image, ((i % columns) * 720, (i // columns) * 470))
    sheet.save(output)


def render(scale_um: float, margin_um: float, device: str) -> dict[str, Any]:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT_ROOT}")
    table = pd.read_parquet(FINAL_TABLE).copy()
    table["section_id"] = pd.to_numeric(table["section_id"], errors="raise").astype(int)
    table = table[~table["section_id"].isin(REMOVED_SECTION_IDS)].sort_values("section_id").reset_index(drop=True)
    section_ids = table["section_id"].astype(int).tolist()
    mapper = FinalCoordinateMapper(table, device=device)
    source_by_sid = {sid: BASE._load_source(row) for sid, row in zip(section_ids, table.itertuples(index=False))}

    sections_dir = OUTPUT_ROOT / "sections"
    pairs_dir = OUTPUT_ROOT / "adjacent_redgreen_pairs"
    contact_dir = OUTPUT_ROOT / "contact_sheets"
    for path in (sections_dir, pairs_dir, contact_dir):
        path.mkdir(parents=True, exist_ok=False)

    bboxes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    audits: list[dict[str, Any]] = []
    section_paths: list[Path] = []
    for row in table.itertuples(index=False):
        sid = int(row.section_id)
        source = source_by_sid[sid]
        boundary_px = BASE._boundary_points(source["mask"], BOUNDARY_POINTS)
        local = BASE._local_pixels_to_um(boundary_px, source)
        global_points = mapper.local_to_global(local, sid)
        low = global_points.min(axis=0) - margin_um
        high = global_points.max(axis=0) + margin_um
        bboxes[sid] = (low, high)
        shape = (
            max(1, int(np.ceil((high[1] - low[1]) / scale_um))),
            max(1, int(np.ceil((high[0] - low[0]) / scale_um))),
        )
        rgb, valid = BASE._sample_source_on_global_grid(
            mapper, source, sid, low, shape, scale_um
        )
        path = sections_dir / f"section-{sid:03d}__he_final_global.png"
        BASE._plot_section(
            path,
            rgb,
            low,
            scale_um,
            f"9957/g0 section {sid:03d} | z={float(row.z_um):.1f} µm | final H&E global frame",
        )
        section_paths.append(path)
        recovered = mapper.global_to_local(global_points[:AUDIT_POINTS], sid)
        roundtrip = np.linalg.norm(recovered - local[:AUDIT_POINTS], axis=1)
        audits.append(
            {
                "section_id": sid,
                "z_um": float(row.z_um),
                "final_he_alignment_applied": bool(sid >= DIRECT_START_SECTION),
                "raw_image_path": source["image_path"],
                "raw_mask_path": source["mask_path"],
                "source_transform_path": str(mapper.sections[sid]["source_transform"].resolve()),
                "output_transform_path": str(Path(str(row.section_transform_path)).resolve()),
                "global_crop_origin_x_um": float(low[0]),
                "global_crop_origin_y_um": float(low[1]),
                "global_crop_max_x_um": float(high[0]),
                "global_crop_max_y_um": float(high[1]),
                "render_scale_um_per_px": float(scale_um),
                "valid_render_fraction": float(valid.mean()),
                "roundtrip_max_um": float(roundtrip.max()) if len(roundtrip) else 0.0,
                "roundtrip_p95_um": float(np.percentile(roundtrip, 95)) if len(roundtrip) else 0.0,
            }
        )

    pair_paths: list[Path] = []
    for previous, current in zip(section_ids[:-1], section_ids[1:]):
        low = np.minimum(bboxes[previous][0], bboxes[current][0])
        high = np.maximum(bboxes[previous][1], bboxes[current][1])
        shape = (
            max(1, int(np.ceil((high[1] - low[1]) / scale_um))),
            max(1, int(np.ceil((high[0] - low[0]) / scale_um))),
        )
        density_a, valid_a = BASE._sample_density_on_global_grid(
            mapper, source_by_sid[previous], previous, low, shape, scale_um
        )
        density_b, valid_b = BASE._sample_density_on_global_grid(
            mapper, source_by_sid[current], current, low, shape, scale_um
        )
        path = pairs_dir / f"pair-{previous:03d}-{current:03d}__he_final_redgreen.png"
        BASE._plot_redgreen(
            path,
            density_a,
            valid_a,
            density_b,
            valid_b,
            low,
            scale_um,
            f"9957/g0 sections {previous:03d} → {current:03d} | final H&E global XY",
        )
        pair_paths.append(path)

    for start in range(0, len(pair_paths), 10):
        page = start // 10 + 1
        _contact_sheet(
            pair_paths[start : start + 10],
            contact_dir / f"he_redgreen_contact_sheet_page-{page:02d}.png",
            f"9957/g0 final H&E red/green pairs {start + 1}-{min(start + 10, len(pair_paths))}",
        )

    audit = pd.DataFrame(audits).sort_values("section_id")
    audit.to_csv(OUTPUT_ROOT / "coordinate_audit.csv", index=False)
    manifest = {
        "status": "rendered_final_he_alignment",
        "release": "9957_final_he_53_55_diagonal_v1_qc",
        "coordinate_frame": "9957__g0_registered_final_he_53_55_diagonal_v1",
        "section_count": len(section_ids),
        "section_ids": section_ids,
        "removed_section_ids": sorted(REMOVED_SECTION_IDS),
        "adjacent_pair_count": len(pair_paths),
        "scale_um_per_px": scale_um,
        "margin_um": margin_um,
        "mapper": mapper.metadata(),
        "outputs": {
            "sections": str(sections_dir.resolve()),
            "adjacent_redgreen_pairs": str(pairs_dir.resolve()),
            "contact_sheets": str(contact_dir.resolve()),
            "coordinate_audit": str((OUTPUT_ROOT / "coordinate_audit.csv").resolve()),
        },
        "st_note": "These figures validate H&E-to-H&E global propagation only; ST-to-H&E residual alignment is intentionally not evaluated here.",
    }
    (OUTPUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)) + "\n", encoding="utf-8")
    (OUTPUT_ROOT / "README.md").write_text(
        "# 9957/g0 final H&E alignment QC\n\n"
        "Raw H&E pixels are mapped into the final global coordinate frame.\n\n"
        "Sections 1-53 keep the reviewed v6 placement. Section 54 is removed. "
        "Sections 55-91 use the approved direct 53->55 STalign map whose active A/map values already include the upper-left/lower-right diagonal reflection.\n\n"
        "The `sections/` directory contains one global-coordinate image per section. `adjacent_redgreen_pairs/` contains one real H&E optical-density overlay per adjacent retained pair; red is the lower-Z section and green is the higher-Z section.\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale-um", type=float, default=DEFAULT_SCALE_UM)
    parser.add_argument("--margin-um", type=float, default=DEFAULT_MARGIN_UM)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = render(args.scale_um, args.margin_um, args.device)
    print(json.dumps({
        "status": result["status"],
        "section_count": result["section_count"],
        "adjacent_pair_count": result["adjacent_pair_count"],
        "output_root": result["outputs"]["sections"].rsplit("/sections", 1)[0],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
