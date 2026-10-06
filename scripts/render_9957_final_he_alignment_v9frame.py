"""Render the final H&E chain in the same frame used by the approved pair.

The approved 53->55 candidate was fitted on v9 registered H&E rasters.  This
renderer therefore uses those rasters as the source and applies the exact
v9->v6 coarse bridge followed by the baked direct 53->55 map.  It does not
replace that source with the separate raw-crop/v6 nonlinear lineage.
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
from PIL import Image, ImageDraw
from scipy.ndimage import map_coordinates


ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = ROOT.parent / "DeepSpatial_v2"
V2_SRC = V2_ROOT / "src"
V2_SCRIPTS = V2_ROOT / "scripts"
for path in (ROOT, V2_SRC, V2_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from deepspatial_v2.data_export.cell_level_alignment import load_stalign_map  # noqa: E402
from deepspatial_v2.serial_registration.stalign_backend import (  # noqa: E402
    transform_points_target_to_source_with_edge,
    transform_points_with_edge,
)
from deepspatial_v2.serial_registration.stalign_types import EdgeAlignment  # noqa: E402


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DIRECT_RUNNER = _load_module(ROOT / "scripts" / "run_9957_direct_53_55_stalign.py", "_direct_9957_runner")
BASE_RENDERER = _load_module(ROOT / "scripts" / "render_9957_v8_true_coordinate_he_sections.py", "_base_9957_renderer")

GROUP = ROOT / "data" / "9957_g0"
ARCHIVE_TOP = ROOT.parent / ".deepSpatial_archives" / "9957_g0_v5_rollback_20261001_132456" / "top_level"
V9_ROOT = V2_ROOT / "outputs" / "registration_9957_pairfix_v9_mask_from_v2_total_v7_affine"
HE_MANIFEST = V2_ROOT / "outputs" / "sequence" / "manifests" / "he_combined_object_manifest.parquet"
V6_TRANSFORM_DIR = ARCHIVE_TOP / "reconstruction_prep_v6_final_manual_alignment" / "registration_transforms_final_manual_alignment_v6"
FINAL_PREP = GROUP / "reconstruction_prep_final_he_53_55_diagonal_v1"
FINAL_TABLE = FINAL_PREP / "he_sections.parquet"
DIRECT_ROOT = FINAL_PREP / "direct_stalign_53_55_final"
OUTPUT_ROOT = GROUP / "qc" / "he_alignment_final_he_53_55_diagonal_v2_v9frame"
REMOVED_SECTION_IDS = {54}
DIRECT_START_SECTION = 55
DEFAULT_SCALE_UM = 6.0
DEFAULT_MARGIN_UM = 80.0


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


def _morphology_to_rgb(channels: np.ndarray) -> np.ndarray:
    value = np.asarray(channels, dtype=float)
    return np.clip(np.rint(np.moveaxis(1.0 - np.clip(value, 0.0, 1.0), 0, -1) * 255), 0, 255).astype(np.uint8)


class V9FrameMapper:
    """Map v9 registered raster coordinates into the final H&E frame."""

    def __init__(self, table: pd.DataFrame, *, device: str):
        self.table = table
        self.device = device
        self.edge, self.direct_summary = _load_edge(DIRECT_ROOT, device)
        self.he_manifest = pd.read_parquet(HE_MANIFEST)
        self.sources: dict[int, dict[str, Any]] = {}
        self.bridges: dict[int, np.ndarray] = {}
        for sid in table["section_id"].astype(int):
            raster = DIRECT_RUNNER._load_v9_source_raster(V9_ROOT, int(sid))
            image, mask, he_meta = DIRECT_RUNNER._load_source_h_and_e(V9_ROOT, int(sid), self.he_manifest)
            bridge_meta = DIRECT_RUNNER._v9_to_v6_bridge(V9_ROOT, V6_TRANSFORM_DIR, int(sid))
            self.sources[int(sid)] = {
                **raster,
                "image": np.asarray(image, dtype=np.float32),
                "mask": np.asarray(mask, dtype=bool),
                "he_meta": he_meta,
            }
            self.bridges[int(sid)] = np.asarray(bridge_meta["bridge_v9_to_v6"], dtype=float)

    def source_to_global(self, points: np.ndarray, sid: int) -> np.ndarray:
        value = DIRECT_RUNNER._apply_affine(np.asarray(points, dtype=float), self.bridges[int(sid)])
        if int(sid) >= DIRECT_START_SECTION:
            value = transform_points_with_edge(value, self.edge)
        return value

    def global_to_source(self, points: np.ndarray, sid: int) -> np.ndarray:
        value = np.asarray(points, dtype=float)
        if int(sid) >= DIRECT_START_SECTION:
            value = transform_points_target_to_source_with_edge(value, self.edge)
        return DIRECT_RUNNER._apply_affine(value, np.linalg.inv(self.bridges[int(sid)]))

    def bbox(self, sid: int, margin_um: float) -> tuple[np.ndarray, np.ndarray]:
        source = self.sources[int(sid)]
        points = np.asarray(source["grid"], dtype=float)[:, source["mask"]].T
        mapped = self.source_to_global(points, int(sid))
        return mapped.min(axis=0) - margin_um, mapped.max(axis=0) + margin_um

    def sample(self, sid: int, origin: np.ndarray, shape: tuple[int, int], scale_um: float) -> tuple[np.ndarray, np.ndarray]:
        source = self.sources[int(sid)]
        image = source["image"]
        mask = source["mask"].astype(np.float32)
        source_grid = source["grid"]
        source_x0 = float(source_grid[0, 0, 0])
        source_y0 = float(source_grid[1, 0, 0])
        height, width = shape
        total = height * width
        output = np.full((total, 3), 255.0, dtype=np.float32)
        output_mask = np.zeros(total, dtype=bool)
        for start in range(0, total, 65536):
            stop = min(start + 65536, total)
            indices = np.arange(start, stop, dtype=np.int64)
            rows = indices // width
            cols = indices % width
            global_points = np.column_stack([origin[0] + cols * scale_um, origin[1] + rows * scale_um])
            source_points = self.global_to_source(global_points, int(sid))
            source_cols = (source_points[:, 0] - source_x0) / float(source["dx"])
            source_rows = (source_points[:, 1] - source_y0) / float(source["dx"])
            inside = (
                (source_cols >= 0.0)
                & (source_cols <= image.shape[2] - 1)
                & (source_rows >= 0.0)
                & (source_rows <= image.shape[1] - 1)
            )
            cols_clip = np.clip(source_cols, 0.0, image.shape[2] - 1)
            rows_clip = np.clip(source_rows, 0.0, image.shape[1] - 1)
            sampled = np.stack(
                [
                    map_coordinates(image[channel], [rows_clip, cols_clip], order=1, mode="constant", cval=0.0)
                    for channel in range(image.shape[0])
                ],
                axis=1,
            )
            sampled_rgb = np.clip(1.0 - sampled, 0.0, 1.0) * 255.0
            sampled_mask = map_coordinates(mask, [rows_clip, cols_clip], order=0, mode="constant", cval=0.0) >= 0.5
            valid = inside & sampled_mask
            output[start:stop][valid] = sampled_rgb[valid]
            output_mask[start:stop] = valid
        return output.reshape(height, width, 3).astype(np.uint8), output_mask.reshape(height, width)

    def metadata(self) -> dict[str, Any]:
        return {
            "coordinate_frame": "9957__g0_registered_final_he_candidate_v9frame_diagonal_v2",
            "mapping_order": [
                "v9 registered H&E raster grid",
                "section-specific v9->v6 coarse bridge",
                "baked direct 53->55 STalign map for section >=55",
            ],
            "direct_summary": self.direct_summary,
        }


def _contact_sheet(paths: list[Path], output: Path, columns: int = 2) -> None:
    tiles = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((700, 430), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (720, 470), "white")
        tile.paste(image, ((720 - image.width) // 2, 30 + (430 - image.height) // 2))
        ImageDraw.Draw(tile).text((10, 8), path.stem, fill="black")
        tiles.append(tile)
    rows = int(np.ceil(len(tiles) / columns))
    sheet = Image.new("RGB", (columns * 720, rows * 470), "white")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * 720, (index // columns) * 470))
    sheet.save(output)


def render(scale_um: float, margin_um: float, device: str) -> dict[str, Any]:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT_ROOT}")
    table = pd.read_parquet(FINAL_TABLE).copy()
    table["section_id"] = pd.to_numeric(table["section_id"], errors="raise").astype(int)
    table = table[~table["section_id"].isin(REMOVED_SECTION_IDS)].sort_values("section_id").reset_index(drop=True)
    section_ids = table["section_id"].astype(int).tolist()
    mapper = V9FrameMapper(table, device=device)
    sections_dir = OUTPUT_ROOT / "sections"
    pairs_dir = OUTPUT_ROOT / "adjacent_redgreen_pairs"
    contact_dir = OUTPUT_ROOT / "contact_sheets"
    for path in (sections_dir, pairs_dir, contact_dir):
        path.mkdir(parents=True, exist_ok=False)

    bboxes = {sid: mapper.bbox(sid, margin_um) for sid in section_ids}
    section_paths: list[Path] = []
    audits: list[dict[str, Any]] = []
    for row in table.itertuples(index=False):
        sid = int(row.section_id)
        low, high = bboxes[sid]
        shape = (
            max(1, int(np.ceil((high[1] - low[1]) / scale_um))),
            max(1, int(np.ceil((high[0] - low[0]) / scale_um))),
        )
        rgb, valid = mapper.sample(sid, low, shape, scale_um)
        path = sections_dir / f"section-{sid:03d}__he_final_candidate_frame.png"
        BASE_RENDERER._plot_section(
            path,
            rgb,
            low,
            scale_um,
            f"9957/g0 section {sid:03d} | z={float(row.z_um):.1f} µm | final candidate H&E frame",
        )
        section_paths.append(path)
        audits.append(
            {
                "section_id": sid,
                "z_um": float(row.z_um),
                "direct_53_55_map_applied": bool(sid >= DIRECT_START_SECTION),
                "global_crop_origin_x_um": float(low[0]),
                "global_crop_origin_y_um": float(low[1]),
                "global_crop_max_x_um": float(high[0]),
                "global_crop_max_y_um": float(high[1]),
                "valid_render_fraction": float(valid.mean()),
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
        first, valid_first = mapper.sample(previous, low, shape, scale_um)
        second, valid_second = mapper.sample(current, low, shape, scale_um)
        density_first = np.clip((255.0 - first.astype(np.float32).mean(axis=2)) / 255.0, 0.0, 1.0)
        density_second = np.clip((255.0 - second.astype(np.float32).mean(axis=2)) / 255.0, 0.0, 1.0)
        path = pairs_dir / f"pair-{previous:03d}-{current:03d}__he_final_candidate_redgreen.png"
        BASE_RENDERER._plot_redgreen(
            path,
            density_first,
            valid_first,
            density_second,
            valid_second,
            low,
            scale_um,
            f"9957/g0 sections {previous:03d} → {current:03d} | approved candidate frame",
        )
        pair_paths.append(path)

    for start in range(0, len(pair_paths), 10):
        _contact_sheet(pair_paths[start : start + 10], contact_dir / f"he_redgreen_contact_sheet_page-{start // 10 + 1:02d}.png")

    pd.DataFrame(audits).sort_values("section_id").to_csv(OUTPUT_ROOT / "coordinate_audit.csv", index=False)
    manifest = {
        "status": "rendered_final_he_alignment_approved_candidate_frame",
        "release": "9957_final_he_53_55_diagonal_v2_qc",
        "coordinate_frame": "9957__g0_registered_final_he_candidate_v9frame_diagonal_v2",
        "section_count": len(section_ids),
        "adjacent_pair_count": len(pair_paths),
        "section_ids": section_ids,
        "removed_section_ids": sorted(REMOVED_SECTION_IDS),
        "outputs": {
            "sections": str(sections_dir.resolve()),
            "adjacent_redgreen_pairs": str(pairs_dir.resolve()),
            "contact_sheets": str(contact_dir.resolve()),
            "coordinate_audit": str((OUTPUT_ROOT / "coordinate_audit.csv").resolve()),
        },
        "mapper": mapper.metadata(),
        "st_note": "H&E-to-H&E final propagation only; ST-to-H&E alignment is not evaluated in this QC.",
    }
    (OUTPUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)) + "\n", encoding="utf-8")
    (OUTPUT_ROOT / "README.md").write_text(
        "# 9957/g0 final H&E alignment QC (approved candidate frame)\n\n"
        "This release uses the same v9 registered H&E raster and v9->v6 coarse bridge that produced the approved 53->55 candidate. Section 54 is removed; the baked direct 53->55 map is applied from section 55 onward.\n\n"
        "`sections/` contains one H&E image per retained section. `adjacent_redgreen_pairs/` contains one red/green H&E overlay per adjacent retained pair.\n",
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
        "output_root": str(OUTPUT_ROOT),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
