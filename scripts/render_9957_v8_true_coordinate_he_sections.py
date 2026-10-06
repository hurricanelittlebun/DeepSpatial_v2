#!/usr/bin/env python3
"""Render auditable 9957/g0 H&E sections in the current v8 physical frame.

This renderer deliberately does not use the ``matrix`` field in a section
transform as the image warp.  That field is a coarse affine summary and is
not the complete nonlinear registration.  The source image is mapped with
the same lineage used by the v1/v6 feature stores:

    raw H&E crop
      -> current saved base STalign chain (preorientation + edge maps)
      -> v1 -> v5 frame affine
      -> v5 -> v6 frame affine
      -> current v8 53->55 chain (when applicable)
      -> current v8 global physical XY frame

The script writes:

* one global-coordinate H&E figure per section;
* one H&E + active-ST overlay for each available ST anchor;
* one red/green image for every adjacent retained section pair;
* a machine-readable manifest and coordinate audit CSV.

The output is diagnostic.  The transform provenance is exact and auditable,
but a poor overlay is reported as a poor registration; it is never hidden by
an automatically fitted display-only affine.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
V2_SRC = ROOT.parent / "DeepSpatial_v2" / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(V2_SRC) not in sys.path:
    sys.path.insert(0, str(V2_SRC))

from deepspatial_v2.data_export.cell_level_alignment import (  # noqa: E402
    SavedTransformStore,
    load_section_transform,
    transform_section_points,
)


GROUP = ROOT / "data" / "9957_g0"
PREP = GROUP / "reconstruction_prep_v8_latest_alignment_cutoff91"
TABLE_PATH = PREP / "he_sections.parquet"
H5AD_PATH = (
    GROUP
    / "registration_correction_he_v8_latest_alignment_cutoff91"
    / "9957_g0__st_to_latest_alignment_cutoff91_v8.h5ad"
)
BASE_REGISTRATION_ROOT = ROOT.parent / "DeepSpatial_v2" / "outputs" / "registration"
ARCHIVE_TOP = (
    ROOT.parent
    / ".deepSpatial_archives"
    / "9957_g0_v5_rollback_20261001_132456"
    / "top_level"
)
V5_FEATURE_STORE = (
    ARCHIVE_TOP
    / "reconstruction_prep_v5_coordinate_repair"
    / "uni2_features_global_v4_repaired.h5"
)
V6_FEATURE_STORE = (
    ARCHIVE_TOP
    / "reconstruction_prep_v6_final_manual_alignment"
    / "uni2_features_final_manual_alignment_v6.h5"
)
OUTPUT_ROOT = GROUP / "qc" / "he_alignment_v8_true_global_sections_v1"
DEFAULT_SCALE_UM = 6.0
DEFAULT_MARGIN_UM = 80.0
BOUNDARY_POINTS = 6000
AUDIT_POINTS = 500


def _load_materializer():
    path = ROOT / "scripts" / "materialize_9957_latest_alignment_cutoff91.py"
    spec = importlib.util.spec_from_file_location("_v8_materializer", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load current v8 materializer from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _apply_affine(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(points, dtype=float)
    matrix = np.asarray(matrix, dtype=float)
    if value.ndim != 2 or value.shape[1] != 2:
        raise ValueError("points must have shape [N, 2]")
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("matrix must be finite with shape [3, 3]")
    homogeneous = np.column_stack([value, np.ones(len(value), dtype=float)])
    output = (homogeneous @ matrix.T)[:, :2]
    if not np.isfinite(output).all():
        raise ValueError("affine transform produced non-finite points")
    return output


def _registered_to_source_points(
    points: np.ndarray, section: Any, store: SavedTransformStore
) -> np.ndarray:
    """Invert the saved preorientation + nonlinear edge chain.

    This is the same operation as the repository's registered-coordinate
    helper, kept local so this QC renderer does not import the training
    package's top-level ``deepspatial`` module.
    """

    value = np.asarray(points, dtype=float)
    if value.ndim != 2 or value.shape[1] != 2 or not np.isfinite(value).all():
        raise ValueError("points must be a finite [N, 2] array")
    opposite = {"forward": "reverse", "reverse": "forward"}
    result = value.copy()
    for edge_key, direction in reversed(section.edge_chain):
        if direction not in opposite:
            raise ValueError(f"Unknown edge direction: {direction}")
        result = store.transform(
            result, store.load_edge(edge_key), opposite[direction]
        )
    return _apply_affine(result, np.linalg.inv(section.preorientation_matrix))


def _matrix_sha256(matrix: np.ndarray) -> str:
    value = np.asarray(matrix, dtype="<f8")
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _read_feature_metadata(path: Path, section_id: int) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        candidates = [
            key
            for key in handle["sections"]
            if str(handle["sections"][key].attrs.get("section_id")) == str(section_id)
        ]
        if len(candidates) != 1:
            raise KeyError(f"Expected one section {section_id} in {path}, got {candidates}")
        group = handle["sections"][candidates[0]]
        metadata = json.loads(str(group.attrs["metadata"]))
        valid = np.asarray(group["valid"][:], dtype=bool)
    metadata["_valid"] = valid
    return metadata


def _load_frame_affines(section_id: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Recover the exact v1->v5 and v5->v6 affine lineage.

    v4/v5 used two materialization modes.  Sections copied unchanged into v4
    carry the v1->v5 affine in v5's frame_resample record.  Sections that were
    inverse-queried into the v4 frame carry the same affine in v4's
    ``composed_target_residual_affine`` field.  Reading both records avoids
    silently treating those later sections as identity.
    """

    v4_path = ARCHIVE_TOP / "reconstruction_prep_v4_manual_rotation" / "uni2_features_v4_registered_v2.h5"
    v5_meta = _read_feature_metadata(V5_FEATURE_STORE, section_id)
    v4_meta = _read_feature_metadata(v4_path, section_id)
    v6_meta = _read_feature_metadata(V6_FEATURE_STORE, section_id)

    v4_provenance = dict(v4_meta.get("provenance", {}))
    if v4_provenance.get("coordinate_operation") == "inverse_query_old_field_in_new_v4_frame":
        v1_to_v5 = np.asarray(
            v4_provenance["composed_target_residual_affine"], dtype=float
        )
        v1_to_v5_source = "v4.provenance.composed_target_residual_affine"
    else:
        record = v5_meta.get("provenance", {}).get("frame_resample")
        if not record:
            raise KeyError(f"No v1->v5 frame_resample record for section {section_id}")
        v1_to_v5 = np.asarray(record["matrix_source_to_output"], dtype=float)
        v1_to_v5_source = "v5.provenance.frame_resample.matrix_source_to_output"

    v5_to_v6_record = v6_meta.get("provenance", {}).get("frame_resample")
    if not v5_to_v6_record:
        raise KeyError(f"No v5->v6 frame_resample record for section {section_id}")
    v5_to_v6 = np.asarray(v5_to_v6_record["matrix_source_to_output"], dtype=float)

    for label, matrix in (("v1_to_v5", v1_to_v5), ("v5_to_v6", v5_to_v6)):
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError(f"Invalid {label} for section {section_id}")
        if abs(float(np.linalg.det(matrix[:2, :2]))) < 1e-12:
            raise ValueError(f"Singular {label} for section {section_id}")

    return v1_to_v5, v5_to_v6, {
        "v1_to_v5_source": v1_to_v5_source,
        "v5_to_v6_source": "v6.provenance.frame_resample.matrix_source_to_output",
        "v1_to_v5": v1_to_v5.tolist(),
        "v5_to_v6": v5_to_v6.tolist(),
        "v1_to_v5_sha256": _matrix_sha256(v1_to_v5),
        "v5_to_v6_sha256": _matrix_sha256(v5_to_v6),
    }


def _load_source(row: Any) -> dict[str, Any]:
    image_path = Path(str(row.source_image_path)).resolve()
    mask_path = Path(str(row.source_mask_path)).resolve()
    if not image_path.is_file() or not mask_path.is_file():
        raise FileNotFoundError(f"Missing H&E/mask for section {row.section_id}")
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    with Image.open(mask_path) as mask_image:
        mask = np.asarray(mask_image.convert("L"), dtype=np.uint8) > 0
    if rgb.shape[:2] != mask.shape:
        raise ValueError(f"H&E/mask shape mismatch for section {row.section_id}")
    sx = float(row.analysis_pixel_size_x_um)
    sy = float(row.analysis_pixel_size_y_um)
    if sx <= 0 or sy <= 0:
        raise ValueError(f"Invalid pixel size for section {row.section_id}")
    return {
        "rgb": rgb,
        "mask": mask,
        "pixel_size_um": np.asarray([sx, sy], dtype=float),
        "local_origin_um": np.asarray(
            [float(row.bbox_x0) * sx, float(row.bbox_y0) * sy], dtype=float
        ),
        "image_path": str(image_path),
        "mask_path": str(mask_path),
    }


class CoordinateMapper:
    """Map current raw H&E local physical coordinates to/from v8 global XY."""

    def __init__(self, table: pd.DataFrame, device: str = "cpu") -> None:
        self.table = table
        self.device = device
        self.materializer = _load_materializer()
        self.store = SavedTransformStore(BASE_REGISTRATION_ROOT, device=device)

        direct_root = PREP / "stalign_final_53_55" / "direct_v3"
        finetune_root = PREP / "stalign_final_53_55" / "finetune_v1"
        direct, direct_summary = self.materializer._load_edge(
            direct_root,
            summary_name="direct_stalign_summary.json",
            affine_name="direct_stalign_affine.npz",
            forward_name="direct_stalign_forward_map.npz",
            reverse_name="direct_stalign_reverse_map.npz",
            status="current_v8_direct",
            device=device,
        )
        finetune, finetune_summary = self.materializer._load_edge(
            finetune_root,
            summary_name="postrotate_finetune_summary.json",
            affine_name="finetune_affine.npz",
            forward_name="finetune_forward_map.npz",
            reverse_name="finetune_reverse_map.npz",
            status="current_v8_finetune",
            device=device,
        )
        self.latest_chain = self.materializer.LatestAlignmentChain(
            direct, finetune, device=device
        )
        self.direct_summary = direct_summary
        self.finetune_summary = finetune_summary
        self.sections: dict[int, dict[str, Any]] = {}

        for row in table.sort_values("section_id").itertuples(index=False):
            sid = int(row.section_id)
            transform_path = Path(str(row.section_transform_path)).resolve()
            section = load_section_transform(transform_path)
            v1_to_v5, v5_to_v6, affine_meta = _load_frame_affines(sid)
            with np.load(transform_path, allow_pickle=False) as payload:
                latest_applied = bool(
                    int(np.asarray(payload["latest_alignment_applied"]).reshape(-1)[0])
                )
                output_frame = str(np.asarray(payload["latest_alignment_coordinate_frame"]).item())
                coarse_matrix = np.asarray(payload["matrix"], dtype=float)
            self.sections[sid] = {
                "row": row,
                "section": section,
                "transform_path": transform_path,
                "v1_to_v5": v1_to_v5,
                "v5_to_v6": v5_to_v6,
                "latest_applied": latest_applied,
                "output_frame": output_frame,
                "coarse_matrix": coarse_matrix,
                "affine_meta": affine_meta,
            }

    def local_to_global(self, points: np.ndarray, section_id: int) -> np.ndarray:
        record = self.sections[int(section_id)]
        value = transform_section_points(
            np.asarray(points, dtype=float),
            record["section"],
            self.store,
        )
        value = _apply_affine(value, record["v1_to_v5"])
        value = _apply_affine(value, record["v5_to_v6"])
        if record["latest_applied"]:
            value = self.latest_chain.forward_chunked(value)
        return value

    def global_to_local(self, points: np.ndarray, section_id: int) -> np.ndarray:
        record = self.sections[int(section_id)]
        value = np.asarray(points, dtype=float)
        if record["latest_applied"]:
            value = self.latest_chain.inverse_chunked(value)
        value = _apply_affine(value, np.linalg.inv(record["v5_to_v6"]))
        value = _apply_affine(value, np.linalg.inv(record["v1_to_v5"]))
        return _registered_to_source_points(value, record["section"], self.store)

    def metadata(self) -> dict[str, Any]:
        return {
            "coordinate_frame": next(iter(self.sections.values()))["output_frame"],
            "base_registration_root": str(BASE_REGISTRATION_ROOT.resolve()),
            "current_he_table": str(TABLE_PATH.resolve()),
            "current_h5ad": str(H5AD_PATH.resolve()),
            "v5_feature_store": str(V5_FEATURE_STORE.resolve()),
            "v6_feature_store": str(V6_FEATURE_STORE.resolve()),
            "mapping_order": [
                "raw H&E crop local physical XY",
                "current saved base preorientation + STalign edge chain",
                "v1 -> v5 frame affine from archived feature-store provenance",
                "v5 -> v6 frame affine from archived feature-store provenance",
                "current v8 53->55 chain when latest_alignment_applied=1",
            ],
            "coarse_section_matrix_used_for_image": False,
            "coarse_matrix_note": (
                "section transform field matrix is retained as an audit value only; "
                "it is not applied separately because it summarizes the affine chain"
            ),
            "latest_chain": self.latest_chain.metadata(),
            "direct_summary": self.direct_summary,
            "finetune_summary": self.finetune_summary,
        }


def _boundary_points(mask: np.ndarray, max_points: int) -> np.ndarray:
    from scipy.ndimage import binary_erosion

    boundary = mask & ~binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))
    rows, cols = np.nonzero(boundary)
    if len(rows) == 0:
        rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        return np.empty((0, 2), dtype=float)
    if len(rows) > max_points:
        selected = np.linspace(0, len(rows) - 1, max_points, dtype=np.int64)
        rows, cols = rows[selected], cols[selected]
    return np.column_stack([cols, rows]).astype(float)


def _local_pixels_to_um(pixel_points: np.ndarray, source: dict[str, Any]) -> np.ndarray:
    return source["local_origin_um"] + pixel_points * source["pixel_size_um"]


def _global_bbox(mapper: CoordinateMapper, source: dict[str, Any], sid: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boundary_px = _boundary_points(source["mask"], BOUNDARY_POINTS)
    local = _local_pixels_to_um(boundary_px, source)
    global_points = mapper.local_to_global(local, sid)
    return global_points.min(axis=0), global_points.max(axis=0), global_points


def _sample_source_on_global_grid(
    mapper: CoordinateMapper,
    source: dict[str, Any],
    sid: int,
    origin: np.ndarray,
    shape: tuple[int, int],
    scale_um: float,
    *,
    chunk_size: int = 65536,
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.ndimage import map_coordinates

    height, width = shape
    rgb = source["rgb"].astype(np.float32)
    mask = source["mask"].astype(np.float32)
    sx, sy = source["pixel_size_um"]
    output = np.full((height * width, 3), 255.0, dtype=np.float32)
    output_mask = np.zeros(height * width, dtype=bool)
    for start in range(0, height * width, chunk_size):
        stop = min(start + chunk_size, height * width)
        indices = np.arange(start, stop, dtype=np.int64)
        rows = indices // width
        cols = indices % width
        global_points = np.column_stack(
            [origin[0] + cols * scale_um, origin[1] + rows * scale_um]
        )
        local = mapper.global_to_local(global_points, sid)
        source_cols = (local[:, 0] - source["local_origin_um"][0]) / sx
        source_rows = (local[:, 1] - source["local_origin_um"][1]) / sy
        inside = (
            (source_cols >= 0)
            & (source_cols <= rgb.shape[1] - 1)
            & (source_rows >= 0)
            & (source_rows <= rgb.shape[0] - 1)
        )
        source_cols_clip = np.clip(source_cols, 0, rgb.shape[1] - 1)
        source_rows_clip = np.clip(source_rows, 0, rgb.shape[0] - 1)
        sampled = np.stack(
            [
                map_coordinates(
                    rgb[..., channel],
                    [source_rows_clip, source_cols_clip],
                    order=1,
                    mode="constant",
                    cval=255.0,
                )
                for channel in range(3)
            ],
            axis=1,
        )
        sampled_mask = map_coordinates(
            mask,
            [source_rows_clip, source_cols_clip],
            order=0,
            mode="constant",
            cval=0.0,
        ) > 0.5
        valid = inside & sampled_mask
        output[start:stop][valid] = sampled[valid]
        output_mask[start:stop] = valid
    return output.reshape(height, width, 3).astype(np.uint8), output_mask.reshape(height, width)


def _sample_density_on_global_grid(
    mapper: CoordinateMapper,
    source: dict[str, Any],
    sid: int,
    origin: np.ndarray,
    shape: tuple[int, int],
    scale_um: float,
) -> tuple[np.ndarray, np.ndarray]:
    rgb, valid = _sample_source_on_global_grid(
        mapper, source, sid, origin, shape, scale_um
    )
    # Optical density-like signal; white background is zero.
    gray = rgb.astype(np.float32).mean(axis=2)
    density = np.clip((255.0 - gray) / 255.0, 0.0, 1.0)
    return density, valid


def _plot_section(
    output: Path,
    rgb: np.ndarray,
    origin: np.ndarray,
    scale_um: float,
    title: str,
    *,
    st_points: np.ndarray | None = None,
    st_label: str | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    height, width = rgb.shape[:2]
    extent = [
        float(origin[0]),
        float(origin[0] + width * scale_um),
        float(origin[1] + height * scale_um),
        float(origin[1]),
    ]
    fig, ax = plt.subplots(figsize=(10, 8), dpi=180)
    ax.imshow(rgb, extent=extent, origin="upper", interpolation="nearest")
    if st_points is not None and len(st_points):
        ax.scatter(
            st_points[:, 0],
            st_points[:, 1],
            s=1.4,
            c="#00d8ff",
            alpha=0.38,
            linewidths=0,
            rasterized=True,
            label=st_label or "active ST",
        )
        ax.legend(loc="upper right", framealpha=0.9, markerscale=4)
    ax.set_xlabel("global X (µm)")
    ax.set_ylabel("global Y (µm)")
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.grid(False)
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_redgreen(
    output: Path,
    density_a: np.ndarray,
    valid_a: np.ndarray,
    density_b: np.ndarray,
    valid_b: np.ndarray,
    origin: np.ndarray,
    scale_um: float,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    image = np.zeros((*density_a.shape, 3), dtype=np.float32)
    image[..., 0] = np.clip(density_a, 0, 1) * valid_a
    image[..., 1] = np.clip(density_b, 0, 1) * valid_b
    image = np.clip(image, 0, 1)
    h, w = image.shape[:2]
    extent = [origin[0], origin[0] + w * scale_um, origin[1] + h * scale_um, origin[1]]
    fig, ax = plt.subplots(figsize=(10, 8), dpi=180)
    ax.imshow(image, extent=extent, origin="upper", interpolation="nearest")
    ax.set_xlabel("global X (µm)")
    ax.set_ylabel("global Y (µm)")
    ax.set_title(title + " | red=previous, green=current", fontsize=10)
    ax.set_aspect("equal")
    fig.savefig(output, bbox_inches="tight", facecolor="black")
    plt.close(fig)


def _classify_st(
    mapper: CoordinateMapper,
    source: dict[str, Any],
    sid: int,
    st_points: np.ndarray,
) -> tuple[int, int, float]:
    if len(st_points) == 0:
        return 0, 0, float("nan")
    local = mapper.global_to_local(st_points, sid)
    sx, sy = source["pixel_size_um"]
    cols = np.rint((local[:, 0] - source["local_origin_um"][0]) / sx).astype(int)
    rows = np.rint((local[:, 1] - source["local_origin_um"][1]) / sy).astype(int)
    valid = (
        (rows >= 0)
        & (rows < source["mask"].shape[0])
        & (cols >= 0)
        & (cols < source["mask"].shape[1])
    )
    inside = np.zeros(len(st_points), dtype=bool)
    inside[valid] = source["mask"][rows[valid], cols[valid]]
    return int(len(st_points)), int((inside & valid).sum()), float((inside & valid).mean())


def _audit_roundtrip(mapper: CoordinateMapper, source: dict[str, Any], sid: int) -> dict[str, float]:
    boundary_px = _boundary_points(source["mask"], AUDIT_POINTS)
    local = _local_pixels_to_um(boundary_px, source)
    global_points = mapper.local_to_global(local, sid)
    recovered = mapper.global_to_local(global_points, sid)
    errors = np.linalg.norm(recovered - local, axis=1)
    return {
        "roundtrip_max_um": float(np.max(errors)) if len(errors) else 0.0,
        "roundtrip_median_um": float(np.median(errors)) if len(errors) else 0.0,
        "roundtrip_p95_um": float(np.percentile(errors, 95)) if len(errors) else 0.0,
    }


def render(args: argparse.Namespace) -> dict[str, Any]:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUTPUT_ROOT}")
    if not TABLE_PATH.is_file() or not H5AD_PATH.is_file():
        raise FileNotFoundError("Current v8 H&E table or H5AD is missing")
    if not V5_FEATURE_STORE.is_file() or not V6_FEATURE_STORE.is_file():
        raise FileNotFoundError("Archived v5/v6 feature store needed for frame audit is missing")

    table = pd.read_parquet(TABLE_PATH).copy()
    table["section_id"] = pd.to_numeric(table["section_id"], errors="raise").astype(int)
    table = table.sort_values("section_id", kind="stable").reset_index(drop=True)
    section_ids = table.section_id.astype(int).tolist()
    mapper = CoordinateMapper(table, device=args.device)

    source_by_sid = {sid: _load_source(row) for sid, row in zip(section_ids, table.itertuples(index=False))}
    h5 = ad.read_h5ad(H5AD_PATH, backed="r")
    try:
        h5_section = pd.to_numeric(h5.obs["section_id"], errors="raise").astype(int).to_numpy()
        h5_points = np.asarray(h5.obsm["spatial_st_corrected"], dtype=float)
        st_by_sid = {sid: h5_points[h5_section == sid] for sid in section_ids}
    finally:
        h5.file.close()

    sections_dir = OUTPUT_ROOT / "sections"
    anchors_dir = OUTPUT_ROOT / "anchors_st_overlay"
    pairs_dir = OUTPUT_ROOT / "adjacent_redgreen_pairs"
    for path in (sections_dir, anchors_dir, pairs_dir):
        path.mkdir(parents=True, exist_ok=False)

    audits: list[dict[str, Any]] = []
    bboxes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    boundary_points: dict[int, np.ndarray] = {}

    for sid in section_ids:
        low, high, boundary = _global_bbox(mapper, source_by_sid[sid], sid)
        bboxes[sid] = (low - args.margin_um, high + args.margin_um)
        boundary_points[sid] = boundary

    # Render every section on its own crop, but the axes are global physical XY.
    for row in table.itertuples(index=False):
        sid = int(row.section_id)
        low, high = bboxes[sid]
        shape = (
            max(1, int(np.ceil((high[1] - low[1]) / args.scale_um))),
            max(1, int(np.ceil((high[0] - low[0]) / args.scale_um))),
        )
        rgb, valid = _sample_source_on_global_grid(
            mapper, source_by_sid[sid], sid, low, shape, args.scale_um
        )
        st = st_by_sid[sid]
        title = (
            f"9957/g0 section {sid:03d} | z={float(row.z_um):.1f} µm | "
            f"v8 global frame | raw H&E in true global coordinates"
        )
        _plot_section(
            sections_dir / f"section-{sid:03d}__v8_true_global_he.png",
            rgb,
            low,
            args.scale_um,
            title,
            st_points=st if len(st) else None,
            st_label="active ST (spatial_st_corrected)" if len(st) else None,
        )
        if len(st):
            _plot_section(
                anchors_dir / f"section-{sid:03d}__he_plus_active_st.png",
                rgb,
                low,
                args.scale_um,
                title + " | cyan=active ST",
                st_points=st,
                st_label="active ST (spatial_st_corrected)",
            )

        n_st, n_inside, fraction = _classify_st(mapper, source_by_sid[sid], sid, st)
        roundtrip = _audit_roundtrip(mapper, source_by_sid[sid], sid)
        record = {
            "section_id": sid,
            "z_um": float(row.z_um),
            "n_st_cells": n_st,
            "n_st_in_transformed_raw_he_mask": n_inside,
            "st_in_mask_fraction": fraction,
            "raw_image_path": source_by_sid[sid]["image_path"],
            "raw_mask_path": source_by_sid[sid]["mask_path"],
            "section_transform_path": str(mapper.sections[sid]["transform_path"]),
            "base_registration_root": str(BASE_REGISTRATION_ROOT.resolve()),
            "output_coordinate_frame": mapper.sections[sid]["output_frame"],
            "latest_alignment_applied": bool(mapper.sections[sid]["latest_applied"]),
            "raw_shape_yx": list(source_by_sid[sid]["mask"].shape),
            "global_crop_origin_x_um": float(low[0]),
            "global_crop_origin_y_um": float(low[1]),
            "global_crop_max_x_um": float(high[0]),
            "global_crop_max_y_um": float(high[1]),
            "render_scale_um_per_px": float(args.scale_um),
            "valid_render_fraction": float(valid.mean()),
            "coarse_section_matrix_applied": False,
            **roundtrip,
            **mapper.sections[sid]["affine_meta"],
        }
        audits.append(record)

    # Adjacent pair red/green figures use a union global crop and exactly the
    # same coordinate mapping as the single-section images.
    for previous, current in zip(section_ids[:-1], section_ids[1:]):
        low = np.minimum(bboxes[previous][0], bboxes[current][0])
        high = np.maximum(bboxes[previous][1], bboxes[current][1])
        shape = (
            max(1, int(np.ceil((high[1] - low[1]) / args.scale_um))),
            max(1, int(np.ceil((high[0] - low[0]) / args.scale_um))),
        )
        density_a, valid_a = _sample_density_on_global_grid(
            mapper, source_by_sid[previous], previous, low, shape, args.scale_um
        )
        density_b, valid_b = _sample_density_on_global_grid(
            mapper, source_by_sid[current], current, low, shape, args.scale_um
        )
        _plot_redgreen(
            pairs_dir / f"pair-{previous:03d}-{current:03d}__v8_true_global_redgreen.png",
            density_a,
            valid_a,
            density_b,
            valid_b,
            low,
            args.scale_um,
            f"9957/g0 sections {previous:03d} → {current:03d} | v8 global physical XY",
        )

    audit_df = pd.DataFrame(audits).sort_values("section_id")
    audit_df.to_csv(OUTPUT_ROOT / "coordinate_audit.csv", index=False)
    manifest = {
        "status": "rendered",
        "release": "9957_v8_true_coordinate_he_qc_v1",
        "output_root": str(OUTPUT_ROOT.resolve()),
        "section_count": len(section_ids),
        "section_ids": section_ids,
        "removed_section_ids": [54],
        "scale_um_per_px": float(args.scale_um),
        "margin_um": float(args.margin_um),
        "coordinate_mapper": mapper.metadata(),
        "outputs": {
            "sections": str(sections_dir.resolve()),
            "anchors_st_overlay": str(anchors_dir.resolve()),
            "adjacent_redgreen_pairs": str(pairs_dir.resolve()),
            "coordinate_audit": str((OUTPUT_ROOT / "coordinate_audit.csv").resolve()),
        },
        "interpretation": {
            "guarantee": (
                "All H&E and ST points in these figures use the recorded current v8 "
                "physical coordinate chain; no display-only affine or per-image auto-fit "
                "was applied."
            ),
            "quality_warning": (
                "Coordinate provenance is guaranteed; biological/image alignment quality "
                "must be judged from the overlay and audit fractions."
            ),
        },
    }
    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))
        + "\n",
        encoding="utf-8",
    )
    (OUTPUT_ROOT / "README.md").write_text(
        "# 9957/g0 v8 true-coordinate H&E QC\n\n"
        "These figures are rendered from raw H&E pixels after the recorded coordinate chain.\n\n"
        "The image chain is: raw crop local XY -> base saved preorientation/STalign edge chain -> "
        "v1-to-v5 frame affine -> v5-to-v6 frame affine -> current v8 53-to-55 chain when active.\n\n"
        "The active Xenium points are read from `spatial_st_corrected` in the current v8 H5AD. "
        "The section NPZ `matrix` is retained for audit but is deliberately not applied as a second "
        "image transform because it is a coarse affine summary.\n\n"
        "A round-trip error close to zero means the coordinate implementation is internally reversible; "
        "it does not by itself prove that the tissue registration is biologically perfect.\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale-um", type=float, default=DEFAULT_SCALE_UM)
    parser.add_argument("--margin-um", type=float, default=DEFAULT_MARGIN_UM)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.scale_um <= 0 or args.margin_um < 0:
        raise ValueError("scale-um must be positive and margin-um non-negative")
    manifest = render(args)
    print(json.dumps({"output_root": manifest["output_root"], "section_count": manifest["section_count"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
