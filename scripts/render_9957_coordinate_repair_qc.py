"""Render 9957/g0 coordinate-repair QC figures before model training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import binary_erosion

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepspatial.histology.frame_resample import transform_points  # noqa: E402


DEFAULT_GROUP = ROOT / "data" / "9957_g0"
DEFAULT_CANDIDATE = (
    DEFAULT_GROUP
    / "registration_correction_he_v5_coordinate_repair"
    / "9957_g0__st_to_global_v4_repaired_candidate.h5ad"
)
DEFAULT_HE_TABLE = (
    DEFAULT_GROUP / "reconstruction_prep_v5_coordinate_repair" / "he_sections.parquet"
)
DEFAULT_PATH_SUMMARY = (
    DEFAULT_GROUP / "reconstruction_prep_v5_coordinate_repair" / "path_cache_summary.json"
)
DEFAULT_UOT_METADATA = (
    DEFAULT_GROUP
    / "reconstruction_prep_v5_coordinate_repair"
    / "uot_cache_topk64"
    / "run_metadata.json"
)
DEFAULT_FEATURE_STORE = (
    DEFAULT_GROUP
    / "reconstruction_prep_v5_coordinate_repair"
    / "uni2_features_global_v4_repaired.h5"
)
DEFAULT_OUTPUT = DEFAULT_GROUP / "qc" / "coordinate_repair_v5"


def mask_pixels_to_global(
    mask: np.ndarray,
    pixel_size_um: tuple[float, float],
    affine: np.ndarray,
    *,
    max_points: int = 6000,
) -> np.ndarray:
    """Return a downsampled tissue-mask boundary in global physical XY."""

    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("mask must be 2-D")
    if max_points <= 0:
        raise ValueError("max_points must be positive")
    scale_x, scale_y = map(float, pixel_size_um)
    if not np.isfinite([scale_x, scale_y]).all() or min(scale_x, scale_y) <= 0:
        raise ValueError("pixel_size_um must contain positive finite values")
    boundary = mask & ~binary_erosion(mask, structure=np.ones((3, 3), bool), border_value=0)
    rows, columns = np.nonzero(boundary)
    if len(rows) == 0:
        return np.empty((0, 2), dtype=float)
    if len(rows) > max_points:
        selected = np.linspace(0, len(rows) - 1, max_points, dtype=np.int64)
        rows, columns = rows[selected], columns[selected]
    local = np.column_stack([columns * scale_x, rows * scale_y])
    return transform_points(local, affine)


def feature_grid_boundary(
    valid_mask: np.ndarray,
    *,
    origin_um: tuple[float, float] | np.ndarray,
    spacing_um: tuple[float, float] | np.ndarray,
    max_points: int = 6000,
) -> np.ndarray:
    """Return the boundary of a feature-grid support mask in global µm.

    ``FeatureStore`` stores the first grid center in ``origin_um`` and uses
    ``spacing_um`` between adjacent centers.  This helper deliberately uses
    that same convention; it does not apply a raw crop transform a second
    time.  The result therefore describes the H&E/UNI2 support actually
    queried by UOT and the morphology path.
    """

    mask = np.asarray(valid_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("valid_mask must be 2-D")
    origin = np.asarray(origin_um, dtype=float)
    spacing = np.asarray(spacing_um, dtype=float)
    if origin.shape != (2,) or spacing.shape != (2,):
        raise ValueError("origin_um and spacing_um must be x/y pairs")
    if not np.isfinite(origin).all() or not np.isfinite(spacing).all():
        raise ValueError("origin_um and spacing_um must be finite")
    if np.any(spacing <= 0):
        raise ValueError("spacing_um must be positive")
    if max_points <= 0:
        raise ValueError("max_points must be positive")

    boundary = mask & ~binary_erosion(
        mask, structure=np.ones((3, 3), bool), border_value=0
    )
    rows, columns = np.nonzero(boundary)
    if len(rows) == 0:
        return np.empty((0, 2), dtype=float)
    if len(rows) > max_points:
        selected = np.linspace(0, len(rows) - 1, max_points, dtype=np.int64)
        rows, columns = rows[selected], columns[selected]
    return np.column_stack(
        [origin[0] + columns * spacing[0], origin[1] + rows * spacing[1]]
    )


def _downsample(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
    return points[indices]


def _read_transform(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        matrix = np.asarray(payload["matrix"], dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"Invalid registration matrix in {path}")
    return matrix


def _classify_against_mask(points_global: np.ndarray, mask: np.ndarray, pixel_size, affine):
    local = transform_points(points_global, np.linalg.inv(affine))
    scale_x, scale_y = map(float, pixel_size)
    columns = np.rint(local[:, 0] / scale_x).astype(np.int64)
    rows = np.rint(local[:, 1] / scale_y).astype(np.int64)
    valid = (
        (rows >= 0)
        & (rows < mask.shape[0])
        & (columns >= 0)
        & (columns < mask.shape[1])
    )
    inside = np.zeros(len(points_global), dtype=bool)
    inside[valid] = mask[rows[valid], columns[valid]]
    return inside, valid


def _load_records(candidate: Path, he_table: Path, max_mask_points: int) -> tuple[list[dict], str]:
    table = pd.read_parquet(he_table).copy()
    table["section_id"] = pd.to_numeric(table["section_id"], errors="raise").astype(int)
    table_by_section = {int(row.section_id): row for row in table.itertuples(index=False)}
    source = ad.read_h5ad(candidate, backed="r")
    try:
        sections = pd.to_numeric(source.obs["section_id"], errors="raise").astype(int).to_numpy()
        points = np.asarray(source.obsm["spatial_st_corrected"], dtype=np.float64)
        frame_values = set(source.obs["coordinate_frame"].astype(str).unique())
        if len(frame_values) != 1:
            raise ValueError(f"Candidate has multiple coordinate frames: {sorted(frame_values)}")
        frame = next(iter(frame_values))
        records = []
        for section_id in sorted(np.unique(sections)):
            if int(section_id) not in table_by_section:
                raise KeyError(f"H&E table lacks section {section_id}")
            row = table_by_section[int(section_id)]
            mask_path = Path(str(row.source_mask_path))
            transform_path = Path(str(row.section_transform_path))
            with Image.open(mask_path) as image:
                mask = np.asarray(image.convert("L")) > 0
            pixel_size = (
                float(row.analysis_pixel_size_x_um),
                float(row.analysis_pixel_size_y_um),
            )
            affine = _read_transform(transform_path)
            section_points = points[sections == int(section_id)]
            inside, valid = _classify_against_mask(section_points, mask, pixel_size, affine)
            records.append(
                {
                    "section_id": int(section_id),
                    "z_um": float(row.z_um),
                    "points": section_points,
                    "mask_boundary": mask_pixels_to_global(
                        mask, pixel_size, affine, max_points=max_mask_points
                    ),
                    "n_cells": int(len(section_points)),
                    "n_inside_mask": int((inside & valid).sum()),
                    "n_valid_mask_queries": int(valid.sum()),
                    "inside_mask_fraction": float((inside & valid).sum() / max(valid.sum(), 1)),
                    "mask_path": str(mask_path),
                    "transform_path": str(transform_path),
                    "affine": affine,
                }
            )
        return records, frame
    finally:
        source.file.close()


def _load_feature_records(
    candidate: Path, feature_store_path: Path, max_boundary_points: int
) -> tuple[list[dict], str]:
    """Load ST anchors and the registered H&E support used by the model."""

    from deepspatial.histology.feature_store import FeatureStore

    store = FeatureStore(feature_store_path, cache_mb=1024)
    source = ad.read_h5ad(candidate, backed="r")
    try:
        sections = pd.to_numeric(source.obs["section_id"], errors="raise").astype(int).to_numpy()
        points = np.asarray(source.obsm["spatial_st_corrected"], dtype=np.float64)
        local_points = (
            np.asarray(source.obsm["spatial_he_local"], dtype=np.float64)
            if "spatial_he_local" in source.obsm
            else None
        )
        frame_values = set(source.obs["coordinate_frame"].astype(str).unique())
        if len(frame_values) != 1:
            raise ValueError(f"Candidate has multiple coordinate frames: {sorted(frame_values)}")
        frame = next(iter(frame_values))
        records = []
        with h5py.File(store.path, "r") as handle:
            for section_id in sorted(np.unique(sections)):
                sid = str(int(section_id))
                if sid not in store.metadata:
                    raise KeyError(f"Feature store lacks anchor section {sid}")
                section_mask = sections == int(section_id)
                section_points = points[section_mask]
                metadata = store.metadata[sid]
                valid_grid = handle["sections"][store._key(sid)]["valid"][:]
                boundary = feature_grid_boundary(
                    valid_grid,
                    origin_um=metadata["origin_um"],
                    spacing_um=metadata["spacing_um"],
                    max_points=max_boundary_points,
                )
                _, feature_valid, quality = store.get_feature(
                    sid,
                    section_points,
                    return_valid=True,
                    return_quality=True,
                    nearest_max_distance_um=0.0,
                )
                feature_valid = np.asarray(feature_valid, dtype=bool)
                quality = np.asarray(quality, dtype=np.uint8)
                records.append(
                    {
                        "section_id": int(section_id),
                        "z_um": float(metadata["z_um"]),
                        "points": section_points,
                        "local_points": local_points[section_mask] if local_points is not None else None,
                        "feature_boundary": boundary,
                        "n_cells": int(len(section_points)),
                        "feature_valid_fraction": float(feature_valid.mean()),
                        "quality_1_fraction": float((quality == 1).mean()),
                        "quality_2_fraction": float((quality == 2).mean()),
                        "quality_3_fraction": float((quality == 3).mean()),
                        "valid_grid_fraction": float(valid_grid.mean()),
                        "feature_store_path": str(feature_store_path),
                    }
                )
        return records, frame
    finally:
        source.file.close()


def _plot_xy(
    records: list[dict],
    output: Path,
    *,
    include_masks: bool,
    title: str,
    boundary_key: str = "mask_boundary",
    metric_key: str = "inside_mask_fraction",
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(records)))
    fig, ax = plt.subplots(figsize=(12, 10), constrained_layout=True)
    for color, record in zip(colors, records):
        boundary = record.get(boundary_key, np.empty((0, 2)))
        if include_masks and len(boundary):
            # Boundary samples are returned in raster order, not contour
            # traversal order.  Connecting them would create artificial long
            # lines across the tissue; draw them as independent support dots.
            ax.scatter(
                boundary[:, 0],
                boundary[:, 1],
                s=2.0,
                color=color,
                alpha=0.65,
                linewidths=0,
                rasterized=True,
            )
        points = _downsample(record["points"], 12000)
        metric = float(record.get(metric_key, np.nan))
        metric_label = f"{metric:.3f}" if np.isfinite(metric) else "NA"
        ax.scatter(
            points[:, 0], points[:, 1], s=0.45, color=color, alpha=0.28,
            linewidths=0, rasterized=True, label=f"section {record['section_id']} ({metric_label})",
        )
        center = record["points"].mean(axis=0)
        ax.text(center[0], center[1], str(record["section_id"]), fontsize=8, color=color)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("global X (µm)")
    ax.set_ylabel("global Y (µm)")
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8, frameon=False)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def _plot_51_61(
    records: list[dict], output: Path, *, boundary_key: str = "mask_boundary"
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    selected = [record for record in records if record["section_id"] in {51, 61}]
    if len(selected) != 2:
        raise ValueError("Expected sections 51 and 61 for seam QC")
    colors = {51: "#e31a1c", 61: "#1a9850"}
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    for ax, mode in zip(axes, ("overlay", "separate")):
        for record in selected:
            sid = record["section_id"]
            color = colors[sid]
            boundary = record.get(boundary_key, np.empty((0, 2)))
            points = _downsample(record["points"], 25000)
            if mode == "separate":
                alpha = 0.7 if sid == 51 else 0.0
                point_alpha = 0.28 if sid == 51 else 0.0
            else:
                alpha = 0.8
                point_alpha = 0.3
            if len(boundary):
                ax.scatter(
                    boundary[:, 0],
                    boundary[:, 1],
                    s=3.0,
                    color=color,
                    alpha=alpha,
                    linewidths=0,
                    rasterized=True,
                )
            ax.scatter(points[:, 0], points[:, 1], s=0.5, color=color, alpha=point_alpha, linewidths=0, rasterized=True)
        if mode == "separate":
            record = selected[1]
            boundary = record.get(boundary_key, np.empty((0, 2)))
            points = _downsample(record["points"], 25000)
            ax.scatter(
                boundary[:, 0],
                boundary[:, 1],
                s=3.0,
                color=colors[61],
                alpha=0.8,
                linewidths=0,
                rasterized=True,
            )
            ax.scatter(
                points[:, 0],
                points[:, 1],
                s=0.5,
                color=colors[61],
                alpha=0.3,
                linewidths=0,
                rasterized=True,
            )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("global X (µm)")
        ax.set_ylabel("global Y (µm)")
        ax.set_title("51 red / 61 green — " + ("overlay" if mode == "overlay" else "separate visibility"))
    fig.savefig(output, dpi=240)
    plt.close(fig)


def _plot_3d(records: list[dict], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(records)))
    fig = plt.figure(figsize=(12, 9), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    for color, record in zip(colors, records):
        points = _downsample(record["points"], 5000)
        ax.scatter(points[:, 0], points[:, 1], np.full(len(points), record["z_um"]), s=0.25, color=color, alpha=0.25, linewidths=0, rasterized=True)
    ax.set_xlabel("X (µm)")
    ax.set_ylabel("Y (µm)")
    ax.set_zlabel("Z (µm)")
    ax.set_title("9957/g0 repaired global-frame ST anchors")
    fig.savefig(output, dpi=220)
    plt.close(fig)


def render_qc(
    candidate: Path,
    he_table: Path,
    feature_store: Path,
    output: Path,
    path_summary: Path,
    uot_metadata: Path,
) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite QC directory: {output}")
    output.mkdir(parents=True)
    records, frame = _load_feature_records(
        candidate, feature_store, max_boundary_points=8000
    )
    _plot_xy(
        records,
        output / "st_anchor_only_global_xy.png",
        include_masks=False,
        metric_key="feature_valid_fraction",
        title=f"9957/g0 ST anchors — {frame}",
    )
    _plot_xy(
        records,
        output / "registered_he_feature_support_and_st_global_xy.png",
        include_masks=True,
        boundary_key="feature_boundary",
        metric_key="feature_valid_fraction",
        title=f"9957/g0 registered H&E/UNI2 support + ST anchors — {frame}",
    )
    _plot_51_61(
        records,
        output / "section-051__section-061__feature_support_st_overlay.png",
        boundary_key="feature_boundary",
    )
    _plot_3d(records, output / "st_anchor_global_3d.png")

    centers = []
    for record in records:
        centers.append({
            "section_id": record["section_id"],
            "z_um": record["z_um"],
            "n_cells": record["n_cells"],
            "feature_valid_fraction": record["feature_valid_fraction"],
            "quality_1_fraction": record["quality_1_fraction"],
            "quality_2_fraction": record["quality_2_fraction"],
            "quality_3_fraction": record["quality_3_fraction"],
            "valid_grid_fraction": record["valid_grid_fraction"],
            "center_x_um": float(record["points"][:, 0].mean()),
            "center_y_um": float(record["points"][:, 1].mean()),
        })
    metrics = pd.DataFrame(centers)
    metrics.to_csv(output / "section_alignment_metrics.csv", index=False)
    center_51 = metrics.loc[metrics.section_id == 51, ["center_x_um", "center_y_um"]].to_numpy()[0]
    center_61 = metrics.loc[metrics.section_id == 61, ["center_x_um", "center_y_um"]].to_numpy()[0]
    manifest = {
        "status": "qc_ready_for_review",
        "candidate": str(candidate.resolve()),
        "he_table": str(he_table.resolve()),
        "coordinate_frame": frame,
        "figures": [
            str((output / "st_anchor_only_global_xy.png").resolve()),
            str((output / "registered_he_feature_support_and_st_global_xy.png").resolve()),
            str((output / "section-051__section-061__feature_support_st_overlay.png").resolve()),
            str((output / "st_anchor_global_3d.png").resolve()),
        ],
        "section_count": len(records),
        "total_st_cells": int(sum(record["n_cells"] for record in records)),
        "feature_store": str(feature_store.resolve()),
        "support_semantics": (
            "valid H&E/UNI2 feature-grid support in the exact global frame used by "
            "morphology-aware UOT and morphology-guided path; not a raw crop mask "
            "transformed by an incomplete affine"
        ),
        "mean_feature_valid_fraction": float(metrics.feature_valid_fraction.mean()),
        "weighted_feature_valid_fraction": float(
            np.sum(metrics.feature_valid_fraction * metrics.n_cells)
            / max(metrics.n_cells.sum(), 1)
        ),
        "section_51_to_61_center_displacement_um": float(np.linalg.norm(center_61 - center_51)),
        "path_cache_summary": str(path_summary.resolve()),
        "uot_metadata": str(uot_metadata.resolve()),
        "path_cache_frame": json.loads(path_summary.read_text())["coordinate_frame"],
        "uot_frame": json.loads(uot_metadata.read_text())["coordinate_frame"],
        "section_metrics": centers,
        "deprecated_raw_mask_overlay": (
            "The earlier he_mask_and_st_global_xy.png was not used here: its "
            "direct crop-pixel-to-global transform omitted the saved nonlinear "
            "registration chain and is retained only as a debugging artifact."
        ),
        "next_step": "Wait for user confirmation before training/reconstruction.",
    }
    (output / "qc_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--he-table", type=Path, default=DEFAULT_HE_TABLE)
    parser.add_argument("--feature-store", type=Path, default=DEFAULT_FEATURE_STORE)
    parser.add_argument("--path-summary", type=Path, default=DEFAULT_PATH_SUMMARY)
    parser.add_argument("--uot-metadata", type=Path, default=DEFAULT_UOT_METADATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = render_qc(
        args.candidate,
        args.he_table,
        args.feature_store,
        args.output,
        args.path_summary,
        args.uot_metadata,
    )
    print(json.dumps({
        "status": result["status"],
        "output": str(args.output.resolve()),
        "coordinate_frame": result["coordinate_frame"],
        "weighted_feature_valid_fraction": result["weighted_feature_valid_fraction"],
        "section_51_to_61_center_displacement_um": result["section_51_to_61_center_displacement_um"],
    }, indent=2))


if __name__ == "__main__":
    main()
