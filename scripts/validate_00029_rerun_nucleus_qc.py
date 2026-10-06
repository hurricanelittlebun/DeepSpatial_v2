"""Validate rerun Cellpose nuclei against same-slice Xenium coordinates.

The figures deliberately use the native level-3 SDPC crop and the local
companion-H&E coordinate frame.  No STalign or residual transform is applied:
this tests the direct claim that the rerun nuclei and same-slice ST occupy the
same H&E crop.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from scipy.spatial import cKDTree
from skimage.segmentation import find_boundaries


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepspatial.histology.coordinate_validation import (  # noqa: E402
    local_um_to_output_pixel_centers,
)
from deepspatial.histology.sdpc import SdpcPyramid  # noqa: E402


ANCHORS = (1, 11, 21, 31, 41, 51, 61, 71, 81, 91)
DEFAULT_DATASET = ROOT / "data" / "00029_g0"
DEFAULT_HE_TABLE = DEFAULT_DATASET / "he_sections.parquet"
DEFAULT_SEGMENTATION = (
    DEFAULT_DATASET / "nucleus_path_highres_rerun_v3" / "segmentation"
)
DEFAULT_OUTPUT = DEFAULT_DATASET / "qc" / "highres_nucleus_rerun_v3"


def _read_local_st(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        value = np.asarray(handle["obsm"]["spatial_he_local"], dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 2 or not np.isfinite(value).all():
        raise ValueError(f"{path}: spatial_he_local must be finite [N, 2]")
    return value


def _rgb(value: np.ndarray) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"Expected RGB/RGBA image, got {image.shape}")
    return image[..., :3].astype(np.uint8, copy=False)


def _valid_points(points: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    return (
        np.isfinite(points).all(axis=1)
        & (points[:, 0] >= 0)
        & (points[:, 0] < width)
        & (points[:, 1] >= 0)
        & (points[:, 1] < height)
    )


def _plot_section(
    path: Path,
    image: np.ndarray,
    labels: np.ndarray,
    st_px: np.ndarray,
    nuclei_px: np.ndarray,
    section_id: int,
    metrics: dict,
) -> None:
    boundary = find_boundaries(labels, mode="inner")
    nucleus_overlay = image.copy()
    nucleus_overlay[boundary] = [255, 40, 40]
    st_stride = max(1, int(np.ceil(len(st_px) / 25000)))
    nuc_stride = max(1, int(np.ceil(len(nuclei_px) / 12000)))

    fig, axes = plt.subplots(2, 2, figsize=(16, 14), constrained_layout=True)
    axes[0, 0].imshow(nucleus_overlay)
    axes[0, 0].set_title(
        f"Native level-3 H&E + Cellpose nuclear boundaries\n"
        f"nuclei={metrics['n_nuclei']:,} | mpp={metrics['mpp_um']:.4f}"
    )

    axes[0, 1].imshow(image)
    axes[0, 1].scatter(
        st_px[::st_stride, 0], st_px[::st_stride, 1],
        s=1.0, c="#00d7ff", alpha=0.45, linewidths=0,
        rasterized=True, label="Xenium spatial_he_local",
    )
    axes[0, 1].scatter(
        nuclei_px[::nuc_stride, 0], nuclei_px[::nuc_stride, 1],
        s=5.0, c="#ffe600", alpha=0.80, linewidths=0,
        rasterized=True, label="Cellpose nucleus center",
    )
    axes[0, 1].set_title(
        "Same local H&E crop: cyan=ST cells, yellow=nucleus centers\n"
        f"ST in image={metrics['st_valid_fraction']:.4f}"
    )
    axes[0, 1].legend(loc="upper right", fontsize=8, framealpha=0.85)

    axes[1, 0].imshow(nucleus_overlay)
    axes[1, 0].scatter(
        st_px[::st_stride, 0], st_px[::st_stride, 1],
        s=1.0, c="#00d7ff", alpha=0.45, linewidths=0,
        rasterized=True, label="ST cell",
    )
    axes[1, 0].set_title(
        "ST cells over Cellpose boundaries\n"
        "red=instance boundary; cyan=ST local coordinate"
    )
    axes[1, 0].legend(loc="upper right", fontsize=8, framealpha=0.85)

    axes[1, 1].imshow(image)
    axes[1, 1].scatter(
        st_px[::st_stride, 0], st_px[::st_stride, 1],
        s=1.0, c="#00d7ff", alpha=0.45, linewidths=0,
        rasterized=True,
    )
    axes[1, 1].scatter(
        nuclei_px[::nuc_stride, 0], nuclei_px[::nuc_stride, 1],
        s=5.0, c="#ffe600", alpha=0.80, linewidths=0,
        rasterized=True,
    )
    axes[1, 1].set_title(
        "Direct coordinate audit\n"
        f"nearest ST↔nucleus median={metrics['nearest_median_um']:.1f} µm; "
        f"p95={metrics['nearest_p95_um']:.1f} µm"
    )

    for axis in axes.flat:
        axis.set_xlim(-0.5, image.shape[1] - 0.5)
        axis.set_ylim(image.shape[0] - 0.5, -0.5)
        axis.set_axis_off()
    fig.suptitle(
        f"00029/g0 section {section_id:03d} | rerun high-resolution nucleus QC",
        fontsize=15,
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_overview(path: Path, records: list[dict]) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(22, 10), constrained_layout=True)
    for axis, record in zip(axes.flat, records):
        image = record["thumbnail"]
        axis.imshow(image)
        scale_x = image.shape[1] / record["image_shape"][1]
        scale_y = image.shape[0] / record["image_shape"][0]
        st = record["st_px"]
        nuc = record["nuclei_px"]
        axis.scatter(
            st[:, 0] * scale_x, st[:, 1] * scale_y,
            s=0.35, c="#00d7ff", alpha=0.35, linewidths=0, rasterized=True,
        )
        axis.scatter(
            nuc[:, 0] * scale_x, nuc[:, 1] * scale_y,
            s=1.5, c="#ffe600", alpha=0.7, linewidths=0, rasterized=True,
        )
        axis.set_title(
            f"section {record['section_id']:03d}\n"
            f"nuclei={record['n_nuclei']:,}, ST-in={record['st_valid_fraction']:.3f}",
            fontsize=9,
        )
        axis.set_axis_off()
    fig.suptitle(
        "00029/g0 native high-resolution H&E: cyan=ST, yellow=nucleus centers",
        fontsize=15,
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict:
    he_table_path = Path(args.he_table).resolve()
    segmentation = Path(args.segmentation_dir).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing QC output: {output}")
    sections = pd.read_parquet(he_table_path).copy()
    sections["section_id"] = pd.to_numeric(sections["section_id"], errors="raise").astype(int)
    rows = sections.set_index("section_id", drop=False)
    output.mkdir(parents=True, exist_ok=False)
    records: list[dict] = []

    grouped = rows.loc[list(ANCHORS)].groupby("raw_source_path", sort=True)
    for raw_path_value, group in grouped:
        raw_path = Path(str(raw_path_value)).resolve()
        with SdpcPyramid(raw_path) as reader:
            raw_mpp = float(reader.mpp_level0)
            for section_id, row in group.sort_index().iterrows():
                section_dir = segmentation / f"section-{int(section_id):03d}"
                meta_path = section_dir / "raw_crop_metadata.json"
                nuclei_path = section_dir / "nuclei.csv"
                labels_path = section_dir / "nuclei.tif"
                anchor_path = DEFAULT_DATASET / "anchors" / f"section-{int(section_id):03d}.h5ad"
                for required in (meta_path, nuclei_path, labels_path, anchor_path):
                    if not required.is_file():
                        raise FileNotFoundError(required)
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                image = _rgb(
                    reader.read_region(
                        tuple(int(v) for v in meta["read_location_level0"]),
                        int(meta["selected_level"]),
                        tuple(int(v) for v in meta["read_size_level_px"]),
                    )
                )
                labels = np.asarray(tifffile.imread(labels_path), dtype=np.uint32)
                if labels.shape != image.shape[:2]:
                    raise ValueError(
                        f"section {section_id}: labels {labels.shape} != image {image.shape[:2]}"
                    )
                st_um = _read_local_st(anchor_path)
                st_px = local_um_to_output_pixel_centers(
                    st_um,
                    local_origin_level0=tuple(float(v) for v in meta["requested_bbox_level0"][:2]),
                    output_origin_level0=tuple(float(v) for v in meta["read_location_level0"]),
                    raw_mpp_um=raw_mpp,
                    output_downsample=float(meta["level_downsample"]),
                )
                nuclei = pd.read_csv(nuclei_path)
                nuclei_px = nuclei[["x_px", "y_px"]].to_numpy(dtype=np.float64)
                st_valid = _valid_points(st_px, image.shape[:2])
                nuc_valid = _valid_points(nuclei_px, image.shape[:2])
                if not st_valid.any() or not nuc_valid.any():
                    raise ValueError(f"section {section_id}: no valid points for QC")
                tree = cKDTree(nuclei_px[nuc_valid])
                nearest_px, _ = tree.query(st_px[st_valid], k=1)
                nearest_um = nearest_px * float(meta["selected_level_mpp_um"])
                metrics = {
                    "section_id": int(section_id),
                    "n_st_cells": int(len(st_px)),
                    "n_nuclei": int(len(nuclei_px)),
                    "st_valid_fraction": float(st_valid.mean()),
                    "nucleus_valid_fraction": float(nuc_valid.mean()),
                    "nearest_median_um": float(np.median(nearest_um)),
                    "nearest_p95_um": float(np.percentile(nearest_um, 95)),
                    "image_shape_hw": [int(v) for v in image.shape[:2]],
                    "mpp_um": float(meta["selected_level_mpp_um"]),
                    "selected_level": int(meta["selected_level"]),
                    "raw_source_path": str(raw_path),
                    "raw_crop_metadata": str(meta_path),
                    "anchor_h5ad": str(anchor_path),
                    "nuclei_csv": str(nuclei_path),
                    "coordinate_chain": (
                        "spatial_he_local (um) -> same requested SDPC crop -> "
                        "native level-3 pixels; nuclei.csv x_px/y_px are native crop pixels"
                    ),
                }
                figure_path = output / f"section-{int(section_id):03d}_he_st_nucleus_qc.png"
                _plot_section(
                    figure_path,
                    image,
                    labels,
                    st_px,
                    nuclei_px,
                    int(section_id),
                    metrics,
                )
                metrics["figure"] = str(figure_path)
                records.append(
                    {
                        **metrics,
                        "thumbnail": np.asarray(
                            Image.fromarray(image).resize(
                                (min(600, image.shape[1]),
                                 int(round(image.shape[0] * min(600, image.shape[1]) / image.shape[1]))),
                                Image.Resampling.BILINEAR,
                            )
                        ),
                        "image_shape": list(image.shape[:2]),
                        "st_px": st_px,
                        "nuclei_px": nuclei_px,
                    }
                )

    records.sort(key=lambda item: item["section_id"])
    frame = pd.DataFrame(
        [{key: value for key, value in record.items() if key not in {"thumbnail", "st_px", "nuclei_px"}}
         for record in records]
    )
    frame.to_csv(output / "metrics.csv", index=False)
    _plot_overview(output / "overview_he_st_nucleus.png", records)
    report = {
        "format": "deepspatial-00029-rerun-nucleus-qc-v1",
        "status": "completed",
        "n_sections": len(records),
        "sections": [
            {key: value for key, value in record.items() if key not in {"thumbnail", "st_px", "nuclei_px"}}
            for record in records
        ],
        "segmentation_dir": str(segmentation),
        "he_table": str(he_table_path),
        "no_registration_transform": True,
        "interpretation": (
            "This is a direct same-slice local-frame audit. It does not claim that "
            "the STalign registered frame equals the unwarped native H&E crop."
        ),
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# 00029/g0 rerun high-resolution nucleus QC\n\n"
        "Cyan points are `spatial_he_local` from the anchor H5AD. Yellow points "
        "are Cellpose nucleus centers from the rerun native SDPC level-3 crop. "
        "Red curves are Cellpose instance boundaries. No STalign or residual "
        "transform is applied in these figures.\n\n"
        "- `overview_he_st_nucleus.png`: all ten anchors.\n"
        "- `section-*_he_st_nucleus_qc.png`: per-anchor four-panel diagnostic.\n"
        "- `metrics.csv`: point validity and nearest-nucleus summaries.\n"
        "- `report.json`: provenance and coordinate chain.\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--he-table", type=Path, default=DEFAULT_HE_TABLE)
    parser.add_argument("--segmentation-dir", type=Path, default=DEFAULT_SEGMENTATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"status": report["status"], "n_sections": report["n_sections"],
                      "output_dir": str(Path(args.output_dir).resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
