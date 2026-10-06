#!/usr/bin/env python3
"""Validate corrected ST, nuclear centroids, and native high-resolution H&E.

The comparison frame is ``00029__g0_fixed_he``.  ST and nuclear centroids are
converted back to the same native H&E crop only for visualization; the source
H&E image and Cellpose instance masks are never warped or overwritten.
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
from PIL import Image
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from deepspatial.histology.frame_resample import transform_points  # noqa: E402


OUTPUT_FRAME = "00029__g0_fixed_he"
ANCHORS = (1, 11, 21, 31, 41, 51, 61, 71, 81, 91)
DEFAULT_DATASET = ROOT / "data/00029_g0"
DEFAULT_NUCLEI = DEFAULT_DATASET / "fixed_he_single_cell_v1"
DEFAULT_CANDIDATE = DEFAULT_DATASET / "anchors_st_to_fixed_he_candidate"
DEFAULT_HE_TABLE = DEFAULT_DATASET / "he_sections.parquet"
DEFAULT_HIGH_ROOT = (
    DEFAULT_DATASET
    / "nucleus_path_raw_v1_coordinate_corrected_v2"
    / "segmentation"
)
DEFAULT_RESIDUAL = DEFAULT_DATASET / "registration_correction_v1"
DEFAULT_FEATURE_FRAME = DEFAULT_DATASET / "fixed_he_feature_frame_v2_interpolated"
DEFAULT_REGISTRATION = Path("/data/buyonggan/DeepSpatial_v2/outputs/registration")
DEFAULT_SDPC = Path("/data/buyonggan/DeepSpatial_v2/data/0106548.sdpc")
DEFAULT_OUTPUT = DEFAULT_DATASET / "qc/fixed_he_three_way_v1"


def validate_fixed_frame_contract(
    st_frame: str,
    nucleus_frame: str,
    feature_frame: str,
) -> str:
    """Require ST, nuclear tables, and morphology stores to share one frame."""

    frames = {
        "st": str(st_frame),
        "nucleus": str(nucleus_frame),
        "feature": str(feature_frame),
    }
    if set(frames.values()) != {OUTPUT_FRAME}:
        raise ValueError(
            "Fixed-frame contract failed: "
            + ", ".join(f"{key}={value}" for key, value in frames.items())
        )
    return OUTPUT_FRAME


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_candidate_st(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read cell IDs and corrected ST coordinates without loading expression."""

    with h5py.File(path, "r") as handle:
        if "obsm/spatial_st_corrected" not in handle:
            raise KeyError(f"{path}: missing obsm/spatial_st_corrected")
        points = np.asarray(handle["obsm/spatial_st_corrected"], dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError(f"{path}: invalid spatial_st_corrected shape or values")
        obs = handle["obs"]
        index_name = obs.attrs.get("_index", "")
        if isinstance(index_name, bytes):
            index_name = index_name.decode()
        index_name = str(index_name)
        if index_name and index_name in obs:
            raw_ids = np.asarray(obs[index_name], dtype=object)
        elif "_index" in obs:
            raw_ids = np.asarray(obs["_index"], dtype=object)
        else:
            raw_ids = np.arange(len(points), dtype=np.int64)
    ids = np.asarray(
        [value.decode() if isinstance(value, bytes) else str(value) for value in raw_ids]
    )
    if len(ids) != len(points):
        raise ValueError(f"{path}: cell ID count does not match coordinate count")
    return ids, points


def _sample_mask(mask: np.ndarray, points_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(points_xy, dtype=np.float64)
    height, width = mask.shape
    valid = (
        np.isfinite(value).all(axis=1)
        & (value[:, 0] >= 0.0)
        & (value[:, 0] <= width - 1)
        & (value[:, 1] >= 0.0)
        & (value[:, 1] <= height - 1)
    )
    inside = np.zeros(len(value), dtype=bool)
    if valid.any():
        columns = np.rint(value[valid, 0]).astype(int)
        rows = np.rint(value[valid, 1]).astype(int)
        inside[valid] = mask[rows, columns]
    return inside, valid


def _load_highres_context(row, high_root: Path, reader):
    """Load a native high-res image and map the reviewed low-res mask to it."""

    from validate_00029_st_highres_qc import _high_mask_from_low, _rgb

    source_image_path = Path(str(row["source_image_path"]))
    low_meta = _read_json(source_image_path.parent / "metadata.json")
    low_mask = np.asarray(
        Image.open(source_image_path.parent / f"he_foreground_mask_level{int(low_meta['crop_level'])}.png").convert("L")
    ) > 0
    high_meta = _read_json(
        high_root / f"section-{int(row['section_id']):03d}" / "raw_crop_metadata.json"
    )
    high_level = int(high_meta["selected_level"])
    high_downsample = float(high_meta["level_downsample"])
    high_origin = tuple(int(value) for value in high_meta["read_location_level0"])
    high_size = tuple(int(value) for value in high_meta["read_size_level_px"])
    image = _rgb(reader.read_region(high_origin, high_level, high_size))
    low_level = int(low_meta["crop_level"])
    low_downsample = float(reader.level_downsamples[low_level])
    requested_bbox = tuple(float(value) for value in low_meta["crop_level0_bbox"])
    from deepspatial.histology.coordinate_validation import effective_level0_origin

    low_origin = effective_level0_origin(requested_bbox[:2], low_downsample)
    high_mask, _ = _high_mask_from_low(
        low_mask,
        image.shape[:2],
        low_origin_level0=low_origin,
        low_downsample=low_downsample,
        high_origin_level0=high_origin,
        high_downsample=high_downsample,
    )
    return image, high_mask, low_meta, high_meta, requested_bbox, high_origin


def _to_highres_pixels(
    fixed_points: np.ndarray,
    matrix_source_to_fixed: np.ndarray,
    *,
    preorientation: np.ndarray,
    chain,
    maps,
    requested_bbox,
    high_origin,
    raw_mpp,
    high_downsample,
):
    from validate_00029_st_highres_qc import source_points_from_registered
    from deepspatial.histology.coordinate_validation import local_um_to_output_pixel_centers

    registered = transform_points(
        fixed_points,
        np.linalg.inv(np.asarray(matrix_source_to_fixed, dtype=np.float64)),
    )
    source = source_points_from_registered(registered, preorientation, chain, maps)
    pixels = local_um_to_output_pixel_centers(
        source,
        local_origin_level0=requested_bbox[:2],
        output_origin_level0=high_origin,
        raw_mpp_um=raw_mpp,
        output_downsample=high_downsample,
    )
    return registered, source, pixels


def _plot_three_way(path, image, st_px, st_inside, st_valid, nuclei_px, nuclei_inside, nuclei_valid, metrics):
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), constrained_layout=True)
    axes[0, 0].imshow(image)
    st_colors = np.where(
        (st_valid & st_inside)[:, None],
        np.asarray([0.1, 0.85, 0.2, 0.55]),
        np.asarray([0.9, 0.1, 0.1, 0.55]),
    )
    axes[0, 0].scatter(st_px[:, 0], st_px[:, 1], s=1.0, c=st_colors, linewidths=0, rasterized=True)
    axes[0, 0].set_title(
        f"candidate ST on native high-res H&E\n"
        f"inside tissue={metrics['st_inside_tissue_given_valid_fraction']:.3f}"
    )

    axes[0, 1].imshow(image)
    nuclei_colors = np.where(
        (nuclei_valid & nuclei_inside)[:, None],
        np.asarray([1.0, 0.75, 0.05, 0.70]),
        np.asarray([0.9, 0.1, 0.1, 0.70]),
    )
    axes[0, 1].scatter(
        nuclei_px[:, 0], nuclei_px[:, 1], s=5.0, c=nuclei_colors, linewidths=0, rasterized=True
    )
    axes[0, 1].set_title(
        f"fixed-frame nuclei on native high-res H&E\n"
        f"inside tissue={metrics['nucleus_inside_tissue_given_valid_fraction']:.3f}"
    )

    axes[1, 0].imshow(image)
    axes[1, 0].scatter(st_px[:, 0], st_px[:, 1], s=0.8, c="#00d7ff", alpha=0.38, linewidths=0, rasterized=True, label="ST")
    axes[1, 0].scatter(nuclei_px[:, 0], nuclei_px[:, 1], s=5.0, c="#ffb000", alpha=0.65, linewidths=0, rasterized=True, label="nucleus")
    axes[1, 0].legend(loc="upper right", fontsize=9)
    axes[1, 0].set_title(
        f"three-way overlay\n"
        f"ST→nearest nucleus median={metrics['nearest_nucleus_distance_median_um']:.1f} µm"
    )

    axes[1, 1].scatter(st_fixed := metrics["st_fixed_xy"][:, 0], metrics["st_fixed_xy"][:, 1], s=1.0, c="#00a6d6", alpha=0.35, linewidths=0, rasterized=True, label="ST fixed")
    axes[1, 1].scatter(metrics["nucleus_fixed_xy"][:, 0], metrics["nucleus_fixed_xy"][:, 1], s=4.0, c="#e69f00", alpha=0.60, linewidths=0, rasterized=True, label="nucleus fixed")
    axes[1, 1].set_aspect("equal")
    axes[1, 1].legend(loc="best", fontsize=9)
    axes[1, 1].set_xlabel("fixed H&E x (µm)")
    axes[1, 1].set_ylabel("fixed H&E y (µm)")
    axes[1, 1].set_title("same fixed physical coordinate frame")
    for axis in axes.flat[:3]:
        axis.set_xlim(-0.5, image.shape[1] - 0.5)
        axis.set_ylim(image.shape[0] - 0.5, -0.5)
        axis.set_axis_off()
    fig.suptitle(
        f"00029/g0 section {int(metrics['section_id']):03d} | "
        "fixed H&E / ST / nuclei consistency QC",
        fontsize=15,
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict:
    dataset = Path(args.dataset_dir).resolve()
    nuclei_dir = Path(args.nuclei_dir).resolve()
    candidate_dir = Path(args.candidate_dir).resolve()
    he_table_path = Path(args.he_table).resolve()
    high_root = Path(args.high_root).resolve()
    residual_dir = Path(args.residual_dir).resolve()
    feature_frame_dir = Path(args.feature_frame_dir).resolve()
    registration_root = Path(args.registration_root).resolve()
    sdpc_path = Path(args.sdpc_path).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.mkdir(parents=True, exist_ok=False)

    nucleus_manifest = _read_json(nuclei_dir / "manifest.json")
    feature_manifest = _read_json(feature_frame_dir / "materialization_manifest.json")
    candidate_manifest = _read_json(candidate_dir / "manifest.json")
    if candidate_manifest.get("coordinate_key") != "spatial_st_corrected":
        raise ValueError("Candidate ST manifest does not select spatial_st_corrected")
    validate_fixed_frame_contract(
        OUTPUT_FRAME,
        nucleus_manifest.get("output_coordinate_frame", ""),
        feature_manifest.get("output_coordinate_frame", ""),
    )

    he_table = pd.read_parquet(he_table_path).copy()
    nuclei = pd.read_parquet(nuclei_dir / "nuclei_fixed.parquet")
    required_nucleus = {"section_id", "cell_id", "x_fixed_he_um", "y_fixed_he_um", "coordinate_frame"}
    missing = sorted(required_nucleus - set(nuclei.columns))
    if missing:
        raise KeyError(f"Fixed nuclear table lacks columns: {missing}")
    if set(nuclei["coordinate_frame"].astype(str)) != {OUTPUT_FRAME}:
        raise ValueError("Fixed nuclear table contains mixed coordinate frames")
    if not np.isfinite(nuclei[["x_fixed_he_um", "y_fixed_he_um"]].to_numpy(float)).all():
        raise ValueError("Fixed nuclear table contains non-finite coordinates")

    from materialize_00029_fixed_he_feature_stores import build_section_affines
    from validate_00029_st_highres_qc import (
        RegistrationMaps,
        _format_axes,
        read_section_transform,
        source_points_from_registered,
    )

    matrices, transform_records = build_section_affines(
        he_table,
        residual_dir,
        interpolate_serial_affine=True,
    )
    transform_by_id = {str(item["section_id"]): item for item in transform_records}
    maps = RegistrationMaps(registration_root, "00029", "g0")
    rows = []
    overview = []

    with __import__("deepspatial.histology.sdpc", fromlist=["SdpcPyramid"]).SdpcPyramid(sdpc_path) as reader:
        raw_mpp = float(reader.mpp_level0)
        for section_id in ANCHORS:
            sid = str(section_id)
            section_row = he_table.loc[he_table["section_id"].eq(section_id)].iloc[0]
            candidate_path = candidate_dir / f"section-{section_id:03d}__st_to_fixed_he_candidate.h5ad"
            if not candidate_path.is_file():
                raise FileNotFoundError(candidate_path)
            st_ids, st_fixed = _read_candidate_st(candidate_path)
            nucleus_frame = nuclei.loc[nuclei["section_id"].astype(int).eq(section_id)].copy()
            nucleus_fixed = nucleus_frame[["x_fixed_he_um", "y_fixed_he_um"]].to_numpy(float)
            matrix = np.asarray(matrices[sid], dtype=float)
            transform_record = transform_by_id[sid]

            section_transform_path = Path(str(section_row["section_transform_path"]))
            preorientation, chain = read_section_transform(section_transform_path)
            image, high_mask, low_meta, high_meta, requested_bbox, high_origin = _load_highres_context(
                section_row, high_root, reader
            )
            st_registered, st_source, st_px = _to_highres_pixels(
                st_fixed,
                matrix,
                preorientation=preorientation,
                chain=chain,
                maps=maps,
                requested_bbox=requested_bbox,
                high_origin=high_origin,
                raw_mpp=raw_mpp,
                high_downsample=float(high_meta["level_downsample"]),
            )
            nuclei_registered, nuclei_source, nuclei_px = _to_highres_pixels(
                nucleus_fixed,
                matrix,
                preorientation=preorientation,
                chain=chain,
                maps=maps,
                requested_bbox=requested_bbox,
                high_origin=high_origin,
                raw_mpp=raw_mpp,
                high_downsample=float(high_meta["level_downsample"]),
            )
            st_inside, st_valid = _sample_mask(high_mask, st_px)
            nucleus_inside, nucleus_valid = _sample_mask(high_mask, nuclei_px)
            st_roundtrip = np.max(
                np.linalg.norm(transform_points(st_registered, matrix) - st_fixed, axis=1)
            )
            nucleus_roundtrip = np.max(
                np.linalg.norm(transform_points(nuclei_registered, matrix) - nucleus_fixed, axis=1)
            ) if len(nucleus_fixed) else 0.0
            if len(nucleus_fixed):
                distances = cKDTree(nucleus_fixed).query(st_fixed, k=1)[0]
                nearest_median = float(np.median(distances))
                nearest_p95 = float(np.percentile(distances, 95))
                nearest_50 = float(np.mean(distances <= 50.0))
            else:
                distances = np.full(len(st_fixed), np.nan)
                nearest_median = float("nan")
                nearest_p95 = float("nan")
                nearest_50 = 0.0
            metrics = {
                "section_id": int(section_id),
                "z_um": float(section_row["z_um"]),
                "n_st_cells": int(len(st_fixed)),
                "n_nuclei": int(len(nucleus_fixed)),
                "st_valid_fraction": float(st_valid.mean()),
                "st_inside_tissue_given_valid_fraction": float(st_inside[st_valid].mean()) if st_valid.any() else 0.0,
                "nucleus_valid_fraction": float(nucleus_valid.mean()) if len(nucleus_valid) else 1.0,
                "nucleus_inside_tissue_given_valid_fraction": float(nucleus_inside[nucleus_valid].mean()) if nucleus_valid.any() else 0.0,
                "nearest_nucleus_distance_median_um": nearest_median,
                "nearest_nucleus_distance_p95_um": nearest_p95,
                "st_within_50um_of_nucleus_fraction": nearest_50,
                "st_fixed_to_registered_roundtrip_max_um": float(st_roundtrip),
                "nucleus_fixed_to_registered_roundtrip_max_um": float(nucleus_roundtrip),
                "fixed_transform_type": transform_record["transform_type"],
                "fixed_transform_sha256": transform_record["matrix_sha256"],
                "st_coordinate_key": "spatial_st_corrected",
                "st_coordinate_frame": OUTPUT_FRAME,
                "nucleus_coordinate_frame": OUTPUT_FRAME,
                "feature_coordinate_frame": OUTPUT_FRAME,
                "highres_level": int(high_meta["selected_level"]),
                "highres_pixel_size_um": float(high_meta["selected_level_mpp_um"]),
            }
            section_output = output / f"section-{section_id:03d}"
            section_output.mkdir(parents=True, exist_ok=False)
            point_table = pd.concat(
                [
                    pd.DataFrame(
                        {
                            "entity_type": "st",
                            "entity_id": st_ids,
                            "x_fixed_he_um": st_fixed[:, 0],
                            "y_fixed_he_um": st_fixed[:, 1],
                            "x_registered_um": st_registered[:, 0],
                            "y_registered_um": st_registered[:, 1],
                            "x_source_he_um": st_source[:, 0],
                            "y_source_he_um": st_source[:, 1],
                            "x_highres_px": st_px[:, 0],
                            "y_highres_px": st_px[:, 1],
                            "inside_highres_tissue": st_inside,
                            "valid_highres_pixel": st_valid,
                            "nearest_nucleus_distance_um": distances,
                        }
                    ),
                    pd.DataFrame(
                        {
                            "entity_type": "nucleus",
                            "entity_id": nucleus_frame["cell_id"].astype(str).to_numpy(),
                            "x_fixed_he_um": nucleus_fixed[:, 0],
                            "y_fixed_he_um": nucleus_fixed[:, 1],
                            "x_registered_um": nuclei_registered[:, 0],
                            "y_registered_um": nuclei_registered[:, 1],
                            "x_source_he_um": nuclei_source[:, 0],
                            "y_source_he_um": nuclei_source[:, 1],
                            "x_highres_px": nuclei_px[:, 0],
                            "y_highres_px": nuclei_px[:, 1],
                            "inside_highres_tissue": nucleus_inside,
                            "valid_highres_pixel": nucleus_valid,
                            "nearest_nucleus_distance_um": np.full(len(nucleus_fixed), np.nan),
                        }
                    ),
                ],
                ignore_index=True,
            )
            point_table.to_parquet(section_output / "three_way_points.parquet", index=False)
            metrics["overlay_path"] = str(section_output / "three_way_overlay.png")
            metrics["point_table_path"] = str(section_output / "three_way_points.parquet")
            plot_metrics = dict(metrics, st_fixed_xy=st_fixed, nucleus_fixed_xy=nucleus_fixed)
            _plot_three_way(
                section_output / "three_way_overlay.png",
                image,
                st_px,
                st_inside,
                st_valid,
                nuclei_px,
                nucleus_inside,
                nucleus_valid,
                plot_metrics,
            )
            rows.append(metrics)
            thumbnail = np.asarray(
                Image.fromarray(image).resize(
                    (min(420, image.shape[1]), int(round(image.shape[0] * min(420, image.shape[1]) / image.shape[1]))),
                    Image.Resampling.BILINEAR,
                )
            )
            overview.append(
                {
                    "section_id": int(section_id),
                    "thumbnail": thumbnail,
                    "image_shape": image.shape[:2],
                    "st_px": st_px,
                    "nuclei_px": nuclei_px,
                    "st_valid": st_valid,
                    "nucleus_valid": nucleus_valid,
                    "st_inside": st_inside,
                    "nucleus_inside": nucleus_inside,
                    "st_fraction": metrics["st_inside_tissue_given_valid_fraction"],
                    "nucleus_fraction": metrics["nucleus_inside_tissue_given_valid_fraction"],
                }
            )

    fig, axes = plt.subplots(2, 5, figsize=(20, 9), constrained_layout=True)
    for axis, record in zip(axes.flat, overview):
        axis.imshow(record["thumbnail"])
        sx = record["thumbnail"].shape[1] / record["image_shape"][1]
        sy = record["thumbnail"].shape[0] / record["image_shape"][0]
        axis.scatter(
            record["st_px"][:, 0] * sx,
            record["st_px"][:, 1] * sy,
            s=0.35,
            c="#00d7ff",
            alpha=0.38,
            linewidths=0,
            rasterized=True,
        )
        axis.scatter(
            record["nuclei_px"][:, 0] * sx,
            record["nuclei_px"][:, 1] * sy,
            s=2.0,
            c="#ffb000",
            alpha=0.55,
            linewidths=0,
            rasterized=True,
        )
        axis.set_title(
            f"s{record['section_id']:03d}\n"
            f"ST={record['st_fraction']:.3f}, nuclei={record['nucleus_fraction']:.3f}",
            fontsize=9,
        )
        axis.set_axis_off()
    fig.suptitle("00029/g0 fixed-frame three-way QC | cyan=corrected ST, orange=fixed nuclei", fontsize=14)
    fig.savefig(output / "overview_three_way.png", dpi=160)
    plt.close(fig)

    frame = pd.DataFrame(rows).sort_values("section_id")
    frame.to_csv(output / "section_metrics.csv", index=False)
    report = {
        "format": "deepspatial-fixed-he-three-way-qc-v1",
        "status": "completed",
        "dataset": str(dataset),
        "coordinate_frame": OUTPUT_FRAME,
        "st_coordinate_key": "spatial_st_corrected",
        "candidate_st_directory": str(candidate_dir),
        "nucleus_directory": str(nuclei_dir),
        "fixed_feature_frame_directory": str(feature_frame_dir),
        "n_sections": int(len(rows)),
        "total_st_cells": int(frame["n_st_cells"].sum()),
        "total_nuclei_in_all_sections": int(len(nuclei)),
        "anchor_nuclei": int(frame["n_nuclei"].sum()),
        "fixed_frame_contract": {
            "st": OUTPUT_FRAME,
            "nuclei": OUTPUT_FRAME,
            "features": OUTPUT_FRAME,
            "passed": True,
        },
        "interpretation": (
            "Mask containment and round-trip checks validate coordinate consistency; "
            "nearest-nucleus distances are a sanity check, not one-to-one cell matching."
        ),
        "sections": rows,
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value) + "\n",
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        "# 00029/g0 fixed-frame three-way QC\n\n"
        "This report uses candidate corrected ST coordinates (`spatial_st_corrected`) "
        "and fixed-frame nuclear centroids in `00029__g0_fixed_he`. Both are inverse "
        "mapped into the same native high-resolution H&E crop for visualization. "
        "The H&E image and Cellpose masks are not overwritten.\n\n"
        "Mask containment demonstrates that points land inside the reviewed tissue; "
        "it is not a claim of one-to-one ST-cell/nucleus matching.\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--nuclei-dir", type=Path, default=DEFAULT_NUCLEI)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--he-table", type=Path, default=DEFAULT_HE_TABLE)
    parser.add_argument("--high-root", type=Path, default=DEFAULT_HIGH_ROOT)
    parser.add_argument("--residual-dir", type=Path, default=DEFAULT_RESIDUAL)
    parser.add_argument("--feature-frame-dir", type=Path, default=DEFAULT_FEATURE_FRAME)
    parser.add_argument("--registration-root", type=Path, default=DEFAULT_REGISTRATION)
    parser.add_argument("--sdpc-path", type=Path, default=DEFAULT_SDPC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "n_sections": report["n_sections"],
                "total_st_cells": report["total_st_cells"],
                "total_nuclei_in_all_sections": report["total_nuclei_in_all_sections"],
                "coordinate_frame": report["coordinate_frame"],
                "output_dir": str(Path(args.output_dir).resolve()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
