"""Segment 00029/g0 H&E crops and build a registered nuclear feature grid.

The script is intentionally offline. Cellpose is imported from the execution
environment, while DeepSpatial consumes only the resulting HDF5 descriptor
store. Nuclear descriptors describe local anatomy and are not cell tracking.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
# Make direct ``python scripts/...py`` execution behave like module execution.
# This is needed by the materialization stage, which imports the local
# DeepSpatial package after the Cellpose-only stage has finished.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_HE_TABLE = ROOT / "data/00029_g0/he_sections.parquet"
DEFAULT_UNI2_STORE = ROOT / "data/00029_g0/uni2_features.h5"
DEFAULT_OUTPUT = ROOT / "data/00029_g0/nucleus_path_v1"
DEFAULT_REGISTRATION = Path("/data/buyonggan/DeepSpatial_v2/outputs/registration")
DEFAULT_V2_SRC = Path("/data/buyonggan/DeepSpatial_v2/src")
DEFAULT_CELLPOSE_MODEL = Path("/data/buyonggan/.cellpose/models/nucleitorch_0")


def _grid_from_metadata(metadata):
    height, width, _ = metadata["shape"]
    yy, xx = np.mgrid[:height, :width]
    return np.stack(
        [
            metadata["origin_um"][0] + xx * metadata["spacing_um"][0],
            metadata["origin_um"][1] + yy * metadata["spacing_um"][1],
        ],
        axis=-1,
    ).astype(np.float64)


def _clean_paths(value):
    return str(Path(value).expanduser().resolve())


def _load_local_nucleus_module():
    """Load nucleus helpers without importing deepspatial/__init__.py.

    The Cellpose environment is intentionally smaller than the training
    environment and does not provide pytorch-lightning.  The segmentation
    stage only needs the dependency-light nucleus module, so importing the
    file directly keeps the two environments decoupled.
    """

    path = ROOT / "deepspatial" / "histology" / "nucleus.py"
    spec = importlib.util.spec_from_file_location("deepspatial_nucleus_local", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load local nucleus module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_local_sdpc_module():
    """Load the dependency-light SDPC adapter without importing the package."""

    path = ROOT / "deepspatial" / "histology" / "sdpc.py"
    spec = importlib.util.spec_from_file_location("deepspatial_sdpc_local", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load local SDPC module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _enable_parquet_site_packages(path):
    """Add an optional Parquet engine after the active env's NumPy is loaded."""

    if path is None:
        return
    value = str(Path(path).expanduser().resolve())
    if value not in sys.path:
        # Inserting after the interpreter's standard entries preserves the
        # Cellpose environment's NumPy/OpenCV ABI while making pyarrow
        # available to pandas and the external registration manifest loader.
        sys.path.insert(1, value)


def _read_table(path, *, parquet_site_packages=None):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    try:
        return pd.read_parquet(path)
    except ImportError:
        _enable_parquet_site_packages(parquet_site_packages)
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pandas()


def _load_external_registration(v2_src, registration_root):
    v2_src = str(Path(v2_src).resolve())
    if v2_src not in sys.path:
        sys.path.insert(0, v2_src)
    from deepspatial_v2.data_export.cell_level_alignment import (
        SavedTransformStore,
        load_section_transform,
        transform_section_points,
    )

    return (
        SavedTransformStore(Path(registration_root), device="cpu"),
        load_section_transform,
        transform_section_points,
    )


def _save_qc(image, labels, output_path):
    from skimage.segmentation import find_boundaries

    overlay = image.copy()
    overlay[find_boundaries(labels, mode="inner")] = [0, 255, 0]
    panel = np.concatenate([image, overlay], axis=1)
    Image.fromarray(panel).save(output_path)


def _resize_mask(mask, size):
    """Resize a reviewed low-resolution mask without inventing gray pixels."""

    image = Image.fromarray((np.asarray(mask, dtype=bool) * 255).astype(np.uint8))
    resized = image.resize(tuple(map(int, size)), Image.Resampling.NEAREST)
    return np.asarray(resized, dtype=np.uint8) > 0


def _segment_one(row, model, output_dir, diameter_um, series_id="00029__g0"):
    nucleus = _load_local_nucleus_module()
    import tifffile

    section_id = str(row.section_id)
    section_dir = output_dir / f"section-{int(row.section_id):03d}"
    section_dir.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(Image.open(row.source_image_path).convert("RGB"), dtype=np.uint8)
    tissue = np.asarray(Image.open(row.source_mask_path).convert("L"), dtype=np.uint8) > 0
    mpp = np.asarray(
        [float(row.analysis_pixel_size_x_um), float(row.analysis_pixel_size_y_um)],
        dtype=float,
    )
    if tissue.shape != rgb.shape[:2]:
        raise ValueError(f"{section_id}: image and mask shape mismatch")
    labels, score = nucleus.segment_h_and_e(
        rgb,
        tissue,
        model,
        diameter_px=float(diameter_um) / float(np.mean(mpp)),
    )
    table = nucleus.nuclei_table_from_mask(
        labels,
        score,
        mpp_um=mpp,
        section_id=f"{series_id.replace('__', '_')}_{section_id}",
    )
    if len(table):
        table["section_id"] = section_id
        table["z_um"] = float(row.z_um)
        table["x_source_um"] = (
            table["x_px"].to_numpy(float) + float(row.bbox_x0)
        ) * float(row.analysis_pixel_size_x_um)
        table["y_source_um"] = (
            table["y_px"].to_numpy(float) + float(row.bbox_y0)
        ) * float(row.analysis_pixel_size_y_um)
    else:
        table["section_id"] = pd.Series(dtype=str)
        table["z_um"] = pd.Series(dtype=float)
        table["x_source_um"] = pd.Series(dtype=float)
        table["y_source_um"] = pd.Series(dtype=float)
    table.to_csv(section_dir / "nuclei.csv", index=False)
    tifffile.imwrite(
        section_dir / "nuclei.tif",
        labels.astype(np.uint32),
        compression="zlib",
    )
    if score is not None:
        np.save(section_dir / "cellprob_score.npy", np.asarray(score, dtype=np.float32))
    clean = rgb.copy()
    clean[~tissue] = 255
    _save_qc(clean, labels, section_dir / "qc_original_and_boundaries.png")
    return table, {
        "section_id": section_id,
        "n_nuclei": int(len(table)),
        "image_shape": [int(x) for x in rgb.shape[:2]],
        "mpp_um": mpp.tolist(),
        "status": "segmented",
    }


def _segment_one_raw(
    row,
    model,
    output_dir,
    diameter_um,
    reader,
    sdpc_module,
    *,
    target_mpp,
    save_input_crop=False,
    series_id="00029__g0",
):
    """Segment one reviewed object from its native SDPC crop.

    The reviewed analysis mask supplies the tissue ROI, while the pixels come
    from a common SDPC pyramid level.  This keeps all sections in one physical
    source frame and avoids running Cellpose on an accidentally over-downsampled
    analysis crop.
    """

    nucleus = _load_local_nucleus_module()
    import tifffile

    section_id = str(row.section_id)
    section_dir = output_dir / f"section-{int(row.section_id):03d}"
    section_dir.mkdir(parents=True, exist_ok=True)
    level = sdpc_module.choose_level_for_mpp(
        reader.level_downsamples,
        float(reader.mpp_level0),
        float(target_mpp),
    )
    downsample = float(reader.level_downsamples[level])
    level_mpp = float(reader.mpp_level0) * downsample
    level0_dimensions = tuple(map(int, reader.level_dimensions[0]))
    bbox = (
        float(row.raw_level0_x0),
        float(row.raw_level0_y0),
        float(row.raw_level0_x1),
        float(row.raw_level0_y1),
    )
    location, size, effective_bbox = sdpc_module.level0_bbox_to_read_region(
        bbox,
        downsample=downsample,
        level0_dimensions=level0_dimensions,
    )
    rgb = np.asarray(reader.read_region(location, level, size), dtype=np.uint8)
    lowres_tissue = np.asarray(
        Image.open(row.source_mask_path).convert("L"), dtype=np.uint8
    ) > 0
    tissue = _resize_mask(lowres_tissue, (rgb.shape[1], rgb.shape[0]))
    labels, score = nucleus.segment_h_and_e(
        rgb,
        tissue,
        model,
        diameter_px=float(diameter_um) / level_mpp,
    )
    table = nucleus.nuclei_table_from_mask(
        labels,
        score,
        mpp_um=(level_mpp, level_mpp),
        section_id=f"{series_id.replace('__', '_')}_{section_id}",
    )
    source_coordinate_mode = (
        "companion_crop_local_physical_um"
        if str(row.source_kind) == "xenium_companion_he"
        else "serial_sdpc_global_physical_um"
    )
    if len(table):
        table["section_id"] = section_id
        table["z_um"] = float(row.z_um)
        if str(row.source_kind) == "xenium_companion_he":
            # Companion crops are the local source frame used by the saved
            # H&E/Xenium transform.  Their raw SDPC bbox is only an extraction
            # locator; adding its global origin here would mix two frames.
            source = sdpc_module.level_px_to_source_um(
                table[["x_px", "y_px"]].to_numpy(float),
                level0_origin=(0.0, 0.0),
                downsample=downsample,
                mpp_level0=float(reader.mpp_level0),
            )
        else:
            source = sdpc_module.level_px_to_source_um(
                table[["x_px", "y_px"]].to_numpy(float),
                level0_origin=location,
                downsample=downsample,
                mpp_level0=float(reader.mpp_level0),
            )
        table["x_source_um"], table["y_source_um"] = source.T
    else:
        table["section_id"] = pd.Series(dtype=str)
        table["z_um"] = pd.Series(dtype=float)
        table["x_source_um"] = pd.Series(dtype=float)
        table["y_source_um"] = pd.Series(dtype=float)

    table.to_csv(section_dir / "nuclei.csv", index=False)
    tifffile.imwrite(
        section_dir / "nuclei.tif",
        labels.astype(np.uint32),
        compression="zlib",
    )
    if score is not None:
        np.save(section_dir / "cellprob_score.npy", np.asarray(score, dtype=np.float32))
    clean = rgb.copy()
    clean[~tissue] = 255
    _save_qc(clean, labels, section_dir / "qc_original_and_boundaries.png")
    Image.fromarray((tissue * 255).astype(np.uint8), mode="L").save(
        section_dir / "tissue_mask_highres.png"
    )
    if save_input_crop:
        tifffile.imwrite(
            section_dir / f"input_rgb_level{level}.tif",
            rgb,
            photometric="rgb",
            compression="zlib",
            bigtiff=bool(rgb.nbytes > 2**32),
        )
    (section_dir / "raw_crop_metadata.json").write_text(
        json.dumps(
            {
                "source": "raw_sdpc",
                "raw_source_path": str(Path(row.raw_source_path).resolve()),
                "target_mpp_um": float(target_mpp),
                "selected_level": int(level),
                "selected_level_mpp_um": float(level_mpp),
                "raw_mpp_um": float(reader.mpp_level0),
                "level_downsample": float(downsample),
                "level0_dimensions": list(level0_dimensions),
                "requested_bbox_level0": list(map(float, bbox)),
                "read_location_level0": list(map(int, location)),
                "read_size_level_px": list(map(int, size)),
                "effective_bbox_level0": list(map(int, effective_bbox)),
                "analysis_mask_path": str(Path(row.source_mask_path).resolve()),
                "analysis_mask_shape": [int(x) for x in lowres_tissue.shape],
                "highres_image_shape": [int(x) for x in rgb.shape[:2]],
                "mask_resize": "nearest_neighbor_reviewed_analysis_mask",
                "source_coordinate_mode": source_coordinate_mode,
            },
            indent=2,
        )
        + "\n"
    )
    return table, {
        "section_id": section_id,
        "n_nuclei": int(len(table)),
        "image_shape": [int(x) for x in rgb.shape[:2]],
        "mpp_um": [float(level_mpp), float(level_mpp)],
        "status": "segmented",
        "segmentation_source": "raw_sdpc",
        "raw_source_path": str(Path(row.raw_source_path).resolve()),
        "raw_level": int(level),
        "raw_mpp_um": float(reader.mpp_level0),
        "selected_level_mpp_um": float(level_mpp),
        "raw_read_location_level0": list(map(int, location)),
        "raw_read_size_level_px": list(map(int, size)),
        "raw_effective_bbox_level0": list(map(int, effective_bbox)),
    }


def _empty_table():
    return pd.DataFrame(
        columns=[
            "cell_id",
            "nucleus_id",
            "x_px",
            "y_px",
            "area_um2",
            "perimeter_um",
            "eccentricity",
            "cellpose_score_raw",
            "touches_image_edge",
            "section_id",
            "z_um",
            "x_source_um",
            "y_source_um",
            "x_registered_um",
            "y_registered_um",
        ]
    )


def _selected_he_table(args):
    he_table = _read_table(
        args.he_table, parquet_site_packages=args.parquet_site_packages
    ).sort_values("z_um").reset_index(drop=True)
    if args.sections:
        wanted = {str(int(x)) for x in args.sections}
        he_table = he_table[he_table.section_id.astype(str).isin(wanted)].copy()
    if he_table.empty:
        raise ValueError("No requested H&E sections")
    return he_table


def _write_segmentation_manifest(output, args, summaries):
    frame = pd.DataFrame(summaries).sort_values("section_id")
    frame.to_csv(output / "segmentation_summary.csv", index=False)
    (output / "segmentation_manifest.json").write_text(
        json.dumps(
            {
                "format": "deepspatial-nucleus-segmentation-v1",
                "he_table": str(Path(args.he_table).resolve()),
                "registration_root": str(Path(args.registration_root).resolve()),
                "cellpose_model": str(Path(args.cellpose_model).resolve()),
                "diameter_um": float(args.diameter_um),
                "image_source": str(args.image_source),
                "raw_target_mpp_um": float(args.raw_target_mpp),
                "save_input_crops": bool(args.save_input_crops),
                "sections": summaries,
            },
            indent=2,
        )
        + "\n"
    )


def segment(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    import torch
    from cellpose import models

    he_table = _selected_he_table(args)
    torch_device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model = models.CellposeModel(
        gpu=torch_device == "cuda", pretrained_model=str(args.cellpose_model)
    )
    logging.info("Cellpose model=%s device=%s", args.cellpose_model, torch_device)

    registration_store, load_section_transform, transform_section_points = (
        _load_external_registration(args.v2_src, args.registration_root)
    )
    segmentation_dir = output / "segmentation"
    summaries = []

    def save_registered(row, table):
        if len(table):
            section = load_section_transform(row.section_transform_path)
            registered = transform_section_points(
                table[["x_source_um", "y_source_um"]].to_numpy(float),
                section,
                registration_store,
            )
            table["x_registered_um"], table["y_registered_um"] = registered.T
        else:
            table["x_registered_um"] = pd.Series(dtype=float)
            table["y_registered_um"] = pd.Series(dtype=float)
        table.to_csv(
            segmentation_dir / f"section-{int(row.section_id):03d}" / "nuclei_registered.csv",
            index=False,
        )

    def handle_one(row, segment_fn):
        sid = str(row.section_id)
        try:
            table, summary = segment_fn(row)
            save_registered(row, table)
            logging.info("section %s: segmented %d nuclei", sid, len(table))
        except Exception as error:
            logging.exception("section %s failed; storing unavailable nuclear support", sid)
            table = _empty_table()
            summary = {
                "section_id": sid,
                "n_nuclei": 0,
                "image_shape": None,
                "mpp_um": [
                    float(row.analysis_pixel_size_x_um),
                    float(row.analysis_pixel_size_y_um),
                ],
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "segmentation_source": str(args.image_source),
            }
            section_dir = segmentation_dir / f"section-{int(row.section_id):03d}"
            section_dir.mkdir(parents=True, exist_ok=True)
            table.to_csv(section_dir / "nuclei_registered.csv", index=False)
        summaries.append(summary)

    if args.image_source == "analysis":
        for _, row in he_table.iterrows():
            handle_one(
                row,
                lambda current: _segment_one(
                    current, model, segmentation_dir, args.diameter_um, args.series_id
                ),
            )
    else:
        required = {
            "raw_source_path",
            "raw_level0_x0",
            "raw_level0_y0",
            "raw_level0_x1",
            "raw_level0_y1",
        }
        missing = sorted(required - set(he_table.columns))
        if missing:
            raise KeyError(
                "raw_sdpc image source requires metadata columns: " + ", ".join(missing)
            )
        sdpc_module = _load_local_sdpc_module()
        for raw_path, group in he_table.groupby("raw_source_path", sort=True):
            raw_path = Path(raw_path).resolve()
            if not raw_path.is_file():
                raise FileNotFoundError(f"Missing raw SDPC source: {raw_path}")
            logging.info("opening raw SDPC %s for %d sections", raw_path, len(group))
            with sdpc_module.SdpcPyramid(raw_path) as reader:
                for _, row in group.sort_values("section_id").iterrows():
                    handle_one(
                        row,
                        lambda current, opened=reader: _segment_one_raw(
                            current,
                            model,
                            segmentation_dir,
                            args.diameter_um,
                            opened,
                            sdpc_module,
                            target_mpp=args.raw_target_mpp,
                            save_input_crop=args.save_input_crops,
                            series_id=args.series_id,
                        ),
                    )

    _write_segmentation_manifest(output, args, summaries)
    logging.info("wrote segmentation outputs for %d sections", len(summaries))


def materialize(args):
    """Build the DeepSpatial feature grid from already segmented nuclei."""

    from deepspatial.histology import FeatureStore
    from deepspatial.histology.nucleus import descriptor_grid_from_nuclei

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    store_path = output / "nucleus_features.h5"
    if store_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite {store_path}; choose a new output directory"
        )

    he_table = _selected_he_table(args)
    uni_store = FeatureStore(args.uni2_store)
    segmentation_source = (
        Path(args.segmentation_source).resolve()
        if args.segmentation_source is not None
        else output
    )
    segmentation_dir = segmentation_source / "segmentation"
    summary_path = segmentation_source / "segmentation_summary.csv"
    if summary_path.exists():
        summary_frame = pd.read_csv(summary_path)
        summary_by_id = {
            str(row.section_id): row.to_dict()
            for _, row in summary_frame.iterrows()
        }
    else:
        summary_by_id = {}

    feature_store = FeatureStore(store_path, mode="a")
    summaries = []
    for _, row in he_table.iterrows():
        sid = str(row.section_id)
        if sid not in uni_store.metadata:
            raise ValueError(f"Missing UNI2 feature section {sid}")
        meta = uni_store.metadata[sid]
        grid = _grid_from_metadata(meta)
        _, he_valid = uni_store.get_feature(
            sid, grid.reshape(-1, 2), return_valid=True
        )
        he_valid = he_valid.numpy().reshape(grid.shape[:2])
        section_dir = segmentation_dir / f"section-{int(row.section_id):03d}"
        registered_path = section_dir / "nuclei_registered.csv"
        status = str(summary_by_id.get(sid, {}).get("status", "missing"))
        if registered_path.exists():
            table = pd.read_csv(registered_path)
        else:
            table = _empty_table()
        if status == "missing":
            status = "segmented" if registered_path.exists() else "failed"
        if status != "segmented":
            he_valid = np.zeros_like(he_valid, dtype=bool)
            xy = np.empty((0, 2), dtype=float)
            areas = np.empty(0, dtype=float)
            eccentricity = np.empty(0, dtype=float)
        elif len(table):
            xy = table[["x_registered_um", "y_registered_um"]].to_numpy(float)
            areas = table["area_um2"].to_numpy(float)
            eccentricity = table["eccentricity"].to_numpy(float)
        else:
            xy = np.empty((0, 2), dtype=float)
            areas = np.empty(0, dtype=float)
            eccentricity = np.empty(0, dtype=float)

        descriptors, nuclear_valid = descriptor_grid_from_nuclei(
            xy,
            areas,
            eccentricity,
            grid,
            radius_um=args.radius_um,
            tissue_mask=he_valid,
        )
        if status != "segmented":
            descriptors[...] = 0
            nuclear_valid[...] = False
        he_valid &= nuclear_valid
        feature_store.add_section(
            sid,
            descriptors,
            z_um=float(meta["z_um"]),
            origin_um=meta["origin_um"],
            spacing_um=meta["spacing_um"],
            patch_size_um=float(meta["patch_size_um"]),
            mpp=meta["mpp"],
            coordinate_frame=meta["coordinate_frame"],
            valid_mask=he_valid,
            provenance={
                "source": "cellpose_nuclear_structure",
                "cellpose_model": str(args.cellpose_model),
                "diameter_um": float(args.diameter_um),
                "descriptor_radius_um": float(args.radius_um),
                "he_table": str(Path(args.he_table).resolve()),
                "uni2_store": str(Path(args.uni2_store).resolve()),
                "segmentation_source": str(segmentation_source),
                "section_transform": str(row.section_transform_path),
                "semantics": "local nuclear anatomy; not cell lineage",
                "segmentation_status": status,
            },
        )
        summary = dict(summary_by_id.get(sid, {}))
        summary.update(
            section_id=sid,
            n_nuclei=int(len(table)) if status == "segmented" else 0,
            feature_valid_fraction=float(he_valid.mean()),
            status=status,
        )
        summaries.append(summary)

    summary_frame = pd.DataFrame(summaries).sort_values("section_id")
    summary_frame.to_csv(output / "segmentation_summary.csv", index=False)
    coordinate_frames = sorted(
        {str(section_meta["coordinate_frame"]) for section_meta in uni_store.metadata.values()}
    )
    if len(coordinate_frames) != 1:
        raise ValueError(
            "Nucleus feature materialization requires one coordinate frame; "
            f"found {coordinate_frames}"
        )
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "format": "deepspatial-nucleus-path-v1",
                "series_id": str(args.series_id),
                "coordinate_frame": coordinate_frames[0],
                "he_table": str(Path(args.he_table).resolve()),
                "uni2_store": str(Path(args.uni2_store).resolve()),
                "nucleus_feature_store": str(store_path),
                "segmentation_source": str(segmentation_source),
                "registration_root": str(Path(args.registration_root).resolve()),
                "cellpose_model": str(Path(args.cellpose_model).resolve()),
                "diameter_um": float(args.diameter_um),
                "descriptor_radius_um": float(args.radius_um),
                "sections": summaries,
            },
            indent=2,
        )
        + "\n"
    )
    logging.info("wrote %s and %d sections", store_path, len(summaries))


def build(args):
    _enable_parquet_site_packages(args.parquet_site_packages)
    if args.stage in {"segment", "all"}:
        segment(args)
    if args.stage in {"materialize", "all"}:
        materialize(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--he-table", type=Path, default=DEFAULT_HE_TABLE)
    parser.add_argument("--uni2-store", type=Path, default=DEFAULT_UNI2_STORE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--series-id", default="00029__g0")
    parser.add_argument("--registration-root", type=Path, default=DEFAULT_REGISTRATION)
    parser.add_argument("--v2-src", type=Path, default=DEFAULT_V2_SRC)
    parser.add_argument("--cellpose-model", type=Path, default=DEFAULT_CELLPOSE_MODEL)
    parser.add_argument("--sections", nargs="*", type=int)
    parser.add_argument(
        "--image-source",
        choices=["analysis", "raw_sdpc"],
        default="analysis",
        help="Segment the reviewed analysis crop or a crop read from raw SDPC",
    )
    parser.add_argument(
        "--raw-target-mpp",
        type=float,
        default=1.1213235294117647,
        help="Target physical pixel size for raw SDPC segmentation",
    )
    parser.add_argument(
        "--save-input-crops",
        action="store_true",
        help="Save each raw SDPC RGB crop alongside its segmentation",
    )
    parser.add_argument(
        "--stage",
        choices=["segment", "materialize", "all"],
        default="all",
        help="Run Cellpose segmentation, feature materialization, or both",
    )
    parser.add_argument(
        "--parquet-site-packages",
        type=Path,
        help="Optional site-packages containing pyarrow; added after active NumPy",
    )
    parser.add_argument(
        "--segmentation-source",
        type=Path,
        help="Existing segment-stage output used by materialize (defaults to output)",
    )
    parser.add_argument("--diameter-um", type=float, default=8.0)
    parser.add_argument("--radius-um", type=float, default=56.0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    Path(args.output).resolve().mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(Path(args.output) / "run.log")],
    )
    build(args)


if __name__ == "__main__":
    main()
