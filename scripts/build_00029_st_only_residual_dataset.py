#!/usr/bin/env python3
"""Build a non-destructive 00029/g0 dataset with ST-only residual correction.

The existing serial H&E registration defines the canonical frame.  This
builder materializes each anchor's residual affine into a new
``obsm['spatial_st_corrected']`` array while leaving H&E images, masks, feature
grids, and nuclear centroids in the original registered frame.

No residual transform is applied to H&E-only sections, and no Z interpolation
of residual affines is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import anndata as ad
import numpy as np
import pandas as pd


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ANCHORS = DEFAULT_ROOT / "data/00029_g0/anchors"
DEFAULT_HE_TABLE = DEFAULT_ROOT / "data/00029_g0/he_sections.parquet"
DEFAULT_RESIDUAL_DIR = DEFAULT_ROOT / "data/00029_g0/registration_correction_v1"
DEFAULT_NUCLEI = DEFAULT_ROOT / "data/00029_g0/nucleus_path_highres_rerun_v3/segmentation"
DEFAULT_OUTPUT = DEFAULT_ROOT / "data/00029_g0/st_only_residual_v1"

MATRIX_NAME = "matrix_current_g_to_fixed_he"
ANCHOR_NAME = "spatial_registered"
CORRECTED_NAME = "spatial_st_corrected"
HE_POLICY = "original_registered_only"
CANONICAL_HE_POLICY = "identity_residual_for_all_he_sections"
ST_POLICY = "materialized_residual_affine_per_anchor"


def _as_affine(matrix: object) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (3, 3) or not np.isfinite(value).all():
        raise ValueError("Residual affine must be finite with shape (3, 3)")
    if not np.allclose(value[2], [0.0, 0.0, 1.0], atol=1e-8, rtol=0.0):
        raise ValueError("Residual affine must have homogeneous last row [0, 0, 1]")
    if abs(float(np.linalg.det(value[:2, :2]))) <= 1e-12:
        raise ValueError("Residual affine must be invertible")
    return value


def _sha256_array(value: np.ndarray) -> str:
    array = np.asarray(value, dtype="<f8", order="C")
    return hashlib.sha256(array.tobytes()).hexdigest()


def _sha256_points(points: np.ndarray) -> str:
    return _sha256_array(np.asarray(points, dtype=np.float64))


def load_residual_matrix(path: Path | str) -> np.ndarray:
    """Load and validate the saved ST-to-fixed-H&E affine."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as value:
        if MATRIX_NAME not in value:
            raise KeyError(f"{path} does not contain {MATRIX_NAME!r}")
        return _as_affine(value[MATRIX_NAME])


def apply_st_residual(points: object, matrix: object) -> np.ndarray:
    """Apply a physical XY affine to an ``[N, 2]`` ST coordinate array."""

    value = np.asarray(points, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 2 or not np.isfinite(value).all():
        raise ValueError("ST points must be finite with shape [N, 2]")
    affine = _as_affine(matrix)
    homogeneous = np.concatenate(
        [value, np.ones((len(value), 1), dtype=np.float64)], axis=1
    )
    transformed = homogeneous @ affine.T
    if not np.allclose(transformed[:, 2], 1.0, atol=1e-8, rtol=0.0):
        raise ValueError("Residual affine produced non-unit homogeneous scale")
    return transformed[:, :2]


def materialize_anchor_h5ad(
    source_path: Path | str,
    destination_path: Path | str,
    matrix_path: Path | str,
    coordinate_frame: str,
) -> dict:
    """Write one anchor H5AD with an embedded, ST-only corrected coordinate."""

    source_path = Path(source_path)
    destination_path = Path(destination_path)
    matrix_path = Path(matrix_path)
    if destination_path.exists():
        raise FileExistsError(f"Refusing to overwrite {destination_path}")
    if not coordinate_frame:
        raise ValueError("coordinate_frame must be non-empty")

    value = ad.read_h5ad(source_path)
    if ANCHOR_NAME not in value.obsm:
        raise KeyError(f"{source_path} lacks obsm[{ANCHOR_NAME!r}]")
    original = np.asarray(value.obsm[ANCHOR_NAME], dtype=np.float64)
    if original.shape != (value.n_obs, 2) or not np.isfinite(original).all():
        raise ValueError(
            f"{source_path} has invalid {ANCHOR_NAME!r}; expected finite [n_obs, 2]"
        )

    matrix = load_residual_matrix(matrix_path)
    corrected = apply_st_residual(original, matrix)
    value.obsm[CORRECTED_NAME] = corrected
    matrix_hash = _sha256_array(matrix)
    contract = {
        "coordinate_units": "micrometer",
        "coordinate_frame": str(coordinate_frame),
        "h_and_e_coordinate_frame": str(coordinate_frame),
        "original_coordinate_key": ANCHOR_NAME,
        "corrected_coordinate_key": CORRECTED_NAME,
        "st_residual_applied": True,
        "h_and_e_residual_applied": False,
        "h_and_e_policy": HE_POLICY,
        "transform_direction": "spatial_st_corrected = matrix_current_g_to_fixed_he @ spatial_registered",
        "transform_path": str(matrix_path.resolve()),
        "matrix_sha256": matrix_hash,
        "original_coordinate_sha256": _sha256_points(original),
        "corrected_coordinate_sha256": _sha256_points(corrected),
        "note": "Residual affine is materialized only into ST anchor coordinates; H&E remains fixed.",
    }
    value.uns["deepspatial_coordinate_contract"] = contract
    value.uns["deepspatial_residual_correction"] = contract.copy()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    value.write_h5ad(destination_path, compression="gzip")
    return {
        "section_id": str(value.obs["section_id"].iloc[0])
        if "section_id" in value.obs
        else destination_path.stem,
        "source": str(source_path.resolve()),
        "output": str(destination_path.resolve()),
        "matrix_path": str(matrix_path.resolve()),
        "matrix_sha256": matrix_hash,
        "n_obs": int(value.n_obs),
        "n_vars": int(value.n_vars),
        "original_coordinate_sha256": contract["original_coordinate_sha256"],
        "corrected_coordinate_sha256": contract["corrected_coordinate_sha256"],
        "h_and_e_residual_applied": False,
    }


def build_he_table_contract(table: pd.DataFrame) -> pd.DataFrame:
    """Annotate an H&E manifest without changing any H&E coordinates or paths."""

    result = table.copy()
    result["residual_applied_to_he"] = False
    result["he_coordinate_policy"] = HE_POLICY
    result["he_residual_transform_path"] = None
    result["he_residual_transform_type"] = "not_applied"
    return result


def _section_id_from_path(path: Path) -> int:
    match = re.search(r"section-(\d+)", path.name)
    if match is None:
        raise ValueError(f"Cannot parse section ID from {path}")
    return int(match.group(1))


def aggregate_registered_nuclei(
    segmentation_dir: Path | str,
    output_path: Path | str,
    coordinate_frame: str,
    section_ids: set[int] | None = None,
) -> dict:
    """Aggregate native high-resolution segmentation in the original H&E frame."""

    segmentation_dir = Path(segmentation_dir)
    output_path = Path(output_path)
    if not segmentation_dir.is_dir():
        raise NotADirectoryError(segmentation_dir)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")

    frames = []
    for path in sorted(segmentation_dir.glob("section-*/nuclei_registered.csv")):
        section_id = _section_id_from_path(path.parent)
        if section_ids is not None and section_id not in section_ids:
            continue
        frame = pd.read_csv(path)
        required = {"x_registered_um", "y_registered_um"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise KeyError(f"{path} missing registered coordinates: {missing}")
        # Do not carry fields from an older fixed-frame materialization into the
        # new canonical table.
        frame = frame.drop(
            columns=["x_fixed_he_um", "y_fixed_he_um", "fixed_transform_type"],
            errors="ignore",
        )
        if "section_id" not in frame.columns:
            frame.insert(0, "section_id", section_id)
        frame["section_id"] = section_id
        frame["coordinate_frame"] = str(coordinate_frame)
        frame["residual_applied_to_he"] = False
        frame["he_coordinate_policy"] = HE_POLICY
        frames.append(frame)

    if frames:
        result = pd.concat(frames, ignore_index=True)
    else:
        result = pd.DataFrame(
            columns=[
                "section_id",
                "x_registered_um",
                "y_registered_um",
                "coordinate_frame",
                "residual_applied_to_he",
                "he_coordinate_policy",
            ]
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    return {
        "path": str(output_path.resolve()),
        "n_nuclei": int(len(result)),
        "n_sections": int(result["section_id"].nunique()) if len(result) else 0,
        "coordinate_frame": str(coordinate_frame),
        "residual_applied_to_he": False,
        "coordinate_columns": ["x_registered_um", "y_registered_um"],
    }


def _validate_he_frame(table: pd.DataFrame, coordinate_frame: str) -> pd.DataFrame:
    if "coordinate_frame" not in table.columns:
        table = table.copy()
        table["coordinate_frame"] = str(coordinate_frame)
    frames = set(table["coordinate_frame"].astype(str))
    if frames != {str(coordinate_frame)}:
        raise ValueError(
            "Input H&E table has unexpected coordinate frames: "
            f"{sorted(frames)}; expected {coordinate_frame!r}"
        )
    if "section_id" not in table.columns or "source_kind" not in table.columns:
        raise KeyError("H&E table needs section_id and source_kind")
    return table


def _write_readme(output_dir: Path, manifest: dict) -> None:
    output_dir.joinpath("README.md").write_text(
        "# 00029/g0 ST-only residual dataset\n\n"
        "This is the canonical data package for the residual-corrected ST run.\n\n"
        f"- Coordinate frame: `{manifest['coordinate_frame']}`\n"
        f"- ST coordinate used by DeepSpatial: `{manifest['st_coordinate_key']}`\n"
        "- H&E policy: original registered H&E remains fixed for every section.\n"
        "- Residual affine: materialized once into anchor ST coordinates only.\n"
        "- Serial H&E-only sections: no residual affine and no Z interpolation.\n"
        "- Original ST coordinates remain in `spatial_registered` for baseline/ablation.\n\n"
        "Use `spatial_key=\"spatial_st_corrected\"` for the corrected DeepSpatial run.\n"
        "Do not multiply the residual matrices again.\n"
    )


def build_anchor_manifest(
    *,
    source_dir: Path | str,
    residual_dir: Path | str,
    output_dir: Path | str,
    sections: list[int] | None,
    coordinate_frame: str,
    he_table_path: Path | str | None,
    nuclei_segmentation: Path | str | None = None,
) -> dict:
    """Build the complete non-destructive ST-only package."""

    source_dir = Path(source_dir)
    residual_dir = Path(residual_dir)
    output_dir = Path(output_dir)
    if not source_dir.is_dir():
        raise NotADirectoryError(source_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to write into non-empty output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    anchors_output = output_dir / "anchors"
    anchors_output.mkdir()

    if sections is None:
        source_paths = sorted(source_dir.glob("section-*.h5ad"))
        sections = [_section_id_from_path(path) for path in source_paths]
    sections = sorted({int(value) for value in sections})
    if not sections:
        raise ValueError("No anchor sections supplied")

    anchor_records = []
    for section_id in sections:
        source_path = source_dir / f"section-{section_id:03d}.h5ad"
        matrix_path = residual_dir / "transforms" / (
            f"section-{section_id:03d}__st_to_fixed_he_residual.npz"
        )
        # Check the matrix before reading/writing other package components so a
        # missing anchor transform cannot silently produce a partial dataset.
        load_residual_matrix(matrix_path)
        destination_path = anchors_output / f"section-{section_id:03d}.h5ad"
        anchor_records.append(
            materialize_anchor_h5ad(
                source_path,
                destination_path,
                matrix_path,
                coordinate_frame,
            )
        )

    he_record = None
    he_section_ids: set[int] | None = None
    if he_table_path is not None:
        he_table_path = Path(he_table_path)
        if not he_table_path.is_file():
            raise FileNotFoundError(he_table_path)
        he_table = _validate_he_frame(
            pd.read_parquet(he_table_path), coordinate_frame
        )
        he_table = build_he_table_contract(he_table)
        # The package must contain the complete registered H&E series.  Anchor
        # sections determine which H5ADs are materialized, but they must not be
        # used to filter out the H&E-only sections between anchors.
        he_section_ids = set(he_table["section_id"].astype(int).tolist())
        he_output = output_dir / "he_sections.parquet"
        he_table.to_parquet(he_output, index=False)
        he_record = {
            "source": str(he_table_path.resolve()),
            "output": str(he_output.resolve()),
            "n_sections": int(he_table["section_id"].nunique()),
            "residual_applied_to_he": bool(he_table["residual_applied_to_he"].any()),
            "coordinate_frame": str(coordinate_frame),
        }

    nuclei_record = None
    if nuclei_segmentation is not None:
        nuclei_record = aggregate_registered_nuclei(
            nuclei_segmentation,
            output_dir / "nuclei_registered.parquet",
            coordinate_frame,
            he_section_ids,
        )

    manifest = {
        "format": "deepspatial-00029-st-only-residual-v1",
        "status": "complete",
        "dataset": "00029/g0",
        "coordinate_frame": str(coordinate_frame),
        "coordinate_units": "micrometer",
        "st_coordinate_key": CORRECTED_NAME,
        "original_st_coordinate_key": ANCHOR_NAME,
        "h_and_e_policy": CANONICAL_HE_POLICY,
        "st_policy": ST_POLICY,
        "serial_he_transform_policy": "identity_residual",
        "residual_is_applied_to_he": False,
        "z_interpolated_residual_affine": False,
        "source_anchors": str(source_dir.resolve()),
        "residual_transform_dir": str(residual_dir.resolve()),
        "anchor_sections": sections,
        "anchors": anchor_records,
        "he": he_record,
        "nuclei": nuclei_record,
        "deep_spatial_usage": {
            "spatial_key": CORRECTED_NAME,
            "coordinate_frame": str(coordinate_frame),
            "feature_store_policy": "use unchanged registered-frame H&E UNI2 store",
            "nucleus_store_policy": "use unchanged registered-frame H&E nuclear store",
            "apply_external_residual_at_runtime": False,
            "rebuild_uot_and_path_cache": True,
        },
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    _write_readme(output_dir, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-anchors", type=Path, default=DEFAULT_SOURCE_ANCHORS)
    parser.add_argument("--he-table", type=Path, default=DEFAULT_HE_TABLE)
    parser.add_argument("--residual-dir", type=Path, default=DEFAULT_RESIDUAL_DIR)
    parser.add_argument("--nuclei-segmentation", type=Path, default=DEFAULT_NUCLEI)
    parser.add_argument(
        "--coordinate-frame", default="00029__g0_registered"
    )
    parser.add_argument("--sections", nargs="*", type=int, default=None)
    args = parser.parse_args()
    manifest = build_anchor_manifest(
        source_dir=args.source_anchors,
        residual_dir=args.residual_dir,
        output_dir=args.output_dir,
        sections=args.sections,
        coordinate_frame=args.coordinate_frame,
        he_table_path=args.he_table,
        nuclei_segmentation=args.nuclei_segmentation,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
