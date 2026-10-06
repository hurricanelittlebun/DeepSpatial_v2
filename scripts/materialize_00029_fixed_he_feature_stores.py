"""Materialize 00029/g0 morphology stores in the fixed H&E coordinate frame.

The existing ST anchors use ``spatial_st_corrected`` after a residual affine
correction.  The original UNI2 and nuclear stores were built in the prior
``00029__g0_registered`` frame.  This script applies the same current-to-fixed
affine to the ten Xenium companion sections and an explicit identity transform
to the seventy-two serial H&E sections.  It never overwrites an existing
feature store.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepspatial.histology import (
    FeatureStore,
    load_section_affines,
    resample_feature_store,
)


ALLOWED_SOURCE_KINDS = {"serial_he", "xenium_companion_he"}
COMPANION_KIND = "xenium_companion_he"
SERIAL_KIND = "serial_he"
SOURCE_FRAME = "00029__g0_registered"
OUTPUT_FRAME = "00029__g0_fixed_he"


def _normalise_section_id(value) -> str:
    try:
        return str(int(value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid section_id: {value!r}") from error


def _transform_path(residual_dir: Path, section_id: str) -> Path:
    return residual_dir / "transforms" / (
        f"section-{int(section_id):03d}__st_to_fixed_he_residual.npz"
    )


def _matrix_sha256(matrix: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(matrix, dtype="<f8").tobytes(order="C")
    ).hexdigest()


def build_section_affines(
    he_table: pd.DataFrame,
    residual_dir,
    *,
    interpolate_serial_affine: bool = False,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Build a verified section-to-fixed transform map from ``he_sections``.

    By default, ``serial_he`` receives an explicit identity matrix.  When
    ``interpolate_serial_affine`` is enabled, its matrix is linearly
    interpolated between the two surrounding Xenium residual matrices in Z;
    this keeps the fixed frame continuous between corrected anchors.  A
    Xenium companion section must have a saved residual affine on disk.
    """

    required = {"section_id", "z_um", "source_kind"}
    missing = sorted(required - set(he_table.columns))
    if missing:
        raise KeyError(f"H&E table lacks required columns: {missing}")

    table = he_table.copy()
    table["_section_id"] = table["section_id"].map(_normalise_section_id)
    if table["_section_id"].duplicated().any():
        duplicates = sorted(
            table.loc[table["_section_id"].duplicated(keep=False), "_section_id"].unique()
        )
        raise ValueError(f"Duplicate section_id in H&E table: {duplicates}")
    table["_source_kind"] = table["source_kind"].astype(str)
    unknown = sorted(set(table["_source_kind"]) - ALLOWED_SOURCE_KINDS)
    if unknown:
        raise ValueError(
            "H&E table source_kind contains unsupported values: "
            f"{unknown}; expected {sorted(ALLOWED_SOURCE_KINDS)}"
        )
    table = table.sort_values(["z_um", "_section_id"], kind="stable")
    section_ids = table["_section_id"].tolist()
    identity_sections = table.loc[
        table["_source_kind"] == SERIAL_KIND, "_section_id"
    ].tolist()
    residual_dir = Path(residual_dir).resolve()
    matrices = load_section_affines(
        residual_dir,
        section_ids,
        identity_sections=identity_sections,
    )

    interpolation = {}
    if interpolate_serial_affine:
        anchor_table = table.loc[table["_source_kind"] == COMPANION_KIND]
        if len(anchor_table) < 2:
            raise ValueError(
                "At least two Xenium companion sections are required to "
                "interpolate serial affine transforms"
            )
        anchor_ids = anchor_table["_section_id"].tolist()
        anchor_z = anchor_table["z_um"].to_numpy(dtype=float)
        if not np.all(np.diff(anchor_z) > 0):
            raise ValueError("Xenium companion anchor Z values must be strictly increasing")
        for _, row in table.loc[table["_source_kind"] == SERIAL_KIND].iterrows():
            section_id = row["_section_id"]
            z_um = float(row["z_um"])
            right = int(np.searchsorted(anchor_z, z_um, side="right"))
            left = right - 1
            if left < 0 or right >= len(anchor_ids):
                raise ValueError(
                    f"Serial section {section_id} lies outside companion anchor Z range"
                )
            fraction = float((z_um - anchor_z[left]) / (anchor_z[right] - anchor_z[left]))
            matrices[section_id] = (
                (1.0 - fraction) * matrices[anchor_ids[left]]
                + fraction * matrices[anchor_ids[right]]
            )
            interpolation[section_id] = {
                "bracket_sections": [anchor_ids[left], anchor_ids[right]],
                "interpolation_fraction": fraction,
            }

    records = []
    for _, row in table.iterrows():
        section_id = row["_section_id"]
        source_kind = row["_source_kind"]
        matrix = matrices[section_id]
        path = _transform_path(residual_dir, section_id)
        is_residual = source_kind == COMPANION_KIND
        if is_residual and not path.is_file():
            # ``load_section_affines`` normally raises first; keep this guard
            # close to the record construction so the invariant is explicit.
            raise FileNotFoundError(f"Missing residual transform for section {section_id}: {path}")
        record = {
            "section_id": section_id,
            "z_um": float(row["z_um"]),
            "source_kind": source_kind,
            "transform_type": (
                "residual_affine"
                if is_residual
                else "interpolated_affine"
                if section_id in interpolation
                else "identity"
            ),
            "path": str(path) if is_residual else None,
            "matrix_source_to_output": matrix.tolist(),
            "matrix_sha256": _matrix_sha256(matrix),
        }
        record.update(interpolation.get(section_id, {}))
        records.append(record)
    return matrices, records


def _resolve(path: Path | str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _validate_source_store(path: Path, section_ids: list[str], table_frame: str):
    store = FeatureStore(path)
    if store.sections != section_ids:
        raise ValueError(
            f"{path} sections do not match he_sections.parquet order: "
            f"store={store.sections[:5]}.../{len(store.sections)}, "
            f"table={section_ids[:5]}.../{len(section_ids)}"
        )
    frames = {store.metadata[sid]["coordinate_frame"] for sid in store.sections}
    if frames != {SOURCE_FRAME}:
        raise ValueError(f"{path} has unexpected coordinate frames: {sorted(frames)}")
    if table_frame != SOURCE_FRAME:
        raise ValueError(
            f"he_sections.parquet must describe {SOURCE_FRAME}, got {table_frame!r}"
        )
    return store


def _write_readme(output_dir: Path, manifest: dict) -> None:
    text = f"""# 00029/g0 fixed-frame morphology stores

These stores are a new, offline-resampled version of the registered-frame
stores.  They are aligned to the same fixed H&E frame as the corrected ST
anchor coordinates (`spatial_st_corrected`).

- coordinate frame: `{manifest['output_coordinate_frame']}`
- source frame: `{manifest['source_coordinate_frame']}`
- transform rule: `{manifest['transform_rule']}`
- serial H&E sections: {manifest['serial_section_count']}
- Xenium companion sections: {manifest['companion_section_count']}

The original stores are retained.  The `section_transforms` entry in
`materialization_manifest.json` records the matrix and provenance for every
section.  This is an anatomical coordinate correction, not cell tracking.
"""
    (output_dir / "README.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data/00029_g0")
    parser.add_argument("--he-table", type=Path, default=None)
    parser.add_argument("--residual-dir", type=Path, default=None)
    parser.add_argument(
        "--uni2-source",
        type=Path,
        default=None,
        help="Existing UNI2 feature store in the registered H&E frame.",
    )
    parser.add_argument(
        "--nucleus-source",
        type=Path,
        default=None,
        help="Existing nuclear descriptor store in the registered H&E frame.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-frame", default=OUTPUT_FRAME)
    parser.add_argument(
        "--interpolate-serial-affine",
        action="store_true",
        help="Interpolate residual anchor affines in Z for serial H&E sections.",
    )
    args = parser.parse_args()

    dataset_dir = _resolve(args.dataset_dir)
    he_table_path = _resolve(args.he_table or dataset_dir / "he_sections.parquet")
    residual_dir = _resolve(
        args.residual_dir or dataset_dir / "registration_correction_v1"
    )
    uni2_source = _resolve(
        args.uni2_source or dataset_dir / "uni2_features_raw_corrected.h5"
    )
    nucleus_source = _resolve(
        args.nucleus_source
        or dataset_dir / "nucleus_path_raw_final/nucleus_features.h5"
    )
    output_dir = _resolve(
        args.output_dir or dataset_dir / "fixed_he_feature_frame_v1"
    )

    if not args.output_frame:
        raise ValueError("--output-frame must be non-empty")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to write into non-empty output directory: {output_dir}"
        )
    if not he_table_path.is_file():
        raise FileNotFoundError(he_table_path)
    if not uni2_source.is_file():
        raise FileNotFoundError(uni2_source)
    if not nucleus_source.is_file():
        raise FileNotFoundError(nucleus_source)

    table = pd.read_parquet(he_table_path)
    table_frame_values = set(table["coordinate_frame"].astype(str))
    if table_frame_values != {SOURCE_FRAME}:
        raise ValueError(
            "he_sections.parquet has unexpected coordinate_frame values: "
            f"{sorted(table_frame_values)}"
        )
    table = table.sort_values(["z_um", "section_id"], kind="stable")
    section_ids = [_normalise_section_id(value) for value in table["section_id"]]
    matrices, transform_records = build_section_affines(
        table,
        residual_dir,
        interpolate_serial_affine=args.interpolate_serial_affine,
    )

    uni2 = _validate_source_store(uni2_source, section_ids, SOURCE_FRAME)
    nucleus = _validate_source_store(nucleus_source, section_ids, SOURCE_FRAME)
    if uni2.sections != nucleus.sections:
        raise ValueError("UNI2 and nucleus stores have different section order")

    output_dir.mkdir(parents=True, exist_ok=False)
    uni2_output = output_dir / "uni2_features_fixed.h5"
    nucleus_output = output_dir / "nucleus_features_fixed.h5"
    serial_policy = (
        "z_linear_interpolated_affine"
        if args.interpolate_serial_affine
        else "identity"
    )
    provenance = {
        "pipeline": "fixed_he_feature_frame_v2" if args.interpolate_serial_affine else "fixed_he_feature_frame_v1",
        "he_table": str(he_table_path),
        "residual_transform_dir": str(residual_dir),
        "source_coordinate_frame": SOURCE_FRAME,
        "output_coordinate_frame": str(args.output_frame),
        "transform_rule": (
            "p_fixed = M_current_to_fixed(z) @ p_registered; residual affine "
            "at xenium companion sections; "
            f"{serial_policy} for serial_he sections"
        ),
        "serial_transform_policy": serial_policy,
        "section_transforms": transform_records,
    }
    print("Materializing UNI2 feature store...", flush=True)
    uni2_result = resample_feature_store(
        uni2_source,
        uni2_output,
        matrices,
        output_frame=args.output_frame,
        provenance=provenance,
    )
    print("Materializing nuclear feature store...", flush=True)
    nucleus_result = resample_feature_store(
        nucleus_source,
        nucleus_output,
        matrices,
        output_frame=args.output_frame,
        provenance=provenance,
    )

    companion_count = sum(
        row["source_kind"] == COMPANION_KIND for row in transform_records
    )
    serial_count = sum(row["source_kind"] == SERIAL_KIND for row in transform_records)
    manifest = {
        "format": (
            "deepspatial-fixed-he-feature-frame-v2"
            if args.interpolate_serial_affine
            else "deepspatial-fixed-he-feature-frame-v1"
        ),
        "status": "complete",
        "dataset": _relative(dataset_dir),
        "he_table": _relative(he_table_path),
        "residual_transform_dir": _relative(residual_dir),
        "source_coordinate_frame": SOURCE_FRAME,
        "output_coordinate_frame": str(args.output_frame),
        "transform_rule": provenance["transform_rule"],
        "serial_transform_policy": serial_policy,
        "serial_section_count": serial_count,
        "companion_section_count": companion_count,
        "section_ids": section_ids,
        "section_transforms": transform_records,
        "sources": {
            "uni2": {
                "path": _relative(uni2_source),
                "revision": uni2.revision,
                "feature_dim": int(uni2.feature_dim),
            },
            "nucleus": {
                "path": _relative(nucleus_source),
                "revision": nucleus.revision,
                "feature_dim": int(nucleus.feature_dim),
            },
        },
        "outputs": {
            "uni2": {
                "path": _relative(uni2_output),
                "revision": uni2_result["output_revision"],
            },
            "nucleus": {
                "path": _relative(nucleus_output),
                "revision": nucleus_result["output_revision"],
            },
        },
        "uni2_materialization": uni2_result,
        "nucleus_materialization": nucleus_result,
        "note": "Morphology/anatomical coordinate correction; not cell tracking.",
    }
    (output_dir / "materialization_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    _write_readme(output_dir, manifest)
    print(output_dir / "materialization_manifest.json")
    print(json.dumps({"sections": len(section_ids), "output_frame": args.output_frame}, indent=2))


if __name__ == "__main__":
    main()
