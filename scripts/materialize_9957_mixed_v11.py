"""Materialize the reviewed mixed 9957/g0 alignment release.

The release is deliberately stitched at the shared section 53:

* sections <= 53 are taken from the v10 feature-support release;
* section 55 and all downstream sections are taken from the reviewed v9
  post-rotation release;
* section 54 remains removed.

The v10 residual affine starts at section 55.  Therefore the v10 and v9
coordinates are identical through section 53, which makes section 53 a valid
common stitch point.  Downstream sections must remain on the v9 branch to
preserve the reviewed v9 53 -> 55 alignment; mixing v9 section 55 with v10
section 57+ would create a new frame discontinuity.

This script copies already materialized arrays.  It does not re-run
STalign, re-encode UNI2, or change the original v9/v10 releases.
"""

from __future__ import annotations

import argparse
import json
import shutil
import uuid
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
GROUP = ROOT / "data" / "9957_g0"

V9_PREP = GROUP / "reconstruction_prep_v9_postrotate_ccw90_cutoff91"
V10_PREP = GROUP / "reconstruction_prep_v10_feature_support_residual"
V9_REG = GROUP / "registration_correction_he_v9_postrotate_ccw90_cutoff91"
V10_REG = GROUP / "registration_correction_he_v10_feature_support_residual"

OUTPUT_PREP = GROUP / "reconstruction_prep_v11_mixed_v10_pre53_v9_post55"
OUTPUT_REG = GROUP / "registration_correction_he_v11_mixed_v10_pre53_v9_post55"
OUTPUT_QC = GROUP / "qc" / "he_alignment_all_sections_v8_mixed_v10_pre53_v9_post55"

OUTPUT_FRAME = "9957__g0_registered_mixed_v10_pre53_v9_post55_v11"
RELEASE = "9957_mixed_v10_pre53_v9_post55_v11"
SPLIT_SECTION = 53
REMOVED_SECTION = 54


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _store_key(section_id: int) -> str:
    import hashlib

    return hashlib.sha256(str(section_id).encode()).hexdigest()


def _source_label(section_id: int) -> str:
    return "v10_pre53" if section_id <= SPLIT_SECTION else "v9_post55"


def _source_prep(section_id: int) -> Path:
    return V10_PREP if section_id <= SPLIT_SECTION else V9_PREP


def _source_store_path(prep: Path, kind: str) -> Path:
    if kind == "uni2":
        name = (
            "uni2_features_feature_support_residual_v10.h5"
            if prep == V10_PREP
            else "uni2_features_latest_alignment_cutoff91_v9.h5"
        )
        return prep / name
    if kind == "nucleus":
        name = (
            "nucleus_path_feature_support_residual_v10/nucleus_features.h5"
            if prep == V10_PREP
            else "nucleus_path_latest_alignment_cutoff91_v9/nucleus_features.h5"
        )
        return prep / name
    raise ValueError(kind)


def _source_h5ad() -> Path:
    candidates = sorted(V10_REG.glob("*.h5ad"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected one v10 H5AD in {V10_REG}, got {candidates}")
    return candidates[0]


def _validate_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[int, pd.Series], dict[int, pd.Series]]:
    v9 = pd.read_parquet(V9_PREP / "he_sections.parquet").copy()
    v10 = pd.read_parquet(V10_PREP / "he_sections.parquet").copy()
    for table, label in ((v9, "v9"), (v10, "v10")):
        table["section_id"] = pd.to_numeric(table["section_id"], errors="raise").astype(int)
        if table["section_id"].duplicated().any():
            raise ValueError(f"{label} H&E table has duplicate section IDs")
    ids9 = set(v9["section_id"].tolist())
    ids10 = set(v10["section_id"].tolist())
    if ids9 != ids10:
        raise ValueError(f"v9/v10 H&E section IDs differ: only-v9={sorted(ids9-ids10)}, only-v10={sorted(ids10-ids9)}")
    if REMOVED_SECTION in ids9 or SPLIT_SECTION not in ids9 or 55 not in ids9:
        raise ValueError(f"unexpected boundary section IDs: {sorted(ids9)}")

    rows9 = {int(row.section_id): row for _, row in v9.iterrows()}
    rows10 = {int(row.section_id): row for _, row in v10.iterrows()}
    matrix_diffs = {}
    for sid in sorted(ids9):
        p9 = Path(str(rows9[sid]["section_transform_path"]))
        p10 = Path(str(rows10[sid]["section_transform_path"]))
        if sid <= SPLIT_SECTION:
            m9 = _load_npz(p9)["matrix"]
            m10 = _load_npz(p10)["matrix"]
            diff = float(np.max(np.abs(np.asarray(m9, dtype=float) - np.asarray(m10, dtype=float))))
            matrix_diffs[sid] = diff
            if diff > 1e-7:
                raise AssertionError(f"v10 section {sid} is not the same pre-53 frame as v9; max matrix diff={diff}")
    return v9, v10, rows9, rows10


def _build_table(v9: pd.DataFrame, v10: pd.DataFrame, rows9: dict[int, pd.Series], rows10: dict[int, pd.Series], output_transform_dir: Path) -> tuple[pd.DataFrame, dict[int, dict]]:
    records = []
    provenance = {}
    for sid in sorted(int(value) for value in v9["section_id"]):
        label = _source_label(sid)
        row = (rows10 if sid <= SPLIT_SECTION else rows9)[sid].copy()
        source_prep = _source_prep(sid)
        source_transform = Path(str(row["section_transform_path"])).resolve()
        if not source_transform.is_file():
            raise FileNotFoundError(source_transform)
        output_transform = output_transform_dir / source_transform.name
        arrays = _load_npz(source_transform)
        arrays["mixed_release"] = np.asarray(RELEASE)
        arrays["mixed_source_release"] = np.asarray(label)
        arrays["mixed_coordinate_frame"] = np.asarray(OUTPUT_FRAME)
        arrays["mixed_stitch_section"] = np.asarray([SPLIT_SECTION], dtype=np.int64)
        np.savez_compressed(output_transform, **arrays)

        record = row.to_dict()
        record["section_id"] = sid
        record["section_transform_path"] = str(output_transform.resolve())
        record["coordinate_frame"] = OUTPUT_FRAME
        record["mixed_release"] = RELEASE
        record["mixed_source_release"] = label
        record["mixed_source_prep"] = str(source_prep.resolve())
        record["mixed_stitch_section"] = SPLIT_SECTION
        record["mixed_pair_51_53_source"] = "v10"
        record["mixed_pair_53_55_source"] = "v9"
        record["mixed_downstream_policy"] = "v9_from_section_55_to_preserve_reviewed_53_55_chain"
        records.append(record)
        provenance[sid] = {
            "source_release": label,
            "source_prep": str(source_prep.resolve()),
            "source_transform": str(source_transform),
            "output_transform": str(output_transform.resolve()),
        }

    table = pd.DataFrame(records).sort_values("z_um", kind="stable").reset_index(drop=True)
    if table["section_id"].duplicated().any():
        raise AssertionError("mixed H&E table has duplicate section IDs")
    table.to_parquet(OUTPUT_PREP / "he_sections.parquet", index=False)
    return table, provenance


def _copy_mixed_store(kind: str, section_ids: list[int]) -> dict:
    output_path = OUTPUT_PREP / (
        "uni2_features_mixed_v11.h5"
        if kind == "uni2"
        else "nucleus_path_mixed_v11/nucleus_features.h5"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(output_path)

    source_paths = {"v9": _source_store_path(V9_PREP, kind), "v10": _source_store_path(V10_PREP, kind)}
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    with h5py.File(source_paths["v9"], "r") as v9, h5py.File(source_paths["v10"], "r") as v10, h5py.File(output_path, "w") as out:
        out.attrs["format"] = str(v10.attrs.get("format", "deepspatial-uni2-grid-v1"))
        out.attrs["revision"] = str(uuid.uuid4())
        out.attrs["coordinate_frame"] = OUTPUT_FRAME
        out.attrs["mixed_release"] = RELEASE
        out.attrs["mixed_policy"] = "v10 sections <=53; v9 sections >=55; section 54 removed"
        out_sections = out.create_group("sections")
        copied = []
        for sid in section_ids:
            label = _source_label(sid)
            source = v10 if label == "v10_pre53" else v9
            key = _store_key(sid)
            if key not in source["sections"]:
                raise KeyError(f"{kind}: section {sid} missing from {label} store")
            source.copy(source["sections"][key], out_sections, name=key)
            group = out_sections[key]
            metadata = json.loads(group.attrs["metadata"])
            metadata["coordinate_frame"] = OUTPUT_FRAME
            provenance = dict(metadata.get("provenance", {}))
            provenance.update({
                "mixed_release": RELEASE,
                "mixed_source_release": label,
                "mixed_source_store": str(source_paths["v10" if label == "v10_pre53" else "v9"].resolve()),
                "mixed_stitch_section": SPLIT_SECTION,
            })
            metadata["provenance"] = provenance
            group.attrs["metadata"] = json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
            copied.append({
                "section_id": sid,
                "source_release": label,
                "shape": list(map(int, group["features"].shape)),
                "valid_fraction": float(np.asarray(group["valid"][:], dtype=bool).mean()),
            })
        out.flush()
    return {"output_store": str(output_path.resolve()), "kind": kind, "sections": copied}


def _set_obs_column(adata, key: str, values) -> None:
    array = np.asarray(values)
    if array.dtype.kind in "iufb":
        series = pd.Series(array, index=adata.obs_names)
    else:
        series = pd.Series(array.astype(str), index=adata.obs_names, dtype="string")
    adata.obs[key] = series


def _materialize_h5ad(table: pd.DataFrame, provenance: dict[int, dict]) -> dict:
    source_path = _source_h5ad()
    output_path = OUTPUT_REG / "9957_g0__st_to_mixed_v10_pre53_v9_post55_v11.h5ad"
    source = ad.read_h5ad(source_path)
    section_ids = pd.to_numeric(source.obs["section_id"], errors="raise").astype(int).to_numpy()
    if not set(np.unique(section_ids)).issubset(set(table["section_id"].astype(int))):
        raise AssertionError("H5AD contains a section not present in the mixed H&E table")
    if "spatial_latest_alignment_v9" not in source.obsm:
        raise KeyError("v10 H5AD is missing historical spatial_latest_alignment_v9")
    final = np.asarray(source.obsm["spatial_latest_alignment_v9"], dtype=np.float32).copy()
    source.obsm["spatial_mixed_v11"] = final
    for key in ("spatial", "spatial_registered", "spatial_st_corrected"):
        if key in source.obsm:
            source.obsm[key] = final.copy()

    labels = np.asarray([_source_label(int(sid)) for sid in section_ids], dtype=object)
    paths = [str((OUTPUT_PREP / "registration_transforms_mixed_v11" / f"section-{int(sid)}-9957__g0__section-{int(sid)}.npz").resolve()) for sid in section_ids]
    _set_obs_column(source, "coordinate_frame", np.full(source.n_obs, OUTPUT_FRAME, dtype=object))
    _set_obs_column(source, "mixed_release", np.full(source.n_obs, RELEASE, dtype=object))
    _set_obs_column(source, "mixed_source_release", labels)
    _set_obs_column(source, "mixed_stitch_section", np.full(source.n_obs, SPLIT_SECTION, dtype=np.int64))
    _set_obs_column(source, "registration_transform_path", paths)
    for key, col in (("x_global", 0), ("x_latest_alignment_um", 0), ("x_mixed_v11_um", 0), ("y_global", 1), ("y_latest_alignment_um", 1), ("y_mixed_v11_um", 1)):
        _set_obs_column(source, key, final[:, col].astype(np.float64))
    source.uns["deepspatial_mixed_alignment_v11"] = {
        "release": RELEASE,
        "coordinate_frame": OUTPUT_FRAME,
        "source_h5ad": str(source_path.resolve()),
        "section_policy": "v10 for sections <=53; v9 for sections >=55; section 54 removed",
        "pair_51_53": "v10",
        "pair_53_55": "v9",
        "shared_stitch_section": 53,
        "downstream_policy": "v9 from section 55 onward to preserve the reviewed v9 53->55 geometry",
        "active_coordinate_key": "spatial_mixed_v11",
        "feature_store": str((OUTPUT_PREP / "uni2_features_mixed_v11.h5").resolve()),
        "nucleus_store": str((OUTPUT_PREP / "nucleus_path_mixed_v11/nucleus_features.h5").resolve()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source.write_h5ad(output_path, compression="gzip")

    anchors_dir = OUTPUT_PREP / "anchors"
    anchors_dir.mkdir(parents=True, exist_ok=True)
    anchor_records = []
    for sid in sorted(np.unique(section_ids)):
        anchor = source[section_ids == int(sid)].copy()
        anchor.uns["deepspatial_series"] = {
            "series_id": "9957__g0",
            "section_id": int(sid),
            "section_thickness_um": 5.0,
            "coordinate_frame": OUTPUT_FRAME,
            "release": RELEASE,
            "mixed_source_release": _source_label(int(sid)),
        }
        path = anchors_dir / f"section-{int(sid):03d}.h5ad"
        anchor.write_h5ad(path, compression="gzip")
        anchor_records.append({"section_id": int(sid), "path": str(path.resolve()), "n_obs": int(anchor.n_obs)})
    return {
        "source": str(source_path.resolve()),
        "output": str(output_path.resolve()),
        "n_obs": int(source.n_obs),
        "n_sections": len(anchor_records),
        "anchors": anchor_records,
    }


def _validate_materialized(table: pd.DataFrame, store_results: list[dict], h5ad_result: dict) -> dict:
    ids = [int(value) for value in table["section_id"]]
    if REMOVED_SECTION in ids or len(ids) != len(set(ids)):
        raise AssertionError("mixed section filtering failed")
    if table["coordinate_frame"].nunique() != 1 or table["coordinate_frame"].iloc[0] != OUTPUT_FRAME:
        raise AssertionError("mixed H&E table frame is inconsistent")
    for result in store_results:
        actual = [int(item["section_id"]) for item in result["sections"]]
        if actual != sorted(ids):
            raise AssertionError(f"{result['kind']} store section IDs do not match H&E table")
    if not Path(h5ad_result["output"]).is_file():
        raise AssertionError("mixed H5AD was not written")
    return {
        "section_count": len(ids),
        "section_ids": ids,
        "removed_section_ids": [REMOVED_SECTION],
        "coordinate_frame": OUTPUT_FRAME,
        "pair_51_53_source": "v10",
        "pair_53_55_source": "v9",
        "source_release_by_section": {
            "v10_pre53": [sid for sid in ids if sid <= SPLIT_SECTION],
            "v9_post55": [sid for sid in ids if sid > SPLIT_SECTION],
        },
    }


def materialize() -> dict:
    for path in (OUTPUT_PREP, OUTPUT_REG, OUTPUT_QC):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {path}")
    OUTPUT_PREP.mkdir(parents=True)
    OUTPUT_REG.mkdir(parents=True)
    output_transform_dir = OUTPUT_PREP / "registration_transforms_mixed_v11"
    output_transform_dir.mkdir()

    v9, v10, rows9, rows10 = _validate_inputs()
    table, provenance = _build_table(v9, v10, rows9, rows10, output_transform_dir)
    section_ids = [int(value) for value in table["section_id"]]
    store_results = [_copy_mixed_store(kind, section_ids) for kind in ("uni2", "nucleus")]
    h5ad_result = _materialize_h5ad(table, provenance)
    validation = _validate_materialized(table, store_results, h5ad_result)

    manifest = {
        "status": "materialized",
        "release": RELEASE,
        "coordinate_frame": OUTPUT_FRAME,
        "split_section": SPLIT_SECTION,
        "removed_section_ids": [REMOVED_SECTION],
        "source_releases": {
            "v10": str(V10_PREP.resolve()),
            "v9": str(V9_PREP.resolve()),
        },
        "policy": {
            "sections_le_53": "v10",
            "pair_51_53": "v10",
            "pair_53_55": "v9",
            "sections_ge_55": "v9, preserving the reviewed v9 53->55 branch",
        },
        "he_table": str((OUTPUT_PREP / "he_sections.parquet").resolve()),
        "transform_dir": str(output_transform_dir.resolve()),
        "store_results": store_results,
        "h5ad": h5ad_result,
        "validation": validation,
        "provenance_by_section": provenance,
        "cache_policy": "rebuild UOT/path caches against the mixed v11 feature and nucleus stores",
    }
    _write_json(OUTPUT_PREP / "mixed_alignment_manifest.json", manifest)
    _write_json(OUTPUT_REG / "mixed_alignment_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="materialize the immutable mixed release")
    args = parser.parse_args()
    if not args.run:
        parser.error("pass --run to materialize")
    result = materialize()
    print(json.dumps({
        "status": result["status"],
        "release": result["release"],
        "coordinate_frame": result["coordinate_frame"],
        "he_table": result["he_table"],
        "h5ad": result["h5ad"]["output"],
        "stores": [item["output_store"] for item in result["store_results"]],
        "policy": result["policy"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
