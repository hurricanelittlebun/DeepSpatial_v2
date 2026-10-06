"""Materialize a 9957/g0 release after the feature-support 53 -> 55 fix.

The v9 release is immutable.  This release adds the residual affine fitted on
the actual v9 UNI2 support grids.  Section 53 and every earlier section stay
unchanged; section 55 and all downstream sections receive one explicit
target-frame residual.  Cached UNI2 and nucleus grids are resampled, not
re-encoded.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import types
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
GROUP = ROOT / "data" / "9957_g0"
SOURCE_PREP = GROUP / "reconstruction_prep_v9_postrotate_ccw90_cutoff91"
SOURCE_REG = GROUP / "registration_correction_he_v9_postrotate_ccw90_cutoff91"
RESIDUAL_ROOT = GROUP / "feature_support_stalign_53_55_v3"
OUTPUT_PREP = GROUP / "reconstruction_prep_v10_feature_support_residual"
OUTPUT_REG = GROUP / "registration_correction_he_v10_feature_support_residual"
OUTPUT_QC = GROUP / "qc" / "he_alignment_all_sections_v7_feature_support_residual"
OUTPUT_FRAME = "9957__g0_registered_feature_support_residual_v10"
RELEASE = "9957_feature_support_residual_v10"
VERSION_TAG = "v10"
START_SECTION = 55


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Load histology modules without importing deepspatial.__init__, because the
# standalone materialization environment intentionally does not need the
# training stack.
_HIST_PACKAGE = "_deepspatial_histology_v10"
package = types.ModuleType(_HIST_PACKAGE)
package.__path__ = [str(ROOT / "deepspatial" / "histology")]
sys.modules[_HIST_PACKAGE] = package
_feature_module = _load_module(
    ROOT / "deepspatial" / "histology" / "feature_store.py",
    f"{_HIST_PACKAGE}.feature_store",
)
sys.modules[f"{_HIST_PACKAGE}.feature_store"] = _feature_module
_frame_module = _load_module(
    ROOT / "deepspatial" / "histology" / "frame_resample.py",
    f"{_HIST_PACKAGE}.frame_resample",
)
FeatureStore = _feature_module.FeatureStore
resample_feature_store = _frame_module.resample_feature_store


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _apply_affine(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape [N,2]")
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    return (homogeneous @ np.asarray(matrix, dtype=np.float64).T)[:, :2]


def _load_residual() -> tuple[np.ndarray, dict]:
    summary_path = RESIDUAL_ROOT / "feature_support_residual_summary.json"
    affine_path = RESIDUAL_ROOT / "pair-53-55" / "feature_support_residual_affine.npz"
    if not summary_path.is_file() or not affine_path.is_file():
        raise FileNotFoundError("feature-support residual artifact is incomplete")
    summary = _read_json(summary_path)
    with np.load(affine_path, allow_pickle=False) as payload:
        matrix = np.asarray(payload["affine"], dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("invalid feature-support residual affine")
    if not np.allclose(matrix[2], [0.0, 0.0, 1.0], atol=1e-8, rtol=0.0):
        raise ValueError("feature-support residual is not homogeneous")
    if float(np.linalg.det(matrix[:2, :2])) <= 0:
        raise ValueError("feature-support residual contains a reflection")
    return matrix, summary


def _copy_transforms_and_table(source_table_path: Path, output_table_path: Path, output_transform_dir: Path, residual: np.ndarray, residual_summary: dict) -> tuple[dict, dict[int, np.ndarray]]:
    table = pd.read_parquet(source_table_path).copy()
    table["section_id"] = pd.to_numeric(table["section_id"], errors="raise").astype(int)
    if table["section_id"].duplicated().any():
        raise ValueError("source H&E table has duplicate section IDs")
    output_transform_dir.mkdir(parents=True, exist_ok=False)
    matrices = {}
    residual_json = json.dumps(residual.tolist(), separators=(",", ":"))
    for row in table.itertuples(index=False):
        sid = int(row.section_id)
        source_path = Path(str(row.section_transform_path)).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        arrays = _load_npz(source_path)
        old = np.asarray(arrays["matrix"], dtype=np.float64)
        apply = sid >= START_SECTION
        final = residual @ old if apply else old
        arrays[f"matrix_before_feature_support_residual_{VERSION_TAG}"] = old
        arrays["matrix"] = final
        arrays["feature_support_residual_affine"] = residual
        arrays["feature_support_residual_applied"] = np.asarray([int(apply)], dtype=np.uint8)
        arrays["feature_support_residual_release"] = np.asarray(RELEASE)
        arrays["feature_support_residual_start_section"] = np.asarray([START_SECTION], dtype=np.int64)
        arrays["feature_support_residual_summary_json"] = np.asarray(json.dumps(residual_summary, ensure_ascii=False, separators=(",", ":")))
        output_path = output_transform_dir / source_path.name
        np.savez_compressed(output_path, **arrays)
        matrices[sid] = final
        table.loc[table["section_id"] == sid, "section_transform_path"] = str(output_path.resolve())

    table["coordinate_frame"] = OUTPUT_FRAME
    table["feature_support_residual_release"] = RELEASE
    table["feature_support_residual_applied"] = table["section_id"] >= START_SECTION
    table["feature_support_residual_start_section"] = START_SECTION
    table["feature_support_residual_affine_json"] = residual_json
    table["feature_support_residual_summary_path"] = str((RESIDUAL_ROOT / "feature_support_residual_summary.json").resolve())
    table.to_parquet(output_table_path, index=False)
    return {
        "source_table": str(source_table_path.resolve()),
        "output_table": str(output_table_path.resolve()),
        "source_section_count": int(len(table)),
        "output_section_count": int(len(table)),
        "output_section_ids": [int(v) for v in table["section_id"]],
        "residual_applied_sections": [int(v) for v in table.loc[table["feature_support_residual_applied"], "section_id"]],
    }, matrices


def _resample_store(source_path: Path, output_path: Path, residual: np.ndarray, residual_summary: dict, store_name: str) -> dict:
    source = FeatureStore(source_path, cache_mb=1024)
    matrices = {
        sid: (residual.copy() if int(sid) >= START_SECTION else np.eye(3, dtype=np.float64))
        for sid in source.sections
    }
    result = resample_feature_store(
        source_path,
        output_path,
        matrices,
        output_frame=OUTPUT_FRAME,
        provenance={
            "release": RELEASE,
            "store_name": store_name,
            "source_store": str(source_path.resolve()),
            "feature_support_residual_summary": residual_summary,
            "residual_semantics": "p_new = residual_affine @ p_v9 for sections >=55",
        },
    )
    return result


def _materialize_h5ad(source_path: Path, output_path: Path, anchors_dir: Path, transform_dir: Path, residual: np.ndarray, residual_summary: dict) -> dict:
    source = ad.read_h5ad(source_path)
    if "section_id" not in source.obs or "spatial_registered" not in source.obsm:
        raise KeyError("v9 H5AD requires obs['section_id'] and obsm['spatial_registered']")
    section_ids = pd.to_numeric(source.obs["section_id"], errors="raise").astype(int).to_numpy()
    before = np.asarray(source.obsm["spatial_registered"], dtype=np.float64)
    final = before.copy()
    rows = section_ids >= START_SECTION
    if rows.any():
        final[rows] = _apply_affine(before[rows], residual)
    source.obsm[f"spatial_before_feature_support_residual_{VERSION_TAG}"] = before.astype(np.float32)
    source.obsm[f"spatial_feature_support_residual_{VERSION_TAG}"] = final.astype(np.float32)
    # Keep historical v9 keys; these three are the active global keys used by
    # the reconstruction entry points and therefore move to the new frame.
    for key in ("spatial", "spatial_registered", "spatial_st_corrected"):
        if key in source.obsm:
            source.obsm[key] = final.astype(np.float32)
    source.obs["coordinate_frame"] = OUTPUT_FRAME
    source.obs["feature_support_residual_release"] = RELEASE
    source.obs["feature_support_residual_applied"] = rows
    source.obs["feature_support_residual_start_section"] = START_SECTION
    source.obs["x_feature_support_residual_v10_um"] = final[:, 0]
    source.obs["y_feature_support_residual_v10_um"] = final[:, 1]
    source.obs["registration_transform_path"] = [
        str((transform_dir / f"section-{int(sid)}-9957__g0__section-{int(sid)}.npz").resolve())
        for sid in section_ids
    ]
    source.uns["deepspatial_feature_support_residual_v10"] = {
        "release": RELEASE,
        "coordinate_frame": OUTPUT_FRAME,
        "source_h5ad": str(source_path.resolve()),
        "fixed_section": 53,
        "moving_section": 55,
        "applied_to_sections_ge": START_SECTION,
        "residual_affine": residual.tolist(),
        "residual_summary": residual_summary,
        "historical_coordinate_key": "spatial_latest_alignment_v9",
        "active_coordinate_key": "spatial_feature_support_residual_v10",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source.write_h5ad(output_path, compression="gzip")

    anchors_dir.mkdir(parents=True, exist_ok=False)
    records = []
    for sid in sorted(np.unique(section_ids)):
        anchor = source[section_ids == int(sid)].copy()
        anchor.uns["deepspatial_series"] = {
            "series_id": "9957__g0",
            "section_id": int(sid),
            "section_thickness_um": 5.0,
            "coordinate_frame": OUTPUT_FRAME,
            "release": RELEASE,
            "feature_support_residual_applied": bool(int(sid) >= START_SECTION),
        }
        path = anchors_dir / f"section-{int(sid):03d}.h5ad"
        anchor.write_h5ad(path, compression="gzip")
        records.append({"section_id": int(sid), "path": str(path.resolve()), "n_obs": int(anchor.n_obs)})
    return {
        "source": str(source_path.resolve()),
        "output": str(output_path.resolve()),
        "n_obs": int(source.n_obs),
        "n_sections": len(records),
        "anchors": records,
    }


def _boundary(mask: np.ndarray, origin: list[float], spacing: list[float], max_points: int = 6000) -> np.ndarray:
    from scipy.ndimage import binary_erosion

    value = np.asarray(mask, dtype=bool)
    edge = value & ~binary_erosion(value, structure=np.ones((3, 3), bool), border_value=0)
    rows, cols = np.nonzero(edge)
    if len(rows) > max_points:
        indices = np.linspace(0, len(rows) - 1, max_points, dtype=np.int64)
        rows, cols = rows[indices], cols[indices]
    return np.column_stack([float(origin[0]) + cols * float(spacing[0]), float(origin[1]) + rows * float(spacing[1])]) if len(rows) else np.empty((0, 2), dtype=float)


def _render_qc(table_path: Path, store_path: Path, output_dir: Path) -> dict:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite QC directory: {output_dir}")
    output_dir.mkdir(parents=True)
    table = pd.read_parquet(table_path)
    table["section_id"] = pd.to_numeric(table["section_id"], errors="raise").astype(int)
    table_by_id = {int(row.section_id): row for row in table.itertuples(index=False)}
    store = FeatureStore(store_path, cache_mb=512)
    records = []
    with h5py.File(store_path, "r") as handle:
        for sid_text in store.sections:
            metadata = store.metadata[sid_text]
            sid = int(sid_text)
            valid = np.asarray(handle["sections"][store._key(sid_text)]["valid"][:], dtype=bool)
            records.append({
                "section_id": sid,
                "z_um": float(metadata["z_um"]),
                "boundary": _boundary(valid, metadata["origin_um"], metadata["spacing_um"]),
                "registration_confidence": float(getattr(table_by_id[sid], "registration_confidence", np.nan)),
            })
    records.sort(key=lambda item: item["z_um"])
    all_points = np.concatenate([r["boundary"] for r in records if len(r["boundary"])], axis=0)
    xmin, ymin = all_points.min(axis=0)
    xmax, ymax = all_points.max(axis=0)
    pad = 0.04 * max(xmax - xmin, ymax - ymin, 1.0)
    bounds = (xmin - pad, xmax + pad, ymax + pad, ymin - pad)

    def draw(ax, record, color, size):
        points = record["boundary"]
        if len(points):
            ax.scatter(points[:, 0], points[:, 1], s=size, color=color, linewidths=0, rasterized=True)

    def configure(ax):
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("global X (µm)")
        ax.set_ylabel("global Y (µm)")

    section_dir = output_dir / "sections"
    pair_dir = output_dir / "adjacent_pairs"
    section_dir.mkdir()
    pair_dir.mkdir()
    section_paths = []
    pair_paths = []
    for index, record in enumerate(records):
        previous = records[index - 1] if index else None
        following = records[index + 1] if index + 1 < len(records) else None
        fig, ax = plt.subplots(figsize=(8, 7), dpi=170, constrained_layout=True)
        if previous is not None:
            draw(ax, previous, "#0057b7", 2.0)
        if following is not None:
            draw(ax, following, "#00a651", 2.0)
        draw(ax, record, "#e41a1c", 2.8)
        configure(ax)
        ax.set_title(f"9957/g0 section {record['section_id']:03d} | prev={previous['section_id'] if previous else '—'}, next={following['section_id'] if following else '—'}\nred=current; blue=previous; green=next | v10 feature-support residual", fontsize=11)
        path = section_dir / f"section-{record['section_id']:03d}__he_global_alignment.png"
        fig.savefig(path, facecolor="white")
        plt.close(fig)
        section_paths.append(path)
    for left, right in zip(records[:-1], records[1:]):
        fig, ax = plt.subplots(figsize=(8, 7), dpi=170, constrained_layout=True)
        draw(ax, left, "#e41a1c", 2.8)
        draw(ax, right, "#0057b7", 2.8)
        configure(ax)
        ax.set_title(f"9957/g0 adjacent H&E support: {left['section_id']:03d} red vs {right['section_id']:03d} blue", fontsize=11)
        path = pair_dir / f"pair-{left['section_id']:03d}-{right['section_id']:03d}__he_alignment.png"
        fig.savefig(path, facecolor="white")
        plt.close(fig)
        pair_paths.append(path)

    selected = {sid: next(record for record in records if record["section_id"] == sid) for sid in (51, 53, 55) if sid in {r["section_id"] for r in records}}
    fig, ax = plt.subplots(figsize=(9, 8), dpi=220, constrained_layout=True)
    for sid, color in ((51, "#0057b7"), (53, "#e41a1c"), (55, "#00a651")):
        if sid in selected:
            draw(ax, selected[sid], color, 4.0)
    configure(ax)
    ax.set_title("9957/g0 fixed-bounds diagnostic: section 51 blue, 53 red, 55 green\nv10 residual fitted on actual UNI2 support grids", fontsize=12)
    diagnostic = output_dir / "diagnostics_section-051-053-055_fixed_bounds.png"
    fig.savefig(diagnostic, facecolor="white")
    plt.close(fig)
    manifest = {
        "status": "complete",
        "release": RELEASE,
        "coordinate_frame": OUTPUT_FRAME,
        "he_table": str(table_path.resolve()),
        "feature_store": str(store_path.resolve()),
        "section_count": len(records),
        "section_images": [str(path.resolve()) for path in section_paths],
        "adjacent_pair_images": [str(path.resolve()) for path in pair_paths],
        "fixed_bounds_diagnostic": str(diagnostic.resolve()),
        "semantics": "Boundaries are valid H&E/UNI2 feature-grid supports in the v10 frame; 53 is fixed and the 55+ branch receives one residual affine.",
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def materialize() -> dict:
    for path in (OUTPUT_PREP, OUTPUT_REG, OUTPUT_QC):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    residual, residual_summary = _load_residual()
    source_table = pd.read_parquet(SOURCE_PREP / "he_sections.parquet")
    source_ids = sorted(int(v) for v in source_table["section_id"])
    if 53 not in source_ids or 55 not in source_ids or 54 in source_ids:
        raise ValueError(f"unexpected v9 section IDs around 53/55: {source_ids}")
    source_reg_h5ad = next(SOURCE_REG.glob("*.h5ad"), None)
    if source_reg_h5ad is None:
        raise FileNotFoundError(f"No v9 H5AD in {SOURCE_REG}")

    OUTPUT_PREP.mkdir(parents=True)
    OUTPUT_REG.mkdir(parents=True)
    table_result, matrices = _copy_transforms_and_table(
        SOURCE_PREP / "he_sections.parquet",
        OUTPUT_PREP / "he_sections.parquet",
        OUTPUT_PREP / "registration_transforms_feature_support_residual_v10",
        residual,
        residual_summary,
    )
    if not np.allclose(matrices[51], _load_npz(Path(str(source_table.loc[source_table.section_id == 51, "section_transform_path"].iloc[0]))) ["matrix"]):
        raise AssertionError("section 51 changed unexpectedly")
    if not np.allclose(matrices[53], _load_npz(Path(str(source_table.loc[source_table.section_id == 53, "section_transform_path"].iloc[0]))) ["matrix"]):
        raise AssertionError("section 53 changed unexpectedly")

    audit_target = OUTPUT_PREP / "stalign_final_53_55"
    shutil.copytree(SOURCE_PREP / "stalign_final_53_55", audit_target)
    residual_target = OUTPUT_PREP / "feature_support_stalign_53_55_v3"
    shutil.copytree(RESIDUAL_ROOT, residual_target)

    feature_result = _resample_store(
        SOURCE_PREP / "uni2_features_latest_alignment_cutoff91_v9.h5",
        OUTPUT_PREP / "uni2_features_feature_support_residual_v10.h5",
        residual,
        residual_summary,
        "uni2",
    )
    nucleus_result = _resample_store(
        SOURCE_PREP / "nucleus_path_latest_alignment_cutoff91_v9" / "nucleus_features.h5",
        OUTPUT_PREP / "nucleus_path_feature_support_residual_v10" / "nucleus_features.h5",
        residual,
        residual_summary,
        "nucleus",
    )
    candidate_result = _materialize_h5ad(
        source_reg_h5ad,
        OUTPUT_REG / "9957_g0__st_to_feature_support_residual_v10.h5ad",
        OUTPUT_PREP / "anchors",
        OUTPUT_PREP / "registration_transforms_feature_support_residual_v10",
        residual,
        residual_summary,
    )
    qc = _render_qc(
        OUTPUT_PREP / "he_sections.parquet",
        OUTPUT_PREP / "uni2_features_feature_support_residual_v10.h5",
        OUTPUT_QC,
    )
    manifest = {
        "status": "materialized",
        "release": RELEASE,
        "coordinate_frame": OUTPUT_FRAME,
        "source_prep": str(SOURCE_PREP.resolve()),
        "source_reg": str(SOURCE_REG.resolve()),
        "source_h5ad": str(source_reg_h5ad.resolve()),
        "fixed_section": 53,
        "moving_section": 55,
        "residual_applied_from_section": START_SECTION,
        "residual_affine": residual.tolist(),
        "residual_summary": residual_summary,
        "he_table": table_result,
        "feature_store": feature_result,
        "nucleus_store": nucleus_result,
        "candidate": candidate_result,
        "qc": qc,
        "historical_release_preserved": str(SOURCE_PREP.resolve()),
        "cache_policy": "Rebuild UOT/path caches because the morphology feature frame changed from v9 to v10.",
    }
    _write_json(OUTPUT_PREP / "feature_support_residual_manifest.json", manifest)
    _write_json(OUTPUT_REG / "feature_support_residual_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="materialize the independent v10 release")
    args = parser.parse_args()
    if not args.run:
        parser.error("pass --run to materialize")
    result = materialize()
    print(json.dumps({
        "status": result["status"],
        "release": result["release"],
        "coordinate_frame": result["coordinate_frame"],
        "residual_affine": result["residual_affine"],
        "h5ad": result["candidate"]["output"],
        "uni2": result["feature_store"]["output_store"],
        "nucleus": result["nucleus_store"]["output_store"],
        "diagnostic": result["qc"]["fixed_bounds_diagnostic"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
