"""Render static 3-D QC figures for the reviewed 9957/g0 mixed v11 frame.

The figure uses the same physical coordinate frame as the v11 H&E feature and
nucleus stores.  H&E support is drawn from the feature-grid validity masks;
ST points are read from ``spatial_mixed_v11`` in the mixed H5AD.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
GROUP = ROOT / "data" / "9957_g0"
PREP = GROUP / "reconstruction_prep_v11_mixed_v10_pre53_v9_post55"
REG = GROUP / "registration_correction_he_v11_mixed_v10_pre53_v9_post55"
STORE = PREP / "uni2_features_mixed_v11.h5"
H5AD = REG / "9957_g0__st_to_mixed_v10_pre53_v9_post55_v11.h5ad"
OUTPUT = GROUP / "qc" / "he_alignment_all_sections_v8_mixed_v10_pre53_v9_post55" / "3d_static"

COLORS = [
    "#e41a1c",  # 1
    "#0057b7",  # 11
    "#00a651",  # 21
    "#6a1b9a",  # 31
    "#ff7f00",  # 41
    "#008fb3",  # 51
    "#b8860b",  # 61
    "#4d4d4d",  # 71
    "#e6007e",  # 81
    "#003f5c",  # 91
]


def _feature_store_class():
    package_name = "_mixed_v11_feature_store"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / "deepspatial" / "histology")]
    sys.modules[package_name] = package
    path = ROOT / "deepspatial" / "histology" / "feature_store.py"
    spec = importlib.util.spec_from_file_location(f"{package_name}.feature_store", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import FeatureStore from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FeatureStore


def _boundary(mask: np.ndarray, origin: list[float], spacing: list[float], max_points: int) -> np.ndarray:
    from scipy.ndimage import binary_erosion

    mask = np.asarray(mask, dtype=bool)
    boundary = mask & ~binary_erosion(mask, structure=np.ones((3, 3), bool), border_value=0)
    rows, cols = np.nonzero(boundary)
    if len(rows) > max_points:
        keep = np.linspace(0, len(rows) - 1, max_points, dtype=np.int64)
        rows, cols = rows[keep], cols[keep]
    if not len(rows):
        return np.empty((0, 2), dtype=float)
    return np.column_stack(
        [
            float(origin[0]) + cols * float(spacing[0]),
            float(origin[1]) + rows * float(spacing[1]),
        ]
    )


def _load_he_boundaries(max_points: int) -> tuple[list[dict], str]:
    FeatureStore = _feature_store_class()
    store = FeatureStore(STORE, cache_mb=256)
    records = []
    with h5py.File(STORE, "r") as handle:
        for sid in store.sections:
            metadata = store.metadata[sid]
            key = store._key(sid)
            valid = np.asarray(handle["sections"][key]["valid"][:], dtype=bool)
            points = _boundary(
                valid,
                metadata["origin_um"],
                metadata["spacing_um"],
                max_points,
            )
            records.append(
                {
                    "section_id": int(sid),
                    "z_um": float(metadata["z_um"]),
                    "points": points,
                }
            )
    frame = str(store.metadata[store.sections[0]]["coordinate_frame"])
    records.sort(key=lambda record: record["z_um"])
    return records, frame


def _load_st() -> tuple[np.ndarray, np.ndarray, dict[int, float], str]:
    table = pd.read_parquet(PREP / "he_sections.parquet")
    z_by_section = {
        int(row.section_id): float(row.z_um)
        for row in table.itertuples(index=False)
    }
    adata = ad.read_h5ad(H5AD, backed="r")
    try:
        if "spatial_mixed_v11" not in adata.obsm:
            raise KeyError("mixed H5AD does not contain obsm['spatial_mixed_v11']")
        xy = np.asarray(adata.obsm["spatial_mixed_v11"], dtype=float)
        section_ids = pd.to_numeric(adata.obs["section_id"], errors="raise").astype(int).to_numpy()
        frame_values = set(adata.obs["coordinate_frame"].astype(str).unique())
        if frame_values != {next(iter(frame_values))}:
            raise ValueError(f"H5AD has multiple coordinate frames: {sorted(frame_values)}")
        frame = next(iter(frame_values))
    finally:
        adata.file.close()
    return xy, section_ids, z_by_section, frame


def _draw(
    he_records: list[dict],
    st_xy: np.ndarray,
    st_sections: np.ndarray,
    z_by_section: dict[int, float],
    frame: str,
    output: Path,
    *,
    include_he: bool,
    z_stretch: float,
) -> dict:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    z_all = [record["z_um"] for record in he_records]
    z_all.extend(z_by_section.values())
    z_min, z_max = float(min(z_all)), float(max(z_all))
    z_span = max(z_max - z_min, 1.0)

    fig = plt.figure(figsize=(17, 13), dpi=220)
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection="3d", facecolor="white")

    if include_he:
        for record in he_records:
            points = record["points"]
            if not len(points):
                continue
            z = np.full(len(points), (record["z_um"] - z_min) * z_stretch)
            ax.scatter(
                points[:, 0], points[:, 1], z,
                s=0.50, c="#303030", alpha=0.45,
                depthshade=False, linewidths=0, rasterized=True,
            )

    handles = []
    anchor_ids = sorted(int(value) for value in np.unique(st_sections))
    for index, sid in enumerate(anchor_ids):
        rows = st_sections == sid
        points = st_xy[rows]
        if not len(points):
            continue
        if sid not in z_by_section:
            raise KeyError(f"section {sid} is missing from H&E z metadata")
        color = COLORS[index % len(COLORS)]
        z = np.full(len(points), (z_by_section[sid] - z_min) * z_stretch)
        ax.scatter(
            points[:, 0], points[:, 1], z,
            s=0.34 if include_he else 0.42,
            c=color, alpha=0.95, depthshade=False,
            linewidths=0, rasterized=True,
        )
        handles.append(
            Line2D(
                [0], [0], marker="o", color="none",
                markerfacecolor=color, markeredgecolor=color,
                markersize=7, label=f"ST {sid}",
            )
        )

    x_values = np.concatenate([record["points"][:, 0] for record in he_records if len(record["points"])])
    y_values = np.concatenate([record["points"][:, 1] for record in he_records if len(record["points"])])
    x_values = np.concatenate([x_values, st_xy[:, 0]])
    y_values = np.concatenate([y_values, st_xy[:, 1]])
    x_span = max(float(x_values.max() - x_values.min()), 1.0)
    y_span = max(float(y_values.max() - y_values.min()), 1.0)

    ax.set_xlabel("global X (µm)", labelpad=10, fontsize=12)
    ax.set_ylabel("global Y (µm)", labelpad=10, fontsize=12)
    ax.set_zlabel("Z (µm; display stretched ×5)", labelpad=10, fontsize=12)
    title = "9957/g0 mixed v11 3-D alignment QC\n"
    title += "dark gray = registered H&E/UNI2 support; colored = ST anchors" if include_he else "colored = ST anchors only"
    ax.set_title(title + f"\nframe: {frame}", pad=20, fontsize=15)
    ax.view_init(elev=25, azim=-58)
    ax.set_proj_type("ortho")
    ax.set_box_aspect((x_span, y_span, z_span * z_stretch))
    ax.grid(False)
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9, title="ST sections")
    fig.subplots_adjust(left=0.02, right=0.82, bottom=0.04, top=0.88)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return {
        "output": str(output.resolve()),
        "coordinate_frame": frame,
        "include_he_support": include_he,
        "he_section_count": len(he_records),
        "st_anchor_section_ids": anchor_ids,
        "st_cell_count": int(len(st_xy)),
        "z_display_stretch": z_stretch,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--he-points-per-section", type=int, default=1400)
    parser.add_argument("--z-stretch", type=float, default=5.0)
    args = parser.parse_args()
    if args.he_points_per_section <= 0 or args.z_stretch <= 0:
        raise ValueError("point limit and z stretch must be positive")

    he_records, he_frame = _load_he_boundaries(args.he_points_per_section)
    st_xy, st_sections, z_by_section, st_frame = _load_st()
    if he_frame != st_frame:
        raise ValueError(f"H&E/ST frame mismatch: {he_frame!r} vs {st_frame!r}")

    combined = OUTPUT / "9957_g0_mixed_v11_he_st_alignment_3d.png"
    anchors = OUTPUT / "9957_g0_mixed_v11_st_anchor_only_3d.png"
    results = [
        _draw(he_records, st_xy, st_sections, z_by_section, st_frame, combined, include_he=True, z_stretch=args.z_stretch),
        _draw(he_records, st_xy, st_sections, z_by_section, st_frame, anchors, include_he=False, z_stretch=args.z_stretch),
    ]
    manifest = {
        "status": "created",
        "source_h5ad": str(H5AD.resolve()),
        "source_feature_store": str(STORE.resolve()),
        "he_table": str((PREP / "he_sections.parquet").resolve()),
        "coordinate_frame": st_frame,
        "policy": "available sections <=53 from v10; available sections >=55 from v9; section 54 removed",
        "figures": results,
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
