#!/usr/bin/env python3
"""Render static 3-D alignment figures for the final 9957/g0 v3 frame.

The H&E support is read from the final immutable UNI2 feature store.  The ST
points are read from the newly materialized v3 H5AD.  Both are plotted in the
same physical global XY frame and at the same physical Z values; only the
displayed Z axis is stretched for visibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import anndata as ad
import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
GROUP = ROOT / "data" / "9957_g0"
FRAME = "9957__g0_registered_final_he_53_55_v1_chain_v1"
HE_TABLE = GROUP / "reconstruction_prep_final_he_53_55_v1_chain_v1" / "he_sections.parquet"
FEATURE_STORE = HE_TABLE.parent / "uni2_features_final_he_53_55_v1_chain_v1.h5"
H5AD = (
    GROUP
    / "registration_final_st_residual_v3_global_frame"
    / "9957_g0__st_to_final_he_53_55_v1_chain_v1__st_residual_v3.h5ad"
)
OUTPUT = GROUP / "qc" / "final_st_he_alignment_v3_global_frame_3d"


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def validate_coordinate_frames(he_frame: str, st_frame: str) -> str:
    if str(he_frame) != str(st_frame):
        raise ValueError(f"coordinate frame mismatch: H&E={he_frame!r}, ST={st_frame!r}")
    if not str(he_frame):
        raise ValueError("coordinate frame must be non-empty")
    return str(he_frame)


def z_display_values(z_um: np.ndarray, z_stretch: float) -> np.ndarray:
    values = np.asarray(z_um, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("z_um must be a finite one-dimensional array")
    if not np.isfinite(z_stretch) or z_stretch <= 0:
        raise ValueError("z_stretch must be positive and finite")
    return (values - float(values.min())) * float(z_stretch)


def _boundary(mask: np.ndarray, origin: list[float], spacing: list[float], max_points: int) -> np.ndarray:
    from scipy.ndimage import binary_erosion

    value = np.asarray(mask, dtype=bool)
    if value.ndim != 2:
        raise ValueError(f"H&E valid mask must be 2-D, got {value.shape}")
    edge = value & ~binary_erosion(value, structure=np.ones((3, 3), dtype=bool), border_value=0)
    rows, cols = np.nonzero(edge)
    if len(rows) > max_points:
        keep = np.linspace(0, len(rows) - 1, max_points, dtype=np.int64)
        rows, cols = rows[keep], cols[keep]
    if not len(rows):
        return np.empty((0, 2), dtype=np.float64)
    return np.column_stack(
        [
            float(origin[0]) + cols * float(spacing[0]),
            float(origin[1]) + rows * float(spacing[1]),
        ]
    )


def load_he_records(feature_store: Path, he_table: Path, max_points: int) -> tuple[list[dict[str, Any]], str]:
    table = pd.read_parquet(he_table).copy()
    table["section_id"] = pd.to_numeric(table["section_id"], errors="raise").astype(int)
    table_by_section = {int(row.section_id): row for row in table.itertuples(index=False)}
    if len(table_by_section) != len(table):
        raise ValueError("H&E table has duplicate section_id values")
    table_frames = set(table["coordinate_frame"].astype(str))
    if table_frames != {FRAME}:
        raise ValueError(f"H&E table frames are {sorted(table_frames)}, expected {FRAME!r}")

    records: list[dict[str, Any]] = []
    with h5py.File(feature_store, "r") as handle:
        for key, group in handle["sections"].items():
            if key.startswith("_pending_"):
                continue
            sid = int(group.attrs["section_id"])
            metadata = json.loads(group.attrs["metadata"])
            if sid not in table_by_section:
                raise ValueError(f"feature store section {sid} is absent from H&E table")
            frame = str(metadata["coordinate_frame"])
            validate_coordinate_frames(FRAME, frame)
            points = _boundary(
                np.asarray(group["valid"][:], dtype=bool),
                metadata["origin_um"],
                metadata["spacing_um"],
                max_points,
            )
            z_feature = float(metadata["z_um"])
            z_table = float(table_by_section[sid].z_um)
            if not np.isclose(z_feature, z_table, atol=1e-6):
                raise ValueError(f"section {sid} feature-store z={z_feature} differs from table z={z_table}")
            records.append(
                {
                    "section_id": sid,
                    "z_um": z_feature,
                    "points": points,
                    "coordinate_frame": frame,
                }
            )
    if len(records) != len(table):
        raise ValueError(f"feature store has {len(records)} sections but H&E table has {len(table)}")
    records.sort(key=lambda record: (record["z_um"], record["section_id"]))
    return records, FRAME


def load_st(h5ad: Path, he_table: Path) -> tuple[np.ndarray, np.ndarray, dict[int, float], str]:
    table = pd.read_parquet(he_table)
    z_by_section = {int(row.section_id): float(row.z_um) for row in table.itertuples(index=False)}
    adata = ad.read_h5ad(h5ad, backed="r")
    try:
        key = "spatial_st_residual_v3"
        if key not in adata.obsm:
            raise KeyError(f"final H5AD is missing obsm[{key!r}]")
        xy = np.asarray(adata.obsm[key], dtype=np.float64)
        sections = pd.to_numeric(adata.obs["section_id"], errors="raise").astype(int).to_numpy()
        if not np.isfinite(xy).all() or xy.shape != (adata.n_obs, 2):
            raise ValueError(f"invalid final ST coordinate shape/value: {xy.shape}")
        frames = set(adata.obs["coordinate_frame"].astype(str).unique())
        if frames != {FRAME}:
            raise ValueError(f"final H5AD frames are {sorted(frames)}, expected {FRAME!r}")
        if "st_residual_v3_applied" not in adata.obs or not bool(adata.obs["st_residual_v3_applied"].all()):
            raise ValueError("not every ST row is marked st_residual_v3_applied")
        missing = sorted(set(np.unique(sections)) - set(z_by_section))
        if missing:
            raise ValueError(f"ST sections missing H&E z metadata: {missing}")
    finally:
        adata.file.close()
    return xy, sections, z_by_section, FRAME


def _draw(
    he_records: list[dict[str, Any]],
    st_xy: np.ndarray,
    st_sections: np.ndarray,
    z_by_section: dict[int, float],
    frame: str,
    output: Path,
    *,
    include_he: bool,
    z_stretch: float,
    title_suffix: str,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    he_z = np.asarray([record["z_um"] for record in he_records], dtype=float)
    all_z = np.concatenate([he_z, np.asarray([z_by_section[int(sid)] for sid in np.unique(st_sections)], dtype=float)])
    z_display = z_display_values(all_z, z_stretch)
    z_min = float(all_z.min())
    z_max = float(all_z.max())
    norm = plt.Normalize(vmin=z_min, vmax=z_max if z_max > z_min else z_min + 1.0)
    cmap = plt.get_cmap("turbo")

    fig = plt.figure(figsize=(17, 13), dpi=220)
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection="3d", facecolor="white")

    if include_he:
        for record in he_records:
            points = record["points"]
            if not len(points):
                continue
            z = np.full(len(points), (float(record["z_um"]) - z_min) * z_stretch)
            ax.scatter(
                points[:, 0], points[:, 1], z,
                s=0.58, c=[cmap(norm(float(record["z_um"])))], alpha=0.34,
                depthshade=False, linewidths=0, rasterized=True,
            )

    handles: list[Any] = []
    anchor_ids = sorted(int(value) for value in np.unique(st_sections))
    for sid in anchor_ids:
        rows = st_sections == sid
        points = st_xy[rows]
        z_um = float(z_by_section[sid])
        color = cmap(norm(z_um))
        z = np.full(len(points), (z_um - z_min) * z_stretch)
        ax.scatter(
            points[:, 0], points[:, 1], z,
            s=0.28 if include_he else 0.40, c=[color], alpha=0.92,
            depthshade=False, linewidths=0, rasterized=True,
        )
        handles.append(
            Line2D(
                [0], [0], marker="o", color="none", markerfacecolor=color,
                markeredgecolor=color, markersize=7, label=f"ST anchor {sid}",
            )
        )

    xy_values = [record["points"] for record in he_records if len(record["points"])] if include_he else []
    if xy_values:
        x_values = np.concatenate([points[:, 0] for points in xy_values] + [st_xy[:, 0]])
        y_values = np.concatenate([points[:, 1] for points in xy_values] + [st_xy[:, 1]])
    else:
        x_values, y_values = st_xy[:, 0], st_xy[:, 1]
    x_span = max(float(x_values.max() - x_values.min()), 1.0)
    y_span = max(float(y_values.max() - y_values.min()), 1.0)
    z_span_display = max((z_max - z_min) * z_stretch, 1.0)

    ax.set_xlabel("global X (µm)", labelpad=10, fontsize=12)
    ax.set_ylabel("global Y (µm)", labelpad=10, fontsize=12)
    ax.set_zlabel(f"Z display (physical µm × {z_stretch:g})", labelpad=10, fontsize=12)
    ax.set_title(
        f"9957/g0 final v3 3-D alignment{title_suffix}\n"
        "colored H&E boundaries = all 81 registered sections; colored points = ST anchors\n"
        f"frame: {frame} | physical Z range: {z_min:.1f}–{z_max:.1f} µm",
        pad=20,
        fontsize=15,
    )
    ax.view_init(elev=25, azim=-58)
    ax.set_proj_type("ortho")
    ax.set_box_aspect((x_span, y_span, z_span_display))
    ax.grid(False)
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9, title="ST sections")
    fig.subplots_adjust(left=0.02, right=0.82, bottom=0.04, top=0.88)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return {
        "output": str(output.resolve()),
        "include_he_support": include_he,
        "he_section_count": len(he_records),
        "st_anchor_sections": anchor_ids,
        "st_cell_count": int(len(st_xy)),
        "z_display_stretch": float(z_stretch),
        "physical_z_min_um": z_min,
        "physical_z_max_um": z_max,
        "coordinate_frame": frame,
    }


def render(
    he_table: Path = HE_TABLE,
    feature_store: Path = FEATURE_STORE,
    h5ad: Path = H5AD,
    output: Path = OUTPUT,
    *,
    he_points_per_section: int = 1800,
    z_stretch: float = 5.0,
) -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite 3-D QC directory: {output}")
    if he_points_per_section <= 0 or z_stretch <= 0:
        raise ValueError("point limit and z_stretch must be positive")
    he_records, he_frame = load_he_records(feature_store.resolve(), he_table.resolve(), he_points_per_section)
    st_xy, st_sections, z_by_section, st_frame = load_st(h5ad.resolve(), he_table.resolve())
    frame = validate_coordinate_frames(he_frame, st_frame)
    output.mkdir(parents=True, exist_ok=False)
    outputs = [
        _draw(
            he_records, st_xy, st_sections, z_by_section, frame,
            output / "9957_g0__final_v3__all_81_he_sections_plus_st_anchors_3d.png",
            include_he=True, z_stretch=z_stretch,
            title_suffix=" — all H&E sections + ST anchors",
        ),
        _draw(
            he_records, st_xy, st_sections, z_by_section, frame,
            output / "9957_g0__final_v3__he_sections_only_3d.png",
            include_he=True, z_stretch=z_stretch,
            title_suffix=" — all H&E sections only",
        ),
        _draw(
            he_records, st_xy, st_sections, z_by_section, frame,
            output / "9957_g0__final_v3__st_anchors_only_3d.png",
            include_he=False, z_stretch=z_stretch,
            title_suffix=" — ST anchors only",
        ),
    ]
    manifest = {
        "status": "complete",
        "coordinate_frame": frame,
        "h5ad": str(h5ad.resolve()),
        "h5ad_sha256": sha256(h5ad.resolve()),
        "he_table": str(he_table.resolve()),
        "he_table_sha256": sha256(he_table.resolve()),
        "feature_store": str(feature_store.resolve()),
        "feature_store_sha256": sha256(feature_store.resolve()),
        "h_e_is_fixed": True,
        "he_section_count": len(he_records),
        "he_section_ids": [int(record["section_id"]) for record in he_records],
        "removed_he_section_ids": [54],
        "st_anchor_section_ids": sorted(int(value) for value in np.unique(st_sections)),
        "st_cell_count": int(len(st_xy)),
        "he_points_per_section": int(he_points_per_section),
        "z_display_stretch": float(z_stretch),
        "semantics": (
            "H&E is represented by registered UNI2-valid support boundaries; ST uses "
            "obsm['spatial_st_residual_v3']; both are in the same final global XY frame."
        ),
        "figures": outputs,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--he-table", type=Path, default=HE_TABLE)
    parser.add_argument("--feature-store", type=Path, default=FEATURE_STORE)
    parser.add_argument("--h5ad", type=Path, default=H5AD)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--he-points-per-section", type=int, default=1800)
    parser.add_argument("--z-stretch", type=float, default=5.0)
    args = parser.parse_args()
    print(json.dumps(render(**vars(args)), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
