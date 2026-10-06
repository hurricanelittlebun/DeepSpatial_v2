"""Render a static 3-D PNG for the 9957/g0 v10 H&E/ST coordinate frame."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
GROUP = ROOT / "data" / "9957_g0"
PREP = GROUP / "reconstruction_prep_v10_feature_support_residual"
REG = GROUP / "registration_correction_he_v10_feature_support_residual"
STORE_PATH = PREP / "uni2_features_feature_support_residual_v10.h5"
H5AD_PATH = REG / "9957_g0__st_to_feature_support_residual_v10.h5ad"
OUTPUT = GROUP / "qc" / "he_alignment_all_sections_v7_feature_support_residual" / "9957_g0_v10_he_support_st_3d.png"


def _feature_store_class():
    package_name = "_feature_store_3d_v10"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / "deepspatial" / "histology")]
    sys.modules[package_name] = package
    path = ROOT / "deepspatial" / "histology" / "feature_store.py"
    spec = importlib.util.spec_from_file_location(f"{package_name}.feature_store", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FeatureStore


def _boundary(mask: np.ndarray, origin: list[float], spacing: list[float], max_points: int) -> np.ndarray:
    from scipy.ndimage import binary_erosion

    edge = np.asarray(mask, dtype=bool)
    edge = edge & ~binary_erosion(edge, structure=np.ones((3, 3), bool), border_value=0)
    rows, cols = np.nonzero(edge)
    if len(rows) > max_points:
        keep = np.linspace(0, len(rows) - 1, max_points, dtype=np.int64)
        rows, cols = rows[keep], cols[keep]
    return np.column_stack([
        float(origin[0]) + cols * float(spacing[0]),
        float(origin[1]) + rows * float(spacing[1]),
    ]) if len(rows) else np.empty((0, 2), dtype=float)


def render() -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    FeatureStore = _feature_store_class()
    store = FeatureStore(STORE_PATH, cache_mb=512)
    he_records = []
    with h5py.File(STORE_PATH, "r") as handle:
        for sid_text in store.sections:
            metadata = store.metadata[sid_text]
            valid = np.asarray(handle["sections"][store._key(sid_text)]["valid"][:], dtype=bool)
            points = _boundary(valid, metadata["origin_um"], metadata["spacing_um"], 1400)
            he_records.append({"section_id": int(sid_text), "z_um": float(metadata["z_um"]), "points": points})
    he_records.sort(key=lambda item: item["z_um"])

    adata = ad.read_h5ad(H5AD_PATH, backed="r")
    try:
        st_xy = np.asarray(adata.obsm["spatial_registered"], dtype=float)
        section_ids = pd.to_numeric(adata.obs["section_id"], errors="raise").astype(int).to_numpy()
        table = pd.read_parquet(PREP / "he_sections.parquet")
        z_by_section = {int(row.section_id): float(row.z_um) for row in table.itertuples(index=False)}
    finally:
        adata.file.close()

    z_min = min(item["z_um"] for item in he_records)
    z_max = max(item["z_um"] for item in he_records)
    z_stretch = 5.0
    fig = plt.figure(figsize=(17, 13), dpi=220)
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection="3d", facecolor="white")

    # All H&E support is black: this is the continuous registered morphology.
    for item in he_records:
        points = item["points"]
        z = np.full(len(points), (item["z_um"] - z_min) * z_stretch)
        ax.scatter(points[:, 0], points[:, 1], z, s=0.55, c="#111111", alpha=0.95, depthshade=False, linewidths=0, rasterized=True)

    colors = plt.get_cmap("turbo")(np.linspace(0.03, 0.97, len(sorted(z_by_section))))
    handles = []
    for color, sid in zip(colors, sorted(z_by_section)):
        rows = section_ids == sid
        if not rows.any():
            continue
        xy = st_xy[rows]
        z = np.full(len(xy), (z_by_section[sid] - z_min) * z_stretch)
        ax.scatter(xy[:, 0], xy[:, 1], z, s=0.25, c=[color], alpha=0.9, depthshade=False, linewidths=0, rasterized=True)
        handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor=color, markersize=7, label=f"ST {sid}"))

    ax.set_xlabel("global X (µm)", labelpad=10, fontsize=12)
    ax.set_ylabel("global Y (µm)", labelpad=10, fontsize=12)
    ax.set_zlabel("Z (µm; display stretched ×5)", labelpad=10, fontsize=12)
    ax.set_title(
        "9957/g0 v10 registered 3-D coordinate QC\n"
        "black = all registered H&E/UNI2 support; colored = ST anchors\n"
        "feature-support residual applied from section 55",
        pad=20,
        fontsize=15,
    )
    ax.view_init(elev=25, azim=-58)
    ax.set_proj_type("ortho")
    ax.set_box_aspect((1.0, 1.0, max(0.45, z_stretch * max(z_max - z_min, 1.0) / 4000.0)))
    ax.grid(False)
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9, title="ST anchor sections")
    fig.subplots_adjust(left=0.02, right=0.82, bottom=0.04, top=0.88)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    fig.savefig(OUTPUT, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return OUTPUT


if __name__ == "__main__":
    print(render())
