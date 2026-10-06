"""Render filled, fixed-frame feature-support overlays for 9957/g0 v10."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / "data" / "9957_g0" / "reconstruction_prep_v10_feature_support_residual"
STORE_PATH = PREP / "uni2_features_feature_support_residual_v10.h5"
OUTPUT_DIR = ROOT / "data" / "9957_g0" / "qc" / "he_alignment_all_sections_v7_feature_support_residual" / "diagnostics"


def _load_feature_store():
    package_name = "_feature_store_pair_overlay"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / "deepspatial" / "histology")]
    sys.modules[package_name] = package
    path = ROOT / "deepspatial" / "histology" / "feature_store.py"
    spec = importlib.util.spec_from_file_location(f"{package_name}.feature_store", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FeatureStore


def _load_mask(store, sid: int):
    metadata = store.metadata[str(sid)]
    with h5py.File(store.path, "r") as handle:
        mask = np.asarray(handle["sections"][store._key(str(sid))]["valid"][:], dtype=bool)
    return mask, np.asarray(metadata["origin_um"], dtype=float), np.asarray(metadata["spacing_um"], dtype=float)


def _canvas(sections: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]):
    spacing = float(sections[next(iter(sections))][2][0])
    points = []
    for mask, origin, _ in sections.values():
        rows, cols = np.nonzero(mask)
        points.append(np.column_stack([origin[0] + cols * spacing, origin[1] + rows * spacing]))
    merged = np.concatenate(points, axis=0)
    origin = np.floor(merged.min(axis=0) / spacing) * spacing
    maximum = np.ceil(merged.max(axis=0) / spacing) * spacing
    width = int(round((maximum[0] - origin[0]) / spacing)) + 1
    height = int(round((maximum[1] - origin[1]) / spacing)) + 1
    canvas = {}
    for sid, (mask, section_origin, _) in sections.items():
        out = np.zeros((height, width), dtype=bool)
        rows, cols = np.nonzero(mask)
        x = section_origin[0] + cols * spacing
        y = section_origin[1] + rows * spacing
        out[np.rint((y - origin[1]) / spacing).astype(int), np.rint((x - origin[0]) / spacing).astype(int)] = True
        canvas[sid] = out
    return canvas, origin, spacing


def _render(sids: tuple[int, ...], name: str, colors: dict[int, tuple[float, float, float]], title: str) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    FeatureStore = _load_feature_store()
    store = FeatureStore(STORE_PATH, cache_mb=512)
    sections = {sid: _load_mask(store, sid) for sid in sids}
    masks, origin, spacing = _canvas(sections)
    height, width = next(iter(masks.values())).shape
    rgb = np.ones((height, width, 3), dtype=np.float32)
    occupied = np.zeros((height, width), dtype=bool)
    for sid in sids:
        mask = masks[sid]
        color = np.asarray(colors[sid], dtype=np.float32)
        overlap = occupied & mask
        rgb[mask] = color
        if overlap.any():
            rgb[overlap] = np.array([1.0, 0.82, 0.20], dtype=np.float32)
        occupied |= mask
    fig, ax = plt.subplots(figsize=(10, 8), dpi=220, constrained_layout=True)
    extent = [origin[0] - spacing / 2, origin[0] + (width - 0.5) * spacing, origin[1] + (height - 0.5) * spacing, origin[1] - spacing / 2]
    ax.imshow(rgb, extent=extent, interpolation="none")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("global X (µm)")
    ax.set_ylabel("global Y (µm)")
    ax.set_title(title)
    path = OUTPUT_DIR / name
    if path.exists():
        raise FileExistsError(path)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        _render(
            (53, 55),
            "pair-053-055__filled_global_overlay_v10.png",
            {53: (0.95, 0.05, 0.05), 55: (0.05, 0.80, 0.15)},
            "9957/g0 feature-support overlay | red=53, green=55, yellow=overlap | v10",
        ),
        _render(
            (51, 53, 55),
            "sections-051-053-055__filled_global_overlay_v10.png",
            {51: (0.05, 0.25, 0.90), 53: (0.95, 0.05, 0.05), 55: (0.05, 0.80, 0.15)},
            "9957/g0 feature-support overlay | blue=51, red=53, green=55 | v10",
        ),
    ]
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
