"""Persistent sparse UOT coupling caches and lightweight QC summaries."""

import json
from pathlib import Path

import numpy as np

from .uot_solver import SparseCoupling


def save_sparse_coupling(path, coupling, metadata=None):
    """Atomically save an edge-list coupling as a compressed NPZ file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    payload = {
        "row": coupling.row,
        "col": coupling.col,
        "mass": coupling.mass,
        "shape": np.asarray(coupling.shape, dtype=np.int64),
        "metadata": np.asarray(json.dumps(metadata or {}, sort_keys=True, default=str)),
    }
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)


def load_sparse_coupling(path):
    """Load a coupling cache and its JSON metadata."""
    with np.load(path, allow_pickle=False) as data:
        coupling = SparseCoupling(
            row=data["row"],
            col=data["col"],
            mass=data["mass"],
            shape=tuple(int(x) for x in data["shape"]),
        )
        metadata = json.loads(str(data["metadata"].item()))
    return coupling, metadata


def _weighted_quantile(values, weights, quantile):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if len(values) == 0 or weights.sum() <= 0:
        return float("nan")
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    target = float(quantile) * cumulative[-1]
    index = np.searchsorted(cumulative, target, side="left")
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def summarize_sparse_coupling(coupling, source_xy, target_xy):
    """Return JSON-safe support, mass and distance statistics for QC."""
    source_xy = np.asarray(source_xy, dtype=np.float64)
    target_xy = np.asarray(target_xy, dtype=np.float64)
    if source_xy.shape != (coupling.shape[0], 2) or target_xy.shape != (
        coupling.shape[1],
        2,
    ):
        raise ValueError("XY arrays do not match coupling shape")

    edge_distances = np.linalg.norm(
        source_xy[coupling.row] - target_xy[coupling.col], axis=1
    )
    total_mass = coupling.total_mass
    row_mass = np.bincount(
        coupling.row, weights=coupling.mass, minlength=coupling.shape[0]
    )
    col_mass = np.bincount(
        coupling.col, weights=coupling.mass, minlength=coupling.shape[1]
    )
    dense_pair_count = int(coupling.shape[0]) * int(coupling.shape[1])
    return {
        "source_count": int(coupling.shape[0]),
        "target_count": int(coupling.shape[1]),
        "edge_count": int(len(coupling)),
        "dense_pair_count": dense_pair_count,
        "edge_fraction_of_dense": float(len(coupling) / dense_pair_count),
        "dense_to_sparse_reduction": float(dense_pair_count / max(len(coupling), 1)),
        "total_mass": float(total_mass),
        "source_support_fraction": float(np.mean(row_mass > 0)),
        "target_support_fraction": float(np.mean(col_mass > 0)),
        "positive_mass_edge_fraction": float(np.mean(coupling.mass > 0)),
        "weighted_distance_mean_um": float(
            np.dot(coupling.mass, edge_distances) / total_mass
            if total_mass > 0
            else float("nan")
        ),
        "weighted_distance_p50_um": _weighted_quantile(
            edge_distances, coupling.mass, 0.50
        ),
        "weighted_distance_p95_um": _weighted_quantile(
            edge_distances, coupling.mass, 0.95
        ),
        "unweighted_distance_p95_um": float(np.quantile(edge_distances, 0.95)),
    }
