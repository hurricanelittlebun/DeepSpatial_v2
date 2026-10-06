#!/usr/bin/env python3
"""Fit non-destructive residual ST-to-fixed-H&E affine corrections.

The existing H&E registration is treated as fixed.  The saved STalign maps
are evaluated exactly with NumPy (including the velocity field), so this
script does not replace the nonlinear registration by an affine approximation.

The fitted residual matrix R is defined as H&E-fixed-frame -> current-g-frame.
The corrected ST coordinates are therefore R^{-1} @ spatial_registered.
Original coordinates and all original registration artifacts are preserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.optimize import minimize


ANCHORS = (1, 11, 21, 31, 41, 51, 61, 71, 81, 91)
PRIORITY = (71, 81, 91)


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def apply_homogeneous(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=float)))
    return (homogeneous @ np.asarray(matrix, dtype=float).T)[:, :2]


def bilinear_field(field: np.ndarray, axes: tuple[np.ndarray, np.ndarray], points: np.ndarray) -> np.ndarray:
    """Bilinear interpolation matching STalign's align_corners convention."""

    field = np.asarray(field, dtype=float)
    points = np.asarray(points, dtype=float)
    row_axis, col_axis = axes
    result = np.zeros((len(points), field.shape[-1]), dtype=float)
    if len(points) == 0:
        return result
    valid = (
        np.isfinite(points).all(axis=1)
        & (points[:, 0] >= row_axis[0])
        & (points[:, 0] <= row_axis[-1])
        & (points[:, 1] >= col_axis[0])
        & (points[:, 1] <= col_axis[-1])
    )
    if not valid.any():
        return result
    indices = np.flatnonzero(valid)
    rows = (points[indices, 0] - row_axis[0]) / (row_axis[-1] - row_axis[0]) * (len(row_axis) - 1)
    cols = (points[indices, 1] - col_axis[0]) / (col_axis[-1] - col_axis[0]) * (len(col_axis) - 1)
    r0 = np.floor(rows).astype(int)
    c0 = np.floor(cols).astype(int)
    r1 = np.minimum(r0 + 1, len(row_axis) - 1)
    c1 = np.minimum(c0 + 1, len(col_axis) - 1)
    wr = rows - r0
    wc = cols - c0
    result[indices] = (
        (1.0 - wr)[:, None] * (1.0 - wc)[:, None] * field[r0, c0]
        + (1.0 - wr)[:, None] * wc[:, None] * field[r0, c1]
        + wr[:, None] * (1.0 - wc)[:, None] * field[r1, c0]
        + wr[:, None] * wc[:, None] * field[r1, c1]
    )
    return result


@dataclass
class SavedMap:
    affine: np.ndarray
    velocity: np.ndarray
    axes: tuple[np.ndarray, np.ndarray]

    @classmethod
    def load(cls, path: Path) -> "SavedMap":
        with np.load(path, allow_pickle=False) as source:
            return cls(
                affine=np.asarray(source["A"], dtype=float),
                velocity=np.asarray(source["v"], dtype=float),
                axes=(
                    np.asarray(source["xv_0"], dtype=float),
                    np.asarray(source["xv_1"], dtype=float),
                ),
            )

    def forward_rc(self, points_rc: np.ndarray) -> np.ndarray:
        points = np.asarray(points_rc, dtype=float).copy()
        for velocity in self.velocity:
            points += bilinear_field(velocity, self.axes, points) / self.velocity.shape[0]
        return points @ self.affine[:2, :2].T + self.affine[:2, 2]

    def inverse_rc(self, points_rc: np.ndarray) -> np.ndarray:
        inverse = np.linalg.inv(self.affine)
        points = points_rc @ inverse[:2, :2].T + inverse[:2, 2]
        for velocity in self.velocity[::-1]:
            points += bilinear_field(-velocity, self.axes, points) / self.velocity.shape[0]
        return points


class RegistrationMaps:
    def __init__(self, registration_root: Path, sample: str, group: str):
        self.root = registration_root
        self.folder = registration_root / "transforms" / sample / group
        manifest = pd.read_parquet(registration_root / "manifests" / "stalign_pairwise_edges.parquet")
        selected = manifest[
            manifest["sample_id"].astype(str).eq(sample)
            & manifest["group_id"].astype(str).eq(group)
        ]
        self.rows = selected.set_index("edge_key", drop=False)
        self._forward: dict[str, SavedMap] = {}
        self._reverse: dict[str, SavedMap] = {}

    def _load(self, edge_key: str, reverse: bool) -> SavedMap:
        cache = self._reverse if reverse else self._forward
        if edge_key not in cache:
            row = self.rows.loc[str(edge_key)]
            column = "reverse_map_path" if reverse else "forward_map_path"
            cache[str(edge_key)] = SavedMap.load(self.root / str(row[column]))
        return cache[str(edge_key)]

    def apply_edge(self, points_xy: np.ndarray, edge_key: str, direction: str) -> np.ndarray:
        if direction == "forward":
            mapped = self._load(edge_key, reverse=False).forward_rc(np.asarray(points_xy)[:, ::-1])
        elif direction == "reverse":
            mapped = self._load(edge_key, reverse=True).inverse_rc(np.asarray(points_xy)[:, ::-1])
        else:
            raise ValueError(f"unknown edge direction: {direction}")
        return mapped[:, ::-1]

    def apply_inverse_edge(self, points_xy: np.ndarray, edge_key: str, direction: str) -> np.ndarray:
        if direction == "forward":
            mapped = self._load(edge_key, reverse=True).inverse_rc(np.asarray(points_xy)[:, ::-1])
        elif direction == "reverse":
            mapped = self._load(edge_key, reverse=False).forward_rc(np.asarray(points_xy)[:, ::-1])
        else:
            raise ValueError(f"unknown edge direction: {direction}")
        return mapped[:, ::-1]

    def apply_chain(
        self,
        points_xy: np.ndarray,
        chain: Iterable[tuple[str, str]],
        inverse: bool = False,
    ) -> np.ndarray:
        mapped = np.asarray(points_xy, dtype=float)
        chain = tuple(chain)
        if inverse:
            for edge_key, direction in reversed(chain):
                mapped = self.apply_inverse_edge(mapped, edge_key, direction)
        else:
            for edge_key, direction in chain:
                mapped = self.apply_edge(mapped, edge_key, direction)
        return mapped


def read_h5ad_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        local = np.asarray(handle["obsm/spatial_he_local"], dtype=float)
        registered = np.asarray(handle["obsm/spatial_registered"], dtype=float)
        index_name = handle["obs"].attrs.get("_index", "")
        if isinstance(index_name, bytes):
            index_name = index_name.decode()
        index_name = str(index_name)
        if index_name and index_name in handle["obs"]:
            raw_ids = np.asarray(handle["obs"][index_name], dtype=object)
            cell_ids = np.asarray(
                [value.decode() if isinstance(value, bytes) else str(value) for value in raw_ids],
                dtype=str,
            )
        elif "obs/_index" in handle:
            raw_ids = np.asarray(handle["obs/_index"], dtype=object)
            cell_ids = np.asarray(
                [value.decode() if isinstance(value, bytes) else str(value) for value in raw_ids],
                dtype=str,
            )
        else:
            cell_ids = np.arange(len(local)).astype(str)
    if local.shape != registered.shape or local.shape[1] != 2:
        raise ValueError(f"invalid coordinate shapes in {path}")
    return cell_ids, local, registered


def read_section_transform(path: Path) -> tuple[np.ndarray, tuple[tuple[str, str], ...]]:
    with np.load(path, allow_pickle=False) as source:
        preorientation = np.asarray(source["preorientation_matrix"], dtype=float)
        chain = tuple(
            (str(key), str(direction))
            for key, direction in json.loads(str(source["edge_chain_json"].item()))
        )
    return preorientation, chain


def source_points_from_registered(
    registered: np.ndarray,
    preorientation: np.ndarray,
    chain: tuple[tuple[str, str], ...],
    maps: RegistrationMaps,
) -> np.ndarray:
    points = maps.apply_chain(registered, chain, inverse=True)
    return apply_homogeneous(points, np.linalg.inv(preorientation))


def registered_from_source(
    source: np.ndarray,
    preorientation: np.ndarray,
    chain: tuple[tuple[str, str], ...],
    maps: RegistrationMaps,
) -> np.ndarray:
    points = apply_homogeneous(source, preorientation)
    return maps.apply_chain(points, chain, inverse=False)


def sample_mask(mask: np.ndarray, local_xy: np.ndarray, mpp_x: float, mpp_y: float) -> tuple[np.ndarray, np.ndarray]:
    columns = np.rint(local_xy[:, 0] / float(mpp_x)).astype(int)
    rows = np.rint(local_xy[:, 1] / float(mpp_y)).astype(int)
    valid = (
        (rows >= 0)
        & (rows < mask.shape[0])
        & (columns >= 0)
        & (columns < mask.shape[1])
    )
    inside = np.zeros(len(local_xy), dtype=bool)
    inside[valid] = mask[rows[valid], columns[valid]]
    return inside, valid


def sample_signed_distance(
    signed_distance_px: np.ndarray,
    local_xy: np.ndarray,
    mpp_x: float,
    mpp_y: float,
) -> np.ndarray:
    columns = local_xy[:, 0] / float(mpp_x)
    rows = local_xy[:, 1] / float(mpp_y)
    result = map_coordinates(
        signed_distance_px,
        [rows, columns],
        order=1,
        mode="constant",
        cval=-float(np.max(np.abs(signed_distance_px))),
    )
    return result * float((mpp_x + mpp_y) / 2.0)


def residual_matrix(parameters: np.ndarray, center: np.ndarray) -> np.ndarray:
    theta, log_sx, log_sy, tx, ty = np.asarray(parameters, dtype=float)
    cosine, sine = np.cos(theta), np.sin(theta)
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=float)
    linear = rotation @ np.diag([np.exp(log_sx), np.exp(log_sy)])
    matrix = np.eye(3, dtype=float)
    matrix[:2, :2] = linear
    matrix[:2, 2] = center + np.array([tx, ty]) - linear @ center
    return matrix


def residual_score(
    parameters: np.ndarray,
    registered: np.ndarray,
    center: np.ndarray,
    preorientation: np.ndarray,
    chain: tuple[tuple[str, str], ...],
    maps: RegistrationMaps,
    mask: np.ndarray,
    signed_distance_px: np.ndarray,
    mpp_x: float,
    mpp_y: float,
) -> float:
    matrix = residual_matrix(parameters, center)
    corrected = apply_homogeneous(registered, np.linalg.inv(matrix))
    local = source_points_from_registered(corrected, preorientation, chain, maps)
    inside, valid = sample_mask(mask, local, mpp_x, mpp_y)
    signed = sample_signed_distance(signed_distance_px, local, mpp_x, mpp_y)
    if not valid.any():
        return 1e6
    inside_fraction = float(inside[valid].mean())
    morphology_score = float(np.tanh(np.clip(signed[valid] / 50.0, -8.0, 8.0)).mean())
    prior = (
        (parameters[0] / 0.15) ** 2
        + (parameters[1] / 0.15) ** 2
        + (parameters[2] / 0.15) ** 2
        + (parameters[3] / 200.0) ** 2
        + (parameters[4] / 200.0) ** 2
    )
    return -(0.72 * inside_fraction + 0.28 * ((morphology_score + 1.0) / 2.0) - 0.01 * prior)


def fit_section(
    section_id: int,
    root: Path,
    registration_root: Path,
    output_dir: Path,
    max_opt_points: int,
    seed: int,
) -> dict:
    sections = pd.read_parquet(root / "data/00029_g0/he_sections.parquet")
    row = sections.loc[sections["section_id"].eq(int(section_id))].iloc[0]
    h5ad_path = root / "data/00029_g0/anchors" / f"section-{section_id:03d}.h5ad"
    transform_path = registration_root / str(row["section_transform_path"]).replace(
        "/data/buyonggan/DeepSpatial_v2/outputs/registration/", ""
    )
    if not transform_path.exists():
        transform_path = registration_root / "transforms" / "00029" / "g0" / f"section-{section_id}-00029__g0__section-{section_id}.npz"
    _, local_unused, registered = read_h5ad_arrays(h5ad_path)
    preorientation, chain = read_section_transform(transform_path)
    maps = RegistrationMaps(registration_root, "00029", "g0")
    mask_path = Path(str(row["source_mask_path"]))
    mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) > 0
    mpp_x = float(row["analysis_pixel_size_x_um"])
    mpp_y = float(row["analysis_pixel_size_y_um"])
    inside_distance = distance_transform_edt(mask)
    outside_distance = distance_transform_edt(~mask)
    signed_distance = inside_distance - outside_distance

    rng = np.random.default_rng(seed + int(section_id))
    if len(registered) > max_opt_points:
        opt_indices = np.sort(rng.choice(len(registered), size=max_opt_points, replace=False))
    else:
        opt_indices = np.arange(len(registered))
    opt_points = registered[opt_indices]
    center = np.mean(opt_points, axis=0)
    bounds = [(-0.25, 0.25), (-0.15, 0.15), (-0.15, 0.15), (-200.0, 200.0), (-200.0, 200.0)]
    objective = lambda params: residual_score(
        params,
        opt_points,
        center,
        preorientation,
        chain,
        maps,
        mask,
        signed_distance,
        mpp_x,
        mpp_y,
    )
    result = minimize(
        objective,
        np.zeros(5, dtype=float),
        method="Powell",
        bounds=bounds,
        options={"maxiter": 55, "xtol": 1e-3, "ftol": 1e-4, "disp": False},
    )
    identity = np.zeros(5, dtype=float)
    candidate_parameters = np.asarray(result.x, dtype=float)

    def evaluate(parameters: np.ndarray, points: np.ndarray) -> dict:
        matrix = residual_matrix(parameters, center)
        corrected = apply_homogeneous(points, np.linalg.inv(matrix))
        local = source_points_from_registered(corrected, preorientation, chain, maps)
        inside, valid = sample_mask(mask, local, mpp_x, mpp_y)
        signed = sample_signed_distance(signed_distance, local, mpp_x, mpp_y)
        return {
            "inside_fraction": float(inside.mean()),
            "valid_crop_fraction": float(valid.mean()),
            "inside_valid_fraction": float(inside[valid].mean()) if valid.any() else 0.0,
            "median_signed_distance_um": float(np.median(signed[valid])) if valid.any() else float("nan"),
            "corrected_coordinates": corrected,
        }

    baseline = evaluate(identity, registered)
    candidate = evaluate(candidate_parameters, registered)
    improvement = candidate["inside_fraction"] - baseline["inside_fraction"]
    at_bound = any(
        abs(candidate_parameters[index] - lower) < 2e-3
        or abs(candidate_parameters[index] - upper) < 2e-3
        for index, (lower, upper) in enumerate(bounds)
    )
    status = "provisional_residual_affine" if improvement >= 0.05 else "unresolved"
    if at_bound:
        status = "review_required_bound_hit" if improvement >= 0.05 else "unresolved_bound_hit"

    transform_dir = output_dir / "transforms"
    coord_dir = output_dir / "candidate_coordinates"
    overlay_dir = output_dir / "before_after_overlays"
    transform_dir.mkdir(parents=True, exist_ok=True)
    coord_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    matrix = residual_matrix(candidate_parameters, center)
    np.savez_compressed(
        transform_dir / f"section-{section_id:03d}__st_to_fixed_he_residual.npz",
        matrix_he_to_current_g=matrix,
        matrix_current_g_to_fixed_he=np.linalg.inv(matrix),
        parameters=candidate_parameters,
        center=center,
        baseline_inside_fraction=np.asarray([baseline["inside_fraction"]]),
        candidate_inside_fraction=np.asarray([candidate["inside_fraction"]]),
        status=np.asarray(status),
    )
    cell_ids, _, _ = read_h5ad_arrays(h5ad_path)
    np.savez_compressed(
        coord_dir / f"section-{section_id:03d}__st_corrected_candidate.npz",
        cell_id=cell_ids,
        spatial_registered_original=registered,
        spatial_st_corrected=candidate["corrected_coordinates"],
    )

    raster_manifest = pd.read_parquet(registration_root / "manifests/stalign_section_transforms.parquet")
    raster_row = raster_manifest[
        raster_manifest["sample_id"].astype(str).eq("00029")
        & raster_manifest["group_id"].astype(str).eq("g0")
        & raster_manifest["section_id"].eq(int(section_id))
    ].iloc[0]
    raster_path = registration_root / str(raster_row["raster_path"])
    with np.load(raster_path, allow_pickle=False) as raster:
        grid = np.asarray(raster["grid_um"], dtype=float)
        raster_mask = np.asarray(raster["mask"], dtype=bool)
    raster_points = grid[:, raster_mask].T
    raster_registered = maps.apply_chain(raster_points, chain, inverse=False)
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=180)
    for axis, points, title in (
        (axes[0], registered, f"before | coverage={baseline['inside_fraction']:.3f}"),
        (axes[1], candidate["corrected_coordinates"], f"ST corrected | coverage={candidate['inside_fraction']:.3f}"),
    ):
        axis.scatter(raster_registered[:, 0], raster_registered[:, 1], s=0.25, c="#bdbdbd", alpha=0.4, linewidths=0)
        axis.scatter(points[:, 0], points[:, 1], s=0.35, c="#e34a33", alpha=0.35, linewidths=0)
        axis.set_aspect("equal")
        axis.set_title(title)
        axis.set_xlabel("fixed H&E / g x (µm)")
        axis.set_ylabel("fixed H&E / g y (µm)")
    figure.suptitle(f"00029/g0 section {section_id}: fixed H&E, residual ST correction")
    figure.tight_layout()
    figure.savefig(overlay_dir / f"section-{section_id:03d}__before_after.png")
    plt.close(figure)

    return {
        "section_id": int(section_id),
        "n_cells": int(len(registered)),
        "baseline": {key: value for key, value in baseline.items() if key != "corrected_coordinates"},
        "candidate": {key: value for key, value in candidate.items() if key != "corrected_coordinates"},
        "improvement": float(improvement),
        "status": status,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "parameters": candidate_parameters.tolist(),
        "matrix_he_to_current_g": matrix.tolist(),
        "matrix_current_g_to_fixed_he": np.linalg.inv(matrix).tolist(),
        "mask_path": str(mask_path),
        "transform_path": str(transform_path),
        "priority": int(section_id) in PRIORITY,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("/data/buyonggan/DeepSpatial"))
    parser.add_argument("--registration-root", type=Path, default=Path("/data/buyonggan/DeepSpatial_v2/outputs/registration"))
    parser.add_argument("--output-dir", type=Path, default=Path("/data/buyonggan/DeepSpatial/data/00029_g0/registration_correction_v1"))
    parser.add_argument("--sections", type=int, nargs="+", default=list(PRIORITY))
    parser.add_argument("--max-opt-points", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()

    root = args.project_root.resolve()
    registration_root = args.registration_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for section_id in args.sections:
        print(f"fitting section {section_id}", flush=True)
        records.append(
            fit_section(
                int(section_id), root, registration_root, output_dir, int(args.max_opt_points), int(args.seed)
            )
        )
        print(json.dumps(records[-1], ensure_ascii=False), flush=True)

    with (output_dir / "residual_fit_metrics.json").open("w") as handle:
        json.dump({"series_id": "00029__g0", "sections": records}, handle, indent=2, ensure_ascii=False)

    figure, axis = plt.subplots(figsize=(10, 5), dpi=180)
    ids = [record["section_id"] for record in records]
    before = [record["baseline"]["inside_fraction"] for record in records]
    after = [record["candidate"]["inside_fraction"] for record in records]
    positions = np.arange(len(ids))
    width = 0.38
    axis.bar(positions - width / 2, before, width, label="before", color="#e34a33")
    axis.bar(positions + width / 2, after, width, label="candidate ST→H&E", color="#2ca25f")
    axis.set_xticks(positions, [str(value) for value in ids])
    axis.set_ylim(0, 1.0)
    axis.set_xlabel("section")
    axis.set_ylabel("ST cells inside fixed H&E mask")
    axis.set_title("00029/g0 residual ST-to-fixed-H&E candidate")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "before_after_coverage.png")
    plt.close(figure)

    source_files = sorted(
        {
            str(Path(record["mask_path"])): sha256(Path(record["mask_path"]))
            for record in records
        }.items()
    )
    manifest = {
        "status": "candidate_only",
        "coordinate_semantics": {
            "matrix_he_to_current_g": "fixed registered H&E frame to current g frame",
            "matrix_current_g_to_fixed_he": "matrix applied to spatial_registered ST points",
            "output_coordinate_key": "spatial_st_corrected_candidate",
        },
        "original_files_sha256": dict(source_files),
        "sections": [record["section_id"] for record in records],
        "note": "No original H&E masks, STalign maps, H5ADs, feature stores, or path caches were overwritten.",
    }
    (output_dir / "correction_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    verification = {
        "status": "candidate_generated_not_published",
        "original_masks_rehashed_unchanged": all(
            sha256(Path(path)) == digest for path, digest in source_files
        ),
        "sections": [
            {
                "section_id": record["section_id"],
                "baseline_inside_fraction": record["baseline"]["inside_fraction"],
                "candidate_inside_fraction": record["candidate"]["inside_fraction"],
                "improvement": record["improvement"],
                "status": record["status"],
            }
            for record in records
        ],
    }
    (output_dir / "verification_metrics.json").write_text(json.dumps(verification, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
