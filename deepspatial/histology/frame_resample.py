"""Offline affine resampling of physical morphology feature fields.

The source stores contain grids in one registered physical XY frame.  This
module materializes a new store in another frame without touching the source:

    p_output = M_source_to_output @ p_source

Output grid centers are mapped back with ``M^-1`` and queried using the
existing mask-aware bilinear interpolation.  Nearest-valid recovery is
disabled during resampling so the output validity mask describes the actual
source support rather than a nearby point.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from .feature_store import FeatureStore


_IDENTITY = np.eye(3, dtype=np.float64)


def _as_affine(matrix) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (3, 3) or not np.isfinite(value).all():
        raise ValueError("Affine matrix must be finite with shape (3,3)")
    if not np.allclose(value[2], [0.0, 0.0, 1.0], atol=1e-8, rtol=0.0):
        raise ValueError("Affine matrix must have homogeneous last row [0,0,1]")
    if abs(float(np.linalg.det(value[:2, :2]))) <= 1e-12:
        raise ValueError("Affine matrix must be invertible")
    return value


def transform_points(points, matrix) -> np.ndarray:
    """Apply a 3x3 physical XY affine to an ``[N,2]`` array."""

    value = np.asarray(points, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 2 or not np.isfinite(value).all():
        raise ValueError("points must be finite with shape [N,2]")
    affine = _as_affine(matrix)
    homogeneous = np.concatenate(
        [value, np.ones((len(value), 1), dtype=np.float64)], axis=1
    )
    transformed = homogeneous @ affine.T
    denominator = transformed[:, 2:3]
    if not np.allclose(denominator, 1.0, atol=1e-8, rtol=0.0):
        raise ValueError("Affine transformation produced non-unit homogeneous scale")
    return transformed[:, :2]


def _grid_centers(metadata: dict) -> np.ndarray:
    height, width, _ = metadata["shape"]
    origin = np.asarray(metadata["origin_um"], dtype=np.float64)
    spacing = np.asarray(metadata["spacing_um"], dtype=np.float64)
    yy, xx = np.mgrid[:height, :width]
    return np.stack(
        [origin[0] + xx.reshape(-1) * spacing[0],
         origin[1] + yy.reshape(-1) * spacing[1]],
        axis=1,
    )


def _output_grid(metadata: dict, affine: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """Return output origin and ``(height, width)`` for transformed centers."""

    height, width, _ = metadata["shape"]
    origin = np.asarray(metadata["origin_um"], dtype=np.float64)
    spacing = np.asarray(metadata["spacing_um"], dtype=np.float64)
    if np.allclose(affine, _IDENTITY, atol=1e-12, rtol=0.0):
        return origin, (int(height), int(width))

    corners = np.array(
        [
            [origin[0], origin[1]],
            [origin[0] + (width - 1) * spacing[0], origin[1]],
            [origin[0], origin[1] + (height - 1) * spacing[1]],
            [
                origin[0] + (width - 1) * spacing[0],
                origin[1] + (height - 1) * spacing[1],
            ],
        ],
        dtype=np.float64,
    )
    transformed = transform_points(corners, affine)
    low = np.floor(transformed.min(axis=0) / spacing) * spacing
    high = np.ceil(transformed.max(axis=0) / spacing) * spacing
    dimensions = np.rint((high - low) / spacing).astype(np.int64) + 1
    if (dimensions < 1).any() or int(np.prod(dimensions)) > 100_000_000:
        raise ValueError("Transformed feature grid has an invalid or excessive size")
    return low, (int(dimensions[1]), int(dimensions[0]))


def _make_grid(origin: np.ndarray, spacing: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    yy, xx = np.mgrid[:height, :width]
    return np.stack(
        [origin[0] + xx.reshape(-1) * spacing[0],
         origin[1] + yy.reshape(-1) * spacing[1]],
        axis=1,
    )


def _read_section_arrays(source: FeatureStore, section_id: str):
    with h5py.File(source.path, "r") as handle:
        group = handle["sections"][source._key(section_id)]
        return group["features"][:], group["valid"][:]


def _section_provenance(
    base: dict,
    *,
    source: FeatureStore,
    section_id: str,
    output_frame: str,
    affine: np.ndarray,
    output_origin: np.ndarray,
    output_shape: tuple[int, int],
):
    metadata = source.metadata[section_id]
    value = dict(base)
    value["frame_resample"] = {
        "source_frame": metadata["coordinate_frame"],
        "output_frame": str(output_frame),
        "direction": "p_output = matrix_source_to_output @ p_source",
        "matrix_source_to_output": affine.tolist(),
        "matrix_sha256": hashlib.sha256(affine.astype("<f8").tobytes()).hexdigest(),
        "interpolation": "mask_aware_bilinear",
        "nearest_valid_recovery": False,
        "source_origin_um": list(map(float, metadata["origin_um"])),
        "source_shape": list(map(int, metadata["shape"][:2])),
        "output_origin_um": list(map(float, output_origin)),
        "output_shape": list(map(int, output_shape)),
    }
    return value


def resample_feature_store(
    source_path,
    output_path,
    affine_by_section,
    *,
    output_frame: str,
    provenance: dict | None = None,
) -> dict:
    """Materialize a source feature store in a new affine output frame.

    ``affine_by_section`` maps every source section ID to a matrix that maps
    source physical coordinates to output physical coordinates.  The output
    path must not already exist.
    """

    source_path = Path(source_path).resolve()
    output_path = Path(output_path).resolve()
    if source_path == output_path:
        raise ValueError("Source and output feature stores must be different")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing feature store: {output_path}"
        )
    if not output_frame:
        raise ValueError("output_frame must be non-empty")

    source = FeatureStore(source_path)
    matrices = {str(key): _as_affine(value) for key, value in affine_by_section.items()}
    missing = sorted(set(source.sections) - set(matrices))
    if missing:
        raise KeyError(f"Missing affine for source sections: {missing}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = FeatureStore(output_path, mode="a")
    sections = []
    for section_id in source.sections:
        metadata = source.metadata[section_id]
        affine = matrices[section_id]
        output_origin, output_shape = _output_grid(metadata, affine)
        spacing = np.asarray(metadata["spacing_um"], dtype=np.float64)
        fixed_points = _make_grid(output_origin, spacing, output_shape)
        if np.allclose(affine, _IDENTITY, atol=1e-12, rtol=0.0):
            features, valid = _read_section_arrays(source, section_id)
        else:
            source_points = transform_points(fixed_points, np.linalg.inv(affine))
            queried, query_valid = source.get_feature(
                section_id,
                source_points,
                return_valid=True,
                nearest_max_distance_um=0.0,
            )
            features = queried.detach().cpu().numpy().astype("float32", copy=False)
            valid = query_valid.detach().cpu().numpy().astype(bool, copy=False)
            features = features.reshape(output_shape[0], output_shape[1], -1)
            valid = valid.reshape(output_shape)

        output.add_section(
            section_id,
            features,
            z_um=float(metadata["z_um"]),
            origin_um=output_origin,
            spacing_um=spacing,
            patch_size_um=float(metadata["patch_size_um"]),
            mpp=np.asarray(metadata["mpp"], dtype=np.float64),
            coordinate_frame=str(output_frame),
            valid_mask=valid,
            provenance=_section_provenance(
                provenance or {},
                source=source,
                section_id=section_id,
                output_frame=output_frame,
                affine=affine,
                output_origin=output_origin,
                output_shape=output_shape,
            ),
        )
        sections.append(
            {
                "section_id": section_id,
                "z_um": float(metadata["z_um"]),
                "source_shape": list(map(int, metadata["shape"])),
                "output_shape": [int(output_shape[0]), int(output_shape[1]), int(features.shape[-1])],
                "source_valid_fraction": float(
                    np.asarray(_read_section_arrays(source, section_id)[1]).mean()
                ) if np.allclose(affine, _IDENTITY, atol=1e-12, rtol=0.0) else None,
                "output_valid_fraction": float(np.asarray(valid).mean()),
                "matrix_source_to_output": affine.tolist(),
                "transform_type": "identity" if np.allclose(affine, _IDENTITY, atol=1e-12, rtol=0.0) else "affine",
            }
        )

    output.refresh()
    return {
        "format": "deepspatial-affine-resampled-feature-store-v1",
        "source_store": str(source_path),
        "source_revision": source.revision,
        "output_store": str(output_path),
        "output_revision": output.revision,
        "source_frame": source.metadata[source.sections[0]]["coordinate_frame"],
        "output_frame": str(output_frame),
        "n_sections": len(sections),
        "sections": sections,
    }


def load_section_affines(
    residual_dir,
    section_ids,
    *,
    identity_sections=(),
) -> dict[str, np.ndarray]:
    """Load residual current-to-fixed matrices and explicit identity sections."""

    residual_dir = Path(residual_dir)
    identity_sections = {str(value) for value in identity_sections}
    result = {}
    for value in section_ids:
        section_id = str(value)
        path = residual_dir / "transforms" / (
            f"section-{int(section_id):03d}__st_to_fixed_he_residual.npz"
        )
        if path.is_file():
            with np.load(path, allow_pickle=False) as payload:
                if "matrix_current_g_to_fixed_he" not in payload:
                    raise KeyError(f"Missing matrix_current_g_to_fixed_he in {path}")
                result[section_id] = _as_affine(payload["matrix_current_g_to_fixed_he"])
        elif section_id in identity_sections:
            result[section_id] = _IDENTITY.copy()
        else:
            raise FileNotFoundError(
                f"No residual transform for section {section_id}; "
                "include it in identity_sections only when identity is intentional"
            )
    return result


__all__ = ["load_section_affines", "resample_feature_store", "transform_points"]
