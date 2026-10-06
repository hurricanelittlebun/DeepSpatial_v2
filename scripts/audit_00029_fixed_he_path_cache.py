"""Audit a morphology path cache without changing it."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np


def _text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def _json_attribute(group, name, default):
    value = group.attrs.get(name)
    if value is None:
        return default
    return json.loads(_text(value))


def audit_path_cache(path, expected_frame=None) -> dict:
    """Return path, frame, fallback and endpoint-integrity statistics."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        root_frame = handle.attrs.get("coordinate_frame")
        root_frame = None if root_frame is None else _text(root_frame)
        if expected_frame is not None and root_frame != str(expected_frame):
            raise ValueError(
                "Path cache coordinate frame does not match expected coordinate frame: "
                f"cache={root_frame!r}, expected={expected_frame!r}"
            )
        if "paths" not in handle:
            raise ValueError(f"Path cache has no paths group: {path}")

        path_count = 0
        morphology_fallback_count = 0
        nucleus_fallback_count = 0
        endpoint_position_errors = []
        endpoint_t_errors = []
        pair_counts = Counter()
        metadata_frames = set()
        versions = Counter()
        for key, group in handle["paths"].items():
            path_count += 1
            metadata = json.loads(_text(group.attrs["metadata"]))
            frame = metadata.get("coordinate_frame")
            if frame is not None:
                metadata_frames.add(str(frame))
            if expected_frame is not None and frame != str(expected_frame):
                raise ValueError(
                    f"Path {key} coordinate frame does not match expected frame: {frame!r}"
                )
            versions[str(metadata.get("version"))] += 1
            pair_counts[(str(metadata["s0"]), str(metadata["s1"]))] += 1
            if _json_attribute(group, "morphology_fallback_sections", []):
                morphology_fallback_count += 1
            if _json_attribute(group, "nucleus_fallback_sections", []):
                nucleus_fallback_count += 1

            t = np.asarray(group["t"][:], dtype=np.float64)
            xy = np.asarray(group["xy_um"][:], dtype=np.float64)
            coefficients = np.asarray(group["coefficients"][:], dtype=np.float64)
            if (
                t.ndim != 1
                or xy.shape != (len(t), 2)
                or coefficients.ndim != 3
                or coefficients.shape[0] != 4
                or coefficients.shape[1] != len(t) - 1
                or coefficients.shape[2] != 2
                or not np.isfinite(t).all()
                or not np.isfinite(xy).all()
                or not np.isfinite(coefficients).all()
            ):
                raise ValueError(f"Invalid finite path arrays for {key}")
            endpoint_position_errors.append(
                max(
                    float(np.linalg.norm(xy[0] - np.asarray(metadata["x0"]))),
                    float(np.linalg.norm(xy[-1] - np.asarray(metadata["x1"]))),
                )
            )
            endpoint_t_errors.append(
                max(abs(float(t[0])), abs(float(t[-1]) - 1.0))
            )

    pair_rows = [
        {
            "source_section": source,
            "target_section": target,
            "path_count": int(count),
        }
        for (source, target), count in sorted(pair_counts.items())
    ]
    return {
        "format": "deepspatial-path-cache-audit-v1",
        "path_cache": str(path.resolve()),
        "coordinate_frame": root_frame,
        "metadata_coordinate_frames": sorted(metadata_frames),
        "path_count": int(path_count),
        "morphology_fallback_path_count": int(morphology_fallback_count),
        "morphology_fallback_path_fraction": float(
            morphology_fallback_count / max(path_count, 1)
        ),
        "nucleus_fallback_path_count": int(nucleus_fallback_count),
        "nucleus_fallback_path_fraction": float(
            nucleus_fallback_count / max(path_count, 1)
        ),
        "endpoint_max_position_error_um": float(
            max(endpoint_position_errors, default=0.0)
        ),
        "endpoint_max_t_error": float(max(endpoint_t_errors, default=0.0)),
        "versions": dict(versions),
        "pairs": pair_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path-cache",
        type=Path,
        default=Path("data/00029_g0/train_full_00029_st_corrected_fixed_he_v1/paths.h5"),
    )
    parser.add_argument("--expected-frame", default="00029__g0_fixed_he")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = audit_path_cache(args.path_cache, expected_frame=args.expected_frame)
    output = args.output or args.path_cache.with_name("path_cache_audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
