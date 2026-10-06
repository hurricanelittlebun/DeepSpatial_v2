"""Merge selected normalized tissue/g H5AD files into one AnnData object."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import anndata as ad
import numpy as np


def _validate_inputs(paths: list[Path]) -> list[ad.AnnData]:
    if not paths:
        raise ValueError("No input H5AD files were supplied")
    objects: list[ad.AnnData] = []
    var_names = None
    obs_columns = None
    obsm_keys = None
    all_obs_names: set[str] = set()
    for path in paths:
        value = ad.read_h5ad(path)
        if value.obs_names.has_duplicates:
            raise ValueError(f"Duplicate obs_names within {path}")
        overlap = all_obs_names.intersection(value.obs_names.astype(str))
        if overlap:
            raise ValueError(f"obs_names overlap across inputs; example: {next(iter(overlap))}")
        all_obs_names.update(value.obs_names.astype(str))
        current_var = tuple(value.var_names.astype(str))
        current_obs = tuple(value.obs.columns)
        current_obsm = tuple(value.obsm.keys())
        if var_names is None:
            var_names, obs_columns, obsm_keys = current_var, current_obs, current_obsm
        else:
            if current_var != var_names:
                raise ValueError(f"Gene order differs in {path}")
            if current_obs != obs_columns:
                raise ValueError(f"obs schema differs in {path}")
            if current_obsm != obsm_keys:
                raise ValueError(f"obsm schema differs in {path}")
        for key in current_obsm:
            array = np.asarray(value.obsm[key])
            if array.shape != (value.n_obs, 2):
                raise ValueError(f"obsm[{key!r}] has unexpected shape in {path}: {array.shape}")
        objects.append(value)
    return objects


def merge_group_h5ads(
    input_paths: Iterable[Path | str],
    output_path: Path | str,
    *,
    compression: str | None = "gzip",
    overwrite: bool = False,
) -> dict[str, object]:
    paths = [Path(path).resolve() for path in input_paths]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    target = Path(output_path).resolve()
    if target.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)

    objects = _validate_inputs(paths)
    try:
        merged = ad.concat(
            objects,
            axis=0,
            # The input validator above already requires identical var and
            # obs/obsm schemas.  ``inner`` keeps this compatible with older
            # AnnData releases that do not implement ``join="exact"``.
            join="inner",
            merge="same",
            uns_merge=None,
            index_unique=None,
        )
        if merged.obs_names.has_duplicates:
            raise ValueError("Merged obs_names are not unique")
        if getattr(merged.X, "format", None) != "csr":
            merged.X = merged.X.tocsr()

        records = []
        for path, value in zip(paths, objects):
            series = value.obs["series_id"].astype(str).unique().tolist()
            if len(series) != 1:
                raise ValueError(f"Input {path} must contain exactly one series_id")
            records.append(
                {
                    "series_id": series[0],
                    "source": str(path),
                    "n_obs": int(value.n_obs),
                    "n_vars": int(value.n_vars),
                    "sections": [
                        int(x)
                        for x in sorted(value.obs["section_id"].astype(int).unique().tolist())
                    ],
                }
            )
        merged.uns["deepspatial_merged_groups"] = {
            "format": "deepspatial-normalized-groups-merged-v1",
            "excluded_series": ["00029__g0"],
            "coordinate_units": "micrometer",
            "spatial_key": "spatial",
            "registered_spatial_key": "spatial_registered",
            "counts_semantics": "raw Xenium cell-by-gene counts",
            "group_count": len(records),
            "n_obs": int(merged.n_obs),
            "n_vars": int(merged.n_vars),
            # Keep the H5AD ``uns`` value compatible with older AnnData/HDF5
            # writers.  The full structured records are also written to the
            # adjacent JSON manifest.
            "series_ids": [record["series_id"] for record in records],
            "groups_json": json.dumps(records, ensure_ascii=False),
        }

        temporary = target.with_name(target.name + ".tmp")
        if temporary.exists():
            temporary.unlink()
        merged.write_h5ad(temporary, compression=compression)
        os.replace(temporary, target)
        manifest = {
            "format": "deepspatial-normalized-groups-merged-v1",
            "output": str(target),
            "excluded_series": ["00029__g0"],
            "n_obs": int(merged.n_obs),
            "n_vars": int(merged.n_vars),
            "group_count": len(records),
            "groups": records,
            "size_bytes": int(target.stat().st_size),
        }
        manifest_path = target.with_name(target.stem + ".manifest.json")
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        return manifest
    finally:
        for value in objects:
            if value.isbacked:
                value.file.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude", action="append", default=["00029_g0.h5ad"])
    parser.add_argument("--compression", choices=("gzip", "lzf", "none"), default="gzip")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    excluded = set(args.exclude)
    paths = sorted(path for path in args.input_dir.glob("*.h5ad") if path.name not in excluded)
    compression = None if args.compression == "none" else args.compression
    manifest = merge_group_h5ads(
        paths,
        args.output,
        compression=compression,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
