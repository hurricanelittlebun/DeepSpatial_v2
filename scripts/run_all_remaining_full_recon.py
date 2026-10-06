"""Run the remaining tissue/g full reconstructions in a reproducible order.

This launcher deliberately excludes the completed ``00029/g0`` and
``9589/g1`` outputs.  Each series is processed independently in its own
registered coordinate frame and is trained without a held-out section.
Existing complete stages are reused; incomplete stages are allowed to finish
before the next stage starts.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/data/buyonggan/miniconda3/envs/scellst/bin/python")
CELLPOSE_PYTHON = Path("/data/buyonggan/miniconda3/envs/hespotex_env/bin/python")
PARQUET_SITE_PACKAGES = Path(
    "/data/buyonggan/miniconda3/envs/scellst/lib/python3.10/site-packages"
)
V2_ROOT = Path("/data/buyonggan/DeepSpatial_v2")
UNI2_CHECKPOINT = ROOT / "weights/uni2/pytorch_model.bin"
CELLPOSE_MODEL = Path("/data/buyonggan/.cellpose/models/nucleitorch_0")
GPU = os.environ.get("DEEPSPATIAL_GPU", "5")

SERIES = (
    ("9589__g3", [1, 11, 21, 31, 41]),
    ("9589__g4", [21, 31, 41, 51]),
    ("9625__g0", [1, 11, 21, 31, 41, 51, 61, 71, 81, 91]),
    ("9934__g0", [1, 11, 21, 31, 41, 51, 61, 71, 81, 91]),
    ("9957__g0", [1, 11, 21, 31, 41, 51, 61, 71, 81, 91]),
    ("9985__g1", [1, 11, 21, 31, 41, 51, 61, 71]),
    ("9985__g2", [1, 11, 21, 31]),
)


def _paths(series_id: str):
    tissue, group = series_id.split("__")
    group_dir = ROOT / "data" / f"{tissue}_{group}"
    prep = group_dir / "reconstruction_prep_v1"
    uni2 = prep / "uni2_features.h5"
    nucleus = prep / "nucleus_path_v1"
    uot = prep / "uot_cache_topk64"
    path_cache = prep / "paths.h5"
    final_dir = group_dir / "reconstruction_final_thickness10_v1"
    final = final_dir / (
        f"adata_{series_id.replace('__', '_')}_full_thickness10_celltype_domain_niche.h5ad"
    )
    return group_dir, prep, uni2, nucleus, uot, path_cache, final_dir, final


def _env() -> dict[str, str]:
    value = os.environ.copy()
    value["CUDA_VISIBLE_DEVICES"] = GPU
    # The registered-SDPC reader used by the UNI2 preprocessing example is
    # provided by the sibling DeepSpatial_v2 source tree, not the main repo.
    # Keep both packages visible to every subprocess in the pipeline.
    python_paths = [str(ROOT), str(V2_ROOT / "src")]
    existing = value.get("PYTHONPATH")
    if existing:
        python_paths.append(existing)
    value["PYTHONPATH"] = os.pathsep.join(python_paths)
    sdpc_site = Path(
        os.environ.get(
            "DEEPSPATIAL_SDPC_SITE",
            "/data/buyonggan/miniconda3/envs/spateo_venv/lib/python3.8/site-packages",
        )
    )
    sdpc_root = sdpc_site / "sdpc"
    library_paths = [
        sdpc_root / "so",
        sdpc_root / "so" / "ffmpeg",
        sdpc_root / "so" / "jpeg",
    ]
    old_library_path = value.get("LD_LIBRARY_PATH")
    value["LD_LIBRARY_PATH"] = os.pathsep.join(
        [*(str(path) for path in library_paths), *( [old_library_path] if old_library_path else [] )]
    )
    value["DEEPSPATIAL_SDPC_SITE"] = str(sdpc_site)
    value.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return value


def _run(name: str, command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[run] {name}", flush=True)
    print("      " + " ".join(str(x) for x in command), flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(str(x) for x in command) + "\n")
        process = subprocess.Popen(
            [str(x) for x in command],
            cwd=ROOT,
            env=_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def _complete(manifest: Path, final: Path) -> bool:
    if not manifest.is_file() or not final.is_file():
        return False
    try:
        value = json.loads(manifest.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return value.get("status") == "complete" and value.get("evaluation_mode") == "full_no_holdout"


def _has_all_uot(uot: Path, sections: list[int]) -> bool:
    if not (uot / "summary.csv").is_file() or not (uot / "run_metadata.json").is_file():
        return False
    return all((uot / f"pair-{a:03d}-{b:03d}.npz").is_file() for a, b in zip(sections, sections[1:]))


def _store_has_sections(path: Path, expected_sections: list[str]) -> bool:
    """Treat an HDF5 store as complete only when every section is readable."""

    if not path.is_file():
        return False
    try:
        from deepspatial.histology import FeatureStore

        store = FeatureStore(path)
        return sorted(store.sections) == sorted(expected_sections)
    except Exception as error:
        print(f"[cache-check] invalid/incomplete store {path}: {error}", flush=True)
        return False


def _run_one(series_id: str, sections: list[int]) -> None:
    group_dir, prep, uni2, nucleus, uot, path_cache, final_dir, final = _paths(series_id)
    prep.mkdir(parents=True, exist_ok=True)
    log_dir = prep / "launcher_logs"
    max_section = max(sections)
    frame = f"{series_id}_registered"
    he_table = prep / "he_sections.parquet"
    anchors = prep / "anchors"
    expected_he_sections = [
        str(value)
        for value in pd.read_parquet(he_table)["section_id"].drop_duplicates().tolist()
    ]

    if _complete(final_dir / "reconstruction_manifest.json", final):
        print(f"[skip] {series_id}: complete full reconstruction already exists", flush=True)
        return

    if not _store_has_sections(uni2, expected_he_sections):
        _run(
            f"{series_id} UNI2 feature extraction",
            [
                PYTHON,
                "-u",
                "scripts/run_with_sdpc_env.py",
                "examples/extract_00029_uni2.py",
                "--dataset",
                prep,
                "--through-section",
                str(max_section),
                "--output",
                uni2,
                "--checkpoint",
                UNI2_CHECKPOINT,
                "--device",
                "cuda",
                "--batch-size",
                "64",
            ],
            log_dir / "uni2.log",
        )

    nucleus_store = nucleus / "nucleus_features.h5"
    if not _store_has_sections(nucleus_store, expected_he_sections):
        if nucleus_store.exists():
            raise RuntimeError(
                f"Incomplete nucleus store exists and will not be overwritten automatically: {nucleus_store}"
            )
        _run(
            f"{series_id} Cellpose nucleus feature store",
            [
                CELLPOSE_PYTHON,
                "-u",
                "scripts/run_with_sdpc_env.py",
                "scripts/build_00029_nucleus_feature_store.py",
                "--he-table",
                he_table,
                "--uni2-store",
                uni2,
                "--output",
                nucleus,
                "--series-id",
                series_id,
                "--registration-root",
                V2_ROOT / "outputs/registration",
                "--v2-src",
                V2_ROOT / "data",
                "--cellpose-model",
                CELLPOSE_MODEL,
                "--image-source",
                "raw_sdpc",
                "--raw-target-mpp",
                "1.1213235294117647",
                "--parquet-site-packages",
                PARQUET_SITE_PACKAGES,
            ],
            log_dir / "cellpose.log",
        )

    if not _has_all_uot(uot, sections):
        _run(
            f"{series_id} sparse morphology UOT cache",
            [
                PYTHON,
                "-u",
                "examples/run_00029_uot_qc.py",
                "--dataset",
                prep,
                "--anchor-dir",
                anchors,
                "--spatial-key",
                "spatial_st_corrected",
                "--feature-store",
                uni2,
                "--output",
                uot,
                "--top-k",
                "64",
                "--sweep-k",
                "32",
                "64",
                "128",
                "--coordinate-frame",
                frame,
            ],
            log_dir / "uot.log",
        )

    if not path_cache.is_file() or not (prep / "path_cache_summary.json").is_file():
        _run(
            f"{series_id} nucleus/morphology path cache",
            [
                PYTHON,
                "-u",
                "scripts/build_00029_corrected_path_cache.py",
                "--anchor-dir",
                anchors,
                "--uot-dir",
                uot,
                "--feature-store",
                uni2,
                "--nucleus-feature-store",
                nucleus_store,
                "--output",
                path_cache,
                "--pairs",
                "20000",
                "--seed",
                "0",
                "--coordinate-frame",
                frame,
                "--allow-linear-fallback",
            ],
            log_dir / "path_cache.log",
        )

    _run(
        f"{series_id} full no-holdout training and reconstruction",
        [
            PYTHON,
            "-u",
            "scripts/run_group_celltype_nucleus_uot_path.py",
            "--full",
            "--series-id",
            series_id,
            "--anchor-dir",
            anchors,
            "--feature-store",
            uni2,
            "--nucleus-feature-store",
            nucleus_store,
            "--path-cache",
            path_cache,
            "--output",
            final_dir,
            "--coordinate-frame",
            frame,
            "--sections",
            *[str(x) for x in sections],
            "--epochs",
            "10",
            "--pairs",
            "20000",
            "--seed",
            "0",
            "--nucleus-weight",
            "1",
            "--lambda-c",
            "10",
            "--thickness-um",
            "10",
            "--reconstruction-steps",
            "100",
            "--reconstruction-chunk-size",
            "512",
            "--inference-chunk-size",
            "512",
            "--device",
            "cuda",
        ],
        log_dir / "reconstruction.log",
    )


def main() -> None:
    print(f"DeepSpatial remaining full reconstruction launcher; GPU={GPU}", flush=True)
    for series_id, sections in SERIES:
        _run_one(series_id, sections)
    print("[complete] all remaining series finished", flush=True)


if __name__ == "__main__":
    main()
