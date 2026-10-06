from pathlib import Path
import subprocess
import sys


def test_nucleus_rerun_entrypoint_accepts_final_anchor_directory():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/run_00029_inner031_nucleus_ablation.py"),
            "--help",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--anchor-dir" in result.stdout


def test_nucleus_rerun_anchor_path_is_not_bound_to_old_candidate_package(tmp_path):
    from scripts.run_00029_inner031_nucleus_ablation import _anchor_path

    assert _anchor_path(tmp_path, 31) == tmp_path / "section-031.h5ad"
