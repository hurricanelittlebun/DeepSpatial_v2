import subprocess
import sys


def test_vis_utils_imports_without_ipywidgets():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import deepspatial.vis_utils; print('vis_utils_imported')",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "vis_utils_imported" in result.stdout
