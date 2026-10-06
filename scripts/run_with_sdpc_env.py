"""Run one repository script with the vendor SDPC Python package appended.

The SDPC wheel is installed in a Python 3.8 environment, but its pure-Python
bindings can be imported by the project environments.  Appending (rather
than prepending) that site-packages directory keeps the active environment's
NumPy/POT stack ahead of the vendor environment's unrelated packages.
The native library search path is prepared by the parent launcher before the
Python process starts.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


DEFAULT_SDPC_SITE = Path(
    "/data/buyonggan/miniconda3/envs/spateo_venv/lib/python3.8/site-packages"
)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_with_sdpc_env.py SCRIPT [ARGS ...]")
    script = Path(sys.argv[1]).resolve()
    if not script.is_file():
        raise FileNotFoundError(script)
    site = Path(os.environ.get("DEEPSPATIAL_SDPC_SITE", DEFAULT_SDPC_SITE)).resolve()
    if not site.is_dir():
        raise FileNotFoundError(f"SDPC site-packages not found: {site}")
    site_text = str(site)
    if site_text not in sys.path:
        sys.path.append(site_text)
    sys.argv = [str(script), *sys.argv[2:]]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
