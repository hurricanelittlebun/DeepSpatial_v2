# Repository layout and release boundary

DeepSpatial follows the upstream repository layout:

```text
deepspatial/       importable library and public API
tests/             regression and integration tests
examples/          small runnable examples
scripts/           research/reproduction utilities
docs/              user, API, and developer documentation
assets/            website and documentation assets
data/              local datasets and reconstruction outputs (ignored)
weights/           local model weights (ignored)
artifacts/         local caches, logs, and figures (ignored)
```

## What is pushed to Git

Push the reusable package, tests, examples, scripts, documentation, assets,
license, and CI workflows. The package discovery in `pyproject.toml` includes
`deepspatial*`, so the H&E/UNI2 modules under `deepspatial/histology/` are
included in the Python distribution.

Tests, examples, scripts, and research documentation are Git resources but
are not included in the wheel or source distribution.

## What stays local

Do not push H5AD/HDF5/Zarr/Parquet/TIFF/SDPC files, feature stores, model
weights, checkpoints, generated QC figures, logs, or reconstruction outputs.
The repository ignore rules cover the project data directories and common
generated file extensions. Check the result before staging:

```bash
git status --short
git check-ignore -v data/example.h5ad weights/example.bin artifacts/example.png
```

## Current H&E interface

H&E/UNI2 data can guide morphology-aware UOT and cached morphology/nucleus
paths. The released GiT does not expose an H&E spatial late-fusion head:
spatial, gene, and cell-type velocity heads retain the original DeepSpatial
interface.
