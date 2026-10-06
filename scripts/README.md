# Research scripts

The scripts in this directory are reproducible research utilities rather than
part of the importable `deepspatial` package. Their paths are kept stable
because some experiments and tests import them directly.

## Categories

- `prepare_*`, `build_*`, `materialize_*`: prepare anchors, feature stores,
  coordinate frames, and persistent UOT/path inputs.
- `run_*`: train or reconstruct DeepSpatial experiments.
- `fit_*`, `repair_*`, `bake_*`: fit or materialize registration and
  coordinate corrections that have already been decided outside the model.
- `validate_*`, `audit_*`, `evaluate_*`: data, cache, and metric checks.
- `render_*`, `visualize_*`, `plot_*`: static, 3D, and overlay visualizations.
- `annotate_*`, `merge_*`, `refresh_*`: metadata and AnnData utilities.

## Running scripts

Run a script from the repository root and inspect its help first:

```bash
python scripts/<script>.py --help
```

Most scripts use checkout-relative defaults under `data/`. Scripts that need
the separate DeepSpatial_v2 registration/segmentation tree or a local
Cellpose/UNI2 checkpoint expose those locations as command-line arguments.
Those external inputs are intentionally not bundled with this repository.

Generated H5AD, feature stores, images, logs, checkpoints, and intermediate
manifests belong under the ignored local data/artifacts/weights directories;
they are not source files for the Git release.
