# DeepSpatial release cleanup design

## Status

Design for review. No production-code changes are authorized by this document
alone.

## Context

The repository started from the upstream DeepSpatial release. The upstream
release publishes the reusable `deepspatial` package, documentation, website
assets, and CI workflows. The current working tree adds morphology-aware UOT,
registered H&E/UNI2 feature stores, morphology/nucleus paths, reconstruction
experiments, tests, and many local data-processing scripts.

The current tree is also intentionally dirty: its changes are the user's
ongoing research work and must not be reset or discarded during cleanup.

## Goals

1. Make the repository safe to push as a reusable research codebase.
2. Keep the public DeepSpatial API and original cell-type branch intact.
3. Remove the H&E late-fusion spatial-head option and implementation.
4. Keep the other H&E capabilities: frozen UNI2 feature stores,
   morphology-aware UOT, morphology/nucleus-guided paths, coordinate checks,
   and H&E preprocessing utilities.
5. Make it clear which files belong in Git and which are local generated data.
6. Preserve import paths used by the current experiments and tests wherever
   possible; avoid a risky `src/` migration or blind script relocation.

## Non-goals

- Do not change registration, UOT mathematics, path construction, cell-type
  conditioning, or reconstruction behavior beyond removing the spatial-head
  branch.
- Do not delete or rewrite existing H5AD, HDF5, Zarr, Parquet, TIFF, SDPC,
  checkpoint, QC, or reconstruction outputs.
- Do not push, merge, or commit automatically.
- Do not make the package depend on a local `DeepSpatial_v2` checkout,
  `/data/buyonggan`, a local Cellpose model, or a particular CUDA installation.

## Release boundary

### Included in the Git repository

- `deepspatial/`, including the new `deepspatial.histology` package.
- `tests/` for library and extension regression coverage.
- `examples/` after release-facing examples use command-line paths or paths
  relative to the checkout.
- `docs/`, README, license, and CI workflows.
- `scripts/` as research/reproduction utilities, with a README that separates
  reusable workflows from dataset-specific local scripts.

### Included in the Python distribution

The existing setuptools package discovery remains the source of truth:

```text
deepspatial*
```

Tests, examples, scripts, data, weights, and generated artifacts remain out
of the wheel/sdist package payload, matching the upstream release behavior.
The optional `histology` dependency group remains available for H&E/UNI2
preprocessing and feature-store use.

### Local-only and ignored

The following remain local and are not part of the Git release:

```text
data/
weights/
artifacts/
.venv/
__pycache__/
*.pyc
*.h5ad
*.h5
*.zarr
*.parquet
*.tif
*.tiff
*.sdpc
```

The ignore rules should be checked without broadening them to hide source
code, tests, or documentation.

## Code changes

### Remove H&E spatial late fusion

Remove the public configuration field `use_histology_spatial_head` and the
corresponding implementation:

- `HistologyConfig` no longer exposes or serializes the field.
- `DeepSpatial` no longer derives `histology_dim` from that field.
- `DeepSpatialModule` no longer queries H&E features during the model forward
  pass for the spatial head.
- `GiT` no longer accepts `histology_dim` or `ht`, and no longer constructs
  `he_projection` or adds it to `hx`.
- The ordinary spatial, gene, and cell-type heads remain unchanged.

H&E features continue to be usable by morphology-aware UOT and the cached
morphology/nucleus path runtime. This means H&E still constrains endpoint
correspondence and spatial path targets, but is not concatenated or added to
the learned spatial velocity head.

### Configuration and metadata cleanup

Update active examples, scripts, tests, and release-facing documentation so
they no longer pass or advertise the removed field. Historical experiment
notes may retain their historical wording when they are explicitly marked as
archival; they must not be used as current API documentation.

The existing `use_histology`, `use_morphology_uot`, `use_morphology_path`,
`use_nucleus_path`, and `use_celltype` controls remain.

### Repository structure and documentation

Keep the upstream top-level layout rather than migrating to `src/`:

```text
deepspatial/       reusable library
tests/             regression tests
examples/          small public examples
scripts/           reproducible research utilities
docs/              user and developer documentation
assets/            website/documentation assets
data/              ignored local datasets
weights/           ignored local model weights
artifacts/         ignored local outputs
```

Add concise repository guidance and a `scripts/README.md` index. Do not move
the existing scripts in this cleanup because several are imported by tests or
share checkout-relative assumptions; classify them first and leave their
paths stable.

Replace hard-coded machine-specific paths in release-facing examples and
documentation with CLI arguments or checkout-relative defaults. Scripts that
are intrinsically tied to the local DeepSpatial_v2 data checkout should be
explicitly labeled as local research utilities and require their external
roots as arguments.

## Tests and release checks

Add or update tests for the removed API boundary:

1. `HistologyConfig` rejects the removed field rather than silently accepting
   it.
2. `GiT` has no H&E projection parameter and the normal forward path still
   returns `(v_x, v_g, v_c)` with the same shapes.
3. Morphology UOT/path and cell-type behavior remain available without the
   spatial head.

Then run:

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python -m compileall -q deepspatial examples scripts
.venv/bin/python -m build
git diff --check
```

If optional dependencies are unavailable, report the exact skipped check and
run the remaining import/compile checks. No claim of release readiness will
be made until the results are recorded.

## Acceptance criteria

- No current code, active example, or release-facing documentation refers to
  `use_histology_spatial_head`, `histology_dim`, or `he_projection`.
- Morphology UOT/path, nucleus path, and cell-type branches still import and
  pass their regression tests.
- `pip install -e .`/package build includes `deepspatial.histology` but not
  local data, weights, tests, or generated artifacts.
- No existing user data or experiment output is deleted or overwritten.
- The final status report clearly separates source files to push from local
  files to keep ignored.
