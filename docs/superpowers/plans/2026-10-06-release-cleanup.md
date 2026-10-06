# DeepSpatial Release Cleanup Implementation Plan

> For agentic workers: use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Prepare the DeepSpatial repository for a reproducible Git/PyPI release by removing the obsolete H&E spatial late-fusion head while preserving morphology-aware UOT/path, nucleus path, cell-type flow, and the upstream package boundary.

**Architecture:** Keep the upstream flat deepspatial/ package layout and public import paths. H&E remains an offline/runtime input to morphology-aware UOT and cached anatomical paths, but no H&E feature is passed into GiT's learned spatial velocity head. Keep research scripts in stable paths, document their categories, and exclude local datasets and generated artifacts from Git/package payloads.

**Tech Stack:** Python 3.10+, PyTorch, PyTorch Lightning, AnnData, HDF5, pytest, setuptools, python -m build.

**Spec:** docs/superpowers/specs/2026-10-06-release-cleanup-design.md

## Global Constraints

- Preserve the original spatial, gene, and cell-type heads and the use_celltype option.
- Preserve frozen UNI2 feature stores, morphology-aware UOT, morphology/nucleus paths, coordinate validation, and cached path behavior.
- Remove use_histology_spatial_head, histology_dim, he_projection, and ht from the current API and active code paths.
- Do not change registration, UOT mathematics, path construction, or reconstruction behavior except for removing the late-fusion spatial-head branch.
- Do not delete or rewrite H5AD, HDF5, Zarr, Parquet, TIFF, SDPC, checkpoints, QC, or reconstruction outputs.
- Do not migrate to src/ or blindly relocate scripts; preserve import paths used by current tests and experiments.
- Do not commit, push, merge, or publish automatically; leave a reviewable working-tree diff.
- Do not retain release-facing hard-coded paths to /data/buyonggan, DeepSpatial_v2, local model weights, or local CUDA environments.

## Review Focus

- Removed API keyword: a caller passing the deleted spatial-head option must fail explicitly instead of silently changing behavior; covered by Task 1.
- Model shape regression: GiT must still return spatial, gene, and cell-type velocities with the original shapes; covered by Task 1.
- Morphology runtime isolation: morphology UOT/path may query features, but ordinary model forward must not query H&E features for a spatial head; covered by Task 2.
- Historical ablation cleanup: no active script may expose or serialize the deleted head as a current component; covered by Task 3.
- Release boundary: package build must include deepspatial.histology while excluding data, weights, tests, scripts, and generated artifacts; covered by Task 4 and Task 5.

---

### Task 1: Remove the H&E spatial-head API from configuration and GiT

**Files:**
- Create: tests/test_spatial_head_removed.py
- Modify: deepspatial/histology/config.py
- Modify: deepspatial/models/git.py

**Interfaces:**
- Consumes: existing HistologyConfig dataclass and GiT constructor/forward signatures.
- Produces: HistologyConfig without use_histology_spatial_head; GiT constructor without histology_dim; GiT.forward without ht, returning x, g, c.

- [ ] Step 1: Write failing API-boundary tests.

  Add test_histology_config_rejects_removed_spatial_head_argument. Construct a valid histology config while passing use_histology_spatial_head=True and expect TypeError.

  Add test_git_forward_keeps_all_three_velocity_shapes_without_histology_head. Construct a small GiT with gene_dim=5, patch_size=2, hidden_size=16, depth=1, num_heads=2, num_classes=3; assert the signatures do not contain the removed names; run batch size 4 and assert output shapes (4, 2), (4, 5), and (4, 3).

- [ ] Step 2: Run the tests and verify the expected failure.

  Run: .venv/bin/python -m pytest tests/test_spatial_head_removed.py -q

  Expected: FAIL because the current config accepts the removed keyword and GiT exposes the H&E projection path.

- [ ] Step 3: Remove the production API.

  In HistologyConfig, delete the field and its validation reference. In GiT, remove histology_dim from the constructor, remove he_projection construction, remove ht from forward, and call x_head directly on h[:, :1, :].

- [ ] Step 4: Run the focused tests and verify they pass.

  Run: .venv/bin/python -m pytest tests/test_spatial_head_removed.py -q

  Expected: 2 passed.

- [ ] Step 5: Run original model smoke tests.

  Run: .venv/bin/python -m pytest tests/test_integration.py::test_reconstruction_rejects_different_gene_order tests/test_guards.py -q

  Expected: all selected tests pass.

### Task 2: Remove runtime forwarding while preserving morphology path and cell-type flow

**Files:**
- Create: tests/test_histology_spatial_head_removed.py
- Modify: deepspatial/core.py
- Modify: deepspatial/module.py
- Modify: tests/test_integration.py
- Modify: tests/test_histology.py

**Interfaces:**
- Consumes: Task 1's GiT signature and existing HistologyRuntime.plan interface.
- Produces: DeepSpatialModule._shared_step that uses H&E only for morphology path targets when enabled, calls the model without extra H&E arguments, and still computes loss_x, loss_g, loss_c and reconstruction outputs.

- [ ] Step 1: Write the failing runtime-isolation test.

  Add test_shared_step_does_not_query_histology_features_for_model_forward. Build the toy two-anchor pipeline with morphology path enabled, replace runtime.features with a callable that raises an assertion, and give the test double a legacy config object whose use_histology_spatial_head is True. Run one _shared_step and assert all three losses are finite. Keep runtime.plan available so the test still exercises morphology-guided spatial targets. The legacy flag on the test double makes the old implementation fail before the production field is removed.

- [ ] Step 2: Run the test and verify it fails for the old implementation.

  Run: .venv/bin/python -m pytest tests/test_histology_spatial_head_removed.py::test_shared_step_does_not_query_histology_features_for_model_forward -q

  Expected: FAIL because the current module conditionally calls histology.features during forward.

- [ ] Step 3: Remove runtime/model glue.

  In core.py, remove the conditional histology_dim model-config injection. In module.py, remove extra ht construction and call the model with the original state tensors only. Keep histology.plan, H&E path sampling, reverse ODE handling, gene path, cell-type path, and loss weights unchanged.

- [ ] Step 4: Update integration tests without weakening coverage.

  Replace the old late-fusion dependency test with the Task 1 model-shape test. Remove the full spatial-head-only mode from the toy pipeline or make it an explicit alias of morphology-path mode with no additional head. Update adaptive ODE test field signatures and all test configs to omit the removed field. Keep feature-store query tests because UOT/path preprocessing still uses them.

- [ ] Step 5: Run focused integration and histology tests.

  Run: .venv/bin/python -m pytest tests/test_spatial_head_removed.py tests/test_histology_spatial_head_removed.py tests/test_integration.py tests/test_histology.py -q

  Expected: all selected tests pass, including morphology path, cell-type output, checkpoint reload, and reconstruction smoke tests.

### Task 3: Clean active examples, experiment scripts, and release-facing documentation

**Files:**
- Modify: examples/morphology_train.py
- Modify: README.md
- Modify: docs/morphology.md
- Create: tests/test_release_layout.py
- Modify: active scripts/run_*.py, scripts/validate_*.py, and scripts/*ablation*.py files that pass or serialize the removed field.
- Modify: release-facing examples containing machine-specific default paths.

**Interfaces:**
- Consumes: Task 1's reduced HistologyConfig API and Task 2's model/runtime behavior.
- Produces: morphology_train.py with supported baseline, uot, and path modes; current scripts instantiate valid configs; README/docs describe H&E as UOT/path guidance only.

- [ ] Step 1: Add an active-reference test.

  Add test_no_removed_spatial_head_in_active_sources to tests/test_release_layout.py. Scan deepspatial/, examples/, scripts/, README.md, docs/morphology.md, and docs/source/; assert none contains use_histology_spatial_head, histology_dim, or he_projection.

- [ ] Step 2: Run the scan and verify it fails.

  Run: .venv/bin/python -m pytest tests/test_release_layout.py::test_no_removed_spatial_head_in_active_sources -q

  Expected: FAIL listing current active references.

- [ ] Step 3: Update active code and documentation.

  Remove obsolete config arguments and metadata keys. Remove the dedicated spatial_head_only ablation arm. Keep baseline/UOT/path/nucleus/cell-type arms. Change the example mode list and morphology documentation to describe supported H&E UOT/path/nucleus behavior without a spatial-head switch. Replace release-facing absolute paths with arguments or checkout-relative defaults; label scripts requiring the separate DeepSpatial_v2 checkout as local research utilities.

- [ ] Step 4: Run the reference scan and help checks.

  Run: .venv/bin/python -m pytest tests/test_release_layout.py::test_no_removed_spatial_head_in_active_sources -q
  Run: .venv/bin/python examples/morphology_train.py --help

  Expected: the scan passes and help lists only supported modes.

### Task 4: Make the Git/PyPI release boundary explicit

**Files:**
- Create: scripts/README.md
- Create: docs/repository-layout.md
- Modify: .gitignore
- Modify: README.md
- Modify: pyproject.toml only if packaging checks identify a concrete metadata issue.
- Modify: tests/test_release_layout.py

**Interfaces:**
- Consumes: upstream package discovery in pyproject.toml and active source tree from Tasks 1–3.
- Produces: documented Git layout, documented script categories, and automated checks for ignored local artifacts and package inclusion.

- [ ] Step 1: Write release-boundary tests.

  Add tests that assert deepspatial.histology is importable, data/, weights/, and artifacts/ are ignored by Git, and no source file under deepspatial/ is ignored. Use git check-ignore through subprocess so the tests check actual repository rules.

- [ ] Step 2: Run tests and verify the expected failure.

  Run: .venv/bin/python -m pytest tests/test_release_layout.py -q

  Expected: FAIL until guidance and any missing ignore rules are added.

- [ ] Step 3: Add release guidance and safe ignore rules.

  Document reusable package code, Git-only tests/examples/scripts, and local data/results. Add only standard build/cache/data-artifact patterns such as dist/, build/, *.egg-info/, *.h5ad, *.h5, *.zarr, *.parquet, *.tif, *.tiff, and *.sdpc. Do not ignore Python source or documentation. Keep script paths stable and classify scripts in scripts/README.md.

- [ ] Step 4: Run release-boundary tests.

  Run: .venv/bin/python -m pytest tests/test_release_layout.py -q

  Expected: all release-layout tests pass.

### Task 5: Run complete release verification

**Files:**
- Modify: docs/morphology-validation.md or add a release validation note only if commands/results need recording.

**Interfaces:**
- Consumes: all changes from Tasks 1–4.
- Produces: verified source tree and a final report distinguishing publishable files from local-only outputs.

- [ ] Step 1: Run complete test suite.

  Run: OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests -q

  Expected: zero failures. If an optional dependency is unavailable, record the exact test and environment limitation.

- [ ] Step 2: Compile all checked-in Python source.

  Run: .venv/bin/python -m compileall -q deepspatial examples scripts

  Expected: exit code 0.

- [ ] Step 3: Build the distribution.

  Run: .venv/bin/python -m build

  Expected: wheel and sdist are created; inspect their file lists and confirm deepspatial/histology is included while data, weights, tests, scripts, and generated artifacts are absent.

- [ ] Step 4: Run final repository checks.

  Run: git diff --check; git status --short; and a final active-source search for removed names.

  Expected: no whitespace errors, no obsolete active references, no deleted user outputs, and a concise list of files ready to push versus intentionally local/ignored files.

- [ ] Step 5: Do not commit or push.

  Leave the verified working-tree diff for the user to review, stage, commit, and push.
