# Nucleus-guided Morphology Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional nuclear-structure field to the existing morphology-guided path and run an independent 00029/g0 ablation without changing existing UNI2/UOT results.

**Architecture:** Cellpose produces nuclei centroids and basic shape/QC tables from the already reviewed H&E crops. Registered nuclear centroids are converted into a physical XY descriptor grid compatible with `FeatureStore`. `MorphologyPathCache` optionally compares adjacent-layer nuclear descriptors in addition to UNI2 descriptors; UOT, gene flow, cell-type flow, and spatial head remain unchanged.

**Tech Stack:** Python 3.10, PyTorch, HDF5/`FeatureStore`, SciPy spatial queries, scikit-image, Cellpose 2.2.3 in the existing `hespotex_env`, Pytest.

**Spec:** `docs/morphology-plan.md` and the user's nucleus-structure path proposal.

## Global Constraints

- Do not re-register H&E or alter existing transforms.
- Do not overwrite `data/00029_g0/uni2_features.h5`, existing path caches, checkpoints, or metrics.
- Nuclear descriptors represent local anatomy, not cell lineage or cross-section cell tracking.
- Nuclear information is path-only in this experiment; it does not enter UOT, gene head, or cell-type head.
- Existing behavior must remain unchanged when nuclear-path mode is disabled.
- Use physical micrometre coordinates for descriptor construction and path costs; retain normalized coordinates only at the model interface.
- Precompute segmentation/descriptors and cache them; do not run Cellpose inside training or `Dataset.__getitem__`.

---

### Task 1: Define the optional nuclear path interface

**Files:**
- Modify: `deepspatial/histology/config.py`
- Modify: `deepspatial/histology/runtime.py`
- Modify: `deepspatial/histology/path.py`
- Test: `tests/test_histology.py`

**Interfaces:**
- `HistologyConfig(use_nucleus_path=False, nucleus_feature_path=None, nucleus_weight=0.0)` remains backward compatible.
- `HistologyRuntime` opens an optional second `FeatureStore` and passes it to `MorphologyPathCache`.
- `MorphologyPathCache(..., nucleus_store=None, nucleus_weight=0.0)` adds nuclear cosine distance only when both are enabled.

- [ ] **Step 1: Write the failing test**

Add a synthetic test that builds a UNI2 store and a nuclear descriptor store where the lower-cost route is different only because of the nuclear descriptors. Assert the cached path changes when `nucleus_weight > 0` and is identical to the old route when `nucleus_weight == 0`.

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `pytest -q tests/test_histology.py -k nucleus`

Expected: FAIL because `MorphologyPathCache` does not accept `nucleus_store`.

- [ ] **Step 3: Implement the minimal interface**

Validate that the second store has the same section IDs, physical Z values, coordinate frame, and compatible grid coordinates. Query nuclear features at each local candidate. Keep candidates valid under the existing UNI2 mask; neutralize nuclear cost only where a nuclear query is unavailable. Add:

\[
E = \lambda_s D_{move} + \lambda_h D_{UNI2} + \lambda_n D_{nucleus}
\]

to the dynamic-programming edge cost. Include both feature-store revisions and nuclear parameters in the content-addressed cache metadata.

- [ ] **Step 4: Run the focused test and the existing histology tests**

Run: `pytest -q tests/test_histology.py -k 'nucleus or cached or physical or mask'`

Expected: PASS.

- [ ] **Step 5: Run the complete test suite**

Run: `pytest -q`

Expected: all pre-existing tests pass and nuclear tests pass.

---

### Task 2: Build the 00029/g0 nuclear descriptor pipeline

**Files:**
- Create: `scripts/build_00029_nucleus_feature_store.py`
- Create: `deepspatial/histology/nucleus.py`
- Test: `tests/test_histology.py`

**Interfaces:**
- `segment_h_and_e(rgb, mask, model, diameter_px)` returns an instance mask and Cellpose score map.
- `nuclei_table_from_mask(mask, score, mpp_um, section_id)` returns a table with ID, centroid, area, perimeter, eccentricity, and edge/QC fields.
- `descriptor_grid_from_nuclei(xy_um, areas, eccentricities, grid_xy, radius_um)` returns finite `[H,W,D_nucleus]` descriptors plus a boolean valid grid.
- CLI reads `data/00029_g0/he_sections.parquet`, applies saved `DeepSpatial_v2` transforms to source-crop nucleus centroids, and writes an independent output directory and HDF5 nuclear feature store.

- [ ] **Step 1: Write failing unit tests**

Test that a synthetic nucleus set produces finite descriptors, that a local nucleus changes the center descriptor, that an empty/background location is invalid, and that source-crop to registered micrometre conversion preserves known affine points.

- [ ] **Step 2: Run tests and verify the new API fails**

Run: `pytest -q tests/test_histology.py -k 'descriptor or nucleus_table'`

Expected: FAIL because the new module does not exist.

- [ ] **Step 3: Implement the descriptor functions**

Use fixed physical windows matching the UNI2 path scale. Encode local nuclear density, log-count, mean/std log-area, mean eccentricity, and a small relative nuclear-density stencil. Normalize descriptor channels robustly and mark grid locations valid only when the H&E tissue mask and nuclear query support are present.

- [ ] **Step 4: Implement the Cellpose CLI**

Use the existing Cellpose model `/data/buyonggan/.cellpose/models/nucleitorch_0` in `hespotex_env`. Set pixels outside the reviewed H&E mask to white before hematoxylin deconvolution. Run on each source H&E crop, convert crop pixels to source micrometres using `he_sections.parquet`, apply the saved section transform from `/data/buyonggan/DeepSpatial_v2/outputs/registration`, and rasterize descriptors onto the exact registered feature-grid geometry from `uni2_features.h5`. Save per-section nuclei CSV, optional uint32 instance masks, QC overlays, a summary table, and `nucleus_features.h5` with provenance and revisions.

- [ ] **Step 5: Run unit tests and inspect one pilot output**

Run: `pytest -q tests/test_histology.py -k 'descriptor or nucleus_table'`.

Expected: PASS; inspect section-003 count/overlay against the existing pilot before processing all sections.

---

### Task 3: Run the independent nucleus-enhanced ablation

**Files:**
- Create: `scripts/run_00029_inner031_nucleus_ablation.py`
- Create: `docs/superpowers/plans/2026-09-16-nucleus-guided-path-run.md`
- Output: `data/00029_g0/nucleus_path_v1/`

**Interfaces:**
- Uses the existing `run_00029_inner031_ablation.py` training/evaluation protocol: training sections exclude 031 and 051, held-out section 031 is scored after fitting, seed 0, 10 epochs, sparse top-k 64, and `use_celltype=False` for this internal comparison.
- Configuration enables `use_morphology_uot=True`, `use_morphology_path=True`, `use_nucleus_path=True`, `use_histology_spatial_head=False`.

- [ ] **Step 1: Add a smoke test for configuration serialization**

Assert that the run configuration records `nucleus_feature_path`, `nucleus_weight`, and the four morphology/cell-type flags without changing the original model configuration.

- [ ] **Step 2: Run the smoke test**

Run: `pytest -q tests/test_histology.py -k 'nucleus or config'`.

Expected: PASS after Task 1 and Task 2 are complete.

- [ ] **Step 3: Run the full independent ablation**

Run on an unoccupied GPU with a new output directory. Keep the existing six-arm results and the `uot_path_no_spatial_head` result untouched.

- [ ] **Step 4: Verify artifacts**

Check that segmentation summaries, nuclear feature store, path cache, checkpoint, manifest, and held-out metrics exist; verify endpoints remain exact and no forbidden H&E→gene/cell-type information path was introduced.

- [ ] **Step 5: Compare metrics and path QC**

Report the new result against `uot_only`, `path_only`, `uot_path_no_spatial_head`, and `full`, including Profile Pearson, macro gene Pearson, MAE, RMSE, path tissue coverage, nuclear-support coverage, and fallback counts. Do not call the result a formal conclusion until repeated seeds/held-out sections are run.

---

## Self-review checklist

- The default path and UOT behavior remain unchanged without nuclear mode.
- The nucleus term is precomputed and cached, never run in the training loop.
- Nuclear features are not direct inputs to `v_g`, `v_c`, or gene regression.
- Invalid nuclear coverage is neutral and auditable rather than silently treated as evidence.
- Every route is described as anatomical correspondence, not cell lineage.
- Existing outputs are preserved in place.
