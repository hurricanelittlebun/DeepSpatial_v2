# Fixed-H&E Feature Frame Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resample UNI2 and nuclear morphology fields into the residual-corrected H&E frame, then regenerate 00029/g0 UOT, anatomical paths, and the section-031 holdout.

**Architecture:** Add a small, offline physical-coordinate resampler that reads an existing `FeatureStore`, applies a section-specific current-to-fixed affine to grid centers, and bilinearly queries the source field. It produces new immutable HDF5 stores in `00029__g0_fixed_he`; model code continues to query one frame, so no training-loop coordinate special cases are needed.

**Tech Stack:** Python 3.10, NumPy, SciPy, h5py, PyTorch, AnnData, existing `FeatureStore`, sparse top-k UOT, existing path cache, PyTorch Lightning.

**Spec:** `docs/superpowers/specs/2026-09-16-fixed-he-feature-frame-design.md`

## Global Constraints

- Preserve all original H&E, STalign, feature, UOT, path, and model outputs.
- Use physical XY coordinates in micrometers; output frame is exactly `00029__g0_fixed_he`.
- Use `p_fixed = M_current_to_fixed @ p_registered` and source lookup `M_current_to_fixed^-1 @ p_fixed`.
- Use bilinear interpolation with nearest-valid recovery disabled while resampling validity masks.
- Use the corrected anchor key `spatial_st_corrected` and do not modify expression matrices.
- Keep UNI2 frozen and do not add H&E features to gene or cell-type heads.
- Run tests before declaring the new experiment valid.

---

### Task 1: Add the affine feature-field resampler API

**Files:**
- Create: `deepspatial/histology/frame_resample.py`
- Test: `tests/test_frame_resample.py`

**Interfaces:**
- `load_section_affines(path, section_ids, identity_sections=...) -> dict[str, np.ndarray]`
- `transform_points(points, matrix) -> np.ndarray`
- `resample_feature_store(source_path, output_path, affine_by_section, output_frame, provenance) -> dict`

- [ ] **Step 1: Write failing tests**

  Add tests for: homogeneous affine transformation, identity resampling on a small synthetic `FeatureStore`, transformed-grid value recovery from a known linear field, matching output section metadata, and rejection of a non-invertible affine.

- [ ] **Step 2: Run the focused tests and confirm the expected failure**

  Run:

  ```bash
  .venv/bin/python -m pytest -q tests/test_frame_resample.py
  ```

  Expected: import/API failures because `frame_resample.py` does not exist.

- [ ] **Step 3: Implement the minimal resampler**

  Use `FeatureStore` metadata and HDF5 groups. For each section, transform the four extreme source-grid center corners, create an axis-aligned output grid with the source spacing, map output centers back with the inverse affine, and call `FeatureStore.get_feature(..., return_valid=True, nearest_max_distance_um=0.0)`. Write output features as float32 and the returned validity mask as the output section mask. Preserve Z, patch FOV, MPP, and source metadata while adding matrix and interpolation provenance.

- [ ] **Step 4: Run the focused tests and confirm they pass**

  Run the same focused pytest command; expected: all frame-resampling tests pass.

- [ ] **Step 5: Run the existing regression suite**

  ```bash
  .venv/bin/python -m pytest -q
  ```

  Expected: existing tests remain green.

---

### Task 2: Add the 00029/g0 fixed-frame materialization script

**Files:**
- Create: `scripts/materialize_00029_fixed_he_feature_stores.py`
- Create: `data/00029_g0/fixed_he_feature_frame_v1/README.md`
- Test: `tests/test_00029_fixed_he_materialization.py`

**Interfaces:**
- CLI options: `--uni2-source`, `--nucleus-source`, `--residual-dir`, `--output-dir`, `--output-frame`.
- Outputs: `uni2_features.h5`, `nucleus_features.h5`, `manifest.json`, and `materialization_summary.json` under the new output directory.

- [ ] **Step 1: Write failing tests**

  Test that the script's affine loader returns the ten residual matrices plus identity for serial sections, rejects a missing anchor transform, and records the source revision and matrix file for every section.

- [ ] **Step 2: Run focused tests and confirm failure**

  ```bash
  .venv/bin/python -m pytest -q tests/test_00029_fixed_he_materialization.py
  ```

  Expected: import failure for the new script/module.

- [ ] **Step 3: Implement the materialization script**

  Load the two existing stores and `registration_correction_v1/transforms/section-XXX__st_to_fixed_he_residual.npz`. Read all section IDs from `he_sections.parquet`; assign the residual matrix to companion sections and identity to serial sections. Materialize separate output stores through the Task 1 API, then write a manifest containing source paths, revisions, per-section matrix path/hash, transform type, output frame, grid settings, and counts.

- [ ] **Step 4: Run focused tests and existing tests**

  ```bash
  .venv/bin/python -m pytest -q tests/test_00029_fixed_he_materialization.py tests/test_frame_resample.py
  .venv/bin/python -m pytest -q
  ```

  Expected: all pass.

- [ ] **Step 5: Materialize the real stores without overwriting old data**

  ```bash
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -u scripts/materialize_00029_fixed_he_feature_stores.py \
    --uni2-source data/00029_g0/uni2_features_raw_corrected.h5 \
    --nucleus-source data/00029_g0/nucleus_path_raw_final/nucleus_features.h5 \
    --residual-dir data/00029_g0/registration_correction_v1 \
    --output-dir data/00029_g0/fixed_he_feature_frame_v1 \
    --output-frame 00029__g0_fixed_he
  ```

  Expected: two new HDF5 stores and manifests; old stores remain byte-for-byte present.

- [ ] **Step 6: Audit the materialized stores**

  Check 82 sections, equal section/Z grids between stores, frame equality, finite values, transformed anchor bounds, identity-section reproducibility, and corrected-anchor feature-query validity. Save the audit in `materialization_summary.json`.

---

### Task 3: Regenerate fixed-frame UOT and path caches

**Files:**
- Modify: `scripts/run_00029_uot_qc.py`
- Modify: `scripts/build_00029_corrected_path_cache.py`
- Create: `scripts/audit_00029_fixed_he_path_cache.py`
- Test: `tests/test_fixed_he_cache_metadata.py`

**Interfaces:**
- UOT output: `data/00029_g0/uot_cache_topk64_st_corrected_fixed_he_v1/`.
- Path output: `data/00029_g0/train_full_00029_st_corrected_fixed_he_nucleus_v1/paths.h5`.
- Both outputs must record `spatial_st_corrected`, `00029__g0_fixed_he`, and the two new store revisions.

- [ ] **Step 1: Write failing metadata tests**

  Add tests that reject UOT metadata with a non-fixed feature revision/frame and reject path metadata whose embedded UNI2/nucleus revisions do not match the new stores.

- [ ] **Step 2: Run tests and confirm failure**

  ```bash
  .venv/bin/python -m pytest -q tests/test_fixed_he_cache_metadata.py
  ```

- [ ] **Step 3: Implement metadata/frame validation**

  Add explicit feature-frame arguments and validation to the scripts, preserve the no-overwrite behavior, and make the path summary distinguish the initial builder count from any runtime-added path groups.

- [ ] **Step 4: Run focused and full tests**

  ```bash
  .venv/bin/python -m pytest -q tests/test_fixed_he_cache_metadata.py tests/test_frame_resample.py
  .venv/bin/python -m pytest -q
  ```

- [ ] **Step 5: Recompute sparse morphology UOT**

  ```bash
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
  .venv/bin/python -u examples/run_00029_uot_qc.py \
    --dataset data/00029_g0 \
    --anchor-dir data/00029_g0/anchors_st_to_fixed_he_candidate \
    --spatial-key spatial_st_corrected \
    --feature-store data/00029_g0/fixed_he_feature_frame_v1/uni2_features.h5 \
    --output data/00029_g0/uot_cache_topk64_st_corrected_fixed_he_v1 \
    --top-k 64 --sweep-k 64 --histology-weight 1.0
  ```

- [ ] **Step 6: Build the fixed-frame nucleus-guided path cache**

  ```bash
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
  .venv/bin/python -u scripts/build_00029_corrected_path_cache.py \
    --anchor-dir data/00029_g0/anchors_st_to_fixed_he_candidate \
    --uot-dir data/00029_g0/uot_cache_topk64_st_corrected_fixed_he_v1 \
    --feature-store data/00029_g0/fixed_he_feature_frame_v1/uni2_features.h5 \
    --nucleus-feature-store data/00029_g0/fixed_he_feature_frame_v1/nucleus_features.h5 \
    --output data/00029_g0/train_full_00029_st_corrected_fixed_he_nucleus_v1/paths.h5 \
    --coordinate-frame 00029__g0_fixed_he \
    --pairs 20000 --seed 0 --candidate-radius-um 100 \
    --candidate-spacing-um 25 --move-weight 0.001 \
    --morphology-weight 1.0 --nucleus-weight 1.0 --allow-linear-fallback
  ```

- [ ] **Step 7: Audit cache integrity**

  Verify all embedded revisions, exact endpoints, finite PCHIP coefficients, fallback counts, pair coverage, and no HDF5 lock remains. Write the audit JSON without changing the old cache.

---

### Task 4: Run the corrected fixed-frame holdout

**Files:**
- Modify: `scripts/run_00029_inner031_nucleus_ablation.py`
- Create: `data/00029_g0/ablation_inner031_nucleus_path_fixed_he_v1/`
- Test: `tests/test_fixed_he_experiment_config.py`

**Interfaces:**
- Model frame: `00029__g0_fixed_he`.
- Input coordinate key: `spatial_st_corrected`.
- Output metrics: `ablation_inner031_nucleus_path_fixed_he_v1/nucleus_path/holdout_metrics.json`.

- [ ] **Step 1: Write failing configuration tests**

  Test that the experiment configuration points to both fixed-frame stores, uses the fixed frame, and refuses a registered-frame store in the fixed-frame run.

- [ ] **Step 2: Run focused tests and confirm failure**

  ```bash
  .venv/bin/python -m pytest -q tests/test_fixed_he_experiment_config.py
  ```

- [ ] **Step 3: Implement the minimal script configuration change**

  Parameterize feature, nucleus, path-cache, and coordinate-frame paths; use the fixed-frame defaults only in this new output namespace; preserve old experiment defaults and outputs.

- [ ] **Step 4: Run focused and full tests**

  ```bash
  .venv/bin/python -m pytest -q tests/test_fixed_he_experiment_config.py
  .venv/bin/python -m pytest -q
  ```

- [ ] **Step 5: Run the fixed-frame holdout**

  ```bash
  CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
  .venv/bin/python -u scripts/run_00029_inner031_nucleus_ablation.py \
    --output data/00029_g0/ablation_inner031_nucleus_path_fixed_he_v1 \
    --epochs 10 --pairs 20000 --nucleus-weight 1.0 \
    --feature-store data/00029_g0/fixed_he_feature_frame_v1/uni2_features.h5 \
    --nucleus-feature-store data/00029_g0/fixed_he_feature_frame_v1/nucleus_features.h5 \
    --path-cache data/00029_g0/train_full_00029_st_corrected_fixed_he_nucleus_v1/paths.h5 \
    --coordinate-frame 00029__g0_fixed_he
  ```

- [ ] **Step 6: Verify the result**

  Confirm the manifest, model config, checkpoint, held-out section, feature revisions, path revision, and metrics. Compare against old metrics only as a versioned diagnostic, not as a controlled claim unless all inputs match.

---

### Task 5: Final verification and handoff

**Files:**
- Modify: `data/00029_g0/dataset_manifest.json` only if the new fixed-frame artifacts are verified.
- Create: `docs/superpowers/verification/2026-09-16-fixed-he-feature-frame.md`

- [ ] **Step 1: Run final code verification**

  ```bash
  .venv/bin/python -m pytest -q
  .venv/bin/python -m py_compile deepspatial/histology/frame_resample.py scripts/materialize_00029_fixed_he_feature_stores.py scripts/run_00029_uot_qc.py scripts/build_00029_corrected_path_cache.py scripts/run_00029_inner031_nucleus_ablation.py
  git diff --check
  ```

- [ ] **Step 2: Run final data audit**

  Verify fixed-frame store/cache/config consistency and that all old outputs still exist. Record exact counts, revisions, fallback fractions, and holdout metrics in the verification document.

- [ ] **Step 3: Update the dataset manifest**

  Add the fixed-frame artifact paths and status only after the audits pass; leave the old artifact paths listed for reproducibility.

- [ ] **Step 4: Report limitations**

  State whether the frame mismatch was removed, whether fixed-frame metrics are comparable, and whether any residual fallback or segmentation-quality issue remains.
