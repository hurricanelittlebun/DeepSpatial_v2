# 9957/g0 51–53 Unified Registration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Align H&E section 53 directly to the Xenium anchor section 51, preserve the reviewed 53→55 postrotate chain, and materialize one consistent coordinate frame for ST, H&E, UNI2, and nucleus features without modifying v9/v10.

**Architecture:** Section 51 remains the fixed anchor. A new direct STalign edge maps the existing v6 position of section 53 into the section-51 target frame. For section 53 and all later sections, the new downstream frame is the new 51→53 edge composed after the existing v9 53→55 postrotate chain. Sections before 53 remain unchanged. The new release is independent and auditable.

**Tech Stack:** Python, NumPy, pandas, AnnData, HDF5/FeatureStore, official STalign LDDMM, CUDA, pytest.

**Spec:** User request in the current conversation: execute the plan to fix 51–53, keep the reviewed 53–55 v9 postprocessing, and avoid mixing v9/v10 coordinate frames.

## Global Constraints

- Do not overwrite v9 or v10 outputs.
- Preserve the section-51 Xenium anchor coordinates.
- Preserve the reviewed v9 53→55 chain: horizontal flip → direct STalign → CCW90.
- Apply one final coordinate frame to every generated H5AD, H&E table, UNI2 store, and nucleus store.
- Do not use the v10 feature-support residual in the new release.
- Keep 54 removed and section 91 retained.
- Do not claim alignment success without fresh red/green and transform-contract verification.

## Review Focus

- 51 remains fixed while 53 and downstream sections receive the new correction.
- The v9 53→55 nonlinear map is composed in the correct order, not replaced by its coarse affine.
- H&E crop bbox offsets are applied exactly once.
- Feature and nucleus grids are resampled through inverse composite transforms.
- The new H5AD, table, feature stores, and transforms all advertise the same coordinate frame.

### Task 1: Add transform-composition and source-coordinate regression tests

**Files:**
- Create: `tests/test_9957_unified_alignment.py`
- Create: `scripts/9957_unified_alignment.py`

**Interfaces:**
- Produces `Unified9957Chain.forward_points(points, section_id)` and `.inverse_points(points, section_id)` for tests and materialization.
- Produces `raw_crop_local_to_v6(row, crop_xy_um)` with bbox applied exactly once.

- [ ] **Step 1: Write failing tests** for section predicates, endpoint identity, affine composition order, and raw crop coordinate mapping.
- [ ] **Step 2: Run the focused tests and verify they fail because the new module is absent.
- [ ] **Step 3: Implement the minimal reusable transform wrapper using the existing v9 `LatestAlignmentChain` and new 51→53 `EdgeAlignment`.
- [ ] **Step 4: Run the focused tests and verify they pass.

### Task 2: Run direct STalign for 51→53

**Files:**
- Create: `scripts/run_9957_direct_51_53_stalign.py`
- Output: `data/9957_g0/stalign_direct_51_53_v1/`

**Interfaces:**
- Consumes v9 source rasters and v6 section transforms.
- Produces a saved STalign affine, forward/reverse maps, masks, summary, and red/green QC image with section 51 fixed and 53 moving.

- [ ] **Step 1:** Run Task 1 tests before using the new runner.
- [ ] **Step 2:** Run the official STalign edge on GPU with no extra flip unless the source metadata requires it.
- [ ] **Step 3:** Verify output contains fixed/moving masks, maps, affine, summary, and endpoint metadata.
- [ ] **Step 4:** Render and inspect 51–53 H&E red/green QC; record metrics without using them as the sole acceptance criterion.

### Task 3: Materialize the unified v11 coordinate frame

**Files:**
- Create: `scripts/materialize_9957_unified_51_53_v11.py`
- Output: `data/9957_g0/reconstruction_prep_v11_unified_51_53/`
- Output: `data/9957_g0/registration_correction_he_v11_unified_51_53/`

**Interfaces:**
- Consumes v6 source stores/H5AD, the new 51→53 edge, and the reviewed v9 53→55 chain.
- Produces unified H&E table and transforms, H5AD with active unified spatial coordinates, UNI2 store, nucleus store, anchor H5ADs, and manifest.

- [ ] **Step 1:** Add a dry-run/contract test that checks section 51 is unchanged, 53 is corrected, sections ≥55 use `new_51_to_53 ∘ v9_53_to_55`, section 54 is absent, and all frames match.
- [ ] **Step 2:** Run materialization without overwriting existing outputs.
- [ ] **Step 3:** Validate H5AD, table, feature-store, nucleus-store, and transform-frame metadata.

### Task 4: Generate final pairwise QC and handoff metadata

**Files:**
- Create: `scripts/render_9957_unified_51_53_qc.py`
- Output: `data/9957_g0/qc/he_alignment_all_sections_v11_unified_51_53/`

**Interfaces:**
- Consumes only v11 outputs.
- Produces actual H&E red/green images for 51–53 and 53–55, transform audit JSON, and a compact QC manifest.

- [ ] **Step 1:** Run QC renderer.
- [ ] **Step 2:** Verify pair endpoints and coordinate-frame equality.
- [ ] **Step 3:** Report image paths and any remaining review-required status before any downstream model retraining.
