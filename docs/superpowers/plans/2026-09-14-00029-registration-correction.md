# 00029/g0 Registration Coverage Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Diagnose and produce non-destructive candidate corrections for H&E mask/section registration coverage in `00029/g0`, prioritizing anchor sections 71, 81, and 91.

**Architecture:** Use the existing saved STalign transform as the base. First map registered ST centroids back to each source H&E crop and generate visual overlays with mask inclusion status. Only when the evidence shows a systematic residual displacement will a residual affine candidate be written; otherwise preserve the transform and write a mask candidate. All candidates, metrics, and provenance are stored under a new correction directory and are not used to rebuild features until verified.

**Tech Stack:** Python, AnnData, pandas/Parquet, PIL, NumPy, Matplotlib, existing `SavedTransformStore`, existing registered-coordinate utilities.

**Spec:** In-chat approved workflow from 2026-09-14.

## Global Constraints

- Never overwrite the current registration transforms, masks, feature store, path cache, or reconstruction.
- Do not alter saved STalign nonlinear fields; residual corrections are separate artifacts.
- Do not use ST points alone to fabricate tissue masks.
- Every correction must have before/after coverage metrics and a visual QC output.

### Task 1: Generate anchor diagnostic overlays

**Files:**
- Create: `data/00029_g0/registration_correction_v1/diagnostic_overlays/`
- Create: `data/00029_g0/registration_correction_v1/diagnostic_metrics.json`

- [x] Map all anchor ST centroids through the inverse saved transform to source H&E crop coordinates.
- [x] Render each anchor with original H&E mask, registered ST points, in-mask points, and out-of-mask points.
- [x] Record transform status/confidence, crop inclusion, mask inclusion, and feature-query validity.
- [x] Inspect sections 71, 81, and 91 before selecting a correction.

### Task 2: Produce conservative correction candidates

**Files:**
- Create: `data/00029_g0/registration_correction_v1/masks/`
- Create: `data/00029_g0/registration_correction_v1/transforms/`
- Create: `data/00029_g0/registration_correction_v1/correction_manifest.json`

- [x] Checked for omitted H&E tissue; no mask-only candidate was created because the out-of-mask ST points sampled background rather than omitted tissue.
- [x] If the mask is visually correct but ST points have systematic residual offset, estimate and save a residual affine candidate after the existing transform.
- [x] If neither condition is supported, mark the section as unresolved instead of forcing a correction.

### Task 3: Verify candidate coverage

**Files:**
- Create: `data/00029_g0/registration_correction_v1/verification_metrics.json`
- Create: `data/00029_g0/registration_correction_v1/before_after_coverage.png`

- [x] Recompute ST-in-mask coverage using only candidate artifacts.
- [x] Verify originals are unchanged by checksum/metadata comparison.
- [x] Report whether the candidate is safe for a new UNI2 feature extraction run; do not rebuild features in this task.
