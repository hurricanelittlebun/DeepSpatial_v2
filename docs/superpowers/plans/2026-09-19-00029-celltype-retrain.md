# 00029 Cell-Type Branch Retraining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Retrain the existing 00029 nucleus-guided morphology UOT+path model with the validated `assigned_celltype` annotations while preserving the registered coordinate frame and existing experiment as a separate baseline.

**Architecture:** Use the existing DeepSpatial cell-type branch without changing its network design. Materialize a labeled copy of the ten canonical registered anchor H5ADs by joining annotations on `source_cell_id`; train with `use_celltype=True`, `label_key="cell_class"`, 37 encoded categories, sparse top-k UOT, and the existing UNI2+nucleus path cache. Evaluate the held-out section 31 using both expression metrics and cell-type accuracy, then optionally use the resulting checkpoint for full reconstruction only after compatibility checks pass.

**Tech Stack:** AnnData/H5AD, NumPy, pandas, PyTorch, PyTorch Lightning, existing DeepSpatial histology/UOT/path modules.

**Spec:** User request: retrain the cell-type branch using `/data/buyonggan/DeepSpatial/data/00029_g0/adata_00029_niche_domain.h5ad`, with model/data compatibility validation and no overwrite of the existing final model.

## Global Constraints

- Keep the existing 5,001-gene order and `00029__g0_registered` coordinate frame.
- Join annotation labels by `source_cell_id`, never by row order.
- Preserve the existing nucleus-guided morphology UOT+path configuration.
- Use `assigned_celltype` as the sole model label and materialize it as `cell_class`.
- Preserve the existing no-celltype experiment and reconstruction outputs.
- Do not use `cell_labels`, because it is a unique cell identifier rather than a biological label.
- Treat `unassigned` as an explicit category and report it; do not silently drop it.

## Review Focus

- Annotation-to-anchor join: every anchor must receive exactly one label from the matching `source_cell_id`.
- Model dimension: checkpoint `num_classes` and saved category order must equal the observed label vocabulary.
- Coordinate compatibility: labeled anchors must retain the exact registered XY and physical Z values.
- Holdout integrity: section 31 expression must not enter training, while its labels may only be used for evaluation.
- Output isolation: new checkpoints/metrics must be written under a new experiment directory.

### Task 1: Materialize validated labeled anchors

**Files:**
- Create: `scripts/prepare_00029_celltype_anchors.py`
- Test: `tests/test_celltype_anchor_preparation.py`

**Interfaces:**
- Consumes: annotated combined anchor H5AD, canonical per-section anchor H5ADs.
- Produces: per-section labeled H5ADs with `obs["cell_class"]`, preserving all original matrices, coordinates, and metadata.

- [ ] Write tests for source-ID joins, missing labels, duplicated IDs, gene-order mismatch, coordinate mismatch, and successful category materialization.
- [ ] Run the tests and observe failure before implementation.
- [ ] Implement deterministic validation and materialization.
- [ ] Run focused tests to verify pass.
- [ ] Run the materializer on the real 00029 data into a new output directory.
- [ ] Verify all ten output files, 129,980 cells, 5,001 genes, zero missing labels, and exact coordinate equality.

### Task 2: Retrain and evaluate the cell-type branch

**Files:**
- Create: `scripts/run_00029_celltype_nucleus_uot_path.py`
- Test: `tests/test_celltype_training_entrypoint.py`

**Interfaces:**
- Consumes: Task 1 labeled anchors and existing UNI2/nucleus/path stores.
- Produces: new experiment manifest, model config/checkpoint, expression holdout metrics, and cell-type holdout metrics.

- [ ] Test that the entrypoint declares `use_celltype=True`, `label_key="cell_class"`, and does not reuse the old no-celltype output directory.
- [ ] Run the focused entrypoint test and observe failure before implementation.
- [ ] Implement the holdout protocol matching the existing nucleus-path experiment: train sections 1, 11, 21, 41, 61, 71, 81, 91 and hold out section 31.
- [ ] Configure sparse top-k UOT (`top_k=64`), bidirectional candidates, existing feature stores, existing path cache, and nucleus path.
- [ ] Train for the selected production epoch count with GPU acceleration and save checkpoint/config.
- [ ] Evaluate section 31 expression reconstruction and predicted `cell_class` accuracy/macro-F1.
- [ ] Verify checkpoint metadata reports the correct category vocabulary and `use_celltype=True`.

### Task 3: Final verification and handoff

**Files:**
- Modify: new experiment output only
- Test: existing focused test suite plus real-data verification command

- [ ] Compare the cell-type retrained metrics against `st_only_residual_nucleus_uot_path_v1`.
- [ ] Confirm the old checkpoint and `adata_00029_full_thickness10.h5ad` are unchanged.
- [ ] Report whether the checkpoint is ready for full reconstruction; do not silently replace the existing reconstruction.
