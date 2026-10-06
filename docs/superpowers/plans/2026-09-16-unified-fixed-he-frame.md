# Unified Fixed H&E Coordinate Frame Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put the 00029/g0 high-resolution H&E provenance, nuclear detections, corrected ST anchors, and morphology feature stores into one explicit `00029__g0_fixed_he` physical coordinate frame without overwriting source data.

**Architecture:** Keep native high-resolution H&E images and instance masks in their source pixel frames, and materialize transformed nuclear centroids/tables in the fixed H&E frame. Use the existing residual affine transforms at Xenium companion sections and the existing Z-interpolated section transforms for serial sections. Treat `spatial_st_corrected` and the fixed-frame feature stores as the downstream reference branch.

**Tech Stack:** Python, NumPy, pandas/Parquet, HDF5/Zarr feature stores, existing DeepSpatial registration utilities, pytest, Matplotlib for QC.

**Spec:** User request in the current conversation: unify high-resolution H&E, ST, and nuclear segmentation coordinates.

## Global Constraints

- Preserve all original H&E, masks, H5ADs, STalign maps, and old registered-frame stores.
- Use `00029__g0_fixed_he` as the downstream physical coordinate frame.
- Use `spatial_st_corrected` for corrected ST anchors.
- Do not rerun Cellpose unless coordinate-chain validation fails.
- Keep the native H&E image/mask pixel coordinates and the fixed-frame physical coordinates both auditable.
- Generate per-section and aggregate QC outputs.

### Task 1: Verify the coordinate contract

**Files:**
- Create: `tests/test_fixed_he_single_cell_coordinates.py`
- Inspect: `data/00029_g0/registration_correction_v1/correction_manifest.json`
- Inspect: `data/00029_g0/fixed_he_feature_frame_v2_interpolated/materialization_manifest.json`

**Interfaces:**
- Consumes: residual section transforms, H&E section manifest, raw nuclear tables, candidate ST coordinates.
- Produces: tested transform direction and fixed-frame column names.

- [ ] **Step 1: Write the failing test**

```python
def test_fixed_frame_transform_maps_registered_points_with_homogeneous_matrix():
    point = np.asarray([[10.0, 20.0]])
    matrix = np.asarray([[2.0, 0.0, 3.0], [0.0, 2.0, 4.0], [0.0, 0.0, 1.0]])
    result = apply_homogeneous(point, matrix)
    np.testing.assert_allclose(result, [[23.0, 44.0]])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_fixed_he_single_cell_coordinates.py::test_fixed_frame_transform_maps_registered_points_with_homogeneous_matrix -q`

Expected: FAIL because the tested helper is not yet exposed by the new fixed-frame utility.

- [ ] **Step 3: Implement the minimal transform helper**

Add a dependency-light helper that validates finite `N x 2` points and applies a `3 x 3` homogeneous matrix in the documented direction `p_fixed = M @ p_registered`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_fixed_he_single_cell_coordinates.py::test_fixed_frame_transform_maps_registered_points_with_homogeneous_matrix -q`

Expected: PASS.

### Task 2: Materialize per-nucleus fixed-frame coordinates

**Files:**
- Create: `scripts/materialize_00029_fixed_nucleus_coordinates.py`
- Test: `tests/test_fixed_he_single_cell_coordinates.py`
- Output: `data/00029_g0/fixed_he_single_cell_v1/`

**Interfaces:**
- Consumes: `nucleus_path_raw_v1_coordinate_corrected_v2/segmentation/section-*/nuclei_registered.csv`, `he_sections.parquet`, and `registration_correction_v1` transforms.
- Produces: per-section `nuclei_fixed.csv`, consolidated `nuclei_fixed.parquet`, `x_fixed_he_um`, `y_fixed_he_um`, and a manifest recording source frame and transform provenance.

- [ ] **Step 1: Write the failing test**

```python
def test_materialized_nucleus_table_contains_fixed_coordinates(tmp_path):
    table = pd.DataFrame({"x_registered_um": [10.0], "y_registered_um": [20.0]})
    matrix = np.asarray([[2.0, 0.0, 3.0], [0.0, 2.0, 4.0], [0.0, 0.0, 1.0]])
    result = add_fixed_coordinates(table, matrix)
    assert list(result[["x_fixed_he_um", "y_fixed_he_um"]].iloc[0]) == [23.0, 44.0]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_fixed_he_single_cell_coordinates.py::test_materialized_nucleus_table_contains_fixed_coordinates -q`

Expected: FAIL because the materialization helper does not yet exist.

- [ ] **Step 3: Implement the materialization script**

For each section, load the old registered nuclear table, choose the verified section-to-fixed matrix from the existing fixed-frame materialization logic, add `x_fixed_he_um` and `y_fixed_he_um`, retain all original columns, and write new CSV/Parquet outputs. Do not transform or overwrite `nuclei.tif`; record that its mask pixels remain in native H&E crop coordinates.

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_fixed_he_single_cell_coordinates.py -q`

Expected: PASS.

- [ ] **Step 5: Run the materialization on all 82 sections**

Run: `.venv/bin/python scripts/materialize_00029_fixed_nucleus_coordinates.py --output data/00029_g0/fixed_he_single_cell_v1`

Expected: 82 section tables, one consolidated Parquet table, and a manifest with `output_coordinate_frame = 00029__g0_fixed_he`.

### Task 3: Validate three-way consistency

**Files:**
- Create: `scripts/validate_00029_fixed_he_three_way_qc.py`
- Test: `tests/test_fixed_he_single_cell_coordinates.py`
- Output: `data/00029_g0/qc/fixed_he_three_way_v1/`

**Interfaces:**
- Consumes: candidate ST H5ADs, fixed nuclear tables, high-resolution H&E metadata, residual transforms, and fixed-frame UNI2/nuclear stores.
- Produces: per-anchor H&E/ST/nuclei overlays, coverage statistics, coordinate-frame report, and a machine-readable QC summary.

- [ ] **Step 1: Write the failing test**

```python
def test_candidate_st_and_fixed_nucleus_tables_declare_same_frame():
    assert candidate_frame == "00029__g0_fixed_he"
    assert nucleus_frame == "00029__g0_fixed_he"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_fixed_he_single_cell_coordinates.py::test_candidate_st_and_fixed_nucleus_tables_declare_same_frame -q`

Expected: FAIL until the new manifest and validation report are created.

- [ ] **Step 3: Implement the validation and QC**

Check section IDs, row counts, finite coordinates, fixed-frame metadata, ST candidate presence, nucleus-to-H&E source mapping, and fixed feature-store metadata. Render at least sections 001, 011, 071, 091 and an aggregate overview with ST points and nuclear centroids.

- [ ] **Step 4: Run the test and validation**

Run: `pytest tests/test_fixed_he_single_cell_coordinates.py -q`

Run: `.venv/bin/python scripts/validate_00029_fixed_he_three_way_qc.py --output-dir data/00029_g0/qc/fixed_he_three_way_v1`

Expected: all 82 sections pass structural checks; any visual registration concern is reported rather than silently discarded.

### Task 4: Final downstream pointer and regression verification

**Files:**
- Modify: `data/00029_g0/fixed_he_single_cell_v1/README.md`
- Test: `tests/test_fixed_he_single_cell_coordinates.py`

- [ ] **Step 1: Record the downstream paths**

Document that corrected ST uses `anchors_st_to_fixed_he_candidate/*h5ad` with `obsm["spatial_st_corrected"]`, nuclear single-cell coordinates use `fixed_he_single_cell_v1/nuclei_fixed.parquet`, and morphology grids use `fixed_he_feature_frame_v2_interpolated/`.

- [ ] **Step 2: Run regression verification**

Run: `.venv/bin/python -m pytest -q tests/test_highres_coordinate_validation.py tests/test_registered_coordinates.py tests/test_high_low_he_coordinate_validation.py tests/test_fixed_he_single_cell_coordinates.py`

Run: `git diff --check`

Expected: all selected tests pass and no whitespace errors are reported.
