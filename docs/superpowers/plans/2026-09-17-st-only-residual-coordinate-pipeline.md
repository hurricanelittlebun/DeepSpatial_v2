# ST-Only Residual Coordinate Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a new 00029/g0 DeepSpatial-ready dataset in which the existing registered H&E coordinate system remains fixed and residual affine correction is materialized only into Xenium ST anchor coordinates.

**Architecture:** The existing serial H&E registration remains the sole transformation applied to H&E images, masks, UNI2 grids, and nuclear centroids. Each ST anchor H5AD receives a new `obsm["spatial_st_corrected"]` calculated once from `spatial_registered` and its section-specific residual affine; original coordinates remain available for baseline comparison. UOT and morphology-path caches are rebuilt or materialized against the corrected ST endpoints and the unchanged registered-frame H&E stores.

**Tech Stack:** Python 3, AnnData/H5AD, NumPy, pandas, HDF5, existing DeepSpatial `FeatureStore`, sparse top-k UOT cache format, pytest.

**Spec:** `docs/morphology-plan.md` and the ST-only residual-coordinate requirements supplied in the user request.

## Global Constraints

- Do not overwrite existing data or experiment directories.
- Keep `00029__g0_registered` as the canonical H&E coordinate frame.
- Apply residual affine only to ST anchor coordinates; never to H&E images, masks, H&E nuclear coordinates, or intermediate H&E sections.
- Preserve `spatial_registered` and add `spatial_st_corrected`; do not silently overwrite the original ST coordinates.
- DeepSpatial must receive `spatial_key="spatial_st_corrected"` explicitly for the corrected run.
- Do not apply the residual matrix again during training, UOT, path evaluation, or reconstruction.
- Store transform provenance and coordinate hashes in the generated manifest.
- Do not use Z-interpolated residual affine for H&E-only sections.
- Retain the original cell-type architecture and current no-celltype compatibility.

---

### Task 1: Define and test the ST-only coordinate contract

**Files:**
- Create: `tests/test_st_only_residual_dataset.py`
- Create: `scripts/build_00029_st_only_residual_dataset.py`

**Interfaces:**
- Produces `apply_st_residual(points, matrix) -> np.ndarray`.
- Produces `load_residual_matrix(path) -> np.ndarray`.
- Produces `materialize_anchor_h5ad(source_path: Path, destination_path: Path, matrix_path: Path, coordinate_frame: str) -> dict`.
- Produces `build_he_table_contract(he_table) -> pandas.DataFrame` with an explicit no-residual H&E policy.

- [ ] **Step 1: Write failing tests**

```python
def test_apply_st_residual_changes_st_only():
    result = apply_st_residual(
        np.array([[10.0, 20.0]]),
        np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -2.0], [0.0, 0.0, 1.0]]),
    )
    np.testing.assert_allclose(result, [[15.0, 18.0]])


def test_materialized_anchor_preserves_original_and_adds_corrected_key(tmp_path):
    source = make_tiny_anchor(tmp_path / "source.h5ad")
    matrix_path = write_matrix(tmp_path / "residual.npz")
    destination = tmp_path / "anchor.h5ad"
    record = materialize_anchor_h5ad(source, destination, matrix_path,
                                     coordinate_frame="00029__g0_registered")
    value = ad.read_h5ad(destination)
    np.testing.assert_allclose(value.obsm["spatial_registered"], [[10.0, 20.0]])
    np.testing.assert_allclose(value.obsm["spatial_st_corrected"], [[15.0, 18.0]])
    assert record["h_and_e_residual_applied"] is False
    assert value.uns["deepspatial_coordinate_contract"]["st_residual_applied"] is True


def test_he_contract_for_every_section_is_identity_residual_policy():
    table = pd.DataFrame({"section_id": [1, 3], "source_kind": ["xenium_companion_he", "serial_he"]})
    result = build_he_table_contract(table)
    assert result["residual_applied_to_he"].tolist() == [False, False]
    assert result["he_coordinate_policy"].tolist() == [
        "original_registered_only", "original_registered_only"
    ]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/data/buyonggan/DeepSpatial/.venv/bin/pytest tests/test_st_only_residual_dataset.py -q`

Expected: FAIL because the new ST-only materialization interfaces do not yet exist.

- [ ] **Step 3: Implement the minimal contract**

Implement matrix validation and homogeneous point transformation. Read each source anchor, verify `spatial_registered` shape and finiteness, compute the corrected array once, preserve all original `.obsm` entries, add `spatial_st_corrected`, and write `uns["deepspatial_coordinate_contract"]`. Add H&E manifest columns that explicitly state residual was not applied.

- [ ] **Step 4: Run the focused tests**

Run: `/data/buyonggan/DeepSpatial/.venv/bin/pytest tests/test_st_only_residual_dataset.py -q`

Expected: PASS.

- [ ] **Step 5: Run the existing coordinate tests**

Run: `/data/buyonggan/DeepSpatial/.venv/bin/pytest tests/test_registered_coordinates.py tests/test_fixed_frame_guards.py -q`

Expected: PASS with no regressions.

### Task 2: Add the non-destructive 00029/g0 dataset builder

**Files:**
- Modify: `scripts/build_00029_st_only_residual_dataset.py`
- Modify: `tests/test_st_only_residual_dataset.py`

**Interfaces:**
- CLI inputs: original anchors, residual transform directory, original H&E table, output directory, canonical frame.
- CLI outputs: `anchors/`, `he_sections.parquet`, `nuclei_registered.parquet`, `dataset_manifest.json`, and `README.md`.

- [ ] **Step 1: Add failing integration-contract tests**

```python
def test_builder_rejects_missing_anchor_residual(tmp_path):
    with pytest.raises(FileNotFoundError, match="section-001"):
        build_anchor_manifest(
            source_dir=tmp_path / "anchors",
            residual_dir=tmp_path / "residual",
            output_dir=tmp_path / "out",
            sections=[1],
            coordinate_frame="00029__g0_registered",
        )


def test_builder_marks_only_st_as_corrected(tmp_path):
    source_dir = make_tiny_anchor_directory(tmp_path / "anchors")
    residual_dir = make_tiny_residual_directory(tmp_path / "residual")
    he_table = pd.DataFrame({
        "section_id": [1],
        "source_kind": ["xenium_companion_he"],
        "z_um": [0.0],
        "coordinate_frame": ["00029__g0_registered"],
    })
    he_table_path = tmp_path / "he_sections.parquet"
    he_table.to_parquet(he_table_path)
    result = build_anchor_manifest(
        source_dir=source_dir,
        residual_dir=residual_dir,
        output_dir=tmp_path / "out",
        sections=[1],
        coordinate_frame="00029__g0_registered",
        he_table_path=he_table_path,
    )
    assert result["h_and_e_policy"] == "identity_residual_for_all_he_sections"
    assert result["st_policy"] == "materialized_residual_affine_per_anchor"
```

- [ ] **Step 2: Run the integration tests to verify the intended failures**

Run: `/data/buyonggan/DeepSpatial/.venv/bin/pytest tests/test_st_only_residual_dataset.py -q`

Expected: the new builder-level tests fail because the orchestration functions are not implemented.

- [ ] **Step 3: Implement the builder**

Use the original `data/00029_g0/anchors/section-*.h5ad` as the source, read each `matrix_current_g_to_fixed_he` from `registration_correction_v1/transforms`, materialize a new H5AD under `st_only_residual_v1/anchors`, and copy the original H&E table with explicit policy columns. Aggregate the high-resolution v3 `nuclei_registered.csv` files without applying residual. Record source paths, output paths, section counts, coordinate keys, matrix SHA-256 values, and per-section policy in the manifest.

- [ ] **Step 4: Run focused tests and a dry-run against the real inputs**

Run: `/data/buyonggan/DeepSpatial/.venv/bin/pytest tests/test_st_only_residual_dataset.py -q`

Run: `/data/buyonggan/DeepSpatial/.venv/bin/python scripts/build_00029_st_only_residual_dataset.py --help`

Expected: tests pass and the CLI prints its options without importing unavailable training dependencies.

### Task 3: Materialize the new ST-only anchor/H&E/nucleus data

**Files:**
- Generated: `data/00029_g0/st_only_residual_v1/`

**Interfaces:**
- Consumes original anchors, residual matrices, `he_sections.parquet`, and `nucleus_path_highres_rerun_v3/segmentation/`.
- Produces corrected ST H5ADs and unchanged registered-frame H&E/nuclear coordinate data.

- [ ] **Step 1: Run the builder into a new directory**

Run:

```bash
/data/buyonggan/DeepSpatial/.venv/bin/python scripts/build_00029_st_only_residual_dataset.py \
  --output-dir data/00029_g0/st_only_residual_v1 \
  --source-anchors data/00029_g0/anchors \
  --he-table data/00029_g0/he_sections.parquet \
  --residual-dir data/00029_g0/registration_correction_v1 \
  --nuclei-segmentation data/00029_g0/nucleus_path_highres_rerun_v3/segmentation \
  --coordinate-frame 00029__g0_registered
```

- [ ] **Step 2: Verify generated anchor invariants**

Run a Python check that all ten output H5ADs have equal `spatial_registered` and `spatial_st_corrected` row counts, finite `[N,2]` arrays, unchanged expression matrices, and the expected section IDs. Verify that the output H&E table has `residual_applied_to_he == False` for every row and that the aggregated nuclear table uses only `x_registered_um/y_registered_um`.

- [ ] **Step 3: Verify no existing directory was modified**

Run: `git status --short -- data/00029_g0 data/00029_g0/st_only_residual_v1`

Expected: only the new output directory is created; old candidate/fixed directories remain untouched.

### Task 4: Build a canonical no-residual H&E nuclear feature store

**Files:**
- Generated: `data/00029_g0/st_only_residual_v1/nucleus_features_registered/nucleus_features.h5`
- Generated: `data/00029_g0/st_only_residual_v1/nucleus_feature_manifest.json`

**Interfaces:**
- Consumes high-resolution v3 `nuclei_registered.csv` and the unchanged `uni2_features.h5` grid.
- Produces registered-frame nuclear descriptors with identity residual policy for all H&E sections.

- [ ] **Step 1: Materialize with the existing feature-grid implementation**

Run:

```bash
/data/buyonggan/DeepSpatial/.venv/bin/python scripts/build_00029_nucleus_feature_store.py \
  --he-table data/00029_g0/st_only_residual_v1/he_sections.parquet \
  --uni2-store data/00029_g0/uni2_features.h5 \
  --output data/00029_g0/st_only_residual_v1/nucleus_features_registered \
  --segmentation-source data/00029_g0/nucleus_path_highres_rerun_v3 \
  --stage materialize \
  --diameter-um 8 \
  --radius-um 56
```

- [ ] **Step 2: Verify feature-store frame and section identity**

Run a check that every section in the new nuclear store has coordinate frame `00029__g0_registered`, the same section/Z/grid geometry as `uni2_features.h5`, and no metadata entry contains `interpolated_affine` or `residual_affine` applied to H&E.

### Task 5: Create canonical corrected UOT and morphology path caches

**Files:**
- Create: `scripts/materialize_00029_st_only_uot_cache.py`
- Create: `tests/test_st_only_uot_cache.py`
- Generated: `data/00029_g0/st_only_residual_v1/uot_cache_topk64/`
- Generated: `data/00029_g0/st_only_residual_v1/paths.h5`

**Interfaces:**
- `materialize_verified_uot_cache(source_dir, anchor_dir, output_dir, coordinate_frame, feature_revision) -> dict`.
- Consumes corrected ST H5ADs and the previously computed sparse coupling arrays only after shape/key/revision validation.
- Adds canonical coordinate-frame metadata and does not recompute or alter coupling values.

- [ ] **Step 1: Write failing cache-contract tests**

```python
def test_uot_materializer_adds_canonical_frame_without_changing_edges(tmp_path):
    source = make_tiny_uot_cache(tmp_path / "source", metadata={"spatial_key": "spatial_st_corrected"})
    result = materialize_verified_uot_cache(
        source, anchor_dir=tmp_path / "anchors", output_dir=tmp_path / "out",
        coordinate_frame="00029__g0_registered", feature_revision="rev1",
    )
    old = load_sparse_coupling(source / "pair-001-011.npz")[0]
    new = load_sparse_coupling(tmp_path / "out/pair-001-011.npz")[0]
    np.testing.assert_array_equal(old.row, new.row)
    np.testing.assert_array_equal(old.col, new.col)
    np.testing.assert_allclose(old.mass, new.mass)
    assert result["coordinate_frame"] == "00029__g0_registered"
```

- [ ] **Step 2: Run the cache test to verify it fails**

Run: `/data/buyonggan/DeepSpatial/.venv/bin/pytest tests/test_st_only_uot_cache.py -q`

Expected: FAIL because the cache materializer does not yet exist.

- [ ] **Step 3: Implement verified cache materialization**

Validate every pair against corrected anchor sizes, `spatial_key == "spatial_st_corrected"`, sparse top-k metadata, and the original UNI2 feature revision. Save a new NPZ with identical row/col/mass arrays and enriched metadata including coordinate frame, source cache, source anchor hashes, and `h_and_e_residual_applied: false`.

- [ ] **Step 4: Run the focused cache tests**

Run: `/data/buyonggan/DeepSpatial/.venv/bin/pytest tests/test_st_only_uot_cache.py -q`

Expected: PASS.

- [ ] **Step 5: Materialize the real verified UOT cache**

Run the new cache materializer using `data/00029_g0/uot_cache_topk64_st_corrected_v1` as source and the new anchor directory as the validation target.

- [ ] **Step 6: Build the new path cache**

Run:

```bash
/data/buyonggan/DeepSpatial/.venv/bin/python scripts/build_00029_corrected_path_cache.py \
  --anchor-dir data/00029_g0/st_only_residual_v1/anchors \
  --uot-dir data/00029_g0/st_only_residual_v1/uot_cache_topk64 \
  --feature-store data/00029_g0/uni2_features.h5 \
  --nucleus-feature-store data/00029_g0/st_only_residual_v1/nucleus_features_registered/nucleus_features.h5 \
  --output data/00029_g0/st_only_residual_v1/paths.h5 \
  --pairs 20000 \
  --candidate-radius-um 100 \
  --candidate-spacing-um 25 \
  --move-weight 0.001 \
  --morphology-weight 1.0 \
  --nucleus-weight 1.0 \
  --coordinate-frame 00029__g0_registered \
  --allow-linear-fallback
```

The path builder uses corrected ST endpoints but unchanged registered-frame H&E and nuclear feature stores. It must not receive `fixed_he_feature_frame_v1/v2_interpolated`.

### Task 6: Produce the DeepSpatial-ready configuration and final QC

**Files:**
- Create: `scripts/validate_00029_st_only_residual_dataset.py`
- Create: `tests/test_st_only_final_contract.py`
- Generated: `data/00029_g0/st_only_residual_v1/deepspatial_config.json`
- Generated: `data/00029_g0/st_only_residual_v1/qc/`

**Interfaces:**
- The final configuration explicitly sets `spatial_key="spatial_st_corrected"`, canonical frame `00029__g0_registered`, original `uni2_features.h5`, new registered nuclear store, and new path cache.
- QC compares corrected ST to unchanged H&E support and checks that no middle H&E row carries a residual transform.

- [ ] **Step 1: Add failing final-contract tests**

```python
def test_final_config_never_uses_interpolated_he_affine():
    config = make_final_config(
        anchor_dir="anchors",
        feature_path="../uni2_features.h5",
        nucleus_feature_path="nucleus_features_registered/nucleus_features.h5",
        path_cache="paths.h5",
        coordinate_frame="00029__g0_registered",
    )
    assert config["spatial_key"] == "spatial_st_corrected"
    assert config["coordinate_frame"] == "00029__g0_registered"
    assert config["he_policy"] == "unchanged_registered_he"
    assert config["serial_he_transform_policy"] == "identity_residual"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/data/buyonggan/DeepSpatial/.venv/bin/pytest tests/test_st_only_final_contract.py -q`

Expected: FAIL until the final configuration and validation functions exist.

- [ ] **Step 3: Implement final config/QC validation**

Check every H5AD, UOT pair, feature store, path cache, and manifest against the same canonical frame. Record counts, coordinate hashes, residual matrix hashes, UOT edge counts, path count, fallback count, and H&E policy. Render a compact before/after ST-on-unchanged-H&E QC for the ten anchors.

- [ ] **Step 4: Run all relevant tests**

Run: `/data/buyonggan/DeepSpatial/.venv/bin/pytest tests -q`

Expected: all tests pass, or any pre-existing unrelated failure is reported separately with its exact traceback.

- [ ] **Step 5: Run the final dataset validator**

Run the validator against `data/00029_g0/st_only_residual_v1` and inspect its JSON report and QC images before calling the dataset ready.
