# Fixed-H&E Feature Frame Design

**Date:** 2026-09-16

## Goal

Make the corrected Xenium anchor coordinates and the H&E morphology fields use one explicit physical XY coordinate frame before recomputing morphology-aware UOT, anatomical paths, and the 00029/g0 holdout experiment.

## Problem

The anchor candidates use `obsm["spatial_st_corrected"]`, which applies the saved per-section residual affine `matrix_current_g_to_fixed_he` after the original STalign transform. The current UNI2 and nuclear feature stores are still grids in `00029__g0_registered`. Querying those grids with corrected coordinates mixes two frames. Mask containment is therefore not sufficient evidence that the queried morphology belongs to the ST cell.

## Design

Use physical feature-field resampling rather than changing the training loop's coordinate semantics:

1. Read the existing UNI2 and nuclear HDF5 grids without modifying them.
2. For each of the ten Xenium companion sections, load `matrix_current_g_to_fixed_he`. For each of the 72 serial-only sections, use the identity transform because no section-specific residual candidate exists.
3. Define `p_fixed = M_current_to_fixed @ p_registered`.
4. Build an axis-aligned output grid in the fixed frame from the transformed source-grid centers. For every output center `p_fixed`, query the source grid at `M_current_to_fixed^-1 @ p_fixed` with bilinear interpolation and no nearest-valid recovery. Save the interpolated features and validity mask in a new HDF5 store whose `coordinate_frame` is `00029__g0_fixed_he`.
5. Apply the same matrices and interpolation rules independently to the UNI2 and nuclear fields, preserving section IDs, Z, spacing, physical patch FOV, provenance, source revisions, matrix hashes, and identity-section records.
6. Recompute sparse top-k morphology UOT using `spatial_st_corrected` and the fixed-frame UNI2 store. Rebuild the morphology/nucleus path cache from the fixed-frame stores. Run the same section-031 holdout using the new path cache and fixed-frame configuration.

The original H&E images, original STalign transforms, old feature stores, old UOT/path caches, and old checkpoints remain untouched. This change does not alter expression matrices or introduce H&E-to-gene regression.

### v2 continuity refinement

The first materialization, `fixed_he_feature_frame_v1`, used identity for
serial-only sections. That was useful as a conservative bridge, but it left
piecewise discontinuities between independently corrected Xenium companion
sections. The evaluated `fixed_he_feature_frame_v2_interpolated` variant instead
linearly interpolates the saved companion residual affines in physical Z and
uses those section-specific matrices for all 82 sections. The v1 artifacts are
retained for comparison; v2 is the current fixed-frame experiment.

## Validation

- Every new feature store section declares `00029__g0_fixed_he`.
- UNI2 and nuclear output grids have matching section IDs, Z, origins, spacings, and frame.
- Identity sections reproduce source values and validity at source grid centers.
- Anchor output values at transformed source centers agree with source values within interpolation tolerance.
- Corrected ST endpoint coordinates query the corresponding fixed-frame stores without frame errors.
- New UOT metadata references the fixed-frame UNI2 revision and corrected spatial key.
- New path metadata references both fixed-frame feature revisions and has exact endpoints.
- The existing test suite and new coordinate-bridge tests pass.
