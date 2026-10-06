# 00029/g0 fixed-H&E frame verification

**Date:** 2026-09-16

## Purpose

This run tests whether the earlier degradation was caused by querying H&E feature
fields in `00029__g0_registered` while the corrected Xenium anchors were in a
different residual-affine frame.

## Coordinate-frame correction

The original stores remain unchanged:

- `data/00029_g0/uni2_features_raw_corrected.h5`
- `data/00029_g0/nucleus_path_raw_final/nucleus_features.h5`

Both original stores are in `00029__g0_registered`. The new stores are:

- `data/00029_g0/fixed_he_feature_frame_v2_interpolated/uni2_features_fixed.h5`
- `data/00029_g0/fixed_he_feature_frame_v2_interpolated/nucleus_features_fixed.h5`

They contain all 82 H&E sections in `00029__g0_fixed_he`. Companion sections use
their saved residual affine; serial-only sections use the affine interpolated in
physical Z between adjacent Xenium companion anchors. Feature queries map fixed
coordinates back to the source field and use bilinear interpolation.

The new UOT cache uses `spatial_st_corrected` and the new fixed-frame UNI2 store:

`data/00029_g0/uot_cache_topk64_st_corrected_fixed_he_v2/`

The new path cache and audit are:

- `data/00029_g0/train_full_00029_st_corrected_fixed_he_v2/paths.h5`
- `data/00029_g0/train_full_00029_st_corrected_fixed_he_v2/path_cache_audit.json`

## Path integrity

- Coordinate frame: `00029__g0_fixed_he`
- Paths: 39,946
- Morphology fallback: 1,436 (3.59%)
- Nuclear fallback: 720 (1.80%)
- Maximum endpoint position error: 0.0 µm
- Maximum endpoint `t` error: 0.0
- Path versions: all version 2

Thus the rebuilt cache does not mix registered and corrected frames, and its
endpoints remain anchored exactly.

## Holdout result

Experiment:

`data/00029_g0/ablation_inner031_nucleus_path_fixed_he_v2/`

Configuration: 10 epochs, sparse top-k 64 UOT, section 31 held out, corrected
coordinates, fixed-frame UNI2/nucleus stores, morphology path enabled, nucleus
path enabled, spatial-head late fusion disabled, and cell-type branch disabled for
this existing 00029/g0 ablation input.

| Metric | Fixed-frame v1 | Fixed-frame v2 | Change v2 - v1 |
|---|---:|---:|---:|
| Profile Pearson | 0.51411 | 0.59102 | +0.07691 |
| Macro gene Pearson | 0.001761 | 0.001915 | +0.000153 |
| MAE | 0.75766 | 0.79403 | +0.03637 |
| RMSE | 1.07055 | 1.06683 | -0.00373 |

The v2 profile correlation improvement supports the hypothesis that the v1
identity treatment of the 72 serial-only sections introduced discontinuities
between independently corrected anchor sections. However, MAE did not improve,
so coordinate-frame inconsistency was an important cause but not the only source
of error. The older raw/nucleus experiments are not controlled baselines because
their feature stores, path caches, and sampled inputs differ; they should not be
used to claim a final model improvement.

## Remaining limitations

1. The residual affines are independently estimated at Xenium companion
   sections; Z interpolation makes the frame continuous but does not prove that
   every local tissue region is anatomically optimal.
2. About 3.6% of paths still use morphology fallback and about 1.8% use nuclear
   fallback. These are now explicit path-quality flags rather than silent frame
   mixing.
3. A controlled comparison against DeepSpatial 1.0 still requires identical
   split, pair sampling, training budget, and evaluation code.
