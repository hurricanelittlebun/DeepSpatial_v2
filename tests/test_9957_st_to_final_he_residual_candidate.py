import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from fit_9957_st_to_final_he_residual_candidate import (  # noqa: E402
    apply_homogeneous,
    evaluate_local_residual_candidate,
    mask_alignment_score,
    rasterize_forward_mask,
    residual_matrix,
)


def test_centered_residual_translation_improves_fixed_mask_score():
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:80, 20:80] = True
    points = np.array([[14.0, 50.0], [16.0, 50.0], [18.0, 50.0]])
    center = points.mean(axis=0)

    baseline = mask_alignment_score(points, mask, pixel_size_um=(1.0, 1.0))
    matrix = residual_matrix(np.array([0.0, 0.0, 0.0, 6.0, 0.0]), center)
    corrected = apply_homogeneous(points, matrix)
    candidate = mask_alignment_score(corrected, mask, pixel_size_um=(1.0, 1.0))

    assert candidate["inside_fraction"] > baseline["inside_fraction"]
    assert candidate["median_signed_distance_um"] > baseline["median_signed_distance_um"]


def test_residual_matrix_keeps_rotation_scale_centered():
    points = np.array([[10.0, 20.0], [30.0, 40.0]])
    center = points.mean(axis=0)
    matrix = residual_matrix(np.array([0.1, 0.02, -0.03, 0.0, 0.0]), center)
    transformed = apply_homogeneous(points, matrix)

    np.testing.assert_allclose(transformed.mean(axis=0), center, atol=1e-10)
    np.testing.assert_allclose(matrix[2], [0.0, 0.0, 1.0])


def test_candidate_is_scored_in_fixed_he_local_frame():
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:80, 20:80] = True
    local_points = np.array([[14.0, 50.0], [16.0, 50.0], [18.0, 50.0]])
    center = local_points.mean(axis=0)
    result = evaluate_local_residual_candidate(
        np.array([0.0, 0.0, 0.0, 6.0, 0.0]),
        local_points,
        center,
        mask,
        pixel_size_um=(1.0, 1.0),
    )

    assert result["inside_valid_fraction"] == 1.0
    np.testing.assert_allclose(result["candidate_local"], local_points + [6.0, 0.0])


def test_forward_global_mask_rasterization_has_explicit_fixed_origin():
    forward_points = np.array(
        [[10.0, 20.0], [11.0, 20.0], [10.0, 21.0], [11.0, 21.0]]
    )
    mask, origin = rasterize_forward_mask(
        forward_points, scale_um=1.0, padding_um=1.0, dilation_iterations=0
    )

    assert mask.shape == (3, 3)
    np.testing.assert_allclose(origin, [9.0, 19.0])
    assert mask[1, 1]
    assert mask[2, 2]
