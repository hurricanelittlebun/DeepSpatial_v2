import numpy as np


def test_candidate_coordinate_loader_reads_corrected_branch(tmp_path):
    from deepspatial.histology.coordinate_validation import (
        load_candidate_coordinates,
    )

    path = tmp_path / "section-091__st_corrected_candidate.npz"
    corrected = np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
    np.savez_compressed(
        path,
        cell_id=np.asarray(["cell-1", "cell-2"]),
        spatial_registered_original=corrected - 1.0,
        spatial_st_corrected=corrected,
    )

    result = load_candidate_coordinates(path, expected_n=2)

    np.testing.assert_allclose(result, corrected)


def test_local_um_maps_to_pixel_centers_with_effective_origin_offset():
    from deepspatial.histology.coordinate_validation import (
        local_um_to_output_pixel_centers,
    )

    points = local_um_to_output_pixel_centers(
        np.asarray([[0.0, 0.0], [1.1213235294117647, 2.2426470588235294]]),
        local_origin_level0=(2131.0, 49196.0),
        output_origin_level0=(2128.0, 49192.0),
        raw_mpp_um=0.1401654411764706,
        output_downsample=8.0,
    )

    # The requested crop starts 3 x 4 level-0 pixels after the actual read
    # origin; the returned values are array pixel-center coordinates.
    np.testing.assert_allclose(points[0], [-0.125, 0.0], atol=1e-6)
    np.testing.assert_allclose(points[1], [0.875, 2.0], atol=1e-6)


def test_identity_when_requested_and_read_origins_match():
    from deepspatial.histology.coordinate_validation import (
        local_um_to_output_pixel_centers,
    )

    result = local_um_to_output_pixel_centers(
        np.asarray([[0.0, 0.0], [8.0 * 0.1401654411764706, 16.0 * 0.1401654411764706]]),
        local_origin_level0=(100.0, 200.0),
        output_origin_level0=(100.0, 200.0),
        raw_mpp_um=0.1401654411764706,
        output_downsample=8.0,
    )
    np.testing.assert_allclose(result, [[-0.5, -0.5], [0.5, 1.5]], atol=1e-6)
