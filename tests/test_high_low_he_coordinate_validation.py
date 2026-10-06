import numpy as np


def test_effective_level0_origin_uses_reader_flooring():
    from deepspatial.histology.coordinate_validation import effective_level0_origin

    assert effective_level0_origin((2131, 49196), 16.0) == (2128, 49184)


def test_output_centers_map_to_source_crop_coordinates():
    from deepspatial.histology.coordinate_validation import output_to_source_pixel_centers

    points = output_to_source_pixel_centers(
        output_shape=(1, 2),
        output_origin_level0=(2128, 49184),
        output_downsample=16.0,
        source_origin_level0=(2128, 49192),
        source_downsample=8.0,
    )

    expected = np.asarray(
        [
            [(16.0 - 8.0) / 8.0 - 0.5, 0.0 / 8.0 - 0.5],
            [(32.0 - 8.0) / 8.0 - 0.5, 0.0 / 8.0 - 0.5],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(points[0], expected, atol=1e-6)
