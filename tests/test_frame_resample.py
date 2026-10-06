import h5py
import numpy as np
import pytest


def _make_store(path):
    from deepspatial.histology import FeatureStore

    store = FeatureStore(path, mode="a")
    yy, xx = np.mgrid[:3, :4]
    features = np.stack(
        [xx.astype(float), yy.astype(float), (xx + 2 * yy).astype(float)], axis=-1
    ).astype("float32")
    store.add_section(
        "1",
        features,
        z_um=0.0,
        origin_um=(0.0, 0.0),
        spacing_um=(1.0, 1.0),
        patch_size_um=2.0,
        mpp=(0.5, 0.5),
        coordinate_frame="registered",
        valid_mask=np.ones((3, 4), dtype=bool),
    )
    return store


def test_transform_points_applies_current_to_fixed_affine():
    from deepspatial.histology.frame_resample import transform_points

    matrix = np.array(
        [[2.0, 0.0, 10.0], [0.0, 3.0, -4.0], [0.0, 0.0, 1.0]]
    )
    points = transform_points([[1.0, 2.0], [0.0, 0.0]], matrix)
    np.testing.assert_allclose(points, [[12.0, 2.0], [10.0, -4.0]])


def test_identity_resampling_preserves_source_grid(tmp_path):
    from deepspatial.histology import FeatureStore
    from deepspatial.histology.frame_resample import resample_feature_store

    source = _make_store(tmp_path / "source.h5")
    output = tmp_path / "identity.h5"
    result = resample_feature_store(
        source.path,
        output,
        {"1": np.eye(3)},
        output_frame="fixed",
        provenance={"test": "identity"},
    )

    assert result["n_sections"] == 1
    fixed = FeatureStore(output)
    np.testing.assert_allclose(
        fixed.get_feature("1", [[0.0, 0.0], [3.0, 2.0]]).numpy(),
        [[0.0, 0.0, 0.0], [3.0, 2.0, 7.0]],
    )
    with h5py.File(output, "r") as handle:
        metadata = handle["sections"][fixed._key("1")].attrs["metadata"]
        assert '"coordinate_frame": "fixed"' in metadata


def test_translation_resampling_maps_fixed_grid_back_to_registered_source(tmp_path):
    from deepspatial.histology import FeatureStore
    from deepspatial.histology.frame_resample import resample_feature_store

    source = _make_store(tmp_path / "source.h5")
    output = tmp_path / "translated.h5"
    matrix = np.array(
        [[1.0, 0.0, 10.0], [0.0, 1.0, 20.0], [0.0, 0.0, 1.0]]
    )
    resample_feature_store(
        source.path,
        output,
        {"1": matrix},
        output_frame="fixed",
        provenance={"test": "translation"},
    )

    fixed = FeatureStore(output)
    # p_fixed=(11,22) must use p_registered=(1,2), whose known field is [1,2,5].
    value, valid = fixed.get_feature("1", [[11.0, 22.0]], return_valid=True)
    assert valid.tolist() == [True]
    np.testing.assert_allclose(value.numpy(), [[1.0, 2.0, 5.0]])


def test_resampling_rejects_singular_affine(tmp_path):
    from deepspatial.histology.frame_resample import resample_feature_store

    source = _make_store(tmp_path / "source.h5")
    with pytest.raises(ValueError, match="invertible"):
        resample_feature_store(
            source.path,
            tmp_path / "bad.h5",
            {"1": np.array(
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            )},
            output_frame="fixed",
            provenance={},
        )
