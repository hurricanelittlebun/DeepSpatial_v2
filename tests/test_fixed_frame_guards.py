import numpy as np
import pytest


def _store(path, frame):
    from deepspatial.histology import FeatureStore

    store = FeatureStore(path, mode="a")
    store.add_section(
        "1",
        np.ones((2, 2, 3), dtype=np.float32),
        z_um=0.0,
        origin_um=(0.0, 0.0),
        spacing_um=(1.0, 1.0),
        patch_size_um=2.0,
        mpp=(0.5, 0.5),
        coordinate_frame=frame,
    )
    return store


def test_uot_frame_guard_accepts_expected_frame(tmp_path):
    from examples.run_00029_uot_qc import _validate_feature_store_frame

    store = _store(tmp_path / "features.h5", "fixed")
    assert _validate_feature_store_frame(store, "fixed") == "fixed"


def test_uot_frame_guard_rejects_mixed_or_wrong_frame(tmp_path):
    from examples.run_00029_uot_qc import _validate_feature_store_frame

    store = _store(tmp_path / "features.h5", "registered")
    with pytest.raises(ValueError, match="coordinate frame"):
        _validate_feature_store_frame(store, "fixed")
