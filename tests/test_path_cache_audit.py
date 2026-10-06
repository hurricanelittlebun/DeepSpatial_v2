import numpy as np
import pytest


def _make_cache(tmp_path):
    from deepspatial.histology import FeatureStore, MorphologyPathCache

    store = FeatureStore(tmp_path / "features.h5", mode="a")
    for section_id, z_um in [("1", 0.0), ("2", 5.0)]:
        store.add_section(
            section_id,
            np.ones((2, 2, 3), dtype=np.float32),
            z_um=z_um,
            origin_um=(0.0, 0.0),
            spacing_um=(1.0, 1.0),
            patch_size_um=2.0,
            mpp=(0.5, 0.5),
            coordinate_frame="fixed",
        )
    cache_path = tmp_path / "paths.h5"
    cache = MorphologyPathCache(
        cache_path,
        store,
        coordinate_frame="fixed",
        candidate_radius_um=0.0,
    )
    cache.prepare("1", "2", [[0.0, 0.0]], [[1.0, 1.0]], ["p"])
    return cache_path


def test_audit_path_cache_reports_frame_and_paths(tmp_path):
    from scripts.audit_00029_fixed_he_path_cache import audit_path_cache

    result = audit_path_cache(_make_cache(tmp_path), expected_frame="fixed")
    assert result["coordinate_frame"] == "fixed"
    assert result["path_count"] == 1
    assert result["morphology_fallback_path_count"] == 0


def test_audit_path_cache_rejects_wrong_frame(tmp_path):
    from scripts.audit_00029_fixed_he_path_cache import audit_path_cache

    with pytest.raises(ValueError, match="coordinate frame"):
        audit_path_cache(_make_cache(tmp_path), expected_frame="registered")
