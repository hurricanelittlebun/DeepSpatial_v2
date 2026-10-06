import h5py
import numpy as np


def _store(path):
    from deepspatial.histology import FeatureStore

    store = FeatureStore(path, mode="a")
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
    return store


def test_path_cache_records_coordinate_frame_in_file_and_path_metadata(tmp_path):
    from deepspatial.histology import MorphologyPathCache

    store = _store(tmp_path / "features.h5")
    cache_path = tmp_path / "paths.h5"
    cache = MorphologyPathCache(
        cache_path,
        store,
        candidate_radius_um=0.0,
        coordinate_frame="fixed",
    )
    ids = cache.prepare("1", "2", [[0.0, 0.0]], [[1.0, 1.0]], ["p"])

    with h5py.File(cache_path, "r") as handle:
        assert handle.attrs["coordinate_frame"] == "fixed"
        metadata = handle["paths"][ids[0]].attrs["metadata"]
        assert '"coordinate_frame": "fixed"' in metadata
