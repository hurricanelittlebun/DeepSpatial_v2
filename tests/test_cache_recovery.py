import h5py


def test_unfinished_feature_write_not_treated_as_section(tmp_path):
    from deepspatial.histology import FeatureStore

    path = tmp_path / "f.h5"
    FeatureStore(path, mode="a")
    with h5py.File(path, "a") as f:
        f["sections"].create_group("_pending_interrupted")
    assert FeatureStore(path).sections == []
