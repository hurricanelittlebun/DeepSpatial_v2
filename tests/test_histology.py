import numpy as np
import pytest
import torch


def make_store(tmp_path):
    from deepspatial.histology.feature_store import FeatureStore

    store = FeatureStore(tmp_path / "features.h5", mode="a")
    yy, xx = np.mgrid[:5, :5]
    for sid, z in [("a", 10.0), ("m", 13.0), ("b", 20.0)]:
        f = np.stack([xx + 2 * yy + z, np.ones_like(xx)], -1).astype("float32")
        store.add_section(
            sid,
            f,
            z_um=z,
            origin_um=(100.0, 200.0),
            spacing_um=(2.0, 3.0),
            patch_size_um=20.0,
            mpp=(0.5, 0.5),
            coordinate_frame="toy",
        )
    return store


def test_sdpc_level_selection_and_level0_crop_geometry():
    from deepspatial.histology.sdpc import (
        choose_level_for_mpp,
        level0_bbox_to_read_region,
        level_px_to_source_um,
    )

    level = choose_level_for_mpp((1.0, 2.0, 4.0, 8.0), 0.140165, 1.121324)
    assert level == 3

    location, size, effective_bbox = level0_bbox_to_read_region(
        (2131, 49196, 19093, 67544),
        downsample=8.0,
        level0_dimensions=(176256, 91392),
    )
    assert location == (2128, 49192)
    assert size == (2121, 2294)
    assert effective_bbox == (2128, 49192, 19096, 67544)

    source = level_px_to_source_um(
        [[0.0, 0.0], [1.0, 2.0]],
        level0_origin=(2128, 49192),
        downsample=8.0,
        mpp_level0=0.140165,
    )
    np.testing.assert_allclose(
        source,
        np.array([[2128, 49192], [2136, 49208]]) * 0.140165,
    )


def test_physical_bilinear_and_z_query(tmp_path):
    store = make_store(tmp_path)
    xy = torch.tensor([[101.0, 201.5], [104.0, 206.0]])
    np.testing.assert_allclose(store.get_feature("a", xy), [[11.5, 1], [16, 1]])
    np.testing.assert_allclose(
        store.query(xy, torch.tensor([15.0, 15.0])), [[16.5, 1], [21.0, 1]]
    )
    with pytest.raises(ValueError):
        store.get_feature("a", torch.tensor([[0.0, 0.0]]))


def test_registered_sdpc_precompute_offsets_raw_image_only_for_patch_reads(tmp_path, monkeypatch):
    from PIL import Image

    import deepspatial.histology.registered_preprocessing as preprocessing
    from deepspatial.histology import FeatureStore

    calls = []

    def fake_extract(reader, xy, **kwargs):
        coordinates = np.asarray(xy, dtype=float).copy()
        calls.append(coordinates)
        return [Image.new("RGB", (8, 8), "white") for _ in coordinates]

    monkeypatch.setattr(preprocessing, "extract_sdpc_patches", fake_extract)

    class Encoder:
        feature_dim = 1

        def encode(self, patches, batch_size):
            return torch.zeros((len(patches), 1), dtype=torch.float32)

    class Section:
        preorientation_matrix = np.eye(3)
        edge_chain = []

    class Reader:
        pass

    store = FeatureStore(tmp_path / "offset.h5", mode="a")
    mask = np.ones((3, 3), dtype=bool)
    preprocessing.precompute_registered_sdpc_section(
        Reader(),
        Encoder(),
        store,
        "1",
        section=Section(),
        transform_store=None,
        source_mask=mask,
        source_mask_mpp=(1.0, 1.0),
        source_mask_bbox_origin_px=(0.0, 0.0),
        z_um=0.0,
        grid_spacing_um=1.0,
        patch_size_um=2.0,
        target_mpp=0.5,
        coordinate_frame="toy",
        batch_size=8,
        source_image_offset_um=(10.0, 20.0),
    )

    assert calls
    coordinates = np.concatenate(calls, axis=0)
    assert np.all(coordinates >= np.array([10.0, 20.0]))


def test_mask_aware_bilinear_renormalizes_available_corners(tmp_path):
    """Catches rejecting an edge query merely because one corner is masked."""
    from deepspatial.histology import FeatureStore

    store = FeatureStore(tmp_path / "partial.h5", mode="a")
    features = np.array(
        [[[1.0, 0.0], [3.0, 0.0]], [[5.0, 0.0], [7.0, 0.0]]],
        dtype=np.float32,
    )
    store.add_section(
        "1", features, z_um=0, origin_um=(0, 0), spacing_um=(1, 1),
        patch_size_um=2, mpp=(0.5, 0.5), coordinate_frame="registered",
        valid_mask=np.array([[True, True], [False, False]]),
    )

    value, valid, method = store.get_feature(
        "1", [[0.5, 0.5]], return_quality=True, nearest_max_distance_um=0
    )

    np.testing.assert_allclose(value.numpy(), [[2.0, 0.0]])
    assert valid.tolist() == [True]
    assert method.tolist() == [2]  # partial mask-aware bilinear


def test_feature_query_uses_bounded_nearest_valid_fallback(tmp_path):
    """Catches returning zero when a valid morphology point is nearby."""
    from deepspatial.histology import FeatureStore

    store = FeatureStore(tmp_path / "nearest.h5", mode="a")
    features = np.zeros((3, 3, 2), dtype=np.float32)
    features[1, 1] = [9.0, 4.0]
    valid_mask = np.zeros((3, 3), dtype=bool)
    valid_mask[1, 1] = True
    store.add_section(
        "1", features, z_um=0, origin_um=(0, 0), spacing_um=(1, 1),
        patch_size_um=2, mpp=(0.5, 0.5), coordinate_frame="registered",
        valid_mask=valid_mask,
    )

    value, valid, method = store.get_feature(
        "1", [[0.0, 0.0], [-10.0, -10.0]], return_quality=True,
        nearest_max_distance_um=1.5,
    )

    np.testing.assert_allclose(value[0].numpy(), [9.0, 4.0])
    assert valid.tolist() == [True, False]
    assert method.tolist() == [3, 0]  # bounded nearest, then unavailable


def test_histology_runtime_neutralizes_unavailable_spatial_features(tmp_path):
    from deepspatial.histology import FeatureStore, HistologyConfig
    from deepspatial.histology.runtime import HistologyRuntime

    store = FeatureStore(tmp_path / "runtime.h5", mode="a")
    features = np.ones((3, 3, 2), dtype=np.float32)
    store.add_section(
        "1",
        features,
        z_um=0,
        origin_um=(0, 0),
        spacing_um=(1, 1),
        patch_size_um=2,
        mpp=(0.5, 0.5),
        coordinate_frame="toy",
        valid_mask=np.zeros((3, 3), dtype=bool),
    )
    runtime = HistologyRuntime(
        HistologyConfig(
            use_histology=True,
            feature_path=str(store.path),
            coordinate_frame="toy",
        ),
        {"x_min": 0.0, "x_range": 2.0, "y_min": 0.0, "y_range": 2.0, "z_min": 0.0, "z_range": 1.0},
    )

    output = runtime.features(torch.tensor([[1.0, 1.0]]), torch.tensor([[0.0]]))

    assert output.shape == (1, 2)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, torch.zeros_like(output))


def test_cost_baseline_and_morphology_coupling():
    from deepspatial.data_utils.uot_solver import (
        compute_cost_matrix,
        compute_uot_coupling,
    )

    x = np.array([[0.0, 0.0], [1.0, 0.0]])
    g = np.eye(2)
    c = np.eye(2)
    baseline = compute_cost_matrix(x, g, c, x, g, c)
    np.testing.assert_allclose(baseline, [[0.0, 11.0], [11.0, 0.0]], atol=1e-7)
    np.testing.assert_array_equal(
        baseline, compute_cost_matrix(x, g, c, x, g, c, histology_weight=0)
    )
    kw = dict(
        spatial_weight=0.0,
        gene_weight=0.0,
        class_weight=0.0,
        histology_weight=2.0,
        h0=g,
        uot_reg=0.1,
    )
    p = compute_uot_coupling(x, g, c, x, g, c, h1=g, **kw)
    q = compute_uot_coupling(x, g, c, x, g, c, h1=g[::-1].copy(), **kw)
    assert p.trace() > p.sum() * 0.9
    assert q.trace() < q.sum() * 0.1


def test_cached_curved_path_endpoints_and_velocity(tmp_path):
    from deepspatial.histology.feature_store import FeatureStore
    from deepspatial.histology.path import MorphologyPathCache

    store = FeatureStore(tmp_path / "corridor.h5", mode="a")
    for sid, z, y in [("a", 0.0, 2), ("m", 3.0, 4), ("b", 10.0, 2)]:
        f = np.zeros((7, 7, 2), dtype="float32")
        f[..., 1] = 1
        f[y, :, 0] = 1
        f[y, :, 1] = 0
        store.add_section(
            sid,
            f,
            z_um=z,
            origin_um=(0, 0),
            spacing_um=(1, 1),
            patch_size_um=1,
            mpp=(1, 1),
            coordinate_frame="toy",
        )
    cache = MorphologyPathCache(
        tmp_path / "paths.h5",
        store,
        candidate_radius_um=2.0,
        candidate_spacing_um=1.0,
        move_weight=0.01,
        morphology_weight=10.0,
    )
    ids = cache.prepare(
        "a", "b", np.array([[1.0, 2.0]]), np.array([[5.0, 2.0]]), ["pair"]
    )
    x0, _ = cache.evaluate(ids, torch.tensor([0.0]))
    x1, _ = cache.evaluate(ids, torch.tensor([1.0]))
    mid, _ = cache.evaluate(ids, torch.tensor([0.3]))
    np.testing.assert_allclose(x0, [[1, 2]], atol=1e-6)
    np.testing.assert_allclose(x1, [[5, 2]], atol=1e-6)
    assert mid[0, 1] > 3.5
    t = torch.tensor([0.6], dtype=torch.float64)
    eps = 1e-5
    _, v = cache.evaluate(ids, t)
    plus, _ = cache.evaluate(ids, t + eps)
    minus, _ = cache.evaluate(ids, t - eps)
    torch.testing.assert_close(v, (plus - minus) / (2 * eps), rtol=1e-5, atol=1e-5)
    assert (
        cache.prepare(
            "a", "b", np.array([[1.0, 2.0]]), np.array([[5.0, 2.0]]), ["pair"]
        )
        == ids
    )


def test_cached_path_keeps_endpoints_when_anchor_morphology_is_unavailable(tmp_path):
    """Anchor coordinates remain usable when the endpoint tissue mask is partial."""
    from deepspatial.histology.feature_store import FeatureStore
    from deepspatial.histology.path import MorphologyPathCache

    store = FeatureStore(tmp_path / "partial_path.h5", mode="a")
    features = np.zeros((5, 5, 2), dtype="float32")
    features[..., 0] = 1.0
    for sid, z, valid in [
        ("a", 0.0, np.zeros((5, 5), dtype=bool)),
        ("m", 5.0, np.ones((5, 5), dtype=bool)),
        ("b", 10.0, np.zeros((5, 5), dtype=bool)),
    ]:
        store.add_section(
            sid,
            features,
            z_um=z,
            origin_um=(0, 0),
            spacing_um=(1, 1),
            patch_size_um=5,
            mpp=(1, 1),
            coordinate_frame="toy",
            valid_mask=valid,
        )

    cache = MorphologyPathCache(
        tmp_path / "partial_paths.h5",
        store,
        candidate_radius_um=1.0,
        candidate_spacing_um=1.0,
    )
    ids = cache.prepare(
        "a", "b", np.array([[1.0, 1.0]]), np.array([[3.0, 1.0]]), ["pair"]
    )
    x0, _ = cache.evaluate(ids, torch.tensor([0.0]))
    x1, _ = cache.evaluate(ids, torch.tensor([1.0]))
    np.testing.assert_allclose(x0, [[1.0, 1.0]])
    np.testing.assert_allclose(x1, [[3.0, 1.0]])


def test_nucleus_descriptor_grid_is_local_and_marks_empty_support_invalid():
    from deepspatial.histology.nucleus import descriptor_grid_from_nuclei

    grid_x, grid_y = np.meshgrid(np.arange(5.0), np.arange(5.0))
    grid = np.stack([grid_x, grid_y], axis=-1)
    descriptors, valid = descriptor_grid_from_nuclei(
        np.array([[2.0, 2.0], [2.5, 2.0]]),
        np.array([20.0, 30.0]),
        np.array([0.2, 0.4]),
        grid,
        radius_um=0.75,
    )

    assert descriptors.shape[:2] == (5, 5)
    assert descriptors.shape[-1] >= 5
    assert valid.shape == (5, 5)
    assert valid[2, 2]
    assert not valid[0, 0]
    assert np.isfinite(descriptors).all()
    assert descriptors[2, 2, 0] > descriptors[0, 0, 0]


def test_nucleus_descriptor_requires_nuclear_support_inside_tissue():
    from deepspatial.histology.nucleus import descriptor_grid_from_nuclei

    grid_x, grid_y = np.meshgrid(np.arange(3.0), np.arange(3.0))
    grid = np.stack([grid_x, grid_y], axis=-1)
    descriptors, valid = descriptor_grid_from_nuclei(
        np.empty((0, 2)),
        np.empty(0),
        np.empty(0),
        grid,
        radius_um=1.0,
        tissue_mask=np.ones((3, 3), dtype=bool),
    )

    assert np.isfinite(descriptors).all()
    assert not valid.any()


def test_nuclei_table_from_mask_contains_geometry_and_score():
    from deepspatial.histology.nucleus import nuclei_table_from_mask

    labels = np.zeros((5, 6), dtype=np.uint32)
    labels[1:3, 2:5] = 1
    score = np.zeros_like(labels, dtype=np.float32)
    score[labels == 1] = 0.75
    table = nuclei_table_from_mask(
        labels,
        score,
        mpp_um=(0.5, 0.5),
        section_id="s1",
    )

    assert len(table) == 1
    row = table.iloc[0]
    assert row.cell_id == "s1_n1"
    assert row.area_um2 == pytest.approx(1.5)
    assert row.cellpose_score_raw == pytest.approx(0.75)
    assert not row.touches_image_edge


def test_nucleus_path_term_can_change_anatomical_route(tmp_path):
    from deepspatial.histology.feature_store import FeatureStore
    from deepspatial.histology.path import MorphologyPathCache

    def add_store(path, middle_row):
        store = FeatureStore(path, mode="a")
        for sid, z in [("a", 0.0), ("m", 5.0), ("b", 10.0)]:
            features = np.zeros((7, 7, 2), dtype="float32")
            features[..., 1] = 1.0
            row = 2 if sid != "m" else middle_row
            features[row, :, 0] = 1.0
            features[row, :, 1] = 0.0
            store.add_section(
                sid,
                features,
                z_um=z,
                origin_um=(0.0, 0.0),
                spacing_um=(1.0, 1.0),
                patch_size_um=2.0,
                mpp=(1.0, 1.0),
                coordinate_frame="toy",
            )
        return store

    uni2 = add_store(tmp_path / "uni2.h5", middle_row=2)
    nuclei = add_store(tmp_path / "nucleus.h5", middle_row=3)
    cache = MorphologyPathCache(
        tmp_path / "paths.h5",
        uni2,
        nucleus_store=nuclei,
        candidate_radius_um=2.0,
        candidate_spacing_um=1.0,
        move_weight=0.01,
        morphology_weight=10.0,
        nucleus_weight=20.0,
    )
    ids = cache.prepare(
        "a", "b", np.array([[1.0, 2.0]]), np.array([[5.0, 2.0]]), ["pair"]
    )
    mid, _ = cache.evaluate(ids, torch.tensor([0.5]))
    assert mid[0, 1] > 2.5


def test_nucleus_config_roundtrip_keeps_baseline_independent():
    from deepspatial.histology import HistologyConfig

    baseline = HistologyConfig().to_dict()
    nucleus = HistologyConfig(
        use_histology=True,
        use_morphology_uot=True,
        use_morphology_path=True,
        use_nucleus_path=True,
        feature_path="uni2.h5",
        nucleus_feature_path="nucleus.h5",
        path_cache="paths.h5",
        coordinate_frame="registered",
        nucleus_weight=1.0,
    ).to_dict()

    assert baseline["use_nucleus_path"] is False
    assert baseline["nucleus_feature_path"] is None
    assert nucleus["use_nucleus_path"] is True
    assert nucleus["nucleus_feature_path"] == "nucleus.h5"
    assert nucleus["nucleus_weight"] == 1.0
