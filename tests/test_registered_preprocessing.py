from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image


@dataclass(frozen=True)
class Section:
    preorientation_matrix: np.ndarray
    edge_chain: tuple = ()


class Store:
    def load_edge(self, key):  # pragma: no cover - identity has no edges
        raise AssertionError


class Reader:
    level_dimensions = ((20, 20),)
    level_downsamples = (1.0,)
    mpp_level0 = 1.0

    def read_region(self, location, level, size):
        return np.full((size[1], size[0], 3), 128, dtype=np.uint8)


class Encoder:
    feature_dim = 2

    def encode(self, patches, batch_size):
        return torch.ones((len(patches), 2))


def test_registered_sdpc_grid_is_queryable_in_registered_coordinates(tmp_path):
    from deepspatial.histology import FeatureStore
    from deepspatial.histology.registered_preprocessing import (
        precompute_registered_sdpc_section,
    )

    mask = np.ones((20, 20), dtype=bool)
    feature_store = FeatureStore(tmp_path / "features.h5", mode="a")
    precompute_registered_sdpc_section(
        Reader(),
        Encoder(),
        feature_store,
        "3",
        section=Section(np.eye(3)),
        transform_store=Store(),
        source_mask=mask,
        source_mask_mpp=(1.0, 1.0),
        source_mask_bbox_origin_px=(0.0, 0.0),
        z_um=10.0,
        grid_spacing_um=5.0,
        patch_size_um=4.0,
        target_mpp=1.0,
        coordinate_frame="toy_registered",
        batch_size=2,
    )
    features, valid = feature_store.get_feature(
        "3", torch.tensor([[5.0, 5.0], [10.0, 10.0]]), return_valid=True
    )
    assert valid.all()
    np.testing.assert_allclose(features.numpy(), 1.0)
