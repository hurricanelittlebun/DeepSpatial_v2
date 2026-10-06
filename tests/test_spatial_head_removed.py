import inspect

import torch
import pytest


def test_histology_config_rejects_removed_spatial_head_argument():
    from deepspatial.histology import HistologyConfig

    with pytest.raises(TypeError):
        HistologyConfig(
            use_histology=True,
            feature_path="features.h5",
            coordinate_frame="toy",
            use_histology_spatial_head=True,
        )


def test_git_forward_keeps_all_three_velocity_shapes_without_histology_head():
    from deepspatial.models.git import GiT

    init_parameters = inspect.signature(GiT.__init__).parameters
    forward_parameters = inspect.signature(GiT.forward).parameters
    assert "histology_dim" not in init_parameters
    assert "ht" not in forward_parameters

    torch.manual_seed(3)
    model = GiT(
        gene_dim=5,
        patch_size=2,
        hidden_size=16,
        depth=1,
        num_heads=2,
        num_classes=3,
    )
    batch_size = 4
    x, g, c = model(
        xt=torch.rand(batch_size, 2),
        gt=torch.rand(batch_size, 5),
        t=torch.rand(batch_size),
        zt=torch.rand(batch_size, 1),
        delta_z=torch.ones(batch_size, 1),
        ct=torch.eye(3)[[0, 1, 2, 0]],
    )

    assert x.shape == (batch_size, 2)
    assert g.shape == (batch_size, 5)
    assert c.shape == (batch_size, 3)
