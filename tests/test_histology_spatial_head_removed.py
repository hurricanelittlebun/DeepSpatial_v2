from types import SimpleNamespace

import torch


def test_shared_step_does_not_query_histology_features_for_model_forward():
    from deepspatial.module import DeepSpatialModule

    class RuntimeDouble:
        config = SimpleNamespace(
            use_morphology_path=True,
            use_histology_spatial_head=True,
        )

        def plan(self, path_id, t):
            x0 = torch.zeros((len(path_id), 2), dtype=t.dtype)
            return x0 + t[:, None], torch.ones_like(x0)

        def features(self, *_args, **_kwargs):
            raise AssertionError("H&E features must not enter the model forward")

    class ZeroVelocityModel(torch.nn.Module):
        def forward(self, xt, gt, t, zt, delta_z, ct):
            return torch.zeros_like(xt), torch.zeros_like(gt), torch.zeros_like(ct)

    module = DeepSpatialModule(
        {
            "path_type": "Linear",
            "prediction": "velocity",
            "train_eps": 0.02,
            "sample_eps": 0.02,
            "use_celltype": True,
            "lambda_g": 0.1,
            "lambda_c": 10.0,
        },
        ZeroVelocityModel(),
        histology_runtime=RuntimeDouble(),
    )

    batch_size = 3
    batch = {
        "x0": torch.zeros(batch_size, 2),
        "x1": torch.ones(batch_size, 2),
        "g0": torch.zeros(batch_size, 4),
        "g1": torch.ones(batch_size, 4),
        "c0": torch.eye(2)[[0, 1, 0]],
        "c1": torch.eye(2)[[1, 0, 1]],
        "z0": torch.zeros(batch_size, 1),
        "z1": torch.ones(batch_size, 1),
        "delta_z": torch.ones(batch_size, 1),
        "path_id": torch.zeros(batch_size, dtype=torch.long),
    }

    losses = module._shared_step(batch)

    for name in ("loss", "loss_x", "loss_g", "loss_c"):
        assert torch.isfinite(losses[name])
