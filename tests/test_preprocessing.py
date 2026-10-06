import numpy as np
import torch
from PIL import Image


def test_fixed_physical_fov_and_white_padding():
    from deepspatial.histology.preprocessing import extract_physical_patch

    class Reader:
        def read_region(self, start, level, size):
            self.start = start
            self.size = size
            return Image.new("RGBA", size, (0, 0, 0, 0))

    r = Reader()
    p = extract_physical_patch(
        r,
        (110.0, 220.0),
        mpp=(0.5, 1.0),
        image_origin_um=(100.0, 200.0),
        patch_size_um=10.0,
    )
    assert r.start == (10, 15) and r.size == (20, 10)
    assert p.size == (224, 224) and np.asarray(p).min() == 255


def test_frozen_uni2_wrapper_without_downloading_weights(monkeypatch):
    """Mock only gated 681M-weight construction, not preprocessing/freezing/encode."""
    import timm
    from deepspatial.histology.uni2 import UNI2Encoder

    class Tiny(torch.nn.Module):
        pretrained_cfg = {
            "input_size": (3, 224, 224),
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
        }

        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.ones(1536))

        def forward(self, x):
            return x.mean((1, 2, 3))[:, None] * self.p

    monkeypatch.setattr(timm, "create_model", lambda *a, **k: Tiny())
    enc = UNI2Encoder(device="cpu")
    enc.model.train()
    out = enc.encode([Image.new("RGB", (224, 224), "white")] * 3, batch_size=2)
    assert out.shape == (3, 1536) and not out.requires_grad
    assert not enc.model.training and not any(
        p.requires_grad for p in enc.model.parameters()
    )
    expected = np.mean([(1 - 0.485) / 0.229, (1 - 0.456) / 0.224, (1 - 0.406) / 0.225])
    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5)


def test_streamed_preprocess_roundtrip(tmp_path):
    from deepspatial.histology.preprocessing import precompute_section
    from deepspatial.histology import FeatureStore

    class Reader:
        dimensions = (100, 100)

        def read_region(self, start, level, size):
            return Image.new("RGB", size, "white")

    class Encoder:
        feature_dim = 2

        def encode(self, patches, batch_size):
            return torch.ones(len(patches), 2)

    s = FeatureStore(tmp_path / "features.h5", mode="a")
    precompute_section(
        Reader(),
        Encoder(),
        s,
        "a",
        z_um=10,
        mpp=(0.5, 0.5),
        image_origin_um=(0, 0),
        grid_origin_um=(5, 5),
        grid_shape=(3, 3),
        grid_spacing_um=(5, 5),
        patch_size_um=10,
        coordinate_frame="toy",
        batch_size=2,
    )
    s2 = FeatureStore(s.path, cache_mb=0.00001)  # exercise true lazy HDF5 gathers
    np.testing.assert_allclose(s2.get_feature("a", [[7.5, 7.5], [10, 10]]), 1)
