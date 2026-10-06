import numpy as np
from PIL import Image


class FakeSdpc:
    level_dimensions = ((100, 80), (50, 40), (25, 20))
    level_downsamples = (1.0, 2.0, 4.0)
    mpp_level0 = 0.5

    def read_region(self, location, level, size):
        self.call = (location, level, size)
        return np.full((size[1], size[0], 3), 127, dtype=np.uint8)


def test_sdpc_patch_uses_physical_fov_and_nearest_target_mpp():
    from deepspatial.histology.sdpc import extract_sdpc_patch

    reader = FakeSdpc()
    patch = extract_sdpc_patch(
        reader, source_xy_um=(25.0, 20.0), patch_size_um=20.0, target_mpp=1.0
    )
    assert reader.call == ((30, 20), 1, (20, 20))
    assert isinstance(patch, Image.Image) and patch.size == (224, 224)
    assert np.asarray(patch).mean() == 127


def test_sdpc_patch_pads_outside_slide_white():
    from deepspatial.histology.sdpc import extract_sdpc_patch

    reader = FakeSdpc()
    patch = extract_sdpc_patch(
        reader, source_xy_um=(2.0, 2.0), patch_size_um=20.0, target_mpp=1.0
    )
    image = np.asarray(patch)
    assert image[0, 0].min() == 255
    assert image[-1, -1].max() == 127
