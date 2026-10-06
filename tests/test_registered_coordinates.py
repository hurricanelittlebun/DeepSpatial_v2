from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Section:
    preorientation_matrix: np.ndarray
    edge_chain: tuple[tuple[str, str], ...]


class TranslationStore:
    def __init__(self, translations):
        self.translations = translations

    def load_edge(self, edge_key):
        return edge_key

    def transform(self, points, edge, direction):
        delta = np.asarray(self.translations[edge], dtype=float)
        return points + delta if direction == "forward" else points - delta


def test_inverse_registered_coordinates_reverses_complete_chain():
    from deepspatial.histology.registered_coordinates import (
        registered_to_source_points,
        source_to_registered_points,
    )

    preorientation = np.array(
        [[0.0, -1.0, 20.0], [1.0, 0.0, -4.0], [0.0, 0.0, 1.0]]
    )
    section = Section(
        preorientation_matrix=preorientation,
        edge_chain=(("a", "forward"), ("b", "reverse")),
    )
    store = TranslationStore({"a": (3.0, 5.0), "b": (-2.0, 7.0)})
    source = np.array([[1.0, 2.0], [7.5, -3.0], [0.0, 0.0]])

    registered = source_to_registered_points(source, section, store, batch_size=2)
    recovered = registered_to_source_points(registered, section, store, batch_size=2)

    np.testing.assert_allclose(recovered, source, atol=1e-12)


def test_crop_pixel_coordinate_conversion_uses_bbox_and_mpp():
    from deepspatial.histology.registered_coordinates import source_um_to_crop_px

    points_um = np.array([[500.0, 1000.0], [504.0, 1006.0]])
    pixels = source_um_to_crop_px(
        points_um, mpp=(2.0, 2.0), bbox_origin_px=(200.0, 400.0)
    )
    np.testing.assert_allclose(pixels, [[50.0, 100.0], [52.0, 103.0]])
