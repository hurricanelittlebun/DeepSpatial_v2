import numpy as np
import pandas as pd
import anndata as ad


def test_prepare_one_tissue_group_uses_physical_z_and_keeps_groups_separate(tmp_path):
    from deepspatial.data_utils.series_preparation import prepare_series

    obs = pd.DataFrame(
        {
            "joint_series_id": ["00029__g0", "00029__g0", "00029__g1"],
            "section_id": [1, 11, 1],
        },
        index=["a", "b", "c"],
    )
    source = ad.AnnData(
        np.eye(3, dtype=np.float32),
        obs=obs,
        obsm={"spatial_registered": np.array([[1, 2], [3, 4], [9, 9]], dtype=float)},
    )
    h5ad = tmp_path / "all.h5ad"
    source.write_h5ad(h5ad)
    he = pd.DataFrame(
        {
            "sample_id": ["00029", "00029", "00029"],
            "tissue_id": [None, "00029", "00029"],
            "group_id": ["g0", "g0", "g1"],
            "section_id": [1, 3, 1],
            "source_kind": ["xenium_companion_he", "serial_he", "serial_he"],
            "masked_crop_path": ["/anchor.tif", "files/serial.png", "files/wrong.png"],
            "mask_path": ["/anchor-mask.png", "files/serial-mask.png", "files/wrong-mask.png"],
            "analysis_pixel_size_x_um": [2.0, 1.0, 1.0],
            "analysis_pixel_size_y_um": [2.0, 1.0, 1.0],
            "bbox_x0": [0, 10, 0],
            "bbox_y0": [0, 20, 0],
            "bbox_x1": [100, 110, 10],
            "bbox_y1": [100, 120, 10],
        }
    )
    transforms = pd.DataFrame(
        {
            "sample_id": ["00029", "00029"],
            "group_id": ["g0", "g0"],
            "section_id": [1, 3],
            "affine_path": ["transforms/1.npz", "transforms/3.npz"],
            "registration_status": ["accepted", "review_required"],
            "registration_confidence": [1.0, 0.8],
        }
    )

    result = prepare_series(
        h5ad,
        he,
        transforms,
        tmp_path / "dataset",
        series_id="00029__g0",
        section_thickness_um=5,
        he_geometry_root=tmp_path / "geometry",
        registration_root=tmp_path / "registration",
        validate_paths=False,
    )

    assert result["anchor_sections"] == [1, 11]
    assert result["he_sections"] == [1, 3]
    assert result["missing_sections"] == [2, 4, 5, 6, 7, 8, 9, 10]
    a0 = ad.read_h5ad(tmp_path / "dataset" / "anchors" / "section-001.h5ad")
    a1 = ad.read_h5ad(tmp_path / "dataset" / "anchors" / "section-011.h5ad")
    assert a0.n_obs == a1.n_obs == 1
    assert a0.obs["z_um"].iloc[0] == 0
    assert a1.obs["z_um"].iloc[0] == 50
    np.testing.assert_array_equal(a0.obsm["spatial"], [[1, 2]])
    manifest = pd.read_parquet(tmp_path / "dataset" / "he_sections.parquet")
    assert manifest["section_id"].tolist() == [1, 3]
    assert manifest["z_um"].tolist() == [0, 10]
    assert manifest["source_image_path"].tolist() == [
        "/anchor.tif",
        str(tmp_path / "geometry" / "files/serial.png"),
    ]
