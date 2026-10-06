import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scripts.prepare_00029_celltype_anchors import label_anchor_from_annotations


def _make_pair(*, annotation_ids=None, anchor_ids=None, gene_names=None, anchor_gene_names=None,
               annotation_xy=None, anchor_xy=None, labels=None):
    annotation_ids = annotation_ids or ["a", "b", "c"]
    anchor_ids = anchor_ids or ["c", "a", "b"]
    gene_names = gene_names or ["G1", "G2"]
    anchor_gene_names = anchor_gene_names or gene_names
    labels = labels or {"a": "Epi", "b": "T", "c": "CAF"}
    domains = {"a": "Tumor", "b": "Stroma", "c": "Tumor"}
    niches = {"a": "Tumor nest", "b": "CAF enriched", "c": "Tumor nest"}
    domain_conf = {"a": 0.9, "b": 0.8, "c": 0.7}
    niche_conf = {"a": 0.6, "b": 0.7, "c": 0.8}
    if annotation_xy is None:
        base_xy = {"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [5.0, 6.0]}
        annotation_xy = np.array([base_xy[x] for x in annotation_ids], dtype=np.float32)
    anchor_xy = anchor_xy if anchor_xy is not None else np.array(
        [[5.0, 6.0], [1.0, 2.0], [3.0, 4.0]], dtype=np.float32
    )
    annotation = ad.AnnData(
        X=sp.csr_matrix(np.ones((len(annotation_ids), len(gene_names)), dtype=np.float32)),
        obs=pd.DataFrame(
            {"source_cell_id": annotation_ids, "section_id": 1,
             "assigned_celltype": [labels[x] for x in annotation_ids],
             "pred_domain": [domains[x] for x in annotation_ids],
             "pred_niche": [niches[x] for x in annotation_ids],
             "pred_domain_conf": [domain_conf[x] for x in annotation_ids],
             "pred_niche_conf": [niche_conf[x] for x in annotation_ids]},
            index=[f"ann-{i}" for i in range(len(annotation_ids))],
        ),
        var=pd.DataFrame(index=gene_names),
        obsm={"spatial_st_corrected": annotation_xy},
    )
    anchor = ad.AnnData(
        X=sp.csr_matrix(np.ones((len(anchor_ids), len(anchor_gene_names)), dtype=np.float32)),
        obs=pd.DataFrame(
            {"source_cell_id": anchor_ids, "section_id": 1, "z_um": 0.0},
            index=[f"anchor-{i}" for i in range(len(anchor_ids))],
        ),
        var=pd.DataFrame(index=anchor_gene_names),
        obsm={"spatial_st_corrected": anchor_xy},
    )
    return annotation, anchor


def test_label_join_uses_source_id_and_preserves_anchor_order():
    annotation, anchor = _make_pair()
    result = label_anchor_from_annotations(annotation, anchor)
    assert result.obs["cell_class"].astype(str).tolist() == ["CAF", "Epi", "T"]
    assert result.obs_names.tolist() == anchor.obs_names.tolist()
    assert result.obsm["spatial_st_corrected"].tolist() == anchor.obsm["spatial_st_corrected"].tolist()


def test_label_join_propagates_domain_and_niche_in_anchor_order():
    annotation, anchor = _make_pair()
    result = label_anchor_from_annotations(annotation, anchor)

    assert result.obs["pred_domain"].astype(str).tolist() == ["Tumor", "Tumor", "Stroma"]
    assert result.obs["pred_niche"].astype(str).tolist() == [
        "Tumor nest", "Tumor nest", "CAF enriched"
    ]
    np.testing.assert_allclose(
        result.obs["pred_domain_conf"].to_numpy(), [0.7, 0.9, 0.8]
    )
    np.testing.assert_allclose(
        result.obs["pred_niche_conf"].to_numpy(), [0.8, 0.6, 0.7]
    )


def test_missing_annotation_id_is_rejected():
    annotation, anchor = _make_pair(annotation_ids=["a", "b"])
    with pytest.raises(ValueError, match="missing annotation labels"):
        label_anchor_from_annotations(annotation, anchor)


def test_duplicate_annotation_id_is_rejected():
    annotation, anchor = _make_pair(annotation_ids=["a", "a", "b"])
    with pytest.raises(ValueError, match="duplicate source_cell_id"):
        label_anchor_from_annotations(annotation, anchor)


def test_gene_order_mismatch_is_rejected():
    annotation, anchor = _make_pair(anchor_gene_names=["G2", "G1"])
    with pytest.raises(ValueError, match="gene order"):
        label_anchor_from_annotations(annotation, anchor)


def test_coordinate_mismatch_is_rejected():
    annotation, anchor = _make_pair(
        anchor_xy=np.array([[5.0, 6.1], [1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    )
    with pytest.raises(ValueError, match="spatial_st_corrected"):
        label_anchor_from_annotations(annotation, anchor)


def test_missing_label_value_is_rejected():
    annotation, anchor = _make_pair()
    annotation.obs.loc[annotation.obs["source_cell_id"] == "b", "assigned_celltype"] = None
    with pytest.raises(ValueError, match="missing annotation labels"):
        label_anchor_from_annotations(annotation, anchor)
