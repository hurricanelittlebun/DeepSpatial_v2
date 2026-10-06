from scripts.run_00029_celltype_nucleus_uot_path import (
    EXPERIMENT_NAME,
    TRAINING_SECTIONS,
    build_training_spec,
)


def test_celltype_training_spec_uses_annotated_branch_without_overwriting_baseline():
    spec = build_training_spec()
    assert spec["use_celltype"] is True
    assert spec["label_key"] == "cell_class"
    assert spec["uot_solver"] == "sparse_topk"
    assert spec["uot_top_k"] == 64
    assert spec["use_nucleus_path"] is True
    assert spec["output_name"] == EXPERIMENT_NAME
    assert spec["output_name"] != "st_only_residual_nucleus_uot_path_v1"
    assert tuple(spec["training_sections"]) == TRAINING_SECTIONS


def test_celltype_training_spec_keeps_section_31_out_of_training():
    spec = build_training_spec()
    assert 31 not in spec["training_sections"]
    assert spec["heldout_section"] == 31


def test_celltype_training_spec_allows_overriding_flow_celltype_loss_weight():
    spec = build_training_spec(lambda_c=2.0)
    assert spec["lambda_c"] == 2.0
