from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest


def test_final_config_uses_materialized_st_key_and_fixed_he_policy():
    from scripts.validate_00029_st_only_residual_dataset import make_final_config

    config = make_final_config(
        anchor_dir="anchors",
        feature_path="../uni2_features.h5",
        nucleus_feature_path="nucleus_features_registered/nucleus_features.h5",
        path_cache="paths.h5",
        uot_dir="uot_cache_topk64",
        coordinate_frame="00029__g0_registered",
    )
    assert config["spatial_key"] == "spatial_st_corrected"
    assert config["original_spatial_key"] == "spatial_registered"
    assert config["coordinate_frame"] == "00029__g0_registered"
    assert config["he_policy"] == "unchanged_registered_he"
    assert config["serial_he_transform_policy"] == "identity_residual"
    assert config["apply_external_residual_at_runtime"] is False


def test_he_table_contract_rejects_any_residual_applied_to_he():
    from scripts.validate_00029_st_only_residual_dataset import validate_he_table

    table = pd.DataFrame(
        {
            "section_id": [1],
            "coordinate_frame": ["00029__g0_registered"],
            "residual_applied_to_he": [True],
        }
    )
    with pytest.raises(ValueError, match="H&E residual"):
        validate_he_table(table, "00029__g0_registered")


def test_validator_cli_can_start_when_invoked_by_script_path():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/validate_00029_st_only_residual_dataset.py"),
            "--help",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_final_qc_cli_exposes_materialized_anchor_directory():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/render_00029_residual_contrast_qc.py"),
            "--help",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--anchor-dir" in result.stdout
