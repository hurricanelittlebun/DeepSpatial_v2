import numpy as np
import pandas as pd
import pytest


def test_materialized_nucleus_table_contains_fixed_coordinates():
    from scripts.materialize_00029_fixed_nucleus_coordinates import (
        add_fixed_coordinates,
    )

    table = pd.DataFrame({"x_registered_um": [10.0], "y_registered_um": [20.0]})
    matrix = np.asarray(
        [[2.0, 0.0, 3.0], [0.0, 2.0, 4.0], [0.0, 0.0, 1.0]]
    )

    result = add_fixed_coordinates(table, matrix)

    np.testing.assert_allclose(
        result[["x_fixed_he_um", "y_fixed_he_um"]].to_numpy(),
        [[23.0, 44.0]],
    )


def test_materialized_nucleus_table_rejects_missing_registered_coordinates():
    from scripts.materialize_00029_fixed_nucleus_coordinates import (
        add_fixed_coordinates,
    )

    with pytest.raises(KeyError, match="y_registered_um"):
        add_fixed_coordinates(pd.DataFrame({"x_registered_um": [1.0]}), np.eye(3))


def test_three_way_frame_contract_requires_one_fixed_frame():
    from scripts.validate_00029_fixed_he_three_way_qc import (
        validate_fixed_frame_contract,
    )

    assert validate_fixed_frame_contract(
        "00029__g0_fixed_he",
        "00029__g0_fixed_he",
        "00029__g0_fixed_he",
    ) == "00029__g0_fixed_he"
