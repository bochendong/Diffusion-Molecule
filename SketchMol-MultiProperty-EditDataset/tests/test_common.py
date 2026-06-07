from sketchmol_multiproperty_dataset.common import (
    active_property_deltas,
    pair_quality_tier,
    render_instruction,
    sketchmol_condition_columns,
    source_similarity_bin,
    strict_property_success,
)


def test_active_property_deltas_and_instruction():
    source = {"MW": 200.0, "LogP": 3.0, "QED": 0.4, "TPSA": 50.0, "HBD": 1.0, "HBA": 2.0, "RB": 4.0}
    target = {"MW": 250.0, "LogP": 2.0, "QED": 0.48, "TPSA": 70.0, "HBD": 1.0, "HBA": 3.0, "RB": 3.0}

    active = active_property_deltas(source, target)
    assert active["MW"] == 50.0
    assert active["LogP"] == -1.0
    assert active["QED"] == 0.07999999999999996
    assert "HBD" not in active

    instruction = render_instruction(
        selected_props=["LogP", "QED"],
        source_props=source,
        target_props=target,
        deltas=active,
    )
    assert "decrease LogP" in instruction
    assert "increase QED" in instruction
    assert instruction.startswith("Starting from the source molecule")

    columns = sketchmol_condition_columns(target, ["LogP", "QED"])
    assert columns["logp_None"] == "False"
    assert columns["QED_None"] == "False"
    assert columns["TPSA_None"] == "True"


def test_source_similarity_quality_helpers():
    assert source_similarity_bin(0.72) == "high_similarity"
    assert source_similarity_bin(0.52) == "medium_similarity"
    assert source_similarity_bin(0.42) == "hard_similarity"
    assert source_similarity_bin(0.30) == "too_distant"
    assert pair_quality_tier(0.55, same_scaffold=True) == "same_scaffold_medium_plus"
    assert pair_quality_tier(0.55, same_scaffold=False) == "cross_scaffold_medium_similarity"


def test_strict_property_success_uses_sketchmol_tolerances():
    target = {"MW": 250.0, "LogP": 2.0}
    assert strict_property_success({"MW": 280.0, "LogP": 2.9}, target, ["MW", "LogP"])
    assert not strict_property_success({"MW": 290.1, "LogP": 2.9}, target, ["MW", "LogP"])
