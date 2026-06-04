from sketchmol_multiproperty_dataset.common import active_property_deltas, render_instruction, sketchmol_condition_columns


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

    columns = sketchmol_condition_columns(target, ["LogP", "QED"])
    assert columns["logp_None"] == "False"
    assert columns["QED_None"] == "False"
    assert columns["TPSA_None"] == "True"
