#!/usr/bin/env python3
"""CPU-only contract test for P8.1.5 inversion and support semantics."""

from prepare_roundtrip_data import cycle_row, invert_direction, invert_text


def main() -> int:
    assert invert_direction("increase") == "decrease"
    assert invert_direction("-1") == "1"
    text = invert_text("increase QED and decrease MW")
    assert text == "decrease QED and increase MW", text
    forward = {
        "condition_id": "e1", "source_smiles": "CC", "target_smiles": "CCC",
        "task_mode": "edit", "QED_direction": "increase",
        "instruction": "increase QED", "external_property_directions_json": '{"QED":"increase"}',
    }
    inverse = cycle_row(forward)
    assert inverse["source_smiles"] == "CCC"
    assert inverse["target_smiles"] == "CC"
    assert inverse["condition_id"] == "e1__cycle"
    assert inverse["QED_direction"] == "decrease"
    assert inverse["instruction"] == "decrease QED"
    assert inverse["external_property_directions_json"] == '{"QED": "decrease"}'
    print("P8.1.5 contract OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
