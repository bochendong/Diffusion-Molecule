import importlib.util
from pathlib import Path


CONDITION_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_condition_rows.py"
CONDITION_SPEC = importlib.util.spec_from_file_location("build_condition_rows", CONDITION_SCRIPT)
condition_builder = importlib.util.module_from_spec(CONDITION_SPEC)
assert CONDITION_SPEC.loader is not None
CONDITION_SPEC.loader.exec_module(condition_builder)

MANIFEST_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_diffusion_edit_manifest.py"
MANIFEST_SPEC = importlib.util.spec_from_file_location("export_diffusion_edit_manifest", MANIFEST_SCRIPT)
manifest_export = importlib.util.module_from_spec(MANIFEST_SPEC)
assert MANIFEST_SPEC.loader is not None
MANIFEST_SPEC.loader.exec_module(manifest_export)


def test_condition_row_carries_source_neighbor_quality_fields():
    pair = {
        "pair_id": "mpair_0001",
        "split": "eval",
        "source_smiles": "CCO",
        "target_smiles": "CCN",
        "source_scaffold": "s1",
        "target_scaffold": "s1",
        "same_scaffold": "True",
        "scaffold_relation": "same_scaffold",
        "source_tanimoto": "0.62",
        "source_similarity_bin": "medium_similarity",
        "pair_quality_tier": "same_scaffold_medium_plus",
        "selection_reason": "same_scaffold_source_neighbor",
        "same_scaffold_neighbor_count": "12",
        "source_neighbor_count_t04": "10",
        "source_neighbor_count_t05": "8",
        "source_neighbor_count_t06": "3",
        "target_neighbor_rank_by_tanimoto": "2",
        "active_properties": "MW,LogP",
        "source_MW": "100",
        "target_MW": "150",
        "delta_MW": "50",
        "source_LogP": "1",
        "target_LogP": "2",
        "delta_LogP": "1",
        "source_QED": "0.5",
        "target_QED": "0.5",
        "delta_QED": "0",
        "source_TPSA": "10",
        "target_TPSA": "10",
        "delta_TPSA": "0",
        "source_HBD": "0",
        "target_HBD": "0",
        "delta_HBD": "0",
        "source_HBA": "1",
        "target_HBA": "1",
        "delta_HBA": "0",
        "source_RB": "1",
        "target_RB": "1",
        "delta_RB": "0",
    }

    row = condition_builder._condition_row(pair, ["MW", "LogP"], 0)

    assert row["source_tanimoto"] == "0.62"
    assert row["pair_quality_tier"] == "same_scaffold_medium_plus"
    assert row["preservation_constraint"] == "keep_same_scaffold_or_source_tanimoto_ge_0_4"
    assert row["instruction_template_id"] == "local_edit_numeric_v1"
    assert "property_constraints_json" in row


def test_manifest_carries_quality_fields():
    row = {
        "condition_id": "cond_1",
        "pair_id": "pair_1",
        "split": "eval",
        "source_smiles": "CCO",
        "target_smiles": "CCN",
        "instruction": "Starting from the source molecule, make a local edit.",
        "condition_properties": "MW,LogP",
        "property_count": "2",
        "source_tanimoto": "0.62",
        "source_similarity_bin": "medium_similarity",
        "pair_quality_tier": "same_scaffold_medium_plus",
        "strict_candidate_count_t04": "3",
        "oracle_strict_success_t04": "True",
        "preservation_constraint": "keep_same_scaffold_or_source_tanimoto_ge_0_4",
    }
    for prop in manifest_export.PROPERTY_COLUMNS:
        row[f"source_{prop}"] = "1"
        row[f"target_{prop}"] = "2"
        row[f"delta_{prop}"] = "1"
        row[f"{prop}_active"] = "False"
        row[f"{prop}_direction"] = ""
        value_col, none_col = manifest_export.SKETCHMOL_SETTING_COLUMNS[prop]
        row[value_col] = ""
        row[none_col] = "True"

    out = manifest_export._manifest_row(row, source_tanimoto=0.62)

    assert out["pair_quality_tier"] == "same_scaffold_medium_plus"
    assert out["strict_candidate_count_t04"] == "3"
    assert out["oracle_strict_success_t04"] == "True"
    assert out["preservation_constraint"] == "keep_same_scaffold_or_source_tanimoto_ge_0_4"
