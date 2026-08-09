from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = (
    ROOT
    / "SketchMol-Understanding-Condition"
    / "experiments"
    / "unified_constraint_agent"
)


def load_module(name: str):
    path = MODULE_ROOT / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ir_module = load_module("molecular_constraint_ir")
audit = load_module("audit_candidate_pools")
trajectory = load_module("build_verifier_trajectories")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_constraint_ir_separates_design_and_edit_actions() -> None:
    design = ir_module.build_constraint_ir(
        {
            "condition_id": "design-1",
            "task_mode": "de_novo",
            "condition_properties": "MW,QED",
            "target_MW": "320",
            "target_QED": "0.7",
            "instruction": "Generate a molecule near the requested targets.",
        }
    )
    edit = ir_module.build_constraint_ir(
        {
            "condition_id": "edit-1",
            "source_smiles": "CCO",
            "external_task_properties": "QED,BBBP",
            "external_property_objectives_json": json.dumps({"QED": "improve", "BBBP": "maintain"}),
            "external_property_directions_json": json.dumps({"QED": 1, "BBBP": 1}),
        }
    )

    assert design.task_mode == "de_novo"
    assert design.action_space == "smiles"
    assert [item.objective for item in design.constraints] == ["target", "target"]
    assert edit.task_mode == "edit"
    assert edit.action_space == "graph_edit_dsl"
    assert [(item.property, item.objective, item.direction) for item in edit.constraints] == [
        ("QED", "improve", 1),
        ("BBBP", "maintain", 0),
    ]


def make_config(tmp_path: Path) -> Path:
    candidate_csv = tmp_path / "candidates.csv"
    rows = [
        {
            "condition_id": "a",
            "task_mode": "de_novo",
            "condition_properties": "MW,QED",
            "generated_smiles": "C",
            "generation_rank": "1",
            "candidate_rank": "2",
            "valid_smiles": "True",
            "unified_property_success_fraction": "0.5",
            "unified_property_distance": "0.3",
        },
        {
            "condition_id": "a",
            "task_mode": "de_novo",
            "condition_properties": "MW,QED",
            "generated_smiles": "CC",
            "generation_rank": "2",
            "candidate_rank": "1",
            "candidate_selected": "True",
            "valid_smiles": "True",
            "unified_property_success_fraction": "1.0",
            "unified_property_distance": "0.0",
        },
        {
            "condition_id": "b",
            "task_mode": "de_novo",
            "condition_properties": "MW,QED",
            "generated_smiles": "CCC",
            "generation_rank": "1",
            "candidate_rank": "1",
            "candidate_selected": "True",
            "valid_smiles": "True",
            "unified_property_success_fraction": "0.5",
            "unified_property_distance": "0.1",
        },
        {
            "condition_id": "b",
            "task_mode": "de_novo",
            "condition_properties": "MW,QED",
            "generated_smiles": "CCCC",
            "generation_rank": "2",
            "candidate_rank": "2",
            "valid_smiles": "True",
            "unified_property_success_fraction": "0.0",
            "unified_property_distance": "0.8",
        },
    ]
    write_csv(candidate_csv, rows)
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "name": "teacher",
                        "suite": "denovo",
                        "candidate_csv": str(candidate_csv),
                        "budget": 2,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return config


def test_candidate_audit_decomposes_support_and_selection(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    spec = audit.load_specs(config)[0]
    rows, manifest = audit.audit_run(spec)
    overall = next(row for row in rows if row["split"] == "all" and row["property_count"] == "all")

    assert manifest["condition_groups"] == 2
    assert overall["raw_at_1"] == 0.0
    assert overall["any_hit_at_k"] == 0.5
    assert overall["selected_at_k"] == 0.5
    assert overall["selection_miss"] == 0.0


def test_table1_strict_success_requires_source_similarity() -> None:
    base = {
        "source_smiles": "CCO",
        "generated_smiles": "CCN",
        "valid_smiles": "True",
        "unified_property_success_fraction": "1.0",
    }

    assert not audit.strict_success({**base, "source_similarity_success": "False"})
    assert audit.strict_success({**base, "source_similarity_success": "True"})
    assert audit.strict_success({**base, "external_official_success": "True"})


def test_trajectory_builder_requires_strict_positive_and_keeps_revision_case(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    spec = audit.load_specs(config)[0]
    rows, summary = trajectory.build_run(spec)

    assert summary == {
        "groups": 2,
        "strict_preferences": 1,
        "revision_cases": 1,
        "skipped_no_negative": 0,
    }
    preference = next(row for row in rows if row["trajectory_type"] == "strict_preference")
    revision = next(row for row in rows if row["trajectory_type"] == "revision_needed")
    assert preference["chosen_smiles"] == "CC"
    assert preference["rejected_smiles"] == "C"
    assert revision["chosen_smiles"] == ""
    assert revision["rejected_smiles"] == "CCC"
