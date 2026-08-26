from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

pytest.importorskip("rdkit")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_stage1_data as builder  # noqa: E402
import augment_denovo_pool as augment  # noqa: E402
import p23_protocol as protocol  # noqa: E402


def edit_row(tasks: list[dict[str, str]] | None = None) -> dict[str, str]:
    return {
        "example_id": "edit-1", "source_smiles": "CCO", "target_smiles": "CCN",
        "instruction_tasks": json.dumps(tasks or [{"property": "GSK3β", "direction": "increase"}]),
        "computed_active_properties": "MW|LogP|QED|TPSA|HBA|RB",
        "MW_active": "True", "MW_direction": "decrease",
    }


def denovo_row(example_id: str = "denovo-1", target: str = "CCO", count: int = 2) -> dict[str, str]:
    row = {"example_id": example_id, "target_smiles": target}
    for prop, value in list(zip(("MW", "QED", "LogP"), (40.0, 0.6, 1.0)))[:count]:
        row[f"{prop}_active"] = "True"
        row[f"target_{prop}"] = str(value)
    return row


def test_denovo_replacement_projection_is_exactly_2p():
    donor = denovo_row("candidate-7p", "CCCO", count=3)
    for prop in ("TPSA", "HBD", "HBA", "RB"):
        donor[f"{prop}_active"] = "True"
        donor[f"target_{prop}"] = "1.0"
    projected = augment.project_two_property(donor)
    program = protocol.legacy_denovo_conditions(projected)
    assert [item["property"] for item in program] == ["MW", "QED"]
    assert projected["property_count"] == "2"
    assert projected["split"] == "train"


def test_edit_prompt_uses_explicit_tasks_and_keeps_assays():
    row = edit_row([
        {"property": "GSK3β", "direction": "increase"},
        {"property": "DRD2", "direction": "decrease"},
        {"property": "JNK3", "direction": "down"},
    ])
    messages, source, mode = protocol.build_prompt(row)
    payload = json.loads(messages[1]["content"])
    assert source == "CCO" and mode == "edit"
    assert payload["conditions"] == [
        {"property": "GSK3B", "goal": "increase"},
        {"property": "DRD2", "goal": "decrease"},
        {"property": "JNK3", "goal": "decrease"},
    ]
    serialized = json.dumps(messages)
    assert "CCN" not in serialized
    assert "MW" not in serialized  # incidental target delta must not become an instruction


def test_table1_task_key_is_official_order():
    program = protocol.explicit_instruction_conditions(edit_row([
        {"property": "SA", "direction": "decrease"},
        {"property": "DRD2", "direction": "decrease"},
        {"property": "MW", "direction": "decrease"},
    ]))
    assert protocol.task_key(program) == "DRD2:decrease+MW:decrease+SA:decrease"


def test_edit_fails_closed_without_explicit_instruction_metadata():
    row = edit_row()
    row["instruction_tasks"] = ""
    with pytest.raises(ValueError, match="no explicit instruction tasks"):
        protocol.build_prompt(row)


def test_malformed_instruction_metadata_fails_closed():
    row = edit_row()
    row["instruction_tasks"] = "not-json"
    with pytest.raises(ValueError, match="malformed explicit instruction metadata"):
        protocol.build_prompt(row)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_builder_balances_modes_and_excludes_heldout(tmp_path: Path):
    denovo_path, edit_path, heldout_path = tmp_path / "denovo.csv", tmp_path / "edit.csv", tmp_path / "eval.csv"
    write_csv(denovo_path, [
        denovo_row("d1", "CCO"), denovo_row("d2", "CCN"), denovo_row("d3", "CCC"),
    ])
    edit_rows = [
        {**edit_row([{"property": "GSK3B", "direction": "increase"}]), "example_id": "e1", "target_smiles": "CCCl", "source_target_tanimoto": "0.8"},
        {**edit_row([{"property": "GSK3B", "direction": "increase"}]), "example_id": "e2", "source_smiles": "CCC", "target_smiles": "CCBr", "source_target_tanimoto": "0.8"},
        {**edit_row([{"property": "GSK3B", "direction": "decrease"}]), "example_id": "e3", "source_smiles": "CCF", "target_smiles": "CCI", "source_target_tanimoto": "0.8"},
    ]
    write_csv(edit_path, edit_rows)
    write_csv(heldout_path, [{"source_smiles": "CCO", "target_smiles": "CO"}])
    output = tmp_path / "out"
    assert builder.main([
        "--denovo-csv", str(denovo_path), "--edit-csv", str(edit_path),
        "--heldout-csv", str(heldout_path), "--output-dir", str(output),
        "--rows-per-mode", "2", "--minimum-edit-task-rows", "1", "--seed", "7",
    ]) == 0
    sft = [json.loads(line) for line in (output / "train.sft.jsonl").read_text().splitlines()]
    contrastive = [json.loads(line) for line in (output / "train.contrastive.jsonl").read_text().splitlines()]
    manifest = json.loads((output / "manifest.json").read_text())
    assert Counter(row["task_mode"] for row in sft) == {"de_novo": 2, "edit": 2}
    assert all(row["source_smiles"] != "CCO" for row in sft if row["task_mode"] == "edit")
    assert manifest["heldout_overlap"] == {"source": 0, "target": 0}
    assert "invalid_corruption" in {row["negative_type"] for row in contrastive}
    assert "opposite_program_target" in {row["negative_type"] for row in contrastive}


def test_stratum_counts_respect_reserved_projected_pairs(tmp_path: Path):
    edit_path = tmp_path / "edit.csv"
    broad_task = [{"property": "GSK3B", "direction": "decrease"}]
    edit_rows = [
        {**edit_row(broad_task), "example_id": "e1", "target_smiles": "CCCl", "source_target_tanimoto": "0.8"},
        {**edit_row(broad_task), "example_id": "e2", "source_smiles": "CCC", "target_smiles": "CCBr", "source_target_tanimoto": "0.8"},
    ]
    write_csv(edit_path, edit_rows)
    reserved = {builder.pair_key(edit_rows[0])}
    counts, audit = builder.count_strata(
        edit_path, "edit", set(), set(), 0.65, excluded_pairs=reserved,
    )
    assert counts == {"broad::GSK3B:decrease": 1}
    assert audit["excluded_projected_pair"] == 1


def test_edit500_runner_exports_and_checks_required_assay_oracles():
    for name in ("run_edit500_prepare.sh", "run_edit500_score.sh"):
        runner = (HERE / name).read_text(encoding="utf-8")
        assert 'export SUCC_GSK3B_ORACLE_PATH=' in runner
        assert 'export SUCC_DRD2_ORACLE_PATH=' in runner
        assert '"$SUCC_GSK3B_ORACLE_PATH" "$SUCC_DRD2_ORACLE_PATH"' in runner
