import importlib.util
import csv
import subprocess
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_external_multiproperty_predictions.py"


def load_evaluator():
    spec = importlib.util.spec_from_file_location("evaluate_external_multiproperty_predictions", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_official_sr_groups_candidates_and_does_not_require_similarity(monkeypatch):
    evaluator = load_evaluator()
    props = {
        "SRC1": {"QED": 0.2, "LogP": 1.0, "SA": 0.0},
        "FAIL1": {"QED": 0.25, "LogP": 1.0, "SA": 0.0},
        "HIT1": {"QED": 0.4, "LogP": 1.0, "SA": 0.0},
        "SRC2": {"QED": 0.2, "LogP": 1.0, "SA": 0.0},
        "FAIL2": {"QED": 0.22, "LogP": 1.0, "SA": 0.0},
    }
    sims = {("SRC1", "HIT1"): 0.2, ("SRC1", "FAIL1"): 0.7, ("SRC2", "FAIL2"): 0.8}
    monkeypatch.setattr(evaluator, "canonical_smiles", lambda smiles: str(smiles or "").strip() or None)
    monkeypatch.setattr(evaluator, "molecular_properties", lambda smiles: props.get(smiles, {}))
    monkeypatch.setattr(evaluator, "morgan_tanimoto", lambda left, right: sims.get((left, right), 0.0))

    base = {
        "external_suite": "mumo",
        "external_task_split": "ind",
        "external_task_id": "Q",
        "external_task_properties": "qed",
        "external_property_directions_json": '{"qed":"increase"}',
        "external_property_thresholds_json": '{"qed":0.1}',
    }
    prediction_rows = [
        {**base, "condition_id": "input_1", "source_smiles": "SRC1", "generated_smiles": "FAIL1"},
        {**base, "condition_id": "input_1", "source_smiles": "SRC1", "generated_smiles": "HIT1"},
        {**base, "condition_id": "input_2", "source_smiles": "SRC2", "generated_smiles": "FAIL2"},
    ]

    detail = [
        evaluator.evaluate_row(
            row,
            generated_props={},
            source_props_lookup={},
            smiles_column="generated_smiles",
            source_smiles_column="source_smiles",
            min_source_tanimoto=0.4,
        )
        for row in prediction_rows
    ]
    summary = evaluator.summarize(detail, group_column="condition_id")
    task_summary = next(row for row in summary if row["external_task_id"] == "Q")

    assert task_summary["input_groups"] == 2
    assert task_summary["candidate_rows"] == 3
    assert task_summary["validity"] == "1"
    assert task_summary["success_rate"] == "0.5"
    assert task_summary["similarity"] == "0.2"
    assert task_summary["relative_improvement"] == "1"
    assert task_summary["strict_success_rate"] == "0"


def test_missing_oracle_marks_success_rate_as_lower_bound(monkeypatch):
    evaluator = load_evaluator()
    monkeypatch.setattr(evaluator, "canonical_smiles", lambda smiles: str(smiles or "").strip() or None)
    monkeypatch.setattr(evaluator, "molecular_properties", lambda _smiles: {})
    monkeypatch.setattr(evaluator, "morgan_tanimoto", lambda _left, _right: 1.0)

    row = {
        "condition_id": "input_1",
        "source_smiles": "SRC",
        "generated_smiles": "GEN",
        "external_suite": "mumo",
        "external_task_split": "ind",
        "external_task_id": "B",
        "external_task_properties": "bbbp",
        "external_property_directions_json": '{"bbbp":"increase"}',
        "external_property_thresholds_json": '{"bbbp":0.1}',
    }

    detail = [
        evaluator.evaluate_row(
            row,
            generated_props={},
            source_props_lookup={},
            smiles_column="generated_smiles",
            source_smiles_column="source_smiles",
            min_source_tanimoto=0.4,
        )
    ]
    summary = evaluator.summarize(detail, group_column="condition_id")
    task_summary = next(item for item in summary if item["external_task_id"] == "B")

    assert detail[0]["external_full_property_coverage"] == "False"
    assert task_summary["official_evaluable_rate"] == "0"
    assert task_summary["success_rate_status"] == "lower_bound_missing_oracle"
    assert task_summary["missing_oracle_properties"] == "bbbp"


def test_invalid_candidates_do_not_mark_covered_best_of_k_as_missing_oracle(monkeypatch):
    evaluator = load_evaluator()
    monkeypatch.setattr(evaluator, "canonical_smiles", lambda smiles: str(smiles or "").strip() or None)
    monkeypatch.setattr(evaluator, "molecular_properties", lambda _smiles: {})
    monkeypatch.setattr(evaluator, "morgan_tanimoto", lambda _left, _right: 1.0)

    base = {
        "condition_id": "input_1",
        "source_smiles": "SRC",
        "external_suite": "mumo",
        "external_task_split": "ind",
        "external_task_id": "B",
        "external_task_properties": "bbbp",
        "external_property_directions_json": '{"bbbp":"increase"}',
        "external_property_thresholds_json": '{"bbbp":0.1}',
    }
    detail = [
        evaluator.evaluate_row(
            {**base, "generated_smiles": generated},
            generated_props={"GEN": {"bbbp": 0.8}},
            source_props_lookup={"SRC": {"bbbp": 0.5}},
            smiles_column="generated_smiles",
            source_smiles_column="source_smiles",
            min_source_tanimoto=0.4,
        )
        for generated in ("", "GEN")
    ]
    summary = evaluator.summarize(detail, group_column="condition_id")
    task_summary = next(item for item in summary if item["external_task_id"] == "B")

    assert task_summary["validity"] == "1"
    assert task_summary["success_rate"] == "1"
    assert task_summary["official_evaluable_rate"] == "1"
    assert task_summary["success_rate_status"] == "official"
    assert task_summary["missing_oracle_properties"] == ""


def test_missing_source_does_not_count_as_similarity_success(monkeypatch):
    evaluator = load_evaluator()
    monkeypatch.setattr(evaluator, "canonical_smiles", lambda smiles: str(smiles or "").strip() or None)
    monkeypatch.setattr(evaluator, "molecular_properties", lambda smiles: {"QED": 0.5} if smiles == "GEN" else {})
    monkeypatch.setattr(evaluator, "morgan_tanimoto", lambda _left, _right: None)

    detail = evaluator.evaluate_row(
        {
            "condition_id": "input_1",
            "source_smiles": "",
            "generated_smiles": "GEN",
            "external_suite": "cmumo",
            "external_task_split": "ind",
            "external_task_id": "Q",
            "external_task_properties": "qed",
            "external_property_directions_json": '{"qed":"increase"}',
            "external_property_thresholds_json": '{"qed":0.1}',
            "external_target_qed": "0.4",
        },
        generated_props={},
        source_props_lookup={},
        smiles_column="generated_smiles",
        source_smiles_column="source_smiles",
        min_source_tanimoto=0.4,
    )
    summary = evaluator.summarize([detail], group_column="condition_id")
    task_summary = next(item for item in summary if item["external_task_id"] == "Q")

    assert detail["external_valid"] == "True"
    assert detail["external_source_available"] == "False"
    assert detail["external_source_similarity_success"] == "False"
    assert detail["external_official_success"] == "True"
    assert detail["external_strict_success"] == "False"
    assert task_summary["source_available_rate"] == "0"
    assert task_summary["source_similarity_success_rate"] == "0"
    assert task_summary["strict_success_rate"] == "0"


def test_cmumo_maintain_objective_allows_small_non_degrading_change(monkeypatch):
    evaluator = load_evaluator()
    props = {
        "SRC": {"QED": 0.8},
        "KEEP": {"QED": 0.74},
        "DROP": {"QED": 0.65},
    }
    monkeypatch.setattr(evaluator, "canonical_smiles", lambda smiles: str(smiles or "").strip() or None)
    monkeypatch.setattr(evaluator, "molecular_properties", lambda smiles: props.get(smiles, {}))
    monkeypatch.setattr(evaluator, "morgan_tanimoto", lambda _left, _right: 0.8)

    base = {
        "external_suite": "cmumo",
        "external_task_split": "ind",
        "external_task_id": "Q-maintain",
        "external_task_properties": "qed",
        "external_property_directions_json": '{"qed":"increase"}',
        "external_property_objectives_json": '{"qed":"maintain"}',
        "external_property_thresholds_json": '{"qed":0.1}',
        "source_smiles": "SRC",
    }
    detail = [
        evaluator.evaluate_row(
            {**base, "condition_id": "input_1", "generated_smiles": "KEEP"},
            generated_props={},
            source_props_lookup={},
            smiles_column="generated_smiles",
            source_smiles_column="source_smiles",
            min_source_tanimoto=0.4,
        ),
        evaluator.evaluate_row(
            {**base, "condition_id": "input_2", "generated_smiles": "DROP"},
            generated_props={},
            source_props_lookup={},
            smiles_column="generated_smiles",
            source_smiles_column="source_smiles",
            min_source_tanimoto=0.4,
        ),
    ]
    summary = evaluator.summarize(detail, group_column="condition_id")
    task_summary = next(item for item in summary if item["external_task_id"] == "Q-maintain")

    assert detail[0]["external_all_property_success"] == "True"
    assert detail[0]["external_mean_relative_improvement"] == ""
    assert detail[1]["external_all_property_success"] == "False"
    assert task_summary["success_rate"] == "0.5"


def test_relative_improvement_is_undefined_for_near_zero_source():
    evaluator = load_evaluator()

    assert (
        evaluator.relative_improvement(
            source_value=0.0,
            generated_value=0.5,
            direction="increase",
        )
        is None
    )


def test_aligned_report_builder_emits_failure_breakdown(tmp_path):
    eval_dir = tmp_path / "ours_eval"
    report_dir = tmp_path / "aligned"
    eval_dir.mkdir()
    _write_rows(
        eval_dir / "external_multiproperty_summary.csv",
        [
            {
                "external_suite": "mumo",
                "external_task_split": "ind",
                "external_task_id": "Q",
                "input_groups": "4",
                "candidate_rows": "4",
                "validity": "0.75",
                "source_available_rate": "1",
                "success_rate": "0.25",
                "similarity": "0.2",
                "relative_improvement": "1.0",
                "source_similarity_success_rate": "0.5",
                "all_property_success_rate": "0.25",
                "strict_success_rate": "0",
                "official_evaluable_rate": "0.75",
                "mean_evaluated_property_fraction": "0.75",
                "missing_oracle_properties": "",
                "success_rate_status": "official",
            },
            {
                "external_suite": "all",
                "external_task_split": "all",
                "external_task_id": "all",
                "input_groups": "4",
                "candidate_rows": "4",
                "validity": "0.75",
                "source_available_rate": "1",
                "success_rate": "0.25",
                "similarity": "0.2",
                "relative_improvement": "1.0",
                "source_similarity_success_rate": "0.5",
                "all_property_success_rate": "0.25",
                "strict_success_rate": "0",
                "official_evaluable_rate": "0.75",
                "mean_evaluated_property_fraction": "0.75",
                "missing_oracle_properties": "",
                "success_rate_status": "official",
            },
        ],
    )
    _write_rows(
        eval_dir / "external_multiproperty_detail.csv",
        [
            {
                "condition_id": "a",
                "external_suite": "mumo",
                "external_task_split": "ind",
                "external_task_id": "Q",
                "external_valid": "True",
                "external_full_property_coverage": "True",
                "external_all_property_success": "True",
                "external_source_similarity_success": "False",
                "external_official_success": "True",
                "external_strict_success": "False",
            },
            {
                "condition_id": "b",
                "external_suite": "mumo",
                "external_task_split": "ind",
                "external_task_id": "Q",
                "external_valid": "False",
                "external_full_property_coverage": "False",
                "external_all_property_success": "False",
                "external_source_similarity_success": "False",
                "external_official_success": "False",
                "external_strict_success": "False",
            },
            {
                "condition_id": "c",
                "external_suite": "mumo",
                "external_task_split": "ind",
                "external_task_id": "Q",
                "external_valid": "True",
                "external_full_property_coverage": "False",
                "external_all_property_success": "False",
                "external_source_similarity_success": "True",
                "external_official_success": "False",
                "external_strict_success": "False",
            },
            {
                "condition_id": "d",
                "external_suite": "mumo",
                "external_task_split": "ind",
                "external_task_id": "Q",
                "external_valid": "True",
                "external_full_property_coverage": "True",
                "external_all_property_success": "False",
                "external_source_similarity_success": "True",
                "external_official_success": "False",
                "external_strict_success": "False",
            },
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/summarize_external_multiproperty_aligned_runs.py",
            "--run",
            f"ours={eval_dir}",
            "--output-dir",
            str(report_dir),
        ],
        cwd="SketchMol-Understanding-Condition",
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    comparison_rows = list(
        csv.DictReader((report_dir / "external_multiproperty_aligned_comparison.csv").open(newline="", encoding="utf-8"))
    )
    failure_rows = list(
        csv.DictReader((report_dir / "external_multiproperty_failure_breakdown.csv").open(newline="", encoding="utf-8"))
    )
    overall = next(row for row in comparison_rows if row["external_suite"] == "all")
    failure_overall = next(row for row in failure_rows if row["external_suite"] == "all")

    assert overall["run_label"] == "ours"
    assert overall["SR"] == "0.25"
    assert failure_overall["official_success_groups"] == "1"
    assert failure_overall["official_success_similarity_failed_groups"] == "1"
    assert failure_overall["invalid_groups"] == "1"
    assert failure_overall["missing_oracle_groups"] == "1"
    assert failure_overall["similarity_pass_property_failed_groups"] == "1"


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
