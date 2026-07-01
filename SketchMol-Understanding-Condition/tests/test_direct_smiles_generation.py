from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch


def test_direct_smiles_entrypoints_exist():
    assert Path("SketchMol-Understanding-Condition/scripts/train_direct_smiles_generator.py").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/train_direct_smiles_generator_rl.py").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/build_direct_smiles_preference_dataset.py").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/train_direct_smiles_generator_dpo.py").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/export_external_multiproperty_benchmark_rows.py").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/evaluate_external_multiproperty_predictions.py").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/score_external_multiproperty_oracles.py").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/build_external_agentic_revise_predictions.py").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/build_external_graph_edit_agent_predictions.py").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/build_external_source_copy_predictions.py").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/build_external_target_copy_predictions.py").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/run_direct_smiles_external_multiproperty_benchmark.sh").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_multiproperty_benchmark.sh").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/run_direct_smiles_external_multiproperty_group_rl.sh").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_multiproperty_group_rl.sh").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_mumo_one_shot_baseline.sh").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/run_direct_smiles_external_multiproperty_source_edit_sft.sh").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_mumo_source_edit_sft.sh").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_mumo_source_edit_sft_group_rl.sh").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/run_direct_smiles_external_mumo_agentic_revise.sh").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_mumo_agentic_revise.sh").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_mumo_agentic_revise_rich.sh").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/run_direct_smiles_external_mumo_graph_edit_agent.sh").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_mumo_graph_edit_agent.sh").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_mumo_graph_edit_policy.sh").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/run_external_multiproperty_source_copy_sanity.sh").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/submit_external_mumo_source_copy_sanity.sh").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/submit_external_multiproperty_official_suite.sh").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/run_direct_smiles_denovo_2p7p_benchmark.sh").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/run_direct_smiles_denovo_ood_benchmark.sh").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/run_direct_smiles_denovo_2p7p_v2_benchmark.sh").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/run_direct_smiles_denovo_2p7p_v2_n128_benchmark.sh").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/run_direct_smiles_denovo_2p7p_v2_conservative_benchmark.sh").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/run_direct_smiles_denovo_ood_v2_benchmark.sh").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/run_direct_smiles_denovo_ood_v2_n128_benchmark.sh").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/run_direct_smiles_denovo_ood_v2_validity_repair_benchmark.sh").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/run_direct_smiles_denovo_ood_v2_balanced_benchmark.sh").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/run_direct_smiles_rl_2p7p_v2.sh").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/run_direct_smiles_group_rl_2p7p_v2.sh").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/run_direct_smiles_group_rl_ood_v2.sh").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/run_direct_smiles_group_rl_ood_v2_conservative_benchmark.sh").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/run_direct_smiles_group_rl_ood_v2_n256_benchmark.sh").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/run_direct_smiles_group_rl_ood_v2_conservative_n256_benchmark.sh").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/run_direct_smiles_preference_dpo_2p7p.sh").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/submit_direct_smiles_denovo_2p7p_v2_benchmark.sh").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/submit_direct_smiles_denovo_2p7p_v2_n128_benchmark.sh").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/submit_direct_smiles_denovo_2p7p_v2_conservative_benchmark.sh").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/submit_direct_smiles_denovo_ood_v2_benchmark.sh").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/submit_direct_smiles_denovo_ood_v2_n128_benchmark.sh").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/submit_direct_smiles_denovo_ood_v2_validity_repair_benchmark.sh").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/submit_direct_smiles_denovo_ood_v2_balanced_benchmark.sh").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/submit_direct_smiles_rl_2p7p_v2.sh").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/submit_direct_smiles_group_rl_2p7p_v2.sh").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/submit_direct_smiles_group_rl_ood_v2.sh").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/submit_direct_smiles_group_rl_ood_v2_conservative_benchmark.sh").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/submit_direct_smiles_group_rl_ood_v2_n256_benchmark.sh").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/submit_direct_smiles_group_rl_ood_v2_conservative_n256_benchmark.sh").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/submit_univideo_moledit_v3_fair_table1_extension.sh").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/submit_direct_smiles_denovo_v2_rerun_suite.sh").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/collect_direct_smiles_denovo_v2_suite_results.py").exists()
    assert Path("SketchMol-Understanding-Condition/scripts/submit_direct_smiles_preference_dpo_2p7p.sh").exists()


def test_collect_direct_smiles_v2_suite_results(tmp_path):
    suite_root = tmp_path / "suite"
    report_path = tmp_path / "suite_report.md"
    csv_path = tmp_path / "suite_summary.csv"

    _write_rows(
        suite_root / "2p7p_default_n128" / "benchmark_direct_smiles" / "benchmark_summary.csv",
        [
            {"method": "direct_smiles_mllm", "property_count": "2", "strict_success_rate": "0.965"},
            {"method": "direct_smiles_mllm", "property_count": "7", "strict_success_rate": "0.584"},
            {
                "method": "direct_smiles_mllm",
                "property_count": "all",
                "strict_success_rate": "0.791",
                "validity": "0.997",
                "success_rate_strict_in_valid_mols": "0.793",
                "unique_valid_smiles": "5981",
                "uniqueness_in_valid_mols": "0.999",
            },
        ],
    )
    _write_rows(
        suite_root / "2p7p_conservative_n128" / "benchmark_direct_smiles" / "benchmark_summary.csv",
        [
            {"method": "direct_smiles_mllm", "property_count": "2", "strict_success_rate": "0.972"},
            {"method": "direct_smiles_mllm", "property_count": "7", "strict_success_rate": "0.601"},
            {
                "method": "direct_smiles_mllm",
                "property_count": "all",
                "strict_success_rate": "0.812",
                "validity": "0.998",
                "success_rate_strict_in_valid_mols": "0.814",
                "unique_valid_smiles": "5988",
                "uniqueness_in_valid_mols": "0.999",
            },
        ],
    )
    _write_rows(
        suite_root / "ood_conservative_n128" / "benchmark_direct_smiles" / "benchmark_summary.csv",
        [
            {"method": "direct_smiles_mllm", "property_count": "2", "strict_success_rate": "0.633"},
            {"method": "direct_smiles_mllm", "property_count": "7", "strict_success_rate": "0.720"},
            {
                "method": "direct_smiles_mllm",
                "property_count": "all",
                "strict_success_rate": "0.741",
                "validity": "0.979",
                "success_rate_strict_in_valid_mols": "0.757",
                "unique_valid_smiles": "944",
                "uniqueness_in_valid_mols": "0.964",
            },
        ],
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/collect_direct_smiles_denovo_v2_suite_results.py",
            "--suite-root",
            str(suite_root),
            "--report-path",
            str(report_path),
            "--csv-path",
            str(csv_path),
        ],
        cwd="SketchMol-Understanding-Condition",
        check=True,
    )

    report_text = report_path.read_text(encoding="utf-8")
    assert "2p7p v2 default n=128" in report_text
    assert "2p7p v2 conservative n=128" in report_text
    assert "OOD v2 conservative n=128" in report_text
    assert "missing" in report_text

    summary_rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    assert len(summary_rows) == 4
    by_variant = {row["variant"]: row for row in summary_rows}
    assert by_variant["2p7p_default_n128"]["overall_strict"] == "0.791"
    assert by_variant["2p7p_conservative_n128"]["strict_7p"] == "0.601"
    assert by_variant["ood_conservative_n128"]["strict_7p"] == "0.720"
    assert by_variant["ood_default_n128"]["status"] == "missing"


def test_external_multiproperty_exporter_preserves_official_task_properties(tmp_path):
    source_csv = tmp_path / "sources.csv"
    output_csv = tmp_path / "external_rows.csv"
    _write_rows(
        source_csv,
        [
            {
                "sample_id": "mol_a",
                "source_smiles": "CCO",
                "bbbp": "0.2",
                "drd2": "0.1",
                "qed": "0.4",
                "target_qed": "0.62",
            }
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/export_external_multiproperty_benchmark_rows.py",
            "--source-file",
            str(source_csv),
            "--output-csv",
            str(output_csv),
            "--suite",
            "mumo",
            "--tasks",
            "BDQ",
            "--max-rows-per-task",
            "1",
        ],
        cwd="SketchMol-Understanding-Condition",
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    rows = list(csv.DictReader(output_csv.open(newline="", encoding="utf-8")))
    assert len(rows) == 1
    row = rows[0]
    assert row["external_suite"] == "mumo"
    assert row["external_task_id"] == "BDQ"
    assert row["external_task_properties"] == "bbbp,drd2,qed"
    assert json.loads(row["external_property_objectives_json"]) == {
        "bbbp": "improve",
        "drd2": "improve",
        "qed": "improve",
    }
    assert row["external_unsupported_properties"] == "bbbp,drd2"
    assert row["condition_properties"] == "QED"
    assert row["property_count"] == "3"
    assert row["target_QED"] == "0.62"
    assert row["external_target_placeholder"] == "True"


def test_external_multiproperty_exporter_filters_input_split(tmp_path):
    source_csv = tmp_path / "sources.csv"
    output_csv = tmp_path / "external_rows.csv"
    _write_rows(
        source_csv,
        [
            {"sample_id": "train_a", "split": "train", "source_smiles": "CCO", "qed": "0.4"},
            {"sample_id": "test_a", "split": "test", "source_smiles": "CCN", "qed": "0.4"},
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/export_external_multiproperty_benchmark_rows.py",
            "--source-file",
            str(source_csv),
            "--output-csv",
            str(output_csv),
            "--suite",
            "mumo",
            "--tasks",
            "DPQ",
            "--input-split",
            "test",
            "--max-rows-per-task",
            "10",
        ],
        cwd="SketchMol-Understanding-Condition",
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    rows = list(csv.DictReader(output_csv.open(newline="", encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["molecule_id"] == "test_a"
    assert rows[0]["source_smiles"] == "CCN"


def test_external_multiproperty_exporter_preserves_property_objectives(tmp_path):
    source_csv = tmp_path / "sources.csv"
    output_csv = tmp_path / "external_rows.csv"
    _write_rows(
        source_csv,
        [
            {
                "sample_id": "mol_a",
                "source_smiles": "CCO",
                "qed": "0.8",
                "qed_objective": "maintain",
            }
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/export_external_multiproperty_benchmark_rows.py",
            "--source-file",
            str(source_csv),
            "--output-csv",
            str(output_csv),
            "--suite",
            "cmumo",
            "--tasks",
            "BPQ",
            "--max-rows-per-task",
            "1",
        ],
        cwd="SketchMol-Understanding-Condition",
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    row = next(csv.DictReader(output_csv.open(newline="", encoding="utf-8")))
    objectives = json.loads(row["external_property_objectives_json"])
    assert objectives["qed"] == "maintain"
    assert "maintain high QED" in row["instruction"]


def test_external_multiproperty_evaluator_uses_generated_properties_csv(tmp_path):
    predictions_csv = tmp_path / "predictions.csv"
    generated_props_csv = tmp_path / "generated_props.csv"
    output_dir = tmp_path / "eval"
    _write_rows(
        predictions_csv,
        [
            {
                "condition_id": "external_mumo_bdq_000000",
                "source_smiles": "CCO",
                "generated_smiles": "CCN",
                "external_suite": "mumo",
                "external_task_split": "ind",
                "external_task_id": "BDQ",
                "external_task_properties": "bbbp,qed",
                "external_property_directions_json": json.dumps({"bbbp": "increase", "qed": "increase"}),
                "external_property_thresholds_json": json.dumps({"bbbp": 0.2, "qed": 0.1}),
                "external_source_bbbp": "0.2",
                "external_source_qed": "0.4",
            }
        ],
    )
    _write_rows(
        generated_props_csv,
        [
            {
                "smiles": "CCN",
                "bbbp": "0.45",
                "qed": "0.62",
            }
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_external_multiproperty_predictions.py",
            "--prediction-csv",
            str(predictions_csv),
            "--generated-properties-csv",
            str(generated_props_csv),
            "--output-dir",
            str(output_dir),
            "--min-source-tanimoto",
            "0",
        ],
        cwd="SketchMol-Understanding-Condition",
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary_rows = list(csv.DictReader((output_dir / "external_multiproperty_summary.csv").open(newline="", encoding="utf-8")))
    overall = [row for row in summary_rows if row["external_suite"] == "all"][0]
    assert overall["strict_success_rate"] == "1"
    assert overall["missing_oracle_properties"] == ""


def test_external_multiproperty_evaluator_uses_generated_properties_as_source_fallback(tmp_path):
    predictions_csv = tmp_path / "predictions.csv"
    generated_props_csv = tmp_path / "generated_props.csv"
    output_dir = tmp_path / "eval"
    _write_rows(
        predictions_csv,
        [
            {
                "condition_id": "external_cmumo_bpq_000000",
                "source_smiles": "CCO",
                "generated_smiles": "CCN",
                "external_suite": "cmumo",
                "external_task_split": "ind",
                "external_task_id": "BPQ",
                "external_task_properties": "bbbp",
                "external_property_directions_json": json.dumps({"bbbp": "increase"}),
                "external_property_objectives_json": json.dumps({"bbbp": "maintain"}),
                "external_property_thresholds_json": json.dumps({"bbbp": 0.1}),
            }
        ],
    )
    _write_rows(
        generated_props_csv,
        [
            {"smiles": "CCO", "bbbp": "0.8"},
            {"smiles": "CCN", "bbbp": "0.75"},
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_external_multiproperty_predictions.py",
            "--prediction-csv",
            str(predictions_csv),
            "--generated-properties-csv",
            str(generated_props_csv),
            "--output-dir",
            str(output_dir),
            "--min-source-tanimoto",
            "0",
        ],
        cwd="SketchMol-Understanding-Condition",
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary_rows = list(csv.DictReader((output_dir / "external_multiproperty_summary.csv").open(newline="", encoding="utf-8")))
    overall = [row for row in summary_rows if row["external_suite"] == "all"][0]
    assert overall["success_rate"] == "1"
    assert overall["success_rate_status"] == "official"
    assert overall["missing_oracle_properties"] == ""


def test_external_oracle_scorer_merges_precomputed_properties(tmp_path):
    predictions_csv = tmp_path / "predictions.csv"
    merge_csv = tmp_path / "precomputed.csv"
    output_csv = tmp_path / "oracle_props.csv"
    _write_rows(predictions_csv, [{"generated_smiles": "CCO"}])
    _write_rows(merge_csv, [{"smiles": "CCO", "bbbp": "0.77"}])

    result = subprocess.run(
        [
            sys.executable,
            "scripts/score_external_multiproperty_oracles.py",
            "--input-csv",
            str(predictions_csv),
            "--merge-properties-csv",
            str(merge_csv),
            "--output-csv",
            str(output_csv),
            "--required-properties",
            "bbbp",
            "--tdc-oracles",
            "never",
        ],
        cwd="SketchMol-Understanding-Condition",
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    row = next(csv.DictReader(output_csv.open(newline="", encoding="utf-8")))
    assert row["bbbp"] == "0.77"


def test_external_oracle_collector_keeps_all_smiles_columns_per_row(tmp_path):
    predictions_csv = tmp_path / "predictions.csv"
    output_csv = tmp_path / "unique_smiles.csv"
    _write_rows(
        predictions_csv,
        [
            {
                "generated_smiles": "CCN",
                "source_smiles": "CCO",
                "target_smiles": "CCC",
            }
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/collect_external_multiproperty_oracle_smiles.py",
            "--input-csv",
            str(predictions_csv),
            "--output-csv",
            str(output_csv),
        ],
        cwd="SketchMol-Understanding-Condition",
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    rows = list(csv.DictReader(output_csv.open(newline="", encoding="utf-8")))
    assert [row["smiles"] for row in rows] == ["CCN", "CCO", "CCC"]


def test_external_oracle_collector_can_skip_precovered_smiles(tmp_path):
    predictions_csv = tmp_path / "predictions.csv"
    covered_csv = tmp_path / "covered.csv"
    output_csv = tmp_path / "unique_smiles.csv"
    _write_rows(
        predictions_csv,
        [
            {
                "generated_smiles": "CCN",
                "source_smiles": "CCO",
                "target_smiles": "CCC",
            }
        ],
    )
    _write_rows(covered_csv, [{"smiles": "CCN", "bbbp": "0.8", "hia": "0.7"}])

    result = subprocess.run(
        [
            sys.executable,
            "scripts/collect_external_multiproperty_oracle_smiles.py",
            "--input-csv",
            str(predictions_csv),
            "--exclude-properties-csv",
            str(covered_csv),
            "--required-properties",
            "bbbp,hia",
            "--output-csv",
            str(output_csv),
        ],
        cwd="SketchMol-Understanding-Condition",
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    rows = list(csv.DictReader(output_csv.open(newline="", encoding="utf-8")))
    assert [row["smiles"] for row in rows] == ["CCO", "CCC"]


def test_external_source_copy_predictions_use_source_smiles(tmp_path):
    rows_csv = tmp_path / "rows.csv"
    predictions_csv = tmp_path / "source_copy_predictions.csv"
    _write_rows(
        rows_csv,
        [
            {
                "condition_id": "external_mumo_dpq_000000",
                "source_smiles": "CCO",
                "target_smiles": "CCO",
            }
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_external_source_copy_predictions.py",
            "--rows-csv",
            str(rows_csv),
            "--prediction-csv",
            str(predictions_csv),
        ],
        cwd="SketchMol-Understanding-Condition",
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    rows = list(csv.DictReader(predictions_csv.open(newline="", encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["generated_smiles"] == "CCO"
    assert rows[0]["method"] == "source_copy_sanity"
    assert rows[0]["direct_candidate_count"] == "1"


def test_agentic_revise_prefers_source_similar_property_success(monkeypatch, tmp_path):
    module = _load_agentic_revise_module()
    monkeypatch.setattr(module, "safe_canonical", lambda value: str(value or ""))
    monkeypatch.setattr(
        module,
        "safe_tanimoto",
        lambda left, right: 0.8 if right == "near_good" else (1.0 if left == right else 0.0),
    )
    monkeypatch.setattr(
        module,
        "local_properties",
        lambda smiles: {"plogp": {"source": 0.0, "near_good": 1.2, "far_good": 1.4}.get(smiles, 0.0)},
    )
    monkeypatch.setattr(
        module,
        "local_edit_candidates",
        lambda smiles, max_candidates, action_profile="basic": [("near_good", f"{action_profile}:add_C")]
        if smiles == "source"
        else [],
    )
    args = module.parse_args(
        [
            "--rows-csv",
            str(tmp_path / "rows.csv"),
            "--prediction-csv",
            str(tmp_path / "predictions.csv"),
            "--max-steps",
            "1",
            "--edit-action-profile",
            "rich",
        ]
    )
    row = {
        "condition_id": "a",
        "source_smiles": "source",
        "external_task_properties": "plogp",
        "external_property_directions_json": json.dumps({"plogp": "increase"}),
        "external_property_thresholds_json": json.dumps({"plogp": 1.0}),
    }

    result = module.revise_row(
        row,
        direct_row={"condition_id": "a", "generated_smiles": "far_good"},
        args=args,
        rng=__import__("random").Random(7),
    )

    assert result["generated_smiles"] == "near_good"
    assert result["agentic_source_similarity_success"] == "True"
    assert result["agentic_all_evaluated_local_success"] == "True"


def test_agentic_revise_similarity_first_requires_local_success(monkeypatch, tmp_path):
    module = _load_agentic_revise_module()
    monkeypatch.setattr(module, "safe_canonical", lambda value: str(value or ""))
    monkeypatch.setattr(
        module,
        "safe_tanimoto",
        lambda left, right: {"source": 1.0, "near_bad": 0.9, "near_good": 0.8, "far_good": 0.0}.get(right, 0.0),
    )
    monkeypatch.setattr(
        module,
        "local_properties",
        lambda smiles: {"plogp": {"source": 0.0, "near_bad": 0.2, "near_good": 1.1, "far_good": 1.5}.get(smiles, 0.0)},
    )
    args = module.parse_args(
        [
            "--rows-csv",
            str(tmp_path / "rows.csv"),
            "--prediction-csv",
            str(tmp_path / "predictions.csv"),
            "--max-steps",
            "0",
            "--selection-mode",
            "similarity_first",
            "--similarity-first-min-local-success-fraction",
            "1.0",
        ]
    )
    row = {
        "condition_id": "a",
        "source_smiles": "source",
        "external_task_properties": "plogp",
        "external_property_directions_json": json.dumps({"plogp": "increase"}),
        "external_property_thresholds_json": json.dumps({"plogp": 1.0}),
    }
    ranked = module.rank_candidates(
        row,
        [
            module.Candidate("source", "source_copy", "source_copy"),
            module.Candidate("near_bad", "near_bad", "test"),
            module.Candidate("near_good", "near_good", "test"),
            module.Candidate("far_good", "far_good", "test"),
        ],
        args=args,
    )

    assert ranked[0].canonical_smiles == "near_good"


def test_graph_edit_planner_emits_structured_dsl_actions():
    module = _load_graph_edit_agent_module()
    row = {
        "source_smiles": "CCO",
        "external_task_properties": "plogp,qed",
        "external_property_directions_json": json.dumps({"plogp": "increase", "qed": "increase"}),
    }

    actions = module.plan_graph_edit_actions(row, source_smiles="CCO", site_limit=2, max_plans_per_property=4)

    assert actions
    assert any(action.op in {"add_atom", "add_fragment"} for action in actions)
    assert all(json.loads(action.to_json())["op"] for action in actions)
    assert any(action.prop == "plogp" for action in actions)


def test_graph_edit_policy_planner_scores_rich_dsl_actions(monkeypatch):
    module = _load_graph_edit_agent_module()
    monkeypatch.setattr(module, "editable_atom_sites", lambda _source_smiles, site_limit: [0, 1])
    monkeypatch.setattr(module, "editable_bond_sites", lambda _source_smiles, site_limit: [(0, 1)])
    row = {
        "source_smiles": "CCO",
        "external_task_properties": "plogp,qed",
        "external_property_directions_json": json.dumps({"plogp": "increase", "qed": "increase"}),
    }

    actions = module.plan_graph_edit_actions(
        row,
        source_smiles="CCO",
        site_limit=2,
        max_plans_per_property=32,
        planner_mode="policy_graph_dsl",
    )

    assert actions
    assert actions[0].policy_score >= actions[-1].policy_score
    assert any(action.fragment in {"C#N", "C(=O)N", "c1ccccc1"} for action in actions)
    assert any(action.op == "change_bond_order" for action in actions)


def test_graph_edit_agent_prefers_source_similar_property_success(monkeypatch, tmp_path):
    module = _load_graph_edit_agent_module()
    monkeypatch.setattr(module.revise, "safe_canonical", lambda value: str(value or ""))
    monkeypatch.setattr(
        module.revise,
        "safe_tanimoto",
        lambda _left, right: {"source": 1.0, "near_good": 0.8, "far_good": 0.0}.get(right, 0.0),
    )
    monkeypatch.setattr(
        module.revise,
        "local_properties",
        lambda smiles: {"plogp": {"source": 0.0, "near_good": 1.2, "far_good": 1.4}.get(smiles, 0.0)},
    )
    monkeypatch.setattr(module, "editable_atom_sites", lambda _source_smiles, site_limit: [0])
    monkeypatch.setattr(module, "editable_bond_sites", lambda _source_smiles, site_limit: [])

    def fake_execute(_source_smiles, action):
        return "near_good" if action.op == "add_fragment" and action.fragment == "C" else ""

    monkeypatch.setattr(module, "execute_graph_edit_action", fake_execute)
    args = module.parse_args(
        [
            "--rows-csv",
            str(tmp_path / "rows.csv"),
            "--prediction-csv",
            str(tmp_path / "predictions.csv"),
            "--selection-mode",
            "similarity_first",
            "--similarity-first-min-local-success-fraction",
            "1.0",
            "--site-limit",
            "1",
            "--max-plans-per-property",
            "8",
        ]
    )
    row = {
        "condition_id": "a",
        "source_smiles": "source",
        "external_task_properties": "plogp",
        "external_property_directions_json": json.dumps({"plogp": "increase"}),
        "external_property_thresholds_json": json.dumps({"plogp": 1.0}),
    }

    result, records = module.predict_row(
        row,
        direct_row={"condition_id": "a", "generated_smiles": "far_good"},
        args=args,
        rng=__import__("random").Random(7),
    )

    assert result["generated_smiles"] == "near_good"
    assert result["graph_edit_candidate_source"] == "graph_edit_dsl"
    assert result["graph_edit_source_similarity_success"] == "True"
    assert any(record["generated_smiles"] == "near_good" for record in records)


def test_graph_edit_agent_emits_top_k_candidate_rows(monkeypatch, tmp_path):
    module = _load_graph_edit_agent_module()
    monkeypatch.setattr(module.revise, "safe_canonical", lambda value: str(value or ""))
    monkeypatch.setattr(
        module.revise,
        "safe_tanimoto",
        lambda _left, right: {"source": 1.0, "near_good": 0.8, "far_good": 0.0}.get(right, 0.0),
    )
    monkeypatch.setattr(
        module.revise,
        "local_properties",
        lambda smiles: {"plogp": {"source": 0.0, "near_good": 1.2, "far_good": 1.4}.get(smiles, 0.0)},
    )
    monkeypatch.setattr(module, "editable_atom_sites", lambda _source_smiles, site_limit: [0])
    monkeypatch.setattr(module, "editable_bond_sites", lambda _source_smiles, site_limit: [])
    monkeypatch.setattr(module, "execute_graph_edit_action", lambda _source_smiles, _action: "near_good")
    args = module.parse_args(
        [
            "--rows-csv",
            str(tmp_path / "rows.csv"),
            "--prediction-csv",
            str(tmp_path / "predictions.csv"),
            "--candidate-output-csv",
            str(tmp_path / "candidate_predictions.csv"),
            "--top-k-candidates",
            "2",
            "--selection-mode",
            "similarity_first",
            "--similarity-first-min-local-success-fraction",
            "1.0",
        ]
    )
    row = {
        "condition_id": "external_mumo_dpq_000000",
        "source_smiles": "source",
        "external_task_properties": "plogp",
        "external_property_directions_json": json.dumps({"plogp": "increase"}),
        "external_property_thresholds_json": json.dumps({"plogp": 1.0}),
    }

    selected, _records, candidates = module.predict_row_full(
        row,
        direct_row={"condition_id": "external_mumo_dpq_000000", "generated_smiles": "far_good"},
        args=args,
        rng=__import__("random").Random(7),
    )

    assert selected["generated_smiles"] == "near_good"
    assert len(candidates) == 2
    assert candidates[0]["condition_id"] == "external_mumo_dpq_000000"
    assert candidates[0]["candidate_rank"] == 1
    assert candidates[0]["candidate_selected"] == "True"


def test_graph_edit_policy_agent_expands_from_beam_parent(monkeypatch, tmp_path):
    module = _load_graph_edit_agent_module()
    monkeypatch.setattr(module.revise, "safe_canonical", lambda value: str(value or ""))
    monkeypatch.setattr(
        module.revise,
        "safe_tanimoto",
        lambda _left, right: {"source": 1.0, "mid": 0.92, "near_good": 0.84, "far_bad": 0.0}.get(right, 0.0),
    )
    monkeypatch.setattr(
        module.revise,
        "local_properties",
        lambda smiles: {"plogp": {"source": 0.0, "mid": 0.6, "near_good": 1.2, "far_bad": 0.0}.get(smiles, 0.0)},
    )

    def fake_plan(_row, *, source_smiles, site_limit, max_plans_per_property, planner_mode="heuristic_graph_dsl"):
        if source_smiles == "source":
            return [module.GraphEditAction("add_fragment", site=0, fragment="C", policy_score=4.0)]
        if source_smiles == "mid":
            return [module.GraphEditAction("add_atom", site=0, atom="O", policy_score=3.0)]
        return []

    def fake_execute(parent, action):
        if parent == "source" and action.fragment == "C":
            return "mid"
        if parent == "mid" and action.atom == "O":
            return "near_good"
        return ""

    monkeypatch.setattr(module, "plan_graph_edit_actions", fake_plan)
    monkeypatch.setattr(module, "execute_graph_edit_action", fake_execute)
    args = module.parse_args(
        [
            "--rows-csv",
            str(tmp_path / "rows.csv"),
            "--prediction-csv",
            str(tmp_path / "predictions.csv"),
            "--planner-mode",
            "policy_graph_dsl",
            "--planner-steps",
            "2",
            "--beam-size",
            "1",
            "--selection-mode",
            "similarity_first",
            "--similarity-first-min-local-success-fraction",
            "1.0",
        ]
    )
    row = {
        "condition_id": "a",
        "source_smiles": "source",
        "external_task_properties": "plogp",
        "external_property_directions_json": json.dumps({"plogp": "increase"}),
        "external_property_thresholds_json": json.dumps({"plogp": 1.0}),
    }

    result, records = module.predict_row(
        row,
        direct_row={"condition_id": "a", "generated_smiles": "far_bad"},
        args=args,
        rng=__import__("random").Random(7),
    )

    assert result["generated_smiles"] == "near_good"
    assert result["graph_edit_planner_mode"] == "policy_graph_dsl"
    assert result["graph_edit_planner_steps"] == 2
    assert any(record["parent_smiles"] == "mid" and record["generated_smiles"] == "near_good" for record in records)


def test_smiles_tokenization_roundtrip():
    if not TORCH_AVAILABLE:
        pytest.skip("torch is required to import direct_smiles_generation")
    from sketchmol_understanding_condition.direct_smiles_generation import detokenize_smiles, tokenize_smiles

    smiles = "CC(Cl)Br"
    tokens = tokenize_smiles(smiles)
    assert tokens == ["C", "C", "(", "Cl", ")", "Br"]
    assert detokenize_smiles(tokens) == smiles


def test_conditioned_smiles_decoder_forward_and_generate():
    if not TORCH_AVAILABLE:
        pytest.skip("torch is required for the direct SMILES decoder")
    from sketchmol_understanding_condition.direct_smiles_generation import ConditionedSmilesDecoder, build_vocabulary, tokenize_smiles

    vocab = build_vocabulary(["CCO", "CCN"])
    model = ConditionedSmilesDecoder(
        vocab_size=len(vocab.token_to_id),
        condition_dim=16,
        d_model=32,
        num_layers=1,
        num_heads=4,
        dim_feedforward=64,
        pad_id=vocab.pad_id,
        max_length=16,
    )
    condition = torch.randn(2, 3, 16)
    decoder_input = torch.tensor(
        [
            vocab.encode(tokenize_smiles("CCO"), add_bos=True),
            vocab.encode(tokenize_smiles("CCN"), add_bos=True),
        ],
        dtype=torch.long,
    )
    logits = model(condition, decoder_input)
    assert logits.shape == (2, decoder_input.shape[1], len(vocab.token_to_id))
    generated = model.generate(
        condition,
        bos_id=vocab.bos_id,
        eos_id=vocab.eos_id,
        max_new_tokens=5,
        temperature=0.9,
        top_k=4,
        top_p=0.95,
        repetition_penalty=1.1,
        no_repeat_ngram_size=3,
        min_new_tokens=1,
        suppress_ids=[vocab.token_to_id["<unk>"]],
    )
    assert generated.shape[0] == 2
    assert generated.shape[1] >= 2


def test_direct_smiles_property_rerank_prefers_strict_candidate(monkeypatch):
    if not TORCH_AVAILABLE:
        pytest.skip("torch is required to import the direct SMILES training script")

    module = _load_train_module()
    monkeypatch.setattr(module, "canonical_smiles", lambda value: str(value or "") or None)

    def fake_properties(smiles: str):
        if smiles == "far":
            return {"MolWt": 200.0, "LogP": 5.0}
        if smiles == "near":
            return {"MolWt": 101.0, "LogP": 2.1}
        return None

    monkeypatch.setattr(module, "molecular_properties", fake_properties)
    selected = module.select_generated_candidate(
        {
            "condition_properties": "MW,LogP",
            "target_MW": "100",
            "target_LogP": "2.0",
            "MW_active": "True",
            "LogP_active": "True",
        },
        ["far", "near"],
    )

    assert selected["generated_smiles"] == "near"
    assert selected["valid_candidate_count"] == 2
    assert selected["strict_fraction"] == 1.0


def test_mixed_condition_appends_property_program_tokens():
    if not TORCH_AVAILABLE:
        pytest.skip("torch is required to import the direct SMILES training script")

    module = _load_train_module()

    class DummyStore:
        input_hidden_dim = 4

        def get(self, _condition_id):
            return [[1.0, 2.0, 3.0, 4.0]]

    row = {
        "condition_id": "a",
        "condition_properties": "MW,LogP",
        "property_count": "2",
        "target_MW": "46",
        "target_LogP": "0.0",
        "MW_active": "True",
        "LogP_active": "True",
    }
    condition = module.condition_array_for_row(
        row,
        DummyStore(),
        4,
        condition_mixing_mode="append_property_program",
    )

    assert condition.shape == (10, 4)
    assert condition[0].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert any(abs(float(value)) > 0 for value in condition[1].tolist())


def test_source_mixed_condition_appends_source_and_property_program_tokens():
    if not TORCH_AVAILABLE:
        pytest.skip("torch is required to import the direct SMILES training script")

    module = _load_train_module()

    class DummyStore:
        input_hidden_dim = 8

        def get(self, _condition_id):
            return [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]

    row = {
        "condition_id": "source_edit",
        "source_smiles": "CCO",
        "condition_properties": "MW",
        "property_count": "1",
        "target_MW": "52",
        "MW_active": "True",
    }
    condition = module.condition_array_for_row(
        row,
        DummyStore(),
        8,
        condition_mixing_mode="append_source_property_program",
    )

    assert condition.shape == (13, 8)
    assert condition[0].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    assert condition[1, 0] == pytest.approx(2.0)
    assert condition[1, 1] == pytest.approx(1.0 / 3.0)
    assert condition[4, 0] == pytest.approx(0.25)


def test_property_count_curriculum_weight_prefers_high_count():
    if not TORCH_AVAILABLE:
        pytest.skip("torch is required to import the direct SMILES training script")

    module = _load_train_module()
    assert module.property_count_curriculum_weight(2, power=1.0, baseline=2.0) == pytest.approx(1.0)
    assert module.property_count_curriculum_weight(7, power=1.0, baseline=2.0) == pytest.approx(3.5)


def test_train_direct_smiles_generator_smoke(tmp_path):
    if not TORCH_AVAILABLE:
        pytest.skip("torch is required for the direct SMILES training smoke test")

    train_csv = tmp_path / "train.csv"
    eval_csv = tmp_path / "eval.csv"
    output_dir = tmp_path / "model"
    prediction_csv = tmp_path / "predictions.csv"
    rows = [
        {
            "sample_id": "a",
            "condition_id": "a",
            "target_smiles": "CCO",
            "condition_properties": "MW,LogP",
            "property_count": "2",
            "target_MW": "46",
            "target_LogP": "0.0",
            "MW_active": "True",
            "LogP_active": "True",
        },
        {
            "sample_id": "b",
            "condition_id": "b",
            "target_smiles": "CCN",
            "condition_properties": "MW,LogP",
            "property_count": "2",
            "target_MW": "45",
            "target_LogP": "0.1",
            "MW_active": "True",
            "LogP_active": "True",
        },
    ]
    _write_rows(train_csv, rows)
    _write_rows(eval_csv, rows)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_direct_smiles_generator.py",
            "--train-csv",
            str(train_csv),
            "--eval-csv",
            str(eval_csv),
            "--output-dir",
            str(output_dir),
            "--prediction-csv",
            str(prediction_csv),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--eval-batch-size",
            "2",
            "--d-model",
            "32",
            "--num-layers",
            "1",
            "--num-heads",
            "4",
            "--dim-feedforward",
            "64",
            "--max-smiles-length",
            "16",
            "--max-new-tokens",
            "8",
            "--num-samples",
            "2",
            "--temperature",
            "0.9",
            "--top-k",
            "4",
            "--top-p",
            "0.95",
            "--repetition-penalty",
            "1.1",
            "--no-repeat-ngram-size",
            "3",
            "--min-new-tokens",
            "1",
            "--device",
            "cpu",
        ],
        cwd="SketchMol-Understanding-Condition",
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / "direct_smiles_generator.pt").exists()
    prediction_rows = list(csv.DictReader(prediction_csv.open(newline="", encoding="utf-8")))
    assert len(prediction_rows) == 2
    assert "generated_smiles" in prediction_rows[0]
    assert prediction_rows[0]["method"] == "direct_smiles_mllm_sampled_rerank"
    assert "direct_valid_candidate_count" in prediction_rows[0]


def test_rl_sample_rollouts_preserves_prompt_grouping():
    if not TORCH_AVAILABLE:
        pytest.skip("torch is required for the RL rollout test")

    module = _load_rl_module()

    class FakeModel:
        def __init__(self) -> None:
            self.call_index = 0

        def generate(self, condition_tokens, **_: object):
            self.call_index += 1
            row_ids = condition_tokens[:, 0, 0].to(dtype=torch.long).cpu()
            call_ids = torch.full_like(row_ids, self.call_index)
            bos = torch.zeros_like(row_ids)
            return torch.stack([bos, row_ids, call_ids], dim=1)

    batch = {
        "condition": torch.tensor([[[1.0]], [[2.0]]], dtype=torch.float32),
        "condition_mask": torch.tensor([[True], [True]]),
    }
    generated = module.sample_rollouts(
        FakeModel(),
        batch,
        bos_id=0,
        eos_id=99,
        rollouts_per_prompt=3,
        parallel_samples=2,
        max_parallel_sequences=8,
        max_new_tokens=4,
        temperature=0.9,
        top_k=4,
        top_p=0.95,
        repetition_penalty=1.1,
        no_repeat_ngram_size=3,
        min_new_tokens=1,
        suppress_ids=[],
    )

    assert generated[:, 1].tolist() == [1, 1, 1, 2, 2, 2]
    assert generated[:, 2].tolist() == [1, 1, 2, 1, 1, 2]


def test_rl_source_similarity_reward_is_optional_and_source_conditioned(monkeypatch):
    if not TORCH_AVAILABLE:
        pytest.skip("torch is required for the RL reward test")

    module = _load_rl_module()
    monkeypatch.setattr(module, "_safe_canonical_smiles", lambda value: str(value or "") or "")
    monkeypatch.setattr(module, "property_score_components", lambda _row, _smiles: (0.0, 0.0))
    monkeypatch.setattr(module, "morgan_tanimoto", lambda _source, smiles: 1.0 if smiles == "near" else 0.0)
    row = {"source_smiles": "CCO"}

    base_near = module.reward_for_smiles(
        row,
        "near",
        reward_valid_weight=0.25,
        reward_strict_weight=0.0,
        reward_distance_weight=0.0,
        reward_distance_clip=10.0,
    )
    base_far = module.reward_for_smiles(
        row,
        "far",
        reward_valid_weight=0.25,
        reward_strict_weight=0.0,
        reward_distance_weight=0.0,
        reward_distance_clip=10.0,
    )
    source_near = module.reward_for_smiles(
        row,
        "near",
        reward_valid_weight=0.25,
        reward_strict_weight=0.0,
        reward_distance_weight=0.0,
        reward_distance_clip=10.0,
        reward_source_similarity_weight=1.0,
        reward_source_similarity_threshold=0.4,
    )
    source_far = module.reward_for_smiles(
        row,
        "far",
        reward_valid_weight=0.25,
        reward_strict_weight=0.0,
        reward_distance_weight=0.0,
        reward_distance_clip=10.0,
        reward_source_similarity_weight=1.0,
        reward_source_similarity_threshold=0.4,
    )

    assert base_near == pytest.approx(base_far)
    assert source_near > source_far


def test_rl_reward_prefers_strict_and_valid_candidates(monkeypatch):
    if not TORCH_AVAILABLE:
        pytest.skip("torch is required for the RL reward test")

    module = _load_rl_module()
    monkeypatch.setattr(module, "_safe_canonical_smiles", lambda smiles: "" if smiles == "bad" else smiles)
    monkeypatch.setattr(
        module,
        "property_score_components",
        lambda _row, smiles: (1.0, 0.2) if smiles == "good" else (0.0, 0.1),
    )

    good = module.reward_for_smiles({}, "good", reward_valid_weight=1.0, reward_strict_weight=1.0, reward_distance_weight=0.1, reward_distance_clip=10.0)
    weak = module.reward_for_smiles({}, "weak", reward_valid_weight=1.0, reward_strict_weight=1.0, reward_distance_weight=0.1, reward_distance_clip=10.0)
    bad = module.reward_for_smiles({}, "bad", reward_valid_weight=1.0, reward_strict_weight=1.0, reward_distance_weight=0.1, reward_distance_clip=10.0)

    assert good > weak
    assert bad == -1.0


def test_rl_restores_mixed_condition_mode_from_checkpoint_args():
    if not TORCH_AVAILABLE:
        pytest.skip("torch is required for the RL condition-mode restore test")
    module = _load_rl_module()
    args = module.parse_args(
        [
            "--train-csv",
            "train.csv",
            "--eval-csv",
            "eval.csv",
            "--output-dir",
            "out",
            "--resume-checkpoint",
            "resume.pt",
        ]
    )

    restored = module.resolve_condition_mixing_mode(
        args,
        {"condition_mixing_mode": "append_property_program"},
    )

    assert restored == "append_property_program"


def test_sft_warm_start_reset_flag_parses():
    if not TORCH_AVAILABLE:
        pytest.skip("torch is required to import the direct SMILES training script")
    module = _load_train_module()
    args = module.parse_args(
        [
            "--train-csv",
            "train.csv",
            "--output-dir",
            "out",
            "--resume-checkpoint",
            "resume.pt",
            "--reset-training-state",
        ]
    )

    assert args.reset_training_state is True


def test_group_relative_advantages_zero_mean_and_respects_clipping():
    if not TORCH_AVAILABLE:
        pytest.skip("torch is required for the group-relative advantage test")
    module = _load_rl_module()
    rewards = torch.tensor([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0]], dtype=torch.float32)

    centered = module.group_relative_advantages(rewards, mode="group_center", clip=0.0)
    zscore = module.group_relative_advantages(rewards, mode="group_zscore", clip=1.0)

    assert torch.allclose(centered[0], torch.tensor([-1.0, 0.0, 1.0]))
    assert torch.allclose(centered[1], torch.zeros(3))
    assert bool(torch.all(zscore.abs() <= 1.0 + 1e-6))


def test_preference_builder_prefers_hard_valid_negative():
    if not TORCH_AVAILABLE:
        pytest.skip("torch is required to import the preference builder script")
    module = _load_preference_builder_module()
    scored = [
        {
            "raw_smiles": "best",
            "canonical_smiles": "best",
            "rank": 0,
            "score": 12.0,
            "strict_fraction": 1.0,
            "normalized_property_distance": 0.0,
        },
        {
            "raw_smiles": "weak_valid",
            "canonical_smiles": "weak_valid",
            "rank": 1,
            "score": 4.0,
            "strict_fraction": 0.25,
            "normalized_property_distance": 1.0,
        },
        {
            "raw_smiles": "bad_invalid",
            "canonical_smiles": "",
            "rank": 2,
            "score": -100.0,
            "strict_fraction": 0.0,
            "normalized_property_distance": float("inf"),
        },
    ]
    pair = module.select_preference_pair({}, scored, rejected_strategy="hard_valid", min_score_gap=0.5)
    assert pair is not None
    assert pair["chosen_smiles"] == "best"
    assert pair["rejected_smiles"] == "weak_valid"
    assert pair["rejected_valid"] == "True"


def test_dpo_logit_prefers_chosen_over_rejected():
    if not TORCH_AVAILABLE:
        pytest.skip("torch is required for the DPO helper test")

    policy_chosen = torch.tensor([4.0, 3.0])
    policy_rejected = torch.tensor([1.0, 2.0])
    ref_chosen = torch.tensor([3.5, 2.5])
    ref_rejected = torch.tensor([1.0, 2.0])
    logits = 0.1 * ((policy_chosen - policy_rejected) - (ref_chosen - ref_rejected))
    assert bool((logits > 0).all())


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_train_module():
    path = Path("SketchMol-Understanding-Condition/scripts/train_direct_smiles_generator.py")
    spec = importlib.util.spec_from_file_location("train_direct_smiles_generator", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_rl_module():
    path = Path("SketchMol-Understanding-Condition/scripts/train_direct_smiles_generator_rl.py")
    spec = importlib.util.spec_from_file_location("train_direct_smiles_generator_rl", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_preference_builder_module():
    path = Path("SketchMol-Understanding-Condition/scripts/build_direct_smiles_preference_dataset.py")
    spec = importlib.util.spec_from_file_location("build_direct_smiles_preference_dataset", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_agentic_revise_module():
    path = Path("SketchMol-Understanding-Condition/scripts/build_external_agentic_revise_predictions.py")
    spec = importlib.util.spec_from_file_location("build_external_agentic_revise_predictions", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_graph_edit_agent_module():
    path = Path("SketchMol-Understanding-Condition/scripts/build_external_graph_edit_agent_predictions.py")
    spec = importlib.util.spec_from_file_location("build_external_graph_edit_agent_predictions", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
