from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "unified_latent_table1"
COLLECTOR = EXPERIMENT / "collect_table1_anyk_robustness.py"
PARTICLE_RUNNER = EXPERIMENT / "run_finalize_existing_particle_coverage.sh"
ANYK_RUNNER = EXPERIMENT / "run_table1_anyk_robustness.sh"
SUBMITTER = EXPERIMENT / "submit_paper_cpu_audits.sh"


def load_collector():
    spec = importlib.util.spec_from_file_location("table1_anyk_robustness", COLLECTOR)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_jobs_are_cpu_only_and_parallel() -> None:
    source = SUBMITTER.read_text(encoding="utf-8")
    assert "uca-particle-finalize" in source
    assert "uca-table1-anyk-audit" in source
    assert "--gres" not in source
    assert "--dependency" not in source
    assert "--cpus-per-task=4" in source
    assert "--mem=16G" in source


def test_runners_only_score_existing_frozen_candidates() -> None:
    particle = PARTICLE_RUNNER.read_text(encoding="utf-8")
    anyk = ANYK_RUNNER.read_text(encoding="utf-8")
    assert "eval_b41_particle_coverage.py" not in particle
    assert "collect_anyk_budget.py" in particle
    assert "collect_b41_particle_coverage.py" in particle
    assert "collect_anyk_budget.py" in anyk
    assert "collect_table1_anyk_robustness.py" in anyk
    assert "reuse_existing_curve=" in anyk
    assert "d0_b41_table1_n20_candidates.csv" in anyk
    assert "b41_canonical_table1_n20_candidates.csv" in anyk
    assert "d3_event_kernel_energy_table1_n20_candidates.csv" in anyk


def test_comparison_collector_records_no_selection_contract(tmp_path: Path, monkeypatch) -> None:
    module = load_collector()
    ks = [1, 2, 5, 10, 20]
    curve = {
        "model": "test",
        "ks": ks,
        "candidate_conditions": 2,
        "evaluated_conditions": 2,
        "real5_anyk_t0_65": {str(k): 0.5 for k in ks},
        "gsk3b_anyk_t0_65": {str(k): 0.25 for k in ks},
        "auc_real5_t0_65": 0.5,
        "auc_gsk3b_t0_65": 0.25,
        "mean_unique_smiles": 10.0,
    }
    paths = []
    candidates = []
    for index in range(3):
        curve_path = tmp_path / f"curve-{index}.json"
        curve_path.write_text(json.dumps(curve), encoding="utf-8")
        paths.append(curve_path)
        candidate = tmp_path / f"candidate-{index}.csv"
        candidate.write_text("id,smiles\n1,CC\n", encoding="utf-8")
        candidates.append(candidate)
    output = tmp_path / "summary.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            str(COLLECTOR),
            "--b41-curve",
            str(paths[0]),
            "--canonical-curve",
            str(paths[1]),
            "--d3-curve",
            str(paths[2]),
            "--b41-candidates",
            str(candidates[0]),
            "--canonical-candidates",
            str(candidates[1]),
            "--d3-candidates",
            str(candidates[2]),
            "--output-json",
            str(output),
        ],
    )
    assert module.main() == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["contract"]["new_generation"] is False
    assert payload["contract"]["model_training"] is False
    assert payload["contract"]["molecular_candidate_ranking"] is False
    assert payload["contract"]["oracle_selection"] is False
    assert payload["contract"]["exact_max_attempts_per_condition"] == 20
    assert payload["coverage_alignment"]["passed"] is True
    assert payload["coverage_alignment"]["minimum_relative_coverage"] == 1.0
    assert payload["claim_policy"] == "diagnostic_only_no_method_selection_or_retuning"


def test_comparison_accepts_one_missing_historical_condition(tmp_path: Path, monkeypatch) -> None:
    module = load_collector()
    ks = [1, 2, 5, 10, 20]
    paths = []
    candidates = []
    for index, evaluated in enumerate((988, 988, 987)):
        curve = {
            "model": f"arm-{index}",
            "ks": ks,
            "candidate_conditions": 997 - index,
            "evaluated_conditions": evaluated,
            "real5_anyk_t0_65": {str(k): 0.5 for k in ks},
            "gsk3b_anyk_t0_65": {str(k): 0.25 for k in ks},
            "auc_real5_t0_65": 0.5,
            "auc_gsk3b_t0_65": 0.25,
            "mean_unique_smiles": 10.0,
        }
        curve_path = tmp_path / f"curve-{index}.json"
        curve_path.write_text(json.dumps(curve), encoding="utf-8")
        paths.append(curve_path)
        candidate = tmp_path / f"candidate-{index}.csv"
        candidate.write_text("id,smiles\n1,CC\n", encoding="utf-8")
        candidates.append(candidate)
    output = tmp_path / "summary.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            str(COLLECTOR),
            "--b41-curve",
            str(paths[0]),
            "--canonical-curve",
            str(paths[1]),
            "--d3-curve",
            str(paths[2]),
            "--b41-candidates",
            str(candidates[0]),
            "--canonical-candidates",
            str(candidates[1]),
            "--d3-candidates",
            str(candidates[2]),
            "--output-json",
            str(output),
        ],
    )
    assert module.main() == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["coverage_alignment"]["evaluated_conditions"] == {
        "b41": 988,
        "canonical": 988,
        "d3_grpo": 987,
    }
    assert payload["coverage_alignment"]["minimum_relative_coverage"] > 0.998
