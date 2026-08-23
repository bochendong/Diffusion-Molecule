from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "p1_property_program_group_rl" / "evaluate_p1_sampling_scaling.py"
VALIDATOR_SCRIPT = ROOT / "experiments" / "p1_property_program_group_rl" / "validate_p1_recovered_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("p1_sampling", SCRIPT)
assert SPEC and SPEC.loader
p1 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = p1
SPEC.loader.exec_module(p1)
VALIDATOR_SPEC = importlib.util.spec_from_file_location("p1_validator", VALIDATOR_SCRIPT)
assert VALIDATOR_SPEC and VALIDATOR_SPEC.loader
validator = importlib.util.module_from_spec(VALIDATOR_SPEC)
sys.modules[VALIDATOR_SPEC.name] = validator
VALIDATOR_SPEC.loader.exec_module(validator)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def candidate_rows(condition_ids: list[str], successes: dict[str, set[int]]) -> list[dict[str, object]]:
    rows = []
    for condition_id in condition_ids:
        for index in range(256):
            strict = index in successes.get(condition_id, set())
            rows.append(
                {
                    "condition_id": condition_id,
                    "direct_candidate_index": index,
                    "direct_candidate_raw_smiles": f"C{index + 1}" if index % 11 else "",
                    "direct_candidate_canonical_smiles": f"C{index + 1}" if index % 11 else "",
                    "direct_candidate_strict_fraction": 1.0 if strict else 0.0,
                }
            )
    return rows


def test_pass_at_k_estimator_boundaries() -> None:
    assert p1.estimated_pass_at_k(256, 0, 20) == 0.0
    assert p1.estimated_pass_at_k(256, 256, 20) == 1.0
    assert p1.estimated_pass_at_k(4, 1, 4) == 1.0
    assert 0.0 < p1.estimated_pass_at_k(256, 4, 8) < 1.0


def test_recovered_checkpoint_guard_rejects_metric_drift() -> None:
    expected = dict(validator.EXPECTED["two_p_to_seven_p"])
    payload = {
        "args": {
            "seed": 7,
            "epochs": 1,
            "rollouts_per_prompt": 16,
            "condition_mixing_mode": "append_property_program",
            "advantage_mode": "group_zscore",
            "reference_kl_weight": 0.05,
            "sft_weight": 1.0,
        },
        "history": [{"epoch": 1, **expected}],
    }
    passed = validator.validate_payload(payload, "two_p_to_seven_p")
    assert passed["status"] == "pass"
    payload["history"][0]["eval_mean_reward"] = 0.5
    failed = validator.validate_payload(payload, "two_p_to_seven_p")
    assert failed["status"] == "fail"
    assert any("eval_mean_reward" in item for item in failed["failures"])


def test_p1_end_to_end_uses_raw_first_candidate_and_emits_gate(tmp_path: Path) -> None:
    two_p_ids = ["2p", "3p", "4p", "5p", "6p", "7p"]
    ood_ids = ["extreme", "rare", "reverse"]
    two_p_eval = [
        {"condition_id": condition_id, "property_count": count, "ood_bucket": ""}
        for count, condition_id in enumerate(two_p_ids, start=2)
    ]
    ood_eval = [
        {"condition_id": "extreme", "property_count": 1, "ood_bucket": "forward_extreme"},
        {"condition_id": "rare", "property_count": 2, "ood_bucket": "rare_combo"},
        {"condition_id": "reverse", "property_count": 4, "ood_bucket": "reverse_stimulation"},
    ]
    write_csv(tmp_path / "two_p_eval.csv", two_p_eval)
    write_csv(tmp_path / "ood_eval.csv", ood_eval)

    sft_two_p = {key: {30} for key in two_p_ids}
    rl_two_p = {key: {0, 2, 6, 15, 30} for key in two_p_ids}
    sft_ood = {key: {30} for key in ood_ids}
    rl_ood = {key: {0, 3, 7, 18, 30} for key in ood_ids}
    paths = {
        "two_p_sft": tmp_path / "two_p_sft.csv",
        "two_p_rl": tmp_path / "two_p_rl.csv",
        "ood_sft": tmp_path / "ood_sft.csv",
        "ood_rl": tmp_path / "ood_rl.csv",
    }
    write_csv(paths["two_p_sft"], candidate_rows(two_p_ids, sft_two_p))
    write_csv(paths["two_p_rl"], candidate_rows(two_p_ids, rl_two_p))
    write_csv(paths["ood_sft"], candidate_rows(ood_ids, sft_ood))
    write_csv(paths["ood_rl"], candidate_rows(ood_ids, rl_ood))

    output_dir = tmp_path / "out"
    rc = p1.main(
        [
            "--two-p-seven-p-eval-csv",
            str(tmp_path / "two_p_eval.csv"),
            "--ood-eval-csv",
            str(tmp_path / "ood_eval.csv"),
            "--two-p-seven-p-sft-candidates",
            str(paths["two_p_sft"]),
            "--two-p-seven-p-group-rl-candidates",
            str(paths["two_p_rl"]),
            "--ood-sft-candidates",
            str(paths["ood_sft"]),
            "--ood-group-rl-candidates",
            str(paths["ood_rl"]),
            "--output-dir",
            str(output_dir),
            "--bootstrap-resamples",
            "50",
        ]
    )
    assert rc == 0
    gate = json.loads((output_dir / "p1_gate.json").read_text(encoding="utf-8"))
    assert gate["verdict"] in {"strong_single_seed_signal", "promising_single_seed_signal"}
    report = (output_dir / "p1_report.md").read_text(encoding="utf-8")
    assert "`k=1` is the first raw draw" in report
    assert "property-reranked selected molecule is diagnostic-only" in report
    assert "Validity metric alignment" in report
    validity_rows = list(csv.DictReader((output_dir / "p1_validity_audit.csv").open(encoding="utf-8")))
    row = next(
        item
        for item in validity_rows
        if item["benchmark"] == "two_p_to_seven_p"
        and item["model"] == "sft"
        and item["group_type"] == "overall"
        and item["candidate_budget"] == "20"
    )
    assert float(row["raw_candidate_validity"]) == pytest.approx(0.9)
    assert float(row["selected_validity_at_k"]) == 1.0
    assert float(row["empty_raw_fraction"]) == pytest.approx(0.1)
