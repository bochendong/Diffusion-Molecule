#!/usr/bin/env python3
"""Apply the final V11 science gate while keeping negative science Complete."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence


PROTOCOL = "train_only_elementwise_additive_semantic_set_compiler_v11"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-manifest", required=True, type=Path)
    parser.add_argument("--execution-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    preregistration = read_json(args.protocol_manifest)
    if preregistration.get("protocol") != PROTOCOL:
        raise ValueError("V11 gate protocol drift")
    if preregistration.get("gate_implementation_sha256") != file_sha256(Path(__file__).resolve()):
        raise ValueError("V11 gate implementation drift")
    execution = read_json(args.execution_summary)
    if execution.get("protocol") != PROTOCOL or execution.get("execution_status") != "completed":
        raise ValueError("V11 execution summary incomplete")
    contract = dict(execution["contract"])
    expected = {
        "common_llm_frozen": True,
        "singleton_language_fit_only": True,
        "composition_supervision": False,
        "v10_probe_reuse": False,
        "source_manifest_role": "constraint_text_and_sources_without_targets",
        "molecule_target_path_accepted": False,
        "molecule_target_access": False,
        "property_oracle_access": False,
        "generation_target_access": False,
        "molecule_generation": False,
        "molecular_candidate_ranking": False,
        "oracle_selection": False,
        "threshold_search": False,
        "official_test_access": False,
        "final_representation_attempt": True,
    }
    contract_checks = {name: contract.get(name) == value for name, value in expected.items()}
    preview = dict(execution["representation_gate_preview"])
    science_checks = {str(name): bool(value) for name, value in dict(preview["checks"]).items()}
    passed = all(contract_checks.values()) and all(science_checks.values())
    failures = sorted(
        [name for name, value in contract_checks.items() if not value]
        + [name for name, value in science_checks.items() if not value]
    )
    decision = (
        "advance_frozen_elementwise_compiler_to_three_exact_n20_pilots"
        if passed
        else "final_stop_llm_core_generation_mechanism_claim"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "gate_summary.json"
    if output.exists():
        raise ValueError(f"Completed V11 gate exists: {output}")
    payload = {
        "protocol": PROTOCOL,
        "stage": "separate_final_scientific_gate",
        "execution_status": "completed",
        "decision": decision,
        "science_gate": {
            "passed": passed,
            "checks": science_checks,
            "thresholds": preview["thresholds"],
            "failures": failures,
        },
        "contract_gate": {"passed": all(contract_checks.values()), "checks": contract_checks},
        "execution_summary_sha256": file_sha256(args.execution_summary),
        "failure_policy": {
            "if_failed": "no_more_representation_retries_and_no_molecule_pilots",
            "if_passed": "freeze_checkpoint_then_run_three_small_single_seed_exact_n20_pilots",
        },
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
