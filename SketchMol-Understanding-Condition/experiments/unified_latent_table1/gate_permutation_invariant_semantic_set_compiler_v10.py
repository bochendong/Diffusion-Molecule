#!/usr/bin/env python3
"""Apply the preregistered V10 science gate without failing the execution job."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence


PROTOCOL = "train_only_permutation_invariant_semantic_set_compiler_v10"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        raise ValueError("V10 gate protocol drift")
    actual_gate_hash = file_sha256(Path(__file__).resolve())
    if preregistration.get("gate_implementation_sha256") != actual_gate_hash:
        raise ValueError("V10 gate implementation drift")
    execution = read_json(args.execution_summary)
    if (
        execution.get("protocol") != PROTOCOL
        or execution.get("execution_status") != "completed"
    ):
        raise ValueError("V10 execution summary is incomplete")
    contract = dict(execution["contract"])
    expected_contract: Mapping[str, object] = {
        "common_llm_frozen": True,
        "numeric_canonical_distillation": True,
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
        "token_slot_training": False,
    }
    contract_checks = {
        name: contract.get(name) == expected for name, expected in expected_contract.items()
    }
    preview = dict(execution["representation_gate_preview"])
    science_checks = {str(name): bool(value) for name, value in dict(preview["checks"]).items()}
    passed = all(contract_checks.values()) and all(science_checks.values())
    decision = (
        "advance_semantic_set_compiler_to_three_exact_n20_pilots"
        if passed
        else "stop_llm_core_generation_claim_before_molecule_pilots"
    )
    failures = sorted(
        [name for name, value in contract_checks.items() if not value]
        + [name for name, value in science_checks.items() if not value]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "gate_summary.json"
    if output_path.exists():
        raise ValueError(f"Completed V10 gate exists: {output_path}")
    payload = {
        "protocol": PROTOCOL,
        "stage": "separate_scientific_gate",
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
        "next_stage_contract": {
            "only_if_passed": True,
            "pilots": ["mumo_ood_n20", "moledit_table1_n20", "denovo_2p7p_n20"],
            "single_seed_small_subset_first": True,
            "exact_raw_attempts_per_condition": 20,
            "candidate_pool_before_selection": 20,
            "molecular_candidate_ranking": False,
            "oracle_selection": False,
            "official_test_access": False,
        },
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
