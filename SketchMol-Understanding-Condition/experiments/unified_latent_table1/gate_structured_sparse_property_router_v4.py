#!/usr/bin/env python3
"""Aggregate four completed v4 arms and apply the preregistered science gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence


PROTOCOL = "structured_sparse_property_router_v4_science_gate"
ARM_PROTOCOL = "train_only_structured_sparse_property_router_v4"
ARMS = ("full", "no_lora", "no_token_slots", "no_composition")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-manifest", required=True, type=Path)
    parser.add_argument("--arms-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_manifest(path: Path) -> dict[str, object]:
    payload = read_json(path)
    if payload.get("protocol") != "train_only_structured_sparse_property_router_v4":
        raise ValueError("Structured-router gate manifest protocol drift")
    actual = file_sha256(Path(__file__).resolve())
    if payload.get("gate_implementation_sha256") != actual:
        raise ValueError(
            f"Gate implementation drift: expected {payload.get('gate_implementation_sha256')}, "
            f"found {actual}"
        )
    return payload


def nested(payload: Mapping[str, object], *keys: str) -> float:
    value: object = payload
    for key in keys:
        value = dict(value)[key]
    return float(value)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = read_manifest(args.protocol_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "gate_summary.json"
    if summary_path.exists():
        raise ValueError(f"Completed structured-router gate exists: {summary_path}")
    summaries: dict[str, dict[str, object]] = {}
    hashes: dict[str, str] = {}
    for arm in ARMS:
        path = args.arms_root / arm / "summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing completed arm summary: {path}")
        payload = read_json(path)
        if payload.get("protocol") != ARM_PROTOCOL:
            raise ValueError(f"Arm {arm} protocol drift")
        if payload.get("arm") != arm or payload.get("execution_status") != "completed":
            raise ValueError(f"Arm {arm} execution contract drift")
        contract = dict(payload["contract"])
        required_false = (
            "support_threshold_search",
            "molecule_generation",
            "molecular_candidate_ranking",
            "oracle_selection",
            "official_test_access",
        )
        if any(bool(contract[key]) for key in required_false):
            raise ValueError(f"Arm {arm} violated a forbidden-access contract")
        if not bool(contract["explicit_cardinality"]) or not bool(
            contract["exact_topk_support"]
        ):
            raise ValueError(f"Arm {arm} lacks structured sparse routing")
        summaries[arm] = payload
        hashes[f"{arm}_summary_sha256"] = file_sha256(path)
    full = summaries["full"]
    no_lora = summaries["no_lora"]
    no_slots = summaries["no_token_slots"]
    no_composition = summaries["no_composition"]
    gates = dict(manifest["science_gates"])
    full_matched_support = nested(
        full, "graph_probe_routing", "matched", "exact_support_rate"
    )
    full_matched_precision = nested(
        full, "graph_probe_routing", "matched", "support_precision"
    )
    full_matched_recall = nested(
        full, "graph_probe_routing", "matched", "support_recall"
    )
    full_multi_support = nested(
        full, "multicardinality_probe", "exact_support_rate"
    )
    full_multi_cardinality = nested(
        full, "multicardinality_probe", "cardinality_exact_rate"
    )
    full_multi_sign = nested(
        full, "multicardinality_probe", "active_sign_accuracy"
    )
    full_token_ratio = nested(
        full, "graph_probe_tokens", "language_mse_ratio_vs_intercept"
    )
    full_flow_ratio = nested(
        full, "graph_probe_flow", "language_flow_ratio_vs_intercept"
    )
    full_flow_advantage = nested(full, "graph_probe_flow", "matched_flow_advantage")
    oracle_flow_error = nested(
        full, "graph_probe_flow", "oracle_canonical_flow_relative_error"
    )
    lora_token_delta = nested(
        no_lora, "graph_probe_tokens", "language_mse_ratio_vs_intercept"
    ) - full_token_ratio
    token_slot_support_delta = full_matched_support - nested(
        no_slots, "graph_probe_routing", "matched", "exact_support_rate"
    )
    composition_support_delta = full_multi_support - nested(
        no_composition, "multicardinality_probe", "exact_support_rate"
    )
    checks = {
        "full_matched_support_rate": full_matched_support
        >= float(gates["full_matched_support_rate"]),
        "full_matched_support_precision": full_matched_precision
        >= float(gates["full_matched_support_precision"]),
        "full_matched_support_recall": full_matched_recall
        >= float(gates["full_matched_support_recall"]),
        "full_multicardinality_support_rate": full_multi_support
        >= float(gates["full_multicardinality_support_rate"]),
        "full_multicardinality_exact_rate": full_multi_cardinality
        >= float(gates["full_multicardinality_exact_rate"]),
        "full_multicardinality_sign_accuracy": full_multi_sign
        >= float(gates["full_multicardinality_sign_accuracy"]),
        "full_language_token_ratio": full_token_ratio
        <= float(gates["full_language_token_ratio"]),
        "full_language_flow_ratio": full_flow_ratio
        <= float(gates["full_language_flow_ratio"]),
        "full_matched_flow_advantage": full_flow_advantage
        >= float(gates["full_matched_flow_advantage"]),
        "oracle_canonical_flow_relative_error": oracle_flow_error
        <= float(gates["oracle_canonical_flow_relative_error"]),
        "lora_ablation_delta": lora_token_delta
        >= float(gates["lora_ablation_delta"]),
        "token_slot_ablation_delta": token_slot_support_delta
        >= float(gates["token_slot_ablation_delta"]),
        "composition_ablation_delta": composition_support_delta
        >= float(gates["composition_ablation_delta"]),
    }
    passed = all(checks.values())
    summary = {
        "protocol": PROTOCOL,
        "execution_status": "completed",
        "science_gate": {"passed": passed, "checks": checks, "thresholds": gates},
        "headline": {
            "full_matched_support_rate": full_matched_support,
            "full_matched_support_precision": full_matched_precision,
            "full_matched_support_recall": full_matched_recall,
            "full_multicardinality_support_rate": full_multi_support,
            "full_multicardinality_exact_rate": full_multi_cardinality,
            "full_multicardinality_sign_accuracy": full_multi_sign,
            "full_language_token_ratio": full_token_ratio,
            "full_language_flow_ratio": full_flow_ratio,
            "full_matched_flow_advantage": full_flow_advantage,
            "oracle_canonical_flow_relative_error": oracle_flow_error,
        },
        "ablation_deltas": {
            "no_lora_token_ratio_minus_full": lora_token_delta,
            "full_minus_no_token_slots_support_rate": token_slot_support_delta,
            "full_minus_no_composition_multicardinality_support_rate": composition_support_delta,
        },
        "decision": (
            "unlock_target_isolated_exact_n20_generation"
            if passed
            else "stop_before_molecule_generation"
        ),
        "arm_summary_hashes": hashes,
        "contract": {
            "execution_and_science_gate_jobs_separated": True,
            "support_threshold_search": False,
            "molecule_generation": False,
            "generation_target_access": False,
            "official_test_access": False,
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if passed:
        generation_contract = dict(manifest["generation_unlock_contract"])
        unlock = {
            "protocol": "target_isolated_exact_n20_generation_unlock_v4",
            "status": "unlocked_not_executed",
            "source_gate_sha256": file_sha256(summary_path),
            **generation_contract,
        }
        (args.output_dir / "generation_unlock.json").write_text(
            json.dumps(unlock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
