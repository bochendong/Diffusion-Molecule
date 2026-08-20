#!/usr/bin/env python3
"""Apply the preregistered V5 science gate after all arm executions finish."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence


PROTOCOL = "mass_conserving_property_set_router_v5_science_gate"
ARM_PROTOCOL = "train_only_mass_conserving_property_set_router_v5"
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
    if payload.get("protocol") != ARM_PROTOCOL:
        raise ValueError("V5 gate manifest protocol drift")
    actual = file_sha256(Path(__file__).resolve())
    if payload.get("gate_implementation_sha256") != actual:
        raise ValueError(
            f"V5 gate implementation drift: expected "
            f"{payload.get('gate_implementation_sha256')}, found {actual}"
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
        raise ValueError(f"Completed V5 science gate exists: {summary_path}")
    summaries: dict[str, dict[str, object]] = {}
    hashes: dict[str, str] = {}
    lineage_hashes: set[str] = set()
    for arm in ARMS:
        path = args.arms_root / arm / "summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing completed V5 arm summary: {path}")
        payload = read_json(path)
        if (
            payload.get("protocol") != ARM_PROTOCOL
            or payload.get("arm") != arm
            or payload.get("execution_status") != "completed"
        ):
            raise ValueError(f"V5 arm {arm} execution contract drift")
        contract = dict(payload["contract"])
        if any(
            bool(contract[key])
            for key in (
                "support_threshold_search",
                "molecule_generation",
                "molecular_candidate_ranking",
                "oracle_selection",
                "official_test_access",
                "categorical_cardinality_head",
            )
        ):
            raise ValueError(f"V5 arm {arm} violated a forbidden contract")
        if not all(
            bool(contract[key])
            for key in (
                "mass_conserving_cardinality",
                "fresh_probe_excludes_v4_fit_and_probe",
                "reused_graph_probe_is_regression_only",
            )
        ):
            raise ValueError(f"V5 arm {arm} lacks the structural/fresh-probe contract")
        lineage = dict(payload["fresh_probe_lineage"])
        if (
            int(lineage["fresh_v4_fit_overlap"]) != 0
            or int(lineage["fresh_v4_probe_overlap"]) != 0
            or int(lineage["fresh_v4_fit_property_set_overlap"]) != 0
            or int(lineage["fresh_v4_probe_property_set_overlap"]) != 0
        ):
            raise ValueError(f"V5 arm {arm} fresh probe overlaps V4")
        lineage_hashes.add(str(lineage["fresh_probe_sha256"]))
        summaries[arm] = payload
        hashes[f"{arm}_summary_sha256"] = file_sha256(path)
    if len(lineage_hashes) != 1:
        raise ValueError("V5 arms did not use the identical fresh probe")

    full = summaries["full"]
    no_lora = summaries["no_lora"]
    no_slots = summaries["no_token_slots"]
    no_composition = summaries["no_composition"]
    gates = dict(manifest["science_gates"])
    full_fresh_support = nested(
        full, "multicardinality_probe", "exact_support_rate"
    )
    full_fresh_cardinality = nested(
        full, "multicardinality_probe", "cardinality_exact_rate"
    )
    full_fresh_sign = nested(
        full, "multicardinality_probe", "active_sign_accuracy"
    )
    full_fresh_signed_set = nested(
        full, "multicardinality_probe", "exact_signed_support_rate"
    )
    full_graph_support = nested(
        full, "graph_probe_routing", "matched", "exact_support_rate"
    )
    full_graph_signed_set = nested(
        full, "graph_probe_routing", "matched", "exact_signed_support_rate"
    )
    full_token_ratio = nested(
        full, "graph_probe_tokens", "language_mse_ratio_vs_intercept"
    )
    full_flow_ratio = nested(
        full, "graph_probe_flow", "language_flow_ratio_vs_intercept"
    )
    full_flow_advantage = nested(
        full, "graph_probe_flow", "matched_flow_advantage"
    )
    lora_delta = full_fresh_signed_set - nested(
        no_lora, "multicardinality_probe", "exact_signed_support_rate"
    )
    token_slot_delta = full_fresh_signed_set - nested(
        no_slots, "multicardinality_probe", "exact_signed_support_rate"
    )
    composition_delta = full_fresh_support - nested(
        no_composition, "multicardinality_probe", "exact_support_rate"
    )
    checks = {
        "full_fresh_exact_support_rate": full_fresh_support
        >= float(gates["full_fresh_exact_support_rate"]),
        "full_fresh_cardinality_exact_rate": full_fresh_cardinality
        >= float(gates["full_fresh_cardinality_exact_rate"]),
        "full_fresh_active_sign_accuracy": full_fresh_sign
        >= float(gates["full_fresh_active_sign_accuracy"]),
        "full_fresh_exact_signed_support_rate": full_fresh_signed_set
        >= float(gates["full_fresh_exact_signed_support_rate"]),
        "full_graph_replay_exact_support_rate": full_graph_support
        >= float(gates["full_graph_replay_exact_support_rate"]),
        "full_graph_replay_exact_signed_support_rate": full_graph_signed_set
        >= float(gates["full_graph_replay_exact_signed_support_rate"]),
        "full_graph_token_ratio": full_token_ratio
        <= float(gates["full_graph_token_ratio"]),
        "full_graph_flow_ratio": full_flow_ratio
        <= float(gates["full_graph_flow_ratio"]),
        "full_graph_flow_advantage": full_flow_advantage
        >= float(gates["full_graph_flow_advantage"]),
        "lora_ablation_signed_set_delta": lora_delta
        >= float(gates["lora_ablation_signed_set_delta"]),
        "token_slot_ablation_signed_set_delta": token_slot_delta
        >= float(gates["token_slot_ablation_signed_set_delta"]),
        "composition_ablation_support_delta": composition_delta
        >= float(gates["composition_ablation_support_delta"]),
    }
    passed = all(checks.values())
    summary = {
        "protocol": PROTOCOL,
        "execution_status": "completed",
        "science_gate": {"passed": passed, "checks": checks, "thresholds": gates},
        "headline": {
            "full_fresh_exact_support_rate": full_fresh_support,
            "full_fresh_cardinality_exact_rate": full_fresh_cardinality,
            "full_fresh_active_sign_accuracy": full_fresh_sign,
            "full_fresh_exact_signed_support_rate": full_fresh_signed_set,
            "full_graph_replay_exact_support_rate": full_graph_support,
            "full_graph_replay_exact_signed_support_rate": full_graph_signed_set,
            "full_graph_token_ratio": full_token_ratio,
            "full_graph_flow_ratio": full_flow_ratio,
            "full_graph_flow_advantage": full_flow_advantage,
        },
        "ablation_deltas": {
            "full_minus_no_lora_fresh_signed_set": lora_delta,
            "full_minus_no_token_slots_fresh_signed_set": token_slot_delta,
            "full_minus_no_composition_fresh_support": composition_delta,
        },
        "decision": (
            "unlock_target_isolated_exact_n20_generation"
            if passed
            else "stop_before_molecule_generation"
        ),
        "arm_summary_hashes": hashes,
        "fresh_probe_sha256": next(iter(lineage_hashes)),
        "contract": {
            "execution_and_science_gate_jobs_separated": True,
            "categorical_cardinality_head": False,
            "support_threshold_search": False,
            "fresh_probe_excludes_v4_fit_and_probe": True,
            "reused_graph_probe_is_regression_only": True,
            "molecule_generation": False,
            "generation_target_access": False,
            "official_test_access": False,
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if passed:
        unlock = {
            "protocol": "target_isolated_exact_n20_generation_unlock_v5",
            "status": "unlocked_not_executed",
            "source_gate_sha256": file_sha256(summary_path),
            **dict(manifest["generation_unlock_contract"]),
        }
        (args.output_dir / "generation_unlock.json").write_text(
            json.dumps(unlock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
