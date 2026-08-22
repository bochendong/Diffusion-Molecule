#!/usr/bin/env python3
"""Validate a reconstructed P1 Group-RL checkpoint against historical job logs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence


EXPECTED = {
    "two_p_to_seven_p": {
        "source_job": "16583941",
        "batches": 1500,
        "loss": 0.7516919071674347,
        "pg_loss": -0.05318761052377522,
        "sft_loss": 0.8077205394903819,
        "kl_loss": -0.002841021431920429,
        "mean_reward": -0.44911612481623886,
        "eval_mean_reward": -0.17568006989189444,
    },
    "ood": {
        "source_job": "16742519",
        "batches": 1000,
        "loss": 0.8673587768077851,
        "pg_loss": -0.036179792444221676,
        "sft_loss": 0.906621556699276,
        "kl_loss": -0.003082986883353442,
        "mean_reward": -0.889485376894474,
        "eval_mean_reward": -0.8207629565149546,
    },
}
TOLERANCE = {
    "loss": 0.015,
    "pg_loss": 0.015,
    "sft_loss": 0.015,
    "kl_loss": 0.001,
    "mean_reward": 0.06,
    "eval_mean_reward": 0.06,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--benchmark", required=True, choices=tuple(EXPECTED))
    parser.add_argument("--output-json", required=True, type=Path)
    return parser.parse_args(argv)


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_payload(payload: Mapping[str, object], benchmark: str) -> dict[str, object]:
    expected = EXPECTED[benchmark]
    args = dict(payload.get("args") or {})
    history = list(payload.get("history") or [])
    failures: list[str] = []
    contract = {
        "seed": 7,
        "epochs": 1,
        "rollouts_per_prompt": 16,
        "condition_mixing_mode": "append_property_program",
        "advantage_mode": "group_zscore",
        "reference_kl_weight": 0.05,
        "sft_weight": 1.0,
    }
    for key, wanted in contract.items():
        actual = args.get(key)
        if actual != wanted:
            failures.append(f"args.{key}: expected {wanted!r}, got {actual!r}")
    if len(history) != 1:
        failures.append(f"history: expected exactly one epoch, got {len(history)}")
        record: Mapping[str, object] = history[-1] if history else {}
    else:
        record = history[0]
    if int(record.get("epoch") or 0) != 1:
        failures.append(f"history.epoch: expected 1, got {record.get('epoch')!r}")
    if int(record.get("batches") or 0) != int(expected["batches"]):
        failures.append(f"history.batches: expected {expected['batches']}, got {record.get('batches')!r}")
    deviations: dict[str, float] = {}
    for metric, tolerance in TOLERANCE.items():
        actual = float(record.get(metric, math.nan))
        target = float(expected[metric])
        deviation = abs(actual - target)
        deviations[metric] = deviation
        if not math.isfinite(actual) or deviation > tolerance:
            failures.append(
                f"history.{metric}: expected {target:.9f} +/- {tolerance:.3f}, got {actual:.9f}"
            )
    return {
        "benchmark": benchmark,
        "source_job": expected["source_job"],
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "expected_history": expected,
        "actual_history": dict(record),
        "absolute_deviations": deviations,
        "tolerances": TOLERANCE,
        "contract": contract,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    import torch

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    result = validate_payload(payload, args.benchmark)
    result["checkpoint"] = str(args.checkpoint)
    result["checkpoint_sha256"] = checkpoint_sha256(args.checkpoint)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
