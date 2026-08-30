#!/usr/bin/env python3
"""Collect the P33 matched joint-vs-separate decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def summarize(
    evals: Mapping[str, dict[str, object]], trains: Mapping[str, dict[str, object]]
) -> dict[str, object]:
    joint = evals["joint"]["aggregate"]
    denovo = evals["denovo"]["aggregate"]
    edit = evals["edit"]["aggregate"]
    deltas = {
        "denovo_strict_joint_minus_specialist": (
            float(joint["denovo_strict_macro"]) - float(denovo["denovo_strict_macro"])
        ),
        "edit_strict_joint_minus_specialist": (
            float(joint["edit_strict_065_macro"]) - float(edit["edit_strict_065_macro"])
        ),
        "denovo_valid_joint_minus_specialist": (
            float(joint["denovo_valid_macro"]) - float(denovo["denovo_valid_macro"])
        ),
        "edit_valid_joint_minus_specialist": (
            float(joint["edit_valid_macro"]) - float(edit["edit_valid_macro"])
        ),
    }
    joint_params = int(trains["joint"]["trainable_parameters"])
    separate_params = int(trains["denovo"]["trainable_parameters"]) + int(
        trains["edit"]["trainable_parameters"]
    )
    noninferior = (
        deltas["denovo_strict_joint_minus_specialist"] >= -0.02 - 1e-12
        and deltas["edit_strict_joint_minus_specialist"] >= -0.02 - 1e-12
        and deltas["denovo_valid_joint_minus_specialist"] >= -0.02 - 1e-12
        and deltas["edit_valid_joint_minus_specialist"] >= -0.02 - 1e-12
    )
    positive_transfer = (
        deltas["denovo_strict_joint_minus_specialist"] >= 0.02 - 1e-12
        or deltas["edit_strict_joint_minus_specialist"] >= 0.02 - 1e-12
    )
    if noninferior and positive_transfer:
        decision = "SUPPORT_UNIFIED_POSITIVE_TRANSFER_PILOT"
    elif noninferior:
        decision = "SUPPORT_UNIFIED_PARAMETER_EFFICIENCY_PILOT"
    elif (
        deltas["denovo_strict_joint_minus_specialist"] < -0.02
        or deltas["edit_strict_joint_minus_specialist"] < -0.02
    ):
        decision = "ASYMMETRIC_INTERFERENCE_PILOT"
    else:
        decision = "INCONCLUSIVE_PILOT"
    return {
        "protocol": "p33_clean_joint_vs_separate_pilot_v1",
        "joint": joint,
        "specialists": {"denovo": denovo, "edit": edit},
        "deltas": deltas,
        "efficiency": {
            "joint_trainable_parameters": joint_params,
            "two_specialist_trainable_parameters": separate_params,
            "joint_over_separate_adapter_parameter_ratio": joint_params / separate_params,
            "base_backbone_shared_in_both_comparisons": True,
        },
        "checks": {
            "joint_noninferior_within_2pp_on_both_modes_and_validity": noninferior,
            "positive_transfer_at_least_2pp_on_one_mode": positive_transfer,
        },
        "decision": decision,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)
    evals = {
        arm: load(args.output_root / "eval" / arm / "summary.json")
        for arm in ("joint", "denovo", "edit")
    }
    trains = {
        arm: load(args.output_root / "model" / arm / "training_summary.json")
        for arm in ("joint", "denovo", "edit")
    }
    result = summarize(evals, trains)
    output = args.output_root / "result"
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (output / "RESULT_COMPLETE").touch()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
