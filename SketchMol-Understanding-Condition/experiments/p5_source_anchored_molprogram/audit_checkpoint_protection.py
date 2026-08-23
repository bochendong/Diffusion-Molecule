#!/usr/bin/env python3
"""Verify that P5 changes only edit-only source modules."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


SOURCE_PREFIXES = (
    "source_condition_proj.",
    "source_encoder.",
    "source_type",
    "null_source",
    "source_gate.",
    "source_output.",
    "source_adapters.",
    "source_copy_query.",
    "source_copy_key.",
    "source_copy_gate.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", required=True, type=Path)
    parser.add_argument("--trained-checkpoint", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    base = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    trained = torch.load(args.trained_checkpoint, map_location="cpu", weights_only=False)
    if base.get("vocab") != trained.get("vocab"):
        raise ValueError("P5 changed the shared SMILES vocabulary")
    base_state = dict(base["model_state"])
    trained_state = dict(trained["model_state"])
    missing_shared = [name for name in base_state if name not in trained_state and not name.startswith(SOURCE_PREFIXES)]
    changed_shared = [
        name
        for name in base_state
        if name in trained_state
        and not name.startswith(SOURCE_PREFIXES)
        and not torch.equal(base_state[name], trained_state[name])
    ]
    new_forbidden = [name for name in trained_state if name not in base_state and not name.startswith(SOURCE_PREFIXES)]
    changed_source = [
        name
        for name in trained_state
        if name.startswith(SOURCE_PREFIXES)
        and (name not in base_state or not torch.equal(base_state[name], trained_state[name]))
    ]
    config = dict(trained.get("model_config", {}))
    checks = {
        "source_copy_aware": bool(config.get("source_copy_aware", False)),
        "source_adapters_present": int(config.get("source_adapter_layers", 0)) > 0,
        "shared_parameters_unchanged": not changed_shared and not missing_shared and not new_forbidden,
        "source_parameters_changed": bool(changed_source),
    }
    payload = {
        "protocol": "p5_source_only_checkpoint_audit_v1",
        "base_checkpoint": str(args.base_checkpoint),
        "base_checkpoint_sha256": sha256(args.base_checkpoint),
        "trained_checkpoint": str(args.trained_checkpoint),
        "trained_checkpoint_sha256": sha256(args.trained_checkpoint),
        "checks": checks,
        "changed_source_parameters": changed_source,
        "changed_shared_parameters": changed_shared,
        "missing_shared_parameters": missing_shared,
        "new_forbidden_parameters": new_forbidden,
        "de_novo_path_bit_identical": checks["shared_parameters_unchanged"],
        "decision": "go" if all(checks.values()) else "stop",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["decision"] != "go":
        raise SystemExit(2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

