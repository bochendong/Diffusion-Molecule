#!/usr/bin/env python3
"""Verify P4 changed only source-conditioned parameters."""

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
        raise ValueError("P4 changed the SMILES vocabulary")
    if base.get("model_config") != trained.get("model_config"):
        raise ValueError("P4 changed the model configuration")
    base_state = dict(base["model_state"])
    trained_state = dict(trained["model_state"])
    if set(base_state) != set(trained_state):
        raise ValueError("P4 changed model-state keys")
    changed_source = []
    changed_forbidden = []
    for name in sorted(base_state):
        equal = torch.equal(base_state[name], trained_state[name])
        allowed = name.startswith(SOURCE_PREFIXES)
        if not equal and allowed:
            changed_source.append(name)
        elif not equal:
            changed_forbidden.append(name)
    payload = {
        "protocol": "p4_source_only_checkpoint_audit_v1",
        "base_checkpoint": str(args.base_checkpoint),
        "base_checkpoint_sha256": sha256(args.base_checkpoint),
        "trained_checkpoint": str(args.trained_checkpoint),
        "trained_checkpoint_sha256": sha256(args.trained_checkpoint),
        "source_prefixes": list(SOURCE_PREFIXES),
        "changed_source_parameters": changed_source,
        "changed_forbidden_parameters": changed_forbidden,
        "de_novo_path_bit_identical": not changed_forbidden,
        "decision": "go" if changed_source and not changed_forbidden else "stop",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if changed_forbidden or not changed_source:
        raise SystemExit(2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
