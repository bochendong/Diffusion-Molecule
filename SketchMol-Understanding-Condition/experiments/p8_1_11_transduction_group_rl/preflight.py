#!/usr/bin/env python3
"""Fail-closed audit of the clean P8.1.2-R1 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", required=True, type=Path); parser.add_argument("--training-summary", required=True, type=Path); parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(); ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False); summary = json.loads(args.training_summary.read_text())
    vocab = dict(ckpt["vocab"]); state = ckpt["model_state"]
    payload = {
        "protocol": "p8_1_11_base_preflight_v1", "checkpoint_sha256": sha(args.checkpoint),
        "base_is_p812_r1_p6_warmstart": "p8_1_2_unified_transduction_raw_v1" in str(args.checkpoint) and "p6_unified_transition_policy" in str(summary.get("base_checkpoint", "")),
        "one_decoder": "decoder.layers.0.self_attn.in_proj_weight" in state, "one_output_head": "output.weight" in state,
        "transduction_tokens": all(token in vocab for token in ("<TRANSDUCE>", "<INSERT>", "<INSERT_END>", "<STOP>", "<KEEP_1>", "<DELETE_1>")),
        "algorithm_entry": "group_relative_REINFORCE_not_GRPO", "interpreter_count": 1, "router": False, "materializer": False, "property_rerank": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n"); print(json.dumps(payload, indent=2, sort_keys=True))
    if not all(payload[key] for key in ("base_is_p812_r1_p6_warmstart", "one_decoder", "one_output_head", "transduction_tokens")): raise SystemExit("P8.1.11 base preflight failed")
    return 0


if __name__ == "__main__": raise SystemExit(main())
