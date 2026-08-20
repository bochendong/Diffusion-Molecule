#!/usr/bin/env python3
"""E1b: frozen E1 head, keyword vs letter-scrambled instructions.

No training. Same Table1 n=20 and sampling seed as E1 template.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
REPO_DIR = PROJECT_DIR.parent
WORKTREE_LATENT = PROJECT_DIR / "experiments" / "unified_latent_flow"
WORKTREE_PROJECT = PROJECT_DIR
C_DIR = PROJECT_DIR / "experiments" / "unified_action_categorical"
for path in (WORKTREE_LATENT, WORKTREE_PROJECT, C_DIR, PROJECT_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import eval_d0_b41_table1 as d0b  # noqa: E402
import eval_e1_nl_condition_head as e1mod  # noqa: E402
import valid_terminal_molecule_latent_jump as valid_terminal  # noqa: E402
import viability_preserving_interacting_particle_transport as b41  # noqa: E402

base = b41.base
delta = b41.delta
graph = b41.graph
b37 = b41.b37
b39 = b41.b39
b40 = b41.b40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-csv", required=True, type=Path)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--validation-csv", required=True, type=Path)
    parser.add_argument("--representation-checkpoint", required=True, type=Path)
    parser.add_argument("--representation-summary", required=True, type=Path)
    parser.add_argument("--b22-checkpoint", required=True, type=Path)
    parser.add_argument("--b22-summary", required=True, type=Path)
    parser.add_argument("--b36-summary", required=True, type=Path)
    parser.add_argument("--b37-summary", required=True, type=Path)
    parser.add_argument("--b38-checkpoint", required=True, type=Path)
    parser.add_argument("--b38-summary", required=True, type=Path)
    parser.add_argument("--b39-checkpoint", required=True, type=Path)
    parser.add_argument("--b39-summary", required=True, type=Path)
    parser.add_argument("--b39-evaluated-candidates", required=True, type=Path)
    parser.add_argument("--b40-summary", required=True, type=Path)
    parser.add_argument("--b40-evaluated-candidates", required=True, type=Path)
    parser.add_argument("--b41-checkpoint", required=True, type=Path)
    parser.add_argument("--b41-summary", required=True, type=Path)
    parser.add_argument("--b41-evaluated-candidates", required=True, type=Path)
    parser.add_argument("--b41-protocol-manifest", required=True, type=Path)
    parser.add_argument("--e1-protocol-manifest", required=True, type=Path)
    parser.add_argument("--e1b-protocol-manifest", required=True, type=Path)
    parser.add_argument("--e1-head-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-limit", type=int, default=0)
    return parser.parse_args()


def render_keyword(specs: list[tuple[str, int]], e1: dict) -> str:
    names = dict(e1["property_names"])
    parts = [str(names.get(prop, prop)) for prop, _direction in specs]
    return " ".join(parts) if parts else "molecule"


def scramble_token(token: str, rng: random.Random) -> str:
    prefix = ""
    suffix = ""
    body = token
    while body and not body[0].isalnum():
        prefix += body[0]
        body = body[1:]
    while body and not body[-1].isalnum():
        suffix = body[-1] + suffix
        body = body[:-1]
    if len(body) < 3:
        return token
    chars = list(body)
    rng.shuffle(chars)
    if "".join(chars) == body:
        chars = chars[1:] + chars[:1]
    return prefix + "".join(chars) + suffix


def scramble_instruction(text: str, *, seed: int, condition_id: str) -> str:
    digest = hashlib.blake2b(f"{seed}|{condition_id}|{text}".encode("utf-8"), digest_size=8).digest()
    rng = random.Random(int.from_bytes(digest, "little"))
    return " ".join(scramble_token(token, rng) for token in str(text).split())


def instruction_for_variant(
    variant: str,
    *,
    specs: list[tuple[str, int]],
    e1: dict,
    seed: int,
    condition_id: str,
) -> str:
    template = e1mod.render_template(specs, e1, verb_index=0)
    if str(variant) == "keyword":
        return render_keyword(specs, e1)
    if str(variant) == "scrambled":
        return scramble_instruction(template, seed=int(seed), condition_id=str(condition_id))
    raise ValueError(f"Unknown E1b variant: {variant}")


def main() -> int:
    args = parse_args()
    e1 = json.loads(args.e1_protocol_manifest.read_text(encoding="utf-8"))
    e1b = json.loads(args.e1b_protocol_manifest.read_text(encoding="utf-8"))
    b41_prereg = b41.read_preregistration(args.b41_protocol_manifest)
    device = base.resolve_device(str(args.device))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    (
        _b22_summary,
        b22_checkpoint,
        _b36_summary,
        _b37_summary,
        _b39_checkpoint,
        _b40_summary,
    ) = b41.check_locked_inputs(args, b41_prereg)
    b41_checkpoint = torch.load(args.b41_checkpoint, map_location="cpu", weights_only=False)
    head_blob = torch.load(args.e1_head_checkpoint, map_location="cpu", weights_only=False)
    selected_pairs = d0b.reconstruct_support_pairs(args, b41_prereg, b22_checkpoint)
    fit_pairs, _development_pairs, _split = b37.strict_source_group_split(
        selected_pairs,
        seed=int(b41_prereg["development_split_seed"]),
        development_source_limit=int(b41_prereg["development_source_limit"]),
    )
    representation, representation_config, _summary = base.load_representation(
        args.representation_checkpoint, args.representation_summary, device
    )
    representation.eval().requires_grad_(False)
    vocabulary = b37.checkpoint_vocabulary(b22_checkpoint)
    head = e1mod.InstructionConditionHead(
        int(head_blob["feature_dim"]),
        int(e1["head_hidden_dim"]),
        int(head_blob["token_count"]),
        int(head_blob["condition_dim"]),
    ).to(device)
    head.load_state_dict(dict(head_blob["model_state"]), strict=True)
    head.eval().requires_grad_(False)

    support = b40.build_support(fit_pairs, vocabulary)
    support_tensors = b40._device_support(support, device)
    node_action_count, edge_action_count = delta.action_space_sizes(vocabulary)
    model = b39.LatentCardinalityGraphJumpBridge(
        node_dim=int(representation_config["node_dim"]),
        edge_dim=int(representation_config["edge_dim"]),
        condition_dim=int(b41_prereg["condition_dim"]),
        transport_dim=int(b41_prereg["transport_dim"]),
        hidden_dim=int(b41_prereg["hidden_dim"]),
        max_atoms=int(representation_config["max_atoms"]),
        max_jumps=int(b41_prereg["max_jumps"]),
        property_count=len(b41.unified.PROPERTY_COLUMNS),
        node_state_count=node_action_count,
        edge_state_count=edge_action_count,
        message_layers=int(b41_prereg["message_layers"]),
    ).to(device)
    model.load_state_dict(dict(b41_checkpoint["model_state"]), strict=True)
    model.eval().requires_grad_(False)

    conditions = d0b.load_table1_conditions(
        args.eval_csv,
        limit=int(args.eval_limit),
        condition_dim=int(b41_prereg["condition_dim"]),
        graph_fingerprint_bits=int(b41_prereg["fingerprint_bits"]),
        max_atoms=int(representation_config["max_atoms"]),
    )
    examples = {}
    for condition in conditions:
        if str(condition.task) == "GSK3B:increase" and "gsk3b" not in examples:
            specs = e1mod.specs_for_row(dict(condition.row))
            examples["gsk3b"] = {
                "template": e1mod.render_template(specs, e1, verb_index=0),
                "keyword": render_keyword(specs, e1),
                "scrambled": scramble_instruction(
                    e1mod.render_template(specs, e1, verb_index=0),
                    seed=int(e1b["seed"]),
                    condition_id=str(condition.condition_id),
                ),
            }
            break
    (args.output_dir / "instruction_examples.json").write_text(
        json.dumps(examples, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"stage": "examples", **examples}, sort_keys=True), flush=True)

    exact_support = valid_terminal.ExactMoleculeStopSupport(vocabulary)
    original_mask = b41.viability_event_mask
    b41.viability_event_mask = exact_support
    try:
        for variant in e1b["variants"]:
            variant_dir = args.output_dir / str(variant)
            variant_dir.mkdir(parents=True, exist_ok=True)
            rows: list[dict[str, object]] = []
            skipped = 0
            sample_started = time.perf_counter()
            for index, condition in enumerate(conditions):
                specs = e1mod.specs_for_row(dict(condition.row))
                text = instruction_for_variant(
                    str(variant),
                    specs=specs,
                    e1=e1,
                    seed=int(e1b["seed"]),
                    condition_id=str(condition.condition_id),
                )
                features = torch.from_numpy(e1mod.instruction_features(text, e1)[None, :]).to(device)
                with torch.no_grad():
                    tokens = head(features)[0].detach().cpu().numpy().astype(np.float32)
                try:
                    generated = b41.sample_from_source(
                        model,
                        representation,
                        vocabulary,
                        support,
                        support_tensors,
                        condition.source,
                        tokens,
                        b41_prereg,
                        device,
                        int(e1b["seed"]) * 100000 + index,
                    )
                except Exception as exc:
                    print(
                        json.dumps(
                            {
                                "stage": "sample_failed",
                                "variant": variant,
                                "condition_id": condition.condition_id,
                                "error": f"{type(exc).__name__}: {exc}",
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    generated = [{"generated_smiles": ""}] * int(e1b["exact_raw_attempts_per_condition"])
                    skipped += 1
                if len(generated) != int(e1b["exact_raw_attempts_per_condition"]):
                    generated = (list(generated) + [{"generated_smiles": ""}] * 20)[
                        : int(e1b["exact_raw_attempts_per_condition"])
                    ]
                    skipped += 1
                for attempt, candidate in enumerate(generated, start=1):
                    rows.append(
                        {
                            "condition_id": condition.condition_id,
                            "task": condition.task,
                            "source_smiles": condition.source_smiles,
                            "generated_smiles": candidate.get("generated_smiles", "") or "",
                            "sample_index": attempt,
                            "candidate_index": attempt,
                            "method": e1b["protocol"],
                            "family": "b41_nl_condition",
                            "op": "latent_graph_jump",
                            "variant": variant,
                            "instruction": text,
                        }
                    )
                if (index + 1) % 20 == 0 or index + 1 == len(conditions):
                    elapsed = time.perf_counter() - sample_started
                    done = index + 1
                    sec_per = elapsed / done
                    print(
                        json.dumps(
                            {
                                "stage": "sampled",
                                "variant": variant,
                                "done": done,
                                "total": len(conditions),
                                "elapsed_sec": round(elapsed, 1),
                                "sec_per_condition": round(sec_per, 3),
                                "eta_sec": round(sec_per * (len(conditions) - done), 1),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            candidate_path = variant_dir / f"e1b_{variant}_table1_n20_candidates.csv"
            d0b.write_rows(candidate_path, rows)
            sampling = {
                "protocol": e1b["protocol"],
                "variant": variant,
                "device": str(device),
                "eval_csv": str(args.eval_csv),
                "loaded_conditions": len(conditions),
                "candidate_rows": len(rows),
                "attempts_per_condition": int(e1b["exact_raw_attempts_per_condition"]),
                "skipped_count": skipped,
                "candidate_csv": str(candidate_path),
                "molecular_candidate_ranking": False,
                "task_router": False,
                "oracle_in_environment": False,
                "frozen_b41_event_kernel": True,
                "frozen_e1_head": True,
                "not_ours": True,
                "elapsed_sec": round(time.perf_counter() - started, 1),
                "sample_sec": round(time.perf_counter() - sample_started, 1),
                "exact_stop_support": exact_support.manifest(),
            }
            (variant_dir / "sampling_summary.json").write_text(
                json.dumps(sampling, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(json.dumps(sampling, indent=2, sort_keys=True), flush=True)
    finally:
        b41.viability_event_mask = original_mask
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
