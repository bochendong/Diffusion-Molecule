#!/usr/bin/env python3
"""E1: frozen B41 + hashed instruction projector onto property-slot tokens.

Diagnostic only. Does not unfreeze the event kernel. Property-vector baseline
is the locked D0 B41 Table1 summary, compared after sampling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


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
import table1_energy_tilted_latent_transfer as b29  # noqa: E402
import valid_terminal_molecule_latent_jump as valid_terminal  # noqa: E402
import viability_preserving_interacting_particle_transport as b41  # noqa: E402
from sketchmol_understanding_condition.text_features import hashed_text_vector  # noqa: E402

base = b41.base
delta = b41.delta
graph = b41.graph
b37 = b41.b37
b39 = b41.b39
b40 = b41.b40


class InstructionConditionHead(nn.Module):
    """Hashed instruction features → B41 property-slot tokens."""

    def __init__(self, feature_dim: int, hidden_dim: int, token_count: int, condition_dim: int) -> None:
        super().__init__()
        self.token_count = int(token_count)
        self.condition_dim = int(condition_dim)
        self.net = nn.Sequential(
            nn.Linear(int(feature_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(token_count) * int(condition_dim)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).view(-1, self.token_count, self.condition_dim)


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
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-limit", type=int, default=0)
    return parser.parse_args()


def normalize_instruction(text: str) -> str:
    return (
        str(text or "")
        .replace("β", "b")
        .replace("Β", "b")
        .lower()
        .strip()
    )


def hashed_char_ngrams(text: str, dim: int, n: int) -> np.ndarray:
    padded = f"{' ' * (n - 1)}{normalize_instruction(text)}{' ' * (n - 1)}"
    vec = np.zeros(int(dim), dtype=np.float32)
    for index in range(max(0, len(padded) - n + 1)):
        gram = padded[index : index + n]
        digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % max(1, int(dim))
        sign = 1.0 if int.from_bytes(digest[4:], "little") % 2 == 0 else -1.0
        vec[bucket] += sign
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def instruction_features(text: str, e1: dict) -> np.ndarray:
    word = hashed_text_vector(normalize_instruction(text), int(e1["word_hash_dim"]))
    char = hashed_char_ngrams(text, int(e1["char_hash_dim"]), int(e1["char_ngram"]))
    return np.concatenate([word, char]).astype(np.float32)


def direction_name(direction: int) -> str:
    if int(direction) > 0:
        return "increase"
    if int(direction) < 0:
        return "decrease"
    return "maintain"


def render_template(specs: list[tuple[str, int]], e1: dict, verb_index: int) -> str:
    names = dict(e1["property_names"])
    verbs = dict(e1["train_verbs"])
    parts = []
    for prop, direction in specs:
        options = list(verbs.get(direction_name(int(direction)), ["change"]))
        verb = options[int(verb_index) % max(1, len(options))]
        parts.append(f"{verb} {names.get(prop, prop)}")
    if not parts:
        return "Edit the molecule."
    if len(parts) == 1:
        return f"Edit the molecule to {parts[0]}."
    return f"Edit the molecule to {', '.join(parts[:-1])}, and {parts[-1]}."


def specs_for_row(row: dict[str, str]) -> list[tuple[str, int]]:
    specs = list(base.task_specs(row))
    return [(str(prop), int(direction)) for prop, direction in specs if int(direction) != 0]


def paraphrase_for_task(task: str, specs: list[tuple[str, int]], e1: dict) -> str:
    paraphrases = dict(e1["eval_paraphrases"])
    if task in paraphrases:
        return str(paraphrases[task])
    return render_template(specs, e1, verb_index=0)


def main() -> int:
    args = parse_args()
    e1 = json.loads(args.e1_protocol_manifest.read_text(encoding="utf-8"))
    assert_disjoint_instructions(e1)
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

    items = []
    skipped_no_specs = 0
    for pair in fit_pairs:
        specs = specs_for_row(dict(pair.row))
        tokens = np.asarray(pair.condition, dtype=np.float32)
        if tokens.ndim != 2:
            continue
        if not specs:
            skipped_no_specs += 1
            continue
        for verb_index in range(int(e1["train_templates_per_pair"])):
            text = render_template(specs, e1, verb_index=verb_index)
            items.append({"features": instruction_features(text, e1), "tokens": tokens, "text": text})
    print(
        json.dumps(
            {
                "kept": len(items),
                "fit_pairs": len(fit_pairs),
                "skipped_no_specs": skipped_no_specs,
                "stage": "train_labels",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if len(items) < 32:
        raise ValueError(f"Need at least 32 instruction/token pairs, found {len(items)}")
    token_count = int(items[0]["tokens"].shape[0])
    condition_dim = int(items[0]["tokens"].shape[1])
    if any(item["tokens"].shape != (token_count, condition_dim) for item in items):
        raise ValueError("Condition slot packing is not uniform")
    feature_dim = int(items[0]["features"].shape[0])
    head = InstructionConditionHead(
        feature_dim, int(e1["head_hidden_dim"]), token_count, condition_dim
    ).to(device)
    history = train_head(head, items, e1, device)
    train_mse = float(history[-1]["loss"]) if history else None
    paraphrase_mse = paraphrase_token_mse(
        head, fit_pairs, e1, token_count, condition_dim, device
    )
    torch.save(
        {
            "model_state": head.state_dict(),
            "protocol": e1["protocol"],
            "history": history,
            "token_count": token_count,
            "condition_dim": condition_dim,
            "feature_dim": feature_dim,
            "train_mse": train_mse,
            "paraphrase_token_mse": paraphrase_mse,
        },
        args.output_dir / "e1_nl_condition_head.pt",
    )
    (args.output_dir / "train_summary.json").write_text(
        json.dumps(
            {
                "history": history,
                "train_pairs": len(items),
                "fit_pairs": len(fit_pairs),
                "skipped_no_specs": skipped_no_specs,
                "token_count": token_count,
                "condition_dim": condition_dim,
                "train_mse": train_mse,
                "paraphrase_token_mse": paraphrase_mse,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "stage": "head_ready",
                "train_mse": train_mse,
                "paraphrase_token_mse": paraphrase_mse,
            },
            sort_keys=True,
        ),
        flush=True,
    )

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
    exact_support = valid_terminal.ExactMoleculeStopSupport(vocabulary)
    original_mask = b41.viability_event_mask
    b41.viability_event_mask = exact_support
    head.eval()
    try:
        for variant in e1["variants"]:
            variant_dir = args.output_dir / str(variant)
            variant_dir.mkdir(parents=True, exist_ok=True)
            rows: list[dict[str, object]] = []
            skipped = 0
            sample_started = time.perf_counter()
            for index, condition in enumerate(conditions):
                specs = specs_for_row(dict(condition.row))
                if str(variant) == "paraphrase":
                    text = paraphrase_for_task(str(condition.task), specs, e1)
                else:
                    text = render_template(specs, e1, verb_index=0)
                features = torch.from_numpy(instruction_features(text, e1)[None, :]).to(device)
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
                        int(e1["seed"]) * 100000 + index,
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
                    generated = [{"generated_smiles": ""}] * int(e1["exact_raw_attempts_per_condition"])
                    skipped += 1
                if len(generated) != int(e1["exact_raw_attempts_per_condition"]):
                    generated = (list(generated) + [{"generated_smiles": ""}] * 20)[
                        : int(e1["exact_raw_attempts_per_condition"])
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
                            "method": e1["protocol"],
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
            candidate_path = variant_dir / f"e1_{variant}_table1_n20_candidates.csv"
            d0b.write_rows(candidate_path, rows)
            sampling = {
                "protocol": e1["protocol"],
                "variant": variant,
                "device": str(device),
                "eval_csv": str(args.eval_csv),
                "loaded_conditions": len(conditions),
                "candidate_rows": len(rows),
                "attempts_per_condition": int(e1["exact_raw_attempts_per_condition"]),
                "skipped_count": skipped,
                "candidate_csv": str(candidate_path),
                "molecular_candidate_ranking": False,
                "task_router": False,
                "oracle_in_environment": False,
                "frozen_b41_event_kernel": True,
                "not_ours": True,
                "train_mse": train_mse,
                "paraphrase_token_mse": paraphrase_mse,
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


def assert_disjoint_instructions(e1: dict) -> None:
    paraphrases = {normalize_instruction(text) for text in dict(e1["eval_paraphrases"]).values()}
    train_texts = set()
    for specs in table1_spec_bank():
        for verb_index in range(int(e1["train_templates_per_pair"])):
            train_texts.add(normalize_instruction(render_template(specs, e1, verb_index)))
    overlap = paraphrases & train_texts
    if overlap:
        raise ValueError(f"Eval paraphrases overlap train templates: {sorted(overlap)}")


def table1_spec_bank() -> list[list[tuple[str, int]]]:
    return [
        [("GSK3B", 1)],
        [("MW", 1)],
        [("SA", -1)],
        [("RB", -1)],
        [("DRD2", -1), ("MW", -1), ("SA", -1)],
        [("HBA", -1), ("SA", -1)],
        [("QED", 1), ("SA", -1)],
        [("HBA", -1), ("LogP", 1)],
        [("HBA", -1), ("MW", -1)],
        [("HBA", 1), ("MW", 1), ("QED", -1)],
    ]


def paraphrase_token_mse(
    head: InstructionConditionHead,
    fit_pairs,
    e1: dict,
    token_count: int,
    condition_dim: int,
    device: torch.device,
) -> float | None:
    paraphrases = dict(e1["eval_paraphrases"])
    rows = []
    head.eval()
    with torch.no_grad():
        for pair in fit_pairs:
            specs = specs_for_row(dict(pair.row))
            task = b29.table1_task_key(specs)
            if task not in paraphrases:
                continue
            target = np.asarray(pair.condition, dtype=np.float32)
            if target.shape != (token_count, condition_dim):
                continue
            features = torch.from_numpy(instruction_features(str(paraphrases[task]), e1)[None, :]).to(device)
            predicted = head(features)[0].cpu().numpy()
            rows.append(float(np.mean((predicted - target) ** 2)))
    if not rows:
        return None
    return float(sum(rows) / len(rows))


def train_head(head, items, e1, device) -> list[dict[str, float]]:
    batch_size = int(e1["head_batch_size"])
    optimizer = torch.optim.AdamW(head.parameters(), lr=float(e1["head_lr"]))
    history = []
    head.train()
    for epoch in range(1, int(e1["head_epochs"]) + 1):
        order = torch.randperm(len(items)).tolist()
        total = 0.0
        seen = 0
        for start in range(0, len(items), batch_size):
            batch = [items[index] for index in order[start : start + batch_size]]
            features = torch.stack([torch.from_numpy(item["features"]) for item in batch]).to(device)
            target = torch.stack([torch.from_numpy(item["tokens"]) for item in batch]).to(device)
            loss = torch.nn.functional.mse_loss(head(features), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch)
            seen += len(batch)
        row = {"epoch": epoch, "loss": total / max(1, seen)}
        history.append(row)
        print(json.dumps({"stage": "head_epoch", **row}, sort_keys=True), flush=True)
    head.eval()
    return history


if __name__ == "__main__":
    raise SystemExit(main())
