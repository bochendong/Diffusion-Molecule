#!/usr/bin/env python3
"""Raw constrained sampling from one empty/source SELFIES transducer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
P6_DIR = PROJECT_DIR / "experiments" / "p6_unified_molecular_transition_policy"
UNIFIED_DIR = PROJECT_DIR / "experiments" / "unified_smiles_generator"
for path in (SCRIPT_DIR, P6_DIR, UNIFIED_DIR):
    sys.path.insert(0, str(path))

import p6_transition_program as p6  # noqa: E402
import transduction_oracle as transduction  # noqa: E402
import unified_smiles_generator as unified  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--eval-csv", required=True, type=Path)
    parser.add_argument("--eval-features-dir", required=True, type=Path)
    parser.add_argument("--candidate-output-csv", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=188)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-source-tokens", type=int, default=96)
    parser.add_argument("--condition-layout", default="p6_transition")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def observed_selfies_tokens(path: Path) -> set[str]:
    output: set[str] = set()
    for row in p6.read_rows(path):
        try:
            tokens = json.loads(str(row.get("policy_target_tokens_json", "") or "[]"))
        except json.JSONDecodeError:
            continue
        output.update(token for token in tokens if token.startswith("[") and token.endswith("]"))
    return output


def prefix_state(tokens: Sequence[str], source_length: int) -> tuple[str, int, int, int]:
    """Return state, source cursor, emitted length, and current insert length."""
    if not tokens or tokens[0] != transduction.START:
        return "start", 0, 0, 0
    cursor = 0
    emitted = 0
    current_insert = 0
    in_insert = False
    for token in tokens[1:]:
        if token == transduction.STOP:
            return "stopped", cursor, emitted, current_insert
        if in_insert:
            if token == transduction.INSERT_END:
                in_insert = False
                current_insert = 0
            else:
                emitted += 1
                current_insert += 1
            continue
        if token == transduction.INSERT:
            in_insert = True
        elif token.startswith("<KEEP_"):
            count = transduction.parse_count(token, "KEEP")
            cursor += count
            emitted += count
        elif token.startswith("<DELETE_"):
            cursor += transduction.parse_count(token, "DELETE")
    if cursor > source_length:
        return "invalid", cursor, emitted, current_insert
    return ("insert" if in_insert else "boundary"), cursor, emitted, current_insert


def allowed_tokens(
    tokens: Sequence[str], *, source_length: int, semantic_tokens: set[str]
) -> set[str]:
    state, cursor, emitted, current_insert = prefix_state(tokens, source_length)
    if state == "start":
        return {transduction.START}
    if state == "stopped":
        return {"<EOS>"}
    if state == "invalid":
        return set()
    if state == "insert":
        allowed = set(semantic_tokens)
        if current_insert > 0:
            allowed.add(transduction.INSERT_END)
        return allowed
    remaining = source_length - cursor
    allowed = {transduction.INSERT}
    if remaining > 0:
        allowed.update(f"<KEEP_{count}>" for count in range(1, remaining + 1))
        allowed.update(f"<DELETE_{count}>" for count in range(1, remaining + 1))
    elif emitted > 0:
        allowed.add(transduction.STOP)
    return allowed


@torch.no_grad()
def generate(
    model: unified.ConditionedSmilesDecoder,
    condition: torch.Tensor,
    condition_mask: torch.Tensor,
    *,
    vocab: unified.SmilesVocabulary,
    source_length: int,
    semantic_tokens: set[str],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    batch = condition.shape[0]
    generated = torch.full((batch, 1), vocab.bos_id, dtype=torch.long, device=condition.device)
    finished = torch.zeros(batch, dtype=torch.bool, device=condition.device)
    for _step in range(max(1, int(max_new_tokens))):
        logits = model(condition, generated, condition_mask=condition_mask)[:, -1, :]
        for index in range(batch):
            if finished[index]:
                logits[index, :] = -torch.inf
                logits[index, vocab.eos_id] = 0.0
                continue
            decoded = vocab.decode(generated[index].tolist())
            if transduction.START in decoded:
                decoded = decoded[decoded.index(transduction.START):]
            allowed = allowed_tokens(decoded, source_length=source_length, semantic_tokens=semantic_tokens)
            allowed_ids = [vocab.eos_id if token == "<EOS>" else vocab.token_to_id[token] for token in allowed]
            mask = torch.ones(logits.shape[-1], dtype=torch.bool, device=logits.device)
            mask[allowed_ids] = False
            logits[index].masked_fill_(mask, -torch.inf)
        logits = logits / max(float(temperature), 1e-6)
        if 0 < int(top_k) < logits.shape[-1]:
            threshold = torch.topk(logits, int(top_k), dim=-1).values[:, -1:]
            logits = logits.masked_fill(logits < threshold, -torch.inf)
        logits = unified.top_p_filter(logits, top_p=float(top_p))
        probabilities = torch.nan_to_num(torch.softmax(logits, dim=-1), nan=0.0)
        next_ids = torch.multinomial(probabilities, num_samples=1).squeeze(1)
        generated = torch.cat((generated, next_ids[:, None]), dim=1)
        finished |= next_ids.eq(vocab.eos_id)
        if bool(finished.all()):
            break
    return generated


def main() -> int:
    args = parse_args()
    unified.seed_everything(int(args.seed))
    device = unified.resolve_device(str(args.device))
    checkpoint = unified.load_checkpoint(args.checkpoint)
    if checkpoint is None:
        raise FileNotFoundError(args.checkpoint)
    vocab = unified.SmilesVocabulary.from_dict(checkpoint["vocab"])
    config = dict(checkpoint["model_config"])
    model = unified.ConditionedSmilesDecoder(**config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    store = unified.FeatureStore(args.eval_features_dir, array_name="query_tokens", variant="full")
    semantic_tokens = observed_selfies_tokens(args.train_csv)
    rows = p6.read_rows(args.eval_csv)
    output: list[dict[str, object]] = []
    valid = complete = 0
    rows_with_valid = 0
    for row_index, row in enumerate(rows):
        source_smiles = transduction.canonical(p6.source_for_row(row)) if p6.source_for_row(row) else ""
        source_tokens = transduction.sf_tokens(source_smiles) if source_smiles else []
        condition_np = unified.condition_array_for_row(
            row, store, int(config["condition_dim"]),
            max_source_tokens=int(args.max_source_tokens), condition_layout=str(args.condition_layout),
        ).astype(np.float32)
        condition = torch.from_numpy(condition_np)[None, :, :].to(device)
        condition = condition.expand(int(args.num_samples), -1, -1)
        condition_mask = torch.ones(condition.shape[:2], dtype=torch.bool, device=device)
        generated_batch = generate(
            model, condition, condition_mask, vocab=vocab, source_length=len(source_tokens),
            semantic_tokens=semantic_tokens, max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature), top_k=int(args.top_k), top_p=float(args.top_p),
        )
        row_valid = 0
        for candidate_index, generated in enumerate(generated_batch.tolist()):
            tokens = vocab.decode(generated)
            if transduction.START in tokens:
                tokens = tokens[tokens.index(transduction.START):]
            is_complete = transduction.STOP in tokens
            complete += int(is_complete)
            canonical = ""
            try:
                reconstructed = transduction.execute_program(source_tokens, tokens)
                canonical = transduction.canonical(transduction.decode_sf(reconstructed))
            except Exception:
                pass
            valid += int(bool(canonical))
            row_valid += int(bool(canonical))
            item = dict(row)
            item.update({
                "generated_smiles": canonical,
                "candidate_smiles": canonical,
                "direct_candidate_index": candidate_index,
                "generation_rank": candidate_index + 1,
                "candidate_rank": candidate_index + 1,
                "method": "p8_1_2_unified_selfies_transduction",
                "p812_program_tokens_json": json.dumps(tokens),
                "p812_grammar_complete": str(is_complete),
            })
            item.update(unified.candidate_metrics(row, canonical, source_similarity_threshold=0.65))
            item["direct_candidate_strict_fraction"] = item["unified_property_success_fraction"]
            output.append(item)
        rows_with_valid += int(row_valid > 0)
        if (row_index + 1) % 20 == 0 or row_index + 1 == len(rows):
            print(f"[p8.1.2-sample] {row_index + 1}/{len(rows)}", flush=True)
    p6.write_rows(args.candidate_output_csv, output)
    total = len(rows) * int(args.num_samples)
    summary = {
        "protocol": "p8_1_2_raw_unified_selfies_transduction",
        "eval_rows": len(rows), "num_samples": int(args.num_samples),
        "valid_candidates": valid, "candidate_validity": valid / max(total, 1),
        "rows_with_valid_candidate": rows_with_valid,
        "row_validity": rows_with_valid / max(len(rows), 1),
        "grammar_complete": complete / max(total, 1),
        "property_reranking": False,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
