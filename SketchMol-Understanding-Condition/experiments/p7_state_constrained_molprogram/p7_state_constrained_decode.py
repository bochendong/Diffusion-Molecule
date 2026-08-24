#!/usr/bin/env python3
"""P7 state-constrained decoding for the frozen P6 MolProgram policy.

The model, checkpoint, property condition, transition vocabulary, and graph
interpreter are unchanged.  At every autoregressive step we expose only tokens
that form a typed action over sites present in the current molecular state.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
P6_DIR = PROJECT_DIR / "experiments" / "p6_unified_molecular_transition_policy"
UNIFIED_DIR = PROJECT_DIR / "experiments" / "unified_smiles_generator"
for import_dir in (P6_DIR, UNIFIED_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

import p6_transition_program as p6  # noqa: E402
import unified_smiles_generator as unified  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402


RDLogger.DisableLog("rdApp.*")

LEGACY_OPS = {
    "<ADD_FRAGMENT>",
    "<SUBSTITUTE_TERMINAL>",
    "<REPLACE_ATOM>",
    "<DELETE_TERMINAL_ATOM>",
    "<CHANGE_BOND_ORDER>",
}
ORDER_TOKENS = set(p6.TOKEN_TO_ORDER)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--eval-csv", required=True, type=Path)
    parser.add_argument("--eval-features-dir", required=True, type=Path)
    parser.add_argument("--candidate-output-csv", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--condition-layout", default="p6_transition")
    parser.add_argument("--max-source-tokens", type=int, default=96)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=188)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def observed_tokens(path: Path) -> set[str]:
    observed: set[str] = set()
    for row in p6.read_rows(path):
        try:
            observed.update(json.loads(row.get("policy_target_tokens_json", "[]")))
        except json.JSONDecodeError:
            continue
    return observed


def completed_actions(tokens: Sequence[str]) -> list[p6.TransitionAction]:
    if not tokens or tokens[0] != p6.PROGRAM:
        return []
    prefix = list(tokens)
    if p6.STOP not in prefix:
        prefix.append(p6.STOP)
    try:
        return p6.parse_program(prefix, tolerate_incomplete_suffix=True)
    except Exception:
        return []


def atom_count(initial_smiles: str, actions: Sequence[p6.TransitionAction]) -> int:
    mol = Chem.MolFromSmiles(initial_smiles) if initial_smiles else None
    count = int(mol.GetNumAtoms()) if mol is not None else 0
    for action in actions:
        if action.op in {"init_atom", "add_atom"}:
            count += 1
        elif action.op == "delete_terminal_atom":
            count = max(0, count - 1)
    return count


def action_tail(tokens: Sequence[str]) -> list[str]:
    if p6.ACTION_END in tokens:
        last = len(tokens) - 1 - list(reversed(tokens)).index(p6.ACTION_END)
        return list(tokens[last + 1 :])
    return list(tokens[1:]) if tokens and tokens[0] == p6.PROGRAM else list(tokens)


def site_tokens(vocab_tokens: set[str], count: int, *, exclude: str = "") -> set[str]:
    return {
        f"<SITE_{index:03d}>"
        for index in range(max(0, count))
        if f"<SITE_{index:03d}>" in vocab_tokens and f"<SITE_{index:03d}>" != exclude
    }


def source_site_tokens(initial_smiles: str, op: str, *, first_site: str = "") -> set[str]:
    """Enumerate sites on which the requested edit can actually execute."""
    mol = Chem.MolFromSmiles(initial_smiles) if initial_smiles else None
    if mol is None:
        return set()
    if op == "<CHANGE_BOND_ORDER>":
        eligible_bonds = [
            bond
            for bond in mol.GetBonds()
            if not bond.GetIsAromatic() and not bond.IsInRing()
        ]
        if first_site:
            first = p6.parse_site(first_site)
            return {
                f"<SITE_{other:03d}>"
                for bond in eligible_bonds
                for begin, other in (
                    (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
                    (bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()),
                )
                if begin == first
            }
        return {
            f"<SITE_{index:03d}>"
            for bond in eligible_bonds
            for index in (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        }
    atoms = list(mol.GetAtoms())
    if op in {"<DELETE_TERMINAL_ATOM>", "<SUBSTITUTE_TERMINAL>"}:
        atoms = [atom for atom in atoms if atom.GetDegree() == 1 and mol.GetNumAtoms() > 1]
    elif op == "<REPLACE_ATOM>":
        atoms = [atom for atom in atoms if not atom.GetIsAromatic()]
    return {f"<SITE_{atom.GetIdx():03d}>" for atom in atoms}


def allowed_next_tokens(
    tokens: Sequence[str],
    *,
    initial_smiles: str,
    vocab_tokens: set[str],
    observed: set[str],
) -> set[str]:
    """Return the grammar/state-valid next-token set for one sequence."""
    if not tokens:
        return {p6.PROGRAM}
    if tokens[-1] == p6.STOP:
        return {"<EOS>"}
    if tokens[0] != p6.PROGRAM:
        return {p6.PROGRAM}

    actions = completed_actions(tokens)
    count = atom_count(initial_smiles, actions)
    tail = action_tail(tokens)
    source_conditioned = bool(initial_smiles)

    if not tail:
        if count == 0:
            return {p6.INIT_ATOM}
        # The paired edit supervision contains one state transition.  Stop
        # after it instead of sampling untrained multi-edit compositions.
        if source_conditioned and actions:
            return {p6.STOP}
        if source_conditioned:
            operators = {p6.ADD_ATOM}
            operators.update(
                op for op in LEGACY_OPS if source_site_tokens(initial_smiles, op)
            )
            return operators & observed
        operators = {p6.ADD_ATOM, p6.STOP}
        if count >= 2:
            operators.add(p6.ADD_BOND)
        return operators

    op = tail[0]
    atom_states = {token for token in observed if token.startswith("<AS_")}
    fragments = {token for token in observed if token.startswith("<FRAG_")}
    atoms = {token for token in observed if token.startswith("<ATOM_")}
    orders = ORDER_TOKENS & observed

    if op == p6.INIT_ATOM:
        return atom_states if len(tail) == 1 else {p6.ACTION_END}
    if op == p6.ADD_ATOM:
        if len(tail) == 1:
            return site_tokens(vocab_tokens, count)
        if len(tail) == 2:
            return atom_states
        if len(tail) == 3:
            return ({"<ORDER_SINGLE>"} if source_conditioned else orders) & observed
        return {p6.ACTION_END}
    if op == p6.ADD_BOND:
        if len(tail) == 1:
            return site_tokens(vocab_tokens, count)
        if len(tail) == 2:
            return site_tokens(vocab_tokens, count, exclude=tail[1])
        if len(tail) == 3:
            return orders
        return {p6.ACTION_END}
    if op in {"<ADD_FRAGMENT>", "<SUBSTITUTE_TERMINAL>"}:
        if len(tail) == 1:
            return source_site_tokens(initial_smiles, op) & vocab_tokens
        if len(tail) == 2:
            return fragments
        return {p6.ACTION_END}
    if op == "<REPLACE_ATOM>":
        if len(tail) == 1:
            return source_site_tokens(initial_smiles, op) & vocab_tokens
        if len(tail) == 2:
            return atoms
        return {p6.ACTION_END}
    if op == "<DELETE_TERMINAL_ATOM>":
        return (source_site_tokens(initial_smiles, op) & vocab_tokens) if len(tail) == 1 else {p6.ACTION_END}
    if op == "<CHANGE_BOND_ORDER>":
        if len(tail) == 1:
            return source_site_tokens(initial_smiles, op) & vocab_tokens
        if len(tail) == 2:
            return source_site_tokens(initial_smiles, op, first_site=tail[1]) & vocab_tokens
        if len(tail) == 3:
            return orders - {"<ORDER_AROMATIC>"}
        return {p6.ACTION_END}
    return {p6.STOP}


@torch.no_grad()
def constrained_generate(
    model: unified.ConditionedSmilesDecoder,
    condition: torch.Tensor,
    condition_mask: torch.Tensor,
    *,
    vocab: unified.SmilesVocabulary,
    initial_smiles: str,
    observed: set[str],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    batch = condition.shape[0]
    generated = torch.full((batch, 1), vocab.bos_id, dtype=torch.long, device=condition.device)
    finished = torch.zeros(batch, dtype=torch.bool, device=condition.device)
    vocab_tokens = set(vocab.token_to_id)
    for _ in range(max(1, max_new_tokens)):
        logits = model(condition, generated, condition_mask=condition_mask)[:, -1, :]
        for row_index in range(batch):
            if finished[row_index]:
                logits[row_index, :] = -torch.inf
                logits[row_index, vocab.eos_id] = 0.0
                continue
            decoded = vocab.decode(generated[row_index].tolist())
            if p6.PROGRAM in decoded:
                decoded = decoded[decoded.index(p6.PROGRAM) :]
            allowed = allowed_next_tokens(
                decoded,
                initial_smiles=initial_smiles,
                vocab_tokens=vocab_tokens,
                observed=observed,
            )
            allowed_ids = [vocab.eos_id if token == "<EOS>" else vocab.token_to_id[token] for token in allowed]
            mask = torch.ones(logits.shape[-1], dtype=torch.bool, device=logits.device)
            mask[allowed_ids] = False
            logits[row_index].masked_fill_(mask, -torch.inf)
        logits = logits / max(float(temperature), 1e-6)
        if 0 < int(top_k) < logits.shape[-1]:
            threshold = torch.topk(logits, int(top_k), dim=-1).values[:, -1:]
            logits = logits.masked_fill(logits < threshold, -torch.inf)
        logits = unified.top_p_filter(logits, top_p=float(top_p))
        probs = torch.nan_to_num(torch.softmax(logits, dim=-1), nan=0.0, posinf=0.0, neginf=0.0)
        next_ids = torch.multinomial(probs, num_samples=1).squeeze(1)
        generated = torch.cat([generated, next_ids[:, None]], dim=1)
        finished |= next_ids.eq(vocab.eos_id)
        if bool(finished.all()):
            break
    return generated


def sample(args: argparse.Namespace) -> int:
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
    observed = observed_tokens(args.train_csv)
    rows = p6.read_rows(args.eval_csv)
    if int(args.limit) > 0:
        rows = rows[: int(args.limit)]

    output: list[dict[str, object]] = []
    valid_rows = valid_candidates = grammar_complete = 0
    for row_index, row in enumerate(rows):
        condition_np = unified.condition_array_for_row(
            row,
            store,
            int(config["condition_dim"]),
            max_source_tokens=int(args.max_source_tokens),
            condition_layout=str(args.condition_layout),
        ).astype(np.float32)
        condition = torch.from_numpy(condition_np)[None, :, :].to(device)
        condition = condition.expand(int(args.num_samples), -1, -1)
        condition_mask = torch.ones(condition.shape[:2], dtype=torch.bool, device=device)
        generated_batch = constrained_generate(
            model,
            condition,
            condition_mask,
            vocab=vocab,
            initial_smiles=p6.source_for_row(row),
            observed=observed,
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            top_p=float(args.top_p),
        )
        row_valid = 0
        for sample_index, generated in enumerate(generated_batch.tolist()):
            tokens = p6.decoded_program(vocab, generated)
            complete = p6.STOP in tokens
            grammar_complete += int(complete)
            try:
                actions = p6.parse_program(tokens, tolerate_incomplete_suffix=False)
                smiles = p6.execute_program(p6.source_for_row(row), actions)
            except Exception:
                actions, smiles = [], ""
            canonical = unified.safe_canonical_smiles(smiles)
            valid_candidates += int(bool(canonical))
            row_valid += int(bool(canonical))
            item = dict(row)
            item.update(
                {
                    "generated_smiles": canonical,
                    "direct_candidate_index": sample_index,
                    "direct_candidate_raw_smiles": smiles,
                    "direct_candidate_canonical_smiles": canonical,
                    "method": "p7_state_constrained_molprogram",
                    "generation_rank": sample_index + 1,
                    "candidate_rank": sample_index + 1,
                    "p7_program_tokens_json": json.dumps(tokens),
                    "p7_executed_actions_json": json.dumps([asdict(action) for action in actions]),
                    "p7_executed_action_count": len(actions),
                    "p7_grammar_complete": str(complete),
                }
            )
            item.update(unified.candidate_metrics(row, canonical, source_similarity_threshold=0.65))
            item["direct_candidate_strict_fraction"] = item["unified_property_success_fraction"]
            output.append(item)
        valid_rows += int(row_valid > 0)
        if (row_index + 1) % 20 == 0 or row_index + 1 == len(rows):
            print(f"[p7-sample] {row_index + 1}/{len(rows)} rows", flush=True)

    p6.write_rows(args.candidate_output_csv, output)
    total = len(rows) * int(args.num_samples)
    summary = {
        "protocol": "p7_frozen_p6_state_constrained_sampling",
        "checkpoint_changed": False,
        "eval_rows": len(rows),
        "rows_with_valid_candidate": valid_rows,
        "row_validity": valid_rows / max(len(rows), 1),
        "valid_candidates": valid_candidates,
        "candidate_yield": valid_candidates / max(total, 1),
        "grammar_complete_candidates": grammar_complete,
        "grammar_completion": grammar_complete / max(total, 1),
        "num_samples": int(args.num_samples),
        "observed_action_tokens": len(observed),
        "candidate_output_csv": str(args.candidate_output_csv),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return sample(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
