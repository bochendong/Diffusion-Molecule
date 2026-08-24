#!/usr/bin/env python3
"""P6 molecular transition programs for both de-novo design and editing.

The task mode never selects a decoder, head, output representation, or
interpreter.  A row supplies an initial molecular graph (empty for de novo,
the source molecule for editing), and one autoregressive policy emits the same
typed graph-transition language in both cases.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Mapping, Sequence

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
UNIFIED_DIR = PROJECT_DIR / "experiments" / "unified_smiles_generator"
GRAPH_SCRIPT_DIR = PROJECT_DIR / "scripts"
for import_dir in (UNIFIED_DIR, GRAPH_SCRIPT_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

import build_external_graph_edit_agent_predictions as graph  # noqa: E402
import umtp_graph_action_policy as graph_policy  # noqa: E402
import unified_smiles_generator as unified  # noqa: E402

from rdkit import Chem, RDLogger  # noqa: E402


RDLogger.DisableLog("rdApp.*")

PROGRAM = "<MOL_PROGRAM>"
ACTION_END = "<ACTION_END>"
STOP = "<STOP>"
INIT_ATOM = "<INIT_ATOM>"
ADD_ATOM = "<ADD_ATOM>"
ADD_BOND = "<ADD_BOND>"
ORDER_TO_TOKEN = {
    "single": "<ORDER_SINGLE>",
    "double": "<ORDER_DOUBLE>",
    "triple": "<ORDER_TRIPLE>",
    "aromatic": "<ORDER_AROMATIC>",
}
TOKEN_TO_ORDER = {value: key for key, value in ORDER_TO_TOKEN.items()}
ORDER_TO_RDKIT = {
    "single": Chem.BondType.SINGLE,
    "double": Chem.BondType.DOUBLE,
    "triple": Chem.BondType.TRIPLE,
    "aromatic": Chem.BondType.AROMATIC,
}


@dataclass(frozen=True)
class AtomSpec:
    atomic_num: int
    formal_charge: int = 0
    explicit_hs: int = 0
    aromatic: bool = False
    no_implicit: bool = False
    chirality: int = 0


@dataclass(frozen=True)
class TransitionAction:
    op: str
    site: int | None = None
    site_b: int | None = None
    atom: AtomSpec | None = None
    bond_order: str = "single"
    fragment: str = ""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare")
    prepare.add_argument("--input-csv", required=True, type=Path)
    prepare.add_argument("--output-csv", required=True, type=Path)
    prepare.add_argument("--manifest-json", required=True, type=Path)
    prepare.add_argument("--max-program-tokens", type=int, default=188)

    sample = sub.add_parser("sample")
    sample.add_argument("--checkpoint", required=True, type=Path)
    sample.add_argument("--eval-csv", required=True, type=Path)
    sample.add_argument("--eval-features-dir", required=True, type=Path)
    sample.add_argument("--candidate-output-csv", required=True, type=Path)
    sample.add_argument("--summary-json", required=True, type=Path)
    sample.add_argument("--condition-layout", default="transformation")
    sample.add_argument("--max-source-tokens", type=int, default=96)
    sample.add_argument("--num-samples", type=int, default=20)
    sample.add_argument("--max-new-tokens", type=int, default=188)
    sample.add_argument("--temperature", type=float, default=0.8)
    sample.add_argument("--top-k", type=int, default=32)
    sample.add_argument("--top-p", type=float, default=0.95)
    sample.add_argument("--limit", type=int, default=0)
    sample.add_argument("--seed", type=int, default=7)
    sample.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [{key: value or "" for key, value in row.items()} for row in csv.DictReader(handle)]


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def atom_spec(atom: Chem.Atom) -> AtomSpec:
    return AtomSpec(
        atomic_num=int(atom.GetAtomicNum()),
        formal_charge=int(atom.GetFormalCharge()),
        explicit_hs=int(atom.GetNumExplicitHs()),
        aromatic=bool(atom.GetIsAromatic()),
        no_implicit=bool(atom.GetNoImplicit()),
        chirality=int(atom.GetChiralTag()),
    )


def atom_tokens(spec: AtomSpec) -> list[str]:
    return [
        f"<AS_{spec.atomic_num:03d}_{spec.formal_charge:+d}_{spec.explicit_hs}_"
        f"{int(spec.aromatic)}_{int(spec.no_implicit)}_{spec.chirality}>"
    ]


def parse_atom_tokens(tokens: Sequence[str]) -> AtomSpec:
    if len(tokens) != 1 or not tokens[0].startswith("<AS_") or not tokens[0].endswith(">"):
        raise ValueError("atom descriptor must contain one typed state token")
    fields = tokens[0][1:-1].split("_")
    if len(fields) != 7 or fields[0] != "AS":
        raise ValueError(f"invalid atom state token: {tokens[0]}")
    return AtomSpec(
        atomic_num=int(fields[1]),
        formal_charge=int(fields[2]),
        explicit_hs=int(fields[3]),
        aromatic=bool(int(fields[4])),
        no_implicit=bool(int(fields[5])),
        chirality=int(fields[6]),
    )


def bond_order(bond: Chem.Bond) -> str:
    if bond.GetIsAromatic():
        return "aromatic"
    value = float(bond.GetBondTypeAsDouble())
    return "triple" if value >= 2.5 else "double" if value >= 1.5 else "single"


def molecule_to_construction_actions(smiles: str) -> list[TransitionAction]:
    canonical = unified.safe_canonical_smiles(smiles)
    mol = Chem.MolFromSmiles(canonical) if canonical else None
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"invalid target molecule: {smiles!r}")
    tree_edges: set[tuple[int, int]] = set()
    actions: list[TransitionAction] = []
    for index, atom in enumerate(mol.GetAtoms()):
        earlier = sorted(
            (neighbor.GetIdx(), mol.GetBondBetweenAtoms(index, neighbor.GetIdx()))
            for neighbor in atom.GetNeighbors()
            if neighbor.GetIdx() < index
        )
        if not earlier:
            actions.append(TransitionAction("init_atom", atom=atom_spec(atom)))
            continue
        parent, parent_bond = earlier[0]
        tree_edges.add(tuple(sorted((parent, index))))
        actions.append(
            TransitionAction(
                "add_atom",
                site=parent,
                atom=atom_spec(atom),
                bond_order=bond_order(parent_bond),
            )
        )
    for bond in mol.GetBonds():
        edge = tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
        if edge not in tree_edges:
            actions.append(
                TransitionAction(
                    "add_bond",
                    site=edge[0],
                    site_b=edge[1],
                    bond_order=bond_order(bond),
                )
            )
    return actions


def graph_edit_to_transition(action: graph.GraphEditAction) -> TransitionAction:
    if action.op == "add_atom":
        atom = Chem.Atom(str(action.atom or "C"))
        return TransitionAction("add_atom", site=action.site, atom=atom_spec(atom))
    return TransitionAction(
        action.op,
        site=action.site,
        site_b=(action.bond[1] if action.bond else None),
        atom=(atom_spec(Chem.Atom(str(action.atom))) if action.atom else None),
        bond_order=str(action.bond_order or "single"),
        fragment=str(action.fragment or ""),
    )


def action_tokens(action: TransitionAction) -> list[str]:
    if action.op == "init_atom" and action.atom:
        return [INIT_ATOM, *atom_tokens(action.atom), ACTION_END]
    if action.op == "add_atom" and action.atom is not None and action.site is not None:
        return [
            ADD_ATOM,
            graph_policy.site_token(action.site),
            *atom_tokens(action.atom),
            ORDER_TO_TOKEN[action.bond_order],
            ACTION_END,
        ]
    if action.op == "add_bond" and action.site is not None and action.site_b is not None:
        return [
            ADD_BOND,
            graph_policy.site_token(action.site),
            graph_policy.site_token(action.site_b),
            ORDER_TO_TOKEN[action.bond_order],
            ACTION_END,
        ]
    legacy = graph.GraphEditAction(
        action.op,
        site=action.site,
        bond=((action.site, action.site_b) if action.site is not None and action.site_b is not None else None),
        atom=(Chem.GetPeriodicTable().GetElementSymbol(action.atom.atomic_num) if action.atom else ""),
        fragment=action.fragment,
        bond_order=action.bond_order,
    )
    tokens = graph_policy.action_program_tokens(legacy)
    return [*tokens[1:], ACTION_END]


def program_tokens(actions: Sequence[TransitionAction]) -> list[str]:
    return [PROGRAM, *(token for action in actions for token in action_tokens(action)), STOP]


def parse_site(token: str) -> int:
    if not token.startswith("<SITE_") or not token.endswith(">"):
        raise ValueError(f"invalid site token: {token}")
    return int(token[6:-1])


def parse_program(tokens: Sequence[str], *, tolerate_incomplete_suffix: bool = False) -> list[TransitionAction]:
    if not tokens or tokens[0] != PROGRAM:
        raise ValueError("program must start with <MOL_PROGRAM>")
    actions: list[TransitionAction] = []
    index = 1
    while index < len(tokens):
        if tokens[index] == STOP:
            return actions
        try:
            end = tokens.index(ACTION_END, index)
        except ValueError:
            if tolerate_incomplete_suffix:
                return actions
            raise ValueError("unterminated action")
        chunk = list(tokens[index:end])
        try:
            if chunk and chunk[0] == INIT_ATOM:
                actions.append(TransitionAction("init_atom", atom=parse_atom_tokens(chunk[1:])))
            elif chunk and chunk[0] == ADD_ATOM:
                actions.append(
                    TransitionAction(
                        "add_atom",
                        site=parse_site(chunk[1]),
                        atom=parse_atom_tokens(chunk[2:3]),
                        bond_order=TOKEN_TO_ORDER[chunk[3]],
                    )
                )
            elif chunk and chunk[0] == ADD_BOND:
                actions.append(
                    TransitionAction(
                        "add_bond",
                        site=parse_site(chunk[1]),
                        site_b=parse_site(chunk[2]),
                        bond_order=TOKEN_TO_ORDER[chunk[3]],
                    )
                )
            else:
                actions.append(parse_legacy_action(chunk))
        except (IndexError, KeyError, TypeError, ValueError):
            if tolerate_incomplete_suffix:
                return actions
            raise
        index = end + 1
    if tolerate_incomplete_suffix:
        return actions
    raise ValueError("program is missing <STOP>")


def parse_legacy_action(chunk: Sequence[str]) -> TransitionAction:
    inverse = {value: key for key, value in graph_policy.OP_TOKENS.items()}
    op = inverse[chunk[0]]
    if op == "change_bond_order":
        return TransitionAction(op, site=parse_site(chunk[1]), site_b=parse_site(chunk[2]), bond_order=TOKEN_TO_ORDER[chunk[3]])
    site = parse_site(chunk[1])
    if op in {"add_fragment", "substitute_terminal"}:
        fragments = {value: key for key, value in graph_policy.FRAGMENT_TOKENS.items()}
        return TransitionAction(op, site=site, fragment=fragments[chunk[2]])
    if op in {"replace_atom", "add_atom"}:
        symbol = chunk[2][6:-1]
        return TransitionAction(op, site=site, atom=atom_spec(Chem.Atom(symbol)))
    return TransitionAction(op, site=site)


def make_atom(spec: AtomSpec) -> Chem.Atom:
    atom = Chem.Atom(int(spec.atomic_num))
    atom.SetFormalCharge(int(spec.formal_charge))
    atom.SetNumExplicitHs(int(spec.explicit_hs))
    atom.SetIsAromatic(bool(spec.aromatic))
    atom.SetNoImplicit(bool(spec.no_implicit))
    atom.SetChiralTag(Chem.ChiralType(int(spec.chirality)))
    return atom


def execute_program(initial_smiles: str, actions: Sequence[TransitionAction]) -> str:
    initial = unified.safe_canonical_smiles(initial_smiles) if initial_smiles else ""
    base = Chem.MolFromSmiles(initial) if initial else Chem.Mol()
    if base is None:
        return ""
    rw = Chem.RWMol(base)
    for action in actions:
        if action.op == "init_atom" and action.atom is not None:
            rw.AddAtom(make_atom(action.atom))
        elif action.op == "add_atom" and action.atom is not None and action.site is not None:
            if not 0 <= int(action.site) < rw.GetNumAtoms():
                return ""
            new_index = rw.AddAtom(make_atom(action.atom))
            rw.AddBond(int(action.site), new_index, ORDER_TO_RDKIT[action.bond_order])
        elif action.op == "add_bond" and action.site is not None and action.site_b is not None:
            if not (0 <= int(action.site) < rw.GetNumAtoms() and 0 <= int(action.site_b) < rw.GetNumAtoms()):
                return ""
            if rw.GetBondBetweenAtoms(int(action.site), int(action.site_b)) is not None:
                return ""
            rw.AddBond(int(action.site), int(action.site_b), ORDER_TO_RDKIT[action.bond_order])
        else:
            try:
                current = Chem.MolToSmiles(rw.GetMol(), canonical=True)
                legacy = graph.GraphEditAction(
                    action.op,
                    site=action.site,
                    bond=((action.site, action.site_b) if action.site is not None and action.site_b is not None else None),
                    atom=(Chem.GetPeriodicTable().GetElementSymbol(action.atom.atomic_num) if action.atom else ""),
                    fragment=action.fragment,
                    bond_order=action.bond_order,
                )
                updated = graph.execute_graph_edit_action(current, legacy)
                updated_mol = Chem.MolFromSmiles(str(updated or ""))
                if updated_mol is None:
                    return ""
                rw = Chem.RWMol(updated_mol)
            except Exception:
                return ""
    try:
        result = rw.GetMol()
        Chem.SanitizeMol(result)
        return Chem.MolToSmiles(result, canonical=True, isomericSmiles=True)
    except Exception:
        return ""


def source_for_row(row: Mapping[str, str]) -> str:
    return str(row.get("source_smiles", "") or row.get("molecule_smiles", "") or "").strip()


def target_actions_for_row(row: Mapping[str, str]) -> list[TransitionAction]:
    if unified.task_mode_for_row(row) == unified.DE_NOVO_MODE:
        return molecule_to_construction_actions(str(row.get("target_smiles", "") or ""))
    payload = json.loads(str(row.get("policy_target_action_json", "") or "{}"))
    return [graph_edit_to_transition(graph.GraphEditAction(**payload))]


def prepare_command(args: argparse.Namespace) -> int:
    output: list[dict[str, object]] = []
    counts: dict[str, int] = {}
    exact: dict[str, list[bool]] = {}
    skipped: dict[str, int] = {}
    lengths: list[int] = []
    for row in read_rows(args.input_csv):
        mode = unified.task_mode_for_row(row)
        counts[mode] = counts.get(mode, 0) + 1
        try:
            actions = target_actions_for_row(row)
            tokens = program_tokens(actions)
            if len(tokens) > int(args.max_program_tokens):
                raise ValueError("program_too_long")
            generated = execute_program(source_for_row(row), actions)
            expected = (
                str(row.get("target_smiles", "") or "")
                if mode == unified.DE_NOVO_MODE
                else str(row.get("policy_target_smiles", "") or "")
            )
            expected_canonical = unified.safe_canonical_smiles(expected)
            same = bool(generated and expected_canonical and generated == expected_canonical)
            exact.setdefault(mode, []).append(same)
            lengths.append(len(tokens))
            item = dict(row)
            item.update(
                {
                    "policy_target_tokens_json": json.dumps(tokens),
                    "p6_initial_graph": source_for_row(row),
                    "p6_action_count": len(actions),
                    "p6_program_token_count": len(tokens),
                    "p6_reconstructed_smiles": generated,
                    "p6_exact_reconstruction": str(same),
                }
            )
            output.append(item)
        except Exception as exc:
            reason = str(exc) or type(exc).__name__
            skipped[reason] = skipped.get(reason, 0) + 1
    write_rows(args.output_csv, output)
    manifest = {
        "protocol": "p6_unified_molecular_transition_program",
        "input_rows": sum(counts.values()),
        "input_mode_counts": counts,
        "output_rows": len(output),
        "output_mode_counts": unified.task_mode_counts([{"task_mode": unified.task_mode_for_row(row)} for row in output]),
        "coverage": len(output) / max(sum(counts.values()), 1),
        "mean_program_tokens": mean(lengths) if lengths else math.nan,
        "max_program_tokens": max(lengths, default=0),
        "exact_reconstruction_rate_by_mode": {
            mode: sum(values) / max(len(values), 1) for mode, values in exact.items()
        },
        "skipped": skipped,
        "unification_contract": {
            "decoder_count": 1,
            "checkpoint_count": 1,
            "interpreter_count": 1,
            "task_router": False,
            "task_specific_head": False,
            "property_aware_finalizer": False,
            "only_mode_difference": "initial_graph_empty_or_source",
        },
    }
    args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def p6_allowed_tokens(vocab: unified.SmilesVocabulary) -> set[int]:
    prefixes = (
        "<MOL_PROGRAM>", "<ACTION_END>", "<STOP>", "<INIT_ATOM>", "<ADD_", "<SUBSTITUTE_",
        "<REPLACE_", "<DELETE_", "<CHANGE_", "<SITE_", "<AS_", "<ORDER_", "<FRAG_", "<ATOM_",
    )
    return {
        index for token, index in vocab.token_to_id.items()
        if token.startswith(prefixes)
    }


def decoded_program(vocab: unified.SmilesVocabulary, ids: Sequence[int]) -> list[str]:
    tokens = vocab.decode(ids)
    if PROGRAM in tokens:
        tokens = tokens[tokens.index(PROGRAM):]
    return tokens


def sample_command(args: argparse.Namespace) -> int:
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
    allowed = p6_allowed_tokens(vocab)
    blocked = [index for index in range(len(vocab.token_to_id)) if index not in allowed and index != vocab.eos_id]
    rows = read_rows(args.eval_csv)
    if int(args.limit) > 0:
        rows = rows[: int(args.limit)]
    output: list[dict[str, object]] = []
    valid_rows = 0
    valid_candidates = 0
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
        row_valid = 0
        seen: set[str] = set()
        generated_batch = model.generate(
            condition,
            bos_id=vocab.bos_id,
            eos_id=vocab.eos_id,
            max_new_tokens=int(args.max_new_tokens),
            condition_mask=condition_mask,
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            top_p=float(args.top_p),
            repetition_penalty=1.05,
            blocked_token_ids=blocked,
        )
        for sample_index, generated in enumerate(generated_batch.tolist()):
            tokens = decoded_program(vocab, generated)
            try:
                actions = parse_program(tokens, tolerate_incomplete_suffix=True)
                smiles = execute_program(source_for_row(row), actions)
            except Exception:
                actions, smiles = [], ""
            canonical = unified.safe_canonical_smiles(smiles)
            if canonical:
                row_valid += 1
                valid_candidates += 1
                seen.add(canonical)
            item = dict(row)
            item.update(
                {
                    "generated_smiles": canonical,
                    "direct_candidate_index": sample_index,
                    "direct_candidate_raw_smiles": smiles,
                    "direct_candidate_canonical_smiles": canonical,
                    "method": "p6_unified_transition_policy",
                    "generation_rank": sample_index + 1,
                    "candidate_rank": sample_index + 1,
                    "p6_program_tokens_json": json.dumps(tokens),
                    "p6_executed_actions_json": json.dumps([asdict(action) for action in actions]),
                    "p6_executed_action_count": len(actions),
                    "p6_raw_sample_index": sample_index,
                }
            )
            item.update(unified.candidate_metrics(row, canonical, source_similarity_threshold=0.65))
            item["direct_candidate_strict_fraction"] = item["unified_property_success_fraction"]
            output.append(item)
        valid_rows += int(row_valid > 0)
        if (row_index + 1) % 20 == 0 or row_index + 1 == len(rows):
            print(f"[p6-sample] {row_index + 1}/{len(rows)} rows", flush=True)
    write_rows(args.candidate_output_csv, output)
    summary = {
        "protocol": "p6_unified_molecular_transition_sampling",
        "eval_rows": len(rows),
        "rows_with_valid_candidate": valid_rows,
        "row_validity": valid_rows / max(len(rows), 1),
        "valid_candidates": valid_candidates,
        "candidate_yield": valid_candidates / max(len(rows) * int(args.num_samples), 1),
        "num_samples": int(args.num_samples),
        "candidate_output_csv": str(args.candidate_output_csv),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "prepare":
        return prepare_command(args)
    if args.command == "sample":
        return sample_command(args)
    raise ValueError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
