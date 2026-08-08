#!/usr/bin/env python3
"""Train and evaluate a unified SMILES/GraphEditDSL transformation policy.

The same ``ConditionedSmilesDecoder`` keeps its original de-novo contract and
learns a second, source-conditioned contract for editing: score short,
executable graph-edit programs.  Edit decoding is grammar constrained by
enumerating valid one-step programs for the source molecule, then ranking those
programs with the common decoder's sequence likelihood.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
GRAPH_SCRIPT_DIR = PROJECT_DIR / "scripts"
for import_dir in (SCRIPT_DIR, GRAPH_SCRIPT_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

import build_external_graph_edit_agent_predictions as graph  # noqa: E402
import unified_smiles_generator as unified  # noqa: E402


try:
    from rdkit import RDLogger

    # Invalid local edits are an expected part of grammar enumeration and are
    # filtered by the executor. Keep pilot logs focused on progress/metrics.
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    pass


EDIT_TOKEN = "<EDIT>"
OP_TOKENS = {
    "add_atom": "<ADD_ATOM>",
    "add_fragment": "<ADD_FRAGMENT>",
    "substitute_terminal": "<SUBSTITUTE_TERMINAL>",
    "replace_atom": "<REPLACE_ATOM>",
    "delete_terminal_atom": "<DELETE_TERMINAL_ATOM>",
    "change_bond_order": "<CHANGE_BOND_ORDER>",
}
ATOM_VALUES = ("C", "N", "O", "F", "Cl", "Br", "S")
FRAGMENT_TOKENS = {
    "C": "<FRAG_METHYL>",
    "CC": "<FRAG_ETHYL>",
    "O": "<FRAG_HYDROXYL>",
    "N": "<FRAG_AMINO>",
    "C#N": "<FRAG_CYANO>",
    "C(=O)O": "<FRAG_CARBOXYL>",
    "C(=O)N": "<FRAG_AMIDE>",
    "CN": "<FRAG_AMINOMETHYL>",
    "CN(C)C": "<FRAG_DIMETHYLAMINO>",
    "C(F)(F)F": "<FRAG_TRIFLUOROMETHYL>",
    "c1ccccc1": "<FRAG_PHENYL>",
    "C(=N)N": "<FRAG_GUANIDINO>",
    "C(=NN)NN": "<FRAG_AMINOGUANIDINO>",
    "C(=O)NN": "<FRAG_HYDRAZIDE>",
    "NC(=O)C(=O)NN": "<FRAG_OXALYL_HYDRAZIDE>",
    "S(=O)(=O)N": "<FRAG_SULFONAMIDE>",
    "c1ncc[nH]1": "<FRAG_IMIDAZOLE>",
    "C1CC1": "<FRAG_CYCLOPROPYL>",
}
BOND_ORDERS = ("single", "double", "triple")
DEFAULT_SOURCE_SIMILARITY_THRESHOLD = 0.65
VOCAB_PARAMETER_NAMES = {
    "token_embedding.weight",
    "output.weight",
    "output.bias",
    "source_output.weight",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Project paired edits to executable one-step action labels.")
    prepare.add_argument("--input-csv", required=True, type=Path)
    prepare.add_argument("--output-csv", required=True, type=Path)
    prepare.add_argument("--manifest-json", required=True, type=Path)
    prepare.add_argument("--site-limit", type=int, default=32)
    prepare.add_argument("--max-actions-per-row", type=int, default=512)
    prepare.add_argument("--source-similarity-threshold", type=float, default=DEFAULT_SOURCE_SIMILARITY_THRESHOLD)
    prepare.add_argument("--seed", type=int, default=7)

    train = subparsers.add_parser("train", help="Warm-start the common decoder with mixed SMILES/action SFT.")
    train.add_argument("--base-checkpoint", required=True, type=Path)
    train.add_argument("--train-csv", required=True, type=Path)
    train.add_argument("--eval-csv", required=True, type=Path)
    train.add_argument("--train-features-dir", required=True, type=Path)
    train.add_argument("--eval-features-dir", required=True, type=Path)
    train.add_argument("--output-dir", required=True, type=Path)
    train.add_argument("--condition-feature-array", choices=("query_tokens", "pooled"), default="query_tokens")
    train.add_argument("--condition-feature-variant", default="full")
    train.add_argument("--condition-layout", default="transformation")
    train.add_argument("--max-smiles-length", type=int, default=160)
    train.add_argument("--max-source-tokens", type=int, default=96)
    train.add_argument("--max-site-index", type=int, default=127)
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--eval-batch-size", type=int, default=128)
    train.add_argument("--samples-per-epoch", type=int, default=4096)
    train.add_argument("--lr", type=float, default=5e-5)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--grad-clip", type=float, default=1.0)
    train.add_argument("--distill-weight", type=float, default=0.3)
    train.add_argument("--distill-temperature", type=float, default=1.0)
    train.add_argument(
        "--trainable-scope",
        choices=("all", "source_action"),
        default="source_action",
        help="source_action freezes the legacy de-novo path and trains only source-conditioned/action parameters.",
    )
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--device", default="auto")

    rank = subparsers.add_parser("rank", help="Rank executable GraphEditDSL programs with a trained checkpoint.")
    rank.add_argument("--checkpoint", required=True, type=Path)
    rank.add_argument("--eval-csv", required=True, type=Path)
    rank.add_argument("--eval-features-dir", required=True, type=Path)
    rank.add_argument("--candidate-output-csv", required=True, type=Path)
    rank.add_argument("--summary-json", required=True, type=Path)
    rank.add_argument("--condition-feature-array", choices=("query_tokens", "pooled"), default="query_tokens")
    rank.add_argument("--condition-feature-variant", default="full")
    rank.add_argument("--condition-layout", default="transformation")
    rank.add_argument("--max-source-tokens", type=int, default=96)
    rank.add_argument("--site-limit", type=int, default=32)
    rank.add_argument("--max-actions-per-row", type=int, default=512)
    rank.add_argument("--top-candidates", type=int, default=64)
    rank.add_argument("--score-batch-size", type=int, default=256)
    rank.add_argument("--source-similarity-threshold", type=float, default=0.65)
    rank.add_argument(
        "--compact-output",
        action="store_true",
        help="Write only candidate identity/ranking fields instead of duplicating every reference column.",
    )
    rank.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "prepare":
        return prepare_command(args)
    if args.command == "train":
        return train_command(args)
    if args.command == "rank":
        return rank_command(args)
    raise ValueError(f"Unsupported command: {args.command}")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {str(key): "" if value is None else str(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def row_id(row: Mapping[str, object]) -> str:
    for key in ("example_id", "condition_id", "sample_id", "pair_id", "variant_id"):
        value = str(row.get(key, "") or "").strip()
        if value:
            return value.split(":")[-1] if value.startswith("edit:") else value
    return ""


def site_token(site: int) -> str:
    return f"<SITE_{int(site):03d}>"


def atom_token(atom: str) -> str:
    return f"<ATOM_{str(atom or 'C')}>"


def order_token(order: str) -> str:
    return f"<ORDER_{str(order or 'single').upper()}>"


def action_program_tokens(action: graph.GraphEditAction) -> list[str]:
    if action.op not in OP_TOKENS:
        raise ValueError(f"Unsupported graph action: {action.op}")
    tokens = [EDIT_TOKEN, OP_TOKENS[action.op]]
    if action.op in {
        "add_atom",
        "add_fragment",
        "substitute_terminal",
        "replace_atom",
        "delete_terminal_atom",
    }:
        if action.site is None:
            raise ValueError(f"Action {action.op} is missing site")
        tokens.append(site_token(action.site))
    if action.op in {"add_atom", "replace_atom"}:
        tokens.append(atom_token(action.atom))
    elif action.op in {"add_fragment", "substitute_terminal"}:
        if action.fragment not in FRAGMENT_TOKENS:
            raise ValueError(f"Unsupported fragment: {action.fragment}")
        tokens.append(FRAGMENT_TOKENS[action.fragment])
    elif action.op == "change_bond_order":
        if not action.bond:
            raise ValueError("change_bond_order action is missing bond")
        tokens.extend((site_token(action.bond[0]), site_token(action.bond[1]), order_token(action.bond_order)))
    return tokens


def action_vocabulary(*, max_site_index: int = 127) -> list[str]:
    tokens = [EDIT_TOKEN, *OP_TOKENS.values()]
    tokens.extend(site_token(site) for site in range(max(0, int(max_site_index)) + 1))
    tokens.extend(atom_token(atom) for atom in ATOM_VALUES)
    tokens.extend(FRAGMENT_TOKENS.values())
    tokens.extend(order_token(order) for order in BOND_ORDERS)
    return list(dict.fromkeys(tokens))


def action_key(action: graph.GraphEditAction) -> tuple[object, ...]:
    return (
        action.op,
        action.site,
        tuple(action.bond) if action.bond else None,
        action.atom,
        action.fragment,
        action.bond_order,
    )


def normalized_planner_row(row: Mapping[str, str]) -> dict[str, str]:
    out = dict(row)
    specs = [(prop, direction) for prop, direction in unified.instruction_task_specs(row) if prop]
    if not specs:
        specs = [(prop, unified.property_direction(row, prop)) for prop in unified.selected_properties(row)]
    props = list(dict.fromkeys(prop for prop, _direction in specs if prop))
    directions = {
        prop: "increase" if direction > 0 else "decrease" if direction < 0 else "increase"
        for prop, direction in specs
    }
    if props:
        out["external_task_properties"] = ",".join(props)
        out["condition_properties"] = ",".join(props)
        out["external_property_directions_json"] = json.dumps(directions, sort_keys=True)
    return out


def universal_actions(source_smiles: str, *, site_limit: int) -> list[graph.GraphEditAction]:
    sites = graph.editable_atom_sites(source_smiles, site_limit=site_limit)
    terminal_sites = editable_terminal_sites(source_smiles, site_limit=site_limit)
    bonds = graph.editable_bond_sites(source_smiles, site_limit=site_limit)
    actions: list[graph.GraphEditAction] = []
    for site in sites:
        actions.extend(graph.GraphEditAction("add_atom", site=site, atom=atom) for atom in ATOM_VALUES)
        actions.extend(
            graph.GraphEditAction("add_fragment", site=site, fragment=fragment)
            for fragment in FRAGMENT_TOKENS
        )
        actions.extend(graph.GraphEditAction("replace_atom", site=site, atom=atom) for atom in ("C", "N", "O", "S"))
        actions.append(graph.GraphEditAction("delete_terminal_atom", site=site))
    for site in terminal_sites:
        actions.extend(
            graph.GraphEditAction("substitute_terminal", site=site, fragment=fragment)
            for fragment in FRAGMENT_TOKENS
        )
    for bond in bonds:
        actions.extend(
            graph.GraphEditAction("change_bond_order", bond=bond, bond_order=order)
            for order in BOND_ORDERS
        )
    return actions


def editable_terminal_sites(source_smiles: str, *, site_limit: int) -> list[int]:
    """Return heavy-atom leaves that can be replaced by a medicinal fragment."""
    try:
        from rdkit import Chem
    except ImportError:
        return []
    mol = Chem.MolFromSmiles(str(source_smiles or ""))
    if mol is None:
        return []
    return [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetDegree() == 1][
        : max(1, int(site_limit))
    ]


def balanced_action_cap(actions: Sequence[graph.GraphEditAction], limit: int) -> list[graph.GraphEditAction]:
    unique: list[graph.GraphEditAction] = []
    seen = set()
    for action in actions:
        key = action_key(action)
        if key not in seen:
            seen.add(key)
            unique.append(action)
    if limit <= 0 or len(unique) <= limit:
        return unique
    groups: dict[str, list[graph.GraphEditAction]] = defaultdict(list)
    for action in unique:
        groups[action.op].append(action)
    out: list[graph.GraphEditAction] = []
    names = sorted(groups)
    cursors = {name: 0 for name in names}
    while len(out) < int(limit):
        progressed = False
        for name in names:
            cursor = cursors[name]
            if cursor < len(groups[name]):
                out.append(groups[name][cursor])
                cursors[name] = cursor + 1
                progressed = True
                if len(out) >= int(limit):
                    break
        if not progressed:
            break
    return out


def enumerate_action_candidates(
    row: Mapping[str, str],
    *,
    site_limit: int,
    max_actions_per_row: int,
) -> list[tuple[graph.GraphEditAction, str, list[str]]]:
    source = str(row.get("source_smiles", "") or row.get("molecule_smiles", "")).strip()
    if not source:
        return []
    planner_row = normalized_planner_row(row)
    planned = graph.plan_graph_edit_actions(
        planner_row,
        source_smiles=source,
        site_limit=int(site_limit),
        max_plans_per_property=max(32, int(max_actions_per_row)),
        planner_mode="policy_graph_dsl",
    )
    actions = balanced_action_cap(
        [*planned, *universal_actions(source, site_limit=int(site_limit))],
        int(max_actions_per_row),
    )
    out: list[tuple[graph.GraphEditAction, str, list[str]]] = []
    canonical_source = unified.safe_canonical_smiles(source)
    seen_smiles = {canonical_source} if canonical_source else set()
    for action in actions:
        try:
            program = action_program_tokens(action)
        except ValueError:
            continue
        generated = graph.execute_graph_edit_action(source, action)
        canonical = unified.safe_canonical_smiles(generated)
        if not canonical or canonical in seen_smiles:
            continue
        seen_smiles.add(canonical)
        out.append((action, canonical, program))
    return out


def instruction_score_components(row: Mapping[str, str], smiles: str) -> dict[str, object]:
    """Score every official Table1 instruction, including TDC/SA oracles."""
    source = str(row.get("source_smiles", "") or row.get("molecule_smiles", "")).strip()
    successes = 0
    evaluated = 0
    normalized_margins: list[float] = []
    specs = [(prop, direction) for prop, direction in unified.instruction_task_specs(row) if direction]
    for prop, direction in specs:
        candidate_value = unified.score_property(smiles, prop)
        source_value = unified.score_property(source, prop)
        if candidate_value is None or source_value is None:
            normalized_margins.append(-1.0)
            continue
        evaluated += 1
        signed_delta = (float(candidate_value) - float(source_value)) * int(direction)
        successes += int(signed_delta > 0.0)
        normalizer = max(float(unified.PROPERTY_NORMALIZERS.get(prop, 1.0)), 1e-8)
        normalized_margins.append(signed_delta / normalizer)
    property_count = len(specs)
    success_fraction = successes / property_count if property_count else math.nan
    all_success = bool(property_count and evaluated == property_count and successes == property_count)
    return {
        "instruction_property_count": property_count,
        "instruction_evaluated_count": evaluated,
        "instruction_success_fraction": success_fraction,
        "instruction_all_success": all_success,
        "instruction_mean_margin": mean(normalized_margins) if normalized_margins else math.nan,
        "instruction_distance": (
            mean(max(0.0, -value) for value in normalized_margins) if normalized_margins else math.inf
        ),
    }


def supported_instruction_success(row: Mapping[str, str], smiles: str) -> tuple[float, int]:
    """Backward-compatible view of the now full-oracle instruction score."""
    score = instruction_score_components(row, smiles)
    return float(score["instruction_success_fraction"]), int(score["instruction_evaluated_count"])


def action_oracle_record(
    row: Mapping[str, str],
    candidate: tuple[graph.GraphEditAction, str, list[str]],
    *,
    source_similarity_threshold: float = DEFAULT_SOURCE_SIMILARITY_THRESHOLD,
) -> dict[str, object]:
    action, smiles, program = candidate
    source = str(row.get("source_smiles", "") or row.get("molecule_smiles", "")).strip()
    target = str(row.get("target_smiles", "") or "").strip()
    canonical_target = unified.safe_canonical_smiles(target)
    target_similarity = unified.morgan_tanimoto(canonical_target, smiles) if canonical_target else math.nan
    source_similarity = unified.morgan_tanimoto(source, smiles)
    instruction = instruction_score_components(row, smiles)
    source_similarity_success = bool(
        math.isfinite(source_similarity) and source_similarity >= float(source_similarity_threshold)
    )
    strict_success = bool(instruction["instruction_all_success"] and source_similarity_success)
    return {
        "action": action,
        "smiles": smiles,
        "program": program,
        "exact": bool(canonical_target and canonical_target == smiles),
        "target_similarity": target_similarity,
        "source_similarity": source_similarity,
        "source_similarity_success": source_similarity_success,
        "strict_success": strict_success,
        **instruction,
        # Preserve the v1 output names for downstream compatibility.
        "supported_success_fraction": instruction["instruction_success_fraction"],
        "supported_property_count": instruction["instruction_evaluated_count"],
    }


def oracle_rank_key(record: Mapping[str, object]) -> tuple[float, ...]:
    def finite(value: object, fallback: float = -1.0) -> float:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else fallback

    action = record["action"]
    policy_score = float(action.policy_score) if isinstance(action, graph.GraphEditAction) else 0.0
    return (
        float(bool(record["strict_success"])),
        float(bool(record["source_similarity_success"])),
        finite(record["instruction_success_fraction"]),
        float(bool(record["instruction_all_success"])),
        finite(record["source_similarity"]),
        -finite(record["instruction_distance"], fallback=1e6),
        finite(record["instruction_mean_margin"]),
        float(bool(record["exact"])),
        finite(record["target_similarity"]),
        policy_score,
    )


def prepare_action_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    site_limit: int,
    max_actions_per_row: int,
    source_similarity_threshold: float = DEFAULT_SOURCE_SIMILARITY_THRESHOLD,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    output: list[dict[str, object]] = []
    oracle_records: list[dict[str, object]] = []
    skipped = Counter()
    op_counts = Counter()
    task_oracle_stats: dict[str, dict[str, object]] = defaultdict(
        lambda: {
            "rows": 0,
            "fully_evaluable_rows": 0,
            "source_feasible_rows": 0,
            "strict_reachable_rows": 0,
            "selected_strict_rows": 0,
            "best_success_fractions": [],
        }
    )
    for index, row in enumerate(rows):
        mode = unified.task_mode_for_row(row)
        if mode == unified.DE_NOVO_MODE:
            output.append(dict(row))
            continue
        candidates = enumerate_action_candidates(
            row,
            site_limit=int(site_limit),
            max_actions_per_row=int(max_actions_per_row),
        )
        if not candidates:
            skipped["no_executable_action"] += 1
            continue
        records = [
            action_oracle_record(
                row,
                candidate,
                source_similarity_threshold=float(source_similarity_threshold),
            )
            for candidate in candidates
        ]
        best = max(records, key=oracle_rank_key)
        action = best["action"]
        assert isinstance(action, graph.GraphEditAction)
        op_counts[action.op] += 1
        out = dict(row)
        out.update(
            {
                "policy_target_tokens_json": json.dumps(best["program"]),
                "policy_target_action_json": json.dumps(asdict(action), sort_keys=True),
                "policy_target_smiles": best["smiles"],
                "policy_target_exact": str(bool(best["exact"])),
                "policy_target_similarity": format_float(best["target_similarity"]),
                "policy_target_source_tanimoto": format_float(best["source_similarity"]),
                "policy_target_supported_success_fraction": format_float(best["supported_success_fraction"]),
                "policy_target_supported_property_count": int(best["supported_property_count"]),
                "policy_target_instruction_property_count": int(best["instruction_property_count"]),
                "policy_target_instruction_evaluated_count": int(best["instruction_evaluated_count"]),
                "policy_target_instruction_all_success": str(bool(best["instruction_all_success"])),
                "policy_target_source_similarity_success": str(bool(best["source_similarity_success"])),
                "policy_target_strict_success": str(bool(best["strict_success"])),
                "policy_target_instruction_distance": format_float(best["instruction_distance"]),
                "policy_target_instruction_mean_margin": format_float(best["instruction_mean_margin"]),
                "policy_action_candidate_count": len(candidates),
            }
        )
        output.append(out)
        oracle_records.append(best)
        task_key = str(row.get("moledit_task_key", "") or "").strip()
        if not task_key:
            task_key = "+".join(
                f"{prop}:{'increase' if direction > 0 else 'decrease'}"
                for prop, direction in unified.instruction_task_specs(row)
            ) or "unknown"
        task_stats = task_oracle_stats[task_key]
        task_stats["rows"] = int(task_stats["rows"]) + 1
        property_count = int(records[0]["instruction_property_count"])
        if property_count and any(int(record["instruction_evaluated_count"]) == property_count for record in records):
            task_stats["fully_evaluable_rows"] = int(task_stats["fully_evaluable_rows"]) + 1
        if any(bool(record["source_similarity_success"]) for record in records):
            task_stats["source_feasible_rows"] = int(task_stats["source_feasible_rows"]) + 1
        if any(bool(record["strict_success"]) for record in records):
            task_stats["strict_reachable_rows"] = int(task_stats["strict_reachable_rows"]) + 1
        if bool(best["strict_success"]):
            task_stats["selected_strict_rows"] = int(task_stats["selected_strict_rows"]) + 1
        finite_fractions = [
            float(record["instruction_success_fraction"])
            for record in records
            if math.isfinite(float(record["instruction_success_fraction"]))
        ]
        if finite_fractions:
            cast_fractions = task_stats["best_success_fractions"]
            assert isinstance(cast_fractions, list)
            cast_fractions.append(max(finite_fractions))
        if (index + 1) % 50 == 0:
            print(f"[graph-action-prepare] {index + 1}/{len(rows)} rows", flush=True)

    edit_input = sum(unified.task_mode_for_row(row) == unified.EDIT_MODE for row in rows)
    exact = sum(bool(record["exact"]) for record in oracle_records)
    target_sims = [float(record["target_similarity"]) for record in oracle_records]
    target_sims = [value for value in target_sims if math.isfinite(value)]
    source_sims = [float(record["source_similarity"]) for record in oracle_records]
    source_sims = [value for value in source_sims if math.isfinite(value)]
    supported = [
        float(record["supported_success_fraction"])
        for record in oracle_records
        if int(record["supported_property_count"]) > 0
        and math.isfinite(float(record["supported_success_fraction"]))
    ]
    per_task_oracle = {}
    for task_key, raw_stats in sorted(task_oracle_stats.items()):
        row_count = int(raw_stats["rows"])
        best_fractions = list(raw_stats.pop("best_success_fractions"))
        per_task_oracle[task_key] = {
            **raw_stats,
            "fully_evaluable_rate": int(raw_stats["fully_evaluable_rows"]) / max(row_count, 1),
            "source_feasible_rate": int(raw_stats["source_feasible_rows"]) / max(row_count, 1),
            "strict_reachability": int(raw_stats["strict_reachable_rows"]) / max(row_count, 1),
            "selected_strict_rate": int(raw_stats["selected_strict_rows"]) / max(row_count, 1),
            "mean_best_instruction_success_fraction": mean(best_fractions) if best_fractions else math.nan,
        }
    manifest = {
        "input_rows": len(rows),
        "output_rows": len(output),
        "de_novo_rows": sum(unified.task_mode_for_row(row) == unified.DE_NOVO_MODE for row in output),
        "edit_input_rows": edit_input,
        "edit_action_rows": len(oracle_records),
        "edit_action_coverage": len(oracle_records) / max(edit_input, 1),
        "exact_reconstruction_rows": exact,
        "exact_reconstruction_rate": exact / max(len(oracle_records), 1),
        "mean_best_target_similarity": mean(target_sims) if target_sims else math.nan,
        "best_target_similarity_at_0_65": sum(value >= 0.65 for value in target_sims) / max(len(target_sims), 1),
        "mean_best_source_similarity": mean(source_sims) if source_sims else math.nan,
        "best_source_similarity_at_0_65": sum(value >= 0.65 for value in source_sims) / max(len(source_sims), 1),
        "supported_instruction_rows": len(supported),
        "mean_supported_instruction_success": mean(supported) if supported else math.nan,
        "selected_strict_instruction_rows": sum(bool(record["strict_success"]) for record in oracle_records),
        "selected_strict_instruction_rate": (
            sum(bool(record["strict_success"]) for record in oracle_records) / max(len(oracle_records), 1)
        ),
        "source_similarity_threshold": float(source_similarity_threshold),
        "instruction_oracle_by_task": per_task_oracle,
        "selected_action_ops": dict(sorted(op_counts.items())),
        "skipped": dict(sorted(skipped.items())),
        "site_limit": int(site_limit),
        "max_actions_per_row": int(max_actions_per_row),
    }
    return output, manifest


def prepare_command(args: argparse.Namespace) -> int:
    random.seed(int(args.seed))
    rows, manifest = prepare_action_rows(
        read_rows(args.input_csv),
        site_limit=int(args.site_limit),
        max_actions_per_row=int(args.max_actions_per_row),
        source_similarity_threshold=float(args.source_similarity_threshold),
    )
    write_rows(args.output_csv, rows)
    manifest.update({"input_csv": str(args.input_csv), "output_csv": str(args.output_csv), "seed": int(args.seed)})
    args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def expand_checkpoint_model(
    checkpoint: Mapping[str, object],
    *,
    max_site_index: int,
    device: torch.device,
) -> tuple[unified.ConditionedSmilesDecoder, unified.SmilesVocabulary, dict[str, object], int]:
    vocab = unified.SmilesVocabulary.from_dict(checkpoint["vocab"])
    old_vocab_size = len(vocab.token_to_id)
    vocab.update([action_vocabulary(max_site_index=int(max_site_index))])
    config = dict(checkpoint["model_config"])
    config["vocab_size"] = len(vocab.token_to_id)
    model = unified.ConditionedSmilesDecoder(**config).to(device)
    target_state = model.state_dict()
    source_state = checkpoint["model_state"]
    if not isinstance(source_state, Mapping):
        raise TypeError("Checkpoint model_state must be a mapping")
    for name, source_tensor in source_state.items():
        if name not in target_state:
            raise ValueError(f"Unexpected checkpoint parameter: {name}")
        target_tensor = target_state[name]
        if source_tensor.shape == target_tensor.shape:
            target_state[name] = source_tensor.to(device=target_tensor.device, dtype=target_tensor.dtype)
            continue
        if name not in VOCAB_PARAMETER_NAMES:
            raise ValueError(
                f"Only vocabulary parameters may expand; {name} changed {tuple(source_tensor.shape)} -> "
                f"{tuple(target_tensor.shape)}"
            )
        if source_tensor.ndim != target_tensor.ndim or any(
            old > new for old, new in zip(source_tensor.shape, target_tensor.shape)
        ):
            raise ValueError(f"Invalid vocabulary expansion for {name}")
        expanded = target_tensor.clone()
        slices = tuple(slice(0, int(size)) for size in source_tensor.shape)
        expanded[slices] = source_tensor.to(device=target_tensor.device, dtype=target_tensor.dtype)
        target_state[name] = expanded
    model.load_state_dict(target_state)
    return model, vocab, config, old_vocab_size


def build_policy_dataset(
    rows: Sequence[Mapping[str, str]],
    vocab: unified.SmilesVocabulary,
    store: unified.FeatureStore,
    condition_dim: int,
    *,
    max_smiles_length: int,
    max_source_tokens: int,
    condition_layout: str,
) -> list[dict[str, object]]:
    dataset: list[dict[str, object]] = []
    for row in rows:
        mode = unified.task_mode_for_row(row)
        if mode == unified.EDIT_MODE:
            raw = str(row.get("policy_target_tokens_json", "") or "").strip()
            if not raw:
                continue
            try:
                tokens = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
                continue
        else:
            target = str(row.get("target_smiles", "") or "").strip()
            if not target:
                continue
            tokens = unified.tokenize_smiles(target)[: max(1, int(max_smiles_length))]
        decoder_input = vocab.encode(tokens, add_bos=True, add_eos=False)
        target_ids = vocab.encode(tokens, add_bos=False, add_eos=True)
        condition = unified.condition_array_for_row(
            row,
            store,
            int(condition_dim),
            max_source_tokens=int(max_source_tokens),
            condition_layout=str(condition_layout),
        )
        dataset.append(
            {
                "row": dict(row),
                "condition": condition.astype(np.float32),
                "decoder_input_ids": np.asarray(decoder_input, dtype=np.int64),
                "target_ids": np.asarray(target_ids, dtype=np.int64),
                "task_mode": mode,
            }
        )
    return dataset


def load_teacher(checkpoint: Mapping[str, object], device: torch.device) -> unified.ConditionedSmilesDecoder:
    config = dict(checkpoint["model_config"])
    teacher = unified.ConditionedSmilesDecoder(**config).to(device)
    teacher.load_state_dict(checkpoint["model_state"])
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def configure_trainable_scope(
    model: unified.ConditionedSmilesDecoder,
    *,
    scope: str,
    old_vocab_size: int,
) -> dict[str, object]:
    if str(scope) == "all":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        return {
            "scope": "all",
            "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "frozen_parameters": 0,
        }
    if str(scope) != "source_action":
        raise ValueError(f"Unsupported trainable scope: {scope}")
    safe_prefixes = (
        "source_condition_proj.",
        "source_encoder.",
        "source_type",
        "null_source",
        "source_gate.",
        "source_output.",
    )
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(safe_prefixes))

    # Action embeddings are edit-only decoder inputs. Train just their appended
    # rows while keeping every legacy SMILES embedding bit-identical.
    model.token_embedding.weight.requires_grad_(True)
    gradient_mask = torch.zeros_like(model.token_embedding.weight)
    gradient_mask[int(old_vocab_size) :, :] = 1.0
    model.token_embedding.weight.register_hook(lambda gradient: gradient * gradient_mask)

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return {
        "scope": "source_action",
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
        "protected_legacy_vocab_rows": int(old_vocab_size),
    }


def train_command(args: argparse.Namespace) -> int:
    unified.seed_everything(int(args.seed))
    device = unified.resolve_device(str(args.device))
    checkpoint = unified.load_checkpoint(args.base_checkpoint)
    if checkpoint is None:
        raise FileNotFoundError(args.base_checkpoint)
    model, vocab, config, old_vocab_size = expand_checkpoint_model(
        checkpoint,
        max_site_index=int(args.max_site_index),
        device=device,
    )
    teacher = load_teacher(checkpoint, device)
    trainable_scope = configure_trainable_scope(
        model,
        scope=str(args.trainable_scope),
        old_vocab_size=old_vocab_size,
    )
    train_store = unified.FeatureStore(
        args.train_features_dir,
        array_name=str(args.condition_feature_array),
        variant=str(args.condition_feature_variant),
    )
    eval_store = unified.FeatureStore(
        args.eval_features_dir,
        array_name=str(args.condition_feature_array),
        variant=str(args.condition_feature_variant),
    )
    train_dataset = build_policy_dataset(
        read_rows(args.train_csv),
        vocab,
        train_store,
        int(config["condition_dim"]),
        max_smiles_length=int(args.max_smiles_length),
        max_source_tokens=int(args.max_source_tokens),
        condition_layout=str(args.condition_layout),
    )
    eval_dataset = build_policy_dataset(
        read_rows(args.eval_csv),
        vocab,
        eval_store,
        int(config["condition_dim"]),
        max_smiles_length=int(args.max_smiles_length),
        max_source_tokens=int(args.max_source_tokens),
        condition_layout=str(args.condition_layout),
    )
    if not train_dataset:
        raise ValueError("No policy training rows")
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(args.lr),
        # Decoupled decay would move protected embedding rows even with masked
        # gradients, so the source-only contract uses no weight decay.
        weight_decay=0.0 if str(args.trainable_scope) == "source_action" else float(args.weight_decay),
    )
    history: list[dict[str, object]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, int(args.epochs) + 1):
        record: dict[str, object] = unified.train_epoch(
            model,
            teacher,
            train_dataset,
            optimizer,
            batch_size=int(args.batch_size),
            grad_clip=float(args.grad_clip),
            device=device,
            seed=int(args.seed) + epoch,
            sampling_mode="task_balanced",
            samples_per_epoch=int(args.samples_per_epoch),
            distill_weight=float(args.distill_weight),
            distill_temperature=float(args.distill_temperature),
        )
        record["epoch"] = epoch
        record.update(
            {
                f"eval_{key}": value
                for key, value in unified.evaluate_loss(
                    model,
                    eval_dataset,
                    batch_size=int(args.eval_batch_size),
                    device=device,
                ).items()
            }
        )
        history.append(record)
        unified.save_checkpoint(
            args.output_dir / f"checkpoint_epoch_{epoch:03d}.pt",
            model,
            optimizer,
            vocab,
            config,
            epoch,
            history,
            args,
        )
        unified.save_checkpoint(
            args.output_dir / "latest_checkpoint.pt",
            model,
            optimizer,
            vocab,
            config,
            epoch,
            history,
            args,
        )
        print(json.dumps(record, indent=2, sort_keys=True), flush=True)
    checkpoint_path = args.output_dir / "umtp_graph_action_policy.pt"
    unified.save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        vocab,
        config,
        int(args.epochs),
        history,
        args,
    )
    summary = {
        "protocol": "unified_smiles_graph_action_policy",
        "checkpoint": str(checkpoint_path),
        "base_checkpoint": str(args.base_checkpoint),
        "old_vocab_size": old_vocab_size,
        "new_vocab_size": len(vocab.token_to_id),
        "trainable_scope": trainable_scope,
        "train_rows": len(train_dataset),
        "eval_rows": len(eval_dataset),
        "train_mode_counts": unified.task_mode_counts(train_dataset),
        "eval_mode_counts": unified.task_mode_counts(eval_dataset),
        "history": history,
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


@torch.no_grad()
def score_programs(
    model: unified.ConditionedSmilesDecoder,
    vocab: unified.SmilesVocabulary,
    condition: np.ndarray,
    programs: Sequence[Sequence[str]],
    *,
    batch_size: int,
    device: torch.device,
) -> list[float]:
    scores: list[float] = []
    model.eval()
    for start in range(0, len(programs), max(1, int(batch_size))):
        items = []
        for tokens in programs[start : start + max(1, int(batch_size))]:
            items.append(
                {
                    "condition": condition,
                    "decoder_input_ids": np.asarray(vocab.encode(tokens, add_bos=True), dtype=np.int64),
                    "target_ids": np.asarray(vocab.encode(tokens, add_eos=True), dtype=np.int64),
                    "task_mode": unified.EDIT_MODE,
                }
            )
        batch = {key: value.to(device) for key, value in unified.collate_batch(items, model.pad_id).items()}
        logits = model(
            batch["condition"],
            batch["decoder_input_ids"],
            condition_mask=batch["condition_mask"],
        )
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(-1, batch["target_ids"].unsqueeze(-1)).squeeze(-1)
        mask = batch["target_ids"].ne(model.pad_id)
        sequence_scores = (token_log_probs * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        scores.extend(float(value) for value in sequence_scores.cpu())
    return scores


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rank_command(args: argparse.Namespace) -> int:
    device = unified.resolve_device(str(args.device))
    checkpoint = unified.load_checkpoint(args.checkpoint)
    if checkpoint is None:
        raise FileNotFoundError(args.checkpoint)
    vocab = unified.SmilesVocabulary.from_dict(checkpoint["vocab"])
    config = dict(checkpoint["model_config"])
    model = unified.ConditionedSmilesDecoder(**config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    store = unified.FeatureStore(
        args.eval_features_dir,
        array_name=str(args.condition_feature_array),
        variant=str(args.condition_feature_variant),
    )
    output: list[dict[str, object]] = []
    row_summaries: list[dict[str, object]] = []
    rows = [row for row in read_rows(args.eval_csv) if unified.task_mode_for_row(row) == unified.EDIT_MODE]
    fingerprint = checkpoint_sha256(args.checkpoint)[:12]
    for index, row in enumerate(rows):
        candidates = enumerate_action_candidates(
            row,
            site_limit=int(args.site_limit),
            max_actions_per_row=int(args.max_actions_per_row),
        )
        if not candidates:
            row_summaries.append({"row_id": row_id(row), "candidate_count": 0})
            continue
        condition = unified.condition_array_for_row(
            row,
            store,
            int(config["condition_dim"]),
            max_source_tokens=int(args.max_source_tokens),
            condition_layout=str(args.condition_layout),
        ).astype(np.float32)
        programs = [program for _action, _smiles, program in candidates]
        scores = score_programs(
            model,
            vocab,
            condition,
            programs,
            batch_size=int(args.score_batch_size),
            device=device,
        )
        ranked = sorted(zip(candidates, scores), key=lambda item: item[1], reverse=True)
        top = ranked[: max(1, int(args.top_candidates))]
        pool_id = f"graph-action:{fingerprint}:{row_id(row)}"
        pool_hash = hashlib.sha256(
            "\n".join(smiles for ((_action, smiles, _program), _score) in ranked).encode("utf-8")
        ).hexdigest()
        target = str(row.get("target_smiles", "") or "")
        best_target_similarity = max(
            (unified.morgan_tanimoto(target, smiles) for ((_action, smiles, _program), _score) in ranked),
            default=math.nan,
        )
        for rank, ((action, smiles, program), score) in enumerate(top, start=1):
            if bool(args.compact_output):
                candidate_row = {
                    key: row[key]
                    for key in ("example_id", "condition_id", "sample_id", "pair_hash", "variant_id", "pair_id")
                    if str(row.get(key, "") or "").strip()
                }
            else:
                candidate_row = dict(row)
            candidate_row.update(
                {
                    "generated_smiles": smiles,
                    "method": "umtp_graph_action_policy",
                    "generation_rank": rank,
                    "candidate_rank": rank,
                    "candidate_pool_id": pool_id,
                    "candidate_pool_hash": pool_hash,
                    "graph_action_program_tokens_json": json.dumps(program),
                    "graph_action_json": json.dumps(asdict(action), sort_keys=True),
                    "graph_action_policy_logprob": format_float(score),
                    "graph_action_candidate_count": len(ranked),
                }
            )
            candidate_row.update(
                unified.candidate_metrics(
                    row,
                    smiles,
                    source_similarity_threshold=float(args.source_similarity_threshold),
                )
            )
            output.append(candidate_row)
        row_summaries.append(
            {
                "row_id": row_id(row),
                "candidate_count": len(ranked),
                "written_candidates": len(top),
                "best_target_similarity": best_target_similarity,
                "raw_source_similarity": unified.morgan_tanimoto(
                    str(row.get("source_smiles", "") or ""),
                    top[0][0][1],
                ),
            }
        )
        if (index + 1) % 20 == 0 or index + 1 == len(rows):
            print(f"[graph-action-rank] {index + 1}/{len(rows)} rows", flush=True)
    write_rows(args.candidate_output_csv, output)
    counts = [int(row["candidate_count"]) for row in row_summaries]
    target_sims = [float(row["best_target_similarity"]) for row in row_summaries if "best_target_similarity" in row]
    source_sims = [float(row["raw_source_similarity"]) for row in row_summaries if "raw_source_similarity" in row]
    summary = {
        "protocol": "umtp_graph_action_ranking",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha256(args.checkpoint),
        "eval_csv": str(args.eval_csv),
        "eval_rows": len(rows),
        "rows_with_candidates": sum(value > 0 for value in counts),
        "candidate_rows": len(output),
        "mean_executable_candidates": mean(counts) if counts else 0.0,
        "mean_oracle_target_similarity": mean(target_sims) if target_sims else math.nan,
        "oracle_target_similarity_at_0_65": sum(value >= 0.65 for value in target_sims) / max(len(target_sims), 1),
        "raw_source_similarity_at_0_65": sum(value >= 0.65 for value in source_sims) / max(len(source_sims), 1),
        "candidate_output_csv": str(args.candidate_output_csv),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def format_float(value: object, digits: int = 8) -> str:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{parsed:.{digits}g}" if math.isfinite(parsed) else ""


if __name__ == "__main__":
    raise SystemExit(main())
