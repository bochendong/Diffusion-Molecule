#!/usr/bin/env python3
"""P8.1.3: falsify atom trajectories with one source-relative macro transaction.

This is an oracle *representation* audit, not a generative result.  Validation
targets are used only to ask whether one shared graph interpreter can exactly
represent both empty->molecule construction and source->target editing.  The
held-out vocabulary audit then asks whether the transaction payloads can be
named using motifs extracted from the fit partition only.

R1 represents the changed region as one payload.  R2 changes exactly one
factor: it factorizes that same payload on BRICS bonds.  The transaction,
source applicability check, graph interpreter, split, and metrics are fixed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Mapping, Sequence

from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS, rdMMPA


RDLogger.DisableLog("rdApp.*")
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
UCA_DIR = PROJECT_DIR / "experiments" / "unified_constraint_agent"
for path in (PROJECT_DIR, UCA_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import build_retrieved_delta_edit_candidates as mmpa  # noqa: E402


PROTOCOL = "p8_1_3_source_relative_macro_transaction_gate_v1"
PORT_RE = re.compile(r"\[(?:\d+)?\*[^\]]*\]")


@dataclass(frozen=True)
class MacroTransaction:
    """One commit over a virtual-root or a source-exposed region."""

    source_core: str
    source_variable: str
    payload_components: tuple[str, ...]
    payload_links: tuple[tuple[int, str], ...]
    replace_all: bool


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--payload-factor", choices=("whole", "brics"), required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--min-core-heavy-atoms", type=int, default=3)
    parser.add_argument("--max-variable-heavy-atoms", type=int, default=48)
    parser.add_argument("--gate-mode-coverage", type=float, default=0.85)
    parser.add_argument("--gate-exact", type=float, default=0.99)
    parser.add_argument("--gate-validity", type=float, default=0.99)
    parser.add_argument("--gate-holdout-vocab", type=float, default=0.40)
    return parser.parse_args(argv)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def canonical(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) if mol else ""


def port_id(atom: Chem.Atom) -> int:
    return int(atom.GetIsotope() or atom.GetAtomMapNum())


def normalized_motif(smiles: str) -> str:
    """Erase transaction-local port numbers but preserve motif chemistry."""
    mol = Chem.MolFromSmiles(str(smiles or ""), sanitize=False)
    if mol is None:
        return PORT_RE.sub("[*]", str(smiles or ""))
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetIsotope(0)
            atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def factor_payload(smiles: str, factor: str) -> tuple[tuple[str, ...], tuple[tuple[int, str], ...]]:
    """Serialize a region as components plus typed port links."""
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        mol = Chem.MolFromSmiles(str(smiles or ""), sanitize=False)
    if mol is None:
        raise ValueError("invalid_payload")
    # BRICS queries ring membership.  MMPA variables with a dummy attachment
    # can take the sanitize=False fallback, so initialize graph invariants
    # explicitly before FindBRICSBonds instead of turning an RDKit precondition
    # violation into apparent representation non-coverage.
    mol.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(mol)
    if factor == "whole":
        return (Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True),), ()
    candidates = list(BRICS.FindBRICSBonds(mol))
    bond_indices: list[int] = []
    labels: list[tuple[int, int]] = []
    links: list[tuple[int, str]] = []
    next_port = max((port_id(atom) for atom in mol.GetAtoms()), default=0) + 1
    for (left, right), _environment in candidates:
        bond = mol.GetBondBetweenAtoms(int(left), int(right))
        if bond is None:
            continue
        port = next_port
        next_port += 1
        bond_indices.append(int(bond.GetIdx()))
        labels.append((port, port))
        links.append((port, str(bond.GetBondType())))
    if not bond_indices:
        return (Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True),), ()
    fragmented = Chem.FragmentOnBonds(
        mol, bond_indices, addDummies=True, dummyLabels=labels
    )
    components = tuple(
        sorted(
            Chem.MolToSmiles(part, canonical=True, isomericSmiles=True)
            for part in Chem.GetMolFrags(fragmented, asMols=True, sanitizeFrags=False)
        )
    )
    return components, tuple(sorted(links))


def assemble_components(
    components: Sequence[str], links: Sequence[tuple[int, str]]
) -> str:
    mols = [Chem.MolFromSmiles(value, sanitize=False) for value in components]
    if not mols or any(mol is None for mol in mols):
        return ""
    combined = Chem.RWMol(mols[0])
    for mol in mols[1:]:
        combined.InsertMol(mol)
    explicit_link_types = {int(port): value for port, value in links}
    ports: dict[int, list[tuple[int, int, Chem.BondType]]] = defaultdict(list)
    for atom in combined.GetAtoms():
        if atom.GetAtomicNum() != 0:
            continue
        neighbors = list(atom.GetNeighbors())
        if len(neighbors) != 1:
            return ""
        bond = combined.GetBondBetweenAtoms(atom.GetIdx(), neighbors[0].GetIdx())
        ports[port_id(atom)].append(
            (atom.GetIdx(), neighbors[0].GetIdx(), bond.GetBondType())
        )
    removals: list[int] = []
    for port, endpoints in ports.items():
        if len(endpoints) != 2:
            return ""
        (dummy_a, atom_a, type_a), (dummy_b, atom_b, type_b) = endpoints
        if combined.GetBondBetweenAtoms(atom_a, atom_b) is not None:
            return ""
        requested = explicit_link_types.get(port, "")
        bond_type = type_a if type_a == type_b else Chem.BondType.SINGLE
        if requested:
            lookup = {
                "SINGLE": Chem.BondType.SINGLE,
                "DOUBLE": Chem.BondType.DOUBLE,
                "TRIPLE": Chem.BondType.TRIPLE,
                "AROMATIC": Chem.BondType.AROMATIC,
            }
            bond_type = lookup.get(requested, bond_type)
        combined.AddBond(int(atom_a), int(atom_b), bond_type)
        removals.extend((dummy_a, dummy_b))
    for index in sorted(removals, reverse=True):
        combined.RemoveAtom(int(index))
    try:
        result = combined.GetMol()
        Chem.SanitizeMol(result)
        return Chem.MolToSmiles(result, canonical=True, isomericSmiles=True)
    except Exception:
        return ""


def mmpa_splits(smiles: str, min_core: int, max_variable: int) -> tuple[mmpa.FragmentSplit, ...]:
    return mmpa.fragment_splits(smiles, min_core, max_variable)


def oracle_transaction(
    row: Mapping[str, str], factor: str, min_core: int, max_variable: int
) -> tuple[str, str, MacroTransaction]:
    mode = "edit" if str(row.get("source_smiles", "") or row.get("molecule_smiles", "")).strip() else "de_novo"
    source = canonical(str(row.get("source_smiles", "") or row.get("molecule_smiles", "")))
    target_field = "policy_target_smiles" if mode == "edit" else "target_smiles"
    target = canonical(str(row.get(target_field, "")))
    if not target:
        raise ValueError("invalid_target")
    if mode == "de_novo":
        components, links = factor_payload(target, factor)
        return mode, target, MacroTransaction("", "", components, links, True)

    source_by_core: dict[str, set[str]] = defaultdict(set)
    target_by_core: dict[str, set[str]] = defaultdict(set)
    for split in mmpa_splits(source, min_core, max_variable):
        source_by_core[split.core].add(split.variable)
    for split in mmpa_splits(target, min_core, max_variable):
        target_by_core[split.core].add(split.variable)
    choices: list[tuple[int, str, str, str]] = []
    for core in set(source_by_core) & set(target_by_core):
        for source_variable in source_by_core[core]:
            for target_variable in target_by_core[core]:
                if source_variable == target_variable:
                    continue
                if canonical(mmpa.join_fragments(core, target_variable)) != target:
                    continue
                target_mol = Chem.MolFromSmiles(target_variable)
                size = target_mol.GetNumHeavyAtoms() if target_mol else 10**6
                choices.append((size, core, source_variable, target_variable))
    if not choices:
        raise ValueError("no_exact_one_cut_region")
    _size, core, source_variable, target_variable = min(choices)
    components, links = factor_payload(target_variable, factor)
    return mode, target, MacroTransaction(core, source_variable, components, links, False)


def execute_transaction(source: str, transaction: MacroTransaction) -> str:
    """The same commit path handles virtual-root and source-root states."""
    if transaction.replace_all:
        return assemble_components(transaction.payload_components, transaction.payload_links)
    applicable = any(
        split.core == transaction.source_core
        and split.variable == transaction.source_variable
        for split in mmpa_splits(source, 1, 128)
    )
    if not applicable:
        return ""
    # Assign the source-core external port to zero and keep payload-local BRICS
    # ports distinct.  MMPA uses atom-map 1 on both sides.
    core = transaction.source_core
    payload = list(transaction.payload_components)
    core_mol = Chem.MolFromSmiles(core, sanitize=False)
    if core_mol is None:
        return ""
    core_ports = [atom for atom in core_mol.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(core_ports) != 1:
        return ""
    core_ports[0].SetAtomMapNum(1)
    core_ports[0].SetIsotope(0)
    # MMPA payload's external port remains map 1; BRICS ports use isotopes >1.
    return assemble_components(
        [Chem.MolToSmiles(core_mol, canonical=True), *payload],
        transaction.payload_links,
    )


def stable_key(row: Mapping[str, str], index: int) -> str:
    return str(row.get("condition_id", "") or row.get("sample_id", "") or index)


def group_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    covered = [row for row in rows if bool(row["covered"])]
    lengths = [int(row["macro_program_tokens"]) for row in covered]
    baseline = [int(row["p6_program_tokens"]) for row in covered if int(row["p6_program_tokens"]) > 0]
    return {
        "rows": len(rows),
        "covered_rows": len(covered),
        "coverage": len(covered) / max(1, len(rows)),
        "exact_reconstruction": sum(bool(row["exact"]) for row in covered) / max(1, len(covered)),
        "validity_reachability": sum(bool(row["valid"]) for row in covered) / max(1, len(covered)),
        "mean_macro_program_tokens": mean(lengths) if lengths else None,
        "median_macro_program_tokens": median(lengths) if lengths else None,
        "mean_p6_program_tokens": mean(baseline) if baseline else None,
        "token_compression_ratio": (sum(lengths) / sum(baseline)) if baseline and sum(baseline) else None,
        "mean_macro_actions": mean([int(row["macro_actions"]) for row in covered]) if covered else None,
    }


def brics_preflight() -> None:
    """Fail closed before a target-aware audit can mislabel infrastructure errors."""
    probe = "CC(=O)Nc1ccc(O)cc1"
    components, links = factor_payload(probe, "brics")
    reconstructed = assemble_components(components, links)
    if reconstructed != canonical(probe):
        raise RuntimeError(
            f"BRICS preflight reconstruction failed: {reconstructed!r} != {canonical(probe)!r}"
        )
    # Exercise the MMPA-variable path, including its unmatched external port.
    variable_components, _ = factor_payload("[*:1]CC(=O)Nc1ccccc1", "brics")
    if not variable_components:
        raise RuntimeError("BRICS MMPA-variable preflight produced no components")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if str(args.payload_factor) == "brics":
        brics_preflight()
    rows = read_rows(args.input_csv)
    evidence: list[dict[str, object]] = []
    transactions: list[MacroTransaction | None] = []
    failures: Counter[str] = Counter()
    for index, row in enumerate(rows):
        source = canonical(str(row.get("source_smiles", "") or row.get("molecule_smiles", "")))
        try:
            mode, target, transaction = oracle_transaction(
                row,
                str(args.payload_factor),
                int(args.min_core_heavy_atoms),
                int(args.max_variable_heavy_atoms),
            )
            generated = execute_transaction(source, transaction)
            exact = bool(generated and generated == target)
            valid = bool(canonical(generated))
            motif_tokens = tuple(normalized_motif(value) for value in transaction.payload_components)
            macro_tokens = 3 + len(motif_tokens) + len(transaction.payload_links)
            evidence.append(
                {
                    "key": stable_key(row, index),
                    "mode": mode,
                    "covered": True,
                    "exact": exact,
                    "valid": valid,
                    "motif_tokens": motif_tokens,
                    "macro_program_tokens": macro_tokens,
                    "macro_actions": 1,
                    "p6_program_tokens": int(float(row.get("p6_program_token_count", 0) or 0)),
                    "p6_actions": int(float(row.get("p6_action_count", 0) or 0)),
                    "payload_components": len(transaction.payload_components),
                    "payload_links": len(transaction.payload_links),
                }
            )
            transactions.append(transaction)
        except Exception as exc:
            mode = "edit" if source else "de_novo"
            reason = str(exc) or type(exc).__name__
            failures[f"{mode}:{reason}"] += 1
            evidence.append(
                {
                    "key": stable_key(row, index), "mode": mode, "covered": False,
                    "exact": False, "valid": False, "motif_tokens": (),
                    "macro_program_tokens": 0, "macro_actions": 0,
                    "p6_program_tokens": int(float(row.get("p6_program_token_count", 0) or 0)),
                    "p6_actions": int(float(row.get("p6_action_count", 0) or 0)),
                    "payload_components": 0, "payload_links": 0,
                }
            )
            transactions.append(None)

    infrastructure_failures = {
        reason: count
        for reason, count in failures.items()
        if "Pre-condition Violation" in reason or "RingInfo" in reason
    }
    if infrastructure_failures:
        raise RuntimeError(
            "Fail-closed RDKit infrastructure errors: "
            + json.dumps(infrastructure_failures, sort_keys=True)
        )

    # Frozen stratified fit/holdout split.  Targets only label this oracle audit;
    # holdout motifs never enter the vocabulary.
    rng = random.Random(int(args.seed))
    fit_indices: set[int] = set()
    holdout_indices: set[int] = set()
    for mode in ("de_novo", "edit"):
        indices = [i for i, item in enumerate(evidence) if item["mode"] == mode and item["covered"]]
        rng.shuffle(indices)
        holdout_count = max(1, round(len(indices) * float(args.holdout_fraction))) if indices else 0
        holdout_indices.update(indices[:holdout_count])
        fit_indices.update(indices[holdout_count:])
    vocabulary = {
        token
        for index in fit_indices
        for token in evidence[index]["motif_tokens"]
    }
    for index, item in enumerate(evidence):
        tokens = tuple(item["motif_tokens"])
        item["partition"] = "holdout" if index in holdout_indices else "fit"
        item["vocab_reachable"] = bool(tokens) and all(token in vocabulary for token in tokens)

    by_mode = {
        mode: group_summary([item for item in evidence if item["mode"] == mode])
        for mode in ("de_novo", "edit")
    }
    overall = group_summary(evidence)
    holdout_by_mode = {}
    for mode in ("de_novo", "edit"):
        selected = [evidence[index] for index in holdout_indices if evidence[index]["mode"] == mode]
        holdout_by_mode[mode] = {
            "rows": len(selected),
            "vocab_reachable_rows": sum(bool(item["vocab_reachable"]) for item in selected),
            "vocab_reachability": sum(bool(item["vocab_reachable"]) for item in selected) / max(1, len(selected)),
        }

    checks: dict[str, dict[str, object]] = {}
    for mode in ("de_novo", "edit"):
        checks[f"{mode}_coverage"] = {"value": by_mode[mode]["coverage"], "threshold": args.gate_mode_coverage}
        checks[f"{mode}_exact"] = {"value": by_mode[mode]["exact_reconstruction"], "threshold": args.gate_exact}
        checks[f"{mode}_validity"] = {"value": by_mode[mode]["validity_reachability"], "threshold": args.gate_validity}
        checks[f"{mode}_holdout_vocab"] = {"value": holdout_by_mode[mode]["vocab_reachability"], "threshold": args.gate_holdout_vocab}
    failed = [name for name, check in checks.items() if float(check["value"] or 0) < float(check["threshold"])]

    summary = {
        "protocol": PROTOCOL,
        "round": "R1" if args.payload_factor == "whole" else "R2",
        "single_changed_factor": "payload_factorization" if args.payload_factor == "brics" else None,
        "payload_factor": args.payload_factor,
        "input_csv": str(args.input_csv),
        "input_sha256": hashlib.sha256(args.input_csv.read_bytes()).hexdigest(),
        "seed": int(args.seed),
        "unification_contract": {
            "interpreter_count": 1,
            "transaction_type_count": 1,
            "checkpoint_count_required": 1,
            "task_router": False,
            "only_initial_state_difference": "virtual_empty_root_or_source_graph",
        },
        "overall": overall,
        "by_mode": by_mode,
        "heldout_train_only_vocabulary": {
            "fit_rows": len(fit_indices),
            "holdout_rows": len(holdout_indices),
            "unique_fit_motifs": len(vocabulary),
            "by_mode": holdout_by_mode,
        },
        "failure_reasons": dict(failures),
        "gate": {"passed": not failed, "checks": checks, "failures": failed},
        "decision": "GO_TRAIN" if not failed else "NO_GO_TRAIN",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (args.output_dir / "evidence.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [key for key in evidence[0] if key != "motif_tokens"] + ["motif_tokens_json"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in evidence:
            writer.writerow({**{key: value for key, value in item.items() if key != "motif_tokens"}, "motif_tokens_json": json.dumps(item["motif_tokens"])})
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
