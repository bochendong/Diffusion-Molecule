#!/usr/bin/env python3
"""Build a fixed-n source-preserving pool from train-only matched-pair deltas.

The tool learns one-cut side-chain substitutions from paired MuMO training
rows.  At inference it fragments only the evaluation source molecule, retrieves
same-task training deltas by source-fragment similarity, and rejoins the
training target fragment to the untouched evaluation core.  Evaluation targets
are deliberately never read.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
PROJECT_SCRIPTS = PROJECT_DIR / "scripts"
for path in (SCRIPT_DIR, PROJECT_DIR, PROJECT_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import build_external_graph_edit_agent_predictions as graph  # noqa: E402


@dataclass(frozen=True)
class FragmentSplit:
    core: str
    variable: str
    core_heavy_atoms: int
    variable_heavy_atoms: int


@dataclass(frozen=True)
class DeltaTransform:
    task_key: str
    source_variable: str
    target_variable: str
    frequency: int
    train_condition_id: str


@dataclass(frozen=True)
class Candidate:
    smiles: str
    source: str
    source_tanimoto: float
    admet_prior_score: float
    retrieval_similarity: float = 0.0
    transform_frequency: int = 0
    exact_variable_match: bool = False
    query_core: str = ""
    query_variable: str = ""
    source_variable: str = ""
    target_variable: str = ""
    train_condition_id: str = ""
    fallback_rank: int = 0

    @property
    def selection_score(self) -> float:
        return (
            float(self.admet_prior_score)
            + 0.30 * float(self.retrieval_similarity)
            + 0.20 * float(self.source_tanimoto)
            + 0.03 * math.log1p(max(int(self.transform_frequency), 0))
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--eval-csv", required=True, type=Path)
    parser.add_argument("--fallback-candidates-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--enumerated-output-csv", type=Path, default=None)
    parser.add_argument("--manifest-json", required=True, type=Path)
    parser.add_argument("--candidate-budget", type=int, default=20)
    parser.add_argument("--min-retrieval-similarity", type=float, default=0.15)
    parser.add_argument("--max-transforms-per-query", type=int, default=96)
    parser.add_argument("--min-core-heavy-atoms", type=int, default=5)
    parser.add_argument("--max-variable-heavy-atoms", type=int, default=30)
    parser.add_argument("--min-source-tanimoto", type=float, default=0.4)
    return parser.parse_args(argv)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def row_key(row: Mapping[str, object]) -> str:
    value = str(row.get("condition_id", "") or row.get("sample_id", "") or "").strip()
    if not value:
        raise ValueError("Every delta-tool row requires condition_id or sample_id")
    return value


def task_key(row: Mapping[str, object]) -> str:
    value = str(
        row.get("external_task_key", "")
        or row.get("external_task_id", "")
        or row.get("condition_properties", "")
        or ""
    ).strip()
    if not value:
        raise ValueError(f"Row {row_key(row)} is missing a task key")
    return value


@lru_cache(maxsize=200000)
def canonical_smiles(smiles: str) -> str:
    try:
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover - cluster dependency
        raise RuntimeError("RetrievedDeltaEdit requires RDKit") from exc
    mol = Chem.MolFromSmiles(str(smiles or ""))
    return Chem.MolToSmiles(mol, canonical=True) if mol is not None else ""


@lru_cache(maxsize=200000)
def fragment_splits(
    smiles: str,
    min_core_heavy_atoms: int,
    max_variable_heavy_atoms: int,
) -> tuple[FragmentSplit, ...]:
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMMPA
    except ImportError as exc:  # pragma: no cover - cluster dependency
        raise RuntimeError("RetrievedDeltaEdit requires RDKit MMPA") from exc
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return ()
    output: list[FragmentSplit] = []
    seen: set[tuple[str, str]] = set()
    for _core, chains in rdMMPA.FragmentMol(mol, maxCuts=1, resultsAsMols=False):
        parts = str(chains or "").split(".")
        if len(parts) != 2:
            continue
        oriented = ((parts[0], parts[1]), (parts[1], parts[0]))
        for raw_core, raw_variable in oriented:
            core = canonical_smiles(raw_core)
            variable = canonical_smiles(raw_variable)
            if not core or not variable or (core, variable) in seen:
                continue
            core_mol = Chem.MolFromSmiles(core)
            variable_mol = Chem.MolFromSmiles(variable)
            if core_mol is None or variable_mol is None:
                continue
            core_heavy = sum(atom.GetAtomicNum() > 1 for atom in core_mol.GetAtoms())
            variable_heavy = sum(atom.GetAtomicNum() > 1 for atom in variable_mol.GetAtoms())
            if core_heavy < max(1, int(min_core_heavy_atoms)):
                continue
            if variable_heavy > max(1, int(max_variable_heavy_atoms)):
                continue
            if sum(atom.GetAtomicNum() == 0 for atom in core_mol.GetAtoms()) != 1:
                continue
            if sum(atom.GetAtomicNum() == 0 for atom in variable_mol.GetAtoms()) != 1:
                continue
            seen.add((core, variable))
            output.append(FragmentSplit(core, variable, core_heavy, variable_heavy))
    return tuple(output)


def build_transform_index(
    rows: Sequence[Mapping[str, object]],
    *,
    min_core_heavy_atoms: int,
    max_variable_heavy_atoms: int,
) -> tuple[dict[str, list[DeltaTransform]], dict[str, object]]:
    counts: Counter[tuple[str, str, str]] = Counter()
    first_condition: dict[tuple[str, str, str], str] = {}
    rows_with_delta = 0
    for row in rows:
        source = str(row.get("source_smiles", "") or "").strip()
        target = str(row.get("target_smiles", "") or "").strip()
        source_by_core: dict[str, set[str]] = defaultdict(set)
        target_by_core: dict[str, set[str]] = defaultdict(set)
        for split in fragment_splits(source, min_core_heavy_atoms, max_variable_heavy_atoms):
            source_by_core[split.core].add(split.variable)
        for split in fragment_splits(target, min_core_heavy_atoms, max_variable_heavy_atoms):
            target_by_core[split.core].add(split.variable)
        task = task_key(row)
        added = False
        for core in sorted(set(source_by_core) & set(target_by_core)):
            for source_variable in sorted(source_by_core[core]):
                for target_variable in sorted(target_by_core[core]):
                    if source_variable == target_variable:
                        continue
                    key = (task, source_variable, target_variable)
                    counts[key] += 1
                    first_condition.setdefault(key, row_key(row))
                    added = True
        rows_with_delta += int(added)
    by_task: dict[str, list[DeltaTransform]] = defaultdict(list)
    for (task, source_variable, target_variable), frequency in counts.items():
        by_task[task].append(
            DeltaTransform(
                task_key=task,
                source_variable=source_variable,
                target_variable=target_variable,
                frequency=int(frequency),
                train_condition_id=first_condition[(task, source_variable, target_variable)],
            )
        )
    for task in by_task:
        by_task[task].sort(
            key=lambda item: (-item.frequency, item.source_variable, item.target_variable)
        )
    manifest = {
        "training_rows": len(rows),
        "training_rows_with_delta": rows_with_delta,
        "task_count": len(by_task),
        "unique_transforms": len(counts),
        "transform_observations": sum(counts.values()),
        "transforms_by_task": dict(sorted(Counter(task for task, _source, _target in counts).items())),
    }
    return dict(by_task), manifest


@lru_cache(maxsize=200000)
def variable_fingerprint(variable: str):
    try:
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator
    except ImportError as exc:  # pragma: no cover - cluster dependency
        raise RuntimeError("RetrievedDeltaEdit requires RDKit fingerprints") from exc
    mol = Chem.MolFromSmiles(str(variable or ""))
    if mol is None:
        return None
    rw = Chem.RWMol(mol)
    for index in sorted(
        (atom.GetIdx() for atom in rw.GetAtoms() if atom.GetAtomicNum() == 0),
        reverse=True,
    ):
        rw.RemoveAtom(int(index))
    stripped = rw.GetMol()
    if stripped.GetNumHeavyAtoms() == 0:
        return None
    try:
        Chem.SanitizeMol(stripped)
    except Exception:
        return None
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    return generator.GetFingerprint(stripped)


def variable_similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0
    try:
        from rdkit import DataStructs
    except ImportError as exc:  # pragma: no cover - cluster dependency
        raise RuntimeError("RetrievedDeltaEdit requires RDKit DataStructs") from exc
    left_fp = variable_fingerprint(left)
    right_fp = variable_fingerprint(right)
    if left_fp is None or right_fp is None:
        return 0.0
    return float(DataStructs.TanimotoSimilarity(left_fp, right_fp))


def join_fragments(core: str, variable: str) -> str:
    try:
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover - cluster dependency
        raise RuntimeError("RetrievedDeltaEdit requires RDKit") from exc
    core_mol = Chem.MolFromSmiles(str(core or ""))
    variable_mol = Chem.MolFromSmiles(str(variable or ""))
    if core_mol is None or variable_mol is None:
        return ""
    combined = Chem.RWMol(Chem.CombineMols(core_mol, variable_mol))
    dummy_indices: list[int] = []
    neighbor_indices: list[int] = []
    bond_types = []
    for atom in combined.GetAtoms():
        if atom.GetAtomicNum() != 0:
            continue
        neighbors = list(atom.GetNeighbors())
        if len(neighbors) != 1:
            return ""
        dummy_indices.append(atom.GetIdx())
        neighbor_indices.append(neighbors[0].GetIdx())
        bond = combined.GetBondBetweenAtoms(atom.GetIdx(), neighbors[0].GetIdx())
        bond_types.append(bond.GetBondType() if bond is not None else Chem.BondType.SINGLE)
    if len(dummy_indices) != 2:
        return ""
    if combined.GetBondBetweenAtoms(neighbor_indices[0], neighbor_indices[1]) is not None:
        return ""
    try:
        bond_type = bond_types[0] if bond_types[0] == bond_types[1] else Chem.BondType.SINGLE
        combined.AddBond(neighbor_indices[0], neighbor_indices[1], bond_type)
        for index in sorted(dummy_indices, reverse=True):
            combined.RemoveAtom(int(index))
        product = combined.GetMol()
        Chem.SanitizeMol(product)
        return Chem.MolToSmiles(product, canonical=True)
    except Exception:
        return ""


def candidate_for_smiles(
    row: Mapping[str, object],
    smiles: str,
    *,
    source: str,
    **kwargs: object,
) -> Candidate | None:
    canonical = canonical_smiles(smiles)
    source_smiles = str(row.get("source_smiles", "") or "").strip()
    if not canonical or not source_smiles:
        return None
    source_tanimoto = graph.revise.morgan_tanimoto(source_smiles, canonical)
    if not math.isfinite(source_tanimoto):
        return None
    return Candidate(
        smiles=canonical,
        source=source,
        source_tanimoto=float(source_tanimoto),
        admet_prior_score=float(graph.admet_prior_score(row, canonical)),
        **kwargs,
    )


def retrieved_candidates(
    row: Mapping[str, object],
    transforms: Sequence[DeltaTransform],
    *,
    min_retrieval_similarity: float,
    max_transforms_per_query: int,
    min_core_heavy_atoms: int,
    max_variable_heavy_atoms: int,
) -> tuple[list[Candidate], dict[str, object]]:
    source = str(row.get("source_smiles", "") or "").strip()
    candidates: dict[str, Candidate] = {}
    exact_query_matches = 0
    approximate_query_matches = 0
    for split in fragment_splits(source, min_core_heavy_atoms, max_variable_heavy_atoms):
        ranked = []
        for transform in transforms:
            similarity = variable_similarity(split.variable, transform.source_variable)
            if similarity < float(min_retrieval_similarity):
                continue
            ranked.append((similarity, transform.frequency, transform))
        ranked.sort(
            key=lambda item: (
                item[0],
                item[1],
                item[2].source_variable,
                item[2].target_variable,
            ),
            reverse=True,
        )
        for similarity, _frequency, transform in ranked[: max(1, int(max_transforms_per_query))]:
            generated = join_fragments(split.core, transform.target_variable)
            candidate = candidate_for_smiles(
                row,
                generated,
                source="retrieved_delta_edit",
                retrieval_similarity=float(similarity),
                transform_frequency=int(transform.frequency),
                exact_variable_match=bool(split.variable == transform.source_variable),
                query_core=split.core,
                query_variable=split.variable,
                source_variable=transform.source_variable,
                target_variable=transform.target_variable,
                train_condition_id=transform.train_condition_id,
            )
            if candidate is None or candidate.smiles == canonical_smiles(source):
                continue
            previous = candidates.get(candidate.smiles)
            if previous is None or candidate.selection_score > previous.selection_score:
                candidates[candidate.smiles] = candidate
            exact_query_matches += int(split.variable == transform.source_variable)
            approximate_query_matches += int(split.variable != transform.source_variable)
    return list(candidates.values()), {
        "exact_query_transform_matches": exact_query_matches,
        "approximate_query_transform_matches": approximate_query_matches,
    }


def fallback_candidates(
    row: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
) -> list[Candidate]:
    output = []
    for rank, fallback in enumerate(rows, start=1):
        candidate = candidate_for_smiles(
            row,
            str(fallback.get("generated_smiles", "") or ""),
            source="v4_graph_fallback",
            fallback_rank=rank,
        )
        if candidate is not None:
            output.append(candidate)
    return output


def rank_candidates(
    candidates: Sequence[Candidate],
    *,
    min_source_tanimoto: float,
) -> list[Candidate]:
    best_by_smiles: dict[str, Candidate] = {}
    for candidate in candidates:
        previous = best_by_smiles.get(candidate.smiles)
        if previous is None or candidate.selection_score > previous.selection_score:
            best_by_smiles[candidate.smiles] = candidate
    ranked = sorted(
        best_by_smiles.values(),
        key=lambda item: (
            item.source_tanimoto >= float(min_source_tanimoto),
            item.selection_score,
            item.admet_prior_score,
            item.retrieval_similarity,
            item.source_tanimoto,
            item.source == "retrieved_delta_edit",
            -item.fallback_rank,
            item.smiles,
        ),
        reverse=True,
    )
    return ranked


def select_candidates(
    candidates: Sequence[Candidate],
    *,
    candidate_budget: int,
    min_source_tanimoto: float,
) -> list[Candidate]:
    selected = rank_candidates(
        candidates,
        min_source_tanimoto=float(min_source_tanimoto),
    )[: max(1, int(candidate_budget))]
    if len(selected) != int(candidate_budget):
        raise ValueError(
            f"Condition has only {len(selected)} unique candidates; fixed n={candidate_budget} is required"
        )
    return selected


def output_row(
    row: Mapping[str, object],
    candidate: Candidate,
    *,
    rank: int,
) -> dict[str, object]:
    output = dict(row)
    output.update(
        {
            "generated_smiles": candidate.smiles,
            "method": "retrieved_delta_edit_support",
            "candidate_rank": int(rank),
            "candidate_selected": str(int(rank) == 1),
            "graph_edit_candidate_source": candidate.source,
            "delta_source_tanimoto": round(candidate.source_tanimoto, 6),
            "delta_admet_prior_score": round(candidate.admet_prior_score, 6),
            "delta_retrieval_similarity": round(candidate.retrieval_similarity, 6),
            "delta_transform_frequency": int(candidate.transform_frequency),
            "delta_exact_variable_match": str(candidate.exact_variable_match),
            "delta_query_core": candidate.query_core,
            "delta_query_variable": candidate.query_variable,
            "delta_source_variable": candidate.source_variable,
            "delta_target_variable": candidate.target_variable,
            "delta_train_condition_id": candidate.train_condition_id,
            "delta_fallback_rank": int(candidate.fallback_rank),
            "delta_selection_score": round(candidate.selection_score, 6),
        }
    )
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.candidate_budget) != 20:
        raise ValueError("The paper-facing RetrievedDeltaEdit support gate fixes candidate_budget=20")
    train_rows = read_rows(args.train_csv)
    eval_rows = read_rows(args.eval_csv)
    fallback_by_condition: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_rows(args.fallback_candidates_csv):
        fallback_by_condition[row_key(row)].append(row)
    transform_index, transform_manifest = build_transform_index(
        train_rows,
        min_core_heavy_atoms=int(args.min_core_heavy_atoms),
        max_variable_heavy_atoms=int(args.max_variable_heavy_atoms),
    )
    output: list[dict[str, object]] = []
    enumerated_output: list[dict[str, object]] = []
    source_counts: Counter[str] = Counter()
    conditions_with_retrieved = 0
    conditions_with_exact = 0
    retrieved_candidate_counts = []
    match_totals: Counter[str] = Counter()
    for index, row in enumerate(eval_rows, start=1):
        key = row_key(row)
        retrieved, match_summary = retrieved_candidates(
            row,
            transform_index.get(task_key(row), []),
            min_retrieval_similarity=float(args.min_retrieval_similarity),
            max_transforms_per_query=int(args.max_transforms_per_query),
            min_core_heavy_atoms=int(args.min_core_heavy_atoms),
            max_variable_heavy_atoms=int(args.max_variable_heavy_atoms),
        )
        fallback = fallback_candidates(row, fallback_by_condition.get(key, []))
        ranked = rank_candidates(
            [*retrieved, *fallback],
            min_source_tanimoto=float(args.min_source_tanimoto),
        )
        selected = select_candidates(
            ranked,
            candidate_budget=int(args.candidate_budget),
            min_source_tanimoto=float(args.min_source_tanimoto),
        )
        output.extend(output_row(row, candidate, rank=rank) for rank, candidate in enumerate(selected, start=1))
        if args.enumerated_output_csv is not None:
            enumerated_output.extend(
                output_row(row, candidate, rank=rank)
                for rank, candidate in enumerate(ranked, start=1)
            )
        source_counts.update(candidate.source for candidate in selected)
        retrieved_candidate_counts.append(len(retrieved))
        conditions_with_retrieved += int(bool(retrieved))
        conditions_with_exact += int(any(candidate.exact_variable_match for candidate in retrieved))
        match_totals.update({name: int(value) for name, value in match_summary.items()})
        if index % 10 == 0 or index == len(eval_rows):
            print(f"[retrieved-delta] {index}/{len(eval_rows)} conditions", flush=True)
    write_rows(args.output_csv, output)
    if args.enumerated_output_csv is not None:
        write_rows(args.enumerated_output_csv, enumerated_output)
    manifest = {
        "protocol": "retrieved_delta_edit_candidate_builder_v1",
        "data_role": "train_only_to_disjoint_train_audit",
        "evaluation_target_access": False,
        "candidate_budget": int(args.candidate_budget),
        "output_rows": len(output),
        "enumerated_output_rows": len(enumerated_output) if args.enumerated_output_csv is not None else None,
        "enumerated_output_csv": str(args.enumerated_output_csv) if args.enumerated_output_csv else None,
        "evaluation_conditions": len(eval_rows),
        "conditions_with_retrieved_candidate": conditions_with_retrieved,
        "conditions_with_exact_source_variable_match": conditions_with_exact,
        "mean_retrieved_candidates": (
            sum(retrieved_candidate_counts) / max(len(retrieved_candidate_counts), 1)
        ),
        "candidate_source_counts": dict(sorted(source_counts.items())),
        "match_totals": dict(sorted(match_totals.items())),
        "retrieval": {
            "min_similarity": float(args.min_retrieval_similarity),
            "max_transforms_per_query": int(args.max_transforms_per_query),
            "min_core_heavy_atoms": int(args.min_core_heavy_atoms),
            "max_variable_heavy_atoms": int(args.max_variable_heavy_atoms),
            "min_source_tanimoto": float(args.min_source_tanimoto),
        },
        "transform_index": transform_manifest,
    }
    args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
