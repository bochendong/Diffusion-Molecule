"""Build SketchImage-JEPA task CSVs from ordinary molecule CSV files."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chem import DESCRIPTOR_KEYS, canonicalize_smiles, molecular_descriptors, scaffold_key, tanimoto
from .dataset import write_examples_csv
from .schema import BenchmarkExample, TaskType


DEFAULT_TASK_TYPES = ("de_novo", "edit", "inpaint", "fragment_grow")
_DELTA_SCALE = {
    "MW": 120.0,
    "LogP": 2.0,
    "QED": 0.25,
    "TPSA": 60.0,
    "HBD": 2.0,
    "HBA": 3.0,
    "RB": 3.0,
}


@dataclass(frozen=True)
class MoleculeRow:
    idx: int
    smiles: str
    descriptors: dict[str, float]
    scaffold: str


def load_molecule_rows(path: str | Path, smiles_column: str | None = None, limit: int | None = None) -> list[MoleculeRow]:
    path = Path(path)
    rows: list[MoleculeRow] = []
    seen: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header.")
        column = smiles_column or _detect_smiles_column(reader.fieldnames)
        for raw in reader:
            smiles = raw.get(column, "")
            can = canonicalize_smiles(smiles)
            if not can or can in seen:
                continue
            rec = molecular_descriptors(can)
            if not rec.valid:
                continue
            seen.add(rec.smiles)
            rows.append(MoleculeRow(idx=len(rows), smiles=rec.smiles, descriptors=rec.descriptors, scaffold=scaffold_key(rec.smiles)))
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"No valid molecules found in {path}. Expected a SMILES column.")
    return rows


def build_tasks_from_molecules(
    molecules: list[MoleculeRow],
    task_types: tuple[str, ...] = DEFAULT_TASK_TYPES,
    max_tasks: int = 5000,
    pairs_per_source: int = 2,
    pair_candidates: int = 128,
    min_similarity: float = 0.15,
    max_similarity: float = 0.90,
    seed: int = 7,
) -> list[BenchmarkExample]:
    rng = np.random.default_rng(seed)
    task_set = set(task_types)
    budgets = _task_budgets(task_types, max_tasks)
    counts: Counter[str] = Counter()
    examples: list[BenchmarkExample] = []

    if "de_novo" in task_set:
        for mol in molecules:
            if counts["de_novo"] >= budgets.get("de_novo", 0):
                break
            examples.append(_de_novo_task(mol))
            counts["de_novo"] += 1

    pair_budget = max(0, max_tasks - len(examples))
    if pair_budget == 0 or len(molecules) < 2:
        return examples[:max_tasks]

    source_order = rng.permutation(len(molecules)).tolist()
    for source_idx in source_order:
        source = molecules[source_idx]
        pairs = _pick_pairs(source, molecules, rng, pairs_per_source, pair_candidates, min_similarity, max_similarity)
        for target, sim in pairs:
            if "edit" in task_set and counts["edit"] < budgets.get("edit", 0):
                examples.append(_edit_task(source, target, sim))
                counts["edit"] += 1
            if len(examples) >= max_tasks:
                return examples
            if "inpaint" in task_set and counts["inpaint"] < budgets.get("inpaint", 0):
                examples.append(_inpaint_task(source, target, sim))
                counts["inpaint"] += 1
            if len(examples) >= max_tasks:
                return examples
            if "fragment_grow" in task_set and counts["fragment_grow"] < budgets.get("fragment_grow", 0) and _looks_like_growth(source, target):
                examples.append(_fragment_grow_task(source, target, sim))
                counts["fragment_grow"] += 1
            if len(examples) >= max_tasks:
                return examples
        if len(examples) >= max_tasks:
            return examples
    return examples[:max_tasks]


def summarize_tasks(examples: list[BenchmarkExample], molecules: list[MoleculeRow], args: argparse.Namespace) -> dict[str, object]:
    counts = Counter(example.task_type.value for example in examples)
    return {
        "molecules": len(molecules),
        "tasks": len(examples),
        "task_counts": dict(sorted(counts.items())),
        "args": {
            "molecule_csv": str(args.molecule_csv),
            "out": str(args.out),
            "smiles_column": args.smiles_column,
            "limit": args.limit,
            "max_tasks": args.max_tasks,
            "pairs_per_source": args.pairs_per_source,
            "pair_candidates": args.pair_candidates,
            "min_similarity": args.min_similarity,
            "max_similarity": args.max_similarity,
            "seed": args.seed,
            "task_types": args.task_types,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SketchImage-JEPA task CSV from a molecule SMILES CSV.")
    parser.add_argument("--molecule-csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--smiles-column", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-tasks", type=int, default=5000)
    parser.add_argument("--pairs-per-source", type=int, default=2)
    parser.add_argument("--pair-candidates", type=int, default=128)
    parser.add_argument("--min-similarity", type=float, default=0.15)
    parser.add_argument("--max-similarity", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--task-types", default=",".join(DEFAULT_TASK_TYPES))
    args = parser.parse_args()

    task_types = tuple(item.strip() for item in args.task_types.split(",") if item.strip())
    molecules = load_molecule_rows(args.molecule_csv, smiles_column=args.smiles_column, limit=args.limit)
    examples = build_tasks_from_molecules(
        molecules,
        task_types=task_types,
        max_tasks=args.max_tasks,
        pairs_per_source=args.pairs_per_source,
        pair_candidates=args.pair_candidates,
        min_similarity=args.min_similarity,
        max_similarity=args.max_similarity,
        seed=args.seed,
    )
    write_examples_csv(args.out, examples)
    summary = summarize_tasks(examples, molecules, args)
    summary_path = Path(args.out).with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def _detect_smiles_column(fieldnames: list[str]) -> str:
    candidates = ["smiles", "SMILES", "canonical_smiles", "CanonicalSMILES", "mol_smiles", "molecule_smiles"]
    for candidate in candidates:
        if candidate in fieldnames:
            return candidate
    lowered = {name.lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    raise ValueError(f"Could not detect a SMILES column. Available columns: {fieldnames}")


def _task_budgets(task_types: tuple[str, ...], max_tasks: int) -> dict[str, int]:
    clean = [task for task in task_types if task in DEFAULT_TASK_TYPES]
    if not clean:
        raise ValueError(f"No supported task types requested: {task_types}")
    base = max_tasks // len(clean)
    remainder = max_tasks % len(clean)
    return {task: base + (1 if idx < remainder else 0) for idx, task in enumerate(clean)}


def _pick_pairs(
    source: MoleculeRow,
    molecules: list[MoleculeRow],
    rng: np.random.Generator,
    pairs_per_source: int,
    pair_candidates: int,
    min_similarity: float,
    max_similarity: float,
) -> list[tuple[MoleculeRow, float]]:
    if pairs_per_source <= 0:
        return []
    n = len(molecules)
    sample_size = min(max(pair_candidates, pairs_per_source * 8), n - 1)
    pool = [idx for idx in rng.choice(n, size=min(n, sample_size + 1), replace=False).tolist() if idx != source.idx]
    if len(pool) < sample_size:
        extras = rng.permutation(n).tolist()
        pool.extend(idx for idx in extras if idx != source.idx and idx not in pool)
        pool = pool[:sample_size]
    scored: list[tuple[float, MoleculeRow, float]] = []
    for idx in pool:
        target = molecules[idx]
        sim = tanimoto(source.smiles, target.smiles)
        if sim < min_similarity or sim > max_similarity:
            continue
        delta_score, _ = _dominant_delta(source, target)
        same_scaffold_bonus = 0.15 if source.scaffold and source.scaffold == target.scaffold else 0.0
        scored.append((delta_score + same_scaffold_bonus + 0.05 * sim, target, sim))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [(target, sim) for _, target, sim in scored[:pairs_per_source]]


def _de_novo_task(target: MoleculeRow) -> BenchmarkExample:
    desc = target.descriptors
    instruction = (
        f"Generate a molecule with MW around {desc.get('MW', 0.0):.0f}, "
        f"LogP around {desc.get('LogP', 0.0):.1f}, QED around {desc.get('QED', 0.0):.2f}, "
        f"and TPSA around {desc.get('TPSA', 0.0):.0f}."
    )
    return BenchmarkExample(
        task_id=f"denovo_{target.idx}",
        task_type=TaskType.DE_NOVO,
        target_smiles=target.smiles,
        instruction=instruction,
        goals=("de_novo", "property_conditioned", *_property_goal_tags(target)),
    )


def _edit_task(source: MoleculeRow, target: MoleculeRow, sim: float) -> BenchmarkExample:
    _, delta_prop = _dominant_delta(source, target)
    instruction = _delta_instruction(delta_prop, source, target, prefix="Edit the source molecule to")
    if source.scaffold and source.scaffold == target.scaffold:
        instruction += " Preserve the molecular scaffold."
    else:
        instruction += " Keep the molecule structurally related to the source."
    return BenchmarkExample(
        task_id=f"edit_{source.idx}_{target.idx}",
        task_type=TaskType.EDIT,
        source_smiles=source.smiles,
        target_smiles=target.smiles,
        instruction=instruction,
        goals=("edit", f"similarity_{sim:.2f}", *_delta_goal_tags(source, target)),
    )


def _inpaint_task(source: MoleculeRow, target: MoleculeRow, sim: float) -> BenchmarkExample:
    _, delta_prop = _dominant_delta(source, target)
    instruction = _delta_instruction(delta_prop, source, target, prefix="Fill the masked region to")
    instruction += " Preserve the visible source core."
    scaffold = source.scaffold or "source core"
    return BenchmarkExample(
        task_id=f"inpaint_{source.idx}_{target.idx}",
        task_type=TaskType.INPAINT,
        source_smiles=source.smiles,
        target_smiles=target.smiles,
        instruction=instruction,
        mask_hint=f"preserve {scaffold}; mask substituent or growth region",
        goals=("inpaint", "preserve_core", f"similarity_{sim:.2f}", *_delta_goal_tags(source, target)),
    )


def _fragment_grow_task(source: MoleculeRow, target: MoleculeRow, sim: float) -> BenchmarkExample:
    _, delta_prop = _dominant_delta(source, target)
    instruction = _delta_instruction(delta_prop, source, target, prefix="Grow the source fragment to")
    instruction += " Keep the original fragment recognizable."
    return BenchmarkExample(
        task_id=f"fragment_grow_{source.idx}_{target.idx}",
        task_type=TaskType.FRAGMENT_GROW,
        source_smiles=source.smiles,
        target_smiles=target.smiles,
        instruction=instruction,
        mask_hint="grow from source fragment; preserve existing atoms as the visible region",
        goals=("fragment_grow", f"similarity_{sim:.2f}", *_delta_goal_tags(source, target)),
    )


def _dominant_delta(source: MoleculeRow, target: MoleculeRow) -> tuple[float, str]:
    best_prop = "QED"
    best_score = 0.0
    for prop in DESCRIPTOR_KEYS:
        delta = float(target.descriptors.get(prop, 0.0) - source.descriptors.get(prop, 0.0))
        score = abs(delta) / _DELTA_SCALE.get(prop, 1.0)
        if score > best_score:
            best_prop = prop
            best_score = score
    return best_score, best_prop


def _delta_instruction(prop: str, source: MoleculeRow, target: MoleculeRow, prefix: str) -> str:
    delta = float(target.descriptors.get(prop, 0.0) - source.descriptors.get(prop, 0.0))
    direction = "increase" if delta >= 0 else "decrease"
    target_value = target.descriptors.get(prop, 0.0)
    return f"{prefix} {direction} {prop} toward {target_value:.2f}."


def _delta_goal_tags(source: MoleculeRow, target: MoleculeRow) -> tuple[str, ...]:
    _, prop = _dominant_delta(source, target)
    delta = float(target.descriptors.get(prop, 0.0) - source.descriptors.get(prop, 0.0))
    direction = "increase" if delta >= 0 else "decrease"
    return (f"{direction}_{prop}",)


def _property_goal_tags(target: MoleculeRow) -> tuple[str, ...]:
    desc = target.descriptors
    tags = []
    if desc.get("MW", 999.0) <= 500.0:
        tags.append("mw_le_500")
    if desc.get("QED", 0.0) >= 0.5:
        tags.append("qed_ge_0.5")
    if desc.get("TPSA", 999.0) <= 90.0:
        tags.append("tpsa_le_90")
    return tuple(tags)


def _looks_like_growth(source: MoleculeRow, target: MoleculeRow) -> bool:
    mw_delta = float(target.descriptors.get("MW", 0.0) - source.descriptors.get("MW", 0.0))
    return mw_delta >= 20.0 or len(target.smiles) > len(source.smiles) + 3


if __name__ == "__main__":
    main()
