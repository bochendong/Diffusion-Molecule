"""Tiny SketchMol-aligned benchmark fixtures and CSV loading."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Iterable

from .schema import BenchmarkExample, TaskType


def toy_examples() -> list[BenchmarkExample]:
    return [
        BenchmarkExample(
            task_id="denovo_qed_aspirin_like",
            task_type=TaskType.DE_NOVO,
            instruction="Generate a small drug-like aromatic acid with balanced QED.",
            target_smiles="CC(=O)Oc1ccccc1C(=O)O",
            goals=("druglike", "aromatic", "acid"),
        ),
        BenchmarkExample(
            task_id="edit_lower_logp",
            task_type=TaskType.EDIT,
            source_smiles="CCOc1ccccc1",
            target_smiles="CCOc1ccc(O)cc1",
            instruction="Lower LogP while keeping the aromatic core.",
            goals=("lower_logp", "keep_aromatic_core"),
        ),
        BenchmarkExample(
            task_id="edit_raise_polarity",
            task_type=TaskType.EDIT,
            source_smiles="CCN(CC)CC",
            target_smiles="CCN(CCO)CCO",
            instruction="Increase polarity and hydrogen bonding without making the molecule too large.",
            goals=("increase_tpsa", "increase_hbd"),
        ),
        BenchmarkExample(
            task_id="inpaint_keep_benzene",
            task_type=TaskType.INPAINT,
            source_smiles="c1ccccc1",
            target_smiles="Nc1ccccc1O",
            mask_hint="keep benzene ring; fill two substituent positions",
            instruction="Fill the masked positions with polar substituents while preserving the benzene ring.",
            goals=("preserve_scaffold", "polar_substituents"),
        ),
        BenchmarkExample(
            task_id="fragment_grow_indole",
            task_type=TaskType.FRAGMENT_GROW,
            source_smiles="c1ccccc1N",
            target_smiles="c1ccc2[nH]ccc2c1",
            mask_hint="grow fused heteroaromatic ring from an aniline-like fragment",
            instruction="Grow the fragment into a compact heteroaromatic scaffold.",
            goals=("fragment_growth", "heteroaromatic"),
        ),
        BenchmarkExample(
            task_id="denovo_low_tpsa",
            task_type=TaskType.DE_NOVO,
            instruction="Generate a compact low-TPSA molecule with moderate hydrophobicity.",
            target_smiles="CC(C)c1ccccc1",
            goals=("low_tpsa", "moderate_logp"),
        ),
    ]


def load_examples_csv(path: str | Path) -> list[BenchmarkExample]:
    path = Path(path)
    examples: list[BenchmarkExample] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            goals = tuple(item.strip() for item in row.get("goals", "").split(";") if item.strip())
            examples.append(
                BenchmarkExample(
                    task_id=row["task_id"],
                    task_type=TaskType(row["task_type"]),
                    source_smiles=row.get("source_smiles") or None,
                    target_smiles=row["target_smiles"],
                    instruction=row.get("instruction", ""),
                    mask_hint=row.get("mask_hint") or None,
                    image_path=row.get("image_path") or None,
                    goals=goals,
                )
            )
    return examples


def split_examples(examples: Iterable[BenchmarkExample], train_fraction: float = 0.75, seed: int = 7) -> tuple[list[BenchmarkExample], list[BenchmarkExample]]:
    examples = list(examples)
    if not examples:
        return [], []
    if len(examples) == 1:
        return examples, examples
    train_fraction = min(0.95, max(0.05, float(train_fraction)))
    indices = list(range(len(examples)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    train_size = int(round(len(indices) * train_fraction))
    train_size = min(max(1, train_size), len(indices) - 1)
    train_idx = set(indices[:train_size])
    train = [example for idx, example in enumerate(examples) if idx in train_idx]
    eval_examples = [example for idx, example in enumerate(examples) if idx not in train_idx]
    return train, eval_examples


def write_examples_csv(path: str | Path, examples: Iterable[BenchmarkExample]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["task_id", "task_type", "source_smiles", "target_smiles", "instruction", "mask_hint", "image_path", "goals"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for example in examples:
            writer.writerow(
                {
                    "task_id": example.task_id,
                    "task_type": example.task_type.value,
                    "source_smiles": example.source_smiles or "",
                    "target_smiles": example.target_smiles,
                    "instruction": example.instruction,
                    "mask_hint": example.mask_hint or "",
                    "image_path": example.image_path or "",
                    "goals": ";".join(example.goals),
                }
            )
