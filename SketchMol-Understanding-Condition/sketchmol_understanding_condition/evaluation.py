"""Evaluation helpers for understanding-conditioned molecular editing."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable, Optional

from .chem import is_valid_smiles, morgan_tanimoto, scaffold_smiles


@dataclass(frozen=True)
class EditExampleResult:
    """Generated output for one source-target edit example."""

    source_smiles: str
    generated_smiles: str
    target_smiles: Optional[str] = None
    source_property: Optional[float] = None
    generated_property: Optional[float] = None


def validity_rate(smiles_list: Iterable[str]) -> float:
    """Fraction of valid generated SMILES."""

    smiles = list(smiles_list)
    if not smiles:
        return 0.0
    return sum(1 for item in smiles if is_valid_smiles(item)) / len(smiles)


def scaffold_preservation_rate(results: Iterable[EditExampleResult]) -> float:
    """Fraction of generations that preserve the source Bemis-Murcko scaffold."""

    examples = list(results)
    if not examples:
        return 0.0

    hits = 0
    for example in examples:
        source_scaffold = scaffold_smiles(example.source_smiles)
        generated_scaffold = scaffold_smiles(example.generated_smiles)
        if source_scaffold is not None and source_scaffold == generated_scaffold:
            hits += 1
    return hits / len(examples)


def mean_source_generated_similarity(results: Iterable[EditExampleResult]) -> float:
    """Mean source/generated Tanimoto similarity over valid pairs."""

    values = []
    for example in results:
        similarity = morgan_tanimoto(example.source_smiles, example.generated_smiles)
        if similarity is not None:
            values.append(similarity)
    return mean(values) if values else 0.0


def property_success_rate(
    results: Iterable[EditExampleResult],
    *,
    direction: str,
    min_delta: float = 0.0,
) -> float:
    """Fraction of examples improving a scalar property in the desired direction."""

    examples = [
        item
        for item in results
        if item.source_property is not None and item.generated_property is not None
    ]
    if not examples:
        return 0.0

    hits = 0
    for example in examples:
        delta = float(example.generated_property) - float(example.source_property)
        if direction == "increase" and delta >= min_delta:
            hits += 1
        elif direction == "decrease" and -delta >= min_delta:
            hits += 1
    return hits / len(examples)
