"""Automatic invalid-to-valid molecular repair dataset construction."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from .chem import canonicalize_smiles, molecular_descriptors
from .schema import GenerationRequest, TaskType


DEFAULT_CORRUPTION_TYPES = (
    "token_deletion",
    "truncation",
    "branch_parenthesis",
    "ring_closure",
    "aromaticity",
    "atom_replacement",
    "bond_corruption",
    "ocr_confusion",
)


@dataclass(frozen=True)
class RepairExample:
    clean_smiles: str
    corrupted_smiles: str
    corruption_type: str
    instruction_text: str
    instruction_spec_json: str
    split: str
    corrupted_valid: bool

    def to_request_pair(self) -> tuple[GenerationRequest, str]:
        request = GenerationRequest(
            task_type=TaskType.REPAIR,
            source_smiles=self.corrupted_smiles,
            instruction=self.instruction_text,
            metadata={
                "clean_smiles": self.clean_smiles,
                "corruption_type": self.corruption_type,
                "split": self.split,
            },
        )
        return request, self.clean_smiles


def build_repair_examples(
    smiles: list[str],
    corruption_types: str | Iterable[str] | None = None,
    corruptions_per_molecule: int = 2,
    seed: int = 7,
) -> list[RepairExample]:
    """Create deterministic repair examples from valid molecules.

    The source is a corrupted molecular string and the target is the original
    clean canonical SMILES. The corrupted string is allowed to be partial or
    still valid for a subset of operators; the verifier records that fact.
    """

    types = _parse_corruption_types(corruption_types)
    rng = np.random.default_rng(seed)
    examples: list[RepairExample] = []
    for mol_idx, raw_smiles in enumerate(smiles):
        clean = canonicalize_smiles(raw_smiles)
        if not clean:
            continue
        chosen = _choose_types(types, corruptions_per_molecule, rng)
        for type_idx, corruption_type in enumerate(chosen):
            corrupted = corrupt_smiles(clean, corruption_type, rng)
            if not corrupted or corrupted == clean:
                continue
            corrupted_valid = molecular_descriptors(corrupted).valid
            examples.append(
                RepairExample(
                    clean_smiles=clean,
                    corrupted_smiles=corrupted,
                    corruption_type=corruption_type,
                    instruction_text=_instruction_for(corruption_type),
                    instruction_spec_json=json.dumps(_spec_for(corruption_type), sort_keys=True),
                    split=_split_for(mol_idx, type_idx),
                    corrupted_valid=corrupted_valid,
                )
            )
    return examples


def build_repair_requests(
    smiles: list[str],
    corruption_types: str | Iterable[str] | None = None,
    corruptions_per_molecule: int = 2,
    seed: int = 7,
    split: str | None = None,
) -> list[tuple[GenerationRequest, str]]:
    examples = build_repair_examples(
        smiles,
        corruption_types=corruption_types,
        corruptions_per_molecule=corruptions_per_molecule,
        seed=seed,
    )
    if split:
        examples = [example for example in examples if example.split == split]
    return [example.to_request_pair() for example in examples]


def corrupt_smiles(clean_smiles: str, corruption_type: str, rng: np.random.Generator) -> str:
    smiles = str(clean_smiles)
    if corruption_type == "token_deletion":
        return _delete_char(smiles, rng)
    if corruption_type == "truncation":
        if len(smiles) <= 2:
            return _delete_char(smiles, rng)
        keep = int(rng.integers(max(1, len(smiles) // 2), len(smiles)))
        return smiles[:keep]
    if corruption_type == "branch_parenthesis":
        if "(" in smiles or ")" in smiles:
            return _remove_first_of(smiles, ["(", ")"])
        return smiles + ")"
    if corruption_type == "ring_closure":
        digits = [ch for ch in smiles if ch.isdigit()]
        if digits:
            return smiles.replace(digits[0], "", 1)
        return smiles + "1"
    if corruption_type == "aromaticity":
        for old, new in (("c", "C"), ("n", "N"), ("o", "O"), ("s", "S")):
            if old in smiles:
                return smiles.replace(old, new, 1)
        return smiles.lower()
    if corruption_type == "atom_replacement":
        return _replace_atom_like_token(smiles, rng)
    if corruption_type == "bond_corruption":
        if "=" in smiles:
            return smiles.replace("=", "#", 1)
        if "#" in smiles:
            return smiles.replace("#", "=", 1)
        return smiles[:1] + "#" + smiles[1:]
    if corruption_type == "ocr_confusion":
        return _ocr_confuse(smiles)
    return _delete_char(smiles, rng)


def examples_to_rows(examples: list[RepairExample]) -> list[dict[str, object]]:
    return [asdict(example) for example in examples]


def _parse_corruption_types(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return list(DEFAULT_CORRUPTION_TYPES)
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        return parts or list(DEFAULT_CORRUPTION_TYPES)
    return [str(part).strip() for part in value if str(part).strip()] or list(DEFAULT_CORRUPTION_TYPES)


def _choose_types(types: list[str], count: int, rng: np.random.Generator) -> list[str]:
    count = max(1, int(count))
    if count >= len(types):
        return list(types)
    indices = rng.choice(len(types), size=count, replace=False)
    return [types[int(idx)] for idx in sorted(indices)]


def _delete_char(smiles: str, rng: np.random.Generator) -> str:
    if len(smiles) <= 1:
        return smiles + ")"
    idx = int(rng.integers(0, len(smiles)))
    return smiles[:idx] + smiles[idx + 1 :]


def _remove_first_of(smiles: str, tokens: list[str]) -> str:
    positions = [(smiles.find(token), token) for token in tokens if smiles.find(token) >= 0]
    if not positions:
        return smiles
    _, token = min(positions)
    return smiles.replace(token, "", 1)


def _replace_atom_like_token(smiles: str, rng: np.random.Generator) -> str:
    replacements = ("Zz", "*", "Q", "Xx")
    for token in ("Cl", "Br", "C", "N", "O", "S", "F", "I", "c", "n", "o", "s"):
        idx = smiles.find(token)
        if idx >= 0:
            repl = replacements[int(rng.integers(0, len(replacements)))]
            return smiles[:idx] + repl + smiles[idx + len(token) :]
    return smiles + "Zz"


def _ocr_confuse(smiles: str) -> str:
    replacements = (
        ("Cl", "CI"),
        ("Br", "B8"),
        ("O", "0"),
        ("S", "5"),
        ("1", "l"),
        ("0", "O"),
        ("c", "e"),
    )
    for old, new in replacements:
        if old in smiles:
            return smiles.replace(old, new, 1)
    return smiles + "?"


def _instruction_for(corruption_type: str) -> str:
    if corruption_type in {"truncation", "token_deletion"}:
        return "Repair this partial molecular string into a valid molecule while preserving the intended structure."
    if corruption_type == "ocr_confusion":
        return "Fix the OCR-corrupted molecular string and recover a valid plausible molecule."
    return "Repair this corrupted or invalid molecular string into a valid molecule while preserving the original scaffold."


def _spec_for(corruption_type: str) -> dict[str, object]:
    task = "repair_partial_smiles" if corruption_type in {"truncation", "token_deletion"} else "repair_invalid_smiles"
    if corruption_type == "ocr_confusion":
        task = "repair_ocr_like_smiles"
    return {
        "task": task,
        "corruption_type": corruption_type,
        "goals": ["make_valid"],
        "constraints": ["recover_scaffold", "keep_properties_close"],
        "thresholds": {"similarity_min": 0.60, "property_mae_max": 0.20},
    }


def _split_for(molecule_idx: int, corruption_idx: int) -> str:
    bucket = (molecule_idx * 17 + corruption_idx) % 10
    if bucket == 0:
        return "test"
    if bucket == 1:
        return "val"
    return "train"
