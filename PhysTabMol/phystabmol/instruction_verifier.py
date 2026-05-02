"""Deterministic chemistry verifier for instruction-guided editing."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from . import chem as chem_mod
from .chem import molecular_descriptors, passes_druglike_filters, tanimoto
from .instruction_schema import CONSTRAINTS, EDIT_RULES, PROPERTY_GOALS, normalize_spec, threshold
from .schema import TARGET_COLUMNS


def verify_instruction(source_smiles: str, candidate_smiles: str, spec: str | dict[str, Any]) -> dict[str, Any]:
    """Verify a candidate molecule against a structured edit instruction.

    Returns group-level booleans plus detailed failed tags. Invalid source or
    candidate molecules are treated as failed candidates.
    """

    normalized = normalize_spec(spec)
    source = molecular_descriptors(source_smiles)
    candidate = molecular_descriptors(candidate_smiles)
    if not source.valid or not candidate.valid:
        return {
            "valid": False,
            "source_valid": bool(source.valid),
            "candidate_valid": bool(candidate.valid),
            "goal_success": False,
            "constraint_success": False,
            "edit_success": False,
            "overall_success": False,
            "similarity_to_source": 0.0,
            "druglike": False,
            "failed_goals": "|".join(normalized["goals"]),
            "failed_constraints": "|".join(normalized["constraints"]),
            "failed_edits": "|".join(normalized["edits"]),
            "goal_results_json": "{}",
            "constraint_results_json": "{}",
            "edit_results_json": "{}",
            "property_delta_json": "{}",
        }

    sim = tanimoto(source.smiles, candidate.smiles)
    deltas = property_delta(source.descriptors, candidate.descriptors)
    goal_results = {goal: _check_property_goal(goal, source.descriptors, candidate.descriptors, normalized) for goal in normalized["goals"]}
    constraint_results = {
        name: _check_constraint(name, source.smiles, candidate.smiles, source.descriptors, candidate.descriptors, sim, normalized)
        for name in normalized["constraints"]
    }
    edit_results = {name: _check_edit(name, source.descriptors, candidate.descriptors, normalized) for name in normalized["edits"]}
    goal_success = all(goal_results.values()) if goal_results else True
    constraint_success = all(constraint_results.values()) if constraint_results else True
    edit_success = all(edit_results.values()) if edit_results else True
    druglike = passes_druglike_filters(candidate.descriptors)
    return {
        "valid": True,
        "source_valid": True,
        "candidate_valid": True,
        "goal_success": bool(goal_success),
        "constraint_success": bool(constraint_success),
        "edit_success": bool(edit_success),
        "overall_success": bool(goal_success and constraint_success and edit_success),
        "similarity_to_source": float(sim),
        "druglike": bool(druglike),
        "failed_goals": "|".join([k for k, v in goal_results.items() if not v]),
        "failed_constraints": "|".join([k for k, v in constraint_results.items() if not v]),
        "failed_edits": "|".join([k for k, v in edit_results.items() if not v]),
        "goal_results_json": json.dumps(goal_results, sort_keys=True),
        "constraint_results_json": json.dumps(constraint_results, sort_keys=True),
        "edit_results_json": json.dumps(edit_results, sort_keys=True),
        "property_delta_json": json.dumps(deltas, sort_keys=True),
    }


def property_delta(source_desc: dict[str, float], target_desc: dict[str, float]) -> dict[str, float]:
    keys = list(TARGET_COLUMNS) + ["ring_count", "C", "N", "O", "S", "F", "Cl", "Br", "I", "fg_halogen"]
    return {key: float(target_desc.get(key, 0.0) - source_desc.get(key, 0.0)) for key in keys}


def scaffold_smiles(smiles: str) -> str | None:
    if not chem_mod.RDKIT_AVAILABLE:
        return None
    try:  # pragma: no cover - RDKit path is exercised on the server.
        from rdkit.Chem.Scaffolds import MurckoScaffold

        mol = chem_mod.Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        return scaffold or ""
    except Exception:
        return None


def preserves_scaffold(source_smiles: str, candidate_smiles: str, similarity_min: float = 0.6) -> bool:
    source_desc = molecular_descriptors(source_smiles)
    candidate_desc = molecular_descriptors(candidate_smiles)
    if not source_desc.valid or not candidate_desc.valid:
        return False

    source_scaffold = scaffold_smiles(source_desc.smiles)
    candidate_scaffold = scaffold_smiles(candidate_desc.smiles)
    if source_scaffold is not None and candidate_scaffold is not None:
        if source_scaffold and candidate_scaffold:
            return source_scaffold == candidate_scaffold
        return (
            source_scaffold == candidate_scaffold
            and int(source_desc.descriptors.get("ring_count", 0)) == int(candidate_desc.descriptors.get("ring_count", 0))
            and tanimoto(source_desc.smiles, candidate_desc.smiles) >= similarity_min
        )

    return (
        int(source_desc.descriptors.get("scaffold_class", -1)) == int(candidate_desc.descriptors.get("scaffold_class", -2))
        and int(source_desc.descriptors.get("ring_count", -1)) == int(candidate_desc.descriptors.get("ring_count", -2))
        and tanimoto(source_desc.smiles, candidate_desc.smiles) >= similarity_min
    )


def _check_property_goal(goal: str, source: dict[str, float], candidate: dict[str, float], spec: dict[str, Any]) -> bool:
    rule = PROPERTY_GOALS.get(goal)
    if rule is None:
        return False
    delta = float(candidate.get(rule["column"], 0.0) - source.get(rule["column"], 0.0))
    required = threshold(spec, rule["threshold"], rule["default"])
    return bool(rule["direction"] * delta >= required)


def _check_constraint(
    name: str,
    source_smiles: str,
    candidate_smiles: str,
    source: dict[str, float],
    candidate: dict[str, float],
    similarity: float,
    spec: dict[str, Any],
) -> bool:
    if name not in CONSTRAINTS:
        return False
    if name == "preserve_scaffold":
        return preserves_scaffold(source_smiles, candidate_smiles, similarity_min=threshold(spec, "similarity_min"))
    if name == "keep_mw_similar":
        return abs(float(candidate.get("MW", 0.0) - source.get("MW", 0.0))) <= threshold(spec, "delta_mw_abs_max")
    if name == "keep_similarity":
        return similarity >= threshold(spec, "similarity_min")
    if name == "keep_druglike":
        return passes_druglike_filters(candidate)
    return False


def _check_edit(name: str, source: dict[str, float], candidate: dict[str, float], spec: dict[str, Any]) -> bool:
    rule = EDIT_RULES.get(name)
    if rule is None:
        return False
    if "columns" in rule:
        source_value = sum(float(source.get(col, 0.0)) for col in rule["columns"])
        candidate_value = sum(float(candidate.get(col, 0.0)) for col in rule["columns"])
    else:
        source_value = float(source.get(rule["column"], 0.0))
        candidate_value = float(candidate.get(rule["column"], 0.0))
    delta = candidate_value - source_value
    required = threshold(spec, rule["threshold"], rule["default"])
    return bool(rule["direction"] * delta >= required)


def score_verification(result: dict[str, Any]) -> float:
    """A compact score for retrieval baselines; not used as ground truth."""

    if not result.get("valid"):
        return 0.0
    parts = [
        1.0 if result.get("goal_success") else 0.0,
        1.0 if result.get("constraint_success") else 0.0,
        1.0 if result.get("edit_success") else 0.0,
    ]
    return float(10.0 * np.mean(parts) + result.get("similarity_to_source", 0.0))
