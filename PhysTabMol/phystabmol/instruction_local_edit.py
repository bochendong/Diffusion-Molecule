"""Local edit / inpainting-style metrics for instruction-guided molecules.

SketchMol evaluates molecular editing through an image inpainting lens: keep a
visible region and redraw the rest. For PhysTabMol we make the same idea
deterministic in molecule space by using the source/target maximum common
substructure as the preserved region, then measuring whether a candidate keeps
that region while moving toward the target edit.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from . import chem as chem_mod
from .chem import canonicalize_smiles, molecular_descriptors, tanimoto


LOCAL_EDIT_NUMERIC_KEYS = (
    "source_target_similarity",
    "source_candidate_similarity",
    "target_candidate_similarity",
    "source_target_core_fraction",
    "source_candidate_core_fraction",
    "target_candidate_core_fraction",
    "source_target_edit_fraction",
    "source_candidate_edit_fraction",
    "target_candidate_edit_fraction",
    "preservation_threshold",
    "target_recovery_margin",
)


def edit_region_summary(source_smiles: str, target_smiles: str) -> dict[str, Any]:
    """Describe the deterministic keep/edit region between two molecules."""

    source_can = canonicalize_smiles(source_smiles)
    target_can = canonicalize_smiles(target_smiles)
    source_desc = molecular_descriptors(source_smiles)
    target_desc = molecular_descriptors(target_smiles)
    base: dict[str, Any] = {
        "source_smiles": source_can or source_smiles,
        "target_smiles": target_can or target_smiles,
        "source_valid": bool(source_desc.valid),
        "target_valid": bool(target_desc.valid),
        "source_atoms": _heavy_atom_count(source_desc.descriptors),
        "target_atoms": _heavy_atom_count(target_desc.descriptors),
        "source_target_similarity": float(tanimoto(source_smiles, target_smiles)),
        "mcs_available": False,
        "mcs_smarts": "",
        "mcs_atoms": 0,
        "source_core_atoms": 0,
        "target_core_atoms": 0,
        "source_edit_atoms": 0,
        "target_edit_atoms": 0,
        "source_core_fraction": 0.0,
        "target_core_fraction": 0.0,
        "source_edit_fraction": 1.0,
        "target_edit_fraction": 1.0,
        "source_edit_atom_symbols": {},
        "target_edit_atom_symbols": {},
    }
    if not source_desc.valid or not target_desc.valid:
        return base
    if not chem_mod.RDKIT_AVAILABLE:
        return _fallback_edit_region(base, source_desc.descriptors, target_desc.descriptors)
    try:  # pragma: no cover - RDKit path is exercised on the server.
        from rdkit.Chem import rdFMCS

        source_mol = chem_mod.Chem.MolFromSmiles(source_smiles)
        target_mol = chem_mod.Chem.MolFromSmiles(target_smiles)
        if source_mol is None or target_mol is None:
            return base
        mcs = rdFMCS.FindMCS(
            [source_mol, target_mol],
            timeout=1,
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
            matchValences=True,
        )
        if not mcs.smartsString:
            return base
        pattern = chem_mod.Chem.MolFromSmarts(mcs.smartsString)
        if pattern is None:
            return base
        source_match = set(source_mol.GetSubstructMatch(pattern))
        target_match = set(target_mol.GetSubstructMatch(pattern))
        source_atoms = int(source_mol.GetNumAtoms())
        target_atoms = int(target_mol.GetNumAtoms())
        source_edit = source_atoms - len(source_match)
        target_edit = target_atoms - len(target_match)
        base.update(
            {
                "source_atoms": source_atoms,
                "target_atoms": target_atoms,
                "mcs_available": True,
                "mcs_canceled": bool(getattr(mcs, "canceled", False)),
                "mcs_smarts": mcs.smartsString,
                "mcs_atoms": int(mcs.numAtoms),
                "source_core_atoms": int(len(source_match)),
                "target_core_atoms": int(len(target_match)),
                "source_edit_atoms": int(source_edit),
                "target_edit_atoms": int(target_edit),
                "source_core_fraction": _safe_fraction(len(source_match), source_atoms),
                "target_core_fraction": _safe_fraction(len(target_match), target_atoms),
                "source_edit_fraction": _safe_fraction(source_edit, source_atoms),
                "target_edit_fraction": _safe_fraction(target_edit, target_atoms),
                "source_edit_atom_symbols": _unmatched_symbols(source_mol, source_match),
                "target_edit_atom_symbols": _unmatched_symbols(target_mol, target_match),
            }
        )
        return base
    except Exception:
        return _fallback_edit_region(base, source_desc.descriptors, target_desc.descriptors)


def local_edit_metrics(source_smiles: str, target_smiles: str, candidate_smiles: str) -> dict[str, Any]:
    """Return local edit metrics for a generated candidate."""

    source_candidate = edit_region_summary(source_smiles, candidate_smiles)
    source_target = edit_region_summary(source_smiles, target_smiles)
    target_candidate = edit_region_summary(target_smiles, candidate_smiles)
    source_target_sim = float(source_target.get("source_target_similarity", 0.0))
    source_candidate_sim = float(source_candidate.get("source_target_similarity", 0.0))
    target_candidate_sim = float(target_candidate.get("source_target_similarity", 0.0))
    source_target_core = float(source_target.get("source_core_fraction", 0.0))
    source_candidate_core = float(source_candidate.get("source_core_fraction", 0.0))
    target_candidate_core = float(target_candidate.get("source_core_fraction", 0.0))
    source_target_edit = float(source_target.get("target_edit_fraction", 1.0))
    source_candidate_edit = float(source_candidate.get("target_edit_fraction", 1.0))
    target_candidate_edit = float(target_candidate.get("target_edit_fraction", 1.0))
    preserve_threshold = max(0.45, min(0.9, source_target_core - 0.1))
    target_recovery_margin = target_candidate_sim - source_target_sim
    candidate_can = canonicalize_smiles(candidate_smiles) or candidate_smiles
    source_can = canonicalize_smiles(source_smiles) or source_smiles
    target_can = canonicalize_smiles(target_smiles) or target_smiles
    local_preservation_success = source_candidate_core >= preserve_threshold
    target_recovery_success = target_candidate_sim >= max(0.45, source_target_sim + 0.03)
    return {
        "source_target_similarity": source_target_sim,
        "source_candidate_similarity": source_candidate_sim,
        "target_candidate_similarity": target_candidate_sim,
        "source_target_core_fraction": source_target_core,
        "source_candidate_core_fraction": source_candidate_core,
        "target_candidate_core_fraction": target_candidate_core,
        "source_target_edit_fraction": source_target_edit,
        "source_candidate_edit_fraction": source_candidate_edit,
        "target_candidate_edit_fraction": target_candidate_edit,
        "preservation_threshold": float(preserve_threshold),
        "target_recovery_margin": float(target_recovery_margin),
        "local_preservation_success": bool(local_preservation_success),
        "target_recovery_success": bool(target_recovery_success),
        "sketchmol_local_edit_success": bool(local_preservation_success and target_recovery_success),
        "candidate_equals_source": bool(candidate_can == source_can),
        "candidate_equals_target": bool(candidate_can == target_can),
    }


def _fallback_edit_region(base: dict[str, Any], source: dict[str, float], target: dict[str, float]) -> dict[str, Any]:
    source_atoms = max(1, _heavy_atom_count(source))
    target_atoms = max(1, _heavy_atom_count(target))
    overlap = 0
    source_symbols: Counter[str] = Counter()
    target_symbols: Counter[str] = Counter()
    for atom in ("C", "N", "O", "S", "F", "Cl", "Br", "I"):
        s_count = int(source.get(atom, 0.0))
        t_count = int(target.get(atom, 0.0))
        overlap += min(s_count, t_count)
        if s_count > t_count:
            source_symbols[atom] = s_count - t_count
        if t_count > s_count:
            target_symbols[atom] = t_count - s_count
    source_edit = max(0, source_atoms - overlap)
    target_edit = max(0, target_atoms - overlap)
    base.update(
        {
            "source_atoms": source_atoms,
            "target_atoms": target_atoms,
            "mcs_atoms": int(overlap),
            "source_core_atoms": int(overlap),
            "target_core_atoms": int(overlap),
            "source_edit_atoms": int(source_edit),
            "target_edit_atoms": int(target_edit),
            "source_core_fraction": _safe_fraction(overlap, source_atoms),
            "target_core_fraction": _safe_fraction(overlap, target_atoms),
            "source_edit_fraction": _safe_fraction(source_edit, source_atoms),
            "target_edit_fraction": _safe_fraction(target_edit, target_atoms),
            "source_edit_atom_symbols": dict(source_symbols),
            "target_edit_atom_symbols": dict(target_symbols),
        }
    )
    return base


def _unmatched_symbols(mol: Any, matched_atoms: set[int]) -> dict[str, int]:
    counts = Counter(atom.GetSymbol() for atom in mol.GetAtoms() if atom.GetIdx() not in matched_atoms)
    return dict(sorted(counts.items()))


def _heavy_atom_count(desc: dict[str, float]) -> int:
    return int(sum(float(desc.get(atom, 0.0)) for atom in ("C", "N", "O", "S", "F", "Cl", "Br", "I")))


def _safe_fraction(num: int | float, denom: int | float) -> float:
    denom = float(denom)
    if denom <= 0:
        return 0.0
    return float(num) / denom
