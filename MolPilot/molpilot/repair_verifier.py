"""Verified molecular repair metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .chem import DESCRIPTOR_KEYS, canonicalize_smiles, molecular_descriptors, scaffold_key, tanimoto


DESCRIPTOR_SCALES = {
    "MW": 500.0,
    "LogP": 5.0,
    "QED": 1.0,
    "TPSA": 150.0,
    "HBD": 5.0,
    "HBA": 10.0,
    "RB": 10.0,
}


@dataclass
class RepairVerificationResult:
    valid: bool
    hard_verifiable: bool
    goal_success: bool
    constraint_success: bool
    proxy_success: bool | None
    overall_success: bool
    reasons: list[str]
    descriptors: dict[str, float]
    corrupted_valid: bool
    exact_recovery: bool
    scaffold_recovery: bool
    tanimoto_to_clean: float
    property_mae_to_clean: float
    novel: bool
    novel_verified_success: bool
    repair_quality: float
    soft_repair_success: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "hard_verifiable": self.hard_verifiable,
            "goal_success": self.goal_success,
            "constraint_success": self.constraint_success,
            "proxy_success": self.proxy_success,
            "overall_success": self.overall_success,
            "reasons": "|".join(self.reasons),
            "corrupted_valid": self.corrupted_valid,
            "exact_recovery": self.exact_recovery,
            "scaffold_recovery": self.scaffold_recovery,
            "tanimoto_to_clean": self.tanimoto_to_clean,
            "property_mae_to_clean": self.property_mae_to_clean,
            "novel": self.novel,
            "novel_verified_success": self.novel_verified_success,
            "repair_quality": self.repair_quality,
            "soft_repair_success": self.soft_repair_success,
            **{f"candidate_{key}": value for key, value in self.descriptors.items()},
        }


def verify_repair(
    corrupted_smiles: str | None,
    clean_smiles: str,
    candidate_smiles: str,
    known_smiles: Iterable[str] | None = None,
    similarity_min: float = 0.60,
    property_mae_max: float = 0.20,
) -> RepairVerificationResult:
    corrupted = molecular_descriptors(corrupted_smiles)
    clean = molecular_descriptors(clean_smiles)
    candidate = molecular_descriptors(candidate_smiles)
    known = set(known_smiles or [])
    reasons: list[str] = []
    if not clean.valid:
        reasons.append("invalid_clean_target")
    if not candidate.valid:
        return RepairVerificationResult(
            valid=False,
            hard_verifiable=True,
            goal_success=False,
            constraint_success=False,
            proxy_success=None,
            overall_success=False,
            reasons=["invalid_smiles"],
            descriptors={},
            corrupted_valid=corrupted.valid,
            exact_recovery=False,
            scaffold_recovery=False,
            tanimoto_to_clean=0.0,
            property_mae_to_clean=float("inf"),
            novel=False,
            novel_verified_success=False,
            repair_quality=0.0,
            soft_repair_success=False,
        )

    clean_canon = canonicalize_smiles(clean_smiles) or clean_smiles
    corrupted_canon = canonicalize_smiles(corrupted_smiles) or str(corrupted_smiles or "")
    candidate_canon = candidate.smiles
    exact = bool(candidate_canon == clean_canon)
    scaffold_ok = _scaffold_recovered(clean_canon, candidate_canon)
    sim = tanimoto(clean_canon, candidate_canon)
    prop_mae = _property_mae(clean.descriptors, candidate.descriptors) if clean.valid else float("inf")
    if not exact and not scaffold_ok:
        reasons.append("scaffold_not_recovered")
    if not exact and sim < similarity_min:
        reasons.append("low_similarity_to_clean")
    if not exact and prop_mae > property_mae_max:
        reasons.append("property_drift_from_clean")
    unchanged_corruption = bool(candidate_canon == corrupted_canon and candidate_canon != clean_canon)
    if unchanged_corruption:
        reasons.append("unrepaired_corruption")

    constraint_ok = bool((exact or (scaffold_ok and sim >= similarity_min and prop_mae <= property_mae_max)) and not unchanged_corruption)
    novel = bool(candidate_canon not in known and candidate_canon != clean_canon)
    overall = bool(candidate.valid and clean.valid and constraint_ok)
    repair_quality = _repair_quality(
        exact=exact,
        scaffold_ok=scaffold_ok,
        similarity=sim,
        property_mae=prop_mae,
        unchanged_corruption=unchanged_corruption,
        similarity_min=similarity_min,
        property_mae_max=property_mae_max,
    )
    soft_repair = bool(candidate.valid and clean.valid and not unchanged_corruption and repair_quality >= 0.65)
    return RepairVerificationResult(
        valid=True,
        hard_verifiable=True,
        goal_success=True,
        constraint_success=constraint_ok,
        proxy_success=None,
        overall_success=overall,
        reasons=reasons,
        descriptors=candidate.descriptors,
        corrupted_valid=corrupted.valid,
        exact_recovery=exact,
        scaffold_recovery=scaffold_ok,
        tanimoto_to_clean=sim,
        property_mae_to_clean=prop_mae,
        novel=novel,
        novel_verified_success=bool(overall and novel),
        repair_quality=repair_quality,
        soft_repair_success=soft_repair,
    )


def _scaffold_recovered(clean_smiles: str, candidate_smiles: str) -> bool:
    clean_scaffold = scaffold_key(clean_smiles)
    candidate_scaffold = scaffold_key(candidate_smiles)
    if clean_scaffold and candidate_scaffold:
        return clean_scaffold == candidate_scaffold
    return clean_smiles == candidate_smiles


def _property_mae(clean: dict[str, float], candidate: dict[str, float]) -> float:
    values = []
    for key in DESCRIPTOR_KEYS:
        scale = DESCRIPTOR_SCALES[key]
        values.append(abs(candidate.get(key, 0.0) - clean.get(key, 0.0)) / scale)
    return float(sum(values) / max(1, len(values)))


def _repair_quality(
    *,
    exact: bool,
    scaffold_ok: bool,
    similarity: float,
    property_mae: float,
    unchanged_corruption: bool,
    similarity_min: float,
    property_mae_max: float,
) -> float:
    """Continuous repair quality for ranking and soft benchmark reporting.

    The hard repair label remains exact-or-scaffold-constrained. This score is
    a smoother diagnostic: it rewards valid candidates that recover the intended
    chemistry even when a strict Murcko scaffold key is brittle for small or
    highly substituted molecules.
    """

    if property_mae == float("inf"):
        return 0.0
    sim_scale = max(0.90, float(similarity_min))
    sim_score = _clamp01(float(similarity) / max(1e-8, sim_scale))
    prop_score = 1.0 - _clamp01(float(property_mae) / max(1e-8, float(property_mae_max)))
    scaffold_score = 1.0 if scaffold_ok else 0.0
    exact_score = 1.0 if exact else 0.0
    repaired_score = 0.0 if unchanged_corruption else 1.0
    return float(
        0.40 * sim_score
        + 0.25 * prop_score
        + 0.20 * scaffold_score
        + 0.10 * exact_score
        + 0.05 * repaired_score
    )


def _clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))
