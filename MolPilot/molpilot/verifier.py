"""Deterministic and proxy verification for generated molecular candidates."""

from __future__ import annotations

from dataclasses import dataclass

from .chem import molecular_descriptors, passes_druglike_proxy, scaffold_key, tanimoto
from .schema import ObjectiveSpec


@dataclass
class VerificationResult:
    valid: bool
    hard_verifiable: bool
    goal_success: bool
    constraint_success: bool
    proxy_success: bool | None
    overall_success: bool
    reasons: list[str]
    descriptors: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "hard_verifiable": self.hard_verifiable,
            "goal_success": self.goal_success,
            "constraint_success": self.constraint_success,
            "proxy_success": self.proxy_success,
            "overall_success": self.overall_success,
            "reasons": "|".join(self.reasons),
            **{f"candidate_{k}": v for k, v in self.descriptors.items()},
        }


def verify_candidate(source_smiles: str | None, candidate_smiles: str, spec: ObjectiveSpec) -> VerificationResult:
    source = molecular_descriptors(source_smiles) if source_smiles else None
    candidate = molecular_descriptors(candidate_smiles)
    reasons: list[str] = []
    if not candidate.valid:
        return VerificationResult(False, spec.hard_verifiable, False, False, None, False, ["invalid_smiles"], {})

    goal_ok = _goals_ok(source.descriptors if source and source.valid else {}, candidate.descriptors, spec, reasons)
    constraint_ok = _constraints_ok(source_smiles, candidate.smiles, candidate.descriptors, spec, reasons)
    proxy_ok = _proxy_ok(source.descriptors if source and source.valid else {}, candidate.descriptors, spec, reasons)
    if spec.unverifiable_goals:
        reasons.append("contains_unverifiable_goal")

    overall = bool(candidate.valid and goal_ok and constraint_ok and (proxy_ok is not False) and spec.hard_verifiable)
    return VerificationResult(
        valid=True,
        hard_verifiable=spec.hard_verifiable,
        goal_success=goal_ok,
        constraint_success=constraint_ok,
        proxy_success=proxy_ok,
        overall_success=overall,
        reasons=reasons,
        descriptors=candidate.descriptors,
    )


def _goals_ok(source: dict[str, float], cand: dict[str, float], spec: ObjectiveSpec, reasons: list[str]) -> bool:
    ok = True
    for goal in spec.goals:
        if goal == "decrease_logp":
            ok &= _delta_at_most(source, cand, "LogP", -spec.thresholds["delta_logp_min"], reasons, goal)
        elif goal == "increase_logp":
            ok &= _delta_at_least(source, cand, "LogP", spec.thresholds["delta_logp_min"], reasons, goal)
        elif goal == "improve_qed":
            ok &= _delta_at_least(source, cand, "QED", spec.thresholds["delta_qed_min"], reasons, goal)
        elif goal == "reduce_tpsa":
            ok &= _delta_at_most(source, cand, "TPSA", -spec.thresholds["delta_tpsa_min"], reasons, goal)
        elif goal == "increase_tpsa":
            ok &= _delta_at_least(source, cand, "TPSA", spec.thresholds["delta_tpsa_min"], reasons, goal)
        elif goal == "decrease_mw":
            ok &= _delta_at_most(source, cand, "MW", -spec.thresholds["delta_mw_min"], reasons, goal)
        elif goal == "increase_mw":
            ok &= _delta_at_least(source, cand, "MW", spec.thresholds["delta_mw_min"], reasons, goal)
        elif goal == "decrease_rb":
            ok &= _delta_at_most(source, cand, "RB", -spec.thresholds["delta_rb_min"], reasons, goal)
    return bool(ok)


def _constraints_ok(source_smiles: str | None, candidate_smiles: str, cand: dict[str, float], spec: ObjectiveSpec, reasons: list[str]) -> bool:
    ok = True
    source_smiles = source_smiles or ""
    for constraint in spec.constraints:
        if constraint == "keep_similarity":
            sim = tanimoto(source_smiles, candidate_smiles)
            passed = sim >= spec.thresholds["similarity_min"]
            if not passed:
                reasons.append("low_similarity")
            ok &= passed
        elif constraint == "preserve_scaffold":
            passed = bool(source_smiles) and scaffold_key(source_smiles) == scaffold_key(candidate_smiles)
            if not passed:
                reasons.append("scaffold_changed")
            ok &= passed
        elif constraint == "keep_mw_similar":
            src = molecular_descriptors(source_smiles)
            passed = src.valid and abs(cand.get("MW", 0.0) - src.descriptors.get("MW", 0.0)) <= spec.thresholds["delta_mw_abs_max"]
            if not passed:
                reasons.append("mw_drift")
            ok &= passed
        elif constraint == "keep_druglike":
            passed = passes_druglike_proxy(cand)
            if not passed:
                reasons.append("druglike_failed")
            ok &= passed
        elif constraint == "cns_like":
            passed = (
                cand.get("MW", 999.0) <= spec.thresholds["cns_mw_max"]
                and cand.get("TPSA", 999.0) <= spec.thresholds["cns_tpsa_max"]
                and cand.get("HBD", 999.0) <= spec.thresholds["cns_hbd_max"]
            )
            if not passed:
                reasons.append("cns_profile_failed")
            ok &= passed
    return bool(ok)


def _proxy_ok(source: dict[str, float], cand: dict[str, float], spec: ObjectiveSpec, reasons: list[str]) -> bool | None:
    if not spec.proxy_goals:
        return None
    ok = True
    for goal in spec.proxy_goals:
        if goal == "improve_solubility_proxy":
            passed = cand.get("LogP", 99.0) < source.get("LogP", 99.0) or cand.get("TPSA", 0.0) > source.get("TPSA", 0.0)
        elif goal == "improve_bbb_proxy":
            passed = cand.get("TPSA", 999.0) <= 90.0 and cand.get("HBD", 99.0) <= 1.0
        else:
            passed = True
            reasons.append(f"{goal}_needs_learned_predictor")
        ok &= passed
    return bool(ok)


def _delta_at_least(source: dict[str, float], cand: dict[str, float], key: str, threshold: float, reasons: list[str], name: str) -> bool:
    passed = cand.get(key, 0.0) - source.get(key, 0.0) >= threshold
    if not passed:
        reasons.append(f"{name}_failed")
    return bool(passed)


def _delta_at_most(source: dict[str, float], cand: dict[str, float], key: str, threshold: float, reasons: list[str], name: str) -> bool:
    passed = cand.get(key, 0.0) - source.get(key, 0.0) <= threshold
    if not passed:
        reasons.append(f"{name}_failed")
    return bool(passed)

