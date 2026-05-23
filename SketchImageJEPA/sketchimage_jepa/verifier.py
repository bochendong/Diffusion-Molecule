"""Benchmark scoring and verification helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .chem import molecular_descriptors, scaffold_key, tanimoto
from .property_guidance import PROPERTY_KEYS, property_mae, property_success
from .schema import BenchmarkExample, Candidate


@dataclass(frozen=True)
class CandidateScore:
    task_id: str
    smiles: str
    rank: int
    origin: str
    valid: bool
    target_tanimoto: float
    scaffold_match: bool
    score: float
    property_mae: float = 0.0
    property_success: bool = False
    property_errors: dict[str, float] = field(default_factory=dict)

    def to_row(self) -> dict[str, object]:
        return asdict(self)


def score_candidates(example: BenchmarkExample, candidates: list[Candidate]) -> list[CandidateScore]:
    target_scaffold = scaffold_key(example.target_smiles)
    target_desc = molecular_descriptors(example.target_smiles).descriptors
    out: list[CandidateScore] = []
    for candidate in candidates:
        rec = molecular_descriptors(candidate.smiles)
        sim = tanimoto(candidate.smiles, example.target_smiles)
        scaffold_match = bool(target_scaffold and scaffold_key(candidate.smiles) == target_scaffold)
        errors = _absolute_property_errors(rec.descriptors, target_desc)
        out.append(
            CandidateScore(
                task_id=example.task_id,
                smiles=candidate.smiles,
                rank=candidate.rank,
                origin=candidate.origin,
                valid=rec.valid,
                target_tanimoto=sim,
                scaffold_match=scaffold_match,
                score=float(candidate.score),
                property_mae=property_mae(rec.descriptors, target_desc),
                property_success=property_success(rec.descriptors, target_desc),
                property_errors=errors,
            )
        )
    ranked = sorted(out, key=lambda item: item.rank)
    return [
        CandidateScore(
            task_id=item.task_id,
            smiles=item.smiles,
            rank=idx,
            origin=item.origin,
            valid=item.valid,
            target_tanimoto=item.target_tanimoto,
            scaffold_match=item.scaffold_match,
            score=item.score,
            property_mae=item.property_mae,
            property_success=item.property_success,
            property_errors=item.property_errors,
        )
        for idx, item in enumerate(ranked, start=1)
    ]


def summarize_scores(scores_by_task: list[list[CandidateScore]], hit_threshold: float = 0.65) -> dict[str, float]:
    n = max(1, len(scores_by_task))
    top1 = [scores[0] for scores in scores_by_task if scores]
    best = [max(scores, key=lambda item: item.target_tanimoto) for scores in scores_by_task if scores]
    best_property = [min(scores, key=lambda item: item.property_mae) for scores in scores_by_task if scores]
    return {
        "tasks": float(len(scores_by_task)),
        "top1_validity": sum(1.0 for item in top1 if item.valid) / n,
        "top1_target_tanimoto": sum(item.target_tanimoto for item in top1) / n,
        "top1_scaffold_match": sum(1.0 for item in top1 if item.scaffold_match) / n,
        "topk_target_hit": sum(1.0 for item in best if item.target_tanimoto >= hit_threshold) / n,
        "mean_best_tanimoto": sum(item.target_tanimoto for item in best) / n,
        "top1_property_mae": sum(item.property_mae for item in top1) / n,
        "mean_best_property_mae": sum(item.property_mae for item in best_property) / n,
        "top1_property_success": sum(1.0 for item in top1 if item.property_success) / n,
        "topk_property_success": sum(1.0 for item in best_property if item.property_success) / n,
    }


def _absolute_property_errors(candidate: dict[str, float], target: dict[str, float]) -> dict[str, float]:
    return {key: abs(float(candidate.get(key, 0.0)) - float(target.get(key, 0.0))) for key in PROPERTY_KEYS}
