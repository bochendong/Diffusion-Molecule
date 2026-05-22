"""Benchmark scoring and verification helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .chem import molecular_descriptors, scaffold_key, tanimoto
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

    def to_row(self) -> dict[str, object]:
        return asdict(self)


def score_candidates(example: BenchmarkExample, candidates: list[Candidate]) -> list[CandidateScore]:
    target_scaffold = scaffold_key(example.target_smiles)
    out: list[CandidateScore] = []
    for candidate in candidates:
        rec = molecular_descriptors(candidate.smiles)
        sim = tanimoto(candidate.smiles, example.target_smiles)
        scaffold_match = bool(target_scaffold and scaffold_key(candidate.smiles) == target_scaffold)
        score = (1.0 if rec.valid else 0.0) + sim + (0.25 if scaffold_match else 0.0)
        out.append(
            CandidateScore(
                task_id=example.task_id,
                smiles=candidate.smiles,
                rank=candidate.rank,
                origin=candidate.origin,
                valid=rec.valid,
                target_tanimoto=sim,
                scaffold_match=scaffold_match,
                score=score,
            )
        )
    ranked = sorted(out, key=lambda item: item.score, reverse=True)
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
        )
        for idx, item in enumerate(ranked, start=1)
    ]


def summarize_scores(scores_by_task: list[list[CandidateScore]], hit_threshold: float = 0.65) -> dict[str, float]:
    n = max(1, len(scores_by_task))
    top1 = [scores[0] for scores in scores_by_task if scores]
    best = [max(scores, key=lambda item: item.target_tanimoto) for scores in scores_by_task if scores]
    return {
        "tasks": float(len(scores_by_task)),
        "top1_validity": sum(1.0 for item in top1 if item.valid) / n,
        "top1_target_tanimoto": sum(item.target_tanimoto for item in top1) / n,
        "top1_scaffold_match": sum(1.0 for item in top1 if item.scaffold_match) / n,
        "topk_target_hit": sum(1.0 for item in best if item.target_tanimoto >= hit_threshold) / n,
        "mean_best_tanimoto": sum(item.target_tanimoto for item in best) / n,
    }
