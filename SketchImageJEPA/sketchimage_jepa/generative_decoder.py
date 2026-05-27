"""Prototype generative molecular decoder.

This decoder is intentionally small and deterministic. It augments the existing
latent retrieval seeds with local molecule mutations, giving the experiment a
candidate surface that can leave the training pool before we invest in a larger
learned graph or SELFIES decoder.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np

from .chem import Chem, RDKIT_AVAILABLE, canonicalize_smiles, descriptor_delta, molecular_descriptors, scaffold_key, tanimoto
from .decoder import RetrievalDecoder, _asks_to_preserve_source, _cosine_similarity
from .features import molecule_latent, stable_hash_vector
from .property_guidance import PROPERTY_KEYS, PROPERTY_TOLERANCES, parse_property_targets, property_match_score
from .schema import BenchmarkExample, Candidate, TaskType

try:  # pragma: no cover - depends on RDKit being available.
    from rdkit.Chem import rdFMCS
except Exception:  # pragma: no cover
    rdFMCS = None


DE_NOVO_TEMPLATES = (
    "c1ccccc1",
    "c1ccncc1",
    "c1ccoc1",
    "c1ccsc1",
    "C1CCCCC1",
    "CCO",
    "CCN",
    "CC(=O)O",
    "COc1ccccc1",
    "Nc1ccccc1",
    "Oc1ccccc1",
    "CC(=O)Nc1ccccc1",
    "CCOC(=O)c1ccccc1",
)

_DELTA_FIELDS = {
    "MW": "delta_mw",
    "LogP": "delta_logp",
    "QED": "delta_qed",
    "TPSA": "delta_tpsa",
}


@dataclass(frozen=True)
class LearnedTransform:
    kind: str
    task_type: str
    anchor_atomic_num: int = 0
    anchor_is_aromatic: bool = False
    root_atomic_num: int = 0
    fragment_smiles: str = ""
    bond_type: str = "SINGLE"
    component_size: int = 0
    source_token: str = ""
    target_token: str = ""
    support: int = 1
    delta_mw: float = 0.0
    delta_logp: float = 0.0
    delta_qed: float = 0.0
    delta_tpsa: float = 0.0

    def signature(self) -> tuple[object, ...]:
        return (
            self.kind,
            self.task_type,
            self.anchor_atomic_num,
            self.anchor_is_aromatic,
            self.root_atomic_num,
            self.fragment_smiles,
            self.bond_type,
            self.component_size,
            self.source_token,
            self.target_token,
        )

    def property_delta(self) -> dict[str, float]:
        return {
            "MW": float(self.delta_mw),
            "LogP": float(self.delta_logp),
            "QED": float(self.delta_qed),
            "TPSA": float(self.delta_tpsa),
        }


class GenerativeMutationDecoder:
    """Generate candidate molecules from retrieval seeds and local edits.

    The score uses only planner latent similarity plus non-oracle task cues
    (source similarity, requested properties, and source scaffold preservation).
    It never scores against the hidden target molecule.
    """

    def __init__(
        self,
        smiles: list[str],
        latents: np.ndarray,
        de_novo_latent_rerank_weight: float = 0.05,
        source_rerank_weight: float = 0.35,
        property_rerank_weight: float = 0.25,
        scaffold_rerank_bonus: float = 0.15,
        seed_count: int = 24,
        mutation_rounds: int = 1,
        candidates_per_seed: int = 8,
        novelty_bonus: float = 0.05,
        include_retrieval: bool = True,
    ):
        self.retrieval = RetrievalDecoder(
            smiles,
            latents,
            de_novo_latent_rerank_weight=de_novo_latent_rerank_weight,
            source_rerank_weight=source_rerank_weight,
            property_rerank_weight=property_rerank_weight,
            scaffold_rerank_bonus=scaffold_rerank_bonus,
        )
        self.train_smiles = set(self.retrieval.smiles)
        self.latent_dim = int(np.asarray(latents).shape[1])
        self.source_rerank_weight = float(source_rerank_weight)
        self.property_rerank_weight = float(property_rerank_weight)
        self.scaffold_rerank_bonus = float(scaffold_rerank_bonus)
        self.de_novo_latent_rerank_weight = float(de_novo_latent_rerank_weight)
        self.seed_count = max(1, int(seed_count))
        self.mutation_rounds = max(1, int(mutation_rounds))
        self.candidates_per_seed = max(1, int(candidates_per_seed))
        self.novelty_bonus = float(novelty_bonus)
        self.include_retrieval = bool(include_retrieval)

    def decode(
        self,
        pred_latents: np.ndarray,
        source_smiles: list[str | None],
        top_k: int = 5,
        examples: list[BenchmarkExample] | None = None,
        source_latents: np.ndarray | None = None,
    ) -> list[list[Candidate]]:
        pred_latents = np.asarray(pred_latents, dtype=np.float32)
        seed_rows = self.retrieval.decode(
            pred_latents,
            source_smiles,
            top_k=max(top_k, self.seed_count),
            examples=examples,
            source_latents=source_latents,
        )
        all_rows: list[list[Candidate]] = []
        for row_idx, seed_candidates in enumerate(seed_rows):
            example = examples[row_idx] if examples else None
            source = canonicalize_smiles(source_smiles[row_idx]) if source_smiles[row_idx] else None
            row_by_smiles: dict[str, Candidate] = {}

            if self.include_retrieval:
                for candidate in seed_candidates[:top_k]:
                    self._maybe_add(row_by_smiles, candidate)

            seed_smiles = self._seed_smiles(seed_candidates, source, example)
            generated = self._generate_candidates(seed_smiles, example, source, pred_latent=pred_latents[row_idx])
            for smiles, origin in generated:
                if source and smiles == source:
                    continue
                score = self._score(smiles, pred_latents[row_idx], source, example)
                candidate = Candidate(smiles=smiles, origin=origin, score=score)
                self._maybe_add(row_by_smiles, candidate)

            if not row_by_smiles:
                for candidate in seed_candidates[:top_k]:
                    fallback = Candidate(
                        smiles=candidate.smiles,
                        origin="generative_fallback_retrieval",
                        score=candidate.score,
                    )
                    self._maybe_add(row_by_smiles, fallback)

            ranked = sorted(row_by_smiles.values(), key=lambda candidate: (-candidate.score, candidate.smiles))
            all_rows.append([Candidate(smiles=c.smiles, origin=c.origin, score=c.score, rank=idx + 1) for idx, c in enumerate(ranked[:top_k])])
        return all_rows

    def _generate_candidates(
        self,
        seeds: list[str],
        example: BenchmarkExample | None,
        source: str | None,
        pred_latent: np.ndarray | None = None,
    ) -> list[tuple[str, str]]:
        return [(smiles, "generated_mutation") for smiles in self._generate(seeds, example, source)]

    def _seed_smiles(
        self,
        retrieval_candidates: list[Candidate],
        source: str | None,
        example: BenchmarkExample | None,
    ) -> list[str]:
        seeds: list[str] = []
        if source:
            seeds.append(source)
        if example and example.task_type == TaskType.DE_NOVO:
            seeds.extend(DE_NOVO_TEMPLATES)
        seeds.extend(candidate.smiles for candidate in retrieval_candidates[: self.seed_count])
        return _unique_valid(seeds)

    def _generate(
        self,
        seeds: list[str],
        example: BenchmarkExample | None,
        source: str | None,
    ) -> list[str]:
        generated: set[str] = set()
        frontier = list(seeds)
        for round_idx in range(self.mutation_rounds):
            next_frontier: list[str] = []
            for seed in frontier:
                variants = _mutate_smiles(seed, limit=self.candidates_per_seed, salt=f"{round_idx}:{seed}")
                variants = _rank_generation_pool(variants, example, source)
                for variant in variants[: self.candidates_per_seed]:
                    if variant not in generated and variant not in seeds:
                        generated.add(variant)
                        next_frontier.append(variant)
            frontier = next_frontier[: max(self.seed_count, self.candidates_per_seed)]
            if not frontier:
                break
        return _rank_generation_pool(generated, example, source)

    def _score(
        self,
        smiles: str,
        pred_latent: np.ndarray,
        source: str | None,
        example: BenchmarkExample | None,
    ) -> float:
        candidate_latent = molecule_latent(smiles, self.latent_dim).reshape(1, -1)
        latent_score = float(_cosine_similarity(pred_latent.reshape(1, -1), candidate_latent)[0, 0])
        score = latent_score

        if example and example.task_type == TaskType.DE_NOVO:
            targets = parse_property_targets(example.instruction)
            if targets:
                desc = molecular_descriptors(smiles).descriptors
                score = self.de_novo_latent_rerank_weight * latent_score + property_match_score(desc, targets)
        elif example and example.task_type in {TaskType.EDIT, TaskType.INPAINT, TaskType.FRAGMENT_GROW}:
            if source:
                score += self.source_rerank_weight * tanimoto(source, smiles)
            targets = parse_property_targets(example.instruction)
            if targets:
                desc = molecular_descriptors(smiles).descriptors
                score += self.property_rerank_weight * property_match_score(desc, targets)
            if source and _asks_to_preserve_source(example) and scaffold_key(source) == scaffold_key(smiles):
                score += self.scaffold_rerank_bonus

        if smiles not in self.train_smiles:
            score += self.novelty_bonus
        return float(score)

    @staticmethod
    def _maybe_add(row_by_smiles: dict[str, Candidate], candidate: Candidate) -> None:
        smiles = canonicalize_smiles(candidate.smiles)
        if not smiles:
            return
        current = row_by_smiles.get(smiles)
        updated = Candidate(smiles=smiles, origin=candidate.origin, score=float(candidate.score))
        if current is None or updated.score > current.score:
            row_by_smiles[smiles] = updated


class LearnedTransformDecoder(GenerativeMutationDecoder):
    """Generate candidates by applying transforms learned from train pairs."""

    transform_origin = "learned_transform"

    def __init__(
        self,
        smiles: list[str],
        latents: np.ndarray,
        train_examples: list[BenchmarkExample],
        de_novo_latent_rerank_weight: float = 0.05,
        source_rerank_weight: float = 0.35,
        property_rerank_weight: float = 0.25,
        scaffold_rerank_bonus: float = 0.15,
        seed_count: int = 24,
        mutation_rounds: int = 1,
        candidates_per_seed: int = 8,
        novelty_bonus: float = 0.05,
        include_retrieval: bool = True,
        include_mutation_fallback: bool = True,
        max_transform_examples: int = 1200,
    ):
        super().__init__(
            smiles,
            latents,
            de_novo_latent_rerank_weight=de_novo_latent_rerank_weight,
            source_rerank_weight=source_rerank_weight,
            property_rerank_weight=property_rerank_weight,
            scaffold_rerank_bonus=scaffold_rerank_bonus,
            seed_count=seed_count,
            mutation_rounds=mutation_rounds,
            candidates_per_seed=candidates_per_seed,
            novelty_bonus=novelty_bonus,
            include_retrieval=include_retrieval,
        )
        self.transforms = _learn_transform_library(
            train_examples,
            max_transforms=max(32, self.seed_count * 12),
            max_examples=max_transform_examples,
        )
        self.include_mutation_fallback = bool(include_mutation_fallback)

    def _generate_candidates(
        self,
        seeds: list[str],
        example: BenchmarkExample | None,
        source: str | None,
        pred_latent: np.ndarray | None = None,
    ) -> list[tuple[str, str]]:
        generated: dict[str, str] = {}
        bases = self._transform_bases(seeds, example, source)
        transforms = self._select_transforms(example, source=source)
        for base in bases:
            for transform in transforms:
                for smiles in _apply_transform(base, transform):
                    if smiles and smiles != base:
                        generated.setdefault(smiles, self.transform_origin)

        if not generated and self.include_mutation_fallback:
            for smiles in self._generate(seeds, example, source):
                generated.setdefault(smiles, "generated_mutation")

        ranked = _rank_generation_pool(generated.keys(), example, source)
        return [(smiles, generated[smiles]) for smiles in ranked]

    def _transform_bases(
        self,
        seeds: list[str],
        example: BenchmarkExample | None,
        source: str | None,
    ) -> list[str]:
        bases: list[str] = []
        if source:
            bases.append(source)
        if example and example.task_type == TaskType.DE_NOVO:
            bases.extend(DE_NOVO_TEMPLATES)
        bases.extend(seeds[: max(1, min(4, self.seed_count))])
        return _unique_valid(bases)

    def _select_transforms(
        self,
        example: BenchmarkExample | None,
        source: str | None = None,
        pred_latent: np.ndarray | None = None,
    ) -> list[LearnedTransform]:
        if not self.transforms:
            return []
        task_type = example.task_type.value if example else ""
        same_task = [transform for transform in self.transforms if transform.task_type == task_type]
        other = [transform for transform in self.transforms if transform.task_type != task_type]
        return (same_task + other)[: self.seed_count]


class ScaffoldPreservingTransformDecoder(LearnedTransformDecoder):
    """Apply learned transforms while preserving the source scaffold/core."""

    transform_origin = "scaffold_preserving_transform"

    def __init__(
        self,
        smiles: list[str],
        latents: np.ndarray,
        train_examples: list[BenchmarkExample],
        de_novo_latent_rerank_weight: float = 0.05,
        source_rerank_weight: float = 0.35,
        property_rerank_weight: float = 0.25,
        scaffold_rerank_bonus: float = 0.15,
        seed_count: int = 24,
        mutation_rounds: int = 1,
        candidates_per_seed: int = 8,
        novelty_bonus: float = 0.05,
        include_retrieval: bool = True,
        include_mutation_fallback: bool = True,
        max_transform_examples: int = 1200,
        scaffold_retention_bonus: float = 0.75,
    ):
        super().__init__(
            smiles,
            latents,
            train_examples=train_examples,
            de_novo_latent_rerank_weight=de_novo_latent_rerank_weight,
            source_rerank_weight=source_rerank_weight,
            property_rerank_weight=property_rerank_weight,
            scaffold_rerank_bonus=scaffold_rerank_bonus,
            seed_count=seed_count,
            mutation_rounds=mutation_rounds,
            candidates_per_seed=candidates_per_seed,
            novelty_bonus=novelty_bonus,
            include_retrieval=include_retrieval,
            include_mutation_fallback=include_mutation_fallback,
            max_transform_examples=max_transform_examples,
        )
        self.scaffold_retention_bonus = float(scaffold_retention_bonus)

    def _generate_candidates(
        self,
        seeds: list[str],
        example: BenchmarkExample | None,
        source: str | None,
        pred_latent: np.ndarray | None = None,
    ) -> list[tuple[str, str]]:
        if not source or not example or example.task_type == TaskType.DE_NOVO:
            return super()._generate_candidates(seeds, example, source, pred_latent=pred_latent)

        generated: dict[str, str] = {}
        for transform in self._select_transforms(example, source=source):
            for smiles in _apply_transform(source, transform):
                if smiles and smiles != source and source_core_retained(source, smiles):
                    generated.setdefault(smiles, self.transform_origin)

        if not generated and self.include_mutation_fallback:
            for smiles in self._generate([source], example, source):
                if smiles != source and source_core_retained(source, smiles):
                    generated.setdefault(smiles, "generated_mutation")

        ranked = _rank_generation_pool(generated.keys(), example, source)
        return [(smiles, generated[smiles]) for smiles in ranked]

    def _score(
        self,
        smiles: str,
        pred_latent: np.ndarray,
        source: str | None,
        example: BenchmarkExample | None,
    ) -> float:
        score = super()._score(smiles, pred_latent, source, example)
        if source and example and example.task_type in {TaskType.EDIT, TaskType.INPAINT, TaskType.FRAGMENT_GROW}:
            if source_core_retained(source, smiles):
                score += self.scaffold_retention_bonus
            else:
                score -= self.scaffold_retention_bonus
        return score


class PropertyConditionedTransformDecoder(ScaffoldPreservingTransformDecoder):
    """Prefer learned transforms whose property delta matches the requested edit."""

    transform_origin = "property_conditioned_transform"

    def _select_transforms(
        self,
        example: BenchmarkExample | None,
        source: str | None = None,
        pred_latent: np.ndarray | None = None,
    ) -> list[LearnedTransform]:
        if not self.transforms:
            return []
        desired = desired_property_delta(source, example)
        if not desired:
            return super()._select_transforms(example, source=source, pred_latent=pred_latent)
        task_type = example.task_type.value if example else ""

        def sort_key(transform: LearnedTransform) -> tuple[object, ...]:
            delta_mae = _normalized_delta_mae(transform.property_delta(), desired)
            return (
                transform.task_type != task_type,
                delta_mae,
                -transform.support,
                transform.kind,
                transform.fragment_smiles,
                transform.source_token,
                transform.target_token,
            )

        return sorted(self.transforms, key=sort_key)[: self.seed_count]

    def _score(
        self,
        smiles: str,
        pred_latent: np.ndarray,
        source: str | None,
        example: BenchmarkExample | None,
    ) -> float:
        score = super()._score(smiles, pred_latent, source, example)
        desired = desired_property_delta(source, example)
        if desired:
            score += 0.75 * property_delta_match_score(source, smiles, example)
        return float(score)


class LatentConditionedTransformBeamDecoder(PropertyConditionedTransformDecoder):
    """Expand learned source edits with a latent-conditioned beam search."""

    transform_origin = "latent_beam_transform"

    def _generate_candidates(
        self,
        seeds: list[str],
        example: BenchmarkExample | None,
        source: str | None,
        pred_latent: np.ndarray | None = None,
    ) -> list[tuple[str, str]]:
        if pred_latent is None or not source or not example or example.task_type == TaskType.DE_NOVO:
            return super()._generate_candidates(seeds, example, source, pred_latent=pred_latent)

        transforms = self._select_transforms(example, source=source, pred_latent=pred_latent)
        if not transforms:
            return super()._generate_candidates(seeds, example, source, pred_latent=pred_latent)

        generated: dict[str, str] = {}
        best_scores: dict[str, float] = {}
        frontier = [source]
        beam_width = max(1, self.candidates_per_seed)
        for _ in range(max(1, self.mutation_rounds)):
            expanded: dict[str, float] = {}
            for base in frontier:
                for transform in transforms:
                    for smiles in _apply_transform(base, transform):
                        if not smiles or smiles == source or not source_core_retained(source, smiles):
                            continue
                        score = self._beam_score(smiles, pred_latent, source, example)
                        if score > expanded.get(smiles, float("-inf")):
                            expanded[smiles] = score
                        if score > best_scores.get(smiles, float("-inf")):
                            best_scores[smiles] = score
                            generated[smiles] = self.transform_origin
            frontier = [smiles for smiles, _ in sorted(expanded.items(), key=lambda item: (-item[1], item[0]))[:beam_width]]
            if not frontier:
                break

        if not generated and self.include_mutation_fallback:
            for smiles in self._generate([source], example, source):
                if smiles != source and source_core_retained(source, smiles):
                    generated.setdefault(smiles, "generated_mutation")

        ranked = sorted(generated.keys(), key=lambda smiles: (-best_scores.get(smiles, self._score(smiles, pred_latent, source, example)), smiles))
        return [(smiles, generated[smiles]) for smiles in ranked]

    def _score(
        self,
        smiles: str,
        pred_latent: np.ndarray,
        source: str | None,
        example: BenchmarkExample | None,
    ) -> float:
        if source and example and example.task_type in {TaskType.EDIT, TaskType.INPAINT, TaskType.FRAGMENT_GROW}:
            return self._beam_score(smiles, pred_latent, source, example)
        return super()._score(smiles, pred_latent, source, example)

    def _beam_score(
        self,
        smiles: str,
        pred_latent: np.ndarray,
        source: str,
        example: BenchmarkExample,
    ) -> float:
        candidate_latent = molecule_latent(smiles, self.latent_dim)
        pred_unit = _unit_vector(pred_latent)
        candidate_unit = _unit_vector(candidate_latent)
        latent_score = float(_cosine_similarity(pred_unit.reshape(1, -1), candidate_unit.reshape(1, -1))[0, 0])
        latent_distance = float(np.linalg.norm(pred_unit - candidate_unit))
        score = 2.0 * latent_score - latent_distance
        score += 0.35 * tanimoto(source, smiles)

        targets = parse_property_targets(example.instruction)
        if targets:
            desc = molecular_descriptors(smiles).descriptors
            score += 0.25 * property_match_score(desc, targets)

        desired = desired_property_delta(source, example)
        if desired:
            score += 0.75 * property_delta_match_score(source, smiles, example)

        if source_core_retained(source, smiles):
            score += self.scaffold_retention_bonus
        else:
            score -= self.scaffold_retention_bonus

        if smiles not in self.train_smiles:
            score += self.novelty_bonus
        return float(score)


class EditPolicyTransformDecoder(LatentConditionedTransformBeamDecoder):
    """Use supervised source-target edit examples to rank transform actions."""

    transform_origin = "edit_policy_transform"

    def __init__(
        self,
        smiles: list[str],
        latents: np.ndarray,
        train_examples: list[BenchmarkExample],
        de_novo_latent_rerank_weight: float = 0.05,
        source_rerank_weight: float = 0.35,
        property_rerank_weight: float = 0.25,
        scaffold_rerank_bonus: float = 0.15,
        seed_count: int = 24,
        mutation_rounds: int = 1,
        candidates_per_seed: int = 8,
        novelty_bonus: float = 0.05,
        include_retrieval: bool = True,
        include_mutation_fallback: bool = True,
        max_transform_examples: int = 1200,
        scaffold_retention_bonus: float = 0.75,
    ):
        super().__init__(
            smiles,
            latents,
            train_examples=train_examples,
            de_novo_latent_rerank_weight=de_novo_latent_rerank_weight,
            source_rerank_weight=source_rerank_weight,
            property_rerank_weight=property_rerank_weight,
            scaffold_rerank_bonus=scaffold_rerank_bonus,
            seed_count=seed_count,
            mutation_rounds=mutation_rounds,
            candidates_per_seed=candidates_per_seed,
            novelty_bonus=novelty_bonus,
            include_retrieval=include_retrieval,
            include_mutation_fallback=include_mutation_fallback,
            max_transform_examples=max_transform_examples,
            scaffold_retention_bonus=scaffold_retention_bonus,
        )
        self.transform_policy = _learn_transform_policy(train_examples, self.transforms, self.latent_dim)

    def _select_transforms(
        self,
        example: BenchmarkExample | None,
        source: str | None = None,
        pred_latent: np.ndarray | None = None,
    ) -> list[LearnedTransform]:
        if not self.transforms or not source or example is None or example.task_type == TaskType.DE_NOVO:
            return super()._select_transforms(example, source=source, pred_latent=pred_latent)

        query = _edit_policy_feature(example, source, pred_latent, self.latent_dim)
        desired = desired_property_delta(source, example)
        task_type = example.task_type.value

        def score(transform: LearnedTransform) -> float:
            prototype = self.transform_policy.get(transform.signature())
            policy_score = float(np.dot(query, prototype)) if prototype is not None else 0.0
            delta_score = 0.0
            if desired:
                delta_score = max(0.0, 1.0 - _normalized_delta_mae(transform.property_delta(), desired))
            task_score = 1.0 if transform.task_type == task_type else 0.0
            support_score = float(np.log1p(max(0, transform.support)))
            return 2.0 * policy_score + 0.75 * delta_score + 0.35 * task_score + 0.10 * support_score

        ranked = sorted(
            self.transforms,
            key=lambda transform: (
                -score(transform),
                transform.task_type != task_type,
                transform.kind,
                transform.fragment_smiles,
                transform.source_token,
                transform.target_token,
            ),
        )
        return ranked[: self.seed_count]


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    return arr / norm if norm > 0 else arr


def _mutate_smiles(smiles: str, limit: int, salt: str) -> list[str]:
    if RDKIT_AVAILABLE:
        variants = _rdkit_mutations(smiles)
    else:
        variants = _fallback_mutations(smiles)
    return _stable_select(variants, limit=limit, salt=salt)


def _learn_transform_library(
    examples: list[BenchmarkExample],
    max_transforms: int,
    max_examples: int,
) -> list[LearnedTransform]:
    counts: dict[tuple[object, ...], LearnedTransform] = {}
    processed = 0
    for example in examples:
        if not example.source_smiles:
            continue
        source = canonicalize_smiles(example.source_smiles)
        target = canonicalize_smiles(example.target_smiles)
        if not source or not target or source == target:
            continue
        processed += 1
        transforms = _extract_transforms(source, target, example.task_type.value)
        for transform in transforms:
            current = counts.get(transform.signature())
            if current is None:
                counts[transform.signature()] = transform
            else:
                counts[transform.signature()] = _merge_transform(current, transform)
        if len(counts) >= max_transforms or processed >= max_examples:
            break
    return sorted(counts.values(), key=lambda item: (-item.support, item.task_type, item.kind, item.fragment_smiles, item.source_token))[:max_transforms]


def _learn_transform_policy(
    examples: list[BenchmarkExample],
    transforms: list[LearnedTransform],
    latent_dim: int,
) -> dict[tuple[object, ...], np.ndarray]:
    signatures = {transform.signature() for transform in transforms}
    sums: dict[tuple[object, ...], np.ndarray] = {}
    counts: dict[tuple[object, ...], int] = {}
    for example in examples:
        source = canonicalize_smiles(example.source_smiles)
        target = canonicalize_smiles(example.target_smiles)
        if not source or not target or source == target:
            continue
        feature = _edit_policy_feature(example, source, molecule_latent(target, latent_dim), latent_dim)
        for transform in _extract_transforms(source, target, example.task_type.value):
            signature = transform.signature()
            if signature not in signatures:
                continue
            sums[signature] = sums.get(signature, np.zeros(latent_dim, dtype=np.float32)) + feature
            counts[signature] = counts.get(signature, 0) + 1
    return {signature: _unit_vector(total / max(1, counts[signature])) for signature, total in sums.items()}


def _edit_policy_feature(
    example: BenchmarkExample,
    source: str | None,
    pred_latent: np.ndarray | None,
    latent_dim: int,
) -> np.ndarray:
    pred = _unit_vector(pred_latent if pred_latent is not None else np.zeros(latent_dim, dtype=np.float32))
    source_vec = molecule_latent(source, latent_dim)
    task_vec = stable_hash_vector(example.task_type.value, latent_dim, salt="edit-policy-task")
    instruction_vec = stable_hash_vector(example.instruction, latent_dim, salt="edit-policy-instruction")
    delta_vec = _property_delta_vector(desired_property_delta(source, example), latent_dim)
    return _unit_vector(0.40 * pred + 0.20 * source_vec + 0.20 * delta_vec + 0.12 * task_vec + 0.08 * instruction_vec)


def _property_delta_vector(delta: Mapping[str, float], latent_dim: int) -> np.ndarray:
    vec = np.zeros(latent_dim, dtype=np.float32)
    for idx, key in enumerate(PROPERTY_KEYS):
        if idx >= latent_dim:
            break
        if key in delta:
            vec[idx] = float(delta[key]) / PROPERTY_TOLERANCES[key]
    return _unit_vector(vec)


def _extract_transforms(source: str, target: str, task_type: str) -> list[LearnedTransform]:
    delta = descriptor_delta(source, target)
    transforms: list[LearnedTransform] = []
    if RDKIT_AVAILABLE and rdFMCS is not None:
        transforms.extend(_extract_rdkit_transforms(source, target, task_type))
    transforms.extend(_extract_fallback_transforms(source, target, task_type))
    unique = {transform.signature(): transform for transform in transforms}
    return [_with_transform_delta(transform, delta) for transform in unique.values()]


def _with_transform_delta(transform: LearnedTransform, delta: Mapping[str, float]) -> LearnedTransform:
    values = {field: float(delta.get(key, 0.0)) for key, field in _DELTA_FIELDS.items()}
    return replace(transform, **values)


def _merge_transform(current: LearnedTransform, transform: LearnedTransform) -> LearnedTransform:
    support = current.support + transform.support
    values = {}
    for key, field in _DELTA_FIELDS.items():
        current_value = float(getattr(current, field))
        transform_value = float(getattr(transform, field))
        values[field] = (current_value * current.support + transform_value * transform.support) / max(1, support)
    return replace(current, support=support, **values)


def _extract_rdkit_transforms(source: str, target: str, task_type: str) -> list[LearnedTransform]:
    source_mol = Chem.MolFromSmiles(source) if Chem is not None else None
    target_mol = Chem.MolFromSmiles(target) if Chem is not None else None
    if source_mol is None or target_mol is None:
        return []
    try:
        mcs = rdFMCS.FindMCS(
            [source_mol, target_mol],
            timeout=1,
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
        )
    except Exception:
        return []
    if not mcs.smartsString:
        return []
    pattern = Chem.MolFromSmarts(mcs.smartsString)
    if pattern is None:
        return []
    source_match = source_mol.GetSubstructMatch(pattern)
    target_match = target_mol.GetSubstructMatch(pattern)
    if len(source_match) < 2 or len(target_match) < 2:
        return []
    min_atoms = max(1, min(source_mol.GetNumAtoms(), target_mol.GetNumAtoms()))
    if len(source_match) / min_atoms < 0.3:
        return []

    transforms: list[LearnedTransform] = []
    target_core = set(target_match)
    for anchor_idx, root_idx, component, bond_type in _side_components(target_mol, target_core):
        fragment = _component_fragment_smiles(target_mol, component, root_idx)
        if not fragment:
            continue
        anchor = target_mol.GetAtomWithIdx(anchor_idx)
        root = target_mol.GetAtomWithIdx(root_idx)
        transforms.append(
            LearnedTransform(
                kind="attach_fragment",
                task_type=task_type,
                anchor_atomic_num=int(anchor.GetAtomicNum()),
                anchor_is_aromatic=bool(anchor.GetIsAromatic()),
                root_atomic_num=int(root.GetAtomicNum()),
                fragment_smiles=fragment,
                bond_type=str(bond_type),
                component_size=len(component),
            )
        )

    source_core = set(source_match)
    for anchor_idx, root_idx, component, bond_type in _side_components(source_mol, source_core):
        if len(component) > 8:
            continue
        anchor = source_mol.GetAtomWithIdx(anchor_idx)
        root = source_mol.GetAtomWithIdx(root_idx)
        transforms.append(
            LearnedTransform(
                kind="remove_terminal",
                task_type=task_type,
                anchor_atomic_num=int(anchor.GetAtomicNum()),
                anchor_is_aromatic=bool(anchor.GetIsAromatic()),
                root_atomic_num=int(root.GetAtomicNum()),
                bond_type=str(bond_type),
                component_size=len(component),
            )
        )
    return transforms


def _side_components(mol, core_atoms: set[int]) -> list[tuple[int, int, set[int], object]]:
    components: list[tuple[int, int, set[int], object]] = []
    seen_roots: set[int] = set()
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if (begin in core_atoms) == (end in core_atoms):
            continue
        anchor_idx = begin if begin in core_atoms else end
        root_idx = end if begin in core_atoms else begin
        if root_idx in seen_roots:
            continue
        component = _collect_side_component(mol, root_idx, core_atoms)
        seen_roots.update(component)
        if component:
            components.append((anchor_idx, root_idx, component, bond.GetBondType()))
    return components


def _collect_side_component(mol, root_idx: int, core_atoms: set[int]) -> set[int]:
    stack = [root_idx]
    component: set[int] = set()
    while stack:
        atom_idx = stack.pop()
        if atom_idx in component or atom_idx in core_atoms:
            continue
        component.add(atom_idx)
        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if neighbor_idx not in component and neighbor_idx not in core_atoms:
                stack.append(neighbor_idx)
    return component


def _component_fragment_smiles(mol, component: set[int], root_idx: int) -> str:
    try:
        rw = Chem.RWMol()
        ordered = [root_idx] + sorted(idx for idx in component if idx != root_idx)
        old_to_new: dict[int, int] = {}
        for old_idx in ordered:
            atom = mol.GetAtomWithIdx(old_idx)
            new_atom = Chem.Atom(atom.GetAtomicNum())
            new_atom.SetFormalCharge(atom.GetFormalCharge())
            new_atom.SetIsAromatic(atom.GetIsAromatic())
            new_idx = rw.AddAtom(new_atom)
            old_to_new[old_idx] = new_idx
        for bond in mol.GetBonds():
            begin = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            if begin in old_to_new and end in old_to_new:
                rw.AddBond(old_to_new[begin], old_to_new[end], bond.GetBondType())
        frag = rw.GetMol()
        frag.GetAtomWithIdx(old_to_new[root_idx]).SetAtomMapNum(1)
        Chem.SanitizeMol(frag)
        return Chem.MolToSmiles(frag, canonical=True)
    except Exception:
        return ""


def _extract_fallback_transforms(source: str, target: str, task_type: str) -> list[LearnedTransform]:
    for idx, (left, right) in enumerate(zip(source, target)):
        if left != right:
            return [LearnedTransform(kind="replace_token", task_type=task_type, source_token=left, target_token=right, component_size=1)]
    if target.startswith(source) and len(target) > len(source):
        return [LearnedTransform(kind="append_token", task_type=task_type, target_token=target[len(source) :], component_size=len(target) - len(source))]
    if source.startswith(target) and len(source) > len(target):
        return [LearnedTransform(kind="trim_suffix", task_type=task_type, source_token=source[len(target) :], component_size=len(source) - len(target))]
    return []


def _apply_transform(smiles: str, transform: LearnedTransform) -> list[str]:
    if RDKIT_AVAILABLE and transform.kind in {"attach_fragment", "remove_terminal"}:
        if transform.kind == "attach_fragment":
            return _apply_attachment_transform(smiles, transform)
        if transform.kind == "remove_terminal":
            return _apply_removal_transform(smiles, transform)
    return _apply_fallback_transform(smiles, transform)


def _apply_attachment_transform(smiles: str, transform: LearnedTransform) -> list[str]:
    base = Chem.MolFromSmiles(smiles) if Chem is not None else None
    frag = Chem.MolFromSmiles(transform.fragment_smiles) if Chem is not None else None
    if base is None or frag is None:
        return []
    root_idx = next((atom.GetIdx() for atom in frag.GetAtoms() if atom.GetAtomMapNum() == 1), None)
    if root_idx is None:
        return []
    out: set[str] = set()
    base_atoms = base.GetNumAtoms()
    for atom in base.GetAtoms():
        if atom.GetAtomicNum() != transform.anchor_atomic_num:
            continue
        if atom.GetIsAromatic() != transform.anchor_is_aromatic:
            continue
        if atom.GetTotalNumHs() <= 0:
            continue
        try:
            combined = Chem.CombineMols(base, frag)
            rw = Chem.RWMol(combined)
            rw.GetAtomWithIdx(base_atoms + root_idx).SetAtomMapNum(0)
            rw.AddBond(atom.GetIdx(), base_atoms + root_idx, _bond_type_from_name(transform.bond_type))
            mol = rw.GetMol()
            Chem.SanitizeMol(mol)
            out.add(Chem.MolToSmiles(mol, canonical=True))
        except Exception:
            continue
    return sorted(out)


def _apply_removal_transform(smiles: str, transform: LearnedTransform) -> list[str]:
    mol = Chem.MolFromSmiles(smiles) if Chem is not None else None
    if mol is None:
        return []
    out: set[str] = set()
    for bond in mol.GetBonds():
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()
        pairs = ((begin_atom, end_atom), (end_atom, begin_atom))
        for anchor, root in pairs:
            if anchor.GetAtomicNum() != transform.anchor_atomic_num:
                continue
            if anchor.GetIsAromatic() != transform.anchor_is_aromatic:
                continue
            if root.GetAtomicNum() != transform.root_atomic_num:
                continue
            if root.IsInRing():
                continue
            component = _collect_side_component(mol, root.GetIdx(), {anchor.GetIdx()})
            if not component or len(component) > max(1, transform.component_size + 2):
                continue
            try:
                rw = Chem.RWMol(mol)
                for atom_idx in sorted(component, reverse=True):
                    rw.RemoveAtom(atom_idx)
                edited = rw.GetMol()
                Chem.SanitizeMol(edited)
                out.add(Chem.MolToSmiles(edited, canonical=True))
            except Exception:
                continue
    return sorted(out)


def _bond_type_from_name(name: str):
    text = str(name).upper()
    if "DOUBLE" in text:
        return Chem.BondType.DOUBLE
    if "TRIPLE" in text:
        return Chem.BondType.TRIPLE
    if "AROMATIC" in text:
        return Chem.BondType.AROMATIC
    return Chem.BondType.SINGLE


def _apply_fallback_transform(smiles: str, transform: LearnedTransform) -> list[str]:
    text = str(smiles)
    if transform.kind == "replace_token" and transform.source_token in text:
        return _unique_valid([text.replace(transform.source_token, transform.target_token, 1)])
    if transform.kind == "append_token" and transform.target_token:
        return _unique_valid([text + transform.target_token])
    if transform.kind == "trim_suffix" and transform.source_token and text.endswith(transform.source_token):
        return _unique_valid([text[: -len(transform.source_token)]])
    return []


def desired_property_delta(source_smiles: str | None, example: BenchmarkExample | None) -> dict[str, float]:
    """Return requested target-property movement relative to the source."""

    source = canonicalize_smiles(source_smiles)
    if not source or example is None:
        return {}
    targets = parse_property_targets(example.instruction)
    if not targets:
        return {}
    source_desc = molecular_descriptors(source).descriptors
    return {key: float(targets[key]) - float(source_desc.get(key, 0.0)) for key in PROPERTY_KEYS if key in targets}


def property_delta_mae(source_smiles: str | None, candidate_smiles: str | None, example: BenchmarkExample | None) -> float:
    desired = desired_property_delta(source_smiles, example)
    source = canonicalize_smiles(source_smiles)
    candidate = canonicalize_smiles(candidate_smiles)
    if not desired or not source or not candidate:
        return 0.0
    actual = descriptor_delta(source, candidate)
    return _normalized_delta_mae(actual, desired)


def property_delta_match_score(source_smiles: str | None, candidate_smiles: str | None, example: BenchmarkExample | None) -> float:
    desired = desired_property_delta(source_smiles, example)
    if not desired:
        return 0.0
    return max(0.0, 1.0 - property_delta_mae(source_smiles, candidate_smiles, example))


def _normalized_delta_mae(actual: Mapping[str, float], desired: Mapping[str, float]) -> float:
    errors = []
    for key in PROPERTY_KEYS:
        if key not in desired:
            continue
        errors.append(abs(float(actual.get(key, 0.0)) - float(desired[key])) / PROPERTY_TOLERANCES[key])
    return float(sum(errors) / len(errors)) if errors else 0.0


def source_core_retained(source_smiles: str | None, candidate_smiles: str | None) -> bool:
    source = canonicalize_smiles(source_smiles)
    candidate = canonicalize_smiles(candidate_smiles)
    if not source or not candidate:
        return False
    if source == candidate:
        return True

    source_scaffold = scaffold_key(source)
    candidate_scaffold = scaffold_key(candidate)
    if source_scaffold and candidate_scaffold:
        if source_scaffold == candidate_scaffold:
            return True
        if _is_substructure(source_scaffold, candidate):
            return True

    return tanimoto(source, candidate) >= 0.30


def _is_substructure(query_smiles: str, candidate_smiles: str) -> bool:
    if not RDKIT_AVAILABLE or Chem is None:
        return False
    query = Chem.MolFromSmiles(query_smiles)
    candidate = Chem.MolFromSmiles(candidate_smiles)
    if query is None or candidate is None:
        return False
    return bool(candidate.HasSubstructMatch(query))


def _rdkit_mutations(smiles: str) -> set[str]:
    mol = Chem.MolFromSmiles(smiles) if Chem is not None else None
    if mol is None:
        return set()
    variants: set[str] = set()
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        if atom.GetTotalNumHs() > 0:
            for symbol in ("C", "O", "N", "F", "Cl"):
                _add_atom_variant(mol, atom_idx, symbol, variants)
        if not atom.GetIsAromatic() and atom.GetAtomicNum() in {6, 7, 8, 16}:
            for atomic_num in (6, 7, 8, 16):
                if atomic_num != atom.GetAtomicNum():
                    _replace_atom_variant(mol, atom_idx, atomic_num, variants)
        if atom.GetDegree() == 1 and not atom.IsInRing() and mol.GetNumAtoms() > 2:
            _remove_atom_variant(mol, atom_idx, variants)
    return variants


def _add_atom_variant(mol, atom_idx: int, symbol: str, variants: set[str]) -> None:
    rw = Chem.RWMol(mol)
    new_idx = rw.AddAtom(Chem.Atom(symbol))
    rw.AddBond(atom_idx, new_idx, Chem.BondType.SINGLE)
    _add_sanitized(rw, variants)


def _replace_atom_variant(mol, atom_idx: int, atomic_num: int, variants: set[str]) -> None:
    rw = Chem.RWMol(mol)
    rw.GetAtomWithIdx(atom_idx).SetAtomicNum(atomic_num)
    _add_sanitized(rw, variants)


def _remove_atom_variant(mol, atom_idx: int, variants: set[str]) -> None:
    rw = Chem.RWMol(mol)
    rw.RemoveAtom(atom_idx)
    _add_sanitized(rw, variants)


def _add_sanitized(rw_mol, variants: set[str]) -> None:
    try:
        mol = rw_mol.GetMol()
        Chem.SanitizeMol(mol)
        smiles = Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return
    if smiles:
        variants.add(smiles)


def _fallback_mutations(smiles: str) -> set[str]:
    text = smiles.strip()
    if not text:
        return set()
    variants = {text + suffix for suffix in ("C", "O", "N", "F", "Cl")}
    if len(text) > 2:
        variants.add(text[:-1])
    replacements = {"O": "N", "N": "O", "C": "N"}
    for old, new in replacements.items():
        if old in text:
            variants.add(text.replace(old, new, 1))
    return {variant for variant in variants if canonicalize_smiles(variant)}


def _rank_generation_pool(
    smiles_values: Iterable[str],
    example: BenchmarkExample | None,
    source: str | None,
) -> list[str]:
    targets = parse_property_targets(example.instruction) if example else {}
    source_scaffold = scaffold_key(source) if source else ""

    def score(smiles: str) -> tuple[float, str]:
        value = 0.0
        if targets:
            value += property_match_score(molecular_descriptors(smiles).descriptors, targets)
        if source:
            value += 0.25 * tanimoto(source, smiles)
        if example and source_scaffold and _asks_to_preserve_source(example) and scaffold_key(smiles) == source_scaffold:
            value += 0.25
        return (-value, smiles)

    return sorted(_unique_valid(smiles_values), key=score)


def _stable_select(smiles_values: Iterable[str], limit: int, salt: str) -> list[str]:
    values = _unique_valid(smiles_values)
    return sorted(values, key=lambda smiles: (_stable_hash(smiles, salt), smiles))[:limit]


def _unique_valid(smiles_values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for smiles in smiles_values:
        canonical = canonicalize_smiles(smiles)
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        out.append(canonical)
    return out


def _stable_hash(text: str, salt: str) -> int:
    digest = hashlib.sha256(f"{salt}:{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")
