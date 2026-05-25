"""Prototype generative molecular decoder.

This decoder is intentionally small and deterministic. It augments the existing
latent retrieval seeds with local molecule mutations, giving the experiment a
candidate surface that can leave the training pool before we invest in a larger
learned graph or SELFIES decoder.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

import numpy as np

from .chem import Chem, RDKIT_AVAILABLE, canonicalize_smiles, molecular_descriptors, scaffold_key, tanimoto
from .decoder import RetrievalDecoder, _asks_to_preserve_source, _cosine_similarity
from .features import molecule_latent
from .property_guidance import parse_property_targets, property_match_score
from .schema import BenchmarkExample, Candidate, TaskType


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
            generated = self._generate(seed_smiles, example, source)
            for smiles in generated:
                if source and smiles == source:
                    continue
                score = self._score(smiles, pred_latents[row_idx], source, example)
                candidate = Candidate(smiles=smiles, origin="generated_mutation", score=score)
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


def _mutate_smiles(smiles: str, limit: int, salt: str) -> list[str]:
    if RDKIT_AVAILABLE:
        variants = _rdkit_mutations(smiles)
    else:
        variants = _fallback_mutations(smiles)
    return _stable_select(variants, limit=limit, salt=salt)


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
