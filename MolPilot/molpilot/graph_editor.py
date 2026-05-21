"""RDKit-backed source graph editor for local molecular edits.

This is the first MolPilot decoder that treats the source molecule as a locked
graph rather than merely a conditioning vector. It preserves the source core by
removing one side chain at a Murcko-scaffold attachment point and replacing it
with a small R-group selected from the instruction goals. When RDKit is not
available, all functions return no candidates so local smoke tests still run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .chem import (
    Chem,
    RDKIT_AVAILABLE,
    canonicalize_smiles,
    molecular_descriptors,
    scaffold_key,
    tanimoto,
)
from .schema import ObjectiveSpec


@dataclass(frozen=True)
class GraphCandidate:
    smiles: str
    origin: str


@dataclass(frozen=True)
class FragmentTemplate:
    name: str
    smiles: str


POLAR_FRAGMENTS = (
    FragmentTemplate("hydroxy", "[*:1]O"),
    FragmentTemplate("methoxy", "[*:1]OC"),
    FragmentTemplate("amine", "[*:1]N"),
    FragmentTemplate("methylamine", "[*:1]NC"),
    FragmentTemplate("amide", "[*:1]C(=O)N"),
    FragmentTemplate("acid", "[*:1]C(=O)O"),
    FragmentTemplate("sulfonamide", "[*:1]S(=O)(=O)N"),
)

HYDROPHOBIC_FRAGMENTS = (
    FragmentTemplate("methyl", "[*:1]C"),
    FragmentTemplate("ethyl", "[*:1]CC"),
    FragmentTemplate("fluoro", "[*:1]F"),
    FragmentTemplate("chloro", "[*:1]Cl"),
    FragmentTemplate("bromo", "[*:1]Br"),
    FragmentTemplate("trifluoromethyl", "[*:1]C(F)(F)F"),
)

BALANCED_FRAGMENTS = (
    FragmentTemplate("methyl", "[*:1]C"),
    FragmentTemplate("fluoro", "[*:1]F"),
    FragmentTemplate("chloro", "[*:1]Cl"),
    FragmentTemplate("methoxy", "[*:1]OC"),
    FragmentTemplate("nitrile", "[*:1]C#N"),
    FragmentTemplate("amide", "[*:1]C(=O)N"),
)


def generate_graph_edit_candidates(source_smiles: str | None, spec: ObjectiveSpec, limit: int = 96) -> list[GraphCandidate]:
    """Generate local R-group edits while keeping the source core fixed."""

    if limit <= 0 or not RDKIT_AVAILABLE or not source_smiles:
        return []
    source = canonicalize_smiles(source_smiles)
    mol = Chem.MolFromSmiles(source) if source else None
    if mol is None:
        return []

    fragments = _fragments_for_spec(spec)
    candidates: list[GraphCandidate] = []
    for attach_idx, sidechain_atoms in _sidechain_attachment_regions(mol):
        core, new_attach_idx = _remove_atoms(mol, sidechain_atoms, attach_idx)
        if core is None:
            continue
        _append_candidate(candidates, core, "graph_edit_remove", source)
        for fragment in fragments:
            edited = _attach_fragment(core, new_attach_idx, fragment.smiles)
            _append_candidate(candidates, edited, f"graph_edit_{fragment.name}", source)
            if len(candidates) >= limit:
                return _dedupe(candidates)[:limit]

    if not candidates:
        for attach_idx in _fallback_attachment_atoms(mol):
            for fragment in fragments:
                edited = _attach_fragment(mol, attach_idx, fragment.smiles)
                _append_candidate(candidates, edited, f"graph_grow_{fragment.name}", source)
                if len(candidates) >= limit:
                    return _dedupe(candidates)[:limit]
    return _dedupe(candidates)[:limit]


def generate_scaffold_library_candidates(
    source_smiles: str | None,
    spec: ObjectiveSpec,
    library_smiles: Iterable[str],
    limit: int = 32,
) -> list[GraphCandidate]:
    """Return same-scaffold analogs from the fitted molecule library.

    This is intentionally labeled separately from graph edits because it is a
    retrieval-like analog baseline. It is useful experimentally, but should be
    ablated when making claims about generative editing.
    """

    if limit <= 0 or not source_smiles:
        return []
    source = canonicalize_smiles(source_smiles)
    source_record = molecular_descriptors(source)
    if not source or not source_record.valid:
        return []
    source_scaffold = scaffold_key(source)
    scored: list[tuple[float, str]] = []
    for smiles in library_smiles:
        candidate = canonicalize_smiles(smiles)
        if not candidate or candidate == source:
            continue
        if scaffold_key(candidate) != source_scaffold:
            continue
        score = _property_direction_score(source_record.descriptors, candidate, spec)
        score += 0.5 * tanimoto(source, candidate)
        scored.append((score, candidate))
    scored.sort(key=lambda item: (-item[0], item[1]))
    out = [GraphCandidate(smiles, "scaffold_library") for _, smiles in scored[:limit]]
    return _dedupe(out)


def _sidechain_attachment_regions(mol) -> list[tuple[int, set[int]]]:
    scaffold_atoms = _murcko_atom_indices(mol)
    if not scaffold_atoms:
        return []
    regions: list[tuple[int, set[int]]] = []
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if begin in scaffold_atoms and end not in scaffold_atoms:
            regions.append((begin, _collect_sidechain_atoms(mol, end, scaffold_atoms)))
        elif end in scaffold_atoms and begin not in scaffold_atoms:
            regions.append((end, _collect_sidechain_atoms(mol, begin, scaffold_atoms)))
    return [(attach_idx, atoms) for attach_idx, atoms in regions if atoms]


def _murcko_atom_indices(mol) -> set[int]:
    try:
        from rdkit.Chem.Scaffolds import MurckoScaffold

        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            return set()
        matches = mol.GetSubstructMatches(scaffold)
        return set(matches[0]) if matches else set()
    except Exception:
        return set()


def _collect_sidechain_atoms(mol, start_idx: int, blocked_atoms: set[int]) -> set[int]:
    stack = [start_idx]
    visited: set[int] = set()
    while stack:
        idx = stack.pop()
        if idx in visited or idx in blocked_atoms:
            continue
        visited.add(idx)
        atom = mol.GetAtomWithIdx(idx)
        for neighbor in atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if neighbor_idx not in visited and neighbor_idx not in blocked_atoms:
                stack.append(neighbor_idx)
    return visited


def _remove_atoms(mol, remove_atoms: set[int], attach_idx: int):
    try:
        rw = Chem.RWMol(mol)
        for idx in sorted(remove_atoms, reverse=True):
            rw.RemoveAtom(int(idx))
        new_attach_idx = attach_idx - sum(1 for idx in remove_atoms if idx < attach_idx)
        edited = rw.GetMol()
        Chem.SanitizeMol(edited)
        return edited, int(new_attach_idx)
    except Exception:
        return None, -1


def _attach_fragment(core_mol, attach_idx: int, fragment_smiles: str):
    if attach_idx < 0:
        return None
    frag = Chem.MolFromSmiles(fragment_smiles)
    if frag is None:
        return None
    dummy_atoms = [atom.GetIdx() for atom in frag.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(dummy_atoms) != 1:
        return None
    dummy_idx = dummy_atoms[0]
    dummy = frag.GetAtomWithIdx(dummy_idx)
    neighbors = [atom.GetIdx() for atom in dummy.GetNeighbors()]
    if len(neighbors) != 1:
        return None
    neighbor_idx = neighbors[0]
    try:
        combo = Chem.CombineMols(core_mol, frag)
        rw = Chem.RWMol(combo)
        offset = core_mol.GetNumAtoms()
        rw.AddBond(int(attach_idx), int(offset + neighbor_idx), Chem.BondType.SINGLE)
        rw.RemoveAtom(int(offset + dummy_idx))
        edited = rw.GetMol()
        Chem.SanitizeMol(edited)
        return edited
    except Exception:
        return None


def _fallback_attachment_atoms(mol) -> list[int]:
    atoms = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 6 and atom.GetTotalNumHs() > 0:
            atoms.append(atom.GetIdx())
    return atoms[:8]


def _fragments_for_spec(spec: ObjectiveSpec) -> list[FragmentTemplate]:
    goals = set(spec.goals) | set(spec.proxy_goals)
    if {"decrease_logp", "improve_solubility_proxy"} & goals:
        return list(POLAR_FRAGMENTS + BALANCED_FRAGMENTS)
    if {"reduce_tpsa", "improve_bbb_proxy"} & goals:
        return list(HYDROPHOBIC_FRAGMENTS + BALANCED_FRAGMENTS)
    if {"improve_qed", "lower_sa"} & goals:
        return list(BALANCED_FRAGMENTS + POLAR_FRAGMENTS[:3] + HYDROPHOBIC_FRAGMENTS[:4])
    return list(BALANCED_FRAGMENTS + POLAR_FRAGMENTS[:4] + HYDROPHOBIC_FRAGMENTS[:4])


def _property_direction_score(source_desc: dict[str, float], candidate_smiles: str, spec: ObjectiveSpec) -> float:
    cand = molecular_descriptors(candidate_smiles)
    if not cand.valid:
        return -999.0
    score = 0.0
    for goal in spec.goals:
        if goal == "decrease_logp":
            score += source_desc.get("LogP", 0.0) - cand.descriptors.get("LogP", 0.0)
        elif goal == "increase_logp":
            score += cand.descriptors.get("LogP", 0.0) - source_desc.get("LogP", 0.0)
        elif goal == "reduce_tpsa":
            score += 0.05 * (source_desc.get("TPSA", 0.0) - cand.descriptors.get("TPSA", 0.0))
        elif goal == "increase_tpsa":
            score += 0.05 * (cand.descriptors.get("TPSA", 0.0) - source_desc.get("TPSA", 0.0))
        elif goal == "improve_qed":
            score += 5.0 * (cand.descriptors.get("QED", 0.0) - source_desc.get("QED", 0.0))
    if "keep_mw_similar" in spec.constraints:
        score -= 0.01 * abs(cand.descriptors.get("MW", 0.0) - source_desc.get("MW", 0.0))
    return float(score)


def _append_candidate(out: list[GraphCandidate], mol, origin: str, source_smiles: str | None) -> None:
    if mol is None:
        return
    try:
        smiles = Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return
    smiles = canonicalize_smiles(smiles)
    if smiles and smiles != source_smiles:
        out.append(GraphCandidate(smiles, origin))


def _dedupe(candidates: list[GraphCandidate]) -> list[GraphCandidate]:
    out: list[GraphCandidate] = []
    seen: dict[str, int] = {}
    for candidate in candidates:
        if candidate.smiles in seen:
            existing = out[seen[candidate.smiles]]
            origins = existing.origin.split("+")
            if candidate.origin not in origins:
                out[seen[candidate.smiles]] = GraphCandidate(existing.smiles, existing.origin + "+" + candidate.origin)
            continue
        seen[candidate.smiles] = len(out)
        out.append(candidate)
    return out
