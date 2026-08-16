#!/usr/bin/env python3
"""Probe whether aligned B42 rewrites admit reusable closed reaction components.

This is an evidence-only representation probe.  It never trains or evaluates a
generator.  For each locked B42 source/target pair it:

1. reconstructs the MCS-aligned graph slots;
2. decomposes the exact graph difference into connected components;
3. extracts one mapped, radius-bounded RDKit reaction per component; and
4. applies the complete component tuple to the source and checks exact replay.

The important unit is the *tuple*.  Individual components are not treated as
independent molecular candidates, and a molecule is accepted only after every
component has been applied and the final graph sanitizes.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
for path in (SCRIPT_DIR, PROJECT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import categorical_graph_latent_flow as aligned_graph  # noqa: E402
import train_graph_latent_autoencoder as graph  # noqa: E402


NODE_FIELDS = (
    "atomic_number",
    "formal_charge",
    "chirality",
    "aromatic",
    "explicit_hs",
    "no_implicit",
)
EDGE_FIELDS = ("bond", "bond_stereo")


@dataclass(frozen=True)
class ComponentTemplate:
    reaction_smarts: str
    changed_slots: tuple[int, ...]
    context_slots: tuple[int, ...]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--max-atoms", type=int, default=96)
    parser.add_argument("--fingerprint-bits", type=int, default=512)
    parser.add_argument("--mcs-timeout", type=int, default=1)
    parser.add_argument("--min-common-fraction", type=float, default=0.45)
    parser.add_argument("--context-radius", type=int, default=1)
    parser.add_argument("--max-frontier", type=int, default=256)
    return parser.parse_args(argv)


def canonical(smiles: str) -> str:
    return graph.canonical_smiles(str(smiles or ""))


def read_records(path: Path, limit: int) -> list[dict[str, object]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    strict = [row for row in records if bool(row.get("strict"))]
    return strict[: max(0, int(limit))]


def aligned_pair(
    record: Mapping[str, object], args: argparse.Namespace
) -> tuple[object, object] | None:
    value = aligned_graph.align_pair(
        str(record["source_smiles"]),
        str(record["target_smiles"]),
        max_atoms=int(args.max_atoms),
        fingerprint_bits=int(args.fingerprint_bits),
        timeout=int(args.mcs_timeout),
        min_common_fraction=float(args.min_common_fraction),
    )
    if value is None:
        return None
    source, target, _common = value
    return source, target


def changed_slots(source: object, target: object) -> tuple[list[int], np.ndarray]:
    active = (source.atomic_number > 0) | (target.atomic_number > 0)
    adjacency = ((source.bond > 0) | (target.bond > 0)) & active[:, None] & active[None, :]
    changed = np.zeros_like(active, dtype=bool)
    for field in NODE_FIELDS:
        changed |= getattr(source, field) != getattr(target, field)
    for field in EDGE_FIELDS:
        different = getattr(source, field) != getattr(target, field)
        changed |= different.any(axis=0) | different.any(axis=1)
    return np.flatnonzero(changed & active).astype(int).tolist(), adjacency


def connected_components(nodes: Iterable[int], adjacency: np.ndarray) -> list[list[int]]:
    remaining = set(int(node) for node in nodes)
    components: list[list[int]] = []
    while remaining:
        stack = [min(remaining)]
        remaining.remove(stack[0])
        component: list[int] = []
        while stack:
            node = stack.pop()
            component.append(node)
            neighbours = set(np.flatnonzero(adjacency[node]).astype(int).tolist())
            unseen = sorted(neighbours & remaining, reverse=True)
            remaining.difference_update(unseen)
            stack.extend(unseen)
        components.append(sorted(component))
    return components


def expand_context(
    component: Sequence[int], adjacency: np.ndarray, radius: int
) -> list[int]:
    selected = set(int(node) for node in component)
    frontier = set(selected)
    for _ in range(max(0, int(radius))):
        neighbours = {
            int(neighbour)
            for node in frontier
            for neighbour in np.flatnonzero(adjacency[node]).tolist()
        }
        frontier = neighbours - selected
        selected.update(neighbours)
    return sorted(selected)


def graph_molecule(value: object):
    from rdkit import Chem

    active = np.flatnonzero(value.atomic_number > 0).astype(int).tolist()
    editable = Chem.RWMol()
    slot_to_atom: dict[int, int] = {}
    bond_types = {
        graph.BOND_SINGLE: Chem.BondType.SINGLE,
        graph.BOND_DOUBLE: Chem.BondType.DOUBLE,
        graph.BOND_TRIPLE: Chem.BondType.TRIPLE,
        graph.BOND_AROMATIC: Chem.BondType.AROMATIC,
    }
    for slot in active:
        atom = Chem.Atom(int(value.atomic_number[slot]))
        atom.SetFormalCharge(int(value.formal_charge[slot]) - graph.CHARGE_OFFSET)
        atom.SetIsAromatic(bool(value.aromatic[slot]))
        atom.SetNumExplicitHs(int(value.explicit_hs[slot]))
        atom.SetNoImplicit(bool(value.no_implicit[slot]))
        atom.SetAtomMapNum(int(slot) + 1)
        slot_to_atom[slot] = int(editable.AddAtom(atom))
    for offset, left in enumerate(active):
        for right in active[offset + 1 :]:
            bond_value = int(value.bond[left, right])
            if bond_value == graph.BOND_NONE:
                continue
            editable.AddBond(slot_to_atom[left], slot_to_atom[right], bond_types[bond_value])
    molecule = editable.GetMol()
    Chem.SanitizeMol(molecule)
    return molecule, slot_to_atom


def fragment_smarts(molecule, slot_to_atom: Mapping[int, int], slots: Sequence[int]) -> str:
    from rdkit import Chem

    atoms = sorted(slot_to_atom[slot] for slot in slots if slot in slot_to_atom)
    atom_set = set(atoms)
    bonds = sorted(
        bond.GetIdx()
        for bond in molecule.GetBonds()
        if bond.GetBeginAtomIdx() in atom_set and bond.GetEndAtomIdx() in atom_set
    )
    if not atoms:
        return ""
    return Chem.MolFragmentToSmarts(
        molecule,
        atomsToUse=atoms,
        bondsToUse=bonds,
        isomericSmarts=True,
    )


def extract_templates(
    source: object, target: object, context_radius: int
) -> list[ComponentTemplate]:
    from rdkit.Chem import rdChemReactions

    changed, adjacency = changed_slots(source, target)
    if not changed:
        return []
    source_mol, source_slots = graph_molecule(source)
    target_mol, target_slots = graph_molecule(target)
    templates: list[ComponentTemplate] = []
    for component in connected_components(changed, adjacency):
        context = expand_context(component, adjacency, context_radius)
        reactant = fragment_smarts(source_mol, source_slots, context)
        product = fragment_smarts(target_mol, target_slots, context)
        if not reactant or not product:
            return []
        smarts = f"{reactant}>>{product}"
        reaction = rdChemReactions.ReactionFromSmarts(smarts)
        if reaction is None or reaction.GetNumReactantTemplates() != 1:
            return []
        reaction.Initialize()
        templates.append(
            ComponentTemplate(
                reaction_smarts=smarts,
                changed_slots=tuple(component),
                context_slots=tuple(context),
            )
        )
    return templates


def clear_atom_maps(molecule) -> None:
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)


def apply_component_tuple(
    source_smiles: str,
    templates: Sequence[ComponentTemplate],
    *,
    max_frontier: int,
) -> tuple[set[str], int]:
    from rdkit import Chem
    from rdkit.Chem import rdChemReactions

    initial = Chem.MolFromSmiles(source_smiles)
    if initial is None:
        return set(), 0
    frontier = [initial]
    raw_products = 0
    for template in templates:
        reaction = rdChemReactions.ReactionFromSmarts(template.reaction_smarts)
        if reaction is None:
            return set(), raw_products
        by_smiles: dict[str, object] = {}
        for molecule in frontier:
            for products in reaction.RunReactants((molecule,)):
                raw_products += 1
                if len(products) != 1:
                    continue
                product = products[0]
                try:
                    clear_atom_maps(product)
                    Chem.SanitizeMol(product)
                    smiles = Chem.MolToSmiles(product, canonical=True)
                except Exception:
                    continue
                if "." not in smiles:
                    by_smiles.setdefault(smiles, product)
        frontier = [by_smiles[key] for key in sorted(by_smiles)[: int(max_frontier)]]
        if not frontier:
            break
    return {canonical(Chem.MolToSmiles(mol)) for mol in frontier}, raw_products


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    records = read_records(args.records, int(args.limit))
    rows: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    template_counts: Counter[str] = Counter()
    for index, record in enumerate(records, start=1):
        counts["requested"] += 1
        target_smiles = canonical(str(record["target_smiles"]))
        if not target_smiles or "." in target_smiles:
            counts["disconnected_or_invalid_target"] += 1
            continue
        counts["connected_target"] += 1
        pair = aligned_pair(record, args)
        if pair is None:
            counts["alignment_failed"] += 1
            continue
        source, target = pair
        counts["aligned"] += 1
        try:
            templates = extract_templates(source, target, int(args.context_radius))
        except Exception as error:
            counts["extraction_error"] += 1
            rows.append({"index": index - 1, "error": repr(error)})
            continue
        if not templates:
            counts["template_empty"] += 1
            rows.append(
                {
                    "index": index - 1,
                    "task": record.get("task"),
                    "source_smiles": record["source_smiles"],
                    "target_smiles": record["target_smiles"],
                    "components": 0,
                    "exact_replay": False,
                    "error": "empty_component_template",
                }
            )
            continue
        counts["templated"] += 1
        counts["components"] += len(templates)
        for template in templates:
            template_counts[template.reaction_smarts] += 1
        products, raw_products = apply_component_tuple(
            str(record["source_smiles"]),
            templates,
            max_frontier=int(args.max_frontier),
        )
        exact = target_smiles in products
        counts["valid_replay"] += int(bool(products))
        counts["exact_replay"] += int(exact)
        rows.append(
            {
                "index": index - 1,
                "task": record.get("task"),
                "source_smiles": record["source_smiles"],
                "target_smiles": record["target_smiles"],
                "components": len(templates),
                "context_radius": int(args.context_radius),
                "raw_products": raw_products,
                "valid_products": len(products),
                "exact_replay": exact,
                "templates": [template.reaction_smarts for template in templates],
            }
        )
        print(
            json.dumps(
                {
                    "stage": "closed_reaction_template_probe",
                    "pairs": index,
                    "templated": counts["templated"],
                    "exact": counts["exact_replay"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    summary = {
        "protocol": "compositional_closed_reaction_template_probe_v1",
        "records_path": str(args.records),
        "requested": int(counts["requested"]),
        "connected_target": int(counts["connected_target"]),
        "connected_target_rate": counts["connected_target"]
        / max(1, counts["requested"]),
        "aligned": int(counts["aligned"]),
        "templated": int(counts["templated"]),
        "alignment_rate": counts["aligned"] / max(1, counts["connected_target"]),
        "template_rate": counts["templated"] / max(1, counts["aligned"]),
        "valid_replay_rate": counts["valid_replay"] / max(1, counts["templated"]),
        "exact_replay_rate": counts["exact_replay"] / max(1, counts["templated"]),
        "mean_components": counts["components"] / max(1, counts["templated"]),
        "unique_component_templates": len(template_counts),
        "reused_component_templates": sum(value > 1 for value in template_counts.values()),
        "counts": dict(counts),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
