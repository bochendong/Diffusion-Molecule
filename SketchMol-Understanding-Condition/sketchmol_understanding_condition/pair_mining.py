"""Mine scaffold-preserving edit pairs from molecule records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal, Optional

from .chem import canonical_smiles, morgan_tanimoto, scaffold_smiles
from .instructions import EditInstruction, property_direction_from_delta, render_instruction


Direction = Literal["increase", "decrease"]


@dataclass(frozen=True)
class MoleculeRecord:
    """A molecule and optional scalar properties used for pair mining."""

    smiles: str
    mol_id: Optional[str] = None
    properties: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class EditPair:
    """A source-target molecular edit pair."""

    source_smiles: str
    target_smiles: str
    instruction: str
    scaffold: Optional[str]
    similarity: float
    source_id: Optional[str] = None
    target_id: Optional[str] = None
    property_name: Optional[str] = None
    property_delta: Optional[float] = None


def _passes_property_delta(
    source: MoleculeRecord,
    target: MoleculeRecord,
    property_name: Optional[str],
    direction: Direction,
    min_abs_delta: float,
) -> tuple[bool, Optional[float]]:
    if property_name is None:
        return True, None
    if property_name not in source.properties or property_name not in target.properties:
        return False, None

    delta = float(target.properties[property_name]) - float(source.properties[property_name])
    if abs(delta) < min_abs_delta:
        return False, delta
    if direction == "increase":
        return delta > 0, delta
    return delta < 0, delta


def mine_scaffold_edit_pairs(
    records: Iterable[MoleculeRecord],
    *,
    property_name: Optional[str] = None,
    direction: Direction = "increase",
    min_abs_delta: float = 0.1,
    min_similarity: float = 0.25,
    max_similarity: float = 0.9,
    max_pairs: Optional[int] = None,
    max_pairs_per_scaffold: Optional[int] = None,
) -> list[EditPair]:
    """Mine source-target pairs sharing a Bemis-Murcko scaffold.

    The default similarity window avoids both unrelated pairs and near-duplicates.
    Use `property_name=None` to mine generic scaffold-preserving edit pairs.
    """

    normalized: list[tuple[MoleculeRecord, str, str]] = []
    for record in records:
        can = canonical_smiles(record.smiles)
        if can is None:
            continue
        scaffold = scaffold_smiles(can)
        if scaffold is None:
            continue
        normalized.append((record, can, scaffold))

    by_scaffold: dict[str, list[tuple[MoleculeRecord, str]]] = {}
    for record, can, scaffold in normalized:
        by_scaffold.setdefault(scaffold, []).append((record, can))

    pairs: list[EditPair] = []
    for scaffold, group in by_scaffold.items():
        if len(group) < 2:
            continue
        scaffold_pairs = 0
        for source, source_smiles in group:
            for target, target_smiles in group:
                if max_pairs_per_scaffold is not None and scaffold_pairs >= max_pairs_per_scaffold:
                    break
                if source_smiles == target_smiles:
                    continue

                ok, delta = _passes_property_delta(
                    source, target, property_name, direction, min_abs_delta
                )
                if not ok:
                    continue

                similarity = morgan_tanimoto(source_smiles, target_smiles)
                if similarity is None:
                    continue
                if not (min_similarity <= similarity <= max_similarity):
                    continue

                if property_name is None:
                    instruction = render_instruction(
                        EditInstruction(task="scaffold_preserving_edit")
                    )
                else:
                    instruction = render_instruction(
                        EditInstruction(
                            task="property_optimization",
                            property_name=property_name,
                            property_direction=property_direction_from_delta(delta or 0.0),
                        )
                    )

                pairs.append(
                    EditPair(
                        source_smiles=source_smiles,
                        target_smiles=target_smiles,
                        instruction=instruction,
                        scaffold=scaffold,
                        similarity=similarity,
                        source_id=source.mol_id,
                        target_id=target.mol_id,
                        property_name=property_name,
                        property_delta=delta,
                    )
                )
                scaffold_pairs += 1
                if max_pairs is not None and len(pairs) >= max_pairs:
                    return pairs
            if max_pairs_per_scaffold is not None and scaffold_pairs >= max_pairs_per_scaffold:
                break

    return pairs
