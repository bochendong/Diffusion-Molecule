"""Small chemistry helpers with an optional RDKit backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

RDKIT_AVAILABLE = False
Chem = None
Descriptors = None
Crippen = None
Lipinski = None
QED = None
rdMolDescriptors = None
Draw = None
DataStructs = None

try:  # pragma: no cover - depends on server environment.
    from rdkit import Chem as _Chem
    from rdkit import DataStructs as _DataStructs
    from rdkit.Chem import Crippen as _Crippen
    from rdkit.Chem import Descriptors as _Descriptors
    from rdkit.Chem import Draw as _Draw
    from rdkit.Chem import Lipinski as _Lipinski
    from rdkit.Chem import QED as _QED
    from rdkit.Chem import rdMolDescriptors as _rdMolDescriptors

    Chem = _Chem
    DataStructs = _DataStructs
    Descriptors = _Descriptors
    Crippen = _Crippen
    Lipinski = _Lipinski
    QED = _QED
    Draw = _Draw
    rdMolDescriptors = _rdMolDescriptors
    RDKIT_AVAILABLE = True
except Exception:  # pragma: no cover
    pass


DESCRIPTOR_KEYS = ("MW", "LogP", "QED", "TPSA", "HBD", "HBA", "RB")


@dataclass(frozen=True)
class MoleculeRecord:
    smiles: str
    valid: bool
    descriptors: dict[str, float]


def canonicalize_smiles(smiles: str | None) -> str | None:
    if not smiles:
        return None
    smiles = str(smiles).strip()
    if not smiles:
        return None
    if not RDKIT_AVAILABLE:
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def molecular_descriptors(smiles: str | None) -> MoleculeRecord:
    can = canonicalize_smiles(smiles)
    if can is None:
        return MoleculeRecord(smiles=str(smiles or ""), valid=False, descriptors={})
    if not RDKIT_AVAILABLE:
        return MoleculeRecord(smiles=can, valid=True, descriptors=_fallback_descriptors(can))
    mol = Chem.MolFromSmiles(can)
    if mol is None:
        return MoleculeRecord(smiles=can, valid=False, descriptors={})
    desc = {
        "MW": float(Descriptors.MolWt(mol)),
        "LogP": float(Crippen.MolLogP(mol)),
        "QED": float(QED.qed(mol)),
        "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
        "HBD": float(Lipinski.NumHDonors(mol)),
        "HBA": float(Lipinski.NumHAcceptors(mol)),
        "RB": float(Lipinski.NumRotatableBonds(mol)),
    }
    return MoleculeRecord(smiles=can, valid=True, descriptors=desc)


def render_molecule_image(smiles: str, out_path: str | Path, size: tuple[int, int] = (256, 256)) -> str | None:
    if not RDKIT_AVAILABLE or Draw is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image = Draw.MolToImage(mol, size=size)
    image.save(out_path)
    return str(out_path)


def tanimoto(smiles_a: str | None, smiles_b: str | None) -> float:
    if not smiles_a or not smiles_b:
        return 0.0
    if not RDKIT_AVAILABLE:
        return _fallback_string_similarity(str(smiles_a), str(smiles_b))
    ma = Chem.MolFromSmiles(smiles_a)
    mb = Chem.MolFromSmiles(smiles_b)
    if ma is None or mb is None:
        return 0.0
    fa = rdMolDescriptors.GetMorganFingerprintAsBitVect(ma, 2, nBits=2048)
    fb = rdMolDescriptors.GetMorganFingerprintAsBitVect(mb, 2, nBits=2048)
    return float(DataStructs.TanimotoSimilarity(fa, fb))


def scaffold_key(smiles: str | None) -> str:
    if not smiles:
        return ""
    if RDKIT_AVAILABLE:
        try:
            from rdkit.Chem.Scaffolds import MurckoScaffold

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return ""
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold, canonical=True)
        except Exception:
            return ""
    return "".join(ch for ch in str(smiles) if ch.isalpha() or ch.isdigit())[:32]


def passes_druglike_proxy(descriptors: dict[str, float]) -> bool:
    if not descriptors:
        return False
    return (
        descriptors.get("MW", 999.0) <= 500.0
        and descriptors.get("LogP", 99.0) <= 5.0
        and descriptors.get("HBD", 99.0) <= 5.0
        and descriptors.get("HBA", 99.0) <= 10.0
        and descriptors.get("RB", 99.0) <= 10.0
    )


def descriptor_delta(source: dict[str, float], target: dict[str, float]) -> dict[str, float]:
    return {key: float(target.get(key, 0.0) - source.get(key, 0.0)) for key in DESCRIPTOR_KEYS}


def _fallback_descriptors(smiles: str) -> dict[str, float]:
    atoms = {
        "C": smiles.count("C") + smiles.count("c"),
        "N": smiles.count("N") + smiles.count("n"),
        "O": smiles.count("O") + smiles.count("o"),
        "S": smiles.count("S") + smiles.count("s"),
        "F": smiles.count("F"),
        "Cl": smiles.count("Cl"),
        "Br": smiles.count("Br"),
        "I": smiles.count("I"),
    }
    heavy = sum(atoms.values())
    hetero = atoms["N"] + atoms["O"] + atoms["S"]
    halogen = atoms["F"] + atoms["Cl"] + atoms["Br"] + atoms["I"]
    return {
        "MW": float(12 * atoms["C"] + 14 * atoms["N"] + 16 * atoms["O"] + 32 * atoms["S"] + 19 * atoms["F"] + 35 * atoms["Cl"] + 80 * atoms["Br"] + 127 * atoms["I"]),
        "LogP": float(0.35 * atoms["C"] + 0.25 * halogen - 0.35 * hetero),
        "QED": max(0.0, min(1.0, 0.85 - abs(heavy - 25) / 60.0)),
        "TPSA": float(12 * atoms["N"] + 17 * atoms["O"] + 25 * atoms["S"]),
        "HBD": float(min(5, atoms["N"] + atoms["O"])),
        "HBA": float(min(10, hetero)),
        "RB": float(max(0, heavy // 4 - 1)),
    }


def _fallback_string_similarity(a: str, b: str) -> float:
    sa = set(a)
    sb = set(b)
    return float(len(sa & sb) / max(1, len(sa | sb)))

