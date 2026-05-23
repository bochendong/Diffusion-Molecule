"""Small chemistry helpers with deterministic fallbacks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

RDKIT_AVAILABLE = False
Chem = None
Crippen = None
Descriptors = None
Draw = None
Lipinski = None
QED = None
DataStructs = None
rdMolDescriptors = None

try:  # pragma: no cover - depends on local/server environment.
    from rdkit import Chem as _Chem
    from rdkit import DataStructs as _DataStructs
    from rdkit.Chem import Crippen as _Crippen
    from rdkit.Chem import Descriptors as _Descriptors
    from rdkit.Chem import Draw as _Draw
    from rdkit.Chem import Lipinski as _Lipinski
    from rdkit.Chem import QED as _QED
    from rdkit.Chem import rdMolDescriptors as _rdMolDescriptors

    Chem = _Chem
    Crippen = _Crippen
    Descriptors = _Descriptors
    Draw = _Draw
    Lipinski = _Lipinski
    QED = _QED
    DataStructs = _DataStructs
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
    text = str(smiles).strip()
    if not text:
        return None
    if not RDKIT_AVAILABLE:
        return text
    mol = Chem.MolFromSmiles(text)
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
    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)
    if mol_a is None or mol_b is None:
        return 0.0
    fp_a = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol_a, 2, nBits=2048)
    fp_b = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol_b, 2, nBits=2048)
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


def morgan_fingerprint_bits(smiles: str | None, n_bits: int = 2048, radius: int = 2) -> list[float] | None:
    if not smiles or not RDKIT_AVAILABLE:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return [1.0 if bit == "1" else 0.0 for bit in fp.ToBitString()]


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
    return "".join(ch for ch in str(smiles) if ch.isalnum())[:24]


def descriptor_delta(source_smiles: str | None, target_smiles: str | None) -> dict[str, float]:
    source = molecular_descriptors(source_smiles).descriptors
    target = molecular_descriptors(target_smiles).descriptors
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
    heavy = atoms["C"] + atoms["N"] + atoms["O"] + atoms["S"] + atoms["F"] + atoms["Cl"] + atoms["Br"] + atoms["I"]
    hetero = atoms["N"] + atoms["O"] + atoms["S"]
    halogen = atoms["F"] + atoms["Cl"] + atoms["Br"] + atoms["I"]
    return {
        "MW": float(12 * atoms["C"] + 14 * atoms["N"] + 16 * atoms["O"] + 32 * atoms["S"] + 19 * atoms["F"] + 35 * atoms["Cl"] + 80 * atoms["Br"] + 127 * atoms["I"]),
        "LogP": float(0.32 * atoms["C"] + 0.22 * halogen - 0.36 * hetero),
        "QED": max(0.0, min(1.0, 0.86 - abs(heavy - 24) / 60.0)),
        "TPSA": float(12 * atoms["N"] + 17 * atoms["O"] + 25 * atoms["S"]),
        "HBD": float(min(5, atoms["N"] + atoms["O"])),
        "HBA": float(min(10, hetero)),
        "RB": float(max(0, heavy // 4 - 1)),
    }


def _fallback_string_similarity(a: str, b: str) -> float:
    grams_a = _ngrams(a)
    grams_b = _ngrams(b)
    return float(len(grams_a & grams_b) / max(1, len(grams_a | grams_b)))


def _ngrams(text: str, n: int = 2) -> set[str]:
    if len(text) <= n:
        return {text}
    return {text[idx : idx + n] for idx in range(len(text) - n + 1)}
