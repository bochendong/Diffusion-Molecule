"""Descriptor extraction with an RDKit path and a lightweight fallback.

The fallback keeps the demo runnable on machines that do not have RDKit. It is
not a substitute for chemistry-grade evaluation; it is just enough for pipeline
development and unit tests.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass

try:  # pragma: no cover - exercised only when RDKit is installed.
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

    try:
        from rdkit.Chem import rdFingerprintGenerator

        _MORGAN_FP_GEN = rdFingerprintGenerator.GetMorganGenerator(
            radius=2,
            fpSize=1024,
            useCountSimulation=False,
        )
    except Exception:  # pragma: no cover - old RDKit without rdFingerprintGenerator
        _MORGAN_FP_GEN = None

    RDKIT_AVAILABLE = True
except Exception:  # pragma: no cover - fallback is covered in this environment.
    Chem = None
    DataStructs = None
    RDLogger = None
    AllChem = None
    Crippen = None
    Descriptors = None
    Lipinski = None
    QED = None
    rdMolDescriptors = None
    FilterCatalog = None
    FilterCatalogParams = None
    _MORGAN_FP_GEN = None
    RDKIT_AVAILABLE = False


def configure_rdkit_logging(suppress: bool | None = None) -> None:
    """Toggle noisy RDKit C++ logs without changing Python exceptions."""

    if not RDKIT_AVAILABLE or RDLogger is None:
        return
    if suppress is None:
        suppress = os.environ.get("PHYSTABMOL_SUPPRESS_RDKIT_LOGS", "0").strip().lower() in {"1", "true", "yes", "on"}
    if suppress:
        RDLogger.DisableLog("rdApp.debug")
        RDLogger.DisableLog("rdApp.info")
        RDLogger.DisableLog("rdApp.warning")
        RDLogger.DisableLog("rdApp.error")
    else:
        RDLogger.EnableLog("rdApp.debug")
        RDLogger.EnableLog("rdApp.info")
        RDLogger.EnableLog("rdApp.warning")
        RDLogger.EnableLog("rdApp.error")


configure_rdkit_logging()

ATOM_WEIGHTS = {
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "S": 32.06,
    "F": 18.998,
    "Cl": 35.45,
    "Br": 79.904,
    "I": 126.904,
}

ATOM_RE = re.compile(r"Cl|Br|C|N|O|S|F|I|c|n|o|s")


@dataclass(frozen=True)
class MoleculeRecord:
    smiles: str
    descriptors: dict[str, float]
    valid: bool
    pains: str = "unknown"


def canonicalize_smiles(smiles: str) -> str | None:
    if RDKIT_AVAILABLE:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, canonical=True) if mol else None
    return smiles if bool(ATOM_RE.search(smiles)) and _balanced(smiles) and _fallback_syntax_ok(smiles) else None


def molecular_descriptors(smiles: str) -> MoleculeRecord:
    if RDKIT_AVAILABLE:
        return _rdkit_descriptors(smiles)
    return _fallback_descriptors(smiles)


def tanimoto(smiles_a: str, smiles_b: str) -> float:
    if RDKIT_AVAILABLE:
        mol_a = Chem.MolFromSmiles(smiles_a)
        mol_b = Chem.MolFromSmiles(smiles_b)
        if mol_a is None or mol_b is None:
            return 0.0
        if _MORGAN_FP_GEN is not None:
            fp_a = _MORGAN_FP_GEN.GetFingerprint(mol_a)
            fp_b = _MORGAN_FP_GEN.GetFingerprint(mol_b)
        else:
            fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, 2, nBits=1024)
            fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, 2, nBits=1024)
        return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))

    set_a = _char_ngrams(smiles_a)
    set_b = _char_ngrams(smiles_b)
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / max(1, len(set_a | set_b))


def passes_druglike_filters(desc: dict[str, float]) -> bool:
    lipinski = (
        desc["MW"] <= 500
        and desc["LogP"] <= 5
        and desc["HBD"] <= 5
        and desc["HBA"] <= 10
    )
    veber = desc["TPSA"] <= 140 and desc["RB"] <= 10
    synthesizable = desc["SA"] <= 6
    return bool(lipinski and veber and synthesizable)


def atom_counts(smiles: str) -> Counter:
    counts = Counter()
    for atom in ATOM_RE.findall(smiles):
        counts[atom.capitalize()] += 1
    return counts


def _rdkit_descriptors(smiles: str) -> MoleculeRecord:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return MoleculeRecord(smiles=smiles, descriptors={}, valid=False)
    can = Chem.MolToSmiles(mol, canonical=True)
    ring_info = mol.GetRingInfo()
    counts = Counter(atom.GetSymbol() for atom in mol.GetAtoms())
    desc = {
        "MW": float(Descriptors.MolWt(mol)),
        "LogP": float(Crippen.MolLogP(mol)),
        "QED": float(QED.qed(mol)),
        "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
        "HBD": float(Lipinski.NumHDonors(mol)),
        "HBA": float(Lipinski.NumHAcceptors(mol)),
        "RB": float(Lipinski.NumRotatableBonds(mol)),
        "SA": _synthetic_accessibility_proxy(mol),
        "ring_count": float(ring_info.NumRings()),
    }
    for atom in ATOM_WEIGHTS:
        desc[atom] = float(counts.get(atom, 0))
    desc.update(_functional_groups(can))
    desc["scaffold_class"] = float(_scaffold_class(can, int(desc["ring_count"])))
    return MoleculeRecord(smiles=can, descriptors=desc, valid=True, pains="None")


def _fallback_descriptors(smiles: str) -> MoleculeRecord:
    can = canonicalize_smiles(smiles)
    if can is None:
        return MoleculeRecord(smiles=smiles, descriptors={}, valid=False)

    counts = atom_counts(can)
    hetero = counts["N"] + counts["O"] + counts["S"]
    halogens = counts["F"] + counts["Cl"] + counts["Br"] + counts["I"]
    rings = _fallback_ring_count(can)
    mw = sum(ATOM_WEIGHTS[a] * counts[a] for a in ATOM_WEIGHTS)
    logp = 0.54 * counts["C"] + 0.35 * halogens - 1.15 * hetero - 0.01 * mw
    tpsa = 12.0 * counts["N"] + 17.0 * counts["O"] + 25.0 * counts["S"]
    hbd = min(counts["N"] + counts["O"], 6)
    hba = min(counts["N"] + counts["O"] + counts["S"], 12)
    rb = max(0, can.count("C") - 2 * rings - 4)
    sa = 1.5 + 0.04 * max(0.0, mw - 250.0) + 0.35 * rings + 0.15 * hetero
    qed = _bounded(0.95 - abs(mw - 330.0) / 500.0 - max(0.0, logp - 4.0) / 8.0 - max(0.0, tpsa - 120.0) / 180.0)
    desc = {
        "MW": float(mw),
        "LogP": float(logp),
        "QED": float(qed),
        "TPSA": float(tpsa),
        "HBD": float(hbd),
        "HBA": float(hba),
        "RB": float(rb),
        "SA": float(min(8.0, sa)),
        "ring_count": float(rings),
    }
    for atom in ATOM_WEIGHTS:
        desc[atom] = float(counts.get(atom, 0))
    desc.update(_functional_groups(can))
    desc["scaffold_class"] = float(_scaffold_class(can, rings))
    return MoleculeRecord(smiles=can, descriptors=desc, valid=True, pains="unknown")


def _synthetic_accessibility_proxy(mol) -> float:
    rings = mol.GetRingInfo().NumRings()
    heavy = mol.GetNumHeavyAtoms()
    hetero = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() not in {"C", "H"})
    return float(min(8.0, 1.2 + 0.03 * heavy + 0.25 * rings + 0.08 * hetero))


def _functional_groups(smiles: str) -> dict[str, float]:
    return {
        "fg_ester": float(smiles.count("C(=O)O") + smiles.count("O=C(O") + smiles.count("C(=O)OC")),
        "fg_amide": float(smiles.count("C(=O)N") + smiles.count("NC=O") + smiles.count("O=C(N")),
        "fg_amine": float(smiles.count("N") + smiles.count("n")),
        "fg_alcohol": float(smiles.count("CO") + smiles.count("OC")),
        "fg_halogen": float(smiles.count("F") + smiles.count("Cl") + smiles.count("Br") + smiles.count("I")),
    }


def _scaffold_class(smiles: str, rings: int) -> int:
    lower = smiles.lower()
    if rings == 0:
        return 0
    if "n" in lower:
        return 2
    if "s" in lower or "o" in lower:
        return 3
    if rings >= 2:
        return 4
    return 1


def _fallback_ring_count(smiles: str) -> int:
    digits = re.findall(r"[1-9]", smiles)
    return min(5, len(digits) // 2)


def _balanced(smiles: str) -> bool:
    return smiles.count("(") == smiles.count(")") and smiles.count("[") == smiles.count("]")


def _fallback_syntax_ok(smiles: str) -> bool:
    if "==" in smiles or "=O=O" in smiles or smiles.endswith("="):
        return False
    if re.search(r"=(F|Cl|Br|I)", smiles):
        return False
    if re.search(r"=[A-Z][A-Z]", smiles):
        return False
    return True


def _bounded(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _char_ngrams(text: str, n: int = 3) -> set[str]:
    if len(text) <= n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}
