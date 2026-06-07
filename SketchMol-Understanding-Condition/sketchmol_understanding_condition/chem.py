"""Small RDKit-backed chemistry helpers.

The module keeps RDKit optional at import time so non-chemistry utilities and
Torch-only tests can still run in minimal environments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def _rdkit():
    try:
        from rdkit import Chem, DataStructs
        from rdkit import RDLogger
        from rdkit.Chem import AllChem, Crippen, Descriptors, Draw, Lipinski, QED, rdMolDescriptors
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError as exc:
        raise RuntimeError(
            "RDKit is required for chemistry operations. Install rdkit or run "
            "only the encoder utilities."
        ) from exc
    RDLogger.DisableLog("rdApp.warning")
    return Chem, DataStructs, AllChem, Crippen, Descriptors, Draw, Lipinski, QED, rdMolDescriptors, MurckoScaffold


def rdkit_version() -> str:
    """Return the active RDKit version string."""

    try:
        from rdkit import rdBase
    except ImportError as exc:
        raise RuntimeError("RDKit is required to report rdkit_version().") from exc
    return str(rdBase.rdkitVersion)


def canonical_smiles(smiles: str) -> Optional[str]:
    """Return canonical SMILES, or None if parsing fails."""

    text = str(smiles or "").strip()
    if not text:
        return None
    Chem, *_ = _rdkit()
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def scaffold_smiles(smiles: str) -> Optional[str]:
    """Return Bemis-Murcko scaffold SMILES for a molecule."""

    Chem, *_, MurckoScaffold = _rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(scaffold, canonical=True)


def morgan_tanimoto(smiles_a: str, smiles_b: str, radius: int = 2, n_bits: int = 2048) -> Optional[float]:
    """Compute Morgan fingerprint Tanimoto similarity."""

    Chem, DataStructs, AllChem, *_ = _rdkit()
    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)
    if mol_a is None or mol_b is None:
        return None
    fp_a = _morgan_fingerprint(mol_a, radius=radius, n_bits=n_bits, fallback_all_chem=AllChem)
    fp_b = _morgan_fingerprint(mol_b, radius=radius, n_bits=n_bits, fallback_all_chem=AllChem)
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


def is_valid_smiles(smiles: str) -> bool:
    """Return whether RDKit can parse the SMILES."""

    Chem, *_ = _rdkit()
    return Chem.MolFromSmiles(smiles) is not None


def molecular_properties(smiles: str) -> Optional[dict[str, float]]:
    """Compute common SketchMol-style molecular properties."""

    Chem, _, _, Crippen, Descriptors, _, Lipinski, QED, rdMolDescriptors, _ = _rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "MolWt": float(Descriptors.MolWt(mol)),
        "LogP": float(Crippen.MolLogP(mol)),
        "QED": float(QED.qed(mol)),
        "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
        "HBD": float(Lipinski.NumHDonors(mol)),
        "HBA": float(Lipinski.NumHAcceptors(mol)),
        "rotatable": float(Lipinski.NumRotatableBonds(mol)),
    }


def render_molecule_image(smiles: str, output_path: str | Path, image_size: int = 256) -> Optional[str]:
    """Render a 2D molecule image and return the output path."""

    Chem, _, _, _, _, Draw, *_ = _rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Draw.MolToImage(mol, size=(image_size, image_size))
    image.save(output_path)
    return str(output_path)


def morgan_fingerprint_bits(smiles: str, radius: int = 2, n_bits: int = 512) -> Optional[list[float]]:
    """Return Morgan fingerprint bits as floats."""

    Chem, _, AllChem, *_ = _rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = _morgan_fingerprint(mol, radius=radius, n_bits=n_bits, fallback_all_chem=AllChem)
    return [1.0 if bit == "1" else 0.0 for bit in fp.ToBitString()]


def _morgan_fingerprint(mol, radius: int, n_bits: int, fallback_all_chem):
    try:
        from rdkit.Chem import rdFingerprintGenerator

        generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        return generator.GetFingerprint(mol)
    except Exception:
        return fallback_all_chem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
