"""Optional 3D molecule generation and metrics.

These utilities are RDKit-backed. They are no-ops when RDKit is absent so the
rest of the benchmark remains runnable on lightweight machines.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:  # pragma: no cover - RDKit not installed in this local environment.
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors

    RDKIT_3D_AVAILABLE = True
except Exception:  # pragma: no cover
    Chem = None
    AllChem = None
    rdMolDescriptors = None
    RDKIT_3D_AVAILABLE = False


def add_3d_metrics(decoded_df: pd.DataFrame, sdf_path: str | Path | None = None, max_sdf: int = 500) -> pd.DataFrame:
    out = decoded_df.copy()
    metric_rows = []
    writer = None
    if sdf_path and RDKIT_3D_AVAILABLE:
        Path(sdf_path).parent.mkdir(parents=True, exist_ok=True)
        writer = Chem.SDWriter(str(sdf_path))

    for idx, row in out.iterrows():
        metrics, mol = _embed_and_measure(str(row.get("smiles", "")))
        metric_rows.append(metrics)
        if writer is not None and mol is not None and idx < max_sdf:
            mol.SetProp("_Name", f"condition_{row.get('condition_idx', idx)}_sample_{row.get('sample_idx', idx)}")
            writer.write(mol)

    if writer is not None:
        writer.close()
    metrics_df = pd.DataFrame(metric_rows)
    return pd.concat([out.reset_index(drop=True), metrics_df.reset_index(drop=True)], axis=1)


def molecule_3d_metrics(smiles: str) -> dict[str, float]:
    """Return RDKit 3D descriptors for a single molecule, or unavailable flags."""

    metrics, _ = _embed_and_measure(smiles)
    return metrics


def _embed_and_measure(smiles: str):
    empty = {
        "3d_available": float(RDKIT_3D_AVAILABLE),
        "3d_embed_success": 0.0,
        "3d_radius_gyration": float("nan"),
        "3d_asphericity": float("nan"),
        "3d_eccentricity": float("nan"),
        "3d_npr1": float("nan"),
        "3d_npr2": float("nan"),
        "3d_spherocity": float("nan"),
    }
    if not RDKIT_3D_AVAILABLE:
        return empty, None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return empty, None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 17
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        return empty, None
    try:
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        pass
    metrics = {
        "3d_available": 1.0,
        "3d_embed_success": 1.0,
        "3d_radius_gyration": float(rdMolDescriptors.CalcRadiusOfGyration(mol)),
        "3d_asphericity": float(rdMolDescriptors.CalcAsphericity(mol)),
        "3d_eccentricity": float(rdMolDescriptors.CalcEccentricity(mol)),
        "3d_npr1": float(rdMolDescriptors.CalcNPR1(mol)),
        "3d_npr2": float(rdMolDescriptors.CalcNPR2(mol)),
        "3d_spherocity": float(rdMolDescriptors.CalcSpherocityIndex(mol)),
    }
    return metrics, mol
