#!/usr/bin/env python3
"""Extract release-quality PubChem molecules and descriptor values from SDF.gz."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path


ALLOWED_ATOMS = {1, 6, 7, 8, 9, 15, 16, 17, 35, 53}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-sdf-gz", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=24001)
    args = parser.parse_args()

    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    seen: set[int] = set()
    with gzip.open(args.input_sdf_gz, "rb") as source, args.output_jsonl.open(
        "w", encoding="utf-8"
    ) as output:
        supplier = Chem.ForwardSDMolSupplier(source, sanitize=True, removeHs=True)
        for mol in supplier:
            counts["records"] += 1
            if mol is None:
                counts["invalid"] += 1
                continue
            if len(Chem.GetMolFrags(mol)) != 1:
                counts["multi_fragment"] += 1
                continue
            if any(atom.GetAtomicNum() not in ALLOWED_ATOMS for atom in mol.GetAtoms()):
                counts["unsupported_atom"] += 1
                continue
            smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
            if not smiles:
                counts["empty_smiles"] += 1
                continue
            smiles_rank = int.from_bytes(
                hashlib.sha256(smiles.encode()).digest()[:8], "big"
            )
            if smiles_rank in seen:
                counts["duplicate_smiles_hash"] += 1
                continue
            seen.add(smiles_rank)
            values = {
                "MW": round(float(Descriptors.MolWt(mol)), 6),
                "LogP": round(float(Crippen.MolLogP(mol)), 6),
                "QED": round(float(QED.qed(mol)), 6),
                "TPSA": round(float(rdMolDescriptors.CalcTPSA(mol)), 6),
                "HBD": float(Lipinski.NumHDonors(mol)),
                "HBA": float(Lipinski.NumHAcceptors(mol)),
                "RB": float(Lipinski.NumRotatableBonds(mol)),
            }
            if not (
                120.0 <= values["MW"] <= 700.0
                and -3.0 <= values["LogP"] <= 8.0
                and values["QED"] >= 0.10
                and values["TPSA"] <= 180.0
                and values["HBD"] <= 8.0
                and values["HBA"] <= 14.0
                and values["RB"] <= 16.0
            ):
                counts["outside_release_space"] += 1
                continue
            cid = ""
            for key in ("PUBCHEM_COMPOUND_CID", "PUBCHEM_CID", "CID"):
                if mol.HasProp(key):
                    cid = mol.GetProp(key).strip()
                    break
            rank = hashlib.sha256(
                f"{args.seed}:{cid}:{smiles}".encode()
            ).hexdigest()
            output.write(
                json.dumps(
                    {
                        "source_dataset": "PubChem",
                        "source_record_id": cid,
                        "target_smiles": smiles,
                        "target_hash": hashlib.sha256(smiles.encode()).hexdigest(),
                        "descriptor_values": values,
                        "selection_rank": rank,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            counts["eligible"] += 1
            if counts["records"] % 50_000 == 0:
                print(
                    f"[p24-pubchem] records={counts['records']} eligible={counts['eligible']}",
                    flush=True,
                )

    summary = {
        "protocol": "p24_pubchem_descriptor_extract_v1",
        "input": str(args.input_sdf_gz),
        "input_sha256": sha256_file(args.input_sdf_gz),
        "seed": args.seed,
        "counts": dict(counts),
        "output": str(args.output_jsonl),
        "output_sha256": sha256_file(args.output_jsonl),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
