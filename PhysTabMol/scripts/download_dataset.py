#!/usr/bin/env python3
"""Download and prepare SMILES datasets for PhysTabMol.

Examples:
    python scripts/download_dataset.py --source chembl --limit 100000 --out data/molecules.csv
    python scripts/download_dataset.py --source pubchem --limit 100000 --out data/molecules_pubchem_100k.csv
    python scripts/download_dataset.py --source zinc20 --zinc-chunks 1 --limit 100000 --out data/molecules_zinc20_100k.csv
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os
import re
import shutil
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

CHEMBL_LATEST = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/"
PUBCHEM_CID_SMILES = "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz"
ZINC20_ML_SMILES = "https://files.docking.org/zinc20-ML/smiles/"

CHEMBL_RE = re.compile(r'href="(chembl_\d+_chemreps\.txt\.gz)"')
ZINC_CHUNK_RE = re.compile(r'href="([^"]*ZINC20_smiles_chunk[^"]*)"')


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.source == "chembl":
        raw_files = [download_chembl(raw_dir)]
        smiles_iter = iter_chembl(raw_files[0])
    elif args.source == "pubchem":
        raw_files = [download_file(PUBCHEM_CID_SMILES, raw_dir / "CID-SMILES.gz")]
        smiles_iter = iter_pubchem(raw_files[0])
    elif args.source == "zinc20":
        raw_files = download_zinc20(raw_dir, chunks=args.zinc_chunks)
        smiles_iter = iter_zinc20(raw_files)
    else:
        raise ValueError(f"Unsupported source: {args.source}")

    stats = write_smiles_csv(
        smiles_iter=smiles_iter,
        out_path=out_path,
        limit=args.limit,
        deduplicate=not args.no_dedup,
        rdkit_filter=args.rdkit_filter,
    )
    write_manifest(args, raw_files, out_path, stats)
    print(f"Prepared {stats['written']} SMILES -> {out_path}")
    print(f"Raw files saved under: {raw_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download an official SMILES dataset for PhysTabMol.")
    parser.add_argument("--source", choices=["chembl", "pubchem", "zinc20"], default="chembl")
    parser.add_argument("--out", default="data/molecules.csv", help="Output CSV with a single 'smiles' column.")
    parser.add_argument("--raw-dir", default="data/raw", help="Directory for downloaded raw files.")
    parser.add_argument("--limit", type=int, default=100000, help="Maximum number of SMILES to write. Use 0 for all.")
    parser.add_argument("--zinc-chunks", type=int, default=1, help="Number of ZINC20 ML chunks to download.")
    parser.add_argument("--rdkit-filter", action="store_true", help="Use RDKit drug-like-ish filtering if RDKit is installed.")
    parser.add_argument("--no-dedup", action="store_true", help="Do not deduplicate SMILES.")
    return parser.parse_args()


def download_chembl(raw_dir: Path) -> Path:
    index = fetch_text(CHEMBL_LATEST)
    matches = CHEMBL_RE.findall(index)
    if not matches:
        raise RuntimeError("Could not find ChEMBL chemreps file in latest FTP index.")
    filename = sorted(matches)[-1]
    return download_file(CHEMBL_LATEST + filename, raw_dir / "chembl" / filename)


def download_zinc20(raw_dir: Path, chunks: int) -> list[Path]:
    index = fetch_text(ZINC20_ML_SMILES)
    names = []
    for href in ZINC_CHUNK_RE.findall(index):
        href = href.split("?")[0]
        if href not in names:
            names.append(href)
    if not names:
        raise RuntimeError("Could not find ZINC20 smiles chunks in index.")
    selected = names[: max(1, chunks)]
    paths = []
    for name in selected:
        paths.append(download_file(ZINC20_ML_SMILES + name, raw_dir / "zinc20" / name))
    return paths


def download_file(url: str, dest: Path, retries: int = 3) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Using existing file: {dest}")
        return dest

    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            print(f"Downloading {url}")
            with urllib.request.urlopen(url, timeout=60) as response, open(tmp, "wb") as f:
                shutil.copyfileobj(response, f)
            tmp.replace(dest)
            return dest
        except Exception as exc:
            if tmp.exists():
                tmp.unlink()
            if attempt == retries:
                raise
            wait = 5 * attempt
            print(f"Download failed ({exc}); retrying in {wait}s...", file=sys.stderr)
            time.sleep(wait)
    return dest


def fetch_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read().decode("utf-8", errors="replace")


def iter_chembl(path: Path):
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        smiles_col = None
        for name in reader.fieldnames or []:
            if name.lower() in {"canonical_smiles", "smiles"}:
                smiles_col = name
                break
        if smiles_col is None:
            raise RuntimeError(f"Could not find canonical_smiles column in {path}")
        for row in reader:
            smi = clean_smiles(row.get(smiles_col, ""))
            if smi:
                yield smi


def iter_pubchem(path: Path):
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0].isdigit():
                smi = parts[1]
            elif parts:
                smi = parts[0]
            else:
                continue
            smi = clean_smiles(smi)
            if smi:
                yield smi


def iter_zinc20(paths: list[Path]):
    for path in paths:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8", errors="replace") as f:
            header = None
            smiles_idx = None
            for line_no, line in enumerate(f):
                parts = line.strip().replace(",", "\t").split()
                if not parts:
                    continue
                lower = [p.lower() for p in parts]
                if line_no == 0 and "smiles" in lower:
                    header = lower
                    smiles_idx = header.index("smiles")
                    continue
                if smiles_idx is not None and len(parts) > smiles_idx:
                    smi = parts[smiles_idx]
                elif parts[0].upper().startswith("ZINC") and len(parts) > 1:
                    smi = parts[1]
                else:
                    smi = parts[0]
                smi = clean_smiles(smi)
                if smi and smi.lower() != "smiles":
                    yield smi


def write_smiles_csv(smiles_iter, out_path: Path, limit: int, deduplicate: bool, rdkit_filter: bool) -> dict:
    seen = set()
    scanned = 0
    written = 0
    invalid = 0
    with tempfile.NamedTemporaryFile("w", delete=False, newline="", encoding="utf-8", dir=out_path.parent) as tmp:
        writer = csv.writer(tmp)
        writer.writerow(["smiles"])
        tmp_name = tmp.name
        for smi in smiles_iter:
            scanned += 1
            if deduplicate and smi in seen:
                continue
            if deduplicate:
                seen.add(smi)
            if rdkit_filter and not rdkit_keep(smi):
                invalid += 1
                continue
            writer.writerow([smi])
            written += 1
            if limit and written >= limit:
                break
    os.replace(tmp_name, out_path)
    return {"scanned": scanned, "written": written, "invalid_or_filtered": invalid, "deduplicated": deduplicate}


def clean_smiles(value: str) -> str:
    smi = value.strip().strip('"').strip("'")
    if not smi or smi in {"-", "None", "nan", "NaN"}:
        return ""
    if "." in smi:
        return ""
    return smi


def rdkit_keep(smiles: str) -> bool:
    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
    except Exception as exc:
        raise RuntimeError("--rdkit-filter requested, but RDKit is not installed.") from exc

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    allowed = {"C", "N", "O", "S", "F", "Cl", "Br", "I", "H"}
    if any(atom.GetSymbol() not in allowed for atom in mol.GetAtoms()):
        return False
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rb = Lipinski.NumRotatableBonds(mol)
    return 80 <= mw <= 650 and -3 <= logp <= 7 and 0 <= tpsa <= 180 and hbd <= 6 and hba <= 12 and rb <= 15


def write_manifest(args: argparse.Namespace, raw_files: list[Path], out_path: Path, stats: dict) -> None:
    manifest = out_path.with_suffix(".manifest.txt")
    with open(manifest, "w", encoding="utf-8") as f:
        f.write(f"source={args.source}\n")
        f.write(f"out={out_path}\n")
        f.write(f"limit={args.limit}\n")
        f.write(f"rdkit_filter={args.rdkit_filter}\n")
        f.write("raw_files=\n")
        for path in raw_files:
            f.write(f"  {path}\n")
        for key, value in stats.items():
            f.write(f"{key}={value}\n")


if __name__ == "__main__":
    main()
