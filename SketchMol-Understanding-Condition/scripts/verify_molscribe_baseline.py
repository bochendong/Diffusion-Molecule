#!/usr/bin/env python3
"""Verify MolScribe checkpoint/env using SketchMol official predict_csv.py."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PROJECT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from sketchmol_understanding_condition.chem import canonical_smiles, render_molecule_image_pil  # noqa: E402
from sketchmol_understanding_condition.molscribe_images import preprocess_image_for_molscribe  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/scratch/bdong/checkpoints/molscribe/swin_base_char_aux_200k.pth"),
    )
    parser.add_argument(
        "--generated-image",
        type=Path,
        default=PROJECT_DIR
        / "outputs/univideo_molecule_generation_v2_residual_ink/univideo_molecule/eval_latent/generated_images/generated_00000.png",
    )
    parser.add_argument(
        "--complex-smiles",
        default="O=C(c1cccc(Cl)c1)N1C[C@H](CN2CCC(c3ccccc3)CC2)[C@@H](c2ccccc2)C1",
    )
    parser.add_argument("--simple-smiles", default="CCO")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def _save_rgb_array(path: Path, arr) -> None:
    from PIL import Image

    Image.fromarray(arr).save(path)


def _render_variants(smiles: str, out_dir: Path, *, prefix: str) -> dict[str, Path]:
    from PIL import Image
    import numpy as np

    paths: dict[str, Path] = {}
    out_dir.mkdir(parents=True, exist_ok=True)

    # SketchMol-style SVG -> PNG at 300px when cairosvg is available.
    try:
        import cairosvg
        from rdkit import Chem
        from rdkit.Chem import Draw

        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            svg_path = out_dir / f"{prefix}_sketchmol_svg300.svg"
            png_path = out_dir / f"{prefix}_sketchmol_svg300.png"
            Draw.MolToFile(mol, str(svg_path), wedgeBonds=False)
            cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), output_width=300, output_height=300)
            paths["sketchmol_svg300"] = png_path
    except Exception as exc:
        paths["sketchmol_svg300_error"] = Path(str(exc))

    # Default RDKit color render.
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw

        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            color_path = out_dir / f"{prefix}_rdkit_color256.png"
            Draw.MolToImage(mol, size=(256, 256)).save(color_path)
            paths["rdkit_color256"] = color_path
    except Exception as exc:
        paths["rdkit_color256_error"] = Path(str(exc))

    # SUCC B&W Cairo render.
    bw = render_molecule_image_pil(smiles, image_size=256)
    if bw is not None:
        bw_path = out_dir / f"{prefix}_succ_bw256.png"
        bw.save(bw_path)
        paths["succ_bw256"] = bw_path
        pre_path = out_dir / f"{prefix}_succ_bw256_preprocessed.png"
        _save_rgb_array(pre_path, preprocess_image_for_molscribe(np.asarray(bw.convert("RGB"))))
        paths["succ_bw256_preprocessed"] = pre_path

    # Larger B&W render for MolScribe 384 input.
    bw384 = render_molecule_image_pil(smiles, image_size=384)
    if bw384 is not None:
        bw384_path = out_dir / f"{prefix}_succ_bw384.png"
        bw384.save(bw384_path)
        paths["succ_bw384"] = bw384_path

    return {k: v for k, v in paths.items() if isinstance(v, Path) and v.suffix.lower() in {".png", ".svg"}}


def _run_official_predict_csv(
    *,
    python_bin: str,
    predict_csv: Path,
    model_path: Path,
    image_csv: Path,
    batch_size: int,
    molscribe_workdir: Path,
    onmt_overlay: Path | None,
) -> subprocess.CompletedProcess[str]:
    import os

    env = dict(os.environ)
    pythonpath_parts: list[str] = []
    if onmt_overlay is not None and onmt_overlay.is_dir():
        pythonpath_parts.append(str(onmt_overlay))
    pythonpath_parts.append(str(molscribe_workdir))
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(pythonpath_parts)
    return subprocess.run(
        [
            python_bin,
            str(predict_csv),
            "--model_path",
            str(model_path),
            "--image_path",
            str(image_csv),
            "-n",
            str(batch_size),
        ],
        cwd=str(molscribe_workdir),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _summarize_csv(image_csv: Path, *, expected_smiles: str) -> dict[str, object]:
    rows = list(csv.DictReader(image_csv.open(encoding="utf-8")))
    nonempty = 0
    valid = 0
    examples = []
    for row in rows:
        raw = str(row.get("SMILES") or "").strip()
        score = row.get("molscribe_score", "")
        if raw:
            nonempty += 1
        canon = canonical_smiles(raw) if raw else None
        if canon:
            valid += 1
        examples.append({"smiles": raw[:120], "score": score, "valid": bool(canon)})
    exp_canon = canonical_smiles(expected_smiles)
    return {
        "rows": len(rows),
        "nonempty_smiles": nonempty,
        "valid_smiles": valid,
        "examples": examples,
        "expected_canonical": exp_canon,
    }


def main() -> int:
    args = parse_args()
    python_bin = sys.executable
    sketchmol_eval = REPO_DIR / "Research/Molecule Generation/SketchMol/SketchMol-v1-main/evaluate"
    predict_csv = sketchmol_eval / "predict_csv.py"
    onmt_overlay = Path(
        __import__("os").environ.get(
            "SUCC_ONMT_OVERLAY",
            __import__("os").environ.get(
                "SKETCHMOL_ONMT_OVERLAY",
                "/scratch/bdong/python_overlays/onmt220",
            ),
        )
    )
    if not predict_csv.exists():
        raise FileNotFoundError(predict_csv)
    if not args.model_path.exists():
        raise FileNotFoundError(args.model_path)

    work_root = Path(tempfile.mkdtemp(prefix="molscribe_baseline_"))
    results: list[dict[str, object]] = []

    cohorts = [
        ("simple", args.simple_smiles),
        ("complex", args.complex_smiles),
    ]
    image_variants: dict[str, Path] = {}
    for label, smiles in cohorts:
        image_variants.update(_render_variants(smiles, work_root / label, prefix=label))

    if args.generated_image.exists():
        image_variants["generated_raw"] = args.generated_image
        from PIL import Image
        import numpy as np

        gen = np.asarray(Image.open(args.generated_image).convert("RGB"))
        gen_pre = work_root / "generated_preprocessed.png"
        _save_rgb_array(gen_pre, preprocess_image_for_molscribe(gen))
        image_variants["generated_preprocessed"] = gen_pre

    for variant_name, image_path in sorted(image_variants.items()):
        csv_path = work_root / f"{variant_name}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["image_path"])
            writer.writeheader()
            writer.writerow({"image_path": str(image_path.resolve())})

        proc = _run_official_predict_csv(
            python_bin=python_bin,
            predict_csv=predict_csv,
            model_path=args.model_path,
            image_csv=csv_path,
            batch_size=args.batch_size,
            molscribe_workdir=sketchmol_eval,
            onmt_overlay=onmt_overlay,
        )
        summary = _summarize_csv(csv_path, expected_smiles=args.complex_smiles if "complex" in variant_name else args.simple_smiles)
        results.append(
            {
                "variant": variant_name,
                "image_path": str(image_path),
                "returncode": proc.returncode,
                "stdout_tail": proc.stdout.strip()[-500:],
                "stderr_tail": proc.stderr.strip()[-800:],
                **summary,
            }
        )

    payload = {
        "model_path": str(args.model_path),
        "predict_csv": str(predict_csv),
        "python": python_bin,
        "work_dir": str(work_root),
        "results": results,
    }
    output_json = args.output_json or (
        PROJECT_DIR / "outputs/univideo_molecule_generation_v2_residual_ink/univideo_molecule/image_structure_benchmark/molscribe_baseline_verify.json"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report_path = output_json.with_suffix(".md")
    lines = [
        "# MolScribe Baseline Verification (SketchMol predict_csv.py)",
        "",
        f"- model: `{args.model_path}`",
        f"- official script: `{predict_csv}`",
        f"- work_dir: `{work_root}`",
        "",
        "| variant | nonempty | valid | example SMILES | score | rc |",
        "| --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in results:
        example = row["examples"][0] if row["examples"] else {}
        lines.append(
            f"| {row['variant']} | {row['nonempty_smiles']} | {row['valid_smiles']} | "
            f"`{str(example.get('smiles', ''))[:60]}` | {example.get('score', '')} | {row['returncode']} |"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"report": str(report_path), "json": str(output_json)}, indent=2))
    for row in results:
        example = row["examples"][0] if row["examples"] else {}
        print(
            f"{row['variant']}: nonempty={row['nonempty_smiles']} valid={row['valid_smiles']} "
            f"smiles={str(example.get('smiles', ''))[:80]!r} rc={row['returncode']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
