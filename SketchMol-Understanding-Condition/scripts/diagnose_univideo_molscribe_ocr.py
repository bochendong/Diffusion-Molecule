#!/usr/bin/env python3
"""Diagnose MolScribe OCR quality across image sources for UniVideo molecule runs."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PROJECT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from sketchmol_understanding_condition.chem import canonical_smiles, morgan_tanimoto, render_molecule_image  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=PROJECT_DIR / "outputs/univideo_molecule_generation_v2_residual_ink",
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--backend",
        choices=("sketchmol", "custom"),
        default="sketchmol",
    )
    parser.add_argument(
        "--preprocess-images",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def _load_molscribe_module():
    script = PROJECT_DIR / "scripts" / "run_molscribe_ocr.py"
    spec = importlib.util.spec_from_file_location("run_molscribe_ocr", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _read_predictions(path: Path, limit: int) -> list[dict[str, str]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    return rows[:limit]


def _image_stats(path: Path) -> dict[str, float]:
    from PIL import Image

    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    gray = arr.mean(axis=2)
    dark = gray < 0.5
    return {
        "mean_intensity": float(gray.mean()),
        "std_intensity": float(gray.std()),
        "nonwhite_fraction": float((gray < 0.99).mean()),
        "dark_fraction": float(dark.mean()),
        "unique_rgb_colors": float(len({tuple(map(int, px)) for px in arr.reshape(-1, 3)})),
    }


def _valid_smiles(smiles: str) -> bool:
    text = str(smiles or "").strip()
    if not text:
        return False
    return canonical_smiles(text) is not None


def _evaluate_cohort(
    *,
    name: str,
    paths: list[Path],
    expected_smiles: list[str],
    molscribe_module: Any,
    model: Any,
    batch_size: int,
    backend: str,
    preprocess_images: bool,
) -> dict[str, Any]:
    path_strings = [str(path) for path in paths]
    if backend == "sketchmol":
        smiles, scores, diagnostics = molscribe_module._predict_sketchmol(
            model,
            path_strings,
            batch_size=batch_size,
            preprocess_images=preprocess_images,
        )
    else:
        smiles, scores, diagnostics = molscribe_module._predict(
            model,
            path_strings,
            batch_size=batch_size,
            preprocess_images=preprocess_images,
            raw_smiles_fallback=False,
        )

    graph_usable = 0
    raw_usable = 0
    final_valid = 0
    tanimotos: list[float] = []
    exact = 0
    examples: list[dict[str, object]] = []

    for idx, (path, final, diag, expected, score) in enumerate(
        zip(paths, smiles, diagnostics, expected_smiles, scores)
    ):
        graph = str(diag.get("molscribe_graph_smiles", "") or "")
        raw = str(diag.get("molscribe_raw_smiles", "") or "")
        source = str(diag.get("molscribe_decode_source", "") or "")
        if molscribe_module._usable_smiles(graph):
            graph_usable += 1
        if molscribe_module._usable_smiles(raw):
            raw_usable += 1
        if _valid_smiles(final):
            final_valid += 1
            gen_canon = canonical_smiles(final)
            exp_canon = canonical_smiles(expected)
            if gen_canon and exp_canon:
                tani = morgan_tanimoto(gen_canon, exp_canon)
                if tani is not None:
                    tanimotos.append(float(tani))
                if gen_canon == exp_canon:
                    exact += 1
        if idx < 5:
            examples.append(
                {
                    "index": idx,
                    "image_path": str(path),
                    "expected_smiles": expected,
                    "final_smiles": final,
                    "graph_smiles": graph,
                    "raw_smiles": raw[:120],
                    "decode_source": source,
                    "molscribe_score": score,
                    "valid": _valid_smiles(final),
                }
            )

    n = len(paths)
    decode_sources = Counter(str(d.get("molscribe_decode_source", "")) for d in diagnostics)
    return {
        "name": name,
        "n": n,
        "image_stats_mean": _mean_stats(paths),
        "graph_usable_rate": graph_usable / n if n else 0.0,
        "raw_usable_rate": raw_usable / n if n else 0.0,
        "final_valid_rate": final_valid / n if n else 0.0,
        "exact_match_rate": exact / n if n else 0.0,
        "mean_source_tanimoto": float(np.mean(tanimotos)) if tanimotos else None,
        "median_source_tanimoto": float(np.median(tanimotos)) if tanimotos else None,
        "decode_source_counts": dict(decode_sources),
        "examples": examples,
    }


def _mean_stats(paths: list[Path]) -> dict[str, float]:
    if not paths:
        return {}
    stats = [_image_stats(path) for path in paths]
    return {key: float(np.mean([row[key] for row in stats])) for key in stats[0]}


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    eval_dir = run_dir / "univideo_molecule/eval_latent"
    predictions = _read_predictions(eval_dir / "predictions.csv", args.limit)

    temp_dir = Path(tempfile.mkdtemp(prefix="univideo_molscribe_diag_"))
    cohorts: list[tuple[str, list[Path], list[str]]] = []

    rdkit_source_dir = temp_dir / "rdkit_source"
    rdkit_target_dir = temp_dir / "rdkit_target"
    rdkit_source_dir.mkdir(parents=True, exist_ok=True)
    rdkit_target_dir.mkdir(parents=True, exist_ok=True)

    source_oracle_paths: list[Path] = []
    target_oracle_paths: list[Path] = []
    generated_paths: list[Path] = []
    rdkit_source_paths: list[Path] = []
    rdkit_target_paths: list[Path] = []
    source_expected: list[str] = []
    target_expected: list[str] = []

    for idx, row in enumerate(predictions):
        source_oracle = eval_dir / "source_oracle_images" / f"decoded_{idx:05d}.png"
        target_oracle = eval_dir / "target_oracle_images" / f"decoded_{idx:05d}.png"
        generated = eval_dir / "generated_images" / f"generated_{idx:05d}.png"
        if not (source_oracle.exists() and target_oracle.exists() and generated.exists()):
            continue

        source_smiles = row["source_smiles"]
        target_smiles = row["target_smiles"]
        rdkit_source_path = rdkit_source_dir / f"rdkit_source_{idx:05d}.png"
        rdkit_target_path = rdkit_target_dir / f"rdkit_target_{idx:05d}.png"
        render_molecule_image(source_smiles, rdkit_source_path)
        render_molecule_image(target_smiles, rdkit_target_path)

        source_oracle_paths.append(source_oracle)
        target_oracle_paths.append(target_oracle)
        generated_paths.append(generated)
        rdkit_source_paths.append(rdkit_source_path)
        rdkit_target_paths.append(rdkit_target_path)
        source_expected.append(source_smiles)
        target_expected.append(target_smiles)

    cohorts.extend(
        [
            ("rdkit_source", rdkit_source_paths, source_expected),
            ("rdkit_target", rdkit_target_paths, target_expected),
            ("source_oracle", source_oracle_paths, source_expected),
            ("target_oracle", target_oracle_paths, target_expected),
            ("generated", generated_paths, source_expected),
        ]
    )

    import torch
    from molscribe import MolScribe

    molscribe_module = _load_molscribe_module()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = MolScribe(str(args.model_path), device=device)

    results = []
    for name, paths, expected in cohorts:
        print(f"Evaluating cohort: {name} ({len(paths)} images)", flush=True)
        results.append(
            _evaluate_cohort(
                name=name,
                paths=paths,
                expected_smiles=expected,
                molscribe_module=molscribe_module,
                model=model,
                batch_size=args.batch_size,
                backend=args.backend,
                preprocess_images=args.preprocess_images,
            )
        )

    payload = {
        "run_dir": str(run_dir),
        "limit": args.limit,
        "device": str(device),
        "backend": args.backend,
        "preprocess_images": args.preprocess_images,
        "cohorts": results,
    }
    output_json = args.output_json or (
        run_dir / "univideo_molecule/image_structure_benchmark/molscribe_diagnostic.json"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report_path = output_json.with_suffix(".md")
    report_path.write_text(_render_report(payload), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "json": str(output_json)}, indent=2))


def _render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# UniVideo MolScribe OCR Diagnostic",
        "",
        f"- run_dir: `{payload['run_dir']}`",
        f"- samples per cohort: `{payload['limit']}`",
        f"- device: `{payload['device']}`",
        f"- backend: `{payload['backend']}`",
        f"- preprocess_images: `{payload['preprocess_images']}`",
        "",
        "## Summary",
        "",
        "| cohort | n | graph usable | raw usable | final valid | exact match | mean Tanimoto | decode sources |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["cohorts"]:
        tani = row["mean_source_tanimoto"]
        tani_text = "" if tani is None else f"{tani:.3f}"
        sources = ", ".join(f"{k}:{v}" for k, v in sorted(row["decode_source_counts"].items()))
        lines.append(
            f"| {row['name']} | {row['n']} | {row['graph_usable_rate']:.3f} | "
            f"{row['raw_usable_rate']:.3f} | {row['final_valid_rate']:.3f} | "
            f"{row['exact_match_rate']:.3f} | {tani_text} | {sources} |"
        )

    lines.extend(["", "## Image Stats (mean)", ""])
    for row in payload["cohorts"]:
        stats = row["image_stats_mean"]
        lines.append(
            f"- **{row['name']}**: nonwhite={stats.get('nonwhite_fraction', 0):.4f}, "
            f"dark={stats.get('dark_fraction', 0):.4f}, unique_rgb={stats.get('unique_rgb_colors', 0):.0f}"
        )

    lines.extend(["", "## Examples", ""])
    for row in payload["cohorts"]:
        lines.append(f"### {row['name']}")
        lines.append("")
        for example in row["examples"][:3]:
            lines.append(f"- idx={example['index']} source={example['decode_source']} valid={example['valid']}")
            lines.append(f"  expected: `{str(example['expected_smiles'])[:80]}`")
            lines.append(f"  final: `{str(example['final_smiles'])[:80]}`")
            lines.append(f"  graph: `{str(example['graph_smiles'])[:80]}`")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
