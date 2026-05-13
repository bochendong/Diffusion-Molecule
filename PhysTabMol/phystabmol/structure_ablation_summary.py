"""Summarize structure-prompt ablation outputs for a trained run."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SUMMARY_COLUMNS = [
    "ablation",
    "benchmark_task",
    "joint_success_strict",
    "joint_success_sketchmol_tolerance",
    "exact_train_hit_rate",
    "direct_train_decoder_fraction",
    "source_aware_edit_decoder_fraction",
    "mmp_fragment_decoder_fraction",
    "mmp_two_step_fragment_decoder_fraction",
    "sampled_nearest_train_tanimoto",
    "sampled_novelty_at_tanimoto_0_90",
    "novelty",
    "uniqueness",
    "druglike_rate",
]


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    rows = []
    main_path = run_dir / "tables" / "structure_prompt_benchmark" / "structure_prompt_summary.csv"
    rows.extend(_read_summary(main_path, "main"))
    for path in sorted((run_dir / "tables").glob("structure_prompt_ablation_*/structure_prompt_summary.csv")):
        name = path.parent.name.removeprefix("structure_prompt_ablation_")
        rows.extend(_read_summary(path, name))
    if not rows:
        raise FileNotFoundError(f"No structure prompt summary files found under {run_dir}")
    out = pd.concat(rows, ignore_index=True)
    ordered = [col for col in SUMMARY_COLUMNS if col in out.columns]
    out = out[ordered + [col for col in out.columns if col not in ordered]]
    output_path = Path(args.out) if args.out else run_dir / "tables" / "structure_prompt_ablation_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Wrote {len(out)} rows -> {output_path}")
    print(_overall_table(out).to_string(index=False))


def _read_summary(path: Path, name: str) -> list[pd.DataFrame]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    df.insert(0, "ablation", name)
    return [df]


def _overall_table(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "joint_success_strict",
        "joint_success_sketchmol_tolerance",
        "exact_train_hit_rate",
        "mmp_two_step_fragment_decoder_fraction",
        "sampled_novelty_at_tanimoto_0_90",
        "novelty",
        "uniqueness",
    ]
    available = [col for col in metrics if col in df.columns]
    return df.groupby("ablation", dropna=False)[available].mean(numeric_only=True).reset_index().round(4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize PhysTabMol structure-prompt ablation results.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", default="")
    return parser.parse_args()


if __name__ == "__main__":
    main()

