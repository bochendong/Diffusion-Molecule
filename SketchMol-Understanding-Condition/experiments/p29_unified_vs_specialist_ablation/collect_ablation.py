#!/usr/bin/env python3
"""Collect the matched unified-versus-specialist continuation ablation."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def table1(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text())
    return {
        "de_novo_avg_2p_7p": float(payload["average_2p_7p"]),
        **{
            f"de_novo_{count}p": float(payload["strict_success"][f"{count}p"])
            for count in range(2, 8)
        },
    }


def table2(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 10 or len({row["task_key"] for row in rows}) != 10:
        raise ValueError(f"expected ten unique editing tasks in {path}, found {len(rows)}")
    metrics = [
        "Validity", "Acc_all(0.65)", "Acc_valid(0.65)",
        "Acc_all(0.15)", "Acc_valid(0.15)",
    ]
    return {
        f"editing_{metric}": statistics.fmean(float(row[metric]) for row in rows)
        for metric in metrics
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root
    unified = {
        **table1(root / "unified/eval_table1/results/table1/p24_table1.json"),
        **table2(root / "unified/eval_table2/results/moledit_table_summary.csv"),
    }
    specialists = {
        **table1(root / "construction_specialist/eval_table1/results/table1/p24_table1.json"),
        **table2(root / "editing_specialist/eval_table2/results/moledit_table_summary.csv"),
    }
    payload = {
        "protocol": "p29_shared_initialization_unified_vs_specialists_v1",
        "unified": unified,
        "separate_specialist_continuations": specialists,
        "unified_minus_specialists": {
            key: unified[key] - specialists[key] for key in sorted(unified)
        },
    }
    out = root / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "ablation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    rows = [
        {
            "policy": "Separate specialist continuations",
            "checkpoints": 2,
            **specialists,
        },
        {"policy": "Unified MolProgram", "checkpoints": 1, **unified},
    ]
    with (out / "ablation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    md = [
        "# Unified versus specialist continuations",
        "",
        "| Policy | Checkpoints | De novo Avg. | Editing Acc_all(.65) | Editing Acc_all(.15) | Editing validity |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        md.append(
            f"| {row['policy']} | {row['checkpoints']} | "
            f"{100 * row['de_novo_avg_2p_7p']:.1f} | "
            f"{100 * row['editing_Acc_all(0.65)']:.1f} | "
            f"{100 * row['editing_Acc_all(0.15)']:.1f} | "
            f"{100 * row['editing_Validity']:.1f} |"
        )
    (out / "ablation.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
