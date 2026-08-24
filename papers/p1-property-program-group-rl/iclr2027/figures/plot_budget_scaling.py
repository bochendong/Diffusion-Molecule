#!/usr/bin/env python3
"""Render the frozen single-seed candidate-budget scaling figure."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter


HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "budget_scaling.csv"
OUTPUT_PATH = HERE / "budget_scaling.pdf"


def load_rows() -> list[dict[str, str]]:
    with DATA_PATH.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    rows = load_rows()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    styles = {
        "SFT": {"color": "#6B7280", "linestyle": "--", "marker": "o"},
        "GRPO": {"color": "#2563A6", "linestyle": "-", "marker": "s"},
    }
    panels = [("2p-7p", "2p-7p programs"), ("OOD", "OOD programs")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.45), sharey=True)

    for ax, (evaluation, title) in zip(axes, panels):
        panel_rows = [row for row in rows if row["evaluation"] == evaluation]
        for method in ("SFT", "GRPO"):
            method_rows = sorted(
                (row for row in panel_rows if row["method"] == method),
                key=lambda row: int(row["k"]),
            )
            x = [int(row["k"]) for row in method_rows]
            y = [float(row["pass_at_k"]) for row in method_rows]
            ax.plot(
                x,
                y,
                label=method,
                linewidth=1.8,
                markersize=4.5,
                markeredgewidth=0.8,
                **styles[method],
            )
            for x_value, y_value in zip(x, y):
                if x_value == 1:
                    offset = (9, 3) if method == "SFT" else (9, 13)
                    horizontal_alignment = "left"
                    vertical_alignment = "bottom"
                elif y_value < 18:
                    offset = (-6, 6) if method == "SFT" else (6, 6)
                    horizontal_alignment = "right" if method == "SFT" else "left"
                    vertical_alignment = "bottom"
                else:
                    offset = (0, 5 if method == "GRPO" else -10)
                    horizontal_alignment = "center"
                    vertical_alignment = "bottom" if method == "GRPO" else "top"
                ax.annotate(
                    f"{y_value:.1f}",
                    (x_value, y_value),
                    xytext=offset,
                    textcoords="offset points",
                    ha=horizontal_alignment,
                    va=vertical_alignment,
                    fontsize=7.2,
                    color=styles[method]["color"],
                )

        ax.set_xscale("log", base=2)
        budgets = sorted({int(row["k"]) for row in panel_rows})
        ax.xaxis.set_major_locator(FixedLocator(budgets))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value)}"))
        ax.set_xlim(0.82, 310)
        ax.set_ylim(0, 102)
        ax.set_title(title, pad=4)
        ax.set_xlabel("Candidate budget, $k$")
        ax.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.8)
        ax.grid(axis="x", visible=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Strict Pass@$k$ (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        ncol=2,
        handlelength=2.8,
        columnspacing=1.6,
    )
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.20, top=0.80, wspace=0.16)
    fig.savefig(
        OUTPUT_PATH,
        bbox_inches="tight",
        pad_inches=0.02,
        metadata={
            "Creator": "MolProgram plot_budget_scaling.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )


if __name__ == "__main__":
    main()
