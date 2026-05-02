"""LLM paraphrase I/O helpers with deterministic spec-consistency filtering.

This module does not call an LLM. It exports prompt records for any external
LLM workflow and filters returned paraphrases so that only language consistent
with the structured spec can enter the benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .instruction_templates import instruction_text_is_consistent


def main() -> None:
    args = parse_args()
    if args.command == "export-prompts":
        export_paraphrase_prompts(args.dataset, args.out, paraphrases_per_item=args.paraphrases_per_item)
    elif args.command == "filter":
        filter_paraphrases(args.dataset, args.paraphrases, args.out, args.rejected_out)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare and filter LLM paraphrases for instruction editing.")
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export-prompts", help="Write JSONL prompt records for an external LLM.")
    export.add_argument("--dataset", required=True)
    export.add_argument("--out", default="data/instruction_paraphrase_prompts.jsonl")
    export.add_argument("--paraphrases-per-item", type=int, default=8)
    filtr = sub.add_parser("filter", help="Filter returned paraphrases against the executable spec.")
    filtr.add_argument("--dataset", required=True)
    filtr.add_argument("--paraphrases", required=True, help="CSV/JSONL with pair_id and instruction_text.")
    filtr.add_argument("--out", default="data/instruction_editing_llm_verified.csv")
    filtr.add_argument("--rejected-out", default="data/instruction_editing_llm_rejected.csv")
    return parser.parse_args()


def export_paraphrase_prompts(dataset_path: str | Path, out_path: str | Path, paraphrases_per_item: int = 8) -> None:
    df = pd.read_csv(dataset_path)
    unique = df.drop_duplicates("pair_id").reset_index(drop=True)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for _, row in unique.iterrows():
            record = {
                "pair_id": row["pair_id"],
                "instruction_spec_json": row["instruction_spec_json"],
                "seed_instruction_text": row["instruction_text"],
                "paraphrases_per_item": int(paraphrases_per_item),
                "prompt": _prompt(row["pair_id"], row["instruction_spec_json"], row["instruction_text"], paraphrases_per_item),
            }
            f.write(json.dumps(record, sort_keys=True) + "\n")
    print(f"Wrote {len(unique)} paraphrase prompt records -> {out}")


def filter_paraphrases(dataset_path: str | Path, paraphrases_path: str | Path, out_path: str | Path, rejected_out_path: str | Path) -> None:
    dataset = pd.read_csv(dataset_path)
    paraphrases = _read_table(paraphrases_path)
    if "pair_id" not in paraphrases.columns or "instruction_text" not in paraphrases.columns:
        raise ValueError("Paraphrases must contain pair_id and instruction_text columns.")
    base = dataset.drop_duplicates("pair_id").set_index("pair_id")
    accepted = []
    rejected = []
    for _, para in paraphrases.iterrows():
        pair_id = para["pair_id"]
        if pair_id not in base.index:
            continue
        row = base.loc[pair_id].to_dict()
        text = str(para["instruction_text"])
        check = instruction_text_is_consistent(text, row["instruction_spec_json"])
        out = dict(row)
        out["instruction_text"] = text
        out["template_id"] = para.get("template_id", "llm_paraphrase")
        out["language_source"] = "llm_paraphrase"
        out["consistency_check_json"] = json.dumps(check, sort_keys=True)
        if check["consistent"]:
            accepted.append(out)
        else:
            out["rejection_reason"] = "extra_tags:" + "|".join(check["extra_tags"])
            rejected.append(out)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(accepted).to_csv(out, index=False)
    rejected_out = Path(rejected_out_path)
    rejected_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rejected).to_csv(rejected_out, index=False)
    print(f"Accepted {len(accepted)} paraphrases -> {out}")
    print(f"Rejected {len(rejected)} paraphrases -> {rejected_out}")


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return pd.DataFrame(rows)
    return pd.read_csv(path)


def _prompt(pair_id: str, spec_json: str, seed_text: str, paraphrases_per_item: int) -> str:
    return (
        f"Rewrite the molecular editing instruction below into {paraphrases_per_item} concise natural-language paraphrases. "
        "Do not add any new chemistry goals, constraints, properties, or edit operations. "
        "Return JSON lines with fields pair_id and instruction_text only.\n\n"
        f"pair_id: {pair_id}\n"
        f"Executable spec: {spec_json}\n"
        f"Seed instruction: {seed_text}\n"
    )


if __name__ == "__main__":
    main()
