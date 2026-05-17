"""LLM paraphrase I/O helpers with deterministic spec-consistency filtering.

This module does not call an LLM. It exports prompt records for any external
LLM workflow and filters returned paraphrases so that only language consistent
with the structured spec can enter the benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .instruction_actions import ground_instruction_text
from .instruction_schema import spec_tags
from .instruction_templates import instruction_text_is_consistent


def main() -> None:
    args = parse_args()
    if args.command == "export-prompts":
        export_paraphrase_prompts(args.dataset, args.out, paraphrases_per_item=args.paraphrases_per_item)
    elif args.command == "filter":
        filter_paraphrases(
            args.dataset,
            args.paraphrases,
            args.out,
            args.rejected_out,
            allow_missing_tags=args.allow_missing_tags,
        )
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
    filtr.add_argument(
        "--allow-missing-tags",
        action="store_true",
        help="Accept paraphrases that introduce no extra tags even if the conservative tagger misses required tags.",
    )
    return parser.parse_args()


def export_paraphrase_prompts(dataset_path: str | Path, out_path: str | Path, paraphrases_per_item: int = 8) -> None:
    df = pd.read_csv(dataset_path)
    df = _with_paraphrase_base_id(df)
    unique = df.drop_duplicates("paraphrase_base_id").reset_index(drop=True)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for _, row in unique.iterrows():
            record = {
                "paraphrase_base_id": row["paraphrase_base_id"],
                "pair_id": row["pair_id"],
                "instruction_spec_json": row["instruction_spec_json"],
                "seed_instruction_text": row["instruction_text"],
                "difficulty": row.get("difficulty", ""),
                "instruction_combo_key": row.get("instruction_combo_key", ""),
                "paraphrases_per_item": int(paraphrases_per_item),
                "prompt": _prompt(
                    row["paraphrase_base_id"],
                    row["pair_id"],
                    row["instruction_spec_json"],
                    row["instruction_text"],
                    paraphrases_per_item,
                ),
            }
            f.write(json.dumps(record, sort_keys=True) + "\n")
    print(f"Wrote {len(unique)} paraphrase prompt records -> {out}")


def filter_paraphrases(
    dataset_path: str | Path,
    paraphrases_path: str | Path,
    out_path: str | Path,
    rejected_out_path: str | Path,
    allow_missing_tags: bool = False,
) -> None:
    dataset = _with_paraphrase_base_id(pd.read_csv(dataset_path))
    paraphrases = _read_table(paraphrases_path)
    if "pair_id" not in paraphrases.columns or "instruction_text" not in paraphrases.columns:
        raise ValueError("Paraphrases must contain pair_id and instruction_text columns; paraphrase_base_id is strongly recommended.")
    base = dataset.drop_duplicates("paraphrase_base_id").set_index("paraphrase_base_id")
    pair_counts = dataset.drop_duplicates("paraphrase_base_id").groupby("pair_id").size().to_dict()
    pair_base = dataset.drop_duplicates("pair_id").set_index("pair_id")
    accepted = []
    rejected = []
    for para_idx, para in paraphrases.iterrows():
        pair_id = str(para["pair_id"])
        base_id = str(para.get("paraphrase_base_id", "") or "")
        if base_id:
            if base_id not in base.index:
                continue
            row = base.loc[base_id].to_dict()
        elif pair_counts.get(pair_id, 0) == 1 and pair_id in pair_base.index:
            row = pair_base.loc[pair_id].to_dict()
            base_id = str(row["paraphrase_base_id"])
        else:
            rejected.append(
                {
                    "pair_id": pair_id,
                    "instruction_text": str(para["instruction_text"]),
                    "rejection_reason": "ambiguous_pair_id_without_paraphrase_base_id",
                }
            )
            continue
        text = str(para["instruction_text"])
        check = instruction_text_is_consistent(text, row["instruction_spec_json"])
        grounded = ground_instruction_text(text, row["instruction_spec_json"])
        required_tags = set(spec_tags(row["instruction_spec_json"]))
        mentioned_tags = set(check.get("mentioned_tags", []))
        missing_tags = sorted(required_tags - mentioned_tags)
        out = dict(row)
        out["instruction_text"] = text
        out["instruction_id_key"] = f"{base_id}:llm:{para_idx}"
        out["template_id"] = para.get("template_id", "llm_paraphrase")
        out["language_source"] = "llm_paraphrase"
        out["paraphrase_base_id"] = base_id
        out["consistency_check_json"] = json.dumps(check, sort_keys=True)
        out["grounded_instruction_spec_json"] = grounded["instruction_spec_json"]
        out["recognized_instruction_tags"] = "|".join(grounded["recognized_tags"])
        if check["consistent"] and (allow_missing_tags or not missing_tags):
            accepted.append(out)
        else:
            reasons = []
            if check.get("extra_tags"):
                reasons.append("extra_tags:" + "|".join(check["extra_tags"]))
            if missing_tags and not allow_missing_tags:
                reasons.append("missing_tags:" + "|".join(missing_tags))
            out["rejection_reason"] = ";".join(reasons)
            rejected.append(out)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(accepted).to_csv(out, index=False)
    rejected_out = Path(rejected_out_path)
    rejected_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rejected).to_csv(rejected_out, index=False)
    print(f"Accepted {len(accepted)} paraphrases -> {out}")
    print(f"Rejected {len(rejected)} paraphrases -> {rejected_out}")


def _with_paraphrase_base_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "paraphrase_base_id" in out.columns:
        return out
    out["paraphrase_base_id"] = [
        _paraphrase_base_id(str(row["pair_id"]), str(row["instruction_spec_json"]))
        for _, row in out.iterrows()
    ]
    return out


def _paraphrase_base_id(pair_id: str, spec_json: str) -> str:
    digest = hashlib.sha1(spec_json.encode("utf-8")).hexdigest()[:12]
    return f"{pair_id}:{digest}"


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


def _prompt(paraphrase_base_id: str, pair_id: str, spec_json: str, seed_text: str, paraphrases_per_item: int) -> str:
    return (
        f"Rewrite the molecular editing instruction below into {paraphrases_per_item} concise natural-language paraphrases. "
        "Do not add any new chemistry goals, constraints, properties, or edit operations. "
        "Return JSON lines with fields paraphrase_base_id, pair_id, and instruction_text only.\n\n"
        f"paraphrase_base_id: {paraphrase_base_id}\n"
        f"pair_id: {pair_id}\n"
        f"Executable spec: {spec_json}\n"
        f"Seed instruction: {seed_text}\n"
    )


if __name__ == "__main__":
    main()
