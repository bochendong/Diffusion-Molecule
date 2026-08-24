#!/usr/bin/env python3
"""Fail-closed support and one-factor audit for P8.1.6."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

def count(path: Path) -> dict[str, int]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out = {"rows": len(rows), "de_novo": 0, "edit": 0}
    for row in rows:
        mode = str(row.get("task_mode", ""))
        out["edit" if mode == "edit" else "de_novo"] += 1
    return out

def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--macro-r1", required=True, type=Path); p.add_argument("--macro-r2", required=True, type=Path); p.add_argument("--mixed", required=True, type=Path); p.add_argument("--features", required=True, type=Path); p.add_argument("--output", required=True, type=Path); a=p.parse_args()
    r1=json.loads(a.macro_r1.read_text()); r2=json.loads(a.macro_r2.read_text()); support=count(a.mixed)
    payload={"protocol":"p8_1_6_macro_group_rl_preflight_v1","p8_1_3_failure":{"r1_decision":r1.get("decision"),"de_novo_holdout_vocab":r1["heldout_train_only_vocabulary"]["by_mode"]["de_novo"]["vocab_reachability"],"r2_denovo_exact":r2["by_mode"]["de_novo"]["exact_reconstruction"]},"avoidance":{"new_output_tokens":0,"macro_vocabulary":False,"interpreter":False,"materializer":False},"support":support,"feature_files":{name:(a.features/name).exists() for name in ("index.csv","query_tokens.npy","pooled.npy")},"shared_config":{"initial_checkpoint":"P8.1.4-R1","decoder":"one full-SMILES decoder","condition_layout":"unified","raw_candidates":20,"property_rerank":False},"rounds":{"R1":{"reward_aggregation":"joint_bottleneck"},"R2":{"reward_aggregation":"dense_softmin"}},"single_changed_factor":"trajectory reward aggregation"}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n");print(json.dumps(payload,indent=2,sort_keys=True))
    if support["de_novo"] == 0 or support["edit"] == 0 or not all(payload["feature_files"].values()): raise SystemExit("mixed support preflight failed")
    return 0
if __name__ == "__main__": raise SystemExit(main())
