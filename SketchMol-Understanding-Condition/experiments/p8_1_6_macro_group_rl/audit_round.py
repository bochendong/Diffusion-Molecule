#!/usr/bin/env python3
"""Audit unified checkpoint structure and RDKit-valid identity behavior."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import torch
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def canonicalize(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    molecule = Chem.MolFromSmiles(text, sanitize=True)
    if molecule is None:
        return ""
    try:
        Chem.SanitizeMol(molecule)
        return str(Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True))
    except Exception:
        return ""

def main() -> int:
    p=argparse.ArgumentParser();p.add_argument("--base",required=True,type=Path);p.add_argument("--checkpoint",required=True,type=Path);p.add_argument("--candidates",required=True,type=Path);p.add_argument("--expected-aggregation",required=True);p.add_argument("--output",required=True,type=Path);a=p.parse_args()
    base=torch.load(a.base,map_location="cpu",weights_only=False);ck=torch.load(a.checkpoint,map_location="cpu",weights_only=False);state=ck["model_state"];args=dict(ck.get("args",{}));config=dict(ck.get("model_config",{}))
    with a.candidates.open(newline="",encoding="utf-8") as h: rows=list(csv.DictReader(h))
    valid = identity = invalid = 0
    for row in rows:
        candidate = canonicalize(
            row.get("direct_candidate_raw_smiles")
            or row.get("generated_smiles")
            or row.get("direct_candidate_canonical_smiles")
        )
        source = canonicalize(row.get("source_smiles"))
        valid += int(bool(candidate))
        invalid += int(not candidate)
        identity += int(bool(candidate) and bool(source) and candidate == source)
    payload={"protocol":"p8_1_6_unified_audit_v2_rdkit","validity_definition":"RDKit MolFromSmiles plus sanitize","one_checkpoint":True,"one_decoder":"decoder.layers.0.self_attn.in_proj_weight" in state,"one_output_softmax":"output.weight" in state,"same_smiles_vocabulary":base.get("vocab")==ck.get("vocab"),"vocab_size":len(ck.get("vocab",{})),"new_output_tokens":len(ck.get("vocab",{}))-len(base.get("vocab",{})),"condition_layout":args.get("condition_layout"),"explicit_task_token":args.get("condition_layout")=="unified","reward_aggregation":args.get("reward_aggregation"),"expected_reward_aggregation":a.expected_aggregation,"source_copy_pointer":bool(config.get("source_copy_aware",False)),"router":False,"interpreter":False,"materializer":False,"property_rerank":False,"candidate_rows":len(rows),"valid_candidates":valid,"invalid_candidates":invalid,"identity_valid_candidates":identity,"candidate_validity":valid/max(len(rows),1),"identity_copy_rate":identity/max(valid,1),"nonidentity_valid_rate":(valid-identity)/max(len(rows),1)}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n");print(json.dumps(payload,indent=2,sort_keys=True))
    if not(payload["one_decoder"] and payload["one_output_softmax"] and payload["same_smiles_vocabulary"] and payload["new_output_tokens"]==0 and payload["explicit_task_token"] and not payload["source_copy_pointer"] and payload["reward_aggregation"]==a.expected_aggregation): raise SystemExit("unification audit failed")
    return 0
if __name__=="__main__":raise SystemExit(main())
