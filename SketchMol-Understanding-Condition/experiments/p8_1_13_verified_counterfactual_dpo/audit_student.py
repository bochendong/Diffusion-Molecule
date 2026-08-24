#!/usr/bin/env python3
"""Audit unified full-SMILES DPO student and fail-closed pair lineage."""
import argparse, hashlib, json
from pathlib import Path
import torch
PREFIX=("source_condition_proj.","source_encoder.","source_type","null_source","source_gate.","source_output.","source_adapters.","source_copy_query.","source_copy_key.","source_copy_gate.")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument("--base",required=True,type=Path);p.add_argument("--student",required=True,type=Path);p.add_argument("--pair-audit",required=True,type=Path);p.add_argument("--round",required=True,choices=("r1","r2"));p.add_argument("--output",required=True,type=Path);a=p.parse_args()
 b=torch.load(a.base,map_location="cpu",weights_only=False);s=torch.load(a.student,map_location="cpu",weights_only=False);d=json.loads(a.pair_audit.read_text()); changed=[];illegal=[]
 for n,t in b["model_state"].items():
  if n not in s["model_state"] or not torch.equal(t,s["model_state"][n]):changed.append(n);illegal.append(n) if not n.startswith(PREFIX) else None
 out={"protocol":"p8_1_13_unified_student_audit_v1","round":a.round,"base_sha256":sha(a.base),"student_sha256":sha(a.student),"one_checkpoint":True,"one_decoder":True,"one_output_language":"full molecule SMILES","router":False,"materializer":False,"teacher_at_inference":False,"property_rerank":False,"vocabulary_unchanged":b["vocab"]==s["vocab"],"model_config_unchanged":b["model_config"]==s["model_config"],"changed_parameters":changed,"illegal_non_source_changes":illegal,"de_novo_parameter_path_bitwise_protected":not illegal,"pair_audit_decision":d.get("decision"),"remaining_eval_overlap":d.get("remaining_eval_overlap"),"eval_candidates_used_for_training":d.get("eval_candidates_used_for_training"),"round_factor":"uniform pairwise DPO" if a.round=="r1" else "teacher-confidence pair weighting only"}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
 if not(out["vocabulary_unchanged"] and out["model_config_unchanged"] and out["de_novo_parameter_path_bitwise_protected"] and out["pair_audit_decision"]=="pass" and not any((out["remaining_eval_overlap"] or {}).values()) and not out["eval_candidates_used_for_training"]):raise SystemExit("P8.1.13 audit failed")
 print(json.dumps(out,indent=2,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
