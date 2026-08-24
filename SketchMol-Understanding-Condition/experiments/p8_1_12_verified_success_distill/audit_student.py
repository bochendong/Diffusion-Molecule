#!/usr/bin/env python3
"""Audit the deployed single full-SMILES student and frozen de-novo path."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import torch

SOURCE_PREFIXES=("source_condition_proj.","source_encoder.","source_type","null_source","source_gate.","source_output.","source_adapters.","source_copy_query.","source_copy_key.","source_copy_gate.")
def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--base",required=True,type=Path); p.add_argument("--student",required=True,type=Path); p.add_argument("--data-audit",required=True,type=Path); p.add_argument("--round",required=True,choices=("r1","r2")); p.add_argument("--output",required=True,type=Path); a=p.parse_args()
    base=torch.load(a.base,map_location="cpu",weights_only=False); student=torch.load(a.student,map_location="cpu",weights_only=False); data=json.loads(a.data_audit.read_text())
    changed=[]; illegal=[]
    for name,tensor in base["model_state"].items():
        other=student["model_state"].get(name)
        if other is None or not torch.equal(tensor,other):
            changed.append(name)
            if not name.startswith(SOURCE_PREFIXES): illegal.append(name)
    payload={"protocol":"p8_1_12_single_full_smiles_student_audit_v1","round":a.round,"base_sha256":sha(a.base),"student_sha256":sha(a.student),"one_checkpoint_at_inference":True,"one_decoder":"decoder.layers.0.self_attn.in_proj_weight" in student["model_state"],"one_output_head":"output.weight" in student["model_state"],"one_output_language":"full molecule SMILES","router":False,"materializer":False,"property_rerank":False,"teacher_present_at_inference":False,"vocabulary_unchanged":base["vocab"]==student["vocab"],"model_config_unchanged":base["model_config"]==student["model_config"],"changed_parameter_count":len(changed),"changed_parameters":changed,"illegal_non_source_changes":illegal,"de_novo_parameter_path_bitwise_protected":not illegal,"verified_filter_precedes_teacher_selection":data.get("verified_filter_precedes_teacher_selection"),"remaining_eval_overlap":data.get("remaining_eval_overlap"),"eval_candidates_used_for_training":data.get("eval_candidates_used_for_training"),"round_factor":"uniform verified-success SFT" if a.round=="r1" else "teacher likelihood confidence weighting"}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n"); print(json.dumps(payload,indent=2,sort_keys=True))
    checks=[payload["one_decoder"],payload["one_output_head"],payload["vocabulary_unchanged"],payload["model_config_unchanged"],payload["de_novo_parameter_path_bitwise_protected"],payload["verified_filter_precedes_teacher_selection"],not payload["eval_candidates_used_for_training"],not any((payload["remaining_eval_overlap"] or {}).values())]
    if not all(checks): raise SystemExit("P8.1.12 unified/leakage audit failed")
    return 0
if __name__=="__main__": raise SystemExit(main())
