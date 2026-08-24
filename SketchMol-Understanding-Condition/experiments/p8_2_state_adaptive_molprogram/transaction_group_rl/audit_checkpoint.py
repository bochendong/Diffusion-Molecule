#!/usr/bin/env python3
"""P8.2 bit-exact legacy path and shared-checkpoint audit."""
import argparse,csv,hashlib,json
from pathlib import Path
import torch
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument("--legacy-base",required=True,type=Path);p.add_argument("--rl-base",required=True,type=Path);p.add_argument("--checkpoint",required=True,type=Path);p.add_argument("--training-audit",required=True,type=Path);p.add_argument("--denovo-summary",type=Path);p.add_argument("--edit-summary",type=Path);p.add_argument("--output",required=True,type=Path);a=p.parse_args()
 legacy=torch.load(a.legacy_base,map_location="cpu",weights_only=False);base=torch.load(a.rl_base,map_location="cpu",weights_only=False);trained=torch.load(a.checkpoint,map_location="cpu",weights_only=False);old_vocab=len(legacy["vocab"]);bad=[]
 for n,t in legacy["model_state"].items():
  if n not in trained["model_state"]:bad.append("missing:"+n);continue
  q=trained["model_state"][n]
  if n in {"token_embedding.weight","output.weight","output.bias"}:q=q[tuple(slice(0,s) for s in t.shape)]
  if q.shape!=t.shape or not torch.equal(q.cpu(),t.cpu()):bad.append(n)
 changed=[]
 for n,t in base["model_state"].items():
  q=trained["model_state"].get(n)
  if q is None or not torch.equal(t,q):changed.append(n)
 allowed=("source_condition_proj.","source_encoder.","source_type","null_source","source_gate.","source_output.","token_embedding.weight")
 illegal=[n for n in changed if not n.startswith(allowed)];ta=json.loads(a.training_audit.read_text());hashes=[]
 for f in (a.denovo_summary,a.edit_summary):
  if f:hashes.append(json.loads(f.read_text())["checkpoint_sha256"])
 checks={"legacy_denovo_parameters_bit_exact":not bad,"only_source_memory_or_transaction_embedding_changed":not illegal,"one_checkpoint_both_arms":not hashes or all(x==sha(a.checkpoint) for x in hashes),"at_least_four_rollouts":ta.get("rollouts_per_prompt",0)>=4,"no_eval_rows_or_targets_for_training":ta.get("eval_rows_used")==0 and ta.get("eval_targets_used_for_training")==0,"no_target_reward":not ta.get("target_structure_reward_access"),"no_property_rerank":not ta.get("property_rerank_at_inference")}
 out={"protocol":"p8_2_transaction_rl_checkpoint_audit_v1","status":"pass" if all(checks.values()) else "fail","checks":checks,"legacy_mismatches":bad,"changed_from_rl_base":changed,"illegal_changes":illegal,"checkpoint_sha256":sha(a.checkpoint)};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n");print(json.dumps(out,indent=2,sort_keys=True));return 0 if out["status"]=="pass" else 2
if __name__=="__main__":raise SystemExit(main())
