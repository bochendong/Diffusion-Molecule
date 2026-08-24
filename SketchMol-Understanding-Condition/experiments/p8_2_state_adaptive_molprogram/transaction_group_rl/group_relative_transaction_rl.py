#!/usr/bin/env python3
"""Group-relative REINFORCE over executable source-only short transactions."""
from __future__ import annotations
import argparse, csv, json, math, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent; PROJECT=HERE.parents[2]
for path in (PROJECT/"experiments/unified_smiles_generator",PROJECT/"experiments/p8_1_1_short_transaction",PROJECT/"experiments/p8_1_9_transaction_outcome_distill"):
    sys.path.insert(0,str(path))
import build_teacher_pseudopairs as leakage  # noqa:E402
import sample_raw_transactions as transaction  # noqa:E402
import unified_smiles_generator as unified  # noqa:E402
import umtp_graph_action_policy as policy  # noqa:E402

def read(path):
    with path.open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))

def seq_logp(model,vocab,condition,programs,device):
    items=[{"condition":condition,"decoder_input_ids":np.asarray(vocab.encode(p,add_bos=True),dtype=np.int64),"target_ids":np.asarray(vocab.encode(p,add_eos=True),dtype=np.int64),"task_mode":unified.EDIT_MODE} for p in programs]
    batch={k:v.to(device) for k,v in unified.collate_batch(items,model.pad_id).items()}
    logits=model(batch["condition"],batch["decoder_input_ids"],condition_mask=batch["condition_mask"])
    selected=F.log_softmax(logits,dim=-1).gather(-1,batch["target_ids"].unsqueeze(-1)).squeeze(-1); mask=batch["target_ids"].ne(model.pad_id)
    return (selected*mask).sum(1)/mask.sum(1).clamp_min(1)

def components(row,smiles):
    clean=dict(row); clean["target_smiles"]=""; clean["reference_smiles"]=""
    m=unified.candidate_metrics(clean,smiles,source_similarity_threshold=0.65)
    try: prop=float(m.get("unified_property_success_fraction") or 0)
    except ValueError:prop=0
    try: sim=float(m.get("source_tanimoto") or 0)
    except ValueError:sim=0
    return [float(m.get("valid_smiles")=="True"),max(0,min(1,prop)),max(0,min(1,sim))]

def aggregate(values,mode,tau):
    if mode=="joint_bottleneck":return min(values)
    t=max(float(tau),1e-6); return -t*math.log(sum(math.exp(-v/t) for v in values)/len(values))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--base-checkpoint",required=True,type=Path);p.add_argument("--base-training-summary",required=True,type=Path);p.add_argument("--train-csv",required=True,type=Path);p.add_argument("--features-dir",required=True,type=Path);p.add_argument("--edit-eval-csv",required=True,type=Path);p.add_argument("--denovo-eval-csv",required=True,type=Path);p.add_argument("--output-dir",required=True,type=Path);p.add_argument("--audit-output",required=True,type=Path)
    p.add_argument("--reward-aggregation",choices=("joint_bottleneck","dense_softmin"),required=True);p.add_argument("--rollouts",type=int,default=4);p.add_argument("--max-prompts",type=int,default=128);p.add_argument("--lr",type=float,default=2e-6);p.add_argument("--temperature",type=float,default=.8);p.add_argument("--softmin-temperature",type=float,default=.25);p.add_argument("--reference-kl-weight",type=float,default=.05);p.add_argument("--sft-weight",type=float,default=.1);p.add_argument("--seed",type=int,default=7);p.add_argument("--device",default="auto");a=p.parse_args()
    if a.rollouts<4:raise SystemExit("P8.2 requires at least four source-only rollouts per prompt")
    unified.seed_everything(a.seed);device=unified.resolve_device(a.device);ck=unified.load_checkpoint(a.base_checkpoint)
    if ck is None:raise FileNotFoundError(a.base_checkpoint)
    summary=json.loads(a.base_training_summary.read_text());old_vocab=int(summary["old_vocab_size"]);vocab=unified.SmilesVocabulary.from_dict(ck["vocab"]);config=dict(ck["model_config"])
    model=unified.ConditionedSmilesDecoder(**config).to(device);model.load_state_dict(ck["model_state"]);ref=unified.ConditionedSmilesDecoder(**config).to(device);ref.load_state_dict(ck["model_state"]);ref.eval()
    for q in ref.parameters():q.requires_grad_(False)
    scope=policy.configure_trainable_scope(model,scope="source_action",old_vocab_size=old_vocab);opt=torch.optim.AdamW([q for q in model.parameters() if q.requires_grad],lr=a.lr,weight_decay=0)
    eval_rows=read(a.edit_eval_csv)+read(a.denovo_eval_csv);eval_ids={str(r.get(k,"") or "").strip() for r in eval_rows for k in leakage.ID_FIELDS if str(r.get(k,"") or "").strip()};eval_mols={x for r in eval_rows for x in (leakage.canonical(r.get("source_smiles","")),leakage.canonical(r.get("target_smiles",""))) if x}
    eligible=[];rejected_id=rejected_molecule=0
    for row in read(a.train_csv):
        ids={str(row.get(k,"") or "").strip() for k in leakage.ID_FIELDS if str(row.get(k,"") or "").strip()};src=leakage.canonical(row.get("source_smiles",""));tgt=leakage.canonical(row.get("target_smiles",""))
        if ids&eval_ids:rejected_id+=1;continue
        if src in eval_mols or (tgt and tgt in eval_mols):rejected_molecule+=1;continue
        eligible.append(row)
    eligible=eligible[:a.max_prompts];store=unified.FeatureStore(a.features_dir,array_name="query_tokens",variant="full");gen=torch.Generator(device="cpu").manual_seed(a.seed);history=[];active=skipped=0;all_rewards=[]
    model.train()
    for i,row in enumerate(eligible):
        candidates=transaction.source_only_candidates(row,site_limit=32,limit=512)
        if len(candidates)<a.rollouts:skipped+=1;continue
        condition=unified.condition_array_for_row(row,store,int(config["condition_dim"]),max_source_tokens=96,condition_layout="direct_compat").astype(np.float32);programs=[x[2] for x in candidates]
        with torch.no_grad():base_scores=seq_logp(model,vocab,condition,programs,device);probs=torch.softmax(base_scores.cpu()/max(a.temperature,1e-6),dim=0);chosen=torch.multinomial(probs,a.rollouts,replacement=False,generator=gen).tolist()
        selected_programs=[programs[j] for j in chosen];rewards=torch.tensor([aggregate(components(row,candidates[j][1]),a.reward_aggregation,a.softmin_temperature) for j in chosen],dtype=torch.float32,device=device);adv=(rewards-rewards.mean())/rewards.std(unbiased=False).clamp_min(1e-4)
        current=seq_logp(model,vocab,condition,selected_programs,device)
        with torch.no_grad():reference=seq_logp(ref,vocab,condition,selected_programs,device)
        reinforce=-(adv.detach()*current).mean()
        # A true categorical KL on the sampled source-only support group.
        # Reference probabilities are fixed; the policy distribution receives gradients.
        reference_kl=F.kl_div(F.log_softmax(current,dim=0),F.softmax(reference,dim=0),reduction="sum")
        try:oracle=json.loads(str(row.get("policy_target_tokens_json") or "[]"))
        except json.JSONDecodeError:oracle=[]
        sft=-seq_logp(model,vocab,condition,[oracle],device).mean() if oracle else current.new_zeros(())
        loss=reinforce+a.reference_kl_weight*reference_kl+a.sft_weight*sft;opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_([q for q in model.parameters() if q.requires_grad],1);opt.step()
        active+=int(float(rewards.std(unbiased=False))>1e-6);all_rewards.extend(rewards.tolist());history.append({"prompt":i,"loss":float(loss.detach()),"reward_mean":float(rewards.mean()),"reward_std":float(rewards.std(unbiased=False))})
        if (i+1)%16==0:print(f"[p8.2-transaction-rl] {i+1}/{len(eligible)} reward={sum(all_rewards)/len(all_rewards):.4f}",flush=True)
    if not history:raise SystemExit("No eligible source-only rollout groups")
    a.output_dir.mkdir(parents=True,exist_ok=True);out=a.output_dir/"umtp_graph_action_policy.pt";unified.save_checkpoint(out,model,opt,vocab,config,1,history,a)
    audit={"protocol":"p8_2_transaction_group_relative_reinforce_v1","algorithm":"group_relative_REINFORCE_not_GRPO","reward_aggregation":a.reward_aggregation,"rollouts_per_prompt":a.rollouts,"eligible_train_prompts":len(eligible),"updated_prompts":len(history),"active_reward_groups":active,"skipped_small_support":skipped,"rejected_eval_id_overlap":rejected_id,"rejected_eval_source_or_target_overlap":rejected_molecule,"eval_rows_used":0,"eval_targets_used_for_training":0,"eval_targets_used_for_exclusion_only":True,"target_structure_reward_access":False,"source_only_rollout_support":True,"trainable_scope":scope,"reference_penalty":"reference-to-policy categorical KL on sampled source-only group","sft_anchor":True,"property_rerank_at_inference":False,"checkpoint":str(out)}
    a.audit_output.parent.mkdir(parents=True,exist_ok=True);a.audit_output.write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n");(a.output_dir/"training_summary.json").write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n");print(json.dumps(audit,indent=2,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
