#!/usr/bin/env python3
"""Reference-based, length-normalized DPO on verified counterfactual pairs."""
from __future__ import annotations
import argparse, csv, json, os, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent; PROJECT=HERE.parents[1]
for path in (PROJECT,PROJECT/"experiments/unified_smiles_generator",PROJECT/"experiments/p8_1_7_source_clamped_policy"):
    sys.path.insert(0,str(path))
os.environ["P817_SOURCE_CLAMP_SCALE"]="1.0"
import source_clamped_entrypoint  # noqa: F401,E402
import unified_smiles_generator as unified  # noqa: E402

def rows(path):
    with path.open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))

def seq_logp(model,batch,eos):
    logits=model(batch["condition"],batch["decoder_input_ids"],condition_mask=batch["condition_mask"],source_token_ids=batch["source_token_ids"])
    selected=F.log_softmax(logits,dim=-1).gather(-1,batch["target_ids"].unsqueeze(-1)).squeeze(-1)
    mask=batch["target_ids"].ne(model.pad_id)
    return (selected*mask).sum(dim=1)/mask.sum(dim=1).clamp_min(1)

def main()->int:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--base-checkpoint",required=True,type=Path); p.add_argument("--pairs-csv",required=True,type=Path)
    p.add_argument("--features-dir",required=True,type=Path); p.add_argument("--denovo-rehearsal-csv",required=True,type=Path); p.add_argument("--denovo-features-dir",required=True,type=Path)
    p.add_argument("--output-dir",required=True,type=Path); p.add_argument("--pair-weighting",choices=("uniform","teacher_confidence"),required=True)
    p.add_argument("--beta",type=float,default=0.1); p.add_argument("--epochs",type=int,default=4); p.add_argument("--batch-size",type=int,default=32); p.add_argument("--lr",type=float,default=8e-5)
    p.add_argument("--max-pairs",type=int,default=768); p.add_argument("--max-rehearsal",type=int,default=128); p.add_argument("--rehearsal-weight",type=float,default=0.1); p.add_argument("--seed",type=int,default=7); p.add_argument("--device",default="auto"); a=p.parse_args()
    unified.seed_everything(a.seed); device=unified.resolve_device(a.device); ck=unified.load_checkpoint(a.base_checkpoint)
    if ck is None: raise FileNotFoundError(a.base_checkpoint)
    vocab=unified.SmilesVocabulary.from_dict(ck["vocab"]); config=dict(ck["model_config"])
    model=unified.ConditionedSmilesDecoder(**config).to(device); model.load_state_dict(ck["model_state"])
    reference=unified.ConditionedSmilesDecoder(**config).to(device); reference.load_state_dict(ck["model_state"]); reference.eval()
    for q in reference.parameters():q.requires_grad_(False)
    scope=unified.configure_trainable_scope(model,"source_only"); opt=torch.optim.AdamW([q for q in model.parameters() if q.requires_grad],lr=a.lr,weight_decay=0)
    pair_rows=rows(a.pairs_csv)[:a.max_pairs]; store=unified.FeatureStore(a.features_dir,array_name="query_tokens",variant="full")
    positives=[]; negatives=[]
    for row in pair_rows:
        rp=dict(row); rp["target_smiles"]=row["positive_smiles"]; rn=dict(row); rn["target_smiles"]=row["negative_smiles"]
        positives.extend(unified.build_dataset([rp],vocab,store,int(config["condition_dim"]),max_smiles_length=160,max_source_tokens=96,condition_layout="unified"))
        negatives.extend(unified.build_dataset([rn],vocab,store,int(config["condition_dim"]),max_smiles_length=160,max_source_tokens=96,condition_layout="unified"))
    if len(positives)!=len(pair_rows) or len(negatives)!=len(pair_rows):raise SystemExit("pair dataset mismatch")
    rehearsal_rows=[r for r in rows(a.denovo_rehearsal_csv) if unified.task_mode_for_row(r)==unified.DE_NOVO_MODE][:a.max_rehearsal]
    rehearsal_store=unified.FeatureStore(a.denovo_features_dir,array_name="query_tokens",variant="full")
    rehearsal=unified.build_dataset(rehearsal_rows,vocab,rehearsal_store,int(config["condition_dim"]),max_smiles_length=160,max_source_tokens=96,condition_layout="unified")
    rng=np.random.default_rng(a.seed); history=[]; model.train()
    for epoch in range(a.epochs):
        order=rng.permutation(len(positives)); losses=[]; prefs=[]; rehearsals=[]
        for start in range(0,len(order),a.batch_size):
            ids=order[start:start+a.batch_size].tolist(); pb={k:v.to(device) for k,v in unified.collate_batch([positives[i] for i in ids],model.pad_id).items()}; nb={k:v.to(device) for k,v in unified.collate_batch([negatives[i] for i in ids],model.pad_id).items()}
            pi_delta=seq_logp(model,pb,vocab.eos_id)-seq_logp(model,nb,vocab.eos_id)
            with torch.no_grad(): ref_delta=seq_logp(reference,pb,vocab.eos_id)-seq_logp(reference,nb,vocab.eos_id)
            per=-F.logsigmoid(a.beta*(pi_delta-ref_delta))
            if a.pair_weighting=="teacher_confidence":
                w=torch.tensor([float(pair_rows[i]["pair_teacher_confidence"]) for i in ids],dtype=per.dtype,device=device); w=w/w.mean().clamp_min(1e-8)
            else:w=torch.ones_like(per)
            dpo=(w*per).mean(); rehearsal_loss=dpo.new_zeros(())
            if rehearsal:
                rid=[(start+j)%len(rehearsal) for j in range(min(len(ids),len(rehearsal)))]
                rb={k:v.to(device) for k,v in unified.collate_batch([rehearsal[i] for i in rid],model.pad_id).items()}
                rehearsal_loss=-seq_logp(model,rb,vocab.eos_id).mean()
            loss=dpo+a.rehearsal_weight*rehearsal_loss; opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_([q for q in model.parameters() if q.requires_grad],1.0); opt.step()
            losses.append(float(loss.detach())); prefs.append(float((pi_delta>0).float().mean())); rehearsals.append(float(rehearsal_loss.detach()))
        item={"epoch":epoch+1,"loss":sum(losses)/len(losses),"policy_prefers_positive":sum(prefs)/len(prefs),"denovo_rehearsal_loss":sum(rehearsals)/max(len(rehearsals),1)}; history.append(item); print(item,flush=True)
    a.output_dir.mkdir(parents=True,exist_ok=True); path=a.output_dir/"unified_smiles_generator.pt"; unified.save_checkpoint(path,model,opt,vocab,config,a.epochs,history,a)
    summary={"protocol":"p8_1_13_length_normalized_dpo_v1","algorithm":"reference_based_DPO","pair_weighting":a.pair_weighting,"pairs":len(positives),"fixed_denovo_rehearsal_rows":len(rehearsal),"trainable_scope":scope,"one_checkpoint":True,"one_decoder":True,"output_language":"full molecule SMILES","eval_rows_used":0,"eval_targets_used":0,"teacher_at_inference":False,"history":history}
    (a.output_dir/"training_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n"); print(json.dumps(summary,indent=2,sort_keys=True)); return 0
if __name__=="__main__":raise SystemExit(main())
