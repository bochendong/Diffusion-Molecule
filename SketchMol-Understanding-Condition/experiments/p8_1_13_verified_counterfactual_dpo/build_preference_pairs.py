#!/usr/bin/env python3
"""Pair P8.1.12 verified positives with valid same-prompt hard negatives."""
from __future__ import annotations

import argparse, csv, hashlib, json, math, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
for path in (PROJECT, PROJECT / "experiments/unified_smiles_generator",
             PROJECT / "experiments/p8_1_1_short_transaction",
             PROJECT / "experiments/p8_1_9_transaction_outcome_distill"):
    sys.path.insert(0, str(path))
import build_teacher_pseudopairs as common  # noqa: E402
import sample_raw_transactions as sampler  # noqa: E402
import unified_smiles_generator as unified  # noqa: E402
import umtp_graph_action_policy as policy  # noqa: E402


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as f: return list(csv.DictReader(f))


def write_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--p8112-positive-csv", required=True, type=Path)
    p.add_argument("--p8112-audit", required=True, type=Path)
    p.add_argument("--teacher-checkpoint", required=True, type=Path)
    p.add_argument("--student-checkpoint", required=True, type=Path)
    p.add_argument("--features-dir", required=True, type=Path)
    p.add_argument("--eval-csv", required=True, type=Path)
    p.add_argument("--output-csv", required=True, type=Path)
    p.add_argument("--audit-output", required=True, type=Path)
    p.add_argument("--site-limit", type=int, default=32); p.add_argument("--max-actions", type=int, default=512)
    p.add_argument("--score-batch-size", type=int, default=256); p.add_argument("--device", default="auto")
    a=p.parse_args()
    upstream=json.loads(a.p8112_audit.read_text())
    if upstream.get("decision") != "pass" or any((upstream.get("remaining_eval_overlap") or {}).values()):
        raise SystemExit("P8.1.12 audit is absent or not fail-closed PASS")
    positives=read_rows(a.p8112_positive_csv); eval_rows=read_rows(a.eval_csv)
    eval_ids={str(r.get(k,"") or "").strip() for r in eval_rows for k in common.ID_FIELDS if str(r.get(k,"") or "").strip()}
    eval_mols={x for r in eval_rows for x in (common.canonical(r.get("source_smiles","")),common.canonical(r.get("target_smiles",""))) if x}
    ckpt=unified.load_checkpoint(a.teacher_checkpoint); student=unified.load_checkpoint(a.student_checkpoint)
    if ckpt is None or student is None: raise FileNotFoundError("teacher/student checkpoint")
    device=unified.resolve_device(a.device); vocab=unified.SmilesVocabulary.from_dict(ckpt["vocab"])
    student_vocab=unified.SmilesVocabulary.from_dict(student["vocab"]); config=dict(ckpt["model_config"])
    teacher=unified.ConditionedSmilesDecoder(**config).to(device); teacher.load_state_dict(ckpt["model_state"]); teacher.eval()
    store=unified.FeatureStore(a.features_dir,array_name="query_tokens",variant="full")
    pairs=[]; rejected_ids=rejected_mols=rejected_oov=no_negative=0; enumerated=valid_failed=0
    for i,row in enumerate(positives):
        ids={str(row.get(k,"") or "").strip() for k in common.ID_FIELDS if str(row.get(k,"") or "").strip()}
        source=common.canonical(row.get("source_smiles","")); positive=common.canonical(row.get("target_smiles",""))
        if ids & eval_ids: rejected_ids+=1; continue
        if not source or source in eval_mols or positive in eval_mols: rejected_mols+=1; continue
        pm=unified.candidate_metrics(row,positive,source_similarity_threshold=0.65)
        if pm.get("table1_strict_success") != "True" or positive == source:
            raise SystemExit("upstream positive is not strict-success nonidentity")
        candidates=[]
        for action,outcome,program in sampler.source_only_candidates(row,site_limit=a.site_limit,limit=a.max_actions):
            enumerated+=1; outcome=common.canonical(outcome)
            if not outcome or outcome in {source,positive} or outcome in eval_mols: continue
            metrics=unified.candidate_metrics(row,outcome,source_similarity_threshold=0.65)
            if metrics.get("valid_smiles") != "True" or metrics.get("table1_strict_success") == "True": continue
            if any(t not in student_vocab.token_to_id for t in unified.tokenize_smiles(outcome)): rejected_oov+=1; continue
            valid_failed+=1
            try: sim=float(metrics.get("source_tanimoto") or 0.0)
            except ValueError: sim=0.0
            try: prop=float(metrics.get("unified_property_success_fraction") or 0.0)
            except ValueError: prop=0.0
            candidates.append((sim,prop,outcome,program,metrics))
        if not candidates: no_negative+=1; continue
        # Same-prompt hardest negative: closest to source, then closest to full property success.
        candidates.sort(key=lambda x:(x[0],x[1]),reverse=True)
        sim,prop,negative,nprogram,nm=candidates[0]
        condition=unified.condition_array_for_row(row,store,int(config["condition_dim"]),max_source_tokens=96,condition_layout="direct_compat").astype(np.float32)
        try: pprogram=json.loads(str(row.get("teacher_program_tokens_json") or "[]"))
        except json.JSONDecodeError: pprogram=[]
        scores=policy.score_programs(teacher,vocab,condition,[pprogram,nprogram],batch_size=a.score_batch_size,device=device)
        confidence=float(torch.sigmoid(torch.tensor(float(scores[0])-float(scores[1]))))
        # Confidence is the sole R2 weighting signal; normalize later in the trainer.
        item=dict(row); item.update({
            "positive_smiles":positive,"negative_smiles":negative,
            "hard_negative_source_tanimoto":f"{sim:.8g}","hard_negative_property_fraction":f"{prop:.8g}",
            "hard_negative_table1_strict_success":"False","pair_teacher_confidence":f"{confidence:.8g}",
            "positive_teacher_logprob":f"{float(scores[0]):.8g}","negative_teacher_logprob":f"{float(scores[1]):.8g}",
            "pair_origin":"same_train_prompt_verified_counterfactual","eval_target_used":"False",
        }); pairs.append(item)
        if (i+1)%50==0: print(f"[p8.1.13] {i+1}/{len(positives)} pairs={len(pairs)}",flush=True)
    if not pairs: raise SystemExit("No verified counterfactual pairs")
    write_rows(a.output_csv,pairs)
    out_ids={str(r.get(k,"") or "").strip() for r in pairs for k in common.ID_FIELDS if str(r.get(k,"") or "").strip()}
    out_mols={common.canonical(str(r.get(k,"") or "")) for r in pairs for k in ("source_smiles","positive_smiles","negative_smiles")}
    overlap={"id":sorted(out_ids & eval_ids),"source_positive_or_negative":sorted(out_mols & eval_mols)}
    audit={"protocol":"p8_1_13_verified_counterfactual_preference_v1","decision":"pass" if not any(overlap.values()) else "fail",
      "upstream_protocol":upstream.get("protocol"),"upstream_positive_rows":len(positives),"preference_pairs":len(pairs),
      "candidate_enumerated":enumerated,"valid_failed_candidates":valid_failed,"rows_without_negative":no_negative,
      "rejected_eval_identifier":rejected_ids,"rejected_eval_molecule":rejected_mols,"rejected_student_vocab_oov":rejected_oov,
      "remaining_eval_overlap":overlap,"official_positive_predicate":"table1_strict_success at source similarity 0.65 and nonidentity",
      "negative_predicate":"valid and official strict failure under same source/program condition",
      "hard_negative_order":"max source Tanimoto then max property success fraction","eval_candidates_used_for_training":False,
      "eval_targets_used_for_training":False,"teacher_used_at_inference":False,"property_reranking_at_inference":False,
      "r1_factor":"uniform DPO pairs","r2_single_factor":"pair_teacher_confidence loss weighting"}
    a.audit_output.parent.mkdir(parents=True,exist_ok=True); a.audit_output.write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n")
    print(json.dumps(audit,indent=2,sort_keys=True))
    if audit["decision"] != "pass": raise SystemExit("P8.1.13 fail-closed leakage audit")
    return 0
if __name__=="__main__": raise SystemExit(main())
