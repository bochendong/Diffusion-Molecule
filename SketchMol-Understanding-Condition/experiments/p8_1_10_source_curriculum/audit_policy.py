#!/usr/bin/env python3
"""Audit P1 protection, P8.1.7 de-novo retention, and honest edit validity."""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
import torch
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
SOURCE_PREFIXES=("source_condition_proj.","source_encoder.","source_type","null_source","source_gate.","source_output.","source_adapters.","source_copy_")

def rows(path):
    with path.open(newline="",encoding="utf-8") as h:return list(csv.DictReader(h))
def canon(value):
    m=Chem.MolFromSmiles(str(value or "").strip());return Chem.MolToSmiles(m,canonical=True,isomericSmiles=True) if m is not None else ""
def denovo_hash(path):
    fields=[]
    for r in rows(path):fields.append((str(r.get("condition_id") or r.get("sample_id") or ""),str(r.get("direct_candidate_index") or ""),str(r.get("direct_candidate_canonical_smiles") or ""),str(r.get("direct_candidate_strict_fraction") or "")))
    return hashlib.sha256(json.dumps(fields,separators=(",",":"),ensure_ascii=True).encode()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument("--p1",required=True,type=Path);p.add_argument("--base",required=True,type=Path);p.add_argument("--checkpoint",required=True,type=Path);p.add_argument("--denovo",required=True,type=Path);p.add_argument("--denovo-reference",required=True,type=Path);p.add_argument("--edit",required=True,type=Path);p.add_argument("--round",required=True);p.add_argument("--output",required=True,type=Path);a=p.parse_args()
    p1=torch.load(a.p1,map_location="cpu",weights_only=False);base=torch.load(a.base,map_location="cpu",weights_only=False);ck=torch.load(a.checkpoint,map_location="cpu",weights_only=False);state=ck["model_state"]
    common=sorted(set(p1["model_state"])&set(state));changed=[k for k in common if not torch.equal(p1["model_state"][k],state[k])]
    source_changed=[k for k in set(base["model_state"])&set(state) if k.startswith(SOURCE_PREFIXES) and not torch.equal(base["model_state"][k],state[k])]
    valid=identity=strict=relaxed=0;edit_rows=rows(a.edit)
    for r in edit_rows:
        candidate=canon(r.get("direct_candidate_raw_smiles") or r.get("generated_smiles") or r.get("direct_candidate_canonical_smiles"));source=canon(r.get("source_smiles"));is_identity=bool(candidate and source and candidate==source);valid+=int(bool(candidate));identity+=int(is_identity);strict+=int(str(r.get("table1_strict_success","")).lower()=="true" and not is_identity);relaxed+=int(str(r.get("table1_relaxed_success","")).lower()=="true" and not is_identity)
    h=denovo_hash(a.denovo);ref=denovo_hash(a.denovo_reference)
    config=dict(ck.get("model_config",{}));payload={"protocol":"p8_1_10_protection_audit_v1","round":a.round,"one_checkpoint":True,"one_decoder":"decoder.layers.0.self_attn.in_proj_weight" in state,"one_softmax":"output.weight" in state,"same_p1_smiles_vocabulary":p1.get("vocab")==ck.get("vocab"),"p1_common_tensors":len(common),"p1_common_tensors_changed":changed,"p1_shared_path_bit_exact":not changed,"source_parameters_changed":len(source_changed),"null_source_condition_layout":"P8.1.7 direct-compatible base plus property program","denovo_semantic_sha256":h,"p817_reference_denovo_sha256":ref,"denovo_candidates_bit_exact":h==ref,"source_copy_pointer":bool(config.get("source_copy_aware",False)),"router":False,"interpreter":False,"materializer":False,"property_rerank":False,"edit_candidate_rows":len(edit_rows),"edit_validity":valid/max(len(edit_rows),1),"identity_fraction":identity/max(len(edit_rows),1),"strict_nonidentity_fraction":strict/max(len(edit_rows),1),"relaxed_nonidentity_fraction":relaxed/max(len(edit_rows),1)}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n");print(json.dumps(payload,indent=2,sort_keys=True))
    if not(payload["one_decoder"] and payload["one_softmax"] and payload["same_p1_smiles_vocabulary"] and payload["p1_shared_path_bit_exact"] and payload["denovo_candidates_bit_exact"] and not payload["source_copy_pointer"]):raise SystemExit("P8.1.10 protection audit failed")
    return 0
if __name__=="__main__":raise SystemExit(main())
