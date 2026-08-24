#!/usr/bin/env python3
from pathlib import Path
R=Path(__file__).resolve().parent
def main():
 b=(R/"build_preference_pairs.py").read_text();t=(R/"train_dpo.py").read_text();run=(R/"run_round.sh").read_text();s=(R/"submit_queue.sh").read_text()
 assert 'pm.get("table1_strict_success") != "True"' in b and 'outcome in {source,positive}' in b
 assert 'metrics.get("valid_smiles") != "True"' in b and 'metrics.get("table1_strict_success") == "True"' in b
 assert 'outcome in eval_mols' in b and 'remaining_eval_overlap' in b and 'eval_targets_used_for_training":False' in b
 assert 'reference_based_DPO' in t and '-F.logsigmoid' in t and 'source_only' in t
 assert 'teacher_confidence' in t and '--disable-finalizer' in run and '--num-samples 20' in run
 assert 'property-rerank' not in run.lower() and 'afterany:$pre' in s and 'afterany:$r1' in s
 assert 'P8112_PRE_JOB_ID' in s and '[[ ! -e "$UP" ]]' in s and 'mandatory_second_round' in (R/"preregistration.json").read_text()
 print("P8.1.13 static contract: PASS");return 0
if __name__=="__main__":raise SystemExit(main())
