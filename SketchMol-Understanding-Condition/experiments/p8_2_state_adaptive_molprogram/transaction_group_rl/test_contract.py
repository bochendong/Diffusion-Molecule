#!/usr/bin/env python3
from pathlib import Path
R=Path(__file__).resolve().parent
def main():
 t=(R/"group_relative_transaction_rl.py").read_text();run=(R/"run_round.sh").read_text();s=(R/"submit_queue.sh").read_text();p=(R/"preregistration.json").read_text()
 assert 'a.rollouts<4' in t and 'transaction.source_only_candidates' in t
 assert 'clean["target_smiles"]=""' in t and 'eval_targets_used_for_training":0' in t and 'target_structure_reward_access":False' in t
 assert 'policy.configure_trainable_scope(model,scope="source_action"' in t and 'group_relative_REINFORCE_not_GRPO' in t and 'F.kl_div' in t
 assert 'AGG=joint_bottleneck' in run and 'AGG=dense_softmin' in run
 assert '--num-samples 20' in run and '1 8 20' in run and 'property-rerank' not in run.lower()
 assert 'afterany:$r1' in s and 'mandatory_R2' in p
 print("P8.2 transaction RL contract: PASS");return 0
if __name__=="__main__":raise SystemExit(main())
