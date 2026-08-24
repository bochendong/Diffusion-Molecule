# P8.2 — State-adaptive MolProgram

P8.2 promotes the P8.1.1-R2 policy as the unified MolProgram mainline.  The
unified object is the learned policy, not a requirement that every molecular
state expose the same chemically meaningful actions.  One checkpoint shares
the property conditioner, autoregressive decoder, union vocabulary, base
vocabulary projection, and final vocabulary softmax across both boundary
conditions.  A source-conditioned logit residual is active only when a source
is present:

- with an empty source, transaction tokens are masked and the policy emits an
  ordinary complete SMILES literal;
- with a populated source, the legal support is the set of typed transactions
  executable on that source, and literal completions are not candidate actions.

This is a **state-adaptive action space**, not a learned task router.  Source
occupancy deterministically defines the legal support before model scoring.
There is no task classifier, router-selected model, task-specific decoder,
alternate checkpoint, or separate final softmax.  The same decoder scores token
sequences from the same union vocabulary; a learned source adapter adds a
source-conditioned residual before that shared softmax.  Only the chemically
executable support changes with the supplied state, as it does in any
constrained policy.  We do not claim that the source residual is a parameter-
identical path to empty-source decoding.

The promoted seed-7 checkpoint is the P8.1.1-R1 checkpoint; P8.1.1-R2 changes
only the source-transaction sampling temperature and therefore does not create
a second checkpoint.  Its de-novo branch is the legacy P1 Group-RL policy
protected bit-for-bit.  The source-conditioned branch trains only source-memory,
source-residual, and newly appended transaction-token parameters.

## Required audit

`architecture_audit.py` is fail-closed and verifies:

1. both evaluation arms record the identical checkpoint SHA-256;
2. all legacy vocabulary IDs and de-novo parameters remain bit-exact to P1;
3. one decoder, one base vocabulary projection, one union-vocabulary final
   softmax, and the disclosed source-conditioned logit residual are present;
4. empty-source sampling masks typed transactions, while populated-source
   support is constructed deterministically from the source graph;
5. train/evaluation IDs and exact transition fingerprints do not overlap; and
6. neither arm uses the target molecule or property-aware inference reranking.

P8.2 introduces no new result and schedules no GPU job.  Paper tables stay
blank wherever a paper-facing P8.1.1 artifact has not yet been exported.

Example audit invocation on the completed P8.1.1-R2 artifacts:

```bash
python SketchMol-Understanding-Condition/experiments/p8_2_state_adaptive_molprogram/architecture_audit.py \
  --base-checkpoint SketchMol-Understanding-Condition/outputs/direct_smiles_denovo_2p7p_v2_group_rl_v1/direct_smiles_model_group_rl/direct_smiles_generator_rl.pt \
  --checkpoint SketchMol-Understanding-Condition/outputs/p8_1_1_short_transaction_r1/seed_7/policy/umtp_graph_action_policy.pt \
  --train-csv SketchMol-Understanding-Condition/outputs/p8_1_1_short_transaction_r1/seed_7/data/edit_train.csv \
  --denovo-eval-csv SketchMol-Understanding-Condition/outputs/p6_unified_transition_policy_v1/seed_7/data/denovo_hard_gate.csv \
  --edit-eval-csv SketchMol-Understanding-Condition/outputs/p6_unified_transition_policy_v1/seed_7/data/edit_table1_gate.csv \
  --denovo-summary SketchMol-Understanding-Condition/outputs/p8_1_1_short_transaction_r2_temperature/seed_7/eval/denovo/sampling_summary.json \
  --edit-summary SketchMol-Understanding-Condition/outputs/p8_1_1_short_transaction_r2_temperature/seed_7/eval/edit/sampling_summary.json \
  --output SketchMol-Understanding-Condition/outputs/p8_2_state_adaptive_molprogram/seed_7/architecture_audit.json
```
