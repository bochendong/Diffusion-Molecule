# P31.1: frontier-conditioned online RLOO

P31.1 is the final controlled token-level RL pilot after the P31 reward-support
audit.  It starts from the P24 alignment-refresh policy and performs genuinely
online policy-gradient updates from fresh, target-blind rollouts.

The pilot fixes the main confounds in P25--P30:

- sampling is exactly from the current policy (`temperature=1`, no truncation),
  so the optimized log probability matches the behavior policy;
- the policy term uses the **sum** of completion-token log probabilities;
- leave-one-out (RLOO) advantages are computed from one scalar, strict-aligned
  verifier reward, without per-channel z-scoring;
- an update is made only when a 16-sample group contains both strict successes
  and strict failures;
- BUILD and MODIFY use separate mode-conditioned RL adapters initialized from
  the same P24 adapter.  At inference they remain one MolProgram interface and
  are routed by the already declared task mode.

The first pilot stops at 100 informative updates per mode, with frozen joint
gates at 25, 50, and 100 updates.  Each checkpoint bundle is evaluated on both
de-novo construction and editing.  Extension to 200/300 updates and the full
1/4/8/20/40 curve is manual and allowed only after joint promotion.

Protocol amendment 01 was recorded before any GPU training began: the matched
historical P24 pool contains only 20 usable 6p conditions, so the de-novo gate
uses the largest balanced feasible subset, 20 conditions per arity (120 total),
rather than the preregistered 100 per arity.  See
`amendment_01_gate_feasibility.json`; no training or promotion threshold changed.

Protocol amendment 02 adds a numerical safety guard after one MODIFY group
produced finite rewards/loss but non-finite gradients at prospective update 89.
The guard discards that entire group before `optimizer.step`, records it, and
continues online sampling.  The resumed run starts from the last complete,
finite checkpoint (update 50); BUILD is not repeated.  See
`amendment_02_nonfinite_gradient_guard.json`.

```bash
./experiments/p31_1_frontier_online_rloo/submit_frontier_rloo.sh
```
