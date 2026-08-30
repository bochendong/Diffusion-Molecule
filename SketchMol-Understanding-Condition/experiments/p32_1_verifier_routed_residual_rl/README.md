# P32.1: verifier-routed residual graph RL

P32.1 follows the completed P32 pilot without changing its frozen evaluation
gate. The P24 direct proposal remains the primary output. A target-blind
property verifier routes only unsuccessful proposals to one shared BUILD /
MODIFY GraphEditDSL policy; already-strict proposals are returned unchanged.

The residual policy starts from the direct proposal in both modes, receives
the current verifier observation, and may execute at most two property-agnostic
graph actions. It initializes from P32 checkpoint 30 and continues exact
categorical action-value RL with paired PCGrad.

Checkpoint 0 and checkpoints 10, 20, and 30 use the same hard routing rule.
Promotion therefore requires continued RL to improve both modes over both the
frozen P24 direct baseline and residual checkpoint 0.

```bash
./experiments/p32_1_verifier_routed_residual_rl/submit_p32_1.sh
```
