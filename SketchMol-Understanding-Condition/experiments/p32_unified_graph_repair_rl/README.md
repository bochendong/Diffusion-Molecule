# P32: unified verifier-guided graph-repair RL

P32 tests whether one shared action policy can improve MolProgram construction
and editing when both tasks use the same executable GraphEditDSL action space.
It is deliberately a small support-gated pilot.

- A frozen P24 policy supplies the initial direct proposal.
- BUILD starts graph repair from that proposal; MODIFY starts from the source
  and may accept the direct P24 proposal as one executable action.
- The same Qwen2.5-1.5B common-LLM LoRA scores typed graph actions for both
  modes.
- Every visited state executes and scores the complete capped action support.
  Exact categorical policy gradients are combined with paired PCGrad into one
  shared adapter.
- `stop` is always available, so the policy can preserve a good current state.

The frozen pilot gate contains 20 de-novo conditions per arity (120 total) and
five editing conditions per exact Table-2 task (50 total). Before training, a
two-step oracle support audit must find strict-improvement opportunities in
both modes. Checkpoints 0, 10, 20, and 30 are evaluated without property-aware
reranking; checkpoint 0 is the same action policy before RL.

```bash
./experiments/p32_unified_graph_repair_rl/submit_p32.sh
```
