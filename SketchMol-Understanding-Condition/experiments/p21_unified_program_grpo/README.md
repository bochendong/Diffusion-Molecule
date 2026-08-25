# P21 Unified Program GRPO

P21 continues the frozen P18 unified direct-LLM adapter with a small,
critic-free group-relative policy update.  It is designed to convert P18's
large best-of-K headroom into higher raw-policy success without giving up the
single-checkpoint contract.

The update uses only P17's isolated training split.  It samples four raw
responses for each condition, scores them with the same structured molecular
program used by evaluation, z-normalizes rewards within the condition, and
optimizes the sampled response log probability.  A chosen-completion SFT
anchor is retained.  Benchmark targets, static molecular pools, property-aware
selection, routers, and mode-specific adapters are forbidden.

The first frozen gate evaluates:

- P17's ID and strict-family-OOD development views;
- P19's locked 100-row MolEdit Table1 subset at raw budgets 1/4/8;
- P19's locked 40-row hard de-novo subset at raw budgets 1/4/8.

Run on Nibi with:

```bash
bash SketchMol-Understanding-Condition/experiments/p21_unified_program_grpo/submit_p21.sh
```
