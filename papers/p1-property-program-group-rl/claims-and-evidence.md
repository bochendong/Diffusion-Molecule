# P1 claims and evidence ledger

This ledger is the paper's anti-overclaim contract. A claim can enter the
abstract only when its required artifact exists and the protocol column is
satisfied.

| ID | Candidate claim | Current status | Required evidence | Allowed wording |
| --- | --- | --- | --- | --- |
| C1 | Group-RL improves hard 6p/7p generation at k=8/20 | Supported on 256-condition fast subset | `p1_fast_hard_6p7p_seed7/final/paper_main_table.csv`; paired CI | “preliminary hard-set evidence” until full table completes |
| C2 | The gain is not only a k=256 best-of-many effect | Supported | Positive raw and empirical pass deltas at k=8/20 | May be stated directly for the evaluated seed |
| C3 | Group-RL improves OOD one-shot generation | Rejected | OOD k=1 is 3.5% SFT vs 3.2% Group-RL | Must state that one-shot slightly decreases |
| C4 | Group-RL improves OOD success from small candidate groups | Supported | Full 1,000-condition OOD pass@4/8/20 | May be stated for k>=4 and seed 7 |
| C5 | Group-RL improves every 7p candidate-level metric | Not supported | 7p raw CI at k=8/20 must exclude zero | Do not claim; use pass@20 result only |
| C6 | P1 has 96-99% raw candidate validity | False comparison | `p1_validity_audit.csv` | Historical selected validity and raw validity must remain separate |
| C7 | P1 fine-tunes an LLM with Group-RL | False | Method/code audit | State that Qwen is frozen and RL updates the SMILES decoder |
| C8 | P1 uses natural-language property programs alone | False | Method/code audit | State that deterministic numerical program tokens are appended to text features |
| C9 | P1 beats Graph DiT or GeLLM3O | Not established | Same dataset, oracle, and candidate protocol | Related-work comparison only; no SOTA claim |
| C10 | P1 demonstrates 2p-to-7p compositional extrapolation | Pending | Full paired 6,000-condition table and curves | Keep as hypothesis until finalizer completes |

## Frozen preliminary evidence

### Fast hard gate

- 256 total conditions: 128 6p and 128 7p;
- 20 raw candidates per condition;
- SFT vs Group-RL, seed 7;
- overall pass@20: `37.109% -> 50.781%`;
- overall raw@20: `3.6328% -> 4.7461%`;
- overall pass@8: `21.875% -> 28.125%`;
- overall raw@8: `3.4668% -> 4.7363%`;
- verdict: `fast_strong_signal`.

### Full OOD pool

- 1,000 conditions, 256 raw candidates per condition;
- pass@1: `3.5% -> 3.2%`;
- pass@4: `11.7% -> 13.9%`;
- pass@8: `19.7% -> 23.6%`;
- pass@20: `35.2% -> 41.7%`;
- pass@256: `87.6% -> 91.9%`.

## Required before submission

1. Replace every preliminary number from the synchronized machine-readable
   `p1_paper_main_table.csv`.
2. Fill the 2p-to-7p complexity table from the complete paired finalizer.
3. Generate the budget-scaling and complexity-scaling figures from the same
   CSV, not by manual transcription.
4. Include the validity audit with raw candidate validity, selected
   validity@k, empty decode rate, and nonempty RDKit-invalid rate.
5. Add train/evaluation exact-target and exact-condition overlap audits.
6. Record decoder parameter count and total inference candidates/GPU-hours.
7. Keep the single-seed limitation explicit in the abstract, experiments, and
   limitations.

## High-value ablations that do not require multiple seeds

These are ordered by expected paper value per unit of runtime:

1. frozen text features + property program versus property-program-only;
2. SFT curriculum versus SFT without property-count curriculum;
3. Group-RL with versus without the numerical property program;
4. reward decomposition: validity, strict fraction, and distance term;
5. matched-budget comparison against historical SketchMol at k=40.

