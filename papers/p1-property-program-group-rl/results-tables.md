# MolProgram result tables

Draft status: single-seed results. The paper must lead with externally aligned
benchmarks below. SFT-vs-Group-RL and budget-scaling tables are method evidence
and ablations, not external SOTA comparisons.

## Manuscript placement

- Main text: matched-budget SketchMol de novo comparison, MuMO horizontal
  comparison, compact SFT-vs-Group-RL budget table, and 2p-to-7p scaling.
- Appendix: raw validity, G64 negative ablation, full MolEdit-Instruct
  comparison, MuMO source-fidelity diagnostic, GraphEditDSL validation, and
  completed syntax-safe decoding and source-consistency repair tables.
- SketchMol Supplementary Table 1 is not copied into the main comparison:
  DIFFUMOL and DiGress are evaluated for unconditional distribution matching,
  not under the 2p-to-7p property-program protocol.

## Main external comparison: de novo generation

The final external table must cover multiple generation paradigms rather than
presenting only a SketchMol--MolProgram pairwise comparison. We therefore
reserve matched-protocol rows for SPMM (property-vector/SMILES foundation
model), STGG+ (spanning-tree autoregressive generation), and Graph DiT (graph
diffusion). Their published task-specific values must not be copied into these
cells: each method must be rerun on the same 1,000 held-out targets per
complexity, tolerances, and 40-candidate budget. Dashes below mean pending
matched reruns, not zero performance.

| method | representation | status | average | 2p | 3p | 4p | 5p | 6p | 7p |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SPMM | property vector + SMILES | matched rerun pending | -- | -- | -- | -- | -- | -- | -- |
| STGG+ | spanning-tree sequence | matched rerun pending | -- | -- | -- | -- | -- | -- | -- |
| Graph DiT | molecular graph | matched rerun pending | -- | -- | -- | -- | -- | -- | -- |
| SketchMol | molecular image | reported matched reference | **73.1** | 80.4 | 76.8 | **73.6** | **71.6** | **67.8** | **68.5** |
| MolProgram Group-RL | direct SMILES | ours, matched rerun | 68.1 | **91.7** | **83.2** | 72.7 | 63.6 | 51.0 | 46.5 |

The honest reading is mixed: MolProgram wins on 2p and 3p, is 0.9 points lower
on 4p, and loses increasingly badly on 5p--7p. Its average is 5.0 points below
SketchMol. This table supports compositional-scaling analysis, but not an
overall state-of-the-art claim.

Execution priority is SPMM first because its released property-to-SMILES
interface is closest to the present target representation, followed by one
structural family (STGG+ or Graph DiT). The table should contain at least two
completed external families in addition to SketchMol before submission.

## Appendix external comparison: source-conditioned editing

MolEditRL uses the official Table1 aggregate. `first candidate` is honest
one-shot generation. `assisted @20` selects one output using a property-aware
verifier and must never be described as one-shot.

| method | protocol | validity | Acc_all(0.65) | Acc_all(0.15) |
| --- | --- | ---: | ---: | ---: |
| BioT5 | reported | **100.0** | 0.0 | 29.2 |
| MolGen | reported | **100.0** | 3.3 | 22.2 |
| REINVENT4 | reported | 64.1 | 12.5 | 25.0 |
| DrugAssist | reported | 97.9 | 34.9 | 41.1 |
| GeLLM3O-Mistral | reported | 90.2 | 15.2 | 35.4 |
| GeLLM3O-Llama | reported | 90.1 | 14.5 | 40.4 |
| MolEditRL | reported | 96.6 | **45.0** | **72.7** |
| MolProgram Group-RL | first candidate | 23.4 | 0.0 | 3.9 |
| MolProgram SFT | assisted @20 | 96.4 | 0.0 | 17.9 |
| MolProgram Group-RL | assisted @20 | 97.2 | 0.0 | 19.0 |

On the matched seed, Group-RL only improves the assisted relaxed-threshold
score by 1.1 points over SFT. Both checkpoints remain at zero for the official
strict 0.65 threshold and far below MolEditRL. Editing currently supports the
unified-interface claim, not the competitive-performance claim.

## Main external comparison: MuMOInstruct

Do not use a two-column IND/OOD macro-average as the main MuMO table. The main
paper now reproduces the full task-wise structure of GeLLM3O Tables 3 and 4:
five IND tasks and five OOD tasks, each with SR, Sim(success), and RI(success),
and method-family separators for general-purpose LLMs, chemistry foundation
models, task-specific systems, and generalist LLMs. The LaTeX source is
`iclr2027/mumo_tables.tex`.

The MolProgram--P4 row remains blank until the frozen official evaluator
exports all 30 task/metric cells at n=20. The older 32.3 IND / 68.8 OOD
property-SR aggregate must not be reverse-engineered into task-wise cells or
presented without its low source-similarity diagnostic.

## Appendix diagnostic: MuMO source fidelity

The completed official 20-candidate evaluation gives MolProgram Group-RL
50.4% property SR: 32.3% on IND tasks and 68.8% on OOD tasks. This is not a
competitive source-editing result. Mean Tanimoto similarity among successful
outputs is only 0.098, and no input passes the additional source-similarity
diagnostic at 0.4. The result supports property control but shows that direct
decoding usually reaches the target by generating a different molecule rather
than preserving the source. Reported GeLLM3O targets remain 76.8% IND SR and
90.8% OOD SR; these values should only enter a table with the corresponding
official similarity and RI definitions.

## Internal de novo and OOD evidence

Values are percentages. A candidate budget of `k=1` is the first raw
generation, not a property-reranked selection. `Pass@k` is the empirical
fraction of conditions with at least one strict success among the first `k`
generations.

| evaluation | method | k | raw strict success | pass@k | raw validity |
| --- | --- | ---: | ---: | ---: | ---: |
| 2p--7p | SFT | 1 | 7.02 | 7.02 | 29.20 |
| 2p--7p | Group-RL | 1 | **8.93** | **8.93** | **33.53** |
| 2p--7p | SFT | 4 | 7.21 | 22.58 | 28.78 |
| 2p--7p | Group-RL | 4 | **9.15** | **27.60** | **33.89** |
| 2p--7p | SFT | 8 | 7.26 | 35.37 | 28.85 |
| 2p--7p | Group-RL | 8 | **9.28** | **41.72** | **33.86** |
| 2p--7p | SFT | 20 | 7.30 | 54.88 | 29.04 |
| 2p--7p | Group-RL | 20 | **9.41** | **62.52** | **34.25** |
| OOD | SFT | 1 | **3.50** | **3.50** | **12.50** |
| OOD | Group-RL | 1 | 3.20 | 3.20 | 12.10 |
| OOD | SFT | 4 | 3.25 | 11.70 | **11.97** |
| OOD | Group-RL | 4 | **3.80** | **13.90** | 11.87 |
| OOD | SFT | 8 | 3.20 | 19.70 | **11.92** |
| OOD | Group-RL | 8 | **3.77** | **23.60** | 11.63 |
| OOD | SFT | 20 | 3.14 | 35.20 | 11.51 |
| OOD | Group-RL | 20 | **3.79** | **41.70** | **11.59** |

The Group-RL advantage is therefore already visible at `k=4/8/20`; it is not
a `k=256`-only result. The honest exception is OOD `k=1`, where SFT remains
better on the first draw.

## Internal source-conditioned editing scaling

Success requires every requested edit direction and source Tanimoto at least
0.15. Candidate prefixes follow generation order and do not use property-aware
reranking.

| method | k | raw validity | unique valid | raw strict success | pass@k |
| --- | ---: | ---: | ---: | ---: | ---: |
| SFT | 1 | 22.20 | 22.20 | **4.00** | **4.00** |
| Group-RL | 1 | **23.40** | **23.40** | 3.90 | 3.90 |
| SFT | 4 | 21.35 | 21.35 | 2.95 | 11.20 |
| Group-RL | 4 | **23.33** | **23.33** | **3.30** | **12.30** |
| SFT | 8 | 21.25 | 21.25 | 2.76 | 19.00 |
| Group-RL | 8 | **24.16** | **24.16** | **3.29** | **21.70** |
| SFT | 20 | 20.34 | 20.34 | 2.54 | 34.70 |
| Group-RL | 20 | **23.97** | **23.97** | **3.06** | **40.30** |

Both candidate pools now use evaluation seed 7. The strict Tanimoto-0.65
column is zero for both methods and is omitted here; editing currently supports
the broad unified-method claim, but not a strong high-similarity editing claim.

## Hard-program rollout ablation

`G16` is the existing main Group-RL checkpoint. `G64` continues training from
that checkpoint using 64 rollouts per program and targets only 6p/7p rows.

| method | k | raw strict success | overall pass@k | 6p pass@k | 7p pass@k | raw validity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT | 8 | 3.47 | 21.88 | 21.88 | 21.88 | 28.27 |
| Group-RL G16 | 8 | **4.74** | **28.12** | 31.25 | **25.00** | 32.91 |
| hard-only Group-RL G64 | 8 | 4.44 | 25.78 | **32.81** | 18.75 | **36.38** |
| SFT | 20 | 3.63 | 37.11 | 39.84 | 34.38 | 29.00 |
| Group-RL G16 | 20 | 4.75 | **50.78** | **50.78** | **50.78** | 33.26 |
| hard-only Group-RL G64 | 20 | **5.02** | 45.31 | **50.78** | 39.84 | **35.63** |

The larger rollout group increases validity and slightly raises raw success at
`k=20`, but it does not improve empirical pass@k over G16 and significantly
reduces 7p pass@20 on this single-seed subset. G64 should therefore remain a
negative ablation; the current G16 checkpoint stays in the main table.

## Appendix decoder repair: paired P2 evaluation

Syntax-safe decoding modifies the raw autoregressive decoder; it is not a
posthoc validity filter. The paired evaluation uses 384 2p--7p programs and
300 OOD programs.

| evaluation | decoder | validity@1 | strict@1 | pass@8 | validity@20 | pass@20 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2p--7p | legacy | 34.9 | 10.2 | 43.8 | 34.5 | 60.2 |
| 2p--7p | syntax-safe | **48.2** | 10.2 | **51.6** | **50.5** | **68.5** |
| OOD | legacy | 12.0 | 3.3 | 26.7 | 13.1 | 46.3 |
| OOD | syntax-safe | **29.0** | **9.7** | **42.7** | **30.1** | **61.7** |

The repair is scientifically positive but misses the preregistered +20-point
validity gate. The aggregate OOD gain is also not a high-order result: OOD 7p
pass@20 falls from 18.2% to 4.5%, while the strongest improvements occur in
the forward- and reverse-direction subsets.

## Appendix editing repair: paired P1/P2 evaluation

Raw ranking uses no target molecule or output property oracle. `Finalizer@8`
is explicitly assisted source-only selection and must not be described as
one-shot.

| edit policy | protocol | validity | Acc_all(0.65) | Acc_all(0.15) |
| --- | --- | ---: | ---: | ---: |
| protected action | raw@1 | 100.0 | **26.5** | **26.5** |
| source-consistent | raw@1 | 100.0 | 26.0 | 26.0 |
| strong consistent | raw@1 | 100.0 | 23.0 | 23.0 |
| protected action | finalizer@8 | 100.0 | 62.0 | 62.0 |
| source-consistent | finalizer@8 | 100.0 | **66.0** | **66.0** |
| strong consistent | finalizer@8 | 100.0 | 64.0 | 64.0 |

The raw edit gate fails: consistency does not improve first-candidate
accuracy. The 66.0% result is useful evidence for assisted candidate selection
only.

## Evidence paths

- De novo and OOD: `outputs/p1_property_program_group_rl_seed7/final/p1_report.md`
- Matched-seed editing: `outputs/p1_property_program_group_rl_seed7/edit_sampling_scaling_seed7_matched/edit_sampling_paper_table.md`
- G64 vs SFT: `outputs/p1_hard_grpo_g64_seed7/final_vs_sft/report.md`
- G64 vs G16: `outputs/p1_hard_grpo_g64_seed7/final_vs_g16/report.md`
- Syntax-safe decoder: `outputs/p2_validity_edit_repair_seed7/final/p2_report.md`
- Source consistency: `outputs/p1_source_consistency_validity_pilot_v1/seed_7/p1_source_consistency_validity_report.md`
