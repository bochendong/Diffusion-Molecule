# MolProgram result tables

Draft status: single-seed, ordered-prefix results. Values are percentages. A
candidate budget of `k=1` is the first raw generation, not a property-reranked
selection. `Pass@k` is the empirical fraction of conditions with at least one
strict success among the first `k` generations.

## Main de novo and OOD table

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

## Source-conditioned editing table

Success requires every requested edit direction and source Tanimoto at least
0.15. Candidate prefixes follow generation order and do not use property-aware
reranking.

| method | k | raw validity | unique valid | raw strict success | pass@k |
| --- | ---: | ---: | ---: | ---: | ---: |
| SFT | 1 | 22.20 | 22.20 | **4.00** | **4.00** |
| Group-RL | 1 | **22.60** | **22.60** | 3.80 | 3.80 |
| SFT | 4 | 21.35 | 21.35 | 2.95 | 11.20 |
| Group-RL | 4 | **23.97** | **23.97** | **3.35** | **12.80** |
| SFT | 8 | 21.25 | 21.25 | 2.76 | 19.00 |
| Group-RL | 8 | **24.16** | **24.16** | **3.46** | **23.10** |
| SFT | 20 | 20.34 | 20.34 | 2.54 | 34.70 |
| Group-RL | 20 | **24.12** | **24.12** | **3.26** | **42.10** |

This editing block is provisional: SFT uses evaluation seed 7, whereas the
reused Group-RL pool was generated with the historical evaluation seed 23. A
matched seed-7 Group-RL pool is being generated before this table is frozen.
The strict Tanimoto-0.65 column is zero for both methods and is omitted here;
editing currently supports the broad unified-method claim, but not a strong
high-similarity editing claim.

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

## Evidence paths

- De novo and OOD: `outputs/p1_property_program_group_rl_seed7/final/p1_report.md`
- Editing: `outputs/p1_property_program_group_rl_seed7/edit_sampling_scaling/edit_sampling_paper_table.md`
- G64 vs SFT: `outputs/p1_hard_grpo_g64_seed7/final_vs_sft/report.md`
- G64 vs G16: `outputs/p1_hard_grpo_g64_seed7/final_vs_g16/report.md`
