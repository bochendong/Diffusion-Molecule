# P8.1.1--P8.1.13 mandatory two-round audit

Audit date: 2026-08-24 (Asia/Shanghai).  This ledger treats R1 as a diagnostic
arm, never as an accuracy gate.  A direction is round-complete when its R2 has
completed or is already queued/running with one preregistered causal change.
`Artifact complete` means the output/report exists but the original Slurm ID
was not sealed in the repository.  Historical failed or cancelled retries are
shown only when they clarify the final successful job lineage.

| No. | R1 | R1 job / status | R2 single changed factor | R2 job / status | Missing round? |
| --- | --- | --- | --- | --- | --- |
| P8.1.1 | Short source transaction, temperature 1.0 | `20380825` **Complete** | Sampling temperature `1.0 -> 0.25`; checkpoint and candidate support fixed | `20381514` **Complete** | No |
| P8.1.2 | Shared SELFIES transducer with P6 warm-start | Final eval `20386146` **Complete** | Warm-start prior `P6 -> P1`; R2 rows, optimizer, grammar, sampling and evaluators fixed | Final eval `20386148` **Complete** | No |
| P8.1.3 | One whole-payload macro region | `20380538` **Complete** | BRICS-factorize the same payload | `20380539` **Complete** | No |
| P8.1.4 | Source-residual full-SMILES SFT | `20382124` **Complete** | Add one source-only Group-RL objective/stage | `20382125` **Complete** | No |
| P8.1.5 | Forward-only full-SMILES supervision | `20384006` **Complete** | Add inverse-program source reconstruction, `cycle_weight 0 -> 1` | `20384007` **Complete** | No |
| P8.1.6 | Terminal reward aggregation `joint_bottleneck` | `20383503` **Complete** | Reward aggregation `joint_bottleneck -> dense_softmin` | `20383504` **Complete** | No |
| P8.1.7 | Checkpoint-native source residual scale 1.0 | `20386053` **Complete** | Source residual scale `1.0 -> 2.0` | `20386054` **Complete** | No |
| P8.1.8 | Masked SELFIES denoiser, edit corruption 0.35 | `20386283` **Complete** | Edit mask fraction `0.35 -> 0.20`, chosen from the R1 failure mode | `20386284` **Complete** | No |
| P8.1.9 | Uniform SFT on train-only teacher-likelihood outcomes | PRE `20391518`, R1 `20391519`, both **Complete** | Teacher-likelihood confidence weighting; same pseudo-pairs/base/steps | `20391520` **Complete** | No |
| P8.1.10 | Clean source reconstruction curriculum before identical edit SFT | `20391513` **Complete** | Delete one contiguous approximately 12% source-token span only in reconstruction | `20391514` **Complete** | No |
| P8.1.11 | Group-relative REINFORCE with `joint_bottleneck` reward | `20391565` **Complete** | Reward aggregation `joint_bottleneck -> dense_softmin` | `20391566` **Complete** | No |
| P8.1.12 | Uniform distillation of train-only verified-success outcomes | PRE `20392190`, R1 `20392191`, both **Complete** | Success-set teacher-likelihood confidence weighting | `20392192` **Complete** (7m20s, exit 0) | No |
| P8.1.13 | Uniform length-normalized DPO on verified-positive/hard-negative pairs | PRE `20393441`, R1 `20393442`, both **Complete** | Teacher-confidence pair-loss weighting only | `20393443` **Complete** (5m36s, exit 0) | No |

## Accounting notes

- P8.1.1 cancelled R1 attempts `20380396`, `20380490`, and `20380580`
  are superseded by successful R1 `20380825`; they are not additional
  scientific rounds.
- P8.1.2 retained its failed raw-training history rather than hiding it:
  `20380682`, `20380818`, and `20381804` were cancelled;
  `20381228`, `20382924`, and P1-prior retry `20382948` failed.  The sealed
  representation inputs were prepared by `20381650`; after the terminal-vocab
  compatibility repair, the matched final R1/R2 eval jobs `20386146` and
  `20386148` both completed.  Earlier eval retries `20384048` and `20384057`
  failed.  Before the raw pair, its representation-only oracle also completed
  a separate causal diagnostic: canonical target serialization versus
  source-MCS-aligned serialization.  The table uses the final raw pair, whose
  single R2 factor is the P6-to-P1 warm-start prior.  This history does not
  authorize selecting an R2 based on R1 scores.
- P8.1.3 repair job `20380883` corrected the representation audit; it did not
  create a third scientific arm.
- P8.1.4 R2 is also sealed by
  `outputs/p8_1_4_full_smiles_multitask_r2_grpo/seed_7/COMPLETE` and its full
  de-novo/edit evaluations.
- P8.1.6 job IDs and paired negative results are independently recorded in its
  `RESULTS.md`.
- P8.1.12 has exactly one submission chain:
  `20392190 -> 20392191 -> 20392192`.  No duplicate P8.1.12 submission was
  made during this audit.
- P8.1.13 first exposed a completion-marker contract bug: PRE `20393348`
  failed before training because a valid zero-byte marker was tested with
  `-s`; dependent R1 `20393349` and R2 `20393350` were cancelled without
  training.  Commit `f55ac63` changed marker existence checks to `-e`, after
  which the single successful chain `20393441 -> 20393442 -> 20393443`
  completed.  The failed launcher chain is not an extra scientific arm.

## Verdict

No P8.1.1--P8.1.13 number is missing its mandatory R2; all thirteen R2 arms have
completed.  This is a queue-completeness verdict, not a positive scientific
verdict: negative R1 and R2 metrics remain negative and were not used to
eliminate or redefine a round.
