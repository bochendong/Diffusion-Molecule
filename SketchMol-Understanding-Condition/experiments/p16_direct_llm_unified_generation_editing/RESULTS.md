# P16 results

## Decision

**`stop_negative_gate`**

The direct causal-LM experiment produced a much healthier output language than the
older P8 full-SMILES path, but it narrowly failed the preregistered SFT gate. Mixed SFT
showed no meaningful token-NLL interference and reached perfect greedy de-novo validity
and perfect Any@3 validity in both modes. However, its edit greedy validity was 0.125
below the matched edit-only adapter, just beyond the allowed 0.10 loss, and its edit
greedy noncopy rate was only 0.375 versus the required 0.50. Thresholds were not changed
after seeing the result. R2 DPO/ORPO was therefore not started.

## Run and data audit

- successful Slurm job: `20448458`
- state / exit: `COMPLETED`, `0:0`
- elapsed / node: `00:14:59`, `g8`, full H100
- start / end: `2026-08-24 20:45:43–21:00:42 EDT`
- observed batch MaxRSS: `21,335,656 KB`
- cached base: `/scratch/bdong/checkpoints/Qwen2.5-VL-7B-Instruct`
- output: `outputs/p16_direct_llm_unified_generation_editing/seed_1616`
- training: 128 de-novo + 128 edit rows; mixed adapter sees all 256, and each
  matched single-mode adapter sees the identical 128 rows for its mode
- development: 8 de-novo + 8 edit rows
- train/dev condition-hash overlap: 0
- train/dev nonempty-source-hash overlap: 0

The first allocation, `20448199`, failed cleanly after four seconds because Slurm copied
the batch script under `/var/spool`, so a path derived from `BASH_SOURCE` incorrectly
resolved the repository under `/var`. It did not start training or corrupt output. The
submitter now exports the original experiment directory explicitly; the corrected job
completed without retries.

All arms used the same full 7B base and tokenizer, text-only BF16 LoRA, one response
schema, and no mode-specific router or output head. Adapter tensors were finite.

| Arm | Rows | Optimizer steps | Train loss |
| --- | ---: | ---: | ---: |
| mixed unified | 256 | 32 | 0.862338 |
| de-novo only | 128 | 16 | 1.192612 |
| edit only | 128 | 16 | 0.797742 |

## Matched held-out comparison

Each row received one raw greedy decode and exactly three raw sampled decodes. There was
no candidate selection, property reranking, static pool, inference target access, or
official-test access. `Any@3` reports whether at least one of the three outputs passed a
metric; edit similarity is the first valid candidate in generation order.

### De-novo

| Arm | Token NLL | Greedy parse / valid / canonical / exact | Any@3 parse / valid / canonical / exact |
| --- | ---: | ---: | ---: |
| mixed unified | 1.036144 | 1.000 / 1.000 / 0.875 / 0.000 | 1.000 / 1.000 / 1.000 / 0.000 |
| de-novo only | 1.025658 | 1.000 / 1.000 / 1.000 / 0.000 | 1.000 / 0.875 / 0.625 / 0.000 |

Mixed minus single-mode token NLL was only `+0.010486`; greedy validity changed by
`0.000`. Mixed SFT improved sampled validity and canonical-form robustness on this tiny
slice, though neither checkpoint reproduced a held-out target exactly.

### Editing

| Arm | Token NLL | Greedy parse / valid / canonical / exact | Greedy noncopy / similarity | Any@3 parse / valid / canonical / exact | Any@3 noncopy / similarity |
| --- | ---: | ---: | ---: | ---: | ---: |
| mixed unified | 0.480704 | 1.000 / 0.750 / 0.625 / 0.000 | 0.375 / 0.784854 | 1.000 / 1.000 / 0.875 / 0.000 | 1.000 / 0.560177 |
| edit only | 0.493957 | 0.875 / 0.875 / 0.625 / 0.000 | 0.625 / 0.638245 | 1.000 / 1.000 / 0.250 / 0.000 | 1.000 / 0.371581 |

The unified model had slightly better edit token NLL (`-0.013253`) and stronger
source-similarity among its valid greedy outputs, but more greedy rows either failed
validity or copied the source. Its greedy-validity delta was `-0.125`, failing the
preregistered `>= -0.10` interference bound. Its noncopy rate `0.375` also failed the
`>= 0.50` gate. Fixed-K sampling removed both failures (`1.0` Any@3 validity and
noncopy), but the protocol does not allow an Any@3 result to overwrite the greedy gate.

## Interpretation

This is an informative negative gate rather than a dead end. Ordinary mixed LLM SFT can
learn the shared strict JSON/SMILES language and generate RDKit-valid molecules in both
modes far more reliably than P8.1.4's older decoder. The remaining interference is
localized to the edit decision boundary: the unified adapter tends to preserve/copy a
source under greedy decoding. A follow-up should change the training objective or
rehearsal balance around non-identity edit examples, not add a router or silently choose
among sampled molecules. Because R1 failed, no preference-training claim is made.
