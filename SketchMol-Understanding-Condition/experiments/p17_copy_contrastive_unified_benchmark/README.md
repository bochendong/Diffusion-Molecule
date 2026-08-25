# P17: copy-contrastive unified direct LLM

P17 continues the single mixed P16 Qwen2.5-VL-7B LoRA adapter. It does not add a
router, a second model, or a task-specific head. Every training row uses the P16
prompt and JSON/SMILES response contract. Editing rows additionally pair the real
non-identity training target with a synthetic rejected response that copies the
source. De-novo rows remain ordinary completion-CE rehearsal examples.

The optimization objective is completion CE plus an edit-only margin between the
average chosen and rejected completion NLL. This is a train-derived contrastive SFT
loss, not DPO/ORPO. The rejected response is never built from development or benchmark
targets.

The experiment first evaluates the frozen P16 and P17 adapters on two fixed expanded
development views: source-isolated rows whose condition family remains in distribution,
and an OOD diagnostic which takes strict unseen condition families first and then
explicitly reports any fill from unseen exact conditions in known families. Only after
that comparison is written does it run one fixed, inexpensive benchmark pilot: two
rows from each of ten MolEdit Table1 task strata and 20 hard de-novo rows, with raw
candidate-order budgets 1/4/8. These are **pilot estimates**, not full benchmark claims.

The operational gate is applied only to the ID-condition view; the OOD view is
diagnostic because the inherited P16 adapter may already cover most coarse edit
families.

Run preparation with `run_p17_prepare.sh`, then submit the frozen GPU stages with
`submit_p17.sh`. The workflow is fail-closed on leakage audits and cached/offline
model availability.
