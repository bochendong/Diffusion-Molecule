# MolEditRL paper-faithful baseline

MolEditRL does not currently have a confirmed public official implementation in
this workspace. This folder therefore records a paper-faithful reconstruction
contract rather than an official-code reproduction.

Public assets:

- Paper: https://arxiv.org/abs/2505.20131
- Dataset: https://huggingface.co/datasets/FanSiLeC/MolEdit-Instruct

Server entrypoint:

```bash
bash SketchMol-Understanding-Condition/scripts/submit_moleditrl_table1_paper_faithful.sh
```

Without a prediction CSV, the runner materializes the extracted paper baseline
rows into `official_paper_metrics.csv` and writes `official_run_manifest.json`.
With `MOLEDITRL_PREDICTIONS_CSV` set, it also evaluates predictions with the
repo's MolEditRL-style Table1 metric script on the server.
