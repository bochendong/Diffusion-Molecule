# P22 MolEdit Table1 protocol sweep

P22 is a test-only aggregation audit. It does not train, regenerate, rank, deduplicate,
or repair molecules. It reuses frozen ordered candidate streams and evaluates two
different statistical units:

- `candidate`: every raw output in the first `K` ranks remains in the denominator;
- `any`: a source condition succeeds when at least one of its first `K` outputs succeeds.

The candidate-level row is the protocol-compatible comparison to MolEditRL Table 1,
whose displayed decimals resolve to exactly 500 outputs per task. Any@K is a separate
budget diagnostic and must not replace candidate-level `Acc_all` in the external table.

Frozen streams:

- D3 supervised: 997 conditions x 20 raw candidates;
- D3 + GRPO: 997 conditions x 20 raw candidates;
- P19/P17 and P19/P18: 100 conditions x 8 raw candidates.

The Nibi sweep evaluates D3 at K=1/2/4/5/10/20 and P19 at K=1/2/4/5/8 under both
aggregations. `analyze_protocol_sweep.py` compares every completed summary with the
ten task-level MolEditRL values and reports macro distance plus task-vector RMSE.
The completed audit and job provenance are recorded in `RESULTS.md`.

No result in this directory should be described as an official MolEditRL code
reproduction: the authors have not released the training or evaluation implementation.
