# P20 Frozen de-novo 2p-4p table

Paired, frozen P17/P18 pilot for filling the SketchMol main 2p-7p table's
`validity all`, `2p strict`, `3p strict`, and `4p strict` columns. The official
benchmark has 1000 conditions per property count; P20 uses a preregistered,
leak-audited 100-condition subset per count and must be labeled a pilot estimate.

No training, tuning, or static candidate library is allowed. P20 R1's K1 result
is retained only as an ablation. P20 R2 uses the paper-style fair budget of 40
candidates per condition; `best_of_40` with the repository's property-aware
finalizer is the primary table value, while `average_of_40` and @1 are separate
diagnostics.
