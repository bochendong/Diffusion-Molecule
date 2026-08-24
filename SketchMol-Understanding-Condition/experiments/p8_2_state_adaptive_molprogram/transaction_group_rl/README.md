# P8.2 transaction group-relative RL

This branch applies the P8.1.11 group-relative update pattern to P8.1.1's
short, executable transaction grammar. It does **not** use the SELFIES
transducer. Every rollout is selected from support enumerated mechanically from
the supplied source graph. The reward sees the generated molecule and property
program, never the paired target structure.

Both mandatory rounds restart from the same shared P8.1.1-R2 checkpoint. R1
uses the minimum of validity, property success fraction, and source similarity.
R2 changes only this aggregation to dense soft-min. Reference anchoring, SFT
anchoring, rollout count, optimization, sampling temperature, and evaluation
are fixed. Only source-memory parameters and appended transaction-token input
embeddings can update; the legacy direct-SMILES path is audited bit-for-bit.
