# Unified Language-Conditioned Molecular Latent Flow

This is a bounded kill test for the new main direction:

```text
Common-LLM constraint tokens + source molecular latent (optional)
  -> continuous source-to-target latent flow
  -> direct SMILES decoder
  -> exactly 20 independent attempts
```

It intentionally forbids candidate-library materialization, retrieval, a
finalizer, and oracle reranking. De novo starts from a learned noisy prior;
Table1-style edit starts from the encoded source molecule. Both routes share
the same molecular encoder, latent vector field, constraint interface, and
SMILES decoder.

The first run is a single-seed feasibility test on the already frozen Unified
Joint train/validation split. It reuses the Common-LLM condition feature cache
and the stable direct-SMILES vocabulary/decoder checkpoint, while training only
the new molecular encoder and latent dynamics together with a short decoder
adaptation.

```bash
cd /scratch/bdong/projects/Diffusion-Molecule-automation
bash SketchMol-Understanding-Condition/experiments/unified_latent_flow/submit_unified_latent_flow_pilot.sh
```

Primary artifact:

```text
/scratch/bdong/projects/Diffusion-Molecule/SketchMol-Understanding-Condition/
  outputs/unified_latent_flow_pilot_v1/seed_1715/summary.json
```

Read direct validity and unique-valid before property success. The experiment
is a no-go if the decoder repeats the historical token/image diffusion failure
(near-zero direct validity) or if it only copies the edit source. A positive
signal requires non-trivial direct validity/diversity and property any@20 above
an unconditioned/source-copy diagnostic; it does not by itself authorize an
official test run.

## Stage A: molecular latent representation

The first direct-flow result isolated a representation bottleneck: the decoder
was not yet able to turn a compressed molecular state into valid SMILES
reliably. The v2 route therefore separates representation from dynamics before
doing more conditional-flow training.

`train_molecular_latent_autoencoder.py` trains only:

```text
molecule -> continuous latent tokens -> molecule
```

Train and validation molecules are canonicalized, deduplicated, and made
disjoint. The objective combines corrupted-prefix reconstruction, noisy-latent
decoding, Morgan-fingerprint geometry, and a shuffled-latent margin that checks
the decoder actually uses the molecular latent. It has no property condition,
oracle, candidate library, selector, or finalizer.

The held-out molecule is intentionally the input and reconstruction target for
this representation-only test; these numbers are not a MuMO, De novo, or
Table1 benchmark result. The later conditional-generation evaluation remains
target-blind.

Stage B is blocked unless clean/noisy validity, exact reconstruction,
Tanimoto similarity, scaffold retention, and latent usage all pass their fixed
gates.

```bash
bash experiments/unified_latent_flow/submit_molecular_latent_autoencoder_v2.sh
```

If the SMILES representation gate fails because autoregressive syntax errors
compound, v3 keeps the learned molecular latent geometry but replaces only the
output language with SELFIES. This is not a repair finalizer: the neural decoder
generates SELFIES directly, which is deterministically interpreted as a
molecule. It is evaluated under the same fixed reconstruction, Tanimoto,
scaffold, noise, and latent-usage gates before any conditional flow is trained.

```bash
bash experiments/unified_latent_flow/submit_molecular_latent_selfies_autoencoder_v3.sh
```

## Stage A4: graph-native latent representation

The string gates identify a topology bottleneck rather than merely a syntax
problem. The graph-native gate keeps variable-length atom slots and explicit
unordered atom-pair bond slots. A permutation-equivariant encoder and one-shot
categorical decoder reconstruct held-out molecules under clean and noisy
latents. It reports raw-argmax graph validity, connectivity, exact topology,
Morgan similarity, and scaffold retention. No valence repair, finalizer,
property oracle, candidate library, or selector is available.

This bounded representation test is informed by EDM-SyCo (ICLR 2025), DeFoG
(ICML 2025), GrIDDD (NeurIPS 2025), and GraphBSI (ICLR 2026). Passing permits a
later categorical graph-flow experiment; it is not itself a generation result.

```bash
bash experiments/unified_latent_flow/submit_graph_latent_autoencoder_v1.sh
```

The complete-schema v2 adds explicit-H/no-implicit atom state and invariant
R/S plus E/Z stereochemistry. It also evaluates a much stronger latent-noise
stress test and random atom/bond category masking. It writes a separate v2
checkpoint and leaves the v1 result reproducible.

```bash
bash experiments/unified_latent_flow/submit_graph_latent_autoencoder_v2.sh
```

## Stage B1: source-conditioned categorical graph flow

After the complete-schema representation gate passes, the bounded B1 pilot
freezes that autoencoder and trains a permutation-equivariant rectified-flow
velocity over its atom and unordered-pair latent slots. Paired training targets
are MCS-aligned to the source slots, while validation generation accepts only
the source graph and a sanitized property program. The validation target and
property scorers are accessed only after exactly 20 direct raw decodes have
been frozen.

The pilot is deliberately small: one seed, at most 1,500 two- or three-property
training edit pairs and 16 held-out conditions. It has no candidate library, selector, finalizer,
oracle reranking, or valence repair. Its purpose is to decide whether learned
latent motion improves over source-copy target proximity while retaining high
validity and source similarity; it is not a Table1 or MuMO headline result.

```bash
bash experiments/unified_latent_flow/submit_categorical_graph_latent_flow_pilot.sh
```

The v1 signal exposed a size-modeling failure when most paired edits add or
remove atoms. The v2 pilot therefore learns an explicit target-count
distribution and source-slot retention head. At sampling time it draws the
birth/death mask before integrating the latent velocity and leaves all other
inactive slots fixed. Count masking is a learned categorical component of the
generator, not a target-derived mask or a chemical repair pass.

```bash
bash experiments/unified_latent_flow/submit_size_adaptive_graph_latent_flow_pilot.sh
```

## Stage B3: native categorical graph-belief flow

The v2 signal showed that a separate atom-count head can unlock structural
movement, but its independently sampled occupancy mask destabilizes validity.
Stage B3 removes that factorization. Its state and stochastic path are the
atom occupancy/type/attribute and bond/order/stereo categories themselves;
class zero natively represents atom or bond absence. A conditional endpoint
field predicts category distributions through the frozen, gate-passed graph
decoder, and the sampler performs joint categorical birth/death transitions
starting from the source graph.

Empty source slots receive an ordered birth-rank query solely to distinguish
otherwise exchangeable birth locations. There is no target-count prediction,
continuous-latent regression loss, candidate library, selector, finalizer,
oracle reranking, or valence repair. Validation targets are inaccessible to
generation and are opened only after exactly 20 raw candidates are frozen.

```bash
bash experiments/unified_latent_flow/submit_categorical_graph_belief_flow_pilot.sh
```

## Stage B4: coupled local no-edit transitions

The native B3 sampler improved target movement and strict any@20, but sampling
every atom and bond category independently reduced validity. B4 makes source
retention a learned no-edit category. It samples an aligned atom block first,
re-encodes that provisional categorical graph, and then samples its bond blocks
from a distribution conditioned on the sampled atoms. Unchanged node and bond
blocks remain exactly on the source graph inside the generative transition.

This is a bounded 2-property falsification pilot: 1,500 training pairs, 12
held-out conditions, one seed and exact n=20. It does not use a valence rule,
repair pass, candidate selector, finalizer, or property oracle during
generation. The immediate gate is validity at least 80% while retaining source
Tanimoto at least 0.4 and nontrivial strict/target-improvement signal.

```bash
bash experiments/unified_latent_flow/submit_coupled_local_graph_belief_flow_pilot.sh
```

## Stage B5: single-token VQ motif latent

B4 established a strong 2-property signal, but its independent Bernoulli edit
gates accumulated too many simultaneous edits when 2- and 3-property training
were mixed. B5 removes per-atom and per-bond sampling entirely. A train-only
posterior compresses the aligned graph delta into one VQ motif token; a
source-and-condition prior predicts that token; and one shared graph-latent
decoder deterministically maps source plus token to all endpoint categories.
The 20 attempts sample only the latent token.

This pilot uses one seed, 1,500 train pairs and 20 held-out 2p/3p conditions.
The posterior and validation targets are absent during generation. There is no
retrieved transform library, GraphEditDSL action, selector, finalizer, oracle
reranking, independent category sampling, or valence repair.

```bash
bash experiments/unified_latent_flow/submit_vq_motif_graph_belief_flow_pilot.sh
```

The v5b signal adds one code-utilization objective: the correct motif token
must reconstruct its paired endpoint better than a mismatched token by a fixed
margin. This directly tests and prevents the decoder from ignoring the VQ
latent; it does not change candidate budget, generation inputs, or evaluation.

## Stage B6: hierarchical constraint and motif VQ latents

B5b increased motif-code use and target-improvement signal, but one token still
had to represent both the requested multi-property direction and the local
structural realization. B6 separates those roles. A train-only global-delta
posterior learns a small constraint code; a second posterior encodes the
changed local subgraph into a motif code conditioned on that constraint. At
generation time a source-and-condition prior samples the constraint code first,
then a conditional motif prior samples the motif code, and one categorical
graph decoder deterministically produces the endpoint from both codes.

Both codebooks have independent active-code and perplexity diagnostics. The
correct constraint and motif codes must each outperform a mismatched code in
reconstruction, preventing either hierarchy level from being ignored. The
pilot remains one seed, 1,500 train pairs, 20 held-out 2p/3p conditions and
exact n=20. It has no target or property-oracle generation access, candidate
selector, finalizer, independent atom/bond sampling, GraphEditDSL action, or
chemistry repair.

```bash
bash experiments/unified_latent_flow/submit_hierarchical_vq_motif_graph_flow_pilot.sh
```

## Stage B7: source-anchored hierarchical latent decoder

B6 showed that separating constraint and motif tokens restores 3-property
signal and improves structure retention, but raw full-graph decoding still
changed the predicted atom count too often and left validity below the 80%
gate. B7 is a controlled decoder ablation on the exact same seed and held-out
conditions. The two-level posterior, priors, codebooks, n=20 contract and graph
endpoint head remain unchanged.

The new endpoint field jointly learns atom-block and bond-block change logits
from the source graph, decoded endpoint, condition and both latent tokens.
Their deterministic sign selects either the decoded block or the exact source
block. This makes source preservation part of the learned generative decoder;
it is not an after-the-fact source-copy heuristic, chemistry repair, candidate
selector, or ranking stage. Randomness still enters only through the two latent
tokens.

```bash
bash experiments/unified_latent_flow/submit_source_anchored_hierarchical_vq_graph_flow_pilot.sh
```

## Stage B8: connected motif-region latent decoder

B7 raised strict any@20 on the matched 2p/3p pilot but reduced validity because
separately chosen atom and bond blocks broke chemical consistency at their
boundaries. A train-only connectivity audit supports replacing those gates with
one region: 17 of the 18 held-out changed subgraphs are already connected, all
six 3-property cases are connected, and the one exception needs four connector
nodes under a source/target-union shortest-path closure.

B8 predicts one region size and latent-conditioned node scores. A deterministic
graph decoder grows one connected region over the union of source and decoded
endpoint adjacency, then swaps the complete endpoint node blocks and all
internal edge blocks together. Source blocks and boundary bonds outside that
region remain exact. The projection defines the structured decoder support; it
does not inspect chemistry validity, properties or targets at generation time,
and it is not a repair, selector, action plan or ranking step.

The pilot is matched to B6/B7 with seed 1741, the same 1,450 train pairs, the
same 18 held-out conditions, and exact n=20.

```bash
bash experiments/unified_latent_flow/submit_connected_region_hierarchical_vq_graph_flow_pilot.sh
```
