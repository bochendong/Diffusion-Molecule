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

## Stage B9: sparse categorical graph-delta grammar

B8 learned the right connected support scale and improved validity over the
independent B7 block decoder, but replacing every edge inside a region still
changed a dense quadratic set of endpoint pairs. Its invalid samples changed
substantially more internal pairs than its valid samples. B9 keeps the same
two-level constraint/motif latent and connected-region projector, while making
the source-relative edit itself categorical and sparse.

Every selected node predicts one of `KEEP`, `DELETE`, `BIRTH`, or `REPLACE`;
every internal pair predicts `KEEP`, `DELETE`, or `SET`. Illegal operations are
masked deterministically from source occupancy. `KEEP` is a native learned
category, the region boundary remains the exact source graph, and only explicit
non-KEEP pairs are changed. Endpoint categories and delta operations are jointly
conditioned on the source, request and both latent tokens. There is no target or
property oracle at generation time, chemistry repair, candidate selector,
action-plan executor, or ranking stage.

The pilot is a strict matched ablation against B6-B8: seed 1741, the same 1,450
train pairs, the same 18 held-out 2p/3p conditions, and exact n=20.

```bash
bash experiments/unified_latent_flow/submit_categorical_delta_hierarchical_vq_graph_flow_pilot.sh
```

## Stage B10: grammar-native valence-budget delta decoder

B9 recovered strong 2-property support (`strict any@20 = 83.3%`) but reached
only 64.4% candidate validity and 16.7% 3-property strict success. Invalid B9
candidates changed 8.5 internal edges on 2p and 11.8 on 3p, compared with 1.6
and 2.8 for valid candidates. The graph representation is not the bottleneck:
the frozen v2 graph autoencoder has 100% clean validity and 97.25% masked
validity. The failure is independent internal edge composition.

B10 therefore predicts a train-only target total explicit-valence budget for
each node in half-bond units, including explicit hydrogens. After the latent
node operations, internal edge operations are generated in a fixed tensor
order. `KEEP`, `DELETE`, and `SET` remain learned categorical operations, but
an operation outside either endpoint's remaining predicted budget is not in
the decoder support. This is a grammar-native autoregressive decoder, not a
post-hoc sanitization/repair pass, candidate selector, or ranking stage.

The submitter first runs the seed-1741 matched 1,450-pair pilot. A dependent CPU
controller submits a 10,000-pair, 16-epoch scale run only if the pilot reaches
validity >=80%, overall strict any@20 >=65%, and 3p strict any@20 >=50%.

```bash
bash experiments/unified_latent_flow/submit_valence_budget_hierarchical_vq_graph_flow_pilot.sh
```

## Stage B11: latent motif-attachment grammar

B10 reduced invalid-candidate edge edits from 10.35 to 6.52 but improved
validity only from 64.4% to 66.7%; invalid node edits rose from 6.70 to 8.74.
This identifies the remaining failure as atom-by-atom composition rather than
the frozen graph representation or edge valence alone.

B11 treats the edited subgraph as one latent motif. It predicts the number of
active motif atoms, retains one source boundary atom as the attachment anchor,
grows a connected motif support inside the latent-selected region, and emits a
required spanning tree before optional ring-closure edges. Tree and closure
bonds are categorical endpoint predictions constrained by the learned node
valence budgets. Region-external source structure remains exact. The procedure
does not retrieve a motif library, rank molecular candidates, inspect a
property oracle, or sanitize/repair a completed molecule.

The seed-1741 matched pilot uses the same 1,450 train pairs, 18 held-out 2p/3p
conditions and exact n=20 as B6-B10. A dependent controller submits the
10,000-pair, 16-epoch scale run only after validity >=80%, overall strict
any@20 >=65%, and 3p strict any@20 >=50%.

```bash
bash experiments/unified_latent_flow/submit_motif_attachment_hierarchical_vq_graph_flow_pilot.sh
```

## Stage B12: source-aware constraint-token attention

B11 raised matched-pilot validity from 66.7% to 77.2% and strict any@20 from
55.6% to 72.2%.  Its 2-property strict success reached 100%, while 3-property
strict success fell to 16.7%.  The remaining question is therefore not whether
the motif-attachment grammar works, but whether averaging the property program
into one condition vector destroys the composition signal needed by 3p tasks.

B12 is an isolated representation ablation.  It keeps the B11 decoder, losses,
data, seed, training budget, and exact n=20 protocol unchanged.  Each source
atom instead queries the fixed train/test-safe property-program tokens through
multi-head cross-attention; the masked source responses are pooled only after
that interaction and feed the same constraint prior, motif prior, and endpoint
decoder.  Generation remains target- and oracle-blind.

```bash
bash experiments/unified_latent_flow/submit_constraint_attention_motif_graph_flow_pilot.sh
```

## Stage B13: compositional property latent slots

B12 rejected the hypothesis that source-to-condition cross-attention would fix
3p composition: validity fell to 68.1%, 3p strict remained 16.7%, and 3p atom
count MAE rose to 7.96.  B13 therefore removes cross-attention and represents
the request as a set of independently encoded active-property contributions.

The representation contains one global request token and one fixed slot for
each supported property.  Inactive slots are exactly zero.  A shared residual
encoder maps each active slot independently; their normalized sum plus an
explicit active-count residual produces the joint condition supplied to the
unchanged B11 constraint token, motif token, and graph decoder.  The operation
is permutation invariant, target/oracle blind, and uses no candidate ranking.
The slot module is instantiated after all shared B11 modules so seed 1741 gives
identical initialization for every shared parameter.

```bash
bash experiments/unified_latent_flow/submit_property_latent_slots_motif_graph_flow_pilot.sh
```

## Stage B14: B11-preserving residual property slots

B13 confirmed that compositional property slots help the hard regime: 3p
strict doubled from 16.7% to 33.3%, 3p validity reached 92.5%, and 3p atom-count
MAE fell to 2.42.  Its additive raw-slot baseline nevertheless reduced 2p
strict from B11's 100% to 83.3%.

B14 keeps the exact B11 mean-pooled condition as the main path.  Independently
encoded active-property slots and the active-property count can only enter as
zero-initialized residuals.  Thus the initial condition and every shared model
parameter are exactly B11-equivalent, while training can learn a permutation-
invariant compositional correction for 3p.  No property-count branch, candidate
selector, target/oracle access, or graph-decoder change is introduced.

```bash
bash experiments/unified_latent_flow/submit_residual_property_slots_motif_graph_flow_pilot.sh
```

## Stage B15: symmetric property-interaction latents

B14 preserved B11 too strongly: source similarity rose to 0.779, but 3p strict
success fell to 0%.  B13 remains the strongest compositional base, with 92.5%
3p validity and 33.3% 3p strict success.  B15 therefore keeps B13's complete
unary property-slot path and adds one structural capability: an unordered
latent for every pair of active properties.

Each pair is encoded through the symmetric sum, product, and absolute
difference of its two property slots.  A shared network composes these pair
latents with a normalized set sum, making the result invariant to property
order.  Its output layer is initialized to zero, so the initial condition is
exactly B13 and every B11-shared parameter retains the matched seed-1741
initialization.  Training can then learn second-order constraint interactions
without property-count routing, sorting, prompt decisions, candidate ranking,
target/oracle generation access, or any decoder change.

The pilot remains a strict matched ablation: 1,500 train pairs, the same 20
held-out 2p/3p conditions, eight epochs, exact n=20, and one 10GB H100 MIG.

```bash
bash experiments/unified_latent_flow/submit_property_interaction_latents_motif_graph_flow_pilot.sh
```

## Stage B16: train-supported joint atom-state valence grammar

B15 is the strongest matched latent result so far: overall strict any@20 rose
to 77.8% and 3p strict rose to 66.7%.  Its 78.9% candidate validity missed the
80% gate by only four of 360 candidates.  The remaining decoder factorization
chooses atomic number, charge, aromaticity, explicit hydrogens, implicit-H
policy, and total valence as independent categorical predictions.  Each field
can be likely while their assembled atom state never occurs in a valid train
molecule.

B16 keeps the complete B15 model and training dynamics.  It builds a vocabulary
of joint atom states from the selected train source/target molecules only.  At
generation time, the decoder selects one complete state by summed field log
likelihood, then caps its learned valence budget at the maximum valence observed
for that train-supported state before motif edges are generated.  Validation
targets never enter this vocabulary.  The constraint is part of generative
support before graph assembly; it is not RDKit sanitization, post-hoc repair,
candidate filtering, ranking, or a task-specific exception.

The matched pilot retains seed 1741, 1,500 requested train pairs, the same 18
held-out 2p/3p conditions, eight epochs, exact n=20, and one 10GB H100 MIG.

```bash
bash experiments/unified_latent_flow/submit_atom_state_valence_grammar_motif_graph_flow_pilot.sh
```

## Stage B17: train-supported joint node-edge grammar on fresh held-out data

B16 raised overall strict any@20 to 83.3% and 3p strict to 83.3%, but validity
fell to 76.4%.  All nine newly invalid candidates occurred in two already-hard
QED-up/SA-down conditions, while every other condition retained exactly the
same valid-candidate count.  Legal node states therefore improve property
support but do not guarantee that independently decoded bonds are compatible
with both endpoint states.

B17 extends the train-only grammar with bond support for unordered pairs of
joint atom states.  A generated bond must have been observed for the exact
state pair in a train molecule; if that support is absent, the grammar backs
off once to the train-observed `(atomic number, charge, aromatic)` state pair.
If neither support contains a bond, that edge is outside the generative
distribution.  The mask is applied while the motif spanning tree and closure
edges are constructed, before a molecular graph exists.  It is not molecule
repair, filtering, ranking, validation-target access, or a task exception.

To stop tuning on the historical 18 conditions, B17 reproduces and excludes
the old validation selection seed 1742, then freezes a new held-out selection
with seed 2719.  One model training produces two evaluations with identical
latent samples: the B16 node-only grammar and the B17 node-edge grammar.  The
primary gate requires >=80% validity and non-negative overall/3p strict deltas
relative to that matched B16 evaluation.  Candidate budget remains exact n=20.

```bash
bash experiments/unified_latent_flow/submit_node_edge_state_grammar_motif_graph_flow_pilot.sh
```

## Stage B18: set-compositional continuous constraint transport

B17 falsified the node-edge support hypothesis on the fresh development set:
the matched B16 and B17 candidates were identical, candidate validity was
76.4%, mean unique-valid was only 2.22/20, and 3-property strict success was
0%.  The deeper failure is latent collapse: the validation sampler exercised
only two constraint codes and two motif codes, so adding another chemistry mask
cannot create missing structural modes.

B18 removes both VQ codebooks.  A train-only posterior maps the aligned graph
delta to a bounded continuous endpoint.  Conditional flow matching transports
Gaussian noise to that endpoint through three learned components: a
source/global-request field, a normalized set of unary property fields, and a
normalized set of symmetric pairwise interaction fields.  At generation time
20 independent continuous trajectories are integrated from the source and
request alone.  The existing source-anchored motif decoder is retained so the
experiment isolates the latent hypothesis instead of changing the latent and
graph decoder simultaneously.

The seed-2719 conditions are now explicitly treated as development data, not a
final audit.  The kill-test gates are deliberately demanding: validity >=95%,
mean unique-valid >=10/20, overall strict any@20 >=25%, and 3-property strict
any@20 >=20%.  Failure redirects the project to a new representation/decoder
backbone rather than another grammar patch.  Exact n=20 and all target/oracle
access restrictions remain unchanged.

```bash
bash experiments/unified_latent_flow/submit_continuous_constraint_transport_pilot.sh
```
