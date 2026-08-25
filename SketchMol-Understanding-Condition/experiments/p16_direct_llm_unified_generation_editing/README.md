# P16 — Direct LLM unified generation and editing

P16 deliberately tests the user's simplest hypothesis: train molecular generation and
editing the way a causal LLM is normally trained. One cached full
`Qwen2.5-VL-7B-Instruct` base, one tokenizer, one LoRA, and one response language see a
balanced mixture. There is no task router. An empty source means construction and a
populated source means editing.

The single output is strict compact JSON containing a coarse typed plan and a canonical
final SMILES. This keeps P10's useful typed-output lesson while testing direct molecules
instead of P10's unsuccessful motif codebook. Every raw decode is strict-parsed and
checked with RDKit, addressing the validity collapse seen in P8.1.4. P12's tool loop and
P14's graph heads are intentionally absent: this path isolates ordinary causal-LM SFT.

The train CSV is the only data source. Development rows are isolated by condition hash
and nonempty canonical-source hash. Prompts are built only from an explicit property
allowlist and the source; target SMILES appears only in the assistant training label.
Evaluation performs one greedy decode and a fixed three-sample decode without selecting
by properties or oracle outcomes.

R1 trains the unified mixed adapter and matched de-novo-only/edit-only controls with the
same per-mode examples and exposure. It measures negative transfer rather than claiming
that sharing is automatically beneficial. DPO/ORPO is forbidden unless R1 passes.

Run `bash run_p16_pilot.sh` on Nibi. `submit_p16_pilot.sh` defaults to the `def-hup-ab`
account and one 40 GB H100 MIG; resource estimates should be compared before submission.
