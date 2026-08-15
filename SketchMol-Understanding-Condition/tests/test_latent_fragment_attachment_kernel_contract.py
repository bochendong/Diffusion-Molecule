from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = (
    ROOT / "SketchMol-Understanding-Condition" / "experiments" / "unified_latent_flow"
)
MODEL_PATH = EXPERIMENT_DIR / "latent_fragment_attachment_kernel.py"
RUN_PATH = EXPERIMENT_DIR / "run_latent_fragment_attachment_kernel_pilot.sh"
SUBMIT_PATH = EXPERIMENT_DIR / "submit_latent_fragment_attachment_kernel_pilot.sh"


def test_kernel_uses_graph_and_property_latents_for_site_and_fragment_flow() -> None:
    source = MODEL_PATH.read_text(encoding="utf-8")
    ast.parse(source)
    assert "class FragmentAttachmentKernel" in source
    assert "def request_context" in source
    assert "def site_logits" in source
    assert "def transport_velocity" in source
    assert "representation.encode(batch)" in source
    assert "hierarchical.property_latent_slot_tokens" in source
    assert '"continuous_fragment_transport": True' in source
    assert '"single_vq_fragment_decode_per_attempt": True' in source


def test_each_latent_decodes_once_without_candidate_ranking_or_retry() -> None:
    source = MODEL_PATH.read_text(encoding="utf-8")
    assert "selected_tokens = distances.argmin" in source
    assert "product = fragments.join_fragments(site.core, target_fragment)" in source
    assert '"generation_target_access": False' in source
    assert '"property_oracle_generation_access": False' in source
    assert '"generation_rdkit_validity_feedback": False' in source
    assert '"molecular_candidate_ranking": False' in source
    assert '"selector": False' in source
    assert '"finalizer": False' in source
    assert '"failed_attachment_retry": False' in source
    assert '"exact_raw_attempts_per_condition": 20' in source
    assert "if int(args.num_attempts) != 20:" in source


def test_runner_requires_coverage_gate_and_stays_bounded() -> None:
    run_source = RUN_PATH.read_text(encoding="utf-8")
    submit_source = SUBMIT_PATH.read_text(encoding="utf-8")
    assert "fragment_attachment_coverage_v24" in run_source
    assert '--train-limit "${SUCC_FRAGMENT_KERNEL_TRAIN_LIMIT:-1500}"' in run_source
    assert "--gate-validity 0.90" in run_source
    assert "--num-attempts 20" in run_source
    assert "--cpus-per-task=1" in submit_source
    assert "--mem=6G" in submit_source
    assert "00:15:00" in submit_source
    assert "nvidia_h100_80gb_hbm3_1g.10gb:1" in submit_source
    assert "dongbochen1218@gmail.com" in submit_source
