from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FLOW_PATH = (
    ROOT
    / "SketchMol-Understanding-Condition"
    / "experiments"
    / "unified_latent_flow"
    / "unified_latent_flow.py"
)
RUN_PATH = FLOW_PATH.with_name("run_unified_latent_flow_pilot.sh")


def test_latent_flow_source_parses_and_exposes_direct_contract() -> None:
    source = FLOW_PATH.read_text(encoding="utf-8")
    ast.parse(source)
    assert "evaluation_target_access\": False" in source
    assert "candidate_library\": False" in source
    assert "finalizer\": False" in source
    assert "oracle_reranking\": False" in source
    assert "class UnifiedMolecularLatentFlow" in source
    assert "def vector_field" in source
    assert "def sample_latent" in source


def test_runner_locks_exact_n20_without_materializer() -> None:
    source = RUN_PATH.read_text(encoding="utf-8")
    assert "--num-samples 20" in source
    assert "materializer" not in source.lower()
    assert "finalizer" not in source.lower()
