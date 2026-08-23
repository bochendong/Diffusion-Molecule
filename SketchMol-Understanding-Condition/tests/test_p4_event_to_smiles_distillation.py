from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "SketchMol-Understanding-Condition"
EXPERIMENT = PROJECT / "experiments" / "p4_event_to_smiles_distillation"
UNIFIED = PROJECT / "experiments" / "unified_smiles_generator" / "unified_smiles_generator.py"
D3 = PROJECT / "experiments" / "unified_latent_table1"


def test_p4_python_entrypoints_parse() -> None:
    for path in EXPERIMENT.glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_p4_is_one_policy_with_train_only_teacher_and_raw_anyk() -> None:
    runner = (EXPERIMENT / "run_p4_event_to_smiles_distillation.sh").read_text(encoding="utf-8")

    assert "prepare_teacher_subset.py" in runner
    assert "--excluded-eval-csv \"$VALIDATION_SOURCE\"" in runner
    assert "SUCC_D3_FROZEN_MODEL_CHECKPOINT=\"$D3_CHECKPOINT\"" in runner
    assert "SUCC_D3_GENERATE_ONLY=1" in runner
    assert runner.count("--trainable-scope source_only") == 2
    assert "--disable-finalizer" in runner
    assert "--num-samples 20" in runner
    assert "for budget in 1 8 20" in runner
    assert "router" not in runner.lower()


def test_p4_preregisters_the_claim_and_stop_gates() -> None:
    protocol = json.loads((EXPERIMENT / "p4_preregistration.json").read_text(encoding="utf-8"))

    assert protocol["seed"] == 7
    assert protocol["inference"]["single_policy"] is True
    assert protocol["inference"]["teacher_present"] is False
    assert protocol["inference"]["molecular_candidate_ranking"] is False
    assert protocol["evaluation"]["candidate_budgets"] == [1, 8, 20]
    assert protocol["gates"]["minimum_raw_acc_0_65"] == 0.35
    assert protocol["gates"]["strong_raw_acc_0_65"] == 0.45
    assert protocol["gates"]["minimum_any20_acc_0_65"] == 0.70
    assert protocol["gates"]["minimum_validity"] == 0.95


def test_source_only_scope_cannot_update_the_de_novo_path() -> None:
    source = UNIFIED.read_text(encoding="utf-8")
    prefixes = source.split("SOURCE_ONLY_PREFIXES = (", 1)[1].split(")\n\n", 1)[0]
    function = source.split("def configure_trainable_scope(", 1)[1].split("\n\n\nclass FeatureStore", 1)[0]

    assert '"condition_proj.' not in prefixes
    assert '"token_embedding.' not in prefixes
    assert '"decoder.' not in prefixes
    assert '"output.' not in prefixes
    assert "parameter.requires_grad_(name.startswith(SOURCE_ONLY_PREFIXES))" in function
    assert "source_only training requires a source-aware checkpoint" in function


def test_frozen_d3_mode_skips_weight_updates_and_preserves_checkpoint() -> None:
    evaluator = (D3 / "eval_d3_event_kernel_energy.py").read_text(encoding="utf-8")
    runner = (D3 / "run_d3_event_kernel_energy.sh").read_text(encoding="utf-8")

    assert '"--frozen-model-checkpoint"' in evaluator
    assert "args.frozen_model_checkpoint is not None" in evaluator
    assert "if args.frozen_model_checkpoint is None:" in evaluator
    assert 'GENERATE_ONLY="${SUCC_D3_GENERATE_ONLY:-0}"' in runner
    assert 'if [[ "$GENERATE_ONLY" == "1" ]]' in runner
