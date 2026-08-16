import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
FLOW = ROOT / "experiments" / "unified_latent_flow"
SCRIPT = FLOW / "bond_derived_canonical_graph_transport.py"
PREREG = FLOW / "bond_derived_canonical_graph_transport_v42_preregistration.json"
RUN = FLOW / "run_bond_derived_canonical_graph_transport.sh"
SUBMIT = FLOW / "submit_bond_derived_canonical_graph_transport.sh"


def load_b42():
    spec = importlib.util.spec_from_file_location("b42_contract_module", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def tiny_vocabulary(module):
    # State 1 and 2 differ only in the aromatic bit.
    return module.canonical_vocabulary(
        {
            "node_states": np.asarray(
                [
                    [0, module.graph.CHARGE_OFFSET, 0, 0, 0, 0],
                    [6, module.graph.CHARGE_OFFSET, 0, 0, 0, 0],
                    [6, module.graph.CHARGE_OFFSET, 0, 1, 0, 0],
                ],
                dtype=np.int64,
            ),
            "edge_states": np.asarray(
                [[0, 0], [module.graph.BOND_SINGLE, 0], [module.graph.BOND_AROMATIC, 0]],
                dtype=np.int64,
            ),
        }
    )


def graph_batch(module, aromatic):
    torch = pytest.importorskip("torch")
    return {
        "atomic_number": torch.tensor([[6, 6]], dtype=torch.long),
        "formal_charge": torch.full((1, 2), module.graph.CHARGE_OFFSET, dtype=torch.long),
        "chirality": torch.zeros((1, 2), dtype=torch.long),
        "aromatic": torch.full((1, 2), int(aromatic), dtype=torch.long),
        "explicit_hs": torch.zeros((1, 2), dtype=torch.long),
        "no_implicit": torch.zeros((1, 2), dtype=torch.long),
        "bond": torch.zeros((1, 2, 2), dtype=torch.long),
        "bond_stereo": torch.zeros((1, 2, 2), dtype=torch.long),
    }


def test_b42_preregisters_one_structural_intervention_and_locks_b41():
    payload = json.loads(PREREG.read_text(encoding="utf-8"))
    assert payload["protocol"] == "train_only_bond_derived_canonical_graph_transport_v42"
    assert payload["status"] == "preregistered_before_first_run"
    assert payload["frozen_b41_checkpoint"] is True
    assert payload["bond_derived_atom_aromaticity"] is True
    assert payload["aromatic_flag_quotient_node_actions"] is True
    assert payload["canonical_state_used_for_targets_support_stop_and_decode"] is True
    assert payload["b41_particle_transport_frozen"] is True
    assert payload["exact_raw_attempts_per_condition"] == 20
    assert payload["molecular_candidate_ranking"] is False
    assert payload["retry_or_resampling"] is False
    assert payload["posthoc_molecule_repair"] is False
    assert payload["implementation_sha256"] == hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    assert set(key for key in payload["locked_inputs"] if key.startswith("b41_")) == {
        "b41_checkpoint_sha256",
        "b41_evaluated_candidates_sha256",
        "b41_summary_sha256",
    }


def test_b42_quotient_collapses_only_the_redundant_aromatic_flag():
    pytest.importorskip("torch")
    module = load_b42()
    vocabulary = tiny_vocabulary(module)
    representatives = vocabulary["node_state_to_canonical_representative"]
    assert int(representatives[1]) == int(representatives[2])
    assert vocabulary["canonical_node_state_count"] == 1
    assert int(np.asarray(vocabulary["canonical_representative_payload_mask"]).sum()) == 1


def test_b42_target_encoder_ignores_atom_aromatic_flag_and_decoder_derives_it_from_bonds():
    pytest.importorskip("torch")
    module = load_b42()
    vocabulary = tiny_vocabulary(module)
    source = graph_batch(module, aromatic=0)
    target = graph_batch(module, aromatic=1)
    node_actions, edge_actions = module.canonical_delta_action_targets(
        source, target, vocabulary
    )
    assert node_actions.eq(module.delta.NODE_KEEP).all()

    aromatic_action = module.delta.EDGE_SET_OFFSET + 2
    edge_actions[:, 0, 1] = aromatic_action
    edge_actions[:, 1, 0] = aromatic_action
    materialized = module.canonical_apply_delta_actions(
        source, node_actions, edge_actions, vocabulary
    )
    assert materialized["aromatic"].tolist() == [[1, 1]]
    assert materialized["bond"].tolist() == [
        [[0, module.graph.BOND_AROMATIC], [module.graph.BOND_AROMATIC, 0]]
    ]


def test_b42_generation_contract_has_no_pool_ranking_retry_or_repair():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "def canonical_delta_action_targets(" in source
    assert "def canonical_apply_delta_actions(" in source
    assert "def canonical_event_mask(" in source
    assert "install_canonical_state_contract()" in source
    assert "b41_particle_transport_frozen" in source
    assert "topk(" not in source
    assert "argsort(" not in source
    assert '"molecular_candidate_ranking": False' in source
    assert '"retry_or_resampling": False' in source
    assert '"posthoc_molecule_repair": False' in source


def test_b42_runner_uses_one_short_mig_and_exact_locked_b41_paths():
    run = RUN.read_text(encoding="utf-8")
    submit = SUBMIT.read_text(encoding="utf-8")
    assert "bond_derived_canonical_graph_transport_v42_preregistration.json" in run
    assert "viability_preserving_interacting_particle_transport_v41/seed_1991" in run
    assert "--b41-checkpoint" in run
    assert "--b41-summary" in run
    assert "--b41-evaluated-candidates" in run
    assert "--account=def-hup-ab" in submit
    assert "gpu:nvidia_h100_80gb_hbm3_1g.10gb:1" in submit
    assert "00:45:00" in submit
