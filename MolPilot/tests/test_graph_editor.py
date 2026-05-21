import unittest

from molpilot.chem import RDKIT_AVAILABLE, scaffold_key
from molpilot.graph_editor import generate_graph_edit_candidates, generate_scaffold_library_candidates
from molpilot.schema import ObjectiveSpec


@unittest.skipUnless(RDKIT_AVAILABLE, "RDKit is required for graph editor tests")
class GraphEditorTests(unittest.TestCase):
    def test_graph_edit_preserves_murcko_scaffold(self):
        spec = ObjectiveSpec(goals=["decrease_logp"], constraints=["preserve_scaffold"])
        source = "Cc1ccccc1"
        candidates = generate_graph_edit_candidates(source, spec, limit=24)
        self.assertTrue(candidates)
        self.assertTrue(any(candidate.smiles != source for candidate in candidates))
        self.assertTrue(all(scaffold_key(candidate.smiles) == scaffold_key(source) for candidate in candidates))

    def test_scaffold_library_returns_same_scaffold_analogs(self):
        spec = ObjectiveSpec(goals=["decrease_logp"], constraints=["preserve_scaffold"])
        candidates = generate_scaffold_library_candidates(
            "Cc1ccccc1",
            spec,
            ["Oc1ccccc1", "CCO", "Clc1ccccc1"],
            limit=4,
        )
        self.assertTrue(candidates)
        self.assertTrue(all(candidate.origin == "scaffold_library" for candidate in candidates))
        self.assertTrue(all(scaffold_key(candidate.smiles) == scaffold_key("Cc1ccccc1") for candidate in candidates))


if __name__ == "__main__":
    unittest.main()
