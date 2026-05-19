import unittest

from molpilot.chem import RDKIT_AVAILABLE
from molpilot.data import build_verified_requests
from molpilot.understanding import ground_instruction
from molpilot.verifier import verify_candidate


class VerifiedDatasetTests(unittest.TestCase):
    def test_de_novo_relative_goals_do_not_require_source(self):
        spec = ground_instruction("Generate a CNS-like drug-like molecule with MW below 450 and TPSA below 90.")
        result = verify_candidate(None, "CCOc1ccc2nc(S(N)(=O)=O)sc2c1", spec)
        self.assertTrue(result.goal_success)

    @unittest.skipUnless(RDKIT_AVAILABLE, "RDKit required for scaffold-pair construction")
    def test_verified_pairs_pass_their_own_spec(self):
        smiles = [
            "Cc1ccccc1",
            "Oc1ccccc1",
            "CCOc1ccccc1",
            "COc1ccccc1",
        ]
        pairs = build_verified_requests(smiles)
        self.assertGreater(len(pairs), 0)
        for request, target in pairs:
            result = verify_candidate(request.source_smiles, target, ground_instruction(request.instruction))
            self.assertTrue(result.overall_success, result.reasons)


if __name__ == "__main__":
    unittest.main()
