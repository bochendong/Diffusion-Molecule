import unittest

from molpilot.schema import GenerationRequest, TaskType
from molpilot.understanding import UnderstandingStream, ground_instruction


class UnderstandingTests(unittest.TestCase):
    def test_solubility_grounding_is_hard_and_proxy(self):
        spec = ground_instruction("This lead has poor solubility. Lower LogP but preserve the core.")
        self.assertIn("decrease_logp", spec.goals)
        self.assertIn("improve_solubility_proxy", spec.proxy_goals)
        self.assertIn("preserve_scaffold", spec.constraints)
        self.assertTrue(spec.hard_verifiable)

    def test_disease_prompt_is_not_hard_verified(self):
        spec = ground_instruction("Make this compound treat lung cancer.")
        self.assertIn("treat_disease", spec.unverifiable_goals)
        self.assertFalse(spec.hard_verifiable)

    def test_dual_stream_branches_exist(self):
        request = GenerationRequest(
            task_type=TaskType.EDIT,
            source_smiles="CCO",
            instruction="Lower LogP and keep the molecule similar.",
        )
        bundle = UnderstandingStream().encode(request)
        self.assertIn("uncond", bundle.branches)
        self.assertIn("text_spec", bundle.branches)
        self.assertIn("multimodal", bundle.branches)
        self.assertEqual(len(bundle.branches["multimodal"].vector), 256)

    def test_repair_prompt_gets_similarity_constraint(self):
        spec = ground_instruction("Repair this OCR-corrupted invalid SMILES.")
        self.assertIn("keep_similarity", spec.constraints)
        request = GenerationRequest(
            task_type=TaskType.REPAIR,
            source_smiles="CC(",
            instruction="Repair this invalid molecule.",
        )
        bundle = UnderstandingStream().encode(request)
        self.assertIn("multimodal", bundle.branches)


if __name__ == "__main__":
    unittest.main()
