import unittest

from molpilot.chem import RDKIT_AVAILABLE
from molpilot.evaluate import _repair_breakdown
from molpilot.repair_dataset import build_repair_examples, build_repair_requests
from molpilot.repair_verifier import verify_repair
from molpilot.sample import _condition_latent_candidates
from molpilot.schema import TaskType
from molpilot.stage_data import load_smiles_and_pairs


class RepairDatasetTests(unittest.TestCase):
    def test_build_repair_examples_has_required_fields(self):
        examples = build_repair_examples(
            ["CCO", "CCN"],
            corruption_types="token_deletion,ocr_confusion",
            corruptions_per_molecule=2,
            seed=1,
        )
        self.assertGreater(len(examples), 0)
        example = examples[0]
        self.assertTrue(example.clean_smiles)
        self.assertTrue(example.corrupted_smiles)
        self.assertIn(example.split, {"train", "val", "test"})
        self.assertIn("repair_", example.instruction_spec_json)

    def test_repair_requests_use_repair_task(self):
        pairs = build_repair_requests(["CCO"], corruption_types="token_deletion", seed=1)
        self.assertEqual(pairs[0][0].task_type, TaskType.REPAIR)
        self.assertEqual(pairs[0][1], "CCO")

    def test_mixed_task_mode_contains_repair_and_existing_tasks(self):
        _, pairs = load_smiles_and_pairs(
            None,
            task_mode="mixed",
            repair_corruption_types="token_deletion",
            repair_corruptions_per_molecule=1,
            seed=1,
        )
        tasks = {request.task_type for request, _ in pairs}
        self.assertIn(TaskType.REPAIR, tasks)
        self.assertTrue({TaskType.EDIT, TaskType.INPAINT, TaskType.DE_NOVO} & tasks)

    def test_condition_latent_candidates_decode_directly(self):
        class Codec:
            def decode(self, latent, top_k=4):
                return ["CCO"][:top_k]

        candidates = _condition_latent_candidates(Codec(), [0.0], top_k=2)
        self.assertEqual(candidates[0].smiles, "CCO")
        self.assertEqual(candidates[0].origin, "condition_direct")


class RepairVerifierTests(unittest.TestCase):
    def test_exact_recovery_passes(self):
        result = verify_repair("CC", "CCO", "CCO", known_smiles=[])
        self.assertTrue(result.valid)
        self.assertTrue(result.exact_recovery)
        self.assertTrue(result.overall_success)

    @unittest.skipUnless(RDKIT_AVAILABLE, "RDKit required for invalid SMILES check")
    def test_invalid_candidate_fails(self):
        result = verify_repair("CCO", "CCO", "C(", known_smiles=[])
        self.assertFalse(result.valid)
        self.assertFalse(result.overall_success)

    def test_repair_breakdown_reports_topk(self):
        rows = [
            {"request_id": "0", "rank": "0", "task_type": "repair", "valid": "True", "overall_success": "False", "exact_recovery": "False", "scaffold_recovery": "False", "novel_verified_success": "False", "tanimoto_to_clean": "0.1", "property_mae_to_clean": "0.5"},
            {"request_id": "0", "rank": "1", "task_type": "repair", "valid": "True", "overall_success": "True", "exact_recovery": "True", "scaffold_recovery": "True", "novel_verified_success": "False", "tanimoto_to_clean": "1.0", "property_mae_to_clean": "0.0"},
        ]
        metrics = _repair_breakdown(rows)
        self.assertEqual(metrics["repair_validity_at_1"], 1.0)
        self.assertEqual(metrics["exact_recovery_at_1"], 0.0)
        self.assertEqual(metrics["exact_recovery_at_5"], 1.0)
        self.assertEqual(metrics["best_tanimoto_to_clean_at_5"], 1.0)


if __name__ == "__main__":
    unittest.main()
