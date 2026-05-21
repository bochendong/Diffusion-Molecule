import unittest

import numpy as np

from molpilot.diffusion import NearestLatentCodec
from molpilot.schema import GenerationRequest, TaskType
from molpilot.source_guidance import decode_source_guided_candidates, parse_strengths


class SourceGuidanceTests(unittest.TestCase):
    def test_parse_strengths_clamps_values(self):
        self.assertEqual(parse_strengths("-1,0.25,2"), [0.0, 0.25, 1.0])

    def test_source_guided_candidates_add_local_non_source_options(self):
        codec = NearestLatentCodec(latent_dim=16).fit(["CCO", "CCN", "CCCl", "c1ccccc1"])
        request = GenerationRequest(
            task_type=TaskType.EDIT,
            source_smiles="CCO",
            instruction="Lower LogP while preserving scaffold.",
        )
        latents = np.asarray([codec.encode("c1ccccc1")], dtype=np.float32)
        candidates = decode_source_guided_candidates(
            codec,
            request,
            latents,
            top_k=2,
            source_edit_strengths=[0.25],
            source_neighborhood_k=3,
        )
        self.assertTrue(candidates)
        self.assertNotIn("CCO", [candidate.smiles for candidate in candidates])
        self.assertTrue(any("source_neighborhood" in candidate.origin for candidate in candidates))
        self.assertTrue(any(candidate.origin.startswith("source_guided") for candidate in candidates))


if __name__ == "__main__":
    unittest.main()
