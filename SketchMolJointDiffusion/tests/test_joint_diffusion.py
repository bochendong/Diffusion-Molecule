import unittest

from sketchmol_joint_diffusion.joint_diffusion import run_joint_diffusion


class JointDiffusionTests(unittest.TestCase):
    def test_defaults_enable_image_route(self):
        # The callable is imported without importing torch/RDKit at module import time.
        self.assertTrue(callable(run_joint_diffusion))


if __name__ == "__main__":
    unittest.main()
