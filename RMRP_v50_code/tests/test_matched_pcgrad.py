import unittest
from collections import Counter

import torch

from train_matched_restorer import apply_equal_domain_pcgrad


class EqualDomainPCGradTests(unittest.TestCase):
    def test_conflicting_gradients_are_projected_symmetrically(self) -> None:
        parameter = torch.nn.Parameter(torch.zeros(2))
        buffers = {
            "ivcnz": [torch.tensor([1.0, 0.0])],
            "pcm": [torch.tensor([-1.0, 1.0])],
        }
        conflict, cosine = apply_equal_domain_pcgrad(
            [parameter], buffers, Counter({"ivcnz": 1, "pcm": 1})
        )
        self.assertTrue(conflict)
        self.assertLess(cosine, 0.0)
        self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(float(parameter.grad[1]), 0.0)

    def test_aligned_gradients_are_equal_domain_averaged(self) -> None:
        parameter = torch.nn.Parameter(torch.zeros(2))
        buffers = {
            "ivcnz": [torch.tensor([2.0, 0.0])],
            "pcm": [torch.tensor([0.0, 2.0])],
        }
        conflict, cosine = apply_equal_domain_pcgrad(
            [parameter], buffers, Counter({"ivcnz": 2, "pcm": 1})
        )
        self.assertFalse(conflict)
        self.assertAlmostEqual(cosine, 0.0)
        torch.testing.assert_close(parameter.grad, torch.tensor([0.5, 1.0]))


if __name__ == "__main__":
    unittest.main()
