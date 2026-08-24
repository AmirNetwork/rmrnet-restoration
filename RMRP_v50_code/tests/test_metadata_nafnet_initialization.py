from __future__ import annotations

import unittest

import torch

from baselines.nafnet_metadata import MetadataNAFNetRoad
from baselines.nafnet_road import NAFNetRoad


class MetadataNAFNetInitializationTests(unittest.TestCase):
    def test_zero_film_preserves_plain_nafnet_function(self) -> None:
        torch.manual_seed(7)
        plain = NAFNetRoad(width=8, blocks_per_stage=1).eval()
        conditioned = MetadataNAFNetRoad(
            width=8,
            code_dim=82,
            blocks_per_stage=1,
        ).eval()
        conditioned.load_state_dict(plain.state_dict(), strict=False)

        image = torch.rand(2, 3, 32, 32)
        packet_a = torch.rand(2, 82)
        packet_b = torch.rand(2, 82)
        with torch.inference_mode():
            expected = plain(image)
            actual_a = conditioned(image, packet_a)
            actual_b = conditioned(image, packet_b)

        self.assertTrue(torch.equal(expected, actual_a))
        self.assertTrue(torch.equal(expected, actual_b))


if __name__ == "__main__":
    unittest.main()
