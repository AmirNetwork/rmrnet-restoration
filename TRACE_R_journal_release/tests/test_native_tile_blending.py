# Author: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
from __future__ import annotations

import torch

from tools.restore_native_yolo_split import restore_global_residual, restore_tiled


def test_identity_tiled_restoration_preserves_borders() -> None:
    """Raised-edge blending must not vignette native-resolution images."""
    image = torch.linspace(0.05, 0.95, 3 * 53 * 79, dtype=torch.float32).reshape(
        1, 3, 53, 79
    )

    restored = restore_tiled(image, lambda patch: patch, tile=32, overlap=8)

    torch.testing.assert_close(restored, image, rtol=0.0, atol=2e-7)
    torch.testing.assert_close(restored[..., 0, 0], image[..., 0, 0], rtol=0.0, atol=2e-7)
    torch.testing.assert_close(restored[..., -1, -1], image[..., -1, -1], rtol=0.0, atol=2e-7)


def test_halo_tiled_restoration_writes_only_contextualized_core() -> None:
    """Reflected halo context must preserve native coordinates and pixels."""
    image = torch.linspace(0.05, 0.95, 3 * 53 * 79, dtype=torch.float32).reshape(
        1, 3, 53, 79
    )

    restored = restore_tiled(
        image,
        lambda patch: patch,
        tile=32,
        overlap=8,
        halo=6,
    )

    assert restored.shape == image.shape
    torch.testing.assert_close(restored, image, rtol=0.0, atol=2e-7)


def test_global_residual_restoration_preserves_native_identity() -> None:
    image = torch.linspace(0.05, 0.95, 3 * 53 * 79, dtype=torch.float32).reshape(
        1, 3, 53, 79
    )

    restored = restore_global_residual(
        image,
        lambda resized: resized,
        long_side=40,
    )

    assert restored.shape == image.shape
    torch.testing.assert_close(restored, image, rtol=0.0, atol=2e-7)
