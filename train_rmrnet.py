"""Paper-facing RMR-Net training entry point.

The historical implementation lives in ``train_rcadnet.py`` because the
backbone was originally named RCADNet. The manuscript and release artifacts use
RMR-Net, so new commands should call this wrapper.
"""

from __future__ import annotations

from train_rcadnet import main


if __name__ == "__main__":
    main()
