"""Reproducibility helpers.

One call to seed_everything() locks down every source of randomness
in the pipeline. Call it at the very start of any training script.
"""
import logging
import random

import numpy as np
import torch

logger = logging.getLogger("edgevision")


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch for full reproducibility.

    Also enables deterministic CUDA ops — slightly slower,
    but guarantees identical results across runs with the same seed.
    Call this before any model init, data loading, or random operation.

    Args:
        seed: The seed value. Keep it constant across runs you want to compare.

    Example:
        >>> seed_everything(42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic mode — worth the slight slowdown for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Global seed set to %d", seed)
