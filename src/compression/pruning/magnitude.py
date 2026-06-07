"""Global L1 unstructured magnitude pruning utilities.

The core mechanic: rank every weight in the network by absolute value,
zero out the bottom N%, and optionally make that permanent.

PyTorch implements this via a reparameterization trick:
  weight = weight_orig * weight_mask
where weight_orig stores the original floats and weight_mask is a binary
tensor. Gradients flow through weight_orig only — masked weights never
recover through training, which is exactly what we want.
"""
import logging

import torch
import torch.nn as nn
from torch.nn.utils import prune

logger = logging.getLogger("edgevision")


def get_prunable_parameters(model: nn.Module) -> list[tuple[nn.Module, str]]:
    """Return all Conv2d and Linear weight parameter pairs.


    Args:
        model: The model to inspect.

    Returns:
        List of (module, "weight") pairs ready for global_unstructured.
    """
    params = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            params.append((module, "weight"))
    logger.debug("Found %d prunable parameter groups.", len(params))
    return params


def apply_global_l1_pruning(model: nn.Module, target_sparsity: float) -> None:
    """Zero out weights below the L1-magnitude threshold for a cumulative target.


    Args:
        model: Model to prune in-place. Modified directly.
        target_sparsity: Cumulative fraction of weights to zero out (0.0, 1.0).

    Raises:
        ValueError: If target_sparsity is not strictly in (0, 1).
    """
    if not 0.0 < target_sparsity < 1.0:
        raise ValueError(
            f"target_sparsity must be in (0, 1), got {target_sparsity:.3f}."
        )

    params = get_prunable_parameters(model)
    prune.global_unstructured(
        params,
        pruning_method=prune.L1Unstructured,
        amount=target_sparsity,
    )
    actual = get_model_sparsity(model)
    logger.info(
        "Global L1 pruning applied | target=%.3f | actual=%.4f",
        target_sparsity,
        actual,
    )


def make_pruning_permanent(model: nn.Module) -> None:
    """Convert soft masks into hard zeros and remove the reparameterization.

    After this call:
      - weight_orig and weight_mask are removed from every pruned module
      - module.weight contains the masked values (zeros where pruned)
      - The model behaves identically but has no pruning bookkeeping overhead

 

    Args:
        model: Model with active PyTorch pruning reparameterization.
    """
    n_removed = 0
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if prune.is_pruned(module):
                prune.remove(module, "weight")
                n_removed += 1
    logger.info(
        "Pruning made permanent — removed reparameterization from %d layers.",
        n_removed,
    )


def get_model_sparsity(model: nn.Module) -> float:
    """Compute the fraction of exactly-zero weights across all prunable layers.

    Args:
        model: The model to measure.

    Returns:
        Sparsity in [0.0, 1.0]. 0.0 means fully dense, 1.0 means all zeros.
    """
    total = 0
    zeros = 0
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight
            total += w.numel()
            zeros += int((w == 0).sum().item())
    return zeros / total if total > 0 else 0.0


def get_sparsity_schedule(
    target_sparsity: float,
    n_iterations: int,
) -> list[float]:
    """Build a linear cumulative sparsity schedule.

    Returns the cumulative target at each step — not the incremental delta.
    Pass each value directly to apply_global_l1_pruning().



    Args:
        target_sparsity: Desired final sparsity (e.g., 0.5).
        n_iterations: Number of pruning + fine-tuning cycles.

    Returns:
        List of cumulative sparsity targets, length == n_iterations.

    Example:
        >>> get_sparsity_schedule(0.5, 5)
        [0.1, 0.2, 0.3, 0.4, 0.5]
    """
    return [
        target_sparsity * (i + 1) / n_iterations
        for i in range(n_iterations)
    ]