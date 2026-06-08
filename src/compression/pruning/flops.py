"""Simple FLOPs counter for Conv2d and Linear layers.

It uses forward hooks on the model as it runs, so shape changes from pruning
are naturally reflected in the final count.
"""
import logging

import torch
import torch.nn as nn

logger = logging.getLogger("edgevision")


def count_flops(
    model: nn.Module,
    input_shape: tuple[int, ...] = (1, 3, 224, 224),
    device: torch.device = torch.device("cpu"),
) -> int:
    """Count FLOPs for one forward pass using temporary hooks."""
    total: list[int] = [0]
    hooks: list = []

    def _conv_hook(module: nn.Conv2d, inp: tuple, out: torch.Tensor) -> None:
        _, c_out, h_out, w_out = out.shape
        kh, kw = (
            module.kernel_size
            if isinstance(module.kernel_size, tuple)
            else (module.kernel_size, module.kernel_size)
        )
        macs = c_out * h_out * w_out * (module.in_channels // module.groups) * kh * kw
        total[0] += 2 * macs

    def _linear_hook(module: nn.Linear, inp: tuple, out: torch.Tensor) -> None:
        total[0] += 2 * module.in_features * module.out_features

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(_conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(_linear_hook))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(input_shape, device=device))
    if was_training:
        model.train()

    for h in hooks:
        h.remove()

    return total[0]


def format_flops(n: int) -> str:
    """Format a raw FLOPs value into K/M/G units."""
    if n >= 1_000_000_000:
        return f"{n / 1e9:.2f} GFLOPs"
    if n >= 1_000_000:
        return f"{n / 1e6:.1f} MFLOPs"
    return f"{n / 1e3:.1f} KFLOPs"