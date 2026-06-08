"""Structured channel pruning with a first-order Taylor score.

We prune conv2 output channels inside each ResNet50 bottleneck block.
This keeps skip-connection dimensions intact.
"""
import logging
from typing import Optional

import torch
import torch.nn as nn
from torchvision.models.resnet import Bottleneck

logger = logging.getLogger("edgevision")


# ── BatchNorm helpers ─────────────────────────────────────────────────────────


def _clone_bn(bn: nn.BatchNorm2d) -> nn.BatchNorm2d:
    """Return a full copy of a BatchNorm2d layer."""
    new_bn = nn.BatchNorm2d(
        bn.num_features, eps=bn.eps, momentum=bn.momentum, affine=bn.affine
    )
    if bn.affine:
        new_bn.weight.data.copy_(bn.weight.data)
        new_bn.bias.data.copy_(bn.bias.data)
    new_bn.running_mean.data.copy_(bn.running_mean.data)
    new_bn.running_var.data.copy_(bn.running_var.data)
    new_bn.num_batches_tracked.copy_(bn.num_batches_tracked)
    return new_bn


def _clone_bn_subset(bn: nn.BatchNorm2d, keep_idx: torch.Tensor) -> nn.BatchNorm2d:
    """Copy a BatchNorm2d layer but only keep selected channels."""
    n = len(keep_idx)
    new_bn = nn.BatchNorm2d(n, eps=bn.eps, momentum=bn.momentum, affine=bn.affine)
    if bn.affine:
        new_bn.weight.data.copy_(bn.weight.data[keep_idx])
        new_bn.bias.data.copy_(bn.bias.data[keep_idx])
    new_bn.running_mean.data.copy_(bn.running_mean.data[keep_idx])
    new_bn.running_var.data.copy_(bn.running_var.data[keep_idx])
    new_bn.num_batches_tracked.copy_(bn.num_batches_tracked)
    return new_bn


# ── Pruned block ──────────────────────────────────────────────────────────────


class PrunedBottleneck(nn.Module):
    """Bottleneck block with a narrower conv2 channel width.

    Built by _rebuild_bottleneck after channel selection.
    """

    def __init__(
        self,
        conv1: nn.Conv2d,
        bn1: nn.BatchNorm2d,
        conv2: nn.Conv2d,
        bn2: nn.BatchNorm2d,
        conv3: nn.Conv2d,
        bn3: nn.BatchNorm2d,
        downsample: Optional[nn.Module],
    ) -> None:
        super().__init__()
        self.conv1 = conv1
        self.bn1 = bn1
        self.conv2 = conv2
        self.bn2 = bn2
        self.conv3 = conv3
        self.bn3 = bn3
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard bottleneck forward pass."""
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


# ── Taylor importance estimator ───────────────────────────────────────────────


class TaylorImportanceEstimator:
    """Estimate filter importance with first-order Taylor scores."""

    def __init__(self) -> None:
        self._hooks: list = []
        self._activations: dict[str, torch.Tensor] = {}
        self._score_sums: dict[str, torch.Tensor] = {}
        self._counts: dict[str, int] = {}

    def register(self, name: str, module: nn.Conv2d) -> None:
        """Attach hooks to track activations and gradients for one layer."""

        def fwd(mod: nn.Module, inp: tuple, out: torch.Tensor) -> None:
            self._activations[name] = out  # [B, C_out, H, W]

        def bwd(mod: nn.Module, grad_in: tuple, grad_out: tuple) -> None:
            g = grad_out[0]
            a = self._activations.get(name)
            if g is None or a is None:
                return
            # Per-filter score: mean(|activation * gradient|) across batch and space.
            score = (a * g).abs().mean(dim=(0, 2, 3)).detach().cpu()  # [C_out]
            if name not in self._score_sums:
                self._score_sums[name] = score
                self._counts[name] = 1
            else:
                self._score_sums[name] += score
                self._counts[name] += 1

        self._hooks.append(module.register_forward_hook(fwd))
        self._hooks.append(module.register_full_backward_hook(bwd))

    def accumulate(
        self,
        model: nn.Module,
        loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        device: torch.device,
        n_batches: int,
    ) -> None:
        """Run a few train-mode steps to accumulate importance scores."""
        model.train()
        processed = 0
        for batch in loader:
            if processed >= n_batches:
                break
            images = batch["image"].to(device)
            labels = batch["label"].float().to(device)
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)
            model.zero_grad()
            loss.backward()
            processed += 1

        logger.info(
            "Taylor scores accumulated | %d batches | %d layers monitored.",
            processed,
            len(self._score_sums),
        )

    def get_scores(self) -> dict[str, torch.Tensor]:
        """Return the mean score tensor for each registered layer."""
        if not self._score_sums:
            raise RuntimeError(
                "No scores accumulated. Call accumulate() before get_scores()."
            )
        return {
            name: self._score_sums[name] / self._counts[name]
            for name in self._score_sums
        }

    def remove_hooks(self) -> None:
        """Remove all hooks and clear cached activations."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._activations.clear()


# ── Block discovery ───────────────────────────────────────────────────────────


def get_bottleneck_locations(
    model: nn.Module,
) -> list[tuple[nn.Sequential, int, nn.Module]]:
    """Return all bottleneck block locations in feature_extractor order."""
    locations = []
    for child in model.feature_extractor:
        if not isinstance(child, nn.Sequential):
            continue
        for idx, block in enumerate(child):
            if isinstance(block, (Bottleneck, PrunedBottleneck)):
                locations.append((child, idx, block))
    return locations


# ── Keep-index selection ──────────────────────────────────────────────────────


def compute_keep_indices(
    scores: dict[str, torch.Tensor],
    prune_ratio: float,
    min_channels: int = 4,
) -> dict[str, torch.Tensor]:
    """Pick channels to keep using global ranking across all layers."""
    if not 0.0 < prune_ratio < 1.0:
        raise ValueError(f"prune_ratio must be in (0, 1), got {prune_ratio}")

    # Flatten (score, layer_name, channel_idx) across all layers.
    all_entries: list[tuple[float, str, int]] = []
    for name, layer_scores in scores.items():
        for i, val in enumerate(layer_scores.tolist()):
            all_entries.append((val, name, i))

    n_total = len(all_entries)
    n_prune = int(n_total * prune_ratio)
    all_entries.sort(key=lambda e: e[0])  # Lowest score first.

    pruned: dict[str, set[int]] = {name: set() for name in scores}
    for _, name, idx in all_entries[:n_prune]:
        pruned[name].add(idx)

    keep: dict[str, torch.Tensor] = {}
    for name, layer_scores in scores.items():
        n = len(layer_scores)
        kept = [i for i in range(n) if i not in pruned[name]]

        # Keep at least min_channels per layer.
        if len(kept) < min_channels:
            extra = sorted(
                [(layer_scores[i].item(), i) for i in pruned[name]],
                reverse=True,
            )
            for _, i in extra:
                kept.append(i)
                if len(kept) >= min_channels:
                    break

        keep[name] = torch.tensor(sorted(kept), dtype=torch.long)
        logger.debug(
            "%-12s  %d -> %d channels (%.0f%% pruned)",
            name,
            n,
            len(keep[name]),
            100.0 * (n - len(keep[name])) / n,
        )

    return keep


# ── Block rebuild ─────────────────────────────────────────────────────────────


def _rebuild_bottleneck(
    block: nn.Module,
    keep_idx: torch.Tensor,
) -> PrunedBottleneck:
    """Rebuild one bottleneck after pruning conv2 output channels."""
    n_keep = len(keep_idx)

    # conv1 + bn1 stay unchanged.
    new_conv1 = nn.Conv2d(
        block.conv1.in_channels, block.conv1.out_channels, 1, bias=False
    )
    new_conv1.weight.data.copy_(block.conv1.weight.data)
    new_bn1 = _clone_bn(block.bn1)

    # conv2 keeps input width but outputs fewer channels.
    new_conv2 = nn.Conv2d(
        block.conv2.in_channels,
        n_keep,
        kernel_size=3,
        stride=block.conv2.stride[0],
        padding=1,
        bias=False,
    )
    new_conv2.weight.data.copy_(block.conv2.weight.data[keep_idx])
    new_bn2 = _clone_bn_subset(block.bn2, keep_idx)

    # conv3 now reads fewer input channels.
    new_conv3 = nn.Conv2d(n_keep, block.conv3.out_channels, 1, bias=False)
    new_conv3.weight.data.copy_(block.conv3.weight.data[:, keep_idx])
    new_bn3 = _clone_bn(block.bn3)

    return PrunedBottleneck(
        conv1=new_conv1,
        bn1=new_bn1,
        conv2=new_conv2,
        bn2=new_bn2,
        conv3=new_conv3,
        bn3=new_bn3,
        downsample=block.downsample,
    )


# ── Apply to full model ───────────────────────────────────────────────────────


def apply_structured_pruning(
    model: nn.Module,
    keep_indices: dict[str, torch.Tensor],
) -> nn.Module:
    """Replace bottleneck blocks in-place using the provided keep indices."""
    locations = get_bottleneck_locations(model)
    n_modified = 0

    for i, (parent, idx, block) in enumerate(locations):
        key = f"block_{i}"
        if key not in keep_indices:
            continue
        ki = keep_indices[key]
        orig_ch = block.conv2.out_channels
        if len(ki) == orig_ch:
            logger.debug("Block %d: all channels kept, skipping.", i)
            continue

        parent[idx] = _rebuild_bottleneck(block, ki)
        n_modified += 1
        logger.info(
            "Block %2d | conv2: %3d -> %3d channels (%.0f%% removed)",
            i,
            orig_ch,
            len(ki),
            100.0 * (orig_ch - len(ki)) / orig_ch,
        )

    logger.info(
        "Structured pruning complete — %d / %d blocks modified.",
        n_modified,
        len(locations),
    )
    return model