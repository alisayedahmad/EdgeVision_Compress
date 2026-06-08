"""Tests for Module 4 - structured channel pruning.

All tests run on CPU with synthetic or unpretrained models.
No GPU, no MVTec dataset, no checkpoint required.
"""
import pytest
import torch
import torch.nn as nn
from torchvision.models.resnet import Bottleneck

from compression.pruning.flops import count_flops, format_flops
from compression.pruning.structured import (
    PrunedBottleneck,
    TaylorImportanceEstimator,
    _clone_bn,
    _clone_bn_subset,
    _rebuild_bottleneck,
    apply_structured_pruning,
    compute_keep_indices,
    get_bottleneck_locations,
)
from models.resnet import ResNet50AnomalyDetector


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def small_bottleneck() -> Bottleneck:
    """Bottleneck with inplanes=16, planes=4 — fast on CPU."""
    return Bottleneck(inplanes=16, planes=4)


@pytest.fixture()
def resnet_model() -> ResNet50AnomalyDetector:
    """Unpretrained ResNet50 for structural tests — no weight download."""
    return ResNet50AnomalyDetector(pretrained=False)


# ── BN helpers ────────────────────────────────────────────────────────────────


class TestCloneBN:
    def test_full_clone_same_num_features(self) -> None:
        bn = nn.BatchNorm2d(8)
        assert _clone_bn(bn).num_features == 8

    def test_full_clone_weights_match(self) -> None:
        bn = nn.BatchNorm2d(8)
        nn.init.uniform_(bn.weight)
        assert torch.allclose(_clone_bn(bn).weight, bn.weight)

    def test_subset_clone_correct_size(self) -> None:
        bn = nn.BatchNorm2d(16)
        keep = torch.tensor([0, 2, 5, 7])
        assert _clone_bn_subset(bn, keep).num_features == 4

    def test_subset_clone_values_match(self) -> None:
        bn = nn.BatchNorm2d(8)
        nn.init.uniform_(bn.weight)
        keep = torch.tensor([1, 3, 5])
        new_bn = _clone_bn_subset(bn, keep)
        assert torch.allclose(new_bn.weight, bn.weight[keep])

    def test_subset_running_mean_subset(self) -> None:
        bn = nn.BatchNorm2d(8)
        bn.running_mean = torch.arange(8, dtype=torch.float)
        keep = torch.tensor([0, 4, 7])
        new_bn = _clone_bn_subset(bn, keep)
        assert torch.allclose(new_bn.running_mean, torch.tensor([0.0, 4.0, 7.0]))


# ── PrunedBottleneck ──────────────────────────────────────────────────────────


class TestPrunedBottleneck:
    def test_forward_correct_output_shape(self) -> None:
        block = PrunedBottleneck(
            conv1=nn.Conv2d(64, 32, 1, bias=False),
            bn1=nn.BatchNorm2d(32),
            conv2=nn.Conv2d(32, 32, 3, padding=1, bias=False),
            bn2=nn.BatchNorm2d(32),
            conv3=nn.Conv2d(32, 64, 1, bias=False),
            bn3=nn.BatchNorm2d(64),
            downsample=None,
        )
        out = block(torch.randn(2, 64, 8, 8))
        assert out.shape == (2, 64, 8, 8)

    def test_forward_with_downsample(self) -> None:
        ds = nn.Sequential(nn.Conv2d(32, 64, 1, bias=False), nn.BatchNorm2d(64))
        block = PrunedBottleneck(
            conv1=nn.Conv2d(32, 16, 1, bias=False),
            bn1=nn.BatchNorm2d(16),
            conv2=nn.Conv2d(16, 16, 3, padding=1, bias=False),
            bn2=nn.BatchNorm2d(16),
            conv3=nn.Conv2d(16, 64, 1, bias=False),
            bn3=nn.BatchNorm2d(64),
            downsample=ds,
        )
        out = block(torch.randn(2, 32, 8, 8))
        assert out.shape == (2, 64, 8, 8)


# ── _rebuild_bottleneck ───────────────────────────────────────────────────────


class TestRebuildBottleneck:
    def test_conv2_output_channels_reduced(self, small_bottleneck: Bottleneck) -> None:
        keep = torch.tensor([0, 1, 2])
        new_block = _rebuild_bottleneck(small_bottleneck, keep)
        assert new_block.conv2.out_channels == 3

    def test_conv3_input_channels_updated(self, small_bottleneck: Bottleneck) -> None:
        keep = torch.tensor([0, 2])
        new_block = _rebuild_bottleneck(small_bottleneck, keep)
        assert new_block.conv3.in_channels == 2

    def test_conv3_output_channels_unchanged(self, small_bottleneck: Bottleneck) -> None:
        keep = torch.tensor([0, 1])
        new_block = _rebuild_bottleneck(small_bottleneck, keep)
        assert new_block.conv3.out_channels == small_bottleneck.conv3.out_channels

    def test_conv1_completely_unchanged(self, small_bottleneck: Bottleneck) -> None:
        keep = torch.tensor([0, 1])
        new_block = _rebuild_bottleneck(small_bottleneck, keep)
        assert new_block.conv1.in_channels == small_bottleneck.conv1.in_channels
        assert new_block.conv1.out_channels == small_bottleneck.conv1.out_channels

    def test_forward_runs_without_error(self, small_bottleneck: Bottleneck) -> None:
        keep = torch.tensor([0, 1, 2])
        new_block = _rebuild_bottleneck(small_bottleneck, keep)
        x = torch.randn(2, small_bottleneck.conv1.in_channels, 8, 8)
        out = new_block(x)
        assert out.shape == (2, small_bottleneck.conv3.out_channels, 8, 8)

    def test_kept_weights_are_copied(self, small_bottleneck: Bottleneck) -> None:
        keep = torch.tensor([0, 2])
        new_block = _rebuild_bottleneck(small_bottleneck, keep)
        expected = small_bottleneck.conv2.weight.data[keep]
        assert torch.allclose(new_block.conv2.weight.data, expected)


# ── get_bottleneck_locations ──────────────────────────────────────────────────


class TestGetBottleneckLocations:
    def test_resnet50_has_16_blocks(
        self, resnet_model: ResNet50AnomalyDetector
    ) -> None:
        # 3 + 4 + 6 + 3 = 16 Bottleneck blocks in ResNet50
        assert len(get_bottleneck_locations(resnet_model)) == 16

    def test_returns_sequential_parent(
        self, resnet_model: ResNet50AnomalyDetector
    ) -> None:
        for parent, idx, block in get_bottleneck_locations(resnet_model):
            assert isinstance(parent, nn.Sequential)
            assert isinstance(idx, int)


# ── compute_keep_indices ──────────────────────────────────────────────────────


class TestComputeKeepIndices:
    def test_global_removal_approximately_correct(self) -> None:
        scores = {"block_0": torch.rand(32), "block_1": torch.rand(64)}
        keep = compute_keep_indices(scores, prune_ratio=0.3)
        total_before = 96
        total_after = len(keep["block_0"]) + len(keep["block_1"])
        assert int(total_before * 0.65) <= total_after <= int(total_before * 0.75)

    def test_min_channels_floor_respected(self) -> None:
        scores = {"block_0": torch.zeros(8)}
        keep = compute_keep_indices(scores, prune_ratio=0.99, min_channels=4)
        assert len(keep["block_0"]) >= 4

    def test_invalid_ratio_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_keep_indices({"b": torch.rand(8)}, prune_ratio=0.0)

    def test_output_indices_are_sorted(self) -> None:
        scores = {"block_0": torch.rand(16)}
        keep = compute_keep_indices(scores, prune_ratio=0.3)
        idx = keep["block_0"].tolist()
        assert idx == sorted(idx)


# ── TaylorImportanceEstimator ─────────────────────────────────────────────────


class TestTaylorImportanceEstimator:
    def test_scores_correct_shape(self) -> None:
        conv = nn.Conv2d(8, 16, 3, padding=1)
        estimator = TaylorImportanceEstimator()
        estimator.register("test", conv)
        x = torch.randn(2, 8, 4, 4)
        out = conv(x)
        out.mean().backward()
        scores = estimator.get_scores()
        estimator.remove_hooks()
        assert scores["test"].shape == (16,)

    def test_scores_nonnegative(self) -> None:
        conv = nn.Conv2d(4, 8, 3, padding=1)
        estimator = TaylorImportanceEstimator()
        estimator.register("test", conv)
        conv(torch.randn(2, 4, 4, 4)).mean().backward()
        scores = estimator.get_scores()
        estimator.remove_hooks()
        assert (scores["test"] >= 0).all()

    def test_hooks_removed_cleanly(self) -> None:
        conv = nn.Conv2d(4, 8, 3, padding=1)
        n_before = len(conv._forward_hooks) + len(conv._backward_hooks)
        est = TaylorImportanceEstimator()
        est.register("test", conv)
        est.remove_hooks()
        assert len(conv._forward_hooks) + len(conv._backward_hooks) == n_before

    def test_no_accumulation_raises(self) -> None:
        with pytest.raises(RuntimeError, match="No scores accumulated"):
            TaylorImportanceEstimator().get_scores()


# ── FLOPs counter ─────────────────────────────────────────────────────────────


class TestFLOPs:
    def test_flops_positive(self, resnet_model: ResNet50AnomalyDetector) -> None:
        assert count_flops(resnet_model) > 0

    def test_pruned_model_fewer_flops(
        self, resnet_model: ResNet50AnomalyDetector
    ) -> None:
        flops_before = count_flops(resnet_model)
        locations = get_bottleneck_locations(resnet_model)
        parent, idx, block = locations[0]
        n = block.conv2.out_channels
        parent[idx] = _rebuild_bottleneck(block, torch.arange(n // 2))
        assert count_flops(resnet_model) < flops_before

    def test_format_gflops(self) -> None:
        assert "GFLOPs" in format_flops(4_100_000_000)

    def test_format_mflops(self) -> None:
        assert "MFLOPs" in format_flops(823_000_000)