"""Tests for Module 3 — unstructured magnitude pruning.

All tests run on CPU with tiny synthetic models.
No GPU, no MVTec dataset, no checkpoint required.
"""
import pytest
import torch
import torch.nn as nn
from torch.nn.utils import prune as torch_prune

from compression.pruning.magnitude import (
    apply_global_l1_pruning,
    get_model_sparsity,
    get_prunable_parameters,
    get_sparsity_schedule,
    make_pruning_permanent,
)


# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def small_model() -> nn.Module:
    """Minimal conv+linear model — fast to prune and inspect."""
    return nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, 4),
    )


@pytest.fixture()
def model_with_bn() -> nn.Module:
    """Model with BatchNorm — used to verify BN is excluded from pruning."""
    return nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.BatchNorm2d(8),
        nn.Flatten(),
        nn.Linear(8, 2),
    )


# ── get_prunable_parameters ───────────────────────────────────────────────────


class TestGetPrunableParameters:
    def test_returns_conv_and_linear(self, small_model: nn.Module) -> None:
        params = get_prunable_parameters(small_model)
        types = {type(m).__name__ for m, _ in params}
        assert "Conv2d" in types and "Linear" in types

    def test_all_parameter_names_are_weight(self, small_model: nn.Module) -> None:
        params = get_prunable_parameters(small_model)
        assert all(name == "weight" for _, name in params)

    def test_batchnorm_excluded(self, model_with_bn: nn.Module) -> None:
        params = get_prunable_parameters(model_with_bn)
        modules = [m for m, _ in params]
        assert not any(isinstance(m, nn.BatchNorm2d) for m in modules)

    def test_count_matches_conv_linear_layers(self, small_model: nn.Module) -> None:
        # small_model has 1 Conv2d + 1 Linear = 2 prunable groups
        params = get_prunable_parameters(small_model)
        assert len(params) == 2


# ── get_model_sparsity ────────────────────────────────────────────────────────


class TestGetModelSparsity:
    def test_dense_model_is_zero(self, small_model: nn.Module) -> None:
        assert get_model_sparsity(small_model) == pytest.approx(0.0, abs=1e-6)

    def test_after_pruning_sparsity_near_target(
        self, small_model: nn.Module
    ) -> None:
        apply_global_l1_pruning(small_model, 0.3)
        sparsity = get_model_sparsity(small_model)
        # global pruning approximates the target — allow ±5% tolerance
        assert 0.25 <= sparsity <= 0.40

    def test_sparsity_in_unit_interval(self, small_model: nn.Module) -> None:
        apply_global_l1_pruning(small_model, 0.5)
        s = get_model_sparsity(small_model)
        assert 0.0 <= s <= 1.0


# ── apply_global_l1_pruning ───────────────────────────────────────────────────


class TestApplyGlobalL1Pruning:
    def test_zero_sparsity_raises(self, small_model: nn.Module) -> None:
        with pytest.raises(ValueError):
            apply_global_l1_pruning(small_model, 0.0)

    def test_one_sparsity_raises(self, small_model: nn.Module) -> None:
        with pytest.raises(ValueError):
            apply_global_l1_pruning(small_model, 1.0)

    def test_creates_pruning_reparameterization(
        self, small_model: nn.Module
    ) -> None:
        apply_global_l1_pruning(small_model, 0.3)
        for module in small_model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                assert torch_prune.is_pruned(module)

    def test_cumulative_sparsity_is_monotone(
        self, small_model: nn.Module
    ) -> None:
        """Higher cumulative targets must not decrease actual sparsity."""
        apply_global_l1_pruning(small_model, 0.2)
        s1 = get_model_sparsity(small_model)
        apply_global_l1_pruning(small_model, 0.4)
        s2 = get_model_sparsity(small_model)
        assert s2 >= s1

    def test_weights_can_be_used_in_forward(
        self, small_model: nn.Module
    ) -> None:
        """Model must remain usable after pruning (no broken tensors)."""
        apply_global_l1_pruning(small_model, 0.3)
        x = torch.randn(2, 3, 16, 16)
        out = small_model(x)
        assert out.shape == (2, 4)


# ── make_pruning_permanent ────────────────────────────────────────────────────


class TestMakePruningPermanent:
    def test_removes_reparameterization(self, small_model: nn.Module) -> None:
        apply_global_l1_pruning(small_model, 0.3)
        make_pruning_permanent(small_model)
        for module in small_model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                assert not torch_prune.is_pruned(module)

    def test_zeros_preserved_after_permanent(
        self, small_model: nn.Module
    ) -> None:
        apply_global_l1_pruning(small_model, 0.3)
        sparsity_before = get_model_sparsity(small_model)
        make_pruning_permanent(small_model)
        sparsity_after = get_model_sparsity(small_model)
        assert abs(sparsity_before - sparsity_after) < 0.01

    def test_idempotent_on_dense_model(self, small_model: nn.Module) -> None:
        """Must not raise on a model that has never been pruned."""
        make_pruning_permanent(small_model)  # should be a no-op

    def test_weight_orig_removed(self, small_model: nn.Module) -> None:
        """weight_orig must not exist after make_pruning_permanent."""
        apply_global_l1_pruning(small_model, 0.3)
        make_pruning_permanent(small_model)
        for module in small_model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                assert not hasattr(module, "weight_orig")


# ── get_sparsity_schedule ─────────────────────────────────────────────────────


class TestGetSparsitySchedule:
    def test_length_matches_n_iterations(self) -> None:
        assert len(get_sparsity_schedule(0.5, 5)) == 5

    def test_final_value_equals_target(self) -> None:
        schedule = get_sparsity_schedule(0.5, 5)
        assert schedule[-1] == pytest.approx(0.5)

    def test_strictly_increasing(self) -> None:
        schedule = get_sparsity_schedule(0.5, 5)
        assert all(schedule[i] < schedule[i + 1] for i in range(len(schedule) - 1))

    def test_linear_values(self) -> None:
        schedule = get_sparsity_schedule(0.5, 5)
        expected = [0.1, 0.2, 0.3, 0.4, 0.5]
        for got, exp in zip(schedule, expected):
            assert got == pytest.approx(exp, rel=1e-5)

    def test_single_iteration(self) -> None:
        schedule = get_sparsity_schedule(0.5, 1)
        assert len(schedule) == 1
        assert schedule[0] == pytest.approx(0.5)