"""Tests for Module 7 — benchmark utilities."""
import torch
import torch.nn as nn

from benchmark.latency import benchmark_latency, get_model_size_mb
from compression.pruning.flops import count_flops, format_flops


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, 1)

    def forward(self, x):
        return self.fc(self.pool(self.conv(x)).flatten(1))


class TestFullBenchmarkComponents:
    def test_latency_and_flops_consistent(self) -> None:
        m = _TinyModel()
        lat = benchmark_latency(m, input_shape=(1, 3, 16, 16), n_warmup=2, n_runs=5)
        flops = count_flops(m, input_shape=(1, 3, 16, 16))
        assert lat["p50_ms"] > 0
        assert flops > 0

    def test_size_positive(self) -> None:
        assert get_model_size_mb(_TinyModel()) > 0

    def test_format_flops_readable(self) -> None:
        assert "FLOPs" in format_flops(1_000_000)

    def test_different_models_different_flops(self) -> None:
        small = _TinyModel()
        big = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Linear(64, 1),
        )
        assert count_flops(big, input_shape=(1, 3, 16, 16)) > count_flops(small, input_shape=(1, 3, 16, 16))