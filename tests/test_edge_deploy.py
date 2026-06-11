"""Tests for Module 8 — ONNX export and runtime benchmark."""
import pytest
import torch
import torch.nn as nn

from benchmark.onnx_runtime import export_to_onnx, get_onnx_size_mb


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, 1)

    def forward(self, x):
        return self.fc(self.pool(self.conv(x)).flatten(1))


class TestONNXExport:
    def test_file_created(self, tmp_path) -> None:
        p = export_to_onnx(_TinyModel(), tmp_path / "m.onnx", input_shape=(1, 3, 16, 16))
        assert p.exists()

    def test_file_nonempty(self, tmp_path) -> None:
        p = export_to_onnx(_TinyModel(), tmp_path / "m.onnx", input_shape=(1, 3, 16, 16))
        assert p.stat().st_size > 0

    def test_size_positive(self, tmp_path) -> None:
        p = export_to_onnx(_TinyModel(), tmp_path / "m.onnx", input_shape=(1, 3, 16, 16))
        assert get_onnx_size_mb(p) > 0


class TestONNXRuntimeBenchmark:
    @pytest.mark.xfail(reason="onnxruntime may not be installed", raises=ImportError)
    def test_benchmark_returns_keys(self, tmp_path) -> None:
        from benchmark.onnx_runtime import benchmark_onnx
        p = export_to_onnx(_TinyModel(), tmp_path / "m.onnx", input_shape=(1, 3, 16, 16))
        result = benchmark_onnx(p, input_shape=(1, 3, 16, 16), n_warmup=2, n_runs=5)
        assert set(result.keys()) == {"p50_ms", "p95_ms", "p99_ms", "mean_ms", "fps"}

    @pytest.mark.xfail(reason="onnxruntime may not be installed", raises=ImportError)
    def test_benchmark_positive_values(self, tmp_path) -> None:
        from benchmark.onnx_runtime import benchmark_onnx
        p = export_to_onnx(_TinyModel(), tmp_path / "m.onnx", input_shape=(1, 3, 16, 16))
        result = benchmark_onnx(p, input_shape=(1, 3, 16, 16), n_warmup=2, n_runs=5)
        assert all(v > 0 for v in result.values())