"""Tests for Module 5 -- quantization and latency benchmarking."""
import copy
import pytest
import torch
import torch.nn as nn

from benchmark.latency import benchmark_latency, get_model_size_mb
from compression.quantization.ptq import calibrate, convert_ptq, prepare_ptq
from compression.quantization.qat import convert_qat, export_onnx, prepare_qat


class SimpleModel(nn.Module):
    """Minimal traceable model with explicit forward."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8, 1)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.pool(x)
        x = self.flatten(x)
        return self.fc(x)


_EXAMPLE = (torch.randn(1, 3, 32, 32),)


def _fake_loader(n_batches=4):
    return [
        {"image": torch.randn(4, 3, 32, 32), "label": torch.randint(0, 2, (4,))}
        for _ in range(n_batches)
    ]


class TestBenchmarkLatency:
    def test_returns_expected_keys(self):
        result = benchmark_latency(SimpleModel(), input_shape=(1,3,32,32), n_warmup=2, n_runs=5)
        assert set(result.keys()) == {"p50_ms", "p95_ms", "p99_ms", "mean_ms", "fps"}

    def test_all_values_positive(self):
        result = benchmark_latency(SimpleModel(), input_shape=(1,3,32,32), n_warmup=2, n_runs=5)
        assert all(v > 0 for v in result.values())

    def test_percentile_order(self):
        result = benchmark_latency(SimpleModel(), input_shape=(1,3,32,32), n_warmup=2, n_runs=20)
        assert result["p50_ms"] <= result["p95_ms"] <= result["p99_ms"]

    def test_fps_consistent_with_mean(self):
        result = benchmark_latency(SimpleModel(), input_shape=(1,3,32,32), n_warmup=2, n_runs=10)
        assert abs(result["fps"] - 1000.0 / result["mean_ms"]) < 1e-3


class TestGetModelSizeMb:
    def test_positive_size(self):
        assert get_model_size_mb(SimpleModel()) > 0

    def test_larger_model_bigger(self):
        assert get_model_size_mb(nn.Linear(256, 128)) > get_model_size_mb(nn.Linear(8, 1))


class TestPTQ:
    def test_prepare_does_not_raise(self):
        prepare_ptq(SimpleModel().cpu(), example_input=_EXAMPLE)

    def test_calibrate_does_not_raise(self):
        prepared = prepare_ptq(SimpleModel().cpu(), example_input=_EXAMPLE)
        calibrate(prepared, _fake_loader(), torch.device("cpu"), n_batches=2)

    def test_convert_produces_callable_model(self):
        prepared = prepare_ptq(SimpleModel().cpu(), example_input=_EXAMPLE)
        calibrate(prepared, _fake_loader(), torch.device("cpu"), n_batches=2)
        quantized = convert_ptq(prepared)
        out = quantized(torch.zeros(1, 3, 32, 32))
        assert out.shape == (1, 1)

    def test_ptq_model_smaller_than_fp32(self):
        model_fp32 = SimpleModel().cpu()
        prepared = prepare_ptq(copy.deepcopy(model_fp32), example_input=_EXAMPLE)
        calibrate(prepared, _fake_loader(), torch.device("cpu"), n_batches=2)
        model_ptq = convert_ptq(prepared)
        assert get_model_size_mb(model_ptq) < get_model_size_mb(model_fp32)


class TestQAT:
    def test_prepare_does_not_raise(self):
        prepare_qat(SimpleModel().cpu(), example_input=_EXAMPLE)

    def test_forward_runs_during_qat(self):
        prepared = prepare_qat(SimpleModel().cpu(), example_input=_EXAMPLE)
        prepared.train()
        out = prepared(torch.randn(2, 3, 32, 32))
        assert out.shape == (2, 1)

    def test_convert_qat_callable(self):
        prepared = prepare_qat(SimpleModel().cpu(), example_input=_EXAMPLE)
        quantized = convert_qat(prepared)
        out = quantized(torch.zeros(1, 3, 32, 32))
        assert out.shape == (1, 1)

    def test_qat_model_smaller_than_fp32(self):
        model_fp32 = SimpleModel().cpu()
        prepared = prepare_qat(copy.deepcopy(model_fp32), example_input=_EXAMPLE)
        model_qat = convert_qat(prepared)
        assert get_model_size_mb(model_qat) < get_model_size_mb(model_fp32)


class TestONNXExport:
    @pytest.mark.xfail(reason='Quantized ONNX export unsupported on some PyTorch versions', raises=Exception)
    def test_onnx_file_created(self, tmp_path):
        prepared = prepare_qat(SimpleModel().cpu(), example_input=_EXAMPLE)
        quantized = convert_qat(prepared)
        out_path = tmp_path / "model.onnx"
        export_onnx(quantized, out_path, input_shape=(1, 3, 32, 32))
        assert out_path.exists()

    @pytest.mark.xfail(reason='Quantized ONNX export unsupported on some PyTorch versions', raises=Exception)
    def test_onnx_file_nonempty(self, tmp_path):
        prepared = prepare_qat(SimpleModel().cpu(), example_input=_EXAMPLE)
        quantized = convert_qat(prepared)
        out_path = tmp_path / "model.onnx"
        export_onnx(quantized, out_path, input_shape=(1, 3, 32, 32))
        assert out_path.stat().st_size > 0
