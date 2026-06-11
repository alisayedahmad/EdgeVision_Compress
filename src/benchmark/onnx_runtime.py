"""ONNX Runtime inference benchmark for edge deployment simulation.

ONNX Runtime is the standard runtime for edge deployment on ARM
(Raspberry Pi, Jetson Nano). This module exports models to ONNX
and benchmarks them with ORT to simulate real edge performance



"""
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("edgevision")


def export_to_onnx(
    model: nn.Module,
    output_path: Path,
    input_shape: tuple = (1, 3, 224, 224),
    opset_version: int = 17,
) -> Path:
    """Export a PyTorch model to ONNX format.

    Args:
        model: Model to export (must be on CPU).
        output_path: Where to save the .onnx file.
        input_shape: Static input shape.
        opset_version: ONNX opset version.

    Returns:
        Path to saved ONNX file.
    """
    model.eval().cpu()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.zeros(input_shape)
    torch.onnx.export(
        model, dummy, str(output_path),
        input_names=["image"],
        output_names=["logit"],
        opset_version=opset_version,
        do_constant_folding=True,
    )
    size_mb = output_path.stat().st_size / 1e6
    logger.info("ONNX export -> %s (%.1f MB)", output_path, size_mb)
    return output_path


def benchmark_onnx(
    onnx_path: Path,
    input_shape: tuple = (1, 3, 224, 224),
    n_warmup: int = 10,
    n_runs: int = 100,
) -> dict[str, float]:
    """Benchmark an ONNX model using ONNX Runtime on CPU.

    Args:
        onnx_path: Path to the .onnx file.
        input_shape: Input tensor shape.
        n_warmup: Warm-up iterations (discarded).
        n_runs: Timed iterations.

    Returns:
        Dict with p50_ms, p95_ms, p99_ms, mean_ms, fps.
    """
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1  # single-thread = closer to edge CPU
    session = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])

    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(*input_shape).astype(np.float32)

    # warmup
    for _ in range(n_warmup):
        session.run(None, {input_name: dummy})

    # timed
    times_ms = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy})
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    times_ms.sort()
    n = len(times_ms)
    mean_ms = sum(times_ms) / n
    result = {
        "p50_ms": times_ms[int(n * 0.50)],
        "p95_ms": times_ms[int(n * 0.95)],
        "p99_ms": times_ms[int(n * 0.99)],
        "mean_ms": mean_ms,
        "fps": 1000.0 / mean_ms,
    }
    logger.info(
        "ONNX Runtime | p50=%.1f ms  p95=%.1f ms  fps=%.1f",
        result["p50_ms"], result["p95_ms"], result["fps"],
    )
    return result


def get_onnx_size_mb(onnx_path: Path) -> float:
    """Get ONNX file size in MB.

    Args:
        onnx_path: Path to .onnx file.

    Returns:
        File size in megabytes.
    """
    return Path(onnx_path).stat().st_size / 1e6