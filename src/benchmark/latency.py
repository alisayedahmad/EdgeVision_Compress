"""CPU latency benchmarking for quantized and full-precision models.

Measures wall-clock inference time with proper warm-up to eliminate
JIT compilation and cache cold-start effects from the measurements.

All benchmarks run on CPU — that is the target deployment hardware
(Raspberry Pi 4, Jetson Nano in CPU mode).
"""
import logging
import time

import torch
import torch.nn as nn

logger = logging.getLogger("edgevision")


def benchmark_latency(
    model: nn.Module,
    input_shape: tuple[int, ...] = (1, 3, 224, 224),
    n_warmup: int = 10,
    n_runs: int = 100,
    device: torch.device = torch.device("cpu"),
) -> dict[str, float]:
    """Measure inference latency percentiles on CPU.

    Warm-up runs are discarded — they include JIT compilation overhead
    and CPU cache cold-start effects that don't represent steady-state
    performance. After warm-up, we collect n_runs timings and compute
    percentiles.

    Args:
        model: Model to benchmark.
        input_shape: Input tensor shape including batch dimension.
        n_warmup: Number of warm-up forward passes (discarded).
        n_runs: Number of timed forward passes.
        device: Benchmark device. Always CPU for edge deployment targets.

    Returns:
        Dict with keys: p50_ms, p95_ms, p99_ms, mean_ms, fps.
    """
    model = model.to(device)
    model.eval()
    dummy = torch.zeros(input_shape, device=device)

    # warm-up — trigger JIT, fill CPU caches
    with torch.no_grad():
        for _ in range(n_warmup):
            model(dummy)

    # timed runs
    times_ms: list[float] = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(dummy)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    times_ms.sort()
    n = len(times_ms)
    mean_ms = sum(times_ms) / n

    result = {
        "p50_ms":  times_ms[int(n * 0.50)],
        "p95_ms":  times_ms[int(n * 0.95)],
        "p99_ms":  times_ms[int(n * 0.99)],
        "mean_ms": mean_ms,
        "fps":     1000.0 / mean_ms,
    }

    logger.info(
        "Latency | p50=%.1f ms  p95=%.1f ms  p99=%.1f ms  fps=%.1f",
        result["p50_ms"], result["p95_ms"], result["p99_ms"], result["fps"],
    )
    return result


def get_model_size_mb(model: nn.Module) -> float:
    """Compute model size in megabytes by summing all parameter tensors.

    Works for both FP32 and quantized INT8 models because it reads the
    actual dtype of each parameter tensor, not just the count.

    Args:
        model: Any nn.Module.

    Returns:
        Total size of all parameters in megabytes.
    """
    total_bytes = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )
    # include buffers (running mean/var in BN, quantization scales)
    total_bytes += sum(
        b.numel() * b.element_size() for b in model.buffers()
    )
    return total_bytes / 1e6