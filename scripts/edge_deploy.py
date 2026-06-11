"""Edge deployment simulation: export models to ONNX and benchmark with ONNX Runtime.

Simulates Raspberry Pi / Jetson Nano deployment by running ONNX Runtime
in single-threaded CPU mode.

Usage:
    python scripts/edge_deploy.py "mlflow.tracking_uri=sqlite:///mlflow.db"
"""
import logging
import platform
import sys
from pathlib import Path

import hydra
import mlflow
import torch
import torch.nn as nn
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from benchmark.onnx_runtime import benchmark_onnx, export_to_onnx, get_onnx_size_mb
from compression.pruning.structured import (
    apply_structured_pruning,
    get_bottleneck_locations,
)
from models.mobilenet import MobileNetV3Student
from models.resnet import ResNet50AnomalyDetector
from utils.logging_config import setup_logging
from utils.seed import seed_everything

logger = setup_logging(name="edgevision.edge")


def _resolve_uri(raw: str, root: Path) -> str:
    if raw.startswith("sqlite:///"):
        return f"sqlite:///{root / raw.replace('sqlite:///', '', 1)}"
    return str(root / raw)


def _load_resnet(ckpt_path: Path, cfg: DictConfig) -> nn.Module:
    model = ResNet50AnomalyDetector(
        pretrained=False, hidden_dim=cfg.model.hidden_dim, dropout=cfg.model.dropout,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]

    locations = get_bottleneck_locations(model)
    keep = {}
    for i, (parent, bidx, block) in enumerate(locations):
        pidx = next(pi for pi, c in enumerate(model.feature_extractor) if c is parent)
        key = f"feature_extractor.{pidx}.{bidx}.conv2.weight"
        if key in sd and sd[key].shape[0] < block.conv2.out_channels:
            keep[f"block_{i}"] = torch.arange(sd[key].shape[0])
    if keep:
        apply_structured_pruning(model, keep)

    model.load_state_dict(sd)
    return model


@hydra.main(version_base=None, config_path="../configs", config_name="benchmark")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    project_root = Path(get_original_cwd())
    ckpt_dir = project_root / "outputs" / "checkpoints"
    onnx_dir = project_root / "outputs" / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    n_warmup = cfg.benchmark.n_warmup
    n_runs = cfg.benchmark.n_runs
    results = []

    # ── Baseline ResNet50 ─────────────────────────────────────────────────
    bp = ckpt_dir / "best_model.pth"
    if bp.exists():
        logger.info("Exporting baseline ResNet50...")
        m = _load_resnet(bp, cfg)
        onnx_path = export_to_onnx(m, onnx_dir / "baseline.onnx")
        lat = benchmark_onnx(onnx_path, n_warmup=n_warmup, n_runs=n_runs)
        results.append({"name": "ResNet50 baseline", "latency": lat, "size_mb": get_onnx_size_mb(onnx_path)})

    # ── Structured pruned ─────────────────────────────────────────────────
    sp = ckpt_dir / "pruned_structured.pth"
    if sp.exists():
        logger.info("Exporting structured pruned...")
        m = _load_resnet(sp, cfg)
        onnx_path = export_to_onnx(m, onnx_dir / "structured_pruned.onnx")
        lat = benchmark_onnx(onnx_path, n_warmup=n_warmup, n_runs=n_runs)
        results.append({"name": "Structured pruned", "latency": lat, "size_mb": get_onnx_size_mb(onnx_path)})

    # ── Student MobileNetV3 ───────────────────────────────────────────────
    dp = ckpt_dir / "student_distilled.pth"
    if dp.exists():
        logger.info("Exporting student MobileNetV3...")
        m = MobileNetV3Student(pretrained=False, hidden_dim=128, dropout=0.3)
        ckpt = torch.load(dp, map_location="cpu", weights_only=False)
        m.load_state_dict(ckpt["model_state_dict"])
        onnx_path = export_to_onnx(m, onnx_dir / "student.onnx")
        lat = benchmark_onnx(onnx_path, n_warmup=n_warmup, n_runs=n_runs)
        results.append({"name": "MobileNetV3 student", "latency": lat, "size_mb": get_onnx_size_mb(onnx_path)})

    # ── MLflow ────────────────────────────────────────────────────────────
    tracking_uri = _resolve_uri(cfg.mlflow.tracking_uri, project_root)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(run_name="edge_deployment"):
        for r in results:
            prefix = r["name"].lower().replace(" ", "_")
            mlflow.log_metrics({
                f"ort_{prefix}_p50_ms": r["latency"]["p50_ms"],
                f"ort_{prefix}_p95_ms": r["latency"]["p95_ms"],
                f"ort_{prefix}_fps": r["latency"]["fps"],
                f"ort_{prefix}_size_mb": r["size_mb"],
            })

    # ── Results ───────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 75)
    logger.info("EDGE DEPLOYMENT SIMULATION (ONNX Runtime, single-thread CPU)")
    logger.info("=" * 75)
    logger.info("%-22s  %-10s  %-10s  %-10s  %-10s", "Model", "P50 (ms)", "P95 (ms)", "FPS", "Size (MB)")
    logger.info("-" * 75)
    for r in results:
        logger.info(
            "%-22s  %-10.1f  %-10.1f  %-10.1f  %-10.1f",
            r["name"], r["latency"]["p50_ms"], r["latency"]["p95_ms"],
            r["latency"]["fps"], r["size_mb"],
        )
    logger.info("-" * 75)

    if len(results) >= 2:
        first, last = results[0], results[-1]
        speedup = first["latency"]["p50_ms"] / max(last["latency"]["p50_ms"], 0.01)
        compression = first["size_mb"] / max(last["size_mb"], 0.01)
        logger.info("vs baseline: %.1fx faster | %.1fx smaller", speedup, compression)

    # Pi4 estimate (ARM Cortex-A72 is ~5-8x slower than desktop x86)
    PI4_FACTOR = 6.0
    logger.info("")
    logger.info("Raspberry Pi 4 estimate (%.0fx slower than desktop):", PI4_FACTOR)
    for r in results:
        est = r["latency"]["p50_ms"] * PI4_FACTOR
        logger.info(
            "  %-22s  ~%.0f ms  (%.1f FPS)",
            r["name"], est, 1000.0 / est,
        )
    logger.info("=" * 75)


if __name__ == "__main__":
    main()