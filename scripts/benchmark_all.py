"""Unified benchmark: loads every checkpoint and compares all models side by side.

Usage:
    python scripts/benchmark_all.py "mlflow.tracking_uri=sqlite:///mlflow.db"
"""
import logging
import platform
import sys
from pathlib import Path

import hydra
import mlflow
import numpy as np
import torch
import torch.nn as nn
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from benchmark.latency import benchmark_latency, get_model_size_mb
from compression.pruning.flops import count_flops, format_flops
from compression.pruning.structured import (
    apply_structured_pruning,
    get_bottleneck_locations,
)
from data.mvtec import MVTecDataset
from evaluation.metrics import compute_auroc
from models.mobilenet import MobileNetV3Student
from models.resnet import ResNet50AnomalyDetector
from utils.logging_config import setup_logging
from utils.seed import seed_everything

logger = setup_logging(name="edgevision.benchmark")


def _resolve_uri(raw: str, root: Path) -> str:
    if raw.startswith("sqlite:///"):
        return f"sqlite:///{root / raw.replace('sqlite:///', '', 1)}"
    return str(root / raw)


def _load_resnet(ckpt_path: Path, cfg: DictConfig, device: torch.device) -> nn.Module:
    """Load ResNet50, rebuilding pruned architecture if needed."""
    model = ResNet50AnomalyDetector(
        pretrained=False, hidden_dim=cfg.model.hidden_dim, dropout=cfg.model.dropout,
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
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
    return model.to(device)


def _load_student(ckpt_path: Path, device: torch.device) -> nn.Module:
    model = MobileNetV3Student(pretrained=False, hidden_dim=128, dropout=0.3)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device)


@torch.no_grad()
def _eval_auroc(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    scores, labels = [], []
    for batch in loader:
        logits = model(batch["image"].to(device)).squeeze(1)
        scores.extend(torch.sigmoid(logits).cpu().tolist())
        labels.extend(batch["label"].tolist())
    return compute_auroc(labels, scores)


@hydra.main(version_base=None, config_path="../configs", config_name="benchmark")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    cpu = torch.device("cpu")
    project_root = Path(get_original_cwd())
    ckpt_dir = project_root / "outputs" / "checkpoints"

    # data
    data_dir = project_root / cfg.paths.data_dir if hasattr(cfg.paths, "data_dir") else project_root / "data" / "mvtec"
    nw = 0 if platform.system() == "Windows" else 4
    category = cfg.get("data", {}).get("category", "bottle") if hasattr(cfg, "data") else "bottle"
    val_frac = cfg.get("data", {}).get("val_fraction", 0.2) if hasattr(cfg, "data") else 0.2

    test_ds = MVTecDataset(data_dir, category, "test", seed=cfg.seed, val_fraction=val_frac)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=nw)

    n_warmup = cfg.benchmark.n_warmup
    n_runs = cfg.benchmark.n_runs

    # collect results
    results = []

    # ── Baseline ──────────────────────────────────────────────────────────
    bp = ckpt_dir / "best_model.pth"
    if bp.exists():
        m = _load_resnet(bp, cfg, cpu)
        results.append({
            "name": "ResNet50 baseline",
            "auroc": _eval_auroc(m, test_loader, cpu),
            "latency": benchmark_latency(m, n_warmup=n_warmup, n_runs=n_runs, device=cpu),
            "size_mb": get_model_size_mb(m),
            "flops": count_flops(m, device=cpu),
        })

    # ── Structured pruned ─────────────────────────────────────────────────
    sp = ckpt_dir / "pruned_structured.pth"
    if sp.exists():
        m = _load_resnet(sp, cfg, cpu)
        results.append({
            "name": "Structured pruned",
            "auroc": _eval_auroc(m, test_loader, cpu),
            "latency": benchmark_latency(m, n_warmup=n_warmup, n_runs=n_runs, device=cpu),
            "size_mb": get_model_size_mb(m),
            "flops": count_flops(m, device=cpu),
        })

    # ── Student distilled ─────────────────────────────────────────────────
    dp = ckpt_dir / "student_distilled.pth"
    if dp.exists():
        m = _load_student(dp, cpu)
        results.append({
            "name": "MobileNetV3 student",
            "auroc": _eval_auroc(m, test_loader, cpu),
            "latency": benchmark_latency(m, n_warmup=n_warmup, n_runs=n_runs, device=cpu),
            "size_mb": get_model_size_mb(m),
            "flops": count_flops(m, device=cpu),
        })

    # ── MLflow ────────────────────────────────────────────────────────────
    tracking_uri = _resolve_uri(cfg.mlflow.tracking_uri, project_root)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(run_name="full_benchmark"):
        for r in results:
            prefix = r["name"].lower().replace(" ", "_")
            mlflow.log_metrics({
                f"{prefix}_auroc": r["auroc"],
                f"{prefix}_p50_ms": r["latency"]["p50_ms"],
                f"{prefix}_p95_ms": r["latency"]["p95_ms"],
                f"{prefix}_p99_ms": r["latency"]["p99_ms"],
                f"{prefix}_fps": r["latency"]["fps"],
                f"{prefix}_size_mb": r["size_mb"],
                f"{prefix}_flops": r["flops"],
            })

    # ── Final table ───────────────────────────────────────────────────────
    baseline_p50 = results[0]["latency"]["p50_ms"] if results else 1.0
    logger.info("")
    logger.info("=" * 85)
    logger.info("FULL PIPELINE BENCHMARK")
    logger.info("=" * 85)
    logger.info(
        "%-22s  %-8s  %-10s  %-10s  %-10s  %-12s  %-8s",
        "Model", "AUROC", "P50 (ms)", "P95 (ms)", "P99 (ms)", "Size (MB)", "FLOPs",
    )
    logger.info("-" * 85)
    for r in results:
        logger.info(
            "%-22s  %-8.4f  %-10.1f  %-10.1f  %-10.1f  %-12.1f  %-8s",
            r["name"],
            r["auroc"],
            r["latency"]["p50_ms"],
            r["latency"]["p95_ms"],
            r["latency"]["p99_ms"],
            r["size_mb"],
            format_flops(r["flops"]),
        )
    logger.info("-" * 85)

    if len(results) >= 2:
        first, last = results[0], results[-1]
        logger.info(
            "Compression: %.1fx smaller | %.1fx faster | AUROC delta: %+.4f",
            first["size_mb"] / max(last["size_mb"], 0.01),
            first["latency"]["p50_ms"] / max(last["latency"]["p50_ms"], 0.01),
            last["auroc"] - first["auroc"],
        )
    logger.info("=" * 85)


if __name__ == "__main__":
    main()