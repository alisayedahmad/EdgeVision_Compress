"""PTQ and QAT quantization pipeline with CPU latency benchmark.

Usage:
    python scripts/quantize.py "mlflow.tracking_uri=sqlite:///mlflow.db"
"""
import copy
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
from compression.pruning.structured import (
    apply_structured_pruning,
    get_bottleneck_locations,
)
from compression.quantization.ptq import calibrate, convert_ptq, prepare_ptq
from compression.quantization.qat import (
    QATFinetuner,
    convert_qat,
    export_onnx,
    prepare_qat,
)
from data.cutpaste import CutPasteDataset, CutPasteTransform
from data.mvtec import MVTecDataset
from evaluation.metrics import compute_auroc
from models.resnet import ResNet50AnomalyDetector
from utils.logging_config import setup_logging
from utils.seed import seed_everything

logger = setup_logging(name="edgevision.quantize")


def _resolve_tracking_uri(raw_uri: str, project_root: Path) -> str:
    """Resolve MLflow tracking URI for Windows sqlite paths."""
    if raw_uri.startswith("sqlite:///"):
        db_path = project_root / raw_uri.replace("sqlite:///", "", 1)
        return f"sqlite:///{db_path}"
    return str(project_root / raw_uri)


def _load_pruned_model(
    ckpt_path: Path,
    cfg: DictConfig,
    device: torch.device,
) -> nn.Module:
    """Load a checkpoint, rebuilding pruned architecture if needed.

    Reads the state_dict conv2 weight shapes to detect which blocks
    were structurally pruned, then rebuilds the model to match.

    Args:
        ckpt_path: Path to checkpoint file.
        cfg: Hydra config with model settings.
        device: Device to load onto.

    Returns:
        Model with correctly loaded weights.
    """
    model = ResNet50AnomalyDetector(
        pretrained=False,
        hidden_dim=cfg.model.hidden_dim,
        dropout=cfg.model.dropout,
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"]

    # detect structured pruning by comparing conv2 shapes
    locations = get_bottleneck_locations(model)
    keep_indices = {}
    needs_rebuild = False

    for i, (parent, block_idx, block) in enumerate(locations):
        # find parent's index in feature_extractor
        parent_idx = None
        for pi, child in enumerate(model.feature_extractor):
            if child is parent:
                parent_idx = pi
                break

        key = f"feature_extractor.{parent_idx}.{block_idx}.conv2.weight"
        if key in state_dict:
            saved_ch = state_dict[key].shape[0]
            original_ch = block.conv2.out_channels
            if saved_ch < original_ch:
                needs_rebuild = True
                keep_indices[f"block_{i}"] = torch.arange(saved_ch)

    if needs_rebuild:
        logger.info("Rebuilding architecture for %d pruned blocks...", len(keep_indices))
        apply_structured_pruning(model, keep_indices)

    model.load_state_dict(state_dict)
    logger.info("Checkpoint loaded from %s", ckpt_path)
    return model


@torch.no_grad()
def _evaluate_auroc(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Compute image-level AUROC on a DataLoader."""
    model.eval()
    scores, labels = [], []
    for batch in loader:
        images = batch["image"].to(device)
        logits = model(images).squeeze(1)
        scores.extend(torch.sigmoid(logits).cpu().tolist())
        labels.extend(batch["label"].tolist())
    return compute_auroc(labels, scores)


@hydra.main(version_base=None, config_path="../configs", config_name="quantize")
def main(cfg: DictConfig) -> None:
    """Run PTQ and QAT pipelines with CPU latency benchmark."""
    seed_everything(cfg.seed)
    project_root = Path(get_original_cwd())
    cpu = torch.device("cpu")

    logger.info("GPU available: %s", torch.cuda.is_available())

    # ── load checkpoint ───────────────────────────────────────────────────
    ckpt_path = project_root / cfg.paths.checkpoint_dir / "pruned_structured.pth"
    if not ckpt_path.exists():
        ckpt_path = project_root / cfg.paths.checkpoint_dir / "best_model.pth"
        logger.warning("Structured checkpoint not found, using baseline.")

    model_fp32 = _load_pruned_model(ckpt_path, cfg, cpu)

    # ── dataloaders ───────────────────────────────────────────────────────
    data_dir = project_root / cfg.paths.data_dir
    nw = cfg.training.num_workers if platform.system() != "Windows" else 0

    train_base = MVTecDataset(
        data_dir, cfg.data.category, "train",
        seed=cfg.seed, val_fraction=cfg.data.val_fraction,
    )
    train_ds = CutPasteDataset(train_base, CutPasteTransform(p=0.5))
    val_ds = MVTecDataset(
        data_dir, cfg.data.category, "val",
        seed=cfg.seed, val_fraction=cfg.data.val_fraction,
    )
    test_ds = MVTecDataset(
        data_dir, cfg.data.category, "test",
        seed=cfg.seed, val_fraction=cfg.data.val_fraction,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size,
        shuffle=True, num_workers=nw, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size,
        shuffle=False, num_workers=nw,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.training.batch_size,
        shuffle=False, num_workers=nw,
    )

    quant_cfg = cfg.compression.quantization

    # ── FP32 baseline latency ─────────────────────────────────────────────
    logger.info("Benchmarking FP32 on CPU...")
    fp32_latency = benchmark_latency(
        copy.deepcopy(model_fp32), n_warmup=quant_cfg.n_warmup,
        n_runs=quant_cfg.n_benchmark_runs, device=cpu,
    )
    fp32_size = get_model_size_mb(model_fp32)
    fp32_auroc = _evaluate_auroc(model_fp32, test_loader, cpu)
    logger.info("FP32 | size=%.1f MB | auroc=%.4f", fp32_size, fp32_auroc)

    # ── PTQ ───────────────────────────────────────────────────────────────
    logger.info("Running PTQ...")
    model_ptq = copy.deepcopy(model_fp32).cpu()
    model_ptq = prepare_ptq(model_ptq)
    calibrate(model_ptq, train_loader, cpu, n_batches=quant_cfg.ptq_calibration_batches)
    model_ptq = convert_ptq(model_ptq)

    ptq_latency = benchmark_latency(
        model_ptq, n_warmup=quant_cfg.n_warmup,
        n_runs=quant_cfg.n_benchmark_runs, device=cpu,
    )
    ptq_size = get_model_size_mb(model_ptq)
    ptq_auroc = _evaluate_auroc(model_ptq, test_loader, cpu)
    logger.info("PTQ  | size=%.1f MB | auroc=%.4f", ptq_size, ptq_auroc)

    # ── QAT ───────────────────────────────────────────────────────────────
    logger.info("Running QAT for %d epochs...", quant_cfg.qat_epochs)
    model_qat = copy.deepcopy(model_fp32).cpu()
    model_qat = prepare_qat(model_qat)

    finetuner = QATFinetuner(
        model=model_qat, device=cpu,
        lr=quant_cfg.qat_lr, n_epochs=quant_cfg.qat_epochs,
    )
    best_val_auroc = finetuner.run(train_loader, val_loader)
    model_qat = convert_qat(model_qat)

    qat_latency = benchmark_latency(
        model_qat, n_warmup=quant_cfg.n_warmup,
        n_runs=quant_cfg.n_benchmark_runs, device=cpu,
    )
    qat_size = get_model_size_mb(model_qat)
    qat_auroc = _evaluate_auroc(model_qat, test_loader, cpu)
    logger.info("QAT  | size=%.1f MB | auroc=%.4f", qat_size, qat_auroc)

    # ── ONNX export ───────────────────────────────────────────────────────
    onnx_path = project_root / cfg.paths.output_dir / "model_qat_int8.onnx"
    export_onnx(model_qat, onnx_path)

    # ── save checkpoint ───────────────────────────────────────────────────
    ckpt_out = project_root / cfg.paths.checkpoint_dir / "quantized_qat.pth"
    torch.save({
        "model_state_dict": model_qat.state_dict(),
        "test_auroc": qat_auroc,
        "fp32_p50_ms": fp32_latency["p50_ms"],
        "qat_p50_ms": qat_latency["p50_ms"],
    }, ckpt_out)

    # ── MLflow ────────────────────────────────────────────────────────────
    tracking_uri = _resolve_tracking_uri(cfg.mlflow.tracking_uri, project_root)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    run_name = cfg.mlflow.run_name or f"quantize_{cfg.data.category}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "category": cfg.data.category,
            "ptq_calibration_batches": quant_cfg.ptq_calibration_batches,
            "qat_epochs": quant_cfg.qat_epochs,
            "qat_lr": quant_cfg.qat_lr,
            "seed": cfg.seed,
        })
        mlflow.log_metrics({
            "fp32_auroc": fp32_auroc, "fp32_p50_ms": fp32_latency["p50_ms"], "fp32_size_mb": fp32_size,
            "ptq_auroc": ptq_auroc, "ptq_p50_ms": ptq_latency["p50_ms"], "ptq_size_mb": ptq_size,
            "qat_auroc": qat_auroc, "qat_p50_ms": qat_latency["p50_ms"], "qat_size_mb": qat_size,
            "qat_speedup": fp32_latency["p50_ms"] / qat_latency["p50_ms"],
        })
        mlflow.log_artifact(str(ckpt_out))
        mlflow.log_artifact(str(onnx_path))

    # ── summary ───────────────────────────────────────────────────────────
    logger.info("-" * 62)
    logger.info("%-8s  %-10s  %-10s  %-10s  %-8s", "Model", "AUROC", "P50 (ms)", "Size (MB)", "Speedup")
    logger.info("-" * 62)
    logger.info("%-8s  %-10.4f  %-10.1f  %-10.1f  %-8s", "FP32", fp32_auroc, fp32_latency["p50_ms"], fp32_size, "1.0x")
    logger.info("%-8s  %-10.4f  %-10.1f  %-10.1f  %-8.2f", "PTQ", ptq_auroc, ptq_latency["p50_ms"], ptq_size, fp32_latency["p50_ms"]/ptq_latency["p50_ms"])
    logger.info("%-8s  %-10.4f  %-10.1f  %-10.1f  %-8.2f", "QAT", qat_auroc, qat_latency["p50_ms"], qat_size, fp32_latency["p50_ms"]/qat_latency["p50_ms"])
    logger.info("-" * 62)


if __name__ == "__main__":
    main()