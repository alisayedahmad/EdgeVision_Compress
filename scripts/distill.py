"""Knowledge distillation: ResNet50 teacher -> MobileNetV3-Small student.

Usage:
    python scripts/distill.py "mlflow.tracking_uri=sqlite:///mlflow.db"
    python scripts/distill.py distillation.epochs=10 "mlflow.tracking_uri=sqlite:///mlflow.db"
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
from compression.distillation.trainer import DistillationTrainer
from compression.pruning.structured import (
    apply_structured_pruning,
    get_bottleneck_locations,
)
from data.cutpaste import CutPasteDataset, CutPasteTransform
from data.mvtec import MVTecDataset
from evaluation.metrics import compute_auroc, compute_pixel_auroc
from models.mobilenet import MobileNetV3Student
from models.resnet import ResNet50AnomalyDetector
from utils.logging_config import setup_logging
from utils.seed import seed_everything

logger = setup_logging(name="edgevision.distill")


def _resolve_tracking_uri(raw_uri: str, project_root: Path) -> str:
    if raw_uri.startswith("sqlite:///"):
        return f"sqlite:///{project_root / raw_uri.replace('sqlite:///', '', 1)}"
    return str(project_root / raw_uri)


def _load_teacher(ckpt_path: Path, cfg: DictConfig, device: torch.device) -> nn.Module:
    """Load teacher, rebuilding pruned architecture if needed."""
    model = ResNet50AnomalyDetector(
        pretrained=False,
        hidden_dim=cfg.model.hidden_dim,
        dropout=cfg.model.dropout,
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"]

    locations = get_bottleneck_locations(model)
    keep_indices = {}
    for i, (parent, block_idx, block) in enumerate(locations):
        parent_idx = None
        for pi, child in enumerate(model.feature_extractor):
            if child is parent:
                parent_idx = pi
                break
        key = f"feature_extractor.{parent_idx}.{block_idx}.conv2.weight"
        if key in state_dict:
            saved_ch = state_dict[key].shape[0]
            if saved_ch < block.conv2.out_channels:
                keep_indices[f"block_{i}"] = torch.arange(saved_ch)

    if keep_indices:
        apply_structured_pruning(model, keep_indices)

    model.load_state_dict(state_dict)
    return model


@hydra.main(version_base=None, config_path="../configs", config_name="distill")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = Path(get_original_cwd())
    logger.info("Device: %s", device)

    # ── teacher ───────────────────────────────────────────────────────────
    teacher_path = project_root / cfg.paths.checkpoint_dir / "pruned_structured.pth"
    if not teacher_path.exists():
        teacher_path = project_root / cfg.paths.checkpoint_dir / "best_model.pth"
    teacher = _load_teacher(teacher_path, cfg, device)
    teacher.eval()
    logger.info("Teacher loaded from %s", teacher_path)

    # ── student ───────────────────────────────────────────────────────────
    dist_cfg = cfg.distillation
    student = MobileNetV3Student(
        pretrained=dist_cfg.student_pretrained,
        hidden_dim=128,
        dropout=0.3,
    )

    # projector aligns student features (576) to teacher features (2048)
    projector = nn.Linear(576, 2048)

    # ── data ──────────────────────────────────────────────────────────────
    data_dir = project_root / cfg.paths.data_dir
    nw = cfg.training.num_workers if platform.system() != "Windows" else 0
    pin = torch.cuda.is_available()

    train_base = MVTecDataset(data_dir, cfg.data.category, "train",
                              seed=cfg.seed, val_fraction=cfg.data.val_fraction)
    train_ds = CutPasteDataset(train_base, CutPasteTransform(p=0.5))
    val_ds = MVTecDataset(data_dir, cfg.data.category, "val",
                          seed=cfg.seed, val_fraction=cfg.data.val_fraction)
    test_ds = MVTecDataset(data_dir, cfg.data.category, "test",
                           seed=cfg.seed, val_fraction=cfg.data.val_fraction)

    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size,
                              shuffle=True, num_workers=nw, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size,
                            shuffle=False, num_workers=nw, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=cfg.training.batch_size,
                             shuffle=False, num_workers=nw, pin_memory=pin)

    # ── MLflow ────────────────────────────────────────────────────────────
    tracking_uri = _resolve_tracking_uri(cfg.mlflow.tracking_uri, project_root)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    run_name = cfg.mlflow.run_name or f"distill_{cfg.data.category}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "category": cfg.data.category,
            "epochs": dist_cfg.epochs,
            "lr": dist_cfg.lr,
            "alpha": dist_cfg.alpha,
            "beta": dist_cfg.beta,
            "gamma": dist_cfg.gamma,
            "temperature": dist_cfg.temperature,
            "seed": cfg.seed,
        })

        # ── train ─────────────────────────────────────────────────────
        trainer = DistillationTrainer(
            teacher=teacher, student=student,
            feature_projector=projector, cfg=dist_cfg, device=device,
        )
        result = trainer.fit(train_loader, val_loader)
        logger.info("Best val AUROC: %.4f", result["best_val_auroc"])

        # ── test ──────────────────────────────────────────────────────
        student.eval()
        all_scores, all_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(device)
                logits = student(images).squeeze(1)
                all_scores.extend(torch.sigmoid(logits).cpu().tolist())
                all_labels.extend(batch["label"].tolist())

        test_auroc = compute_auroc(all_labels, all_scores)

        # ── benchmark on CPU ──────────────────────────────────────────
        student_cpu = student.cpu()
        latency = benchmark_latency(student_cpu, n_warmup=10, n_runs=100)
        size_mb = get_model_size_mb(student_cpu)

        logger.info("Test AUROC: %.4f | P50: %.1f ms | Size: %.1f MB",
                     test_auroc, latency["p50_ms"], size_mb)

        mlflow.log_metrics({
            "test_auroc": test_auroc,
            "student_p50_ms": latency["p50_ms"],
            "student_size_mb": size_mb,
        })

        # ── save ──────────────────────────────────────────────────────
        ckpt_out = project_root / cfg.paths.checkpoint_dir / "student_distilled.pth"
        torch.save({
            "model_state_dict": student.state_dict(),
            "test_auroc": test_auroc,
            "p50_ms": latency["p50_ms"],
            "size_mb": size_mb,
        }, ckpt_out)
        mlflow.log_artifact(str(ckpt_out))
        logger.info("Student saved -> %s", ckpt_out)


if __name__ == "__main__":
    main()