"""Train the ResNet50 anomaly detection baseline.

Usage (from project root, .venv active):
    python scripts/train_baseline.py
    python scripts/train_baseline.py training.epochs=5
    python scripts/train_baseline.py data.category=capsule
    python scripts/train_baseline.py training.batch_size=16

Note: first run downloads ResNet50 ImageNet weights (~100 MB).
"""
import logging
import platform
import sys
from pathlib import Path

import hydra
import mlflow
import numpy as np
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.cutpaste import CutPasteDataset, CutPasteTransform
from data.mvtec import MVTecDataset
from evaluation.metrics import compute_auroc, compute_pixel_auroc
from models.resnet import ResNet50AnomalyDetector
from training.trainer import Trainer
from utils.logging_config import setup_logging
from utils.seed import seed_everything

logger = setup_logging(name="edgevision.train")


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """Train ResNet50 baseline and log everything to MLflow.

    Args:
        cfg: Hydra config from configs/train.yaml.
    """
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s | category: %s", device, cfg.data.category)

    # Hydra changes cwd to outputs/<date>/<time>/ — resolve paths from root
    project_root   = Path(get_original_cwd())
    data_dir       = project_root / cfg.paths.data_dir
    checkpoint_dir = project_root / cfg.paths.checkpoint_dir

    # ── datasets ──────────────────────────────────────────────────────────────
    train_base = MVTecDataset(
        data_dir, cfg.data.category, "train",
        seed=cfg.seed, val_fraction=cfg.data.val_fraction,
    )
    # wrap with CutPaste — creates synthetic anomalies on-the-fly
    train_ds = CutPasteDataset(train_base, CutPasteTransform(p=0.5))

    val_ds  = MVTecDataset(data_dir, cfg.data.category, "val",
                           seed=cfg.seed, val_fraction=cfg.data.val_fraction)
    test_ds = MVTecDataset(data_dir, cfg.data.category, "test",
                           seed=cfg.seed, val_fraction=cfg.data.val_fraction)

    # Windows: num_workers must be 0 (spawn limitation — see Module 1)
    nw  = cfg.training.num_workers if platform.system() != "Windows" else 0
    pin = torch.cuda.is_available()

    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size,
                              shuffle=True,  num_workers=nw, pin_memory=pin,
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.training.batch_size,
                              shuffle=False, num_workers=nw, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.training.batch_size,
                              shuffle=False, num_workers=nw, pin_memory=pin)

    logger.info(
        "Data ready | train=%d  val=%d  test=%d",
        len(train_ds), len(val_ds), len(test_ds),
    )

    # ── model ─────────────────────────────────────────────────────────────────
    model = ResNet50AnomalyDetector(
        pretrained=cfg.model.pretrained,
        hidden_dim=cfg.model.hidden_dim,
        dropout=cfg.model.dropout,
        freeze_backbone=cfg.model.freeze_backbone,
    )

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri((project_root / cfg.mlflow.tracking_uri).as_uri())
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    run_name = cfg.mlflow.run_name or f"baseline_{cfg.data.category}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "category":        cfg.data.category,
            "model":           cfg.model.name,
            "pretrained":      cfg.model.pretrained,
            "freeze_backbone": cfg.model.freeze_backbone,
            "epochs":          cfg.training.epochs,
            "batch_size":      cfg.training.batch_size,
            "lr":              cfg.training.learning_rate,
            "weight_decay":    cfg.training.weight_decay,
            "seed":            cfg.seed,
        })

        # ── train ──────────────────────────────────────────────────────────
        trainer = Trainer(model, cfg, device, checkpoint_dir=checkpoint_dir)
        trainer.fit(train_loader, val_loader)

        # ── final test evaluation ──────────────────────────────────────────
        ckpt_path = checkpoint_dir / "best_model.pth"
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        all_scores: list[float] = []
        all_labels: list[int]   = []
        all_maps:   list        = []
        all_masks:  list        = []

        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(device)
                logits = model(images).squeeze(1)

                all_scores.extend(torch.sigmoid(logits).cpu().tolist())
                all_labels.extend(batch["label"].tolist())

                # pixel-level: feature-map norm, upsampled to 224×224
                amap = model.get_anomaly_map(images, image_size=224)
                all_maps.append(amap.cpu().numpy())
                all_masks.append(batch["mask"].squeeze(1).numpy())

        test_auroc  = compute_auroc(all_labels, all_scores)
        maps_np     = np.concatenate(all_maps,  axis=0)
        masks_np    = np.concatenate(all_masks, axis=0)
        pixel_auroc = compute_pixel_auroc(masks_np, maps_np)

        logger.info(
            "Test | image_auroc=%.4f  pixel_auroc=%.4f",
            test_auroc, pixel_auroc,
        )
        mlflow.log_metrics({
            "test_image_auroc": test_auroc,
            "test_pixel_auroc": pixel_auroc,
        })
        mlflow.log_artifact(str(ckpt_path))


if __name__ == "__main__":
    main()
