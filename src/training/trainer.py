"""Training orchestrator for the anomaly detector.

Keeps the train script thin — it creates the Trainer and calls fit().
All the epoch logic, LR scheduling, checkpointing, and MLflow logging
live here so they're testable and reusable across modules.
"""
import logging
from pathlib import Path
from typing import Optional

import mlflow
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluation.metrics import compute_auroc

logger = logging.getLogger("edgevision")


class EarlyStopping:
    """Stop training when val AUROC stops improving.

    Args:
        patience: Epochs to wait without improvement before stopping.
        min_delta: Minimum gain to count as progress.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4) -> None:
        self.patience  = patience
        self.min_delta = min_delta
        self._best     = float("-inf")
        self._counter  = 0

    def step(self, metric: float) -> bool:
        """Update and return True if training should stop.

        Args:
            metric: Current val metric (higher = better).
        Returns:
            True if patience exceeded.
        """
        if metric > self._best + self.min_delta:
            self._best    = metric
            self._counter = 0
            return False
        self._counter += 1
        logger.debug("EarlyStopping: %d / %d", self._counter, self.patience)
        return self._counter >= self.patience


class Trainer:
    """Manages the full training loop for anomaly detection models.

    Args:
        model: Model to train.
        cfg: Full Hydra config.
        device: Training device.
        checkpoint_dir: Where to save checkpoints. Overrides cfg path —
            necessary because Hydra changes the working directory.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: DictConfig,
        device: torch.device,
        checkpoint_dir: Optional[Path] = None,
    ) -> None:
        self.model  = model.to(device)
        self.cfg    = cfg
        self.device = device

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = Adam(
            model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )
        # cosine annealing — smooth LR decay, no manual step schedule needed
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.training.epochs,
            eta_min=1e-6,
        )
        self.early_stopping = EarlyStopping(
            patience=cfg.training.early_stopping_patience,
        )
        self.checkpoint_dir = (
            checkpoint_dir if checkpoint_dir is not None
            else Path(cfg.paths.checkpoint_dir)
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._best_auroc = 0.0

    def _train_epoch(self, loader: DataLoader) -> float:
        """One forward+backward pass over the training set.

        Returns:
            Mean training loss for the epoch.
        """
        self.model.train()
        total = 0.0
        for batch in tqdm(loader, desc="  train", leave=False):
            images = batch["image"].to(self.device)
            labels = batch["label"].float().to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images).squeeze(1)
            loss   = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()
            total += loss.item()

        return total / len(loader)

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader) -> tuple[float, float]:
        """Validation pass — loss and image-level AUROC.

        Returns:
            Tuple of (mean_val_loss, image_auroc).
        """
        self.model.eval()
        total   = 0.0
        scores: list[float] = []
        labels: list[int]   = []

        for batch in tqdm(loader, desc="  val  ", leave=False):
            images = batch["image"].to(self.device)
            lbls   = batch["label"]
            logits = self.model(images).squeeze(1)
            loss   = self.criterion(logits, lbls.float().to(self.device))
            total += loss.item()
            scores.extend(torch.sigmoid(logits).cpu().tolist())
            labels.extend(lbls.tolist())

        return total / len(loader), compute_auroc(labels, scores)

    def _save_checkpoint(self, epoch: int, auroc: float) -> Path:
        """Save the best model so far.

        Returns:
            Path to the saved checkpoint.
        """
        path = self.checkpoint_dir / "best_model.pth"
        torch.save({
            "epoch":                epoch,
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_auroc":            auroc,
        }, path)
        logger.info("Checkpoint saved -> %s  (AUROC=%.4f)", path, auroc)
        return path

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> dict[str, float]:
        """Run the full training loop.

        Args:
            train_loader: CutPaste-augmented training loader.
            val_loader: Real MVTec val loader (used for AUROC tracking).
        Returns:
            Dict with 'best_val_auroc' and 'epochs_trained'.
        """
        logger.info("Training for up to %d epochs.", self.cfg.training.epochs)
        epoch = 0
        for epoch in range(1, self.cfg.training.epochs + 1):
            train_loss          = self._train_epoch(train_loader)
            val_loss, val_auroc = self._val_epoch(val_loader)
            self.scheduler.step()
            lr = self.scheduler.get_last_lr()[0]

            logger.info(
                "Epoch %3d | train=%.4f val=%.4f auroc=%.4f lr=%.2e",
                epoch, train_loss, val_loss, val_auroc, lr,
            )
            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss,
                 "val_auroc": val_auroc, "lr": lr},
                step=epoch,
            )

            if val_auroc > self._best_auroc:
                self._best_auroc = val_auroc
                self._save_checkpoint(epoch, val_auroc)

            if self.early_stopping.step(val_auroc):
                logger.info("Early stopping triggered at epoch %d.", epoch)
                break

        logger.info("Training complete. Best val AUROC: %.4f", self._best_auroc)
        return {"best_val_auroc": self._best_auroc, "epochs_trained": epoch}
