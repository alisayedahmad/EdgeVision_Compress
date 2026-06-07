"""Fine-tuning loop after each pruning step."""
import logging

import mlflow
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluation.metrics import compute_auroc

logger = logging.getLogger("edgevision")


class PruningFinetuner:
    """Fine-tunes a pruned model for a fixed number of epochs.


    Args:
        model: Pruned model to fine-tune (already on device).
        device: Training device.
        lr: Fine-tuning learning rate. Should be 10–100× lower than
            the original training LR to avoid overwriting learned features.
        n_epochs: Number of fine-tuning epochs per pruning iteration.
        mlflow_prefix: Metric key prefix for MLflow (e.g., "iter_1").
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        lr: float = 1e-5,
        n_epochs: int = 5,
        mlflow_prefix: str = "",
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.n_epochs = n_epochs
        self.mlflow_prefix = mlflow_prefix
        self.criterion = nn.BCEWithLogitsLoss()
        # fresh optimizer — sees weight_orig parameters in pruned modules
        self.optimizer = Adam(model.parameters(), lr=lr)
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=n_epochs,
            eta_min=lr * 0.01,
        )

    def run(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        step_offset: int = 0,
    ) -> float:
        """Run fine-tuning and return the best val AUROC achieved.

        Args:
            train_loader: CutPaste-augmented training loader.
            val_loader: Clean val loader for AUROC tracking.
            step_offset: Global step offset so MLflow x-axis is continuous
                across all iterations.

        Returns:
            Best val AUROC observed during this fine-tuning phase.
        """
        best_auroc = 0.0
        prefix = f"{self.mlflow_prefix}_" if self.mlflow_prefix else ""

        for epoch in range(1, self.n_epochs + 1):
            train_loss = self._train_epoch(train_loader)
            val_auroc = self._val_auroc(val_loader)
            self.scheduler.step()

            mlflow.log_metrics(
                {
                    f"{prefix}ft_train_loss": train_loss,
                    f"{prefix}ft_val_auroc": val_auroc,
                },
                step=step_offset + epoch,
            )
            logger.info(
                "  FT %2d/%d | train_loss=%.4f  val_auroc=%.4f",
                epoch,
                self.n_epochs,
                train_loss,
                val_auroc,
            )
            if val_auroc > best_auroc:
                best_auroc = val_auroc

        return best_auroc

    def _train_epoch(self, loader: DataLoader) -> float:
        """One pass over the training set.

        Args:
            loader: Training DataLoader.

        Returns:
            Mean loss over all batches.
        """
        self.model.train()
        total = 0.0
        for batch in tqdm(loader, desc="    ft", leave=False):
            images = batch["image"].to(self.device)
            labels = batch["label"].float().to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(images).squeeze(1)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()
            total += loss.item()
        return total / len(loader)

    @torch.no_grad()
    def _val_auroc(self, loader: DataLoader) -> float:
        """Compute image-level AUROC on the validation set.

        Args:
            loader: Validation DataLoader.

        Returns:
            AUROC in [0.0, 1.0].
        """
        self.model.eval()
        scores: list[float] = []
        labels: list[int] = []
        for batch in loader:
            images = batch["image"].to(self.device)
            logits = self.model(images).squeeze(1)
            scores.extend(torch.sigmoid(logits).cpu().tolist())
            labels.extend(batch["label"].tolist())
        return compute_auroc(labels, scores)