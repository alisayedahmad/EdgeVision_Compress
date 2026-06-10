"""Knowledge distillation training loop."""
import logging

import mlflow
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from compression.distillation.loss import DistillationLoss
from evaluation.metrics import compute_auroc

logger = logging.getLogger("edgevision")


class DistillationTrainer:
    """Trains a student model to mimic a frozen teacher.

    Args:
        teacher: Frozen teacher model (ResNet50).
        student: Student model to train (MobileNetV3).
        feature_projector: Linear layer to align feature dimensions
            (student 576 -> teacher 2048).
        cfg: Distillation config section.
        device: Training device.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        feature_projector: nn.Linear,
        cfg,
        device: torch.device,
    ) -> None:
        self.teacher = teacher.to(device).eval()
        self.student = student.to(device)
        self.projector = feature_projector.to(device)
        self.device = device
        self.cfg = cfg

        # freeze teacher completely
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.criterion = DistillationLoss(
            alpha=cfg.alpha,
            beta=cfg.beta,
            gamma=cfg.gamma,
            temperature=cfg.temperature,
        )
        # optimize student + projector together
        self.optimizer = Adam(
            list(self.student.parameters()) + list(self.projector.parameters()),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.01,
        )

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> dict[str, float]:
        """Full training loop. Returns best val AUROC and epochs trained."""
        best_auroc = 0.0
        best_state = None

        for epoch in range(1, self.cfg.epochs + 1):
            losses = self._train_epoch(train_loader)
            val_auroc = self._val_auroc(val_loader)
            self.scheduler.step()

            logger.info(
                "Epoch %2d/%d | total=%.4f task=%.4f kl=%.4f feat=%.4f | val_auroc=%.4f",
                epoch, self.cfg.epochs,
                losses["total"], losses["task"], losses["kl"], losses["feature"],
                val_auroc,
            )
            mlflow.log_metrics({
                "distill_total_loss": losses["total"],
                "distill_task_loss": losses["task"],
                "distill_kl_loss": losses["kl"],
                "distill_feat_loss": losses["feature"],
                "distill_val_auroc": val_auroc,
            }, step=epoch)

            if val_auroc > best_auroc:
                best_auroc = val_auroc
                best_state = {
                    "student": self.student.state_dict(),
                    "projector": self.projector.state_dict(),
                }

        # restore best
        if best_state:
            self.student.load_state_dict(best_state["student"])

        return {"best_val_auroc": best_auroc, "epochs_trained": self.cfg.epochs}

    def _train_epoch(self, loader: DataLoader) -> dict[str, float]:
        """One training epoch. Returns averaged losses."""
        self.student.train()
        self.projector.train()
        sums = {"total": 0.0, "task": 0.0, "kl": 0.0, "feature": 0.0}

        for batch in tqdm(loader, desc="  distill", leave=False):
            images = batch["image"].to(self.device)
            labels = batch["label"].float().to(self.device)

            # teacher forward (no grad)
            with torch.no_grad():
                t_logits = self.teacher(images)
                t_features = self.teacher.get_features(images)

            # student forward
            s_logits = self.student(images)
            s_features = self.student.get_feature_vector(images)
            s_projected = self.projector(s_features)  # 576 -> 2048

            losses = self.criterion(s_logits, t_logits, labels, s_projected, t_features)

            self.optimizer.zero_grad()
            losses["total"].backward()
            self.optimizer.step()

            for k in sums:
                sums[k] += losses[k].item()

        n = len(loader)
        return {k: v / n for k, v in sums.items()}

    @torch.no_grad()
    def _val_auroc(self, loader: DataLoader) -> float:
        """Student AUROC on val set."""
        self.student.eval()
        scores, labels = [], []
        for batch in loader:
            images = batch["image"].to(self.device)
            logits = self.student(images).squeeze(1)
            scores.extend(torch.sigmoid(logits).cpu().tolist())
            labels.extend(batch["label"].tolist())
        return compute_auroc(labels, scores)