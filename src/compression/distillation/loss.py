"""Combined distillation loss: task + KL divergence + feature matching."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """Three-term loss for knowledge distillation.

    Args:
        alpha: Weight for the task loss (BCE).
        beta: Weight for the KL divergence on soft labels.
        gamma: Weight for the feature matching MSE.
        temperature: Softens teacher probabilities. Higher = softer.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.5,
        gamma: float = 0.2,
        temperature: float = 4.0,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute combined loss.

        Args:
            student_logits: Student raw output [B, 1].
            teacher_logits: Teacher raw output [B, 1].
            labels: Ground truth [B].
            student_features: Student feature vector [B, D_s].
            teacher_features: Teacher feature vector [B, D_t].

        Returns:
            Dict with 'total', 'task', 'kl', 'feature' losses.
        """
        s = student_logits.squeeze(1)
        t = teacher_logits.squeeze(1)

        # 1) task loss — student vs hard labels
        task_loss = self.bce(s, labels)

        # 2) KL divergence — student vs teacher soft predictions
        T = self.temperature
        s_soft = torch.sigmoid(s / T)
        t_soft = torch.sigmoid(t / T)
        # binary KL: we treat each output as independent Bernoulli
        kl_loss = F.binary_cross_entropy(s_soft, t_soft.detach()) * (T * T)

        # 3) feature matching — MSE between projected feature vectors
        # teacher features are detached — we only train the student
        feat_loss = F.mse_loss(student_features, teacher_features.detach())

        total = self.alpha * task_loss + self.beta * kl_loss + self.gamma * feat_loss

        return {
            "total": total,
            "task": task_loss.detach(),
            "kl": kl_loss.detach(),
            "feature": feat_loss.detach(),
        }