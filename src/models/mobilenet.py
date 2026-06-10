"""MobileNetV3-Small student model for knowledge distillation."""
import logging

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

logger = logging.getLogger("edgevision")


class MobileNetV3Student(nn.Module):
    """MobileNetV3-Small with anomaly detection head.

    Much smaller than ResNet50 (2.5M vs 25M params) — the whole point
    of distillation is to transfer the teacher's knowledge into this.
    """

    def __init__(
        self,
        pretrained: bool = True,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)

        # everything up to the final classifier
        self.features = backbone.features       # -> [B, 576, 7, 7]
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.Linear(576, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )
        logger.info(
            "MobileNetV3Student | pretrained=%s | %.1fM params",
            pretrained,
            sum(p.numel() for p in self.parameters()) / 1e6,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logit [B, 1]."""
        feat = self.features(x)
        pooled = self.pool(feat).flatten(1)
        return self.head(pooled)

    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Inference-time probability [B]."""
        return torch.sigmoid(self.forward(x)).squeeze(1)

    def get_feature_vector(self, x: torch.Tensor) -> torch.Tensor:
        """Post-GAP feature vector [B, 576] — used for feature matching loss."""
        feat = self.features(x)
        return self.pool(feat).flatten(1)