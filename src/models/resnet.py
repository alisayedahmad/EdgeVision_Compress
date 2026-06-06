"""ResNet50-based anomaly detector.

We strip the classification head and replace it with a small MLP that
outputs a single anomaly score. Raw logit out — sigmoid is applied only
at inference time. BCEWithLogitsLoss during training (numerically stabler
than BCE + sigmoid combo).

The backbone is split before GlobalAvgPool so we can extract spatial
feature maps for pixel-level anomaly heatmaps without adding any hooks.
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50

logger = logging.getLogger("edgevision")


class AnomalyHead(nn.Module):
    """Two-layer MLP: backbone features → single anomaly logit.

    Args:
        in_features: Input dim. Always 2048 for ResNet50.
        hidden_dim: Intermediate layer size.
        dropout: Dropout probability — important because we only train
            on normal images (small dataset), easy to overfit.
    """

    def __init__(
        self,
        in_features: int = 2048,
        hidden_dim: int = 256,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: Feature vector [B, in_features].
        Returns:
            Raw logit [B, 1].
        """
        return self.net(x)


class ResNet50AnomalyDetector(nn.Module):
    """ResNet50 backbone + lightweight anomaly scoring head.

    Trained with CutPaste synthetic anomalies: normal images get label 0,
    cut-pasted images get label 1. The model learns to detect anything
    that looks spatially inconsistent — which generalises well to real
    industrial defects.

    The backbone is split at layer4/avgpool so get_anomaly_map() can
    return spatial heatmaps (7×7 → upsampled to 224×224) without hooks.

    Args:
        pretrained: Load ImageNet weights. Almost always True.
        hidden_dim: Hidden dim of the scoring head.
        dropout: Dropout in the head.
        freeze_backbone: Freeze backbone. Faster but weaker. Useful for
            a quick smoke test.

    Example:
        >>> model = ResNet50AnomalyDetector()
        >>> logits = model(torch.randn(4, 3, 224, 224))      # [4, 1]
        >>> scores = model.anomaly_score(torch.randn(4, 3, 224, 224))  # [4]
    """

    def __init__(
        self,
        pretrained: bool = True,
        hidden_dim: int = 256,
        dropout: float = 0.5,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet50(weights=weights)

        # up to layer4 inclusive — outputs [B, 2048, 7, 7] for 224×224 input
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])
        # global average pool — outputs [B, 2048, 1, 1]
        self.pool = backbone.avgpool

        if freeze_backbone:
            for p in self.feature_extractor.parameters():
                p.requires_grad = False
            logger.info("Backbone frozen — only head is trainable.")

        self.head = AnomalyHead(2048, hidden_dim, dropout)
        logger.info(
            "ResNet50AnomalyDetector | pretrained=%s frozen=%s",
            pretrained, freeze_backbone,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward — returns raw logit.

        Args:
            x: Images [B, 3, H, W], ImageNet-normalised.
        Returns:
            Logit [B, 1]. Use BCEWithLogitsLoss during training.
        """
        feat_map = self.feature_extractor(x)       # [B, 2048, 7, 7]
        pooled   = self.pool(feat_map).flatten(1)  # [B, 2048]
        return self.head(pooled)                   # [B, 1]

    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Inference-time probability — sigmoid of logit.

        Args:
            x: Images [B, 3, H, W].
        Returns:
            Anomaly probabilities [B], in [0, 1].
        """
        return torch.sigmoid(self.forward(x)).squeeze(1)

    @torch.no_grad()
    def get_anomaly_map(self, x: torch.Tensor, image_size: int = 224) -> torch.Tensor:
        """Pixel-level anomaly heatmap via feature-map L2 norm.

        Takes the L2 norm of activations at each of the 7×7 spatial positions
        in the last conv layer, then bilinearly upsamples to full image size.
        Not perfect (image-level training only), but a meaningful baseline
        for pixel-level AUROC — better than a flat score per image.

        Args:
            x: Images [B, 3, H, W].
            image_size: Output spatial resolution.
        Returns:
            Anomaly maps [B, image_size, image_size].
        """
        feat_map  = self.feature_extractor(x)                   # [B, 2048, 7, 7]
        norm_map  = feat_map.norm(dim=1, keepdim=True)           # [B, 1, 7, 7]
        upsampled = F.interpolate(
            norm_map,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )                                                        # [B, 1, H, W]
        return upsampled.squeeze(1)                              # [B, H, W]

    @torch.no_grad()
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Global feature vectors (post-GAP) — used by later modules.

        Args:
            x: Images [B, 3, H, W].
        Returns:
            Feature vectors [B, 2048].
        """
        feat_map = self.feature_extractor(x)
        return self.pool(feat_map).flatten(1)
