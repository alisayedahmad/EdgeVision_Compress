"""Tests for Module 2 — model, CutPaste, and metrics.

All tests run on CPU with pretrained=False (no weight download, no GPU).
"""
import numpy as np
import pytest
import torch
from PIL import Image

from data.cutpaste import CutPasteDataset, CutPasteTransform
from evaluation.metrics import compute_auroc, compute_pixel_auroc
from models.resnet import AnomalyHead, ResNet50AnomalyDetector
from utils.seed import seed_everything


# ── helpers ───────────────────────────────────────────────────────────────────

def _rand_pil(size: int = 224) -> Image.Image:
    arr = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


# ── AnomalyHead ───────────────────────────────────────────────────────────────

class TestAnomalyHead:
    def test_output_shape(self) -> None:
        head = AnomalyHead(in_features=2048, hidden_dim=256)
        assert head(torch.randn(4, 2048)).shape == (4, 1)

    def test_output_is_raw_logit_not_sigmoid(self) -> None:
        # force the last layer to output 100.0 — impossible if sigmoid is inside
        head = AnomalyHead()
        with torch.no_grad():
            head.net[-1].weight.fill_(1.0)
            head.net[-1].bias.fill_(100.0)
        out = head(torch.zeros(1, 2048))
        assert out.item() > 1.0, "Sigmoid must NOT be applied inside the head"


# ── ResNet50AnomalyDetector ───────────────────────────────────────────────────

class TestResNet50AnomalyDetector:
    def test_forward_shape(self) -> None:
        model = ResNet50AnomalyDetector(pretrained=False)
        assert model(torch.randn(2, 3, 224, 224)).shape == (2, 1)

    def test_anomaly_score_range(self) -> None:
        model  = ResNet50AnomalyDetector(pretrained=False)
        scores = model.anomaly_score(torch.randn(4, 3, 224, 224))
        assert scores.shape == (4,)
        assert scores.min() >= 0.0 and scores.max() <= 1.0

    def test_get_features_shape(self) -> None:
        model = ResNet50AnomalyDetector(pretrained=False)
        assert model.get_features(torch.randn(2, 3, 224, 224)).shape == (2, 2048)

    def test_anomaly_map_shape(self) -> None:
        model = ResNet50AnomalyDetector(pretrained=False)
        amap  = model.get_anomaly_map(torch.randn(2, 3, 224, 224), image_size=224)
        assert amap.shape == (2, 224, 224)

    def test_anomaly_map_non_negative(self) -> None:
        # L2 norm is always >= 0
        model = ResNet50AnomalyDetector(pretrained=False)
        amap  = model.get_anomaly_map(torch.randn(2, 3, 224, 224))
        assert amap.min() >= 0.0

    def test_freeze_backbone(self) -> None:
        model = ResNet50AnomalyDetector(pretrained=False, freeze_backbone=True)
        for p in model.feature_extractor.parameters():
            assert not p.requires_grad

    def test_head_always_trainable(self) -> None:
        model = ResNet50AnomalyDetector(pretrained=False, freeze_backbone=True)
        for p in model.head.parameters():
            assert p.requires_grad

    def test_reproducible_with_seed(self) -> None:
        x = torch.randn(2, 3, 224, 224)
        seed_everything(0)
        o1 = ResNet50AnomalyDetector(pretrained=False)(x)
        seed_everything(0)
        o2 = ResNet50AnomalyDetector(pretrained=False)(x)
        assert torch.allclose(o1, o2)


# ── CutPasteTransform ─────────────────────────────────────────────────────────

class TestCutPasteTransform:
    def test_label_1_when_always_applied(self) -> None:
        _, label = CutPasteTransform(p=1.0)(_rand_pil())
        assert label == 1

    def test_label_0_when_never_applied(self) -> None:
        _, label = CutPasteTransform(p=0.0)(_rand_pil())
        assert label == 0

    def test_output_size_unchanged(self) -> None:
        img = _rand_pil(224)
        out, _ = CutPasteTransform(p=1.0)(img)
        assert out.size == img.size

    def test_returns_pil_and_int(self) -> None:
        aug, label = CutPasteTransform(p=1.0)(_rand_pil())
        assert isinstance(aug, Image.Image)
        assert isinstance(label, int)


# ── metrics ───────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_perfect_auroc(self) -> None:
        assert compute_auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0

    def test_random_auroc(self) -> None:
        assert compute_auroc([0, 1], [0.5, 0.5]) == 0.5

    def test_single_class_returns_half(self) -> None:
        assert compute_auroc([0, 0, 0], [0.1, 0.2, 0.3]) == 0.5

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Length mismatch"):
            compute_auroc([0, 1], [0.5])

    def test_pixel_auroc_perfect(self) -> None:
        masks = np.array([[[0, 1], [0, 1]]])
        maps  = np.array([[[0.1, 0.9], [0.1, 0.9]]])
        assert compute_pixel_auroc(masks, maps) == 1.0

    def test_pixel_auroc_no_anomaly_returns_half(self) -> None:
        masks = np.zeros((2, 4, 4), dtype=np.int32)
        maps  = np.random.rand(2, 4, 4).astype(np.float32)
        assert compute_pixel_auroc(masks, maps) == 0.5
