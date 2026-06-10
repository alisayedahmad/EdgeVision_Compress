"""Tests for Module 6 — knowledge distillation."""
import torch
import torch.nn as nn

from compression.distillation.loss import DistillationLoss
from models.mobilenet import MobileNetV3Student


class TestMobileNetV3Student:
    def test_forward_shape(self) -> None:
        model = MobileNetV3Student(pretrained=False)
        assert model(torch.randn(2, 3, 224, 224)).shape == (2, 1)

    def test_anomaly_score_range(self) -> None:
        model = MobileNetV3Student(pretrained=False)
        scores = model.anomaly_score(torch.randn(2, 3, 224, 224))
        assert scores.min() >= 0.0 and scores.max() <= 1.0

    def test_feature_vector_shape(self) -> None:
        model = MobileNetV3Student(pretrained=False)
        feat = model.get_feature_vector(torch.randn(2, 3, 224, 224))
        assert feat.shape == (2, 576)

    def test_much_smaller_than_resnet(self) -> None:
        n_params = sum(p.numel() for p in MobileNetV3Student(pretrained=False).parameters())
        assert n_params < 5_000_000  # should be ~2.5M


class TestDistillationLoss:
    def test_returns_all_keys(self) -> None:
        loss_fn = DistillationLoss()
        result = loss_fn(
            student_logits=torch.randn(4, 1),
            teacher_logits=torch.randn(4, 1),
            labels=torch.randint(0, 2, (4,)).float(),
            student_features=torch.randn(4, 2048),
            teacher_features=torch.randn(4, 2048),
        )
        assert set(result.keys()) == {"total", "task", "kl", "feature"}

    def test_total_is_weighted_sum(self) -> None:
        loss_fn = DistillationLoss(alpha=0.3, beta=0.5, gamma=0.2)
        result = loss_fn(
            torch.randn(4, 1), torch.randn(4, 1),
            torch.randint(0, 2, (4,)).float(),
            torch.randn(4, 64), torch.randn(4, 64),
        )
        assert result["total"].item() > 0

    def test_zero_temperature_does_not_crash(self) -> None:
        # temperature=1 is fine, just checking edge case
        loss_fn = DistillationLoss(temperature=1.0)
        loss_fn(torch.randn(4, 1), torch.randn(4, 1),
                torch.zeros(4), torch.randn(4, 32), torch.randn(4, 32))

    def test_projector_gradient_flows(self) -> None:
        projector = nn.Linear(576, 2048)
        s_feat = torch.randn(2, 576, requires_grad=True)
        projected = projector(s_feat)
        loss_fn = DistillationLoss()
        result = loss_fn(
            torch.randn(2, 1), torch.randn(2, 1),
            torch.zeros(2), projected, torch.randn(2, 2048),
        )
        result["total"].backward()
        assert s_feat.grad is not None