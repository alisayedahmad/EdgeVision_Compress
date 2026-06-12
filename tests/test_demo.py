"""Tests  — Gradio demo inference pipeline."""
import pytest
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from models.mobilenet import MobileNetV3Student


TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _random_image() -> Image.Image:
    return Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))


class TestDemoInference:
    def test_transform_output_shape(self) -> None:
        tensor = TRANSFORM(_random_image())
        assert tensor.shape == (3, 224, 224)

    def test_model_produces_score_in_range(self) -> None:
        model = MobileNetV3Student(pretrained=False)
        model.eval()
        tensor = TRANSFORM(_random_image()).unsqueeze(0)
        with torch.no_grad():
            score = torch.sigmoid(model(tensor).squeeze()).item()
        assert 0.0 <= score <= 1.0

    def test_batch_inference_consistent(self) -> None:
        model = MobileNetV3Student(pretrained=False)
        model.eval()
        img = _random_image()
        tensor = TRANSFORM(img).unsqueeze(0)
        with torch.no_grad():
            s1 = torch.sigmoid(model(tensor).squeeze()).item()
            s2 = torch.sigmoid(model(tensor).squeeze()).item()
        assert abs(s1 - s2) < 1e-5

    def test_prediction_returns_two_classes(self) -> None:
        model = MobileNetV3Student(pretrained=False)
        model.eval()
        tensor = TRANSFORM(_random_image()).unsqueeze(0)
        with torch.no_grad():
            score = torch.sigmoid(model(tensor).squeeze()).item()
        result = {"Normal": 1.0 - score, "Anomaly": score}
        assert set(result.keys()) == {"Normal", "Anomaly"}
        assert abs(result["Normal"] + result["Anomaly"] - 1.0) < 1e-5