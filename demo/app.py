"""EdgeVision-Compress — Interactive anomaly detection demo.

Runs the distilled MobileNetV3-Small student model on uploaded images.
Shows anomaly score, prediction, and model metadata.

Launch locally:
    pip install gradio
    python demo/app.py

Deploy to HuggingFace Spaces:
    Copy demo/ folder, add student_distilled.pth to the Space.
"""
import logging
import sys
import time
from pathlib import Path

import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# allow imports from src/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.mobilenet import MobileNetV3Student

logger = logging.getLogger("edgevision.demo")

# ── config ────────────────────────────────────────────────────────────────────

CHECKPOINT_PATHS = [
    Path(__file__).parent / "student_distilled.pth",
    Path(__file__).parent.parent / "outputs" / "checkpoints" / "student_distilled.pth",
]
DEVICE = torch.device("cpu")
THRESHOLD = 0.5
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# ── model loading ─────────────────────────────────────────────────────────────


def load_model() -> nn.Module:
    """Load the distilled student checkpoint.

    Searches multiple paths so the demo works both locally
    and when deployed to HuggingFace Spaces.

    Returns:
        MobileNetV3Student in eval mode on CPU.

    Raises:
        FileNotFoundError: If no checkpoint is found.
    """
    ckpt_path = None
    for p in CHECKPOINT_PATHS:
        if p.exists():
            ckpt_path = p
            break

    if ckpt_path is None:
        raise FileNotFoundError(
            f"No checkpoint found. Searched: {[str(p) for p in CHECKPOINT_PATHS]}. "
            "Copy student_distilled.pth to the demo/ folder or outputs/checkpoints/."
        )

    model = MobileNetV3Student(pretrained=False, hidden_dim=128, dropout=0.3)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info("Model loaded from %s", ckpt_path)
    return model


MODEL = load_model()

# ── inference ─────────────────────────────────────────────────────────────────


def predict(image: Image.Image) -> dict:
    """Run anomaly detection on an uploaded image.

    Args:
        image: PIL image from Gradio input.

    Returns:
        Dict with label confidences for the Gradio Label component.
    """
    if image is None:
        return {"Error": 1.0}

    img = image.convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)

    t0 = time.perf_counter()
    with torch.no_grad():
        logit = MODEL(tensor).squeeze()
        score = torch.sigmoid(logit).item()
    latency_ms = (time.perf_counter() - t0) * 1000

    label = "Anomaly" if score >= THRESHOLD else "Normal"
    logger.info(
        "Prediction: %s | score=%.4f | latency=%.1f ms",
        label, score, latency_ms,
    )

    return {
        "Normal": 1.0 - score,
        "Anomaly": score,
    }


# ── interface ─────────────────────────────────────────────────────────────────

DESCRIPTION = """
## EdgeVision-Compress — Anomaly Detection Demo

Upload an industrial image to detect anomalies using a **MobileNetV3-Small** student model
distilled from a compressed ResNet50 teacher.

**Pipeline:** ResNet50 → Unstructured Pruning → Structured Pruning → QAT INT8 → Knowledge Distillation → MobileNetV3-Small

| Metric | Value |
|---|---|
| Model | MobileNetV3-Small (distilled) |
| Image AUROC | 0.9890 |
| Size | 4.1 MB |
| Latency (ONNX Runtime) | 4.2 ms |
| Parameters | 1.0M |

Trained on [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) bottle category.
"""

EXAMPLES_DIR = Path(__file__).parent / "examples"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an industrial image"),
    outputs=gr.Label(num_top_classes=2, label="Prediction"),
    title="EdgeVision-Compress",
    description=DESCRIPTION,
    examples=[str(p) for p in sorted(EXAMPLES_DIR.glob("*.png"))] if EXAMPLES_DIR.exists() else None,
    theme=gr.themes.Soft(),
    analytics_enabled=False,
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)