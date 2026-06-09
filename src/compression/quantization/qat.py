"""Quantization-Aware Training via FX Graph Mode.

"""
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.quantize_fx import convert_fx, prepare_qat_fx
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluation.metrics import compute_auroc

logger = logging.getLogger("edgevision")


def prepare_qat(
    model: nn.Module,
    example_input: tuple = (torch.randn(1, 3, 224, 224),),
) -> nn.Module:
    """Insert fake-quantization nodes for QAT via FX tracing.

    Args:
        model: FP32 model on CPU.
        example_input: Tuple of example tensors matching model input.

    Returns:
        QAT-prepared GraphModule.
    """
    model.train()
    qconfig_mapping = QConfigMapping().set_global(
        torch.ao.quantization.get_default_qat_qconfig("x86")
    )
    prepared = prepare_qat_fx(model, qconfig_mapping, example_inputs=example_input)
    logger.info("QAT: fake-quant nodes inserted via FX tracing.")
    return prepared


def convert_qat(model: nn.Module) -> nn.Module:
    """Freeze fake-quant observers and convert to real INT8 operators.

    Args:
        model: QAT-trained GraphModule (after prepare_qat + fine-tuning).

    Returns:
        New INT8 GraphModule.
    """
    model.eval()
    model.cpu()
    quantized = convert_fx(model)
    logger.info("QAT: model converted to INT8.")
    return quantized


def export_onnx(
    model: nn.Module,
    output_path: Path,
    input_shape: tuple = (1, 3, 224, 224),
) -> Path:
    """Export INT8 model to ONNX for edge deployment.

    Args:
        model: Converted INT8 model.
        output_path: Path to write the .onnx file.
        input_shape: Input tensor shape.

    Returns:
        Path to the saved ONNX file.
    """
    model.eval()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        torch.zeros(input_shape),
        str(output_path),
        input_names=["image"],
        output_names=["logit"],
        opset_version=17,
        do_constant_folding=True,
    )
    size_mb = output_path.stat().st_size / 1e6
    logger.info("ONNX export -> %s (%.1f MB)", output_path, size_mb)
    return output_path


class QATFinetuner:
    """Fine-tunes a QAT-prepared GraphModule.

    Args:
        model: QAT-prepared model (after prepare_qat).
        device: Training device. CPU required for quantized ops.
        lr: Fine-tuning learning rate.
        n_epochs: Number of fine-tuning epochs.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        lr: float = 1e-5,
        n_epochs: int = 5,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.n_epochs = n_epochs
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=n_epochs, eta_min=lr * 0.01,
        )

    def run(self, train_loader: DataLoader, val_loader: DataLoader) -> float:
        """Run QAT fine-tuning. Returns best val AUROC.

        Args:
            train_loader: CutPaste-augmented training loader.
            val_loader: Clean val loader for AUROC tracking.

        Returns:
            Best val AUROC observed during fine-tuning.
        """
        best = 0.0
        for epoch in range(1, self.n_epochs + 1):
            loss = self._train(train_loader)
            auroc = self._val(val_loader)
            self.scheduler.step()
            logger.info(
                "  QAT %2d/%d | loss=%.4f  auroc=%.4f",
                epoch, self.n_epochs, loss, auroc,
            )
            if auroc > best:
                best = auroc
        return best

    def _train(self, loader: DataLoader) -> float:
        """One training pass with fake-quantization active.

        Args:
            loader: Training DataLoader.

        Returns:
            Mean loss for the epoch.
        """
        self.model.train()
        total = 0.0
        for batch in tqdm(loader, desc="  qat", leave=False):
            img = batch["image"].to(self.device)
            lbl = batch["label"].float().to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(img).squeeze(1), lbl)
            loss.backward()
            self.optimizer.step()
            total += loss.item()
        return total / len(loader)

    @torch.no_grad()
    def _val(self, loader: DataLoader) -> float:
        """Validation AUROC with fake-quant in eval mode.

        Args:
            loader: Validation DataLoader.

        Returns:
            AUROC in [0.0, 1.0].
        """
        self.model.eval()
        scores, labels = [], []
        for batch in loader:
            logits = self.model(batch["image"].to(self.device)).squeeze(1)
            scores.extend(torch.sigmoid(logits).cpu().tolist())
            labels.extend(batch["label"].tolist())
        return compute_auroc(labels, scores)