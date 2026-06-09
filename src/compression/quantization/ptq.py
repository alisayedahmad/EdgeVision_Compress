"""Post-Training Quantization via FX Graph Mode.
"""
import logging

import torch
import torch.nn as nn
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
from torch.utils.data import DataLoader

logger = logging.getLogger("edgevision")


def prepare_ptq(
    model: nn.Module,
    example_input: tuple = (torch.randn(1, 3, 224, 224),),
) -> nn.Module:
    """Trace the model graph and insert calibration observers.

    Args:
        model: FP32 model on CPU.
        example_input: Tuple of example tensors matching the model input.

    Returns:
        Prepared GraphModule ready for calibration.
    """
    model.eval()
    qconfig_mapping = QConfigMapping().set_global(
        torch.ao.quantization.get_default_qconfig("x86")
    )
    prepared = prepare_fx(model, qconfig_mapping, example_inputs=example_input)
    logger.info("PTQ: observers inserted via FX tracing.")
    return prepared


def calibrate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_batches: int = 32,
) -> None:
    """Run forward passes to collect activation statistics.

    Args:
        model: Prepared model (after prepare_ptq).
        loader: DataLoader with representative training data.
        device: Must be CPU for static quantization.
        n_batches: Number of batches to calibrate on.
    """
    model.eval()
    processed = 0
    with torch.no_grad():
        for batch in loader:
            if processed >= n_batches:
                break
            model(batch["image"].to(device))
            processed += 1
    logger.info("PTQ: calibration complete -- %d batches processed.", processed)


def convert_ptq(model: nn.Module) -> nn.Module:
    """Convert calibrated observers to real INT8 operators.

    Args:
        model: Calibrated model (after prepare_ptq + calibrate).

    Returns:
        New INT8 GraphModule.
    """
    quantized = convert_fx(model)
    logger.info("PTQ: model converted to INT8.")
    return quantized