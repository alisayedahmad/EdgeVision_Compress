"""Run structured channel pruning with Taylor importance.

The script loads a checkpoint, prunes conv2 channels globally across
ResNet50 bottlenecks, fine-tunes, then reports FLOPs and AUROC.
"""
import logging
import platform
import sys
from pathlib import Path

import hydra
import mlflow
import numpy as np
import torch
import torch.nn as nn
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from compression.pruning.finetuner import PruningFinetuner
from compression.pruning.flops import count_flops, format_flops
from compression.pruning.structured import (
    TaylorImportanceEstimator,
    apply_structured_pruning,
    compute_keep_indices,
    get_bottleneck_locations,
)
from data.cutpaste import CutPasteDataset, CutPasteTransform
from data.mvtec import MVTecDataset
from evaluation.metrics import compute_auroc, compute_pixel_auroc
from models.resnet import ResNet50AnomalyDetector
from utils.logging_config import setup_logging
from utils.seed import seed_everything

logger = setup_logging(name="edgevision.structured")


@hydra.main(version_base=None, config_path="../configs", config_name="prune_structured")
def main(cfg: DictConfig) -> None:
    """Run the full structured pruning pipeline from config."""
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = Path(get_original_cwd())
    logger.info("Device: %s | category: %s", device, cfg.data.category)

    # Load checkpoint: prefer unstructured-pruned, fallback to baseline.
    ckpt_path = project_root / cfg.paths.checkpoint_dir / "pruned_unstructured.pth"
    if not ckpt_path.exists():
        ckpt_path = project_root / cfg.paths.checkpoint_dir / "best_model.pth"
        logger.info("Unstructured checkpoint not found, using baseline.")

    model = ResNet50AnomalyDetector(
        pretrained=False,
        hidden_dim=cfg.model.hidden_dim,
        dropout=cfg.model.dropout,
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    logger.info("Checkpoint loaded from %s", ckpt_path)

    flops_before = count_flops(model, device=device)
    logger.info("FLOPs before pruning: %s", format_flops(flops_before))

    # Build data loaders.
    data_dir = project_root / cfg.paths.data_dir
    nw = cfg.training.num_workers if platform.system() != "Windows" else 0
    pin = torch.cuda.is_available()

    train_base = MVTecDataset(
        data_dir, cfg.data.category, "train",
        seed=cfg.seed, val_fraction=cfg.data.val_fraction,
    )
    train_ds = CutPasteDataset(train_base, CutPasteTransform(p=0.5))
    val_ds = MVTecDataset(
        data_dir, cfg.data.category, "val",
        seed=cfg.seed, val_fraction=cfg.data.val_fraction,
    )
    test_ds = MVTecDataset(
        data_dir, cfg.data.category, "test",
        seed=cfg.seed, val_fraction=cfg.data.val_fraction,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size,
        shuffle=True, num_workers=nw, pin_memory=pin, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size,
        shuffle=False, num_workers=nw, pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.training.batch_size,
        shuffle=False, num_workers=nw, pin_memory=pin,
    )
    logger.info(
        "Data | train=%d  val=%d  test=%d | workers=%d",
        len(train_ds), len(val_ds), len(test_ds), nw,
    )

    # Accumulate Taylor scores on conv2 layers.
    struct_cfg = cfg.compression.structured
    criterion = nn.BCEWithLogitsLoss()
    locations = get_bottleneck_locations(model)
    logger.info("Found %d Bottleneck blocks.", len(locations))

    estimator = TaylorImportanceEstimator()
    for i, (_, _, block) in enumerate(locations):
        estimator.register(f"block_{i}", block.conv2)

    estimator.accumulate(
        model, train_loader, criterion, device,
        n_batches=struct_cfg.n_calibration_batches,
    )
    scores = estimator.get_scores()
    estimator.remove_hooks()

    # Compute which channels to keep.
    keep_indices = compute_keep_indices(
        scores,
        prune_ratio=struct_cfg.prune_ratio,
        min_channels=struct_cfg.min_channels,
    )

    total_before = sum(len(s) for s in scores.values())
    total_after = sum(len(ki) for ki in keep_indices.values())
    logger.info(
        "Channel reduction: %d -> %d (%.1f%% removed)",
        total_before, total_after,
        100.0 * (total_before - total_after) / total_before,
    )

    # Apply structured pruning.
    apply_structured_pruning(model, keep_indices)
    model.to(device)  # new layers created on CPU by default — move back to device

    flops_after_pruning = count_flops(model, device=device)
    flops_reduction = 1.0 - flops_after_pruning / flops_before
    logger.info(
        "FLOPs after pruning: %s (%.1f%% reduction)",
        format_flops(flops_after_pruning),
        100.0 * flops_reduction,
    )

    # Configure MLflow.
    raw_uri = cfg.mlflow.tracking_uri
    if raw_uri.startswith("sqlite:///"):
        db_path = project_root / raw_uri.replace("sqlite:///", "", 1)
        tracking_uri = f"sqlite:///{db_path}"
    else:
        tracking_uri = str(project_root / raw_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    run_name = (
        cfg.mlflow.run_name
        or f"structured_{cfg.data.category}_r{struct_cfg.prune_ratio}"
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "category": cfg.data.category,
            "prune_ratio": struct_cfg.prune_ratio,
            "min_channels": struct_cfg.min_channels,
            "n_calibration_batches": struct_cfg.n_calibration_batches,
            "finetune_epochs": struct_cfg.finetune_epochs,
            "finetune_lr": struct_cfg.finetune_lr,
            "seed": cfg.seed,
        })
        mlflow.log_metrics({
            "flops_before": flops_before,
            "flops_after_pruning": flops_after_pruning,
            "flops_reduction_pct": 100.0 * flops_reduction,
            "channels_before": total_before,
            "channels_after": total_after,
        })

        # Fine-tune after pruning.
        logger.info("Fine-tuning for %d epochs...", struct_cfg.finetune_epochs)
        finetuner = PruningFinetuner(
            model=model,
            device=device,
            lr=struct_cfg.finetune_lr,
            n_epochs=struct_cfg.finetune_epochs,
            mlflow_prefix="structured",
        )
        best_val_auroc = finetuner.run(train_loader, val_loader)
        logger.info("Best val AUROC after fine-tuning: %.4f", best_val_auroc)

        # Final test evaluation.
        model.eval()
        all_scores: list[float] = []
        all_labels: list[int] = []
        all_maps: list[np.ndarray] = []
        all_masks: list[np.ndarray] = []

        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(device)
                logits = model(images).squeeze(1)
                all_scores.extend(torch.sigmoid(logits).cpu().tolist())
                all_labels.extend(batch["label"].tolist())
                amap = model.get_anomaly_map(images)
                all_maps.append(amap.cpu().numpy())
                all_masks.append(batch["mask"].squeeze(1).numpy())

        test_image_auroc = compute_auroc(all_labels, all_scores)
        maps_np = np.concatenate(all_maps, axis=0)
        masks_np = np.concatenate(all_masks, axis=0)
        test_pixel_auroc = compute_pixel_auroc(masks_np, maps_np)

        mlflow.log_metrics({
            "test_image_auroc": test_image_auroc,
            "test_pixel_auroc": test_pixel_auroc,
            "best_val_auroc": best_val_auroc,
        })

        # Save checkpoint.
        ckpt_out = project_root / cfg.paths.checkpoint_dir / "pruned_structured.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "test_image_auroc": test_image_auroc,
                "test_pixel_auroc": test_pixel_auroc,
                "flops_before": flops_before,
                "flops_after": flops_after_pruning,
                "flops_reduction_pct": 100.0 * flops_reduction,
                "compression_cfg": dict(struct_cfg),
            },
            ckpt_out,
        )
        mlflow.log_artifact(str(ckpt_out))

        # Summary table.
        logger.info("-" * 55)
        logger.info("%-28s  %s", "Metric", "Value")
        logger.info("-" * 55)
        logger.info("%-28s  %s", "FLOPs before", format_flops(flops_before))
        logger.info("%-28s  %s", "FLOPs after pruning", format_flops(flops_after_pruning))
        logger.info("%-28s  %.1f%%", "FLOPs reduction", 100.0 * flops_reduction)
        logger.info("%-28s  %.4f", "Test image AUROC", test_image_auroc)
        logger.info("%-28s  %.4f", "Test pixel AUROC", test_pixel_auroc)
        logger.info("-" * 55)


if __name__ == "__main__":
    main()