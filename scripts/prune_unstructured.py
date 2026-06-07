"""Iterative L1 unstructured magnitude pruning of the ResNet50 baseline.

Loads the trained baseline, runs N prune→fine-tune cycles toward
a target sparsity, then evaluates on the test set and saves the
pruned checkpoint. Logs the full sparsity-AUROC curve to MLflow.

Usage:
    python scripts/prune_unstructured.py
    python scripts/prune_unstructured.py compression.pruning.target_sparsity=0.7
    python scripts/prune_unstructured.py compression.pruning.n_iterations=3
    python scripts/prune_unstructured.py mlflow.tracking_uri=sqlite:///mlflow.db
"""
import logging
import platform
import sys
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from compression.pruning.finetuner import PruningFinetuner
from compression.pruning.magnitude import (
    apply_global_l1_pruning,
    get_model_sparsity,
    get_sparsity_schedule,
    make_pruning_permanent,
)
from data.cutpaste import CutPasteDataset, CutPasteTransform
from data.mvtec import MVTecDataset
from evaluation.metrics import compute_auroc, compute_pixel_auroc
from models.resnet import ResNet50AnomalyDetector
from utils.logging_config import setup_logging
from utils.seed import seed_everything

logger = setup_logging(name="edgevision.pruning")


def _save_sparsity_curve(
    curve: list[tuple[float, float]],
    output_path: Path,
    baseline_auroc: float,
) -> None:
    """Save a sparsity vs AUROC plot as a PNG artifact.

    Args:
        curve: List of (sparsity, auroc) tuples including the baseline point.
        output_path: Where to save the PNG.
        baseline_auroc: Baseline AUROC for the reference horizontal line.
    """
    sparsities = [s for s, _ in curve]
    aurocs = [a for _, a in curve]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sparsities, aurocs, "o-", linewidth=2, markersize=7, label="Pruned model")
    ax.axhline(
        y=baseline_auroc,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label=f"Baseline AUROC = {baseline_auroc:.4f}",
    )
    ax.set_xlabel("Sparsity (fraction of zero weights)", fontsize=12)
    ax.set_ylabel("Val AUROC", fontsize=12)
    ax.set_title("Unstructured Pruning — Sparsity vs AUROC", fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.02, 0.75)
    ax.set_ylim(0.5, 1.05)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Sparsity-AUROC curve saved -> %s", output_path)


@hydra.main(version_base=None, config_path="../configs", config_name="prune")
def main(cfg: DictConfig) -> None:
    """Run iterative unstructured pruning from the baseline checkpoint.

    Args:
        cfg: Hydra config from configs/prune.yaml.
    """
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = Path(get_original_cwd())
    logger.info("Device: %s | category: %s", device, cfg.data.category)

    # ── load baseline checkpoint ───────────────────────────────────────────
    ckpt_path = project_root / cfg.paths.checkpoint_dir / "best_model.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Baseline checkpoint not found at {ckpt_path}. "
            "Run scripts/train_baseline.py first."
        )

    model = ResNet50AnomalyDetector(
        pretrained=False,
        hidden_dim=cfg.model.hidden_dim,
        dropout=cfg.model.dropout,
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    baseline_val_auroc: float = ckpt["val_auroc"]
    baseline_sparsity = get_model_sparsity(model)
    logger.info(
        "Baseline loaded | val_auroc=%.4f | sparsity=%.4f",
        baseline_val_auroc,
        baseline_sparsity,
    )

    # ── dataloaders ───────────────────────────────────────────────────────
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
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=nw,
        pin_memory=pin,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=pin,
    )

    logger.info(
        "Data ready | train=%d  val=%d  test=%d | workers=%d",
        len(train_ds),
        len(val_ds),
        len(test_ds),
        nw,
    )

    # ── MLflow ────────────────────────────────────────────────────────────
    # on détecte le scheme et on construit le chemin absolu manuellement
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
        or f"pruning_{cfg.data.category}_s{cfg.compression.pruning.target_sparsity}"
    )

    pruning_cfg = cfg.compression.pruning
    schedule = get_sparsity_schedule(
        pruning_cfg.target_sparsity,
        pruning_cfg.n_iterations,
    )
    logger.info("Sparsity schedule: %s", [f"{s:.2f}" for s in schedule])

    # tracks (sparsity, auroc) for the curve — starts with the baseline
    sparsity_auroc_curve: list[tuple[float, float]] = [
        (baseline_sparsity, baseline_val_auroc)
    ]

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "category": cfg.data.category,
            "method": pruning_cfg.method,
            "target_sparsity": pruning_cfg.target_sparsity,
            "n_iterations": pruning_cfg.n_iterations,
            "finetune_epochs": pruning_cfg.finetune_epochs,
            "finetune_lr": pruning_cfg.finetune_lr,
            "baseline_val_auroc": baseline_val_auroc,
            "seed": cfg.seed,
        })

        # log baseline as step 0
        mlflow.log_metrics(
            {"sparsity": baseline_sparsity, "val_auroc": baseline_val_auroc},
            step=0,
        )

        step_offset = 0
        for iteration, target_sparsity in enumerate(schedule, start=1):
            logger.info(
                "-- Iteration %d/%d | cumulative target=%.3f --",
                iteration,
                len(schedule),
                target_sparsity,
            )

            apply_global_l1_pruning(model, target_sparsity)
            actual_sparsity = get_model_sparsity(model)

            finetuner = PruningFinetuner(
                model=model,
                device=device,
                lr=pruning_cfg.finetune_lr,
                n_epochs=pruning_cfg.finetune_epochs,
                mlflow_prefix=f"iter_{iteration}",
            )
            best_auroc = finetuner.run(
                train_loader,
                val_loader,
                step_offset=step_offset,
            )
            step_offset += pruning_cfg.finetune_epochs

            sparsity_auroc_curve.append((actual_sparsity, best_auroc))
            mlflow.log_metrics(
                {"sparsity": actual_sparsity, "val_auroc": best_auroc},
                step=iteration,
            )
            logger.info(
                "Iteration %d done | sparsity=%.4f | val_auroc=%.4f",
                iteration,
                actual_sparsity,
                best_auroc,
            )

        # ── make permanent and run final test evaluation ───────────────
        make_pruning_permanent(model)
        final_sparsity = get_model_sparsity(model)

        all_scores: list[float] = []
        all_labels: list[int] = []
        all_maps: list[np.ndarray] = []
        all_masks: list[np.ndarray] = []

        model.eval()
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

        logger.info(
            "Final test | sparsity=%.4f | image_auroc=%.4f | pixel_auroc=%.4f",
            final_sparsity,
            test_image_auroc,
            test_pixel_auroc,
        )
        mlflow.log_metrics({
            "final_sparsity": final_sparsity,
            "test_image_auroc": test_image_auroc,
            "test_pixel_auroc": test_pixel_auroc,
            "auroc_drop": baseline_val_auroc - test_image_auroc,
        })

        # ── save pruned checkpoint ─────────────────────────────────────
        ckpt_out_dir = project_root / cfg.paths.checkpoint_dir
        ckpt_out_dir.mkdir(parents=True, exist_ok=True)
        pruned_path = ckpt_out_dir / "pruned_unstructured.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "final_sparsity": final_sparsity,
                "test_image_auroc": test_image_auroc,
                "test_pixel_auroc": test_pixel_auroc,
                "sparsity_auroc_curve": sparsity_auroc_curve,
                "compression_cfg": dict(pruning_cfg),
            },
            pruned_path,
        )
        mlflow.log_artifact(str(pruned_path))
        logger.info("Pruned checkpoint saved -> %s", pruned_path)

        # ── sparsity-AUROC curve plot ──────────────────────────────────
        curve_path = project_root / "outputs" / "pruning" / "sparsity_auroc_curve.png"
        _save_sparsity_curve(sparsity_auroc_curve, curve_path, baseline_val_auroc)
        mlflow.log_artifact(str(curve_path))

        # summary table in logs
        logger.info("-" * 50)
        logger.info("%-10s  %-10s", "Sparsity", "Val AUROC")
        for s, a in sparsity_auroc_curve:
            logger.info("%-10.4f  %-10.4f", s, a)
        logger.info("-" * 50)

if __name__ == "__main__":
    main()