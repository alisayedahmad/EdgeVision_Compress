"""Evaluation metrics for anomaly detection.

Two metrics, two levels of granularity:
  - Image AUROC : can the model rank anomalous images above normal ones?
  - Pixel AUROC : can the model locate where the anomaly is?

AUROC is threshold-free — it measures ranking quality over the whole
operating range. That's what we want: a model that consistently scores
anomalies higher than normals, regardless of deployment threshold.
"""
import logging
from typing import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

logger = logging.getLogger("edgevision")


def compute_auroc(
    labels: Sequence[int],
    scores: Sequence[float],
) -> float:
    """Image-level AUROC.

    Probability that a random anomalous image scores higher than a random
    normal image. 0.5 = random ranking. 1.0 = perfect separation.

    Args:
        labels: Ground truth. 0 = normal, 1 = anomaly.
        scores: Predicted anomaly scores. Higher = more anomalous.
    Returns:
        AUROC in [0.0, 1.0]. Returns 0.5 if only one class present.
    Raises:
        ValueError: If labels and scores have different lengths.
    """
    if len(labels) != len(scores):
        raise ValueError(
            f"Length mismatch: labels={len(labels)}, scores={len(scores)}"
        )
    y_true  = np.asarray(labels, dtype=np.int32)
    y_score = np.asarray(scores, dtype=np.float32)

    if len(np.unique(y_true)) < 2:
        logger.warning("AUROC undefined — only one class present. Returning 0.5.")
        return 0.5

    return float(roc_auc_score(y_true, y_score))


def compute_pixel_auroc(
    masks: np.ndarray,
    score_maps: np.ndarray,
) -> float:
    """Pixel-level AUROC.

    Flattens spatial dimensions and computes AUROC pixel by pixel.
    Uses the same protocol as the MVTec AD benchmark paper.

    Args:
        masks: Ground-truth binary masks [N, H, W]. 1 = anomaly pixel.
        score_maps: Predicted anomaly maps [N, H, W]. Higher = more anomalous.
    Returns:
        Pixel-level AUROC in [0.0, 1.0].
    """
    flat_masks  = masks.flatten().astype(np.int32)
    flat_scores = score_maps.flatten().astype(np.float32)

    if len(np.unique(flat_masks)) < 2:
        logger.warning(
            "Pixel AUROC undefined — no anomaly pixels in ground truth. "
            "Returning 0.5."
        )
        return 0.5

    return float(roc_auc_score(flat_masks, flat_scores))
