"""MVTec Anomaly Detection dataset loader.

MVTec AD has 15 categories of industrial objects and textures.
Each category contains:
  train/good/               only normal images, used for training
  test/good/                normal images in the test pool
  test/<defect>/            one folder per defect type
  ground_truth/<defect>/    binary pixel masks for the anomalous regions

MVTec has no official val split, so we carve one out of the test pool.
The carving is stratified (keeps the normal/anomaly ratio) and reproducible
— same seed always produces the same partition.

Reference:
    Bergmann et al., "MVTec AD — A Comprehensive Real-World Dataset for
    Unsupervised Anomaly Detection", CVPR 2019.
"""
import logging
import random
from pathlib import Path
from typing import Optional
import platform

import torch
from PIL import Image
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from data.transforms import get_image_transform, get_mask_transform

logger = logging.getLogger("edgevision")


class MVTecDataset(Dataset):
    """PyTorch Dataset for MVTec AD — one category at a time.

    Each __getitem__ returns a dict with:
      image     FloatTensor [3, H, W]  normalized with ImageNet stats
      label     int                    0 = normal, 1 = anomaly
      mask      FloatTensor [1, H, W]  pixel GT (all zeros for normal images)
      category  str                    e.g. "bottle"

    Args:
        root: Path to the MVTec root directory (parent of all categories).
        category: Which category to load. Must be in CATEGORIES.
        split: "train", "val", or "test".
        transform: Image transform. Defaults to the standard one for the split.
        mask_transform: Mask transform. Defaults to nearest-neighbor pipeline.
        val_fraction: Fraction of the test pool reserved for validation.
        seed: Controls the val/test split. Keep constant for reproducibility.
    """

    CATEGORIES: list[str] = [
        "bottle", "cable", "capsule", "carpet", "grid",
        "hazelnut", "leather", "metal_nut", "pill", "screw",
        "tile", "toothbrush", "transistor", "wood", "zipper",
    ]

    def __init__(
        self,
        root: Path,
        category: str,
        split: str,
        transform: Optional[transforms.Compose] = None,
        mask_transform: Optional[transforms.Compose] = None,
        val_fraction: float = 0.2,
        seed: int = 42,
    ) -> None:
        if category not in self.CATEGORIES:
            raise ValueError(
                f"Unknown category '{category}'. "
                f"Valid options: {self.CATEGORIES}"
            )
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split '{split}'.")

        self.root = Path(root)
        self.category = category
        self.split = split
        self.val_fraction = val_fraction
        self.seed = seed
        self.transform = transform or get_image_transform(split)
        self.mask_transform = mask_transform or get_mask_transform()

        # each entry: (image_path, label, mask_path_or_None)
        self.samples: list[tuple[Path, int, Optional[Path]]] = []
        self._load_samples()

        logger.info(
            "MVTecDataset | category=%-12s split=%-5s | %d samples",
            category, split, len(self.samples),
        )

    def _load_samples(self) -> None:
        """Walk the directory tree and populate self.samples."""
        category_dir = self.root / self.category

        if self.split == "train":
            # train/ contains only normal images — straightforward
            good_dir = category_dir / "train" / "good"
            for img_path in sorted(good_dir.glob("*.png")):
                self.samples.append((img_path, 0, None))
            return

        # for val and test, start from the full test pool
        all_samples: list[tuple[Path, int, Optional[Path]]] = []
        test_dir = category_dir / "test"

        for defect_dir in sorted(test_dir.iterdir()):
            if not defect_dir.is_dir():
                continue

            is_anomaly = defect_dir.name != "good"
            label = int(is_anomaly)

            for img_path in sorted(defect_dir.glob("*.png")):
                mask_path: Optional[Path] = None
                if is_anomaly:
                    # mask lives at ground_truth/<defect>/<stem>_mask.png
                    candidate = (
                        category_dir
                        / "ground_truth"
                        / defect_dir.name
                        / (img_path.stem + "_mask.png")
                    )
                    mask_path = candidate if candidate.exists() else None
                all_samples.append((img_path, label, mask_path))

        self.samples = self._val_test_split(all_samples)

    def _val_test_split(
        self,
        all_samples: list[tuple[Path, int, Optional[Path]]],
    ) -> list[tuple[Path, int, Optional[Path]]]:
        """Split the test pool into val and test subsets, stratified by label.

        Stratification ensures the normal/anomaly ratio is preserved in both
        splits, giving an unbiased AUROC estimate on each.

        Args:
            all_samples: Full list of test samples before splitting.

        Returns:
            The subset corresponding to self.split ("val" or "test").
        """
        rng = random.Random(self.seed)

        normal  = [s for s in all_samples if s[1] == 0]
        anomaly = [s for s in all_samples if s[1] == 1]

        def carve(group: list) -> tuple[list, list]:
            """Shuffle and carve off the val portion."""
            rng.shuffle(group)
            n_val = max(1, int(len(group) * self.val_fraction))
            return group[:n_val], group[n_val:]

        val_normal,  test_normal  = carve(normal)
        val_anomaly, test_anomaly = carve(anomaly)

        pool = (
            val_normal + val_anomaly
            if self.split == "val"
            else test_normal + test_anomaly
        )
        # sort by path for determinism after the shuffle
        return sorted(pool, key=lambda x: str(x[0]))

    def __len__(self) -> int:
        """Return the number of samples in this split."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        """Load and return one sample as a dict.

        Args:
            idx: Index into self.samples.

        Returns:
            Dict with keys: image, label, mask, category.
        """
        img_path, label, mask_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        image_t: torch.Tensor = self.transform(image)

        if mask_path is not None:
            mask_img = Image.open(mask_path).convert("L")
            mask_t: torch.Tensor = self.mask_transform(mask_img)
        else:
            # zero mask for normal images — no anomaly pixels by definition
            _, h, w = image_t.shape
            mask_t = torch.zeros(1, h, w, dtype=torch.float32)

        return {
            "image":    image_t,
            "label":    label,
            "mask":     mask_t,
            "category": self.category,
        }


def get_dataloaders(
    cfg: DictConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Instantiate train, val, test DataLoaders from a Hydra config.

    Workers are set to 0 on Windows — the spawn-based multiprocessing
    used by DataLoader requires a __main__ guard that training scripts
    don't always have.

    Args:
        cfg: Full Hydra config (needs paths.data_dir, data, training).

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    root         = Path(cfg.paths.data_dir)
    category     = cfg.data.category
    val_fraction = cfg.data.val_fraction
    seed         = cfg.seed
    bs           = cfg.training.batch_size
    # Windows + DataLoader workers > 0 can cause silent hangs
    nw  = cfg.training.num_workers if platform.system() != "Windows" else 0
    pin = torch.cuda.is_available()

    train_ds = MVTecDataset(root, category, "train", seed=seed, val_fraction=val_fraction)
    val_ds   = MVTecDataset(root, category, "val",   seed=seed, val_fraction=val_fraction)
    test_ds  = MVTecDataset(root, category, "test",  seed=seed, val_fraction=val_fraction)

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=nw, pin_memory=pin, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds, batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=pin,
    )

    logger.info(
        "DataLoaders ready | train=%d val=%d test=%d | workers=%d",
        len(train_ds), len(val_ds), len(test_ds), nw,
    )
    return train_loader, val_loader, test_loader
