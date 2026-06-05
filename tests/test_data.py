"""Tests for the MVTec AD data pipeline.

All tests run on synthetic data — no real dataset required.
The fake directory tree mirrors the real MVTec structure exactly,
so the Dataset logic is exercised end-to-end.
"""
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from data.mvtec import MVTecDataset
from data.transforms import get_image_transform, get_mask_transform
from utils.seed import seed_everything


# ── helpers ───────────────────────────────────────────────────────────────────

def _rand_png(path: Path, mode: str = "RGB", size: tuple = (64, 64)) -> None:
    """Save a random image as PNG at the given path."""
    if mode == "L":
        arr = np.random.randint(0, 2, size, dtype=np.uint8) * 255
    else:
        arr = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    Image.fromarray(arr, mode=mode).save(path)


@pytest.fixture()
def fake_mvtec(tmp_path: Path) -> Path:
    """Build a minimal synthetic MVTec directory for the bottle category.

    Layout:
        bottle/train/good/       10 normal PNGs
        bottle/test/good/         6 normal PNGs
        bottle/test/broken/       6 anomaly PNGs
        bottle/ground_truth/broken/  6 binary mask PNGs
    """
    cat = "bottle"

    train_good = tmp_path / cat / "train" / "good"
    train_good.mkdir(parents=True)
    for i in range(10):
        _rand_png(train_good / f"{i:03d}.png")

    test_good = tmp_path / cat / "test" / "good"
    test_good.mkdir(parents=True)
    for i in range(6):
        _rand_png(test_good / f"{i:03d}.png")

    test_broken = tmp_path / cat / "test" / "broken"
    test_broken.mkdir(parents=True)
    gt_broken = tmp_path / cat / "ground_truth" / "broken"
    gt_broken.mkdir(parents=True)
    for i in range(6):
        _rand_png(test_broken / f"{i:03d}.png")
        _rand_png(gt_broken / f"{i:03d}_mask.png", mode="L")

    return tmp_path


# ── transform tests ───────────────────────────────────────────────────────────

class TestTransforms:
    """Image and mask transform pipelines."""

    def test_train_output_shape(self) -> None:
        t = get_image_transform("train")
        img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        assert t(img).shape == (3, 224, 224)

    def test_val_output_shape(self) -> None:
        t = get_image_transform("val")
        img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        assert t(img).shape == (3, 224, 224)

    def test_unknown_split_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown split"):
            get_image_transform("predict")

    def test_mask_stays_binary(self) -> None:
        """Nearest-neighbor resize must not create values between 0 and 1."""
        t = get_mask_transform()
        arr = np.zeros((256, 256), dtype=np.uint8)
        arr[64:128, 64:128] = 255
        out = t(Image.fromarray(arr, mode="L"))
        unique = out.unique().tolist()
        for v in unique:
            assert v in (0.0, 1.0), f"Non-binary mask value: {v}"

    def test_val_transform_is_deterministic(self) -> None:
        """Applying val transform twice must give identical tensors."""
        t = get_image_transform("val")
        img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        assert torch.allclose(t(img), t(img))


# ── dataset tests ─────────────────────────────────────────────────────────────

class TestMVTecDataset:
    """MVTecDataset with synthetic data."""

    def test_train_all_normal(self, fake_mvtec: Path) -> None:
        ds = MVTecDataset(fake_mvtec, "bottle", "train")
        assert all(ds[i]["label"] == 0 for i in range(len(ds)))

    def test_train_size(self, fake_mvtec: Path) -> None:
        ds = MVTecDataset(fake_mvtec, "bottle", "train")
        assert len(ds) == 10

    def test_val_plus_test_covers_all(self, fake_mvtec: Path) -> None:
        """No sample must be silently dropped during the split."""
        val  = MVTecDataset(fake_mvtec, "bottle", "val",  seed=42)
        test = MVTecDataset(fake_mvtec, "bottle", "test", seed=42)
        assert len(val) + len(test) == 12   # 6 good + 6 broken

    def test_val_test_no_overlap(self, fake_mvtec: Path) -> None:
        val  = MVTecDataset(fake_mvtec, "bottle", "val",  seed=42)
        test = MVTecDataset(fake_mvtec, "bottle", "test", seed=42)
        val_paths  = {str(s[0]) for s in val.samples}
        test_paths = {str(s[0]) for s in test.samples}
        assert val_paths.isdisjoint(test_paths)

    def test_item_keys(self, fake_mvtec: Path) -> None:
        ds = MVTecDataset(fake_mvtec, "bottle", "train")
        assert set(ds[0].keys()) == {"image", "label", "mask", "category"}

    def test_image_shape(self, fake_mvtec: Path) -> None:
        ds = MVTecDataset(fake_mvtec, "bottle", "train")
        assert ds[0]["image"].shape == (3, 224, 224)

    def test_mask_shape(self, fake_mvtec: Path) -> None:
        ds = MVTecDataset(fake_mvtec, "bottle", "train")
        assert ds[0]["mask"].shape == (1, 224, 224)

    def test_normal_mask_is_zero(self, fake_mvtec: Path) -> None:
        ds = MVTecDataset(fake_mvtec, "bottle", "train")
        assert ds[0]["mask"].sum().item() == 0.0

    def test_unknown_category_raises(self, fake_mvtec: Path) -> None:
        with pytest.raises(ValueError, match="Unknown category"):
            MVTecDataset(fake_mvtec, "spaceship", "train")

    def test_split_reproducible(self, fake_mvtec: Path) -> None:
        """Same seed must always produce identical val sets."""
        ds1 = MVTecDataset(fake_mvtec, "bottle", "val", seed=42)
        ds2 = MVTecDataset(fake_mvtec, "bottle", "val", seed=42)
        assert [str(s[0]) for s in ds1.samples] == [str(s[0]) for s in ds2.samples]

    def test_different_seeds_differ(self, fake_mvtec: Path) -> None:
        ds1 = MVTecDataset(fake_mvtec, "bottle", "val", seed=42)
        ds2 = MVTecDataset(fake_mvtec, "bottle", "val", seed=99)
        assert [str(s[0]) for s in ds1.samples] != [str(s[0]) for s in ds2.samples]


# ── seed utility tests ────────────────────────────────────────────────────────

class TestSeedEverything:
    """Reproducibility utilities."""

    def test_torch_reproducible(self) -> None:
        seed_everything(42)
        t1 = torch.randn(10)
        seed_everything(42)
        t2 = torch.randn(10)
        assert torch.allclose(t1, t2)

    def test_numpy_reproducible(self) -> None:
        seed_everything(42)
        a1 = np.random.rand(10)
        seed_everything(42)
        a2 = np.random.rand(10)
        assert np.allclose(a1, a2)
