"""Image and mask transforms for MVTec AD.

Three modes: train (augmented), val/test (clean).
Masks get their own pipeline — nearest neighbor only to keep them binary.
"""
import logging

from torchvision import transforms

logger = logging.getLogger("edgevision")

# ImageNet stats — safe default since ResNet50 was pretrained on it
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


def get_image_transform(split: str, image_size: int = 224) -> transforms.Compose:
    """Build the image transform pipeline for a given split.

    Training uses mild augmentations — nothing too aggressive since
    anomaly detection is texture-sensitive. Too strong augmentation
    can corrupt the normal pattern the model is supposed to learn.

    Val and test are fully deterministic: resize → center crop → normalize.

    Args:
        split: "train", "val", or "test".
        image_size: Final crop size. 224 is the ResNet standard.

    Returns:
        Composed torchvision transform.

    Raises:
        ValueError: If split is not train/val/test.
    """
    if split not in ("train", "val", "test"):
        raise ValueError(f"Unknown split: '{split}'. Expected train/val/test.")

    if split == "train":
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # subtle color jitter — anomalies are structural, not chromatic
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN, std=_STD),
        ])

    # val and test: pure determinism
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])


def get_mask_transform(image_size: int = 224) -> transforms.Compose:
    """Build the transform for ground truth binary masks.

    Masks are binary (0 or 255). NEAREST interpolation is mandatory —
    any other mode would create intermediate values at boundaries,
    turning a clean binary mask into a blurry gradient and corrupting
    pixel-level AUROC computation downstream.

    Args:
        image_size: Final crop size. Must match get_image_transform().

    Returns:
        Composed transform for PIL mask images.
    """
    return transforms.Compose([
        transforms.Resize(
            256,
            interpolation=transforms.InterpolationMode.NEAREST,
        ),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),  # 0/255 → 0.0/1.0, shape becomes [1, H, W]
    ])
