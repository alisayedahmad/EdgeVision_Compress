"""CutPaste augmentation for self-supervised anomaly detection training.

Reference: Li et al., "CutPaste: Self-Supervised Learning for Anomaly
Detection and Localization", CVPR 2021.

Core idea: cut a patch, paste it somewhere else. The image looks almost
normal but something is wrong — and the model has to learn to notice.
That's exactly the skill we need for anomaly detection.

We use the basic variant (no rotation, no scar) — enough for a solid
baseline and simple to reason about.
"""
import logging
import random
from typing import Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger("edgevision")

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

# spatial transforms applied BEFORE CutPaste — crop only, no normalisation
_SPATIAL = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
])

# applied AFTER CutPaste — tensor conversion + normalisation
_TO_TENSOR = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=_MEAN, std=_STD),
])


class CutPasteTransform:
    """Cut a patch from an image and paste it at a different location.

    With probability p: cuts a random rectangular patch, optionally
    colour-jitters it, and pastes it somewhere else. Returns the modified
    image and label=1. Otherwise returns the original and label=0.

    p=0.5 produces a balanced dataset (50% normal, 50% synthetic anomaly)
    with no extra images needed — training set effectively doubles for free.

    Args:
        p: Application probability.
        area_ratio: (min, max) fraction of image area for the patch.
        aspect_ratio: (min, max) aspect ratio for the patch shape.
        jitter_strength: Colour jitter intensity on the pasted patch.
            A bit of jitter makes the synthetic anomaly more realistic.
    """

    def __init__(
        self,
        p: float = 0.5,
        area_ratio: Tuple[float, float] = (0.02, 0.15),
        aspect_ratio: Tuple[float, float] = (0.3, 3.3),
        jitter_strength: float = 0.1,
    ) -> None:
        self.p = p
        self.area_ratio = area_ratio
        self.aspect_ratio = aspect_ratio
        self._jitter: Optional[transforms.ColorJitter] = None
        if jitter_strength > 0:
            self._jitter = transforms.ColorJitter(
                brightness=jitter_strength,
                contrast=jitter_strength,
                saturation=jitter_strength,
                hue=jitter_strength / 2,
            )

    def __call__(self, image: Image.Image) -> Tuple[Image.Image, int]:
        """Apply CutPaste with probability self.p.

        Args:
            image: PIL RGB image, already spatially cropped.
        Returns:
            (augmented_image, label) — label=1 if CutPaste was applied.
        """
        if random.random() > self.p:
            return image, 0

        img = image.copy()
        w, h = img.size
        area = w * h

        # sample patch size — retry a few times if sampling fails
        pw, ph = 0, 0
        for _ in range(10):
            patch_area = random.uniform(*self.area_ratio) * area
            ar = random.uniform(*self.aspect_ratio)
            pw = int(round((patch_area * ar) ** 0.5))
            ph = int(round((patch_area / ar) ** 0.5))
            if 0 < pw < w and 0 < ph < h:
                break
        else:
            return image, 0  # couldn't sample — skip

        x_src = random.randint(0, w - pw)
        y_src = random.randint(0, h - ph)
        patch = img.crop((x_src, y_src, x_src + pw, y_src + ph))

        if self._jitter is not None:
            patch = self._jitter(patch)

        x_dst = random.randint(0, w - pw)
        y_dst = random.randint(0, h - ph)
        img.paste(patch, (x_dst, y_dst))
        return img, 1


class CutPasteDataset(Dataset):
    """MVTec train split augmented with CutPaste synthetic anomalies.

    The base dataset must be the train split (all-normal by construction).
    We reload the PIL image directly — bypassing the base dataset's
    __getitem__ — so we can apply CutPaste before normalisation.
    Normalisation must happen after, not before, otherwise the pixel
    operations in CutPaste work on normalised values which breaks the
    colour jitter logic.

    Args:
        base_dataset: MVTecDataset with split="train".
        cutpaste: CutPasteTransform instance.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        cutpaste: CutPasteTransform,
    ) -> None:
        self.base     = base_dataset
        self.cutpaste = cutpaste

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        """Load PIL → spatial crop → CutPaste → normalise → tensor.

        Returns:
            Dict with keys: image [3, 224, 224], label int, category str.
        """
        img_path, _, _ = self.base.samples[idx]
        pil_img = Image.open(img_path).convert("RGB")
        pil_img = _SPATIAL(pil_img)                # resize + crop
        pil_img, label = self.cutpaste(pil_img)    # maybe cut-paste
        image_t: torch.Tensor = _TO_TENSOR(pil_img)

        return {
            "image":    image_t,
            "label":    label,
            "category": self.base.category,
        }
