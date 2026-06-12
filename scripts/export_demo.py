"""Prepare the demo folder for deployment.

Copies the student checkpoint and sample images into demo/
so the folder is self-contained for HuggingFace Spaces.

Usage:
    python scripts/export_demo.py
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DEMO = ROOT / "demo"
EXAMPLES = DEMO / "examples"


def main() -> None:
    # copy checkpoint
    src = ROOT / "outputs" / "checkpoints" / "student_distilled.pth"
    dst = DEMO / "student_distilled.pth"
    if src.exists():
        shutil.copy2(src, dst)
        print(f"Checkpoint copied: {dst}")
    else:
        print(f"WARNING: {src} not found. Demo will not work without it.")

    # create examples dir with sample images from test set
    EXAMPLES.mkdir(exist_ok=True)
    test_dir = ROOT / "data" / "mvtec" / "bottle" / "test"
    if test_dir.exists():
        count = 0
        for defect_dir in sorted(test_dir.iterdir()):
            if not defect_dir.is_dir():
                continue
            images = sorted(defect_dir.glob("*.png"))
            if images:
                dst_name = f"{defect_dir.name}_{images[0].name}"
                shutil.copy2(images[0], EXAMPLES / dst_name)
                count += 1
        print(f"Copied {count} example images to {EXAMPLES}")
    else:
        print(f"WARNING: {test_dir} not found. No example images copied.")

    print("\nDemo ready. Launch with:")
    print("  pip install gradio")
    print("  python demo/app.py")


if __name__ == "__main__":
    main()