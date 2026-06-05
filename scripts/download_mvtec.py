"""Download and extract the MVTec Anomaly Detection dataset.

Free for non-commercial use. The full archive is ~5 GB.

Usage:
    python scripts/download_mvtec.py
    python scripts/download_mvtec.py --output data/mvtec
    python scripts/download_mvtec.py --skip-download   # extract only
"""
import argparse
import logging
import sys
import tarfile
import urllib.request
from pathlib import Path

# make sure src/ is on the path when called from the project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.logging_config import setup_logging

logger = setup_logging(name="download")

MVTEC_URL = (
    "https://www.mvtec.com/fileadmin/Datasets/mvtec_anomaly_detection.tar.gz"
)

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]


class _ProgressReporter:
    """Log download progress every 5%."""

    def __init__(self) -> None:
        self._last_pct = -1

    def __call__(self, block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        pct = min(int(block_num * block_size * 100 / total_size), 100)
        if pct != self._last_pct and pct % 5 == 0:
            logger.info("  %d%%", pct)
            self._last_pct = pct


def download_archive(output_dir: Path) -> Path:
    """Download the MVTec AD tar.gz archive.

    Args:
        output_dir: Where to save the archive.

    Returns:
        Path to the downloaded file.

    Raises:
        RuntimeError: If the download fails (includes manual fallback instructions).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "mvtec_anomaly_detection.tar.gz"

    if archive_path.exists():
        logger.info("Archive already present at %s — skipping download.", archive_path)
        return archive_path

    logger.info("Downloading MVTec AD (~5 GB) from %s", MVTEC_URL)

    try:
        urllib.request.urlretrieve(MVTEC_URL, archive_path, _ProgressReporter())
    except Exception as exc:
        archive_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Download failed: {exc}\n\n"
            "Manual fallback:\n"
            f"  1. Open: {MVTEC_URL}\n"
            f"  2. Save the file to: {archive_path}\n"
            "  3. Re-run this script with --skip-download"
        ) from exc

    logger.info("Download complete: %s", archive_path)
    return archive_path


def extract_archive(archive_path: Path, output_dir: Path) -> None:
    """Extract the tar.gz archive into output_dir.

    Args:
        archive_path: Path to the .tar.gz file.
        output_dir: Destination directory.
    """
    logger.info("Extracting to %s ...", output_dir)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=output_dir)
    logger.info("Extraction complete.")


def verify_categories(output_dir: Path) -> bool:
    """Check that all 15 expected category folders are present.

    Args:
        output_dir: The directory where MVTec was extracted.

    Returns:
        True if all categories are found.
    """
    missing = [c for c in CATEGORIES if not (output_dir / c).is_dir()]
    if missing:
        logger.warning("Missing categories after extraction: %s", missing)
        return False
    logger.info("All 15 categories verified in %s", output_dir)
    return True


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Download MVTec AD dataset.")
    parser.add_argument(
        "--output", type=Path, default=Path("data/mvtec"),
        help="Target directory (default: data/mvtec)",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip download and go straight to extraction.",
    )
    args = parser.parse_args()

    if not args.skip_download:
        try:
            archive_path = download_archive(args.output)
        except RuntimeError as exc:
            logger.error(str(exc))
            sys.exit(1)
    else:
        archive_path = args.output / "mvtec_anomaly_detection.tar.gz"
        if not archive_path.exists():
            logger.error("Archive not found at %s", archive_path)
            sys.exit(1)

    extract_archive(archive_path, args.output)
    verify_categories(args.output)

    logger.info("Done. Next: dvc add %s", args.output)


if __name__ == "__main__":
    main()
