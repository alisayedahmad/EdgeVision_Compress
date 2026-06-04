"""Tests for the project scaffold.

Verifies project structure correctness, package importability,
and the logging configuration utility.
"""
import importlib
import logging
from pathlib import Path


ROOT = Path(__file__).parent.parent


class TestProjectStructure:
    """Verify that required directories and files exist."""

    def test_required_directories_exist(self) -> None:
        """All required project directories must exist."""
        required = [
            "configs",
            "configs/model",
            "configs/compression",
            "src/models",
            "src/compression",
            "src/compression/pruning",
            "src/compression/quantization",
            "src/compression/distillation",
            "src/benchmark",
            "src/data",
            "src/utils",
            "notebooks",
            "demo",
            "tests",
            ".github/workflows",
        ]
        for d in required:
            assert (ROOT / d).is_dir(), f"Missing directory: {d}"

    def test_required_files_exist(self) -> None:
        """Base configuration and project files must exist."""
        required = [
            "configs/train.yaml",
            "configs/benchmark.yaml",
            "configs/model/resnet50.yaml",
            "configs/compression/pruning.yaml",
            "pyproject.toml",
            ".gitignore",
            "dvc.yaml",
            "README.md",
        ]
        for f in required:
            assert (ROOT / f).is_file(), f"Missing file: {f}"


class TestPackageImports:
    """Verify that all source packages are importable."""

    def test_all_packages_importable(self) -> None:
        """Every source package must be importable after pip install -e ."""
        packages = [
            "models",
            "compression",
            "compression.pruning",
            "compression.quantization",
            "compression.distillation",
            "benchmark",
            "data",
            "utils",
        ]
        for pkg in packages:
            mod = importlib.import_module(pkg)
            assert mod is not None, f"Cannot import: {pkg}"


class TestLoggingConfig:
    """Verify the logging configuration utility."""

    def test_returns_logger_instance(self) -> None:
        """setup_logging must return a logging.Logger."""
        from utils.logging_config import setup_logging
        logger = setup_logging(name="test_returns")
        assert isinstance(logger, logging.Logger)

    def test_correct_name(self) -> None:
        """Logger name must match the name argument."""
        from utils.logging_config import setup_logging
        logger = setup_logging(name="test_name_check")
        assert logger.name == "test_name_check"

    def test_default_level_is_info(self) -> None:
        """Default log level must be INFO."""
        from utils.logging_config import setup_logging
        logger = setup_logging(name="test_level_default")
        assert logger.level == logging.INFO

    def test_idempotent_no_duplicate_handlers(self) -> None:
        """Calling setup_logging twice must not add duplicate handlers."""
        from utils.logging_config import setup_logging
        logger = setup_logging(name="test_idempotent")
        n = len(logger.handlers)
        setup_logging(name="test_idempotent")
        assert len(logger.handlers) == n

    def test_file_handler_writes_log(self, tmp_path: Path) -> None:
        """Logger must write messages to file when log_file is provided."""
        from utils.logging_config import setup_logging
        log_file = tmp_path / "test.log"
        logger = setup_logging(name="test_file_write", log_file=log_file)
        logger.info("scaffold test message")
        assert log_file.exists()
        content = log_file.read_text(encoding="utf-8")
        assert "scaffold test message" in content
