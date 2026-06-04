"""Root pytest configuration for EdgeVision-Compress.

Adds src/ to sys.path so tests can import project modules
independently of package installation state.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
