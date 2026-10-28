"""
Automated smoke and structural tests for the Streamlit web application.
"""

import py_compile
import subprocess
import sys
from pathlib import Path
import pytest

import config
from app import load_json_safe


def test_app_python_compilation():
    """Verifies that app.py compiles cleanly with no syntax errors."""
    app_file = config.BASE_DIR / "app.py"
    assert app_file.exists()
    # py_compile will raise py_compile.PyCompileError on syntax error
    py_compile.compile(str(app_file), doraise=True)


def test_load_json_safe_behavior(tmp_path):
    """Verifies that load_json_safe safely returns None for missing or corrupted files."""
    missing = tmp_path / "missing.json"
    assert load_json_safe(missing) is None

    bad_json = tmp_path / "bad.json"
    with open(bad_json, "w") as f:
        f.write("{invalid json content")
    assert load_json_safe(bad_json) is None


def test_all_src_modules_compile():
    """Verifies that all Python files inside src/ compile without syntax errors."""
    src_dir = config.BASE_DIR / "src"
    for py_file in src_dir.glob("*.py"):
        py_compile.compile(str(py_file), doraise=True)


def test_config_paths_and_attributes():
    """Verifies that config defines all required paths and constants."""
    assert hasattr(config, "BASE_DIR")
    assert hasattr(config, "IMAGE_SIZE")
    assert hasattr(config, "CLASS_NAMES")
    assert hasattr(config, "CLASS_TO_IDX")
    assert hasattr(config, "IDX_TO_CLASS")
    assert config.IMAGE_SIZE == (224, 224)
    assert "normal" in config.CLASS_NAMES
    assert "defective" in config.CLASS_NAMES
