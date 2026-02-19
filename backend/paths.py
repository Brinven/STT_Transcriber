"""
Centralized path management for the application.

Computes APP_ROOT and DATA_DIR based on whether the app is running
from source or as a PyInstaller-frozen executable. All modules should
import paths from here rather than constructing their own relative paths.
"""

import os
import sys


def _get_app_root() -> str:
    """Determine the application root directory."""
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller bundle — executable lives in the app root
        return os.path.dirname(sys.executable)
    else:
        # Running from source — this file is backend/paths.py, so go up one level
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


APP_ROOT = _get_app_root()
DATA_DIR = os.path.join(APP_ROOT, "data")
CONFIG_DIR = os.path.join(DATA_DIR, "config")
LOGS_DIR = os.path.join(DATA_DIR, "logs")
MODELS_DIR = os.path.join(DATA_DIR, "models")


def ensure_data_dirs() -> None:
    """Create required data directories if they don't exist."""
    os.makedirs(CONFIG_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
