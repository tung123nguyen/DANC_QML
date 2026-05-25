"""Path management.

Reads DATA_ROOT and RESULTS_ROOT from environment variables.
Falls back to local relative paths if not set.

This lets the SAME CODE run on:
- Local: reads/writes ./data, ./results
- Colab: reads/writes Google Drive paths via env vars

Set in Colab notebook:
    os.environ['DATA_ROOT'] = '/content/drive/MyDrive/qnn-ids-thesis/data'
    os.environ['RESULTS_ROOT'] = '/content/drive/MyDrive/qnn-ids-thesis/results'
"""
from __future__ import annotations
import os
import json
import subprocess
from pathlib import Path
from typing import Any


def get_data_root() -> Path:
    return Path(os.environ.get("DATA_ROOT", "./data"))


def get_results_root() -> Path:
    return Path(os.environ.get("RESULTS_ROOT", "./results"))


def get_git_commit() -> str:
    """Return current git commit hash, or 'unknown' if not in git repo."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def save_json(obj: Any, path: Path) -> None:
    """Save dict/list to JSON with pretty formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)
