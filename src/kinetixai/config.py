"""User configuration: endpoint and defaults, stored as JSON."""
from __future__ import annotations

import json
import os
from pathlib import Path

from platformdirs import user_config_dir

APP = "kinetixai"
DEFAULTS = {
    "endpoint": "https://kinetixai.ai",
    "limit": 10,
}


def config_path() -> Path:
    return Path(user_config_dir(APP)) / "config.json"


def load() -> dict:
    cfg = dict(DEFAULTS)
    path = config_path()
    try:
        if path.is_file():
            cfg.update(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        pass  # a broken config falls back to defaults
    # environment wins, so CI and scripts can override without writing a file
    if os.environ.get("KINETIXAI_ENDPOINT"):
        cfg["endpoint"] = os.environ["KINETIXAI_ENDPOINT"]
    return cfg


def save(cfg: dict) -> Path:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return path


def get(key: str, default=None):
    return load().get(key, default)


def set_value(key: str, value) -> Path:
    cfg = load()
    cfg[key] = value
    return save(cfg)
