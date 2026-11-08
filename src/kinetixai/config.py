"""User configuration and per-user directories.

Settings live in a small JSON file in the OS config directory; the paper
library and the local RAG index live in the OS data directory. Both come
from platformdirs, so they land in the right place on each OS:

  Linux    ~/.config/kinetixai          ~/.local/share/kinetixai
  macOS    ~/Library/Application Support/kinetixai (both)
  Windows  %LOCALAPPDATA%\\kinetixai (both)

KX_CONFIG_DIR and KX_DATA_DIR override them (handy for tests and for
keeping a project-specific library).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from platformdirs import user_config_dir, user_data_dir

APP = "kinetixai"
DEFAULTS: dict = {
    "endpoint": "https://kinetixai.ai",
    "limit": 10,
    "ollama_url": "http://localhost:11434",
    "ollama_model": "llama3.1",
    "embed_model": "nomic-embed-text",
}
INT_KEYS = {"limit"}


def config_dir() -> Path:
    env = os.environ.get("KX_CONFIG_DIR")
    return Path(env) if env else Path(user_config_dir(APP, appauthor=False))


def data_dir() -> Path:
    env = os.environ.get("KX_DATA_DIR")
    return Path(env) if env else Path(user_data_dir(APP, appauthor=False))


def config_path() -> Path:
    return config_dir() / "config.json"


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
    if os.environ.get("OLLAMA_HOST"):
        cfg["ollama_url"] = normalise_ollama(os.environ["OLLAMA_HOST"])
    return cfg


def normalise_ollama(host: str) -> str:
    """OLLAMA_HOST may be '0.0.0.0', '127.0.0.1:11434' or a full URL."""
    host = host.strip().rstrip("/")
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    scheme, rest = host.split("://", 1)
    if rest.startswith("0.0.0.0"):
        rest = "localhost" + rest[len("0.0.0.0"):]
    if ":" not in rest.split("/", 1)[0]:
        rest = rest + ":11434"
    return f"{scheme}://{rest}"


def save(cfg: dict) -> Path:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, json.dumps(cfg, indent=2) + "\n")
    return path


def get(key: str, default=None):
    return load().get(key, default)


def set_value(key: str, value) -> Path:
    path = config_path()
    stored: dict = {}
    try:
        if path.is_file():
            stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        stored = {}
    stored[key] = value
    return save(stored)


def write_text_atomic(path: Path, text: str) -> None:
    """Write via a temp file and os.replace, which is atomic on POSIX and Windows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
