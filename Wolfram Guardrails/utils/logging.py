"""Structured logging helpers."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def should_store_prompt_text() -> bool:
    env_value = os.getenv("STORE_PROMPT_TEXT")
    if env_value is not None:
        return env_value.strip().lower() in {"1", "true", "yes", "on"}

    settings_path = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"
    if settings_path.exists():
        with settings_path.open(encoding="utf-8") as handle:
            settings = yaml.safe_load(handle) or {}
        return bool(settings.get("logging", {}).get("store_prompt_text", False))
    return False
