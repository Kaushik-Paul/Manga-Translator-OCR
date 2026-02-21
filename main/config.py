"""
Configuration module - loads settings from .env file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


def _int_env(name: str, default: int) -> int:
    """Read a positive integer from env, falling back to default on invalid values."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _bool_env(name: str, default: bool) -> bool:
    """Read a boolean from env with a sane fallback."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("true", "1", "yes", "on")


@dataclass
class Settings:
    """Application settings loaded from environment and CLI overrides."""

    # OpenRouter
    openrouter_api_key: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY", "")
    )
    # Default to deepseek which handles NSFW text without moderation issues
    translation_model: str = field(
        default_factory=lambda: os.getenv(
            "TRANSLATION_MODEL", "deepseek/deepseek-chat"
        )
    )
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # OCR
    source_lang: str = "ja"  # "ja" for Japanese
    use_modal: bool = field(
        default_factory=lambda: os.getenv("USE_MODAL", "true").lower()
        in ("true", "1", "yes")
    )
    # True -> Modal detector service. False -> local ONNX detector.
    use_detection_model: bool = field(
        default_factory=lambda: _bool_env("USE_DETECTION_MODEL", True)
    )
    # Conservative default for Modal starter/free plan workloads.
    modal_max_parallel_pages: int = field(
        default_factory=lambda: _int_env("MODAL_MAX_PARALLEL_PAGES", 2)
    )

    # Output
    output_dir: Path = field(default_factory=lambda: Path("./output"))

    def validate(self) -> None:
        """Raise ValueError if critical settings are missing."""
        if not self.openrouter_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is required. Set it in .env or pass via environment."
            )
        if self.source_lang != "ja":
            raise ValueError(
                f"Unsupported source language '{self.source_lang}'. Use 'ja'."
            )
        if self.modal_max_parallel_pages < 1:
            raise ValueError("MODAL_MAX_PARALLEL_PAGES must be >= 1.")


# Singleton
settings = Settings()
