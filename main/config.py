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

    # OpenAI-compatible translation API
    base_url: str = field(default_factory=lambda: os.getenv("BASE_URL", ""))
    api_key: str = field(default_factory=lambda: os.getenv("API_KEY", ""))
    model: str = field(default_factory=lambda: os.getenv("MODEL", ""))
    translation_max_concurrent_calls: int = field(
        default_factory=lambda: _int_env("TRANSLATION_MAX_CONCURRENT_CALLS", 3)
    )

    # OCR
    source_lang: str = "ja"  # "ja" for Japanese
    use_modal: bool = field(
        default_factory=lambda: _bool_env("USE_MODAL", False)
    )
    # Only used when USE_MODAL=true:
    # False -> Modal GPU OCR service, True -> Modal CPU OCR service.
    use_mangaocr_cpu: bool = field(
        default_factory=lambda: _bool_env("USE_MANGAOCR_CPU", False)
    )
    # True -> Modal detector service. False -> local ONNX detector.
    use_detection_model: bool = field(
        default_factory=lambda: _bool_env("USE_DETECTION_MODEL", False)
    )
    # Conservative default for Modal starter/free plan workloads.
    modal_max_parallel_pages: int = field(
        default_factory=lambda: _int_env("MODAL_MAX_PARALLEL_PAGES", 2)
    )
    # Parallel pages when running locally (USE_MODAL=false).
    # 2 is ideal for dual-core CPU + 16 GB RAM (overlaps I/O waits).
    local_max_parallel_pages: int = field(
        default_factory=lambda: _int_env("LOCAL_MAX_PARALLEL_PAGES", 2)
    )

    # Gradio UI auth gate for costly actions (translate / download).
    gradio_action_password: str = field(
        default_factory=lambda: os.getenv("GRADIO_ACTION_PASSWORD", "")
    )

    # Output
    output_dir: Path = field(default_factory=lambda: Path("./output"))

    @property
    def active_translation_model(self) -> str:
        """Return the configured translation model."""
        return self.model

    def set_active_translation_model(self, model: str) -> None:
        """Apply a CLI model override."""
        self.model = model

    def validate(self) -> None:
        """Raise ValueError if critical settings are missing."""
        missing = [
            name
            for name, value in (
                ("BASE_URL", self.base_url),
                ("API_KEY", self.api_key),
                ("MODEL", self.model),
            )
            if not value.strip()
        ]
        if missing:
            raise ValueError(
                f"{', '.join(missing)} must be set in .env or the environment."
            )
        if self.modal_max_parallel_pages < 1:
            raise ValueError("MODAL_MAX_PARALLEL_PAGES must be >= 1.")
        if self.local_max_parallel_pages < 1:
            raise ValueError("LOCAL_MAX_PARALLEL_PAGES must be >= 1.")


# Singleton
settings = Settings()
