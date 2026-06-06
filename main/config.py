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

    # Translation provider
    use_openrouter: bool = field(
        default_factory=lambda: _bool_env("USE_OPENROUTER", False)
    )

    # OpenRouter
    openrouter_api_key: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY", "")
    )
    openrouter_model: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat")
    )
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_max_concurrent_calls: int = field(
        default_factory=lambda: _int_env("OPENROUTER_MAX_CONCURRENT_CALLS", 3)
    )

    # OpenCode Go
    opencode_go_api_key: str = field(
        default_factory=lambda: os.getenv("OPENCODE_GO_API_KEY", "")
    )
    opencode_go_model: str = field(
        default_factory=lambda: os.getenv("OPENCODE_GO_MODEL", "deepseek-v4-flash")
    )
    opencode_go_api_style: str = field(
        default_factory=lambda: os.getenv("OPENCODE_GO_API_STYLE", "auto")
    )
    opencode_go_openai_base_url: str = "https://opencode.ai/zen/go/v1"
    opencode_go_anthropic_base_url: str = "https://opencode.ai/zen/go/v1"

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
    def translation_provider(self) -> str:
        """Return the active LLM translation provider name."""
        return "openrouter" if self.use_openrouter else "opencode-go"

    @property
    def active_translation_model(self) -> str:
        """Return the model used by the active translation provider."""
        return self.openrouter_model if self.use_openrouter else self.opencode_go_model

    def set_active_translation_model(self, model: str) -> None:
        """Apply a CLI model override to the active translation provider."""
        if self.use_openrouter:
            self.openrouter_model = model
        else:
            self.opencode_go_model = model

    def validate(self) -> None:
        """Raise ValueError if critical settings are missing."""
        if self.use_openrouter and not self.openrouter_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is required when USE_OPENROUTER=true. "
                "Set it in .env or pass via environment."
            )
        if not self.use_openrouter and not self.opencode_go_api_key:
            raise ValueError(
                "OPENCODE_GO_API_KEY is required when USE_OPENROUTER=false. "
                "Set it in .env or pass via environment."
            )
        if self.opencode_go_api_style.strip().lower() not in {
            "auto",
            "openai",
            "anthropic",
        }:
            raise ValueError("OPENCODE_GO_API_STYLE must be auto, openai, or anthropic.")
        if self.modal_max_parallel_pages < 1:
            raise ValueError("MODAL_MAX_PARALLEL_PAGES must be >= 1.")
        if self.local_max_parallel_pages < 1:
            raise ValueError("LOCAL_MAX_PARALLEL_PAGES must be >= 1.")


# Singleton
settings = Settings()
