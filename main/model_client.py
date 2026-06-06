"""
LLM provider client for translation requests.

Routes translation prompts to OpenCode Go by default or OpenRouter when
USE_OPENROUTER=true.
"""

from __future__ import annotations

from itertools import count
import logging
from threading import BoundedSemaphore
import time

import httpx

from .config import settings

logger = logging.getLogger(__name__)

_TRANSLATION_CALL_SEMAPHORE = BoundedSemaphore(
    value=max(1, int(getattr(settings, "translation_max_concurrent_calls", 1)))
)
_TRANSLATION_CALL_COUNTER = count(1)
_TRANSLATION_LOCK_WAIT_POLL_SEC = 2.0
_TRANSLATION_LOCK_WAIT_LOG_EVERY_SEC = 20.0
_TRANSLATION_HTTP_TIMEOUT = httpx.Timeout(connect=15.0, read=120.0, write=30.0, pool=15.0)
_OPENCODE_GO_ANTHROPIC_MODELS = {"minimax-m2.7", "qwen3.5-plus", "qwen3.6-plus"}


def call_translation_model(
    model: str,
    user_message: str,
    system_prompt: str,
    session_id: str | None = None,
) -> str:
    """Dispatch a translation request to the configured LLM provider."""
    if settings.use_openrouter:
        return _call_openrouter(
            model=model,
            user_message=user_message,
            system_prompt=system_prompt,
            session_id=session_id,
        )
    return _call_opencode_go(
        model=model,
        user_message=user_message,
        system_prompt=system_prompt,
        session_id=session_id,
    )


def _call_openrouter(
    model: str,
    user_message: str,
    system_prompt: str,
    session_id: str | None = None,
) -> str:
    """Make a single API call to OpenRouter."""
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/manga-translator-ocr",
        "X-Title": "Manga Translator OCR",
    }
    if session_id:
        headers["X-Session-Id"] = session_id

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.3,
        "max_tokens": 4096,
        "reasoning": {
            "effort": "none",
        },
        "include_reasoning": False,
    }

    call_id = next(_TRANSLATION_CALL_COUNTER)
    call_started_at = time.monotonic()
    logger.info("Calling OpenRouter API #%d (model: %s)...", call_id, model)
    slot_wait_sec = _acquire_translation_lock("OpenRouter", call_id)

    try:
        logger.info(
            "OpenRouter API #%d slot acquired (wait %.2fs).", call_id, slot_wait_sec
        )
        try:
            with httpx.Client(timeout=_TRANSLATION_HTTP_TIMEOUT) as client:
                response = client.post(
                    f"{settings.openrouter_base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
        except httpx.TimeoutException as e:
            elapsed = time.monotonic() - call_started_at
            raise TimeoutError(
                f"OpenRouter API timeout for call #{call_id} after {elapsed:.2f}s "
                f"(slot wait {slot_wait_sec:.2f}s): {e}"
            ) from e
        except httpx.HTTPStatusError as e:
            response_preview = _truncate_for_log(e.response.text)
            status_code = e.response.status_code
            raise RuntimeError(
                f"OpenRouter API HTTP {status_code} for call #{call_id}: "
                f"{response_preview or '<empty response body>'}"
            ) from e
        except httpx.RequestError as e:
            elapsed = time.monotonic() - call_started_at
            raise RuntimeError(
                f"OpenRouter API request error for call #{call_id} after {elapsed:.2f}s: {e}"
            ) from e
    finally:
        _TRANSLATION_CALL_SEMAPHORE.release()

    try:
        data = response.json()
    except ValueError as e:
        body_preview = _truncate_for_log(response.text)
        raise ValueError(
            f"OpenRouter returned invalid JSON for call #{call_id}: "
            f"{body_preview or '<empty body>'}"
        ) from e

    choices = data.get("choices", [])
    if not choices:
        raise ValueError("No choices returned from OpenRouter API")

    message = choices[0].get("message", {})
    text = _coerce_openai_content(message.get("content", "")).strip()
    if not text:
        raise ValueError("OpenRouter returned empty translation content.")
    logger.info(
        "Translation received from OpenRouter API #%d (%d chars, %.2fs total).",
        call_id,
        len(text),
        time.monotonic() - call_started_at,
    )
    return text


def _call_opencode_go(
    model: str,
    user_message: str,
    system_prompt: str,
    session_id: str | None = None,
) -> str:
    """Make a single API call to OpenCode Go."""
    model_id = _opencode_go_model_id(model)
    api_style = _opencode_go_api_style(model_id)
    if api_style == "anthropic":
        return _call_opencode_go_anthropic(
            model=model_id,
            user_message=user_message,
            system_prompt=system_prompt,
            session_id=session_id,
        )
    return _call_opencode_go_openai(
        model=model_id,
        user_message=user_message,
        system_prompt=system_prompt,
        session_id=session_id,
    )


def _call_opencode_go_openai(
    model: str,
    user_message: str,
    system_prompt: str,
    session_id: str | None,
) -> str:
    """Call an OpenCode Go OpenAI-compatible model."""
    headers = {
        "Authorization": f"Bearer {settings.opencode_go_api_key}",
        "Content-Type": "application/json",
    }
    if session_id:
        headers["X-Session-Id"] = session_id

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.3,
        "max_tokens": 4096,
        "reasoning": {
            "effort": "none",
        },
        "thinking": {
            "type": "disabled",
        },
        "include_reasoning": False,
    }

    call_id = next(_TRANSLATION_CALL_COUNTER)
    call_started_at = time.monotonic()
    logger.info(
        "Calling OpenCode Go API #%d (model: %s, style: openai)...",
        call_id,
        model,
    )
    slot_wait_sec = _acquire_translation_lock("OpenCode Go", call_id)

    try:
        logger.info(
            "OpenCode Go API #%d slot acquired (wait %.2fs).",
            call_id,
            slot_wait_sec,
        )
        try:
            with httpx.Client(timeout=_TRANSLATION_HTTP_TIMEOUT) as client:
                response = client.post(
                    f"{settings.opencode_go_openai_base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
        except httpx.TimeoutException as e:
            elapsed = time.monotonic() - call_started_at
            raise TimeoutError(
                f"OpenCode Go API timeout for call #{call_id} after {elapsed:.2f}s "
                f"(slot wait {slot_wait_sec:.2f}s): {e}"
            ) from e
        except httpx.HTTPStatusError as e:
            response_preview = _truncate_for_log(e.response.text)
            status_code = e.response.status_code
            raise RuntimeError(
                f"OpenCode Go API HTTP {status_code} for call #{call_id}: "
                f"{response_preview or '<empty response body>'}"
            ) from e
        except httpx.RequestError as e:
            elapsed = time.monotonic() - call_started_at
            raise RuntimeError(
                f"OpenCode Go API request error for call #{call_id} "
                f"after {elapsed:.2f}s: {e}"
            ) from e
    finally:
        _TRANSLATION_CALL_SEMAPHORE.release()

    try:
        data = response.json()
    except ValueError as e:
        body_preview = _truncate_for_log(response.text)
        raise ValueError(
            f"OpenCode Go returned invalid JSON for call #{call_id}: "
            f"{body_preview or '<empty body>'}"
        ) from e

    choices = data.get("choices", [])
    if not choices:
        raise ValueError("No choices returned from OpenCode Go API")

    message = choices[0].get("message", {})
    text = _coerce_openai_content(message.get("content", "")).strip()
    if not text:
        raise ValueError("OpenCode Go returned empty translation content.")
    logger.info(
        "Translation received from OpenCode Go API #%d (%d chars, %.2fs total).",
        call_id,
        len(text),
        time.monotonic() - call_started_at,
    )
    return text


def _call_opencode_go_anthropic(
    model: str,
    user_message: str,
    system_prompt: str,
    session_id: str | None,
) -> str:
    """Call an OpenCode Go Anthropic-compatible model."""
    headers = {
        "Authorization": f"Bearer {settings.opencode_go_api_key}",
        "X-Api-Key": settings.opencode_go_api_key,
        "Anthropic-Version": "2023-06-01",
        "Content-Type": "application/json",
    }
    if session_id:
        headers["X-Session-Id"] = session_id

    payload = {
        "model": model,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.3,
        "max_tokens": 4096,
    }

    call_id = next(_TRANSLATION_CALL_COUNTER)
    call_started_at = time.monotonic()
    logger.info(
        "Calling OpenCode Go API #%d (model: %s, style: anthropic)...",
        call_id,
        model,
    )
    slot_wait_sec = _acquire_translation_lock("OpenCode Go", call_id)

    try:
        logger.info(
            "OpenCode Go API #%d slot acquired (wait %.2fs).",
            call_id,
            slot_wait_sec,
        )
        try:
            with httpx.Client(timeout=_TRANSLATION_HTTP_TIMEOUT) as client:
                response = client.post(
                    f"{settings.opencode_go_anthropic_base_url}/messages",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
        except httpx.TimeoutException as e:
            elapsed = time.monotonic() - call_started_at
            raise TimeoutError(
                f"OpenCode Go API timeout for call #{call_id} after {elapsed:.2f}s "
                f"(slot wait {slot_wait_sec:.2f}s): {e}"
            ) from e
        except httpx.HTTPStatusError as e:
            response_preview = _truncate_for_log(e.response.text)
            status_code = e.response.status_code
            raise RuntimeError(
                f"OpenCode Go API HTTP {status_code} for call #{call_id}: "
                f"{response_preview or '<empty response body>'}"
            ) from e
        except httpx.RequestError as e:
            elapsed = time.monotonic() - call_started_at
            raise RuntimeError(
                f"OpenCode Go API request error for call #{call_id} "
                f"after {elapsed:.2f}s: {e}"
            ) from e
    finally:
        _TRANSLATION_CALL_SEMAPHORE.release()

    try:
        data = response.json()
    except ValueError as e:
        body_preview = _truncate_for_log(response.text)
        raise ValueError(
            f"OpenCode Go returned invalid JSON for call #{call_id}: "
            f"{body_preview or '<empty body>'}"
        ) from e

    text = _coerce_anthropic_content(data.get("content", "")).strip()
    if not text:
        raise ValueError("OpenCode Go returned empty translation content.")
    logger.info(
        "Translation received from OpenCode Go API #%d (%d chars, %.2fs total).",
        call_id,
        len(text),
        time.monotonic() - call_started_at,
    )
    return text


def _truncate_for_log(text: str, limit: int = 400) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def _opencode_go_model_id(model: str) -> str:
    """Normalize an OpenCode Go model id for direct API calls."""
    return model.removeprefix("opencode-go/").strip()


def _opencode_go_api_style(model: str) -> str:
    """Return the OpenCode Go API style for the configured model."""
    configured = settings.opencode_go_api_style.strip().lower()
    if configured in {"openai", "anthropic"}:
        return configured
    if configured != "auto":
        raise RuntimeError("OPENCODE_GO_API_STYLE must be auto, openai, or anthropic")

    model_id = _opencode_go_model_id(model).lower()
    if model_id in _OPENCODE_GO_ANTHROPIC_MODELS:
        return "anthropic"
    return "openai"


def _acquire_translation_lock(provider: str, call_id: int) -> float:
    """Wait for a translation concurrency slot, logging periodically while queued."""
    wait_started_at = time.monotonic()
    next_log_after_sec = _TRANSLATION_LOCK_WAIT_LOG_EVERY_SEC

    while True:
        if _TRANSLATION_CALL_SEMAPHORE.acquire(timeout=_TRANSLATION_LOCK_WAIT_POLL_SEC):
            return time.monotonic() - wait_started_at

        waited_sec = time.monotonic() - wait_started_at
        if waited_sec >= next_log_after_sec:
            logger.warning(
                "%s API #%d still waiting on concurrency slot (%.2fs).",
                provider,
                call_id,
                waited_sec,
            )
            next_log_after_sec += _TRANSLATION_LOCK_WAIT_LOG_EVERY_SEC


def _coerce_openai_content(content: object) -> str:
    """Normalize OpenAI-compatible message content to plain text."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            if block.strip():
                parts.append(block.strip())
            continue
        if not isinstance(block, dict):
            continue
        for key in ("text", "output_text", "content"):
            text_value = block.get(key)
            if isinstance(text_value, str) and text_value.strip():
                parts.append(text_value.strip())
                break
    return "\n".join(parts)


def _coerce_anthropic_content(content: object) -> str:
    """Normalize Anthropic-style message content to plain text."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            if block.strip():
                parts.append(block.strip())
            continue
        if not isinstance(block, dict):
            continue
        for key in ("text", "output_text", "content"):
            text_value = block.get(key)
            if isinstance(text_value, str) and text_value.strip():
                parts.append(text_value.strip())
                break
    return "\n".join(parts)
