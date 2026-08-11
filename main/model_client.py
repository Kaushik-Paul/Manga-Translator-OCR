"""OpenAI-compatible client for translation requests."""

from __future__ import annotations

from itertools import count
import logging
from threading import BoundedSemaphore
import time

import httpx

from .config import settings

logger = logging.getLogger(__name__)

_TRANSLATION_CALL_SEMAPHORE = BoundedSemaphore(
    value=max(1, settings.translation_max_concurrent_calls)
)
_TRANSLATION_CALL_COUNTER = count(1)
_TRANSLATION_LOCK_WAIT_POLL_SEC = 2.0
_TRANSLATION_LOCK_WAIT_LOG_EVERY_SEC = 20.0
_TRANSLATION_HTTP_TIMEOUT = httpx.Timeout(
    connect=15.0,
    read=120.0,
    write=30.0,
    pool=15.0,
)


def call_translation_model(
    model: str,
    user_message: str,
    system_prompt: str,
) -> str:
    """Send a translation request to the configured OpenAI-compatible API."""
    settings.validate()
    headers = {
        "Authorization": f"Bearer {settings.api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.3,
        "max_tokens": 4096,
        "reasoning": {"effort": "none"},
    }

    call_id = next(_TRANSLATION_CALL_COUNTER)
    call_started_at = time.monotonic()
    logger.debug("Calling translation API #%d (model: %s)...", call_id, model)
    slot_wait_sec = _acquire_translation_lock(call_id)

    try:
        logger.debug(
            "Translation API #%d slot acquired (wait %.2fs).",
            call_id,
            slot_wait_sec,
        )
        try:
            with httpx.Client(timeout=_TRANSLATION_HTTP_TIMEOUT) as client:
                response = client.post(
                    f"{settings.base_url.rstrip('/')}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
        except httpx.TimeoutException as error:
            elapsed = time.monotonic() - call_started_at
            raise TimeoutError(
                f"Translation API timeout for call #{call_id} after {elapsed:.2f}s "
                f"(slot wait {slot_wait_sec:.2f}s): {error}"
            ) from error
        except httpx.HTTPStatusError as error:
            response_preview = _truncate_for_log(error.response.text)
            raise RuntimeError(
                f"Translation API HTTP {error.response.status_code} for call "
                f"#{call_id}: {response_preview or '<empty response body>'}"
            ) from error
        except httpx.RequestError as error:
            elapsed = time.monotonic() - call_started_at
            raise RuntimeError(
                f"Translation API request error for call #{call_id} after "
                f"{elapsed:.2f}s: {error}"
            ) from error
    finally:
        _TRANSLATION_CALL_SEMAPHORE.release()

    try:
        data = response.json()
    except ValueError as error:
        body_preview = _truncate_for_log(response.text)
        raise ValueError(
            f"Translation API returned invalid JSON for call #{call_id}: "
            f"{body_preview or '<empty body>'}"
        ) from error

    choices = data.get("choices", [])
    if not choices:
        raise ValueError("No choices returned from translation API.")

    message = choices[0].get("message", {})
    text = _coerce_message_content(message.get("content", "")).strip()
    if not text:
        raise ValueError("Translation API returned empty translation content.")

    logger.debug(
        "Translation received from API #%d (%d chars, %.2fs total).",
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


def _acquire_translation_lock(call_id: int) -> float:
    """Wait for a translation concurrency slot, logging periodically while queued."""
    wait_started_at = time.monotonic()
    next_log_after_sec = _TRANSLATION_LOCK_WAIT_LOG_EVERY_SEC

    while True:
        if _TRANSLATION_CALL_SEMAPHORE.acquire(timeout=_TRANSLATION_LOCK_WAIT_POLL_SEC):
            return time.monotonic() - wait_started_at

        waited_sec = time.monotonic() - wait_started_at
        if waited_sec >= next_log_after_sec:
            logger.warning(
                "Translation API #%d still waiting on concurrency slot (%.2fs).",
                call_id,
                waited_sec,
            )
            next_log_after_sec += _TRANSLATION_LOCK_WAIT_LOG_EVERY_SEC


def _coerce_message_content(content: object) -> str:
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
            value = block.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
                break
    return "\n".join(parts)
