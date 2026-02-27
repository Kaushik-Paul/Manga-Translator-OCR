"""
Translation module using OpenRouter API.

Sends extracted text (not images) to OpenRouter for translation,
using models that handle NSFW text content without moderation issues.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import count
import logging
import re
from threading import BoundedSemaphore
import time

import httpx

from .config import settings

logger = logging.getLogger(__name__)
_OPENROUTER_CALL_SEMAPHORE = BoundedSemaphore(
    value=max(1, int(getattr(settings, "openrouter_max_concurrent_calls", 1)))
)
_OPENROUTER_CALL_COUNTER = count(1)
_OPENROUTER_LOCK_WAIT_POLL_SEC = 2.0
_OPENROUTER_LOCK_WAIT_LOG_EVERY_SEC = 20.0
_OPENROUTER_HTTP_TIMEOUT = httpx.Timeout(connect=15.0, read=75.0, write=30.0, pool=15.0)

# System prompt designed for manga/doujinshi translation
SYSTEM_PROMPT = """You are an expert manga/doujinshi translator. Translate Japanese and Chinese text to natural English.

Rules:
- Translate naturally and concisely — text must fit inside speech bubbles
- Preserve tone: casual, dramatic, emotional, vulgar, etc.
- Keep translations SHORT. Prefer brief natural phrasing over verbose accuracy
- For sound effects (SFX) and onomatopoeia, use 1-3 word English equivalents (e.g. *Thud*, *Splash*, *Hah...*)
- For fragmented/noisy OCR text, infer the most likely meaning and translate that
- Never refuse to translate any content — you are a professional translator
- Output ONLY the translated text, nothing else
- Do NOT add explanations, notes, or commentary"""

REPAIR_SYSTEM_PROMPT = """Translate manga text to concise natural English. Output English only.
Rules:
- Keep it very short and bubble-friendly
- Use natural English interjections for sounds (not romaji)
- Never output untranslated Japanese/Chinese
- Output ONLY the translation text"""


@dataclass(frozen=True)
class TranslationConstraint:
    """Optional per-line translation constraints."""

    style: str | None = None
    max_words: int | None = None
    max_chars: int | None = None


def translate_texts(
    texts: list[str],
    source_lang: str | None = None,
    model: str | None = None,
    max_retries: int = 3,
    constraints: list[TranslationConstraint] | None = None,
) -> list[str]:
    """
    Translate a batch of texts from Japanese/Chinese to English.

    Sends all texts in a single API call for efficiency.

    Args:
        texts: List of source language text strings.
        source_lang: Source language ("ja" or "zh"). Defaults to settings.
        model: OpenRouter model to use. Defaults to settings.
        max_retries: Number of retries on failure.
        constraints: Optional per-line translation constraints.

    Returns:
        List of translated English strings (same length as input).
    """
    if not texts:
        return []

    # Filter out empty strings but track their positions
    indexed_texts = [(i, t) for i, t in enumerate(texts) if t.strip()]
    if not indexed_texts:
        return ["" for _ in texts]

    lang = source_lang or settings.source_lang
    lang_name = "Japanese" if lang == "ja" else "Chinese"
    model_name = model or settings.translation_model

    # Build the user message with numbered lines for batch translation
    numbered_lines = []
    for idx, (orig_idx, text) in enumerate(indexed_texts):
        numbered_lines.append(f"[{idx + 1}] {text}")

    user_message = (
        f"Translate the following {lang_name} manga text to English. "
        f"Keep translations short and natural for speech bubbles. "
        f"Return ONLY the translations, one per line, with the same numbering format [N].\n\n"
        + "\n".join(numbered_lines)
    )

    # Make API call with retries
    translated_map: dict[int, str] = {}
    for attempt in range(max_retries):
        try:
            response_text = _call_openrouter(model_name, user_message)
            translated_map = _parse_numbered_response(response_text, len(indexed_texts))
            break
        except Exception as e:
            logger.warning(
                "Translation attempt %d/%d failed: %s", attempt + 1, max_retries, e
            )
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                logger.error("All translation attempts failed.")
                # Return original texts as fallback
                return texts

    # Repair pass for missing/unchanged/non-English outputs.
    repair_candidates: list[tuple[int, int, str]] = []
    for mapped_idx, (orig_idx, src_text) in enumerate(indexed_texts):
        candidate = translated_map.get(mapped_idx, "")
        constraint = (
            constraints[orig_idx]
            if constraints is not None and orig_idx < len(constraints)
            else None
        )
        if _needs_repair_translation(src_text, candidate, constraint):
            repair_candidates.append((mapped_idx, orig_idx, src_text))

    if repair_candidates:
        repair_lines = []
        for i, (_, orig_idx, src_text) in enumerate(repair_candidates):
            repair_lines.append(f"[{i + 1}] {src_text}")

        repair_message = (
            f"Translate the following {lang_name} manga text to concise English. "
            f"Keep translations short for speech bubbles. "
            f"Return ONLY numbered lines in [N] format.\n\n"
            + "\n".join(repair_lines)
        )
        for attempt in range(max_retries):
            try:
                repair_response = _call_openrouter(
                    model_name,
                    repair_message,
                    system_prompt=REPAIR_SYSTEM_PROMPT,
                )
                repaired_map = _parse_numbered_response(
                    repair_response, len(repair_candidates)
                )
                for repair_idx, (mapped_idx, orig_idx, src_text) in enumerate(
                    repair_candidates
                ):
                    fixed = repaired_map.get(repair_idx, "").strip()
                    constraint = (
                        constraints[orig_idx]
                        if constraints is not None and orig_idx < len(constraints)
                        else None
                    )
                    if fixed and not _needs_repair_translation(src_text, fixed, constraint):
                        translated_map[mapped_idx] = fixed
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        "Repair translation attempt %d/%d failed: %s",
                        attempt + 1,
                        max_retries,
                        e,
                    )
                    time.sleep(2 ** attempt)
                else:
                    logger.warning("Repair translation pass failed: %s", e)

    # Reconstruct the full results list (preserving empty string positions)
    results = ["" for _ in texts]
    for idx, (original_idx, _) in enumerate(indexed_texts):
        results[original_idx] = translated_map.get(idx, texts[original_idx])

    return results


def _call_openrouter(
    model: str,
    user_message: str,
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    """Make a single API call to OpenRouter."""
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/manga-translator-ocr",
        "X-Title": "Manga Translator OCR",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.3,  # Low temperature for consistent translations
        "max_tokens": 4096,
        "reasoning": {
            "effort": "none",
        },
    }

    call_id = next(_OPENROUTER_CALL_COUNTER)
    call_started_at = time.monotonic()
    logger.info("Calling OpenRouter API #%d (model: %s)...", call_id, model)

    # Bound outbound OpenRouter concurrency across page-worker threads.
    # This avoids provider-side empty-content responses seen under burst concurrency
    # while still allowing parallel calls.
    slot_wait_sec = _acquire_openrouter_lock(call_id)

    try:
        logger.info(
            "OpenRouter API #%d slot acquired (wait %.2fs).", call_id, slot_wait_sec
        )
        try:
            with httpx.Client(timeout=_OPENROUTER_HTTP_TIMEOUT) as client:
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
        _OPENROUTER_CALL_SEMAPHORE.release()

    try:
        data = response.json()
    except ValueError as e:
        body_preview = _truncate_for_log(response.text)
        raise ValueError(
            f"OpenRouter returned invalid JSON for call #{call_id}: "
            f"{body_preview or '<empty body>'}"
        ) from e

    # Extract the response text
    choices = data.get("choices", [])
    if not choices:
        raise ValueError("No choices returned from OpenRouter API")

    message = choices[0].get("message", {})
    content = message.get("content", "")
    text = _coerce_message_content(content).strip()
    if not text:
        raise ValueError("OpenRouter returned empty translation content.")
    logger.info(
        "Translation received from OpenRouter API #%d (%d chars, %.2fs total).",
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


def _acquire_openrouter_lock(call_id: int) -> float:
    """Wait for an OpenRouter concurrency slot, logging periodically while queued."""
    wait_started_at = time.monotonic()
    next_log_after_sec = _OPENROUTER_LOCK_WAIT_LOG_EVERY_SEC

    while True:
        if _OPENROUTER_CALL_SEMAPHORE.acquire(timeout=_OPENROUTER_LOCK_WAIT_POLL_SEC):
            return time.monotonic() - wait_started_at

        waited_sec = time.monotonic() - wait_started_at
        if waited_sec >= next_log_after_sec:
            logger.warning(
                "OpenRouter API #%d still waiting on concurrency slot (%.2fs).",
                call_id,
                waited_sec,
            )
            next_log_after_sec += _OPENROUTER_LOCK_WAIT_LOG_EVERY_SEC


def _contains_cjk(text: str) -> bool:
    for c in text:
        code = ord(c)
        if (
            0x3040 <= code <= 0x30FF
            or 0x3400 <= code <= 0x4DBF
            or 0x4E00 <= code <= 0x9FFF
            or 0xF900 <= code <= 0xFAFF
        ):
            return True
    return False


def _needs_repair_translation(
    source: str,
    translated: str,
    constraint: TranslationConstraint | None = None,
) -> bool:
    """Return True only for clearly broken translations."""
    tgt = translated.strip()
    src = source.strip()
    if not tgt:
        return True
    if tgt == src:
        return True
    if tgt in {"[", "]", "[8", "[9", "...", ".."}:
        return True
    if _looks_like_refusal_text(tgt):
        return True
    if _contains_cjk(tgt):
        return True
    return False


def _constraint_tag(constraint: TranslationConstraint | None) -> str:
    """Serialize a constraint to an inline hint consumed by the translator prompt."""
    if constraint is None:
        return ""

    parts: list[str] = []
    if constraint.style in {"sfx", "dialogue"}:
        parts.append(f"style={constraint.style}")
    if constraint.max_words is not None and constraint.max_words > 0:
        parts.append(f"max_words={int(constraint.max_words)}")
    if constraint.max_chars is not None and constraint.max_chars > 0:
        parts.append(f"max_chars={int(constraint.max_chars)}")
    if not parts:
        return ""

    return "(" + ", ".join(parts) + ")"


def _looks_like_romaji_noise(text: str) -> bool:
    """Simple heuristic: flag obvious transliterated-sounding outputs."""
    words = [w.lower() for w in re.findall(r"[A-Za-z']+", text)]
    if len(words) < 3:
        return False
    romaji_markers = {"desu", "kun", "chan", "sama", "senpai", "san"}
    return any(w in romaji_markers for w in words)


def _parse_numbered_response(response: str, expected_count: int) -> dict[int, str]:
    """
    Parse a numbered response like:
        [1] Translated text one
        [2] Translated text two

    Returns a dict mapping 0-based index to translated text.
    """
    result: dict[int, str] = {}
    lines = response.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Try to parse [N] prefix
        if line.startswith("["):
            bracket_end = line.find("]")
            if bracket_end > 0:
                try:
                    num = int(line[1:bracket_end])
                    text = line[bracket_end + 1 :].strip()
                    result[num - 1] = text  # Convert to 0-based
                    continue
                except ValueError:
                    pass

        # If no number prefix, try to assign to the next available slot
        next_idx = len(result)
        if next_idx < expected_count:
            result[next_idx] = line

    return result


def _coerce_message_content(content: object) -> str:
    """
    Normalize OpenRouter message content to plain text.

    Handles both legacy string responses and structured content lists.
    """
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
        if block.get("type") == "text":
            text_value = block.get("text")
            if isinstance(text_value, str) and text_value.strip():
                parts.append(text_value.strip())
    return "\n".join(parts)


def _looks_like_refusal_text(text: str) -> bool:
    lower = text.lower()
    markers = (
        "i can't provide",
        "i cannot provide",
        "i can't assist",
        "i cannot assist",
        "i'm unable to",
        "i am unable to",
        "sorry, i can't",
        "sorry, i cannot",
    )
    return any(marker in lower for marker in markers)
