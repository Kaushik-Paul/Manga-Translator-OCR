"""
Translation module using OpenRouter API.

Sends extracted text (not images) to OpenRouter for translation,
using models that handle NSFW text content without moderation issues.
"""

from __future__ import annotations

import logging
import time

import httpx

from .config import settings

logger = logging.getLogger(__name__)

# System prompt designed for manga/doujinshi translation
SYSTEM_PROMPT = """You are an expert manga/doujinshi translator specializing in Japanese and Chinese to English translation.

Rules:
- Translate the given text naturally into English
- Preserve the tone and nuance of the original (casual, dramatic, emotional, etc.)
- Keep translations concise and bubble-friendly; prefer short natural phrasing
- For onomatopoeia/sound effects, provide the English equivalent or romanized form in parentheses
- If the text is a single sound effect, just translate/romanize it
- If OCR text is fragmented/noisy, produce one compact readable line
- Output ONLY the translated text, nothing else
- Do NOT add explanations, notes, or commentary
- Do NOT refuse to translate any content — you are a professional translator"""


def translate_texts(
    texts: list[str],
    source_lang: str | None = None,
    model: str | None = None,
    max_retries: int = 3,
) -> list[str]:
    """
    Translate a batch of texts from Japanese/Chinese to English.

    Sends all texts in a single API call for efficiency.

    Args:
        texts: List of source language text strings.
        source_lang: Source language ("ja" or "zh"). Defaults to settings.
        model: OpenRouter model to use. Defaults to settings.
        max_retries: Number of retries on failure.

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
    for idx, (_, text) in enumerate(indexed_texts):
        numbered_lines.append(f"[{idx + 1}] {text}")

    user_message = (
        f"Translate the following {lang_name} manga text to English. "
        f"Each line is numbered. Return ONLY the translations, one per line, "
        f"with the same numbering format [N].\n\n"
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

    # Reconstruct the full results list (preserving empty string positions)
    results = ["" for _ in texts]
    for idx, (original_idx, _) in enumerate(indexed_texts):
        results[original_idx] = translated_map.get(idx, texts[original_idx])

    return results


def translate_single(
    text: str,
    source_lang: str | None = None,
    model: str | None = None,
) -> str:
    """Translate a single text string. Convenience wrapper."""
    results = translate_texts([text], source_lang=source_lang, model=model)
    return results[0]


def _call_openrouter(model: str, user_message: str) -> str:
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
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.3,  # Low temperature for consistent translations
        "max_tokens": 4096,
    }

    logger.info("Calling OpenRouter API (model: %s)...", model)

    with httpx.Client(timeout=60.0) as client:
        response = client.post(
            f"{settings.openrouter_base_url}/chat/completions",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()

    data = response.json()

    # Extract the response text
    choices = data.get("choices", [])
    if not choices:
        raise ValueError("No choices returned from OpenRouter API")

    content = choices[0].get("message", {}).get("content", "")
    logger.info("Translation received (%d chars).", len(content))
    return content.strip()


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
