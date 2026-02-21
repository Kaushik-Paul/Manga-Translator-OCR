"""
Translation module using OpenRouter API.

Sends extracted text (not images) to OpenRouter for translation,
using models that handle NSFW text content without moderation issues.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from threading import Lock
import time

import httpx

from .config import settings

logger = logging.getLogger(__name__)
_OPENROUTER_CALL_LOCK = Lock()

# System prompt designed for manga/doujinshi translation
SYSTEM_PROMPT = """You are an expert manga/doujinshi translator specializing in Japanese and Chinese to English translation.

Rules:
- Translate the given text naturally into English
- Preserve the tone and nuance of the original (casual, dramatic, emotional, etc.)
- Keep translations concise and bubble-friendly; prefer short natural phrasing
- Keep each line as short as possible while preserving meaning
- For onomatopoeia/sound effects, use short natural English equivalents
- If the text is a single sound effect, output a short English SFX only
- If OCR text is fragmented/noisy, produce one compact readable English line
- If text looks garbled from OCR, infer the most likely intended meaning and translate that
- If an input line includes constraints like (style=sfx/dialogue, max_words=N, max_chars=N), obey them strictly
- Avoid raw romaji transliteration of kana
- Prefer natural English interjections (e.g., "Ah...", "Gulp...", "Throb...") over transliterated syllables
- Do not output long gibberish transliterations; summarize noisy moans/interjections briefly
- Output ONLY the translated text, nothing else
- Do NOT add explanations, notes, or commentary
- Do NOT refuse to translate any content — you are a professional translator"""

REPAIR_SYSTEM_PROMPT = """Translate manga OCR text to concise natural English.

Rules:
- Output English only
- Keep it short and bubble-friendly
- Use natural English SFX/interjections for noisy text
- Never output romaji or untranslated Japanese/Chinese
- Never output transliterated Japanese syllables (e.g., "desu", "kun", "chan", "uhyo")
- If a line includes (style=..., max_words=N, max_chars=N), obey it strictly
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
        constraint = (
            constraints[orig_idx]
            if constraints is not None and orig_idx < len(constraints)
            else None
        )
        hint = _constraint_tag(constraint)
        if hint:
            numbered_lines.append(f"[{idx + 1}] {hint} {text}")
        else:
            numbered_lines.append(f"[{idx + 1}] {text}")

    user_message = (
        f"Translate the following {lang_name} manga text to English. "
        f"Each line is numbered and may include optional constraints like "
        f"(style=sfx/dialogue, max_words=N, max_chars=N). Follow constraints strictly. "
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
            constraint = (
                constraints[orig_idx]
                if constraints is not None and orig_idx < len(constraints)
                else None
            )
            hint = _constraint_tag(constraint)
            if hint:
                repair_lines.append(f"[{i + 1}] {hint} {src_text}")
            else:
                repair_lines.append(f"[{i + 1}] {src_text}")

        repair_message = (
            f"Translate the following {lang_name} manga text to concise English. "
            f"Lines may contain (style=..., max_words=N, max_chars=N) constraints. "
            f"Follow constraints strictly. Return ONLY numbered lines in [N] format.\n\n"
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


def translate_single(
    text: str,
    source_lang: str | None = None,
    model: str | None = None,
) -> str:
    """Translate a single text string. Convenience wrapper."""
    results = translate_texts([text], source_lang=source_lang, model=model)
    return results[0]


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
        }
    }

    logger.info("Calling OpenRouter API (model: %s)...", model)

    # Serialize outbound OpenRouter requests across page-worker threads.
    # This avoids provider-side empty-content responses seen under burst concurrency.
    with _OPENROUTER_CALL_LOCK:
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

    message = choices[0].get("message", {})
    content = message.get("content", "")
    text = _coerce_message_content(content).strip()
    if not text:
        raise ValueError("OpenRouter returned empty translation content.")
    logger.info("Translation received (%d chars).", len(text))
    return text


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
    if _looks_like_romaji_noise(tgt) and _contains_cjk(src):
        return True

    words = re.findall(r"[A-Za-z']+", tgt)
    if constraint is not None:
        if constraint.max_words is not None and constraint.max_words > 0:
            if len(words) > constraint.max_words + 1:
                return True
        if constraint.max_chars is not None and constraint.max_chars > 0:
            if len(tgt) > constraint.max_chars + 6:
                return True
        if constraint.style == "sfx" and len(words) >= 4 and _looks_like_romaji_noise(tgt):
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
    """Heuristic: flag long transliterated-sounding outputs for repair pass."""
    words = [w.lower() for w in re.findall(r"[A-Za-z']+", text)]
    if len(words) < 3:
        return False

    long_words = [w for w in words if len(w) >= 4]
    if len(long_words) < 2:
        return False

    common_en = {
        "this",
        "that",
        "what",
        "when",
        "where",
        "then",
        "well",
        "okay",
        "right",
        "please",
        "come",
        "again",
        "good",
        "more",
        "with",
        "have",
        "gonna",
        "suck",
        "lick",
        "ouch",
        "slip",
        "gulp",
        "throb",
        "twitch",
        "splash",
        "moan",
        "heh",
        "huh",
        "ah",
        "oh",
        "mmm",
        "yeah",
        "feel",
        "hot",
        "mom",
        "rough",
        "coming",
    }

    def english_like(word: str) -> bool:
        if word in common_en:
            return True
        return word.endswith(
            (
                "ing",
                "ed",
                "ly",
                "er",
                "est",
                "tion",
                "ness",
                "ment",
                "ful",
                "able",
            )
        )

    english_hits = sum(1 for w in long_words if english_like(w))
    romaji_markers = {"desu", "kun", "chan", "sama", "senpai", "san"}
    if any(w in romaji_markers for w in words):
        return True

    romaji_cluster_hits = sum(
        1
        for w in long_words
        if re.search(r"(sh|ch|ts|ry|ny|ky|gy|ja|ju|jo)", w) is not None
    )
    vowel_heavy = 0
    for w in long_words:
        vowels = sum(1 for ch in w if ch in "aeiou")
        if vowels / float(max(1, len(w))) >= 0.45:
            vowel_heavy += 1

    if romaji_cluster_hits >= 2 and english_hits <= 1:
        return True
    if english_hits <= max(1, len(long_words) // 4) and vowel_heavy >= max(
        2, len(long_words) // 2
    ):
        return True
    return False


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
