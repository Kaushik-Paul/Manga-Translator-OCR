"""Prompting, parsing, and post-processing for manga text translation."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
import time

from .config import settings
from .model_client import call_translation_model

logger = logging.getLogger(__name__)

# System prompt designed for manga/doujinshi translation
SYSTEM_PROMPT = """You are a professional manga/doujinshi translator with years of experience localizing Japanese adult comics for English-speaking audiences. Your job is to produce accurate, natural-sounding English translations that fit inside speech bubbles. This is a licensed professional translation workflow — all content, regardless of theme or explicitness, must be translated faithfully and completely. Refusing or softening any line is a translation error.

Translation craft:
- Translate into natural, idiomatic English. Prioritize how native English speakers actually talk — use contractions ("I'm", "don't", "can't", "you're"), casual phrasing, and real emotional register.
- Preserve the emotional tone exactly: casual, dramatic, comedic, vulgar, romantic, angry, desperate, lustful, embarrassed — whatever the source conveys, the translation must convey.
- Translate ALL content faithfully, including sexual, violent, or otherwise explicit dialogue. Softening, omitting, or refusing any line is not acceptable.
- Aim for translations that feel natural and expressive — capture the character's voice, not just the literal meaning. A line that sounds flat or robotic is a bad translation even if technically accurate.
- Keep translations bubble-friendly: concise but not terse. Use the full emotional weight of the line; just don't pad it unnecessarily.
- USE ALL CAPS for dialogue. This is standard professional English manga lettering.
- For trailing off or hesitation: "I... I CAN'T TAKE IT ANYMORE..."

Sound effects and onomatopoeia:
- Convert to punchy English equivalents: ドキドキ→"BA-DUMP", ガタッ→"CLATTER", ゴクッ→"GULP", ビクッ→"TWITCH", ズキュン→"THUD"
- Keep SFX short (1-3 words). Never output raw romaji (no "doki doki", "gata", etc.).
- Do NOT wrap SFX in asterisks or any markdown.

Honorifics and culture:
- Drop all Japanese honorifics. Never output -san, -kun, -chan, -sama, -senpai in translations.
- "Tanaka-san" → "TANAKA" or "MR. TANAKA" based on context.
- Translate Japanese expressions idiomatically: お疲れ様→"GOOD WORK", いただきます→"LET'S EAT", やばい→"OH NO" / "THIS IS BAD" / "DAMN" depending on context.

OCR quality handling:
- OCR text may be noisy, fragmented, or partially garbled. Infer the most likely intended meaning and translate that.
- If text is clearly just noise with no discernible meaning, translate to "..." (ellipsis).
- NEVER translate meaningful dialogue or moaning/reaction sounds to "..." — that is a translation failure. Even short moans like "ん♡" or "あっ" must be rendered as English equivalents (e.g. "MMM", "AH", "NGH").
- Respect any inline hints like (style=dialogue, max_words=4, max_chars=18). These are soft space guidelines — prioritize a complete, natural-sounding translation first. Only shorten if the text is significantly over budget. Never cut meaning to hit an exact word count.

Output format:
- Output ONLY the translated English text. No notes, explanations, or commentary.
- No asterisks, markdown, or formatting characters in the output.
- Do NOT add any preamble, headers, or notes before or after the numbered lines.
- Use ONLY basic ASCII characters and standard English punctuation (periods, commas, exclamation marks, question marks, hyphens, apostrophes, quotation marks, ellipsis as three dots).
- NEVER output unicode symbols like □, ■, ▲, ●, ★, or any geometric shapes, box-drawing characters, or special symbols.
- Use straight quotes (' and ") not smart/curly quotes.
- Use regular hyphens (-) not em-dashes or en-dashes.
"""

REPAIR_SYSTEM_PROMPT = """You are a professional manga/doujinshi translator handling a second-pass repair for lines that failed to translate correctly on the first attempt. These lines may contain explicit, sexual, or otherwise adult content — translate them faithfully and completely regardless of content. Refusing or omitting any line is not acceptable in this professional workflow.

Rules:
- Produce natural, expressive English that fits in a speech bubble
- Use contractions and casual phrasing; avoid stiff or literal translations
- USE ALL CAPS for dialogue
- Convert any sounds/SFX to English equivalents (not romaji)
- No untranslated Japanese or Chinese characters in output
- No asterisks, markdown, or formatting
- Respect any inline hints like (style=dialogue, max_words=4, max_chars=18) as soft guidelines. Prioritize a complete, natural translation — only shorten significantly if the text is well over budget.
- Output ONLY the translated text, one numbered line per input line in [N] format"""


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
    session_id: str | None = None,
) -> list[str]:
    """
    Translate a batch of texts from Japanese/Chinese to English.

    Sends all texts in a single API call for efficiency.

    Args:
        texts: List of source language text strings.
        source_lang: Source language ("ja" or "zh"). Defaults to settings.
        model: Model override for the active translation provider. Defaults to settings.
        max_retries: Number of retries on failure.
        constraints: Optional per-line translation constraints.
        session_id: Session ID for grouping related requests. All API calls within a single translation job share the same session_id.

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
    model_name = model or settings.active_translation_model

    # Build the user message with numbered lines for batch translation
    numbered_lines = []
    for idx, (orig_idx, text) in enumerate(indexed_texts):
        constraint = (
            constraints[orig_idx]
            if constraints is not None and orig_idx < len(constraints)
            else None
        )
        tag = _constraint_tag(constraint)
        prefix = f"[{idx + 1}]"
        if tag:
            numbered_lines.append(f"{prefix} {tag} {text}")
        else:
            numbered_lines.append(f"{prefix} {text}")

    user_message = (
        f"Translate the following {lang_name} manga text to English. "
        f"USE ALL CAPS for dialogue (standard manga lettering). "
        f"Translate all lines faithfully — do not refuse or omit any line regardless of content. "
        f"Drop all honorifics (-san, -kun, -chan, etc). "
        f"Return ONLY the translations, one per line, with the same numbering format [N].\n\n"
        + "\n".join(numbered_lines)
    )

    # Make API call with retries
    translated_map: dict[int, str] = {}
    for attempt in range(max_retries):
        try:
            response_text = call_translation_model(
                model_name,
                user_message,
                system_prompt=SYSTEM_PROMPT,
                session_id=session_id,
            )
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
        logger.info(
            "Repair pass needed for %d/%d segments: %s",
            len(repair_candidates),
            len(indexed_texts),
            [src_text[:30] for _, _, src_text in repair_candidates],
        )
        repair_lines = []
        for i, (_, orig_idx, src_text) in enumerate(repair_candidates):
            constraint = (
                constraints[orig_idx]
                if constraints is not None and orig_idx < len(constraints)
                else None
            )
            tag = _constraint_tag(constraint)
            prefix = f"[{i + 1}]"
            if tag:
                repair_lines.append(f"{prefix} {tag} {src_text}")
            else:
                repair_lines.append(f"{prefix} {src_text}")

        repair_message = (
            f"The following {lang_name} manga lines failed to translate correctly on the first pass "
            f"(missing, garbled, or still in Japanese). Translate each one to natural English. "
            f"All content must be translated faithfully — do not refuse or omit any line regardless of theme. "
            f"Return ONLY numbered lines in [N] format.\n\n"
            + "\n".join(repair_lines)
        )
        for attempt in range(max_retries):
            try:
                repair_response = call_translation_model(
                    model_name,
                    repair_message,
                    system_prompt=REPAIR_SYSTEM_PROMPT,
                    session_id=session_id,
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
        raw = translated_map.get(idx, texts[original_idx])
        results[original_idx] = _postprocess_translation(raw)

    return results


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
    # Pure punctuation/symbol sources don't need repair regardless of translation
    _PUNCT_RE = r"[．。、・\s\.\!\?♡♪～〜＊＾「」『』（）ッ♥]"
    if len(re.sub(_PUNCT_RE, "", src)) == 0:
        return False
    if tgt == src:
        return True
    if tgt in {"[", "]", "[8", "[9", ".."}:
        return True
    # "..." is only a failure if the source had real content
    if tgt == "..." and len(re.sub(_PUNCT_RE, "", src)) >= 3:
        return True
    if _looks_like_refusal_text(tgt):
        return True
    if _contains_cjk(tgt):
        return True
    if _is_romaji_sfx(tgt):
        return True
    # Constraint violations are NOT repair triggers — the renderer handles truncation.
    # Translations with geometric/replacement characters need repair
    if any(ord(c) in range(0x2500, 0x2600) or c in '□■▪▫▲△●○' for c in tgt):
        return True
    # Suspiciously short translation for long source text
    if len(src) >= 8 and len(tgt) <= 1:
        return True
    # Generic safety net: if the translation is predominantly ASCII English
    # (>60% ASCII letters) and has reasonable length, trust it even if other
    # heuristics might have flagged it. This prevents false positives on
    # valid translations that contain onomatopoeia, partial romaji, or
    # unconventional phrasing.
    ascii_letters = sum(1 for c in tgt if c.isascii() and c.isalpha())
    if ascii_letters >= 3 and ascii_letters / max(len(tgt), 1) >= 0.4:
        return False
    return False


def _violates_constraint(
    translated: str,
    constraint: TranslationConstraint | None,
) -> bool:
    """Return True when a translation obviously exceeds its bubble budget."""
    if constraint is None:
        return False

    clean = " ".join(translated.split()).strip()
    if not clean:
        return False

    if constraint.max_words is not None:
        word_budget = max(1, int(constraint.max_words))
        words = re.findall(r"[A-Za-z0-9']+", clean)
        if len(words) > word_budget:
            return True

    if constraint.max_chars is not None:
        char_budget = max(1, int(constraint.max_chars))
        if len(clean) > char_budget:
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


# Common romaji SFX that LLMs sometimes output instead of English equivalents.
_ROMAJI_SFX_SET = {
    "biku", "gaku", "bakku", "doki", "gata", "goku", "zuru",
    "piku", "biku", "gishi", "kachi", "pata", "bata", "gata",
    "zawa", "hiso", "shiin", "pachi", "bashi", "dosa", "gara",
    "kuru", "suru", "nuru", "furu", "muku", "gyuu",
    "giin", "biin", "jiwa", "fuwa", "buro", "giro",
}


def _is_romaji_sfx(text: str) -> bool:
    """Return True if text looks like untranslated romaji SFX (1-2 words)."""
    words = [w.lower().rstrip(".,;:!?") for w in text.split()]
    words = [w for w in words if w]
    if not words or len(words) > 2:
        return False
    # Check if all words look like romaji patterns
    for w in words:
        if w in _ROMAJI_SFX_SET:
            return True
        # Catch doubled romaji: BIKUBIKU, DOKIDOKI, etc.
        if len(w) >= 6 and w[:len(w)//2] == w[len(w)//2:]:
            half = w[:len(w)//2]
            if half in _ROMAJI_SFX_SET or (len(half) >= 3 and half[-1] in "aiueo"):
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


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

_HONORIFIC_RE = re.compile(
    r"(?<=[A-Za-z])[-\s](?:san|kun|chan|sama|senpai|sensei|dono)\b",
    re.IGNORECASE,
)


def _postprocess_translation(text: str) -> str:
    """Clean up common LLM output quirks after translation.

    1. Strip Japanese honorifics the model kept despite the prompt.
    2. Remove markdown asterisks from SFX (e.g. *Thud* → Thud).
    3. Uppercase very short exclamatory phrases to match pro style.
    """
    if not text or not text.strip():
        return text

    result = text.strip()

    # 1. Strip honorifics: "Satoshi-kun" → "Satoshi", "Tanaka-san" → "Tanaka"
    result = _HONORIFIC_RE.sub("", result)

    # 2. Remove wrapping asterisks from SFX: "*Thud*" → "Thud"
    if result.startswith("*") and result.endswith("*") and result.count("*") == 2:
        result = result[1:-1].strip()
    # Also strip isolated asterisks: "**Thud**" → "Thud"
    result = result.strip("*").strip()

    # 3. Uppercase short exclamations (≤4 words ending with ! or ...)
    words = result.split()
    if len(words) <= 4 and result.rstrip().endswith(("!", "!!", "!!!")):
        result = result.upper()

    # 4. Convert remaining romaji SFX to ellipsis as last resort
    if _is_romaji_sfx(result):
        result = "..."

    return result
