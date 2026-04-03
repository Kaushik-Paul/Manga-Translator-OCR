"""
Pipeline runner for the Gradio UI.

Wraps the core translation pipeline with:
- GCS download/upload
- Per-image error handling (never stops on failure)
- Batch parallel processing when Modal is enabled
- Progress reporting via generator yields
- Local file cleanup after upload
"""

from __future__ import annotations

import logging
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Settings, settings
from ..gcp_storage import (
    cleanup_local_files,
    download_images,
    generate_download_url,
    upload_translated_images,
)
from ..pipeline import translate_page

logger = logging.getLogger(__name__)


@dataclass
class TranslationResult:
    """Result of a full pipeline run."""

    total: int = 0
    succeeded: int = 0
    failed_images: list[str] = field(default_factory=list)
    download_url: str = ""
    error_message: str = ""


def run_translation_pipeline(
    folder_name: str,
    selected_images: list[str],
    output_dir: Path | None = None,
    config: Settings | None = None,
):
    """
    Generator that runs the translation pipeline for selected images.

    Yields progress strings like "Downloading images...", "2/20 images translated".
    The final yield is a TranslationResult.
    """
    cfg = config or settings
    out_dir = output_dir or cfg.output_dir
    result = TranslationResult(total=len(selected_images))

    # ── Step 1: Download images from GCS ───────────────
    yield f"📥 Downloading {len(selected_images)} image(s) from GCS..."

    try:
        local_paths = download_images(folder_name, selected_images, out_dir)
    except Exception as e:
        result.error_message = f"Failed to download images: {e}"
        logger.error(result.error_message)
        yield result
        return

    if not local_paths:
        result.error_message = "No images were downloaded successfully."
        yield result
        return

    yield f"✅ Downloaded {len(local_paths)} image(s). Starting translation...\n"

    # ── Step 2: Translate images ───────────────────────
    # Output goes to output/translated/folder_name
    translated_dir = out_dir / "translated" / folder_name
    translated_dir.mkdir(parents=True, exist_ok=True)

    # Config override pointing output to the translated dir
    translate_cfg = Settings(
        openrouter_api_key=cfg.openrouter_api_key,
        translation_model=cfg.translation_model,
        source_lang=cfg.source_lang,
        use_modal=cfg.use_modal,
        use_mangaocr_cpu=cfg.use_mangaocr_cpu,
        use_detection_model=cfg.use_detection_model,
        modal_max_parallel_pages=cfg.modal_max_parallel_pages,
        output_dir=translated_dir,
    )

    translated_paths: list[Path] = []
    total = result.total
    job_session_id = str(uuid.uuid4())
    logger.info("Translation job session ID: %s", job_session_id)

    def _translate_one(idx_path: tuple[int, Path]) -> tuple[int, Path, Path | None, Exception | None]:
        i, local_path = idx_path
        try:
            out = translate_page(local_path, config=translate_cfg, session_id=job_session_id)
            return i, local_path, out, None
        except Exception as e:
            return i, local_path, None, e

    # Use batch parallel processing when Modal is enabled (same as CLI)
    if cfg.use_modal and total > 1:
        max_workers = min(cfg.modal_max_parallel_pages, total)
        tasks = list(enumerate(local_paths, start=1))
        total_batches = (total + max_workers - 1) // max_workers

        yield f"⚡ Using batched parallel processing ({max_workers} workers per batch)\n"

        for batch_idx, start in enumerate(range(0, total, max_workers), start=1):
            batch_tasks = tasks[start : start + max_workers]
            batch_names = [p.name for _, p in batch_tasks]
            yield f"📦 Batch {batch_idx}/{total_batches}: {', '.join(batch_names)}"

            with ThreadPoolExecutor(max_workers=len(batch_tasks)) as pool:
                batch_results = list(pool.map(_translate_one, batch_tasks))

            for i, local_path, out, error in batch_results:
                image_name = local_path.name
                if out is not None:
                    translated_paths.append(out)
                    result.succeeded += 1
                    yield (
                        f"✅ [{i}/{total}] {image_name} translated "
                        f"({result.succeeded}/{total} done)"
                    )
                elif error is not None:
                    result.failed_images.append(image_name)
                    tb = traceback.format_exc()
                    logger.error("Failed to translate %s: %s\n%s", image_name, error, tb)
                    yield f"❌ [{i}/{total}] {image_name} FAILED: {error}"
    else:
        # Sequential processing
        for i, local_path in enumerate(local_paths, start=1):
            image_name = local_path.name
            yield f"🔄 [{i}/{total}] Translating {image_name}..."

            try:
                out = translate_page(local_path, config=translate_cfg, session_id=job_session_id)
                translated_paths.append(out)
                result.succeeded += 1
                yield (
                    f"✅ [{i}/{total}] {image_name} translated "
                    f"({result.succeeded}/{total} done)"
                )
            except Exception as e:
                result.failed_images.append(image_name)
                tb = traceback.format_exc()
                logger.error("Failed to translate %s: %s\n%s", image_name, e, tb)
                yield f"❌ [{i}/{total}] {image_name} FAILED: {e}"

    # ── Step 3: Upload translated images to GCS ────────
    if translated_paths:
        yield f"\n📤 Uploading {len(translated_paths)} translated image(s) to GCS..."
        try:
            upload_translated_images(folder_name, translated_paths)
            yield "✅ Upload complete!"
        except Exception as e:
            err = f"Failed to upload translated images: {e}"
            logger.error(err)
            yield f"❌ {err}"

        # ── Step 4: Generate download URL ──────────────
        yield "🔗 Generating download link..."
        try:
            result.download_url = generate_download_url(folder_name)
            yield "✅ Download link generated!"
        except Exception as e:
            err = f"Failed to generate download URL: {e}"
            logger.error(err)
            yield f"⚠️ {err}"

    # ── Step 5: Cleanup local files ────────────────────
    yield "🧹 Cleaning up local files..."
    try:
        cleanup_local_files(out_dir, folder_name)
    except Exception as e:
        logger.warning("Cleanup warning: %s", e)

    yield result
