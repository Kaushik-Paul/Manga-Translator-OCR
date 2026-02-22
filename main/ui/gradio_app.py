"""
Gradio UI for the Manga Translator.

Provides a web interface to:
1. Load manga folders from GCP Cloud Storage
2. Select images for translation
3. Run the translation pipeline with live progress
4. Download translated results
"""

from __future__ import annotations

import logging

import gradio as gr

from ..config import settings
from ..gcp_storage import folder_exists, list_images_in_folder
from .pipeline_runner import TranslationResult, run_translation_pipeline
from .styles import CUSTOM_CSS

logger = logging.getLogger(__name__)


def create_app() -> tuple[gr.Blocks, gr.themes.Soft, str]:
    """Build and return the Gradio Blocks application, theme, and CSS."""

    theme = gr.themes.Soft(
        primary_hue="orange",
        secondary_hue="purple",
        neutral_hue="gray",
    )

    with gr.Blocks(
        title="Manga Translator OCR",
    ) as app:

        # ── State ──────────────────────────────────────
        loaded_folder = gr.State("")
        loaded_images = gr.State([])
        prev_selected = gr.State([])

        # ── Header ─────────────────────────────────────
        gr.HTML(
            """
            <div class="app-header">
                <h1>🎌 Manga Translator OCR</h1>
                <p>Translate manga pages from Japanese to English via GCP Cloud Storage</p>
            </div>
            """
        )

        # ── Section 1: Folder Input ────────────────────
        with gr.Group(elem_classes=["section-card"]):
            gr.HTML('<div class="section-title">📁 Manga Folder</div>')
            with gr.Row():
                folder_input = gr.Textbox(
                    label="Folder name",
                    placeholder="Enter manga folder name (in raw-manga bucket)...",
                    scale=4,
                    interactive=True,
                )
                load_btn = gr.Button(
                    "🔍 Load",
                    variant="primary",
                    elem_classes=["primary-btn"],
                    scale=1,
                    interactive=True,
                )
            folder_status = gr.HTML(visible=False)

        # ── Section 2: Image Selector ──────────────────
        with gr.Group(elem_classes=["section-card"], visible=False) as image_section:
            gr.HTML('<div class="section-title">🖼️ Select Images</div>')
            image_selector = gr.CheckboxGroup(
                label="Images to translate",
                choices=[],
                value=[],
                elem_classes=["image-selector"],
                interactive=True,
            )
            with gr.Row():
                translate_btn = gr.Button(
                    "🚀 Translate Selected",
                    variant="primary",
                    elem_classes=["translate-btn"],
                    scale=2,
                    interactive=True,
                )

        # ── Section 3: Progress & Results ──────────────
        with gr.Group(elem_classes=["section-card"], visible=False) as results_section:
            gr.HTML('<div class="section-title">📊 Progress & Results</div>')
            progress_log = gr.Textbox(
                label="Pipeline Log",
                lines=12,
                max_lines=30,
                interactive=False,
                elem_classes=["progress-box"],
            )
            with gr.Row():
                failed_images_box = gr.Textbox(
                    label="❌ Failed Images",
                    lines=3,
                    interactive=False,
                    visible=False,
                )
            download_html = gr.HTML(visible=False)

        # ── Event Handlers ─────────────────────────────

        def load_folder(folder_name: str):
            """Validate and load images from a GCS folder."""
            folder_name = folder_name.strip()
            if not folder_name:
                return (
                    gr.update(visible=True, value='<span class="result-error">⚠️ Please enter a folder name.</span>'),
                    gr.update(visible=False),
                    gr.update(choices=[], value=[]),
                    "",
                    [],
                )

            try:
                if not folder_exists(folder_name):
                    return (
                        gr.update(
                            visible=True,
                            value=f'<span class="result-error">❌ Folder "<b>{folder_name}</b>" not found in raw-manga/.</span>',
                        ),
                        gr.update(visible=False),
                        gr.update(choices=[], value=[]),
                        "",
                        [],
                    )

                images = list_images_in_folder(folder_name)
                if not images:
                    return (
                        gr.update(
                            visible=True,
                            value=f'<span class="result-error">⚠️ Folder "<b>{folder_name}</b>" has no images.</span>',
                        ),
                        gr.update(visible=False),
                        gr.update(choices=[], value=[]),
                        "",
                        [],
                    )

                # Build choices: "All" + numbered image names
                choices = ["All"] + images
                return (
                    gr.update(
                        visible=True,
                        value=f'<span class="result-success">✅ Found {len(images)} image(s) in "<b>{folder_name}</b>"</span>',
                    ),
                    gr.update(visible=True),
                    gr.update(choices=choices, value=["All"] + images),
                    folder_name,
                    images,
                )
            except Exception as e:
                logger.error("Error loading folder: %s", e)
                return (
                    gr.update(
                        visible=True,
                        value=f'<span class="result-error">❌ Error: {e}</span>',
                    ),
                    gr.update(visible=False),
                    gr.update(choices=[], value=[]),
                    "",
                    [],
                )

        def on_image_selection_change(
            selected: list[str],
            all_images: list[str],
            prev_selected: list[str],
        ):
            """Handle 'All' toggle logic in the checkbox group."""
            if not all_images:
                return gr.update(), selected

            prev_had_all = "All" in (prev_selected or [])
            curr_has_all = "All" in selected
            images_only = [s for s in selected if s != "All"]
            all_with_all = ["All"] + all_images

            # Case 1: User just checked "All"
            if curr_has_all and not prev_had_all:
                return gr.update(value=all_with_all), all_with_all

            # Case 2: User just unchecked "All"
            if not curr_has_all and prev_had_all:
                return gr.update(value=[]), []

            # Case 3: User deselected an individual image (All was on)
            if prev_had_all and curr_has_all and len(images_only) < len(all_images):
                new_val = images_only  # drop "All"
                return gr.update(value=new_val), new_val

            # Case 4: User selected the last remaining image → auto-add "All"
            if not curr_has_all and set(images_only) == set(all_images):
                return gr.update(value=all_with_all), all_with_all

            # Default: no change needed
            return gr.update(), selected

        def run_pipeline(
            folder_name: str,
            selected: list[str],
            all_images: list[str],
        ):
            """Run translation pipeline with streaming progress updates."""
            # Filter out "All" from the selection
            selected_images = [s for s in selected if s != "All"]

            if not selected_images:
                yield (
                    # folder_input, load_btn, image_selector, translate_btn
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    # results_section, progress_log
                    gr.update(visible=True),
                    "⚠️ No images selected.",
                    # failed_images_box, download_html
                    gr.update(visible=False),
                    gr.update(visible=False),
                )
                return

            # Disable inputs while running
            log_lines: list[str] = []

            # Initial yield — disable all inputs, show results section
            yield (
                gr.update(interactive=False),  # folder_input
                gr.update(interactive=False),  # load_btn
                gr.update(interactive=False),  # image_selector
                gr.update(interactive=False),  # translate_btn
                gr.update(visible=True),  # results_section
                f"🚀 Starting translation for {len(selected_images)} image(s) from '{folder_name}'...\n",
                gr.update(visible=False),  # failed_images_box
                gr.update(visible=False),  # download_html
            )

            # Run pipeline generator
            final_result = None
            for update in run_translation_pipeline(
                folder_name=folder_name,
                selected_images=selected_images,
                output_dir=settings.output_dir,
                config=settings,
            ):
                if isinstance(update, TranslationResult):
                    final_result = update
                    break

                log_lines.append(update)
                log_text = "\n".join(log_lines)
                yield (
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(visible=True),
                    log_text,
                    gr.update(visible=False),
                    gr.update(visible=False),
                )

            # ── Final results ──────────────────────────
            if final_result is None:
                final_result = TranslationResult()

            # Summary line
            log_lines.append(
                f"\n{'═' * 40}\n"
                f"🏁 Pipeline complete: {final_result.succeeded}/{final_result.total} images translated\n"
                f"{'═' * 40}"
            )
            log_text = "\n".join(log_lines)

            # Failed images
            failed_visible = len(final_result.failed_images) > 0
            failed_text = "\n".join(final_result.failed_images) if failed_visible else ""

            # Download link
            download_visible = bool(final_result.download_url)
            download_content = ""
            if download_visible:
                download_content = (
                    f'<div class="download-link" style="padding: 1rem; text-align: center;">'
                    f'<a href="{final_result.download_url}" target="_blank" '
                    f'style="font-size: 1.1rem;">📥 Download Translated Manga (CBZ) — Expires in 24 hours</a>'
                    f"</div>"
                )

            # Re-enable inputs
            yield (
                gr.update(interactive=True),  # folder_input
                gr.update(interactive=True),  # load_btn
                gr.update(interactive=True),  # image_selector
                gr.update(interactive=True),  # translate_btn
                gr.update(visible=True),  # results_section
                log_text,
                gr.update(visible=failed_visible, value=failed_text),  # failed_images_box
                gr.update(visible=download_visible, value=download_content),  # download_html
            )

        # ── Wire Events ────────────────────────────────

        load_btn.click(
            fn=load_folder,
            inputs=[folder_input],
            outputs=[
                folder_status,
                image_section,
                image_selector,
                loaded_folder,
                loaded_images,
            ],
        )

        # Also trigger load on Enter key
        folder_input.submit(
            fn=load_folder,
            inputs=[folder_input],
            outputs=[
                folder_status,
                image_section,
                image_selector,
                loaded_folder,
                loaded_images,
            ],
        )

        image_selector.change(
            fn=on_image_selection_change,
            inputs=[image_selector, loaded_images, prev_selected],
            outputs=[image_selector, prev_selected],
        )

        translate_btn.click(
            fn=run_pipeline,
            inputs=[loaded_folder, image_selector, loaded_images],
            outputs=[
                folder_input,
                load_btn,
                image_selector,
                translate_btn,
                results_section,
                progress_log,
                failed_images_box,
                download_html,
            ],
        )

    return app, theme, CUSTOM_CSS
