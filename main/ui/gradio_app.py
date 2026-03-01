"""
Gradio UI for the Manga Translator.

Provides a web interface to:
1. Load manga folders from GCP Cloud Storage
2. Select images for translation
3. Run the translation pipeline with live progress
4. Download translated results
"""

from __future__ import annotations

import hmac
import logging
import threading
from pathlib import Path

import gradio as gr

from ..config import settings
from ..gcp_storage import folder_exists, list_images_in_folder, upload_raw_manga_zip
from .pipeline_runner import TranslationResult, run_translation_pipeline
from .styles import CUSTOM_CSS

logger = logging.getLogger(__name__)

# Global state to share pipeline status across all tabs/users
GLOBAL_STATE = {
    "folder_name": "",
    "loaded_images": [],
    "selected_images": [],
    "is_running": False,
    "log_text": "",
    "failed_text": "",
    "download_url": "",
}



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

        # Hidden state component to trigger auto-updates
        timer = gr.Timer(value=1.0, active=False)
        auth_action = gr.State("")

        # ── Header ─────────────────────────────────────
        gr.HTML(
            """
            <div class="app-header">
                <h1>🎌 Manga Translator OCR</h1>
                <p>Translate manga pages from Japanese to English via GCP Cloud Storage</p>
            </div>
            """
        )

        with gr.Row(elem_classes=["top-actions-row"]):
            upload_zip_btn = gr.Button(
                "📤 Upload ZIP to raw-manga",
                variant="secondary",
                elem_classes=["upload-zip-btn"],
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
            download_btn = gr.Button(
                "📥 Download Translated Manga (CBZ) — Expires in 24 hours",
                variant="primary",
                elem_classes=["download-btn"],
                interactive=True,
                visible=False,
            )

        # Hidden trigger value. JS listener opens this URL in a new tab.
        download_launch_url = gr.Textbox(value="", visible=False, interactive=False)

        # Password popup shown before costly/protected actions.
        with gr.Group(visible=False, elem_classes=["auth-modal-overlay"]) as auth_modal:
            with gr.Group(elem_classes=["auth-modal-card"]):
                gr.HTML('<div class="auth-modal-title">🔒 Password Required</div>')
                auth_prompt = gr.HTML()
                auth_password = gr.Textbox(
                    label="Password",
                    type="password",
                    placeholder="Enter password",
                    interactive=True,
                )
                auth_error = gr.HTML(visible=False)
                with gr.Row():
                    auth_submit_btn = gr.Button(
                        "Continue",
                        variant="primary",
                        elem_classes=["primary-btn"],
                    )
                    auth_cancel_btn = gr.Button("Cancel")

        # ZIP upload popup shown after password authorization.
        with gr.Group(visible=False, elem_classes=["auth-modal-overlay"]) as upload_modal:
            with gr.Group(elem_classes=["auth-modal-card"]):
                gr.HTML('<div class="auth-modal-title">📦 Upload Manga ZIP</div>')
                gr.HTML(
                    '<div class="auth-modal-message">Select a .zip file. '
                    'It will be uploaded to raw-manga/&lt;zip-name&gt;/.</div>'
                )
                upload_zip_file = gr.File(
                    label="ZIP file",
                    file_types=[".zip"],
                    type="filepath",
                )
                upload_status = gr.HTML(visible=False)
                upload_error = gr.HTML(visible=False)
                with gr.Row():
                    upload_confirm_btn = gr.Button(
                        "Confirm",
                        variant="primary",
                        elem_classes=["primary-btn"],
                    )
                    upload_close_btn = gr.Button("Close")

        def _action_prompt(action: str) -> str:
            if action == "download":
                text = "Enter password to open the download link."
            elif action == "upload_zip":
                text = "Enter password to upload a ZIP file."
            else:
                text = "Enter password to start translation."
            return f'<div class="auth-modal-message">{text}</div>'

        def _close_auth_modal():
            return (
                "",
                gr.update(visible=False),
                gr.update(value=""),
                gr.update(value=""),
                gr.update(visible=False, value=""),
            )

        def _close_upload_modal():
            return (
                gr.update(visible=False),
                gr.update(value=None),
                gr.update(visible=False, value=""),
                gr.update(visible=False, value=""),
            )

        def _open_upload_modal():
            return (
                gr.update(visible=True),
                gr.update(value=None),
                gr.update(visible=False, value=""),
                gr.update(visible=False, value=""),
            )

        def _open_auth_modal(action: str):
            if action == "download" and not GLOBAL_STATE.get("download_url"):
                return _close_auth_modal()
            return (
                action,
                gr.update(visible=True),
                gr.update(value=_action_prompt(action)),
                gr.update(value=""),
                gr.update(visible=False, value=""),
            )

        def _validate_action_password(password: str) -> tuple[bool, str]:
            configured_password = settings.gradio_action_password.strip()

            if not configured_password:
                return (
                    False,
                    "⚠️ GRADIO_ACTION_PASSWORD is not configured in .env. "
                    "Set it and restart the app.",
                )
            if not password:
                return False, "⚠️ Password is required."
            if not hmac.compare_digest(password, configured_password):
                return False, "❌ Incorrect password."
            return True, ""

        def sync_state_from_global():
            """Called by Timer or on page load to perfectly reflect global state."""
            is_running = GLOBAL_STATE.get("is_running", False)
            folder_name = GLOBAL_STATE.get("folder_name", "")
            loaded_imgs = GLOBAL_STATE.get("loaded_images", [])
            selected_imgs = GLOBAL_STATE.get("selected_images", [])
            
            show_results = bool(GLOBAL_STATE.get("log_text"))
            failed_vis = bool(GLOBAL_STATE.get("failed_text"))
            dl_vis = bool(GLOBAL_STATE.get("download_url"))
            
            choices = ["All"] + loaded_imgs if loaded_imgs else []
            show_image_sec = bool(choices)
            
            return (
                gr.update(interactive=not is_running, value=folder_name), # folder_input
                gr.update(interactive=not is_running), # load_btn
                gr.update(visible=show_image_sec), # image_section
                gr.update(choices=choices, value=selected_imgs, interactive=not is_running), # image_selector
                gr.update(interactive=not is_running), # translate_btn
                gr.update(visible=show_results), # results_section
                GLOBAL_STATE.get("log_text", ""), # progress_log
                gr.update(visible=failed_vis, value=GLOBAL_STATE.get("failed_text", "")), # failed_images_box
                gr.update(visible=dl_vis, interactive=not is_running), # download_btn
                gr.update(interactive=not is_running), # upload_zip_btn
                gr.update(active=is_running) # timer active while running
            )

        def load_folder(folder_name: str):
            """Validate and load images from a GCS folder."""
            folder_name = folder_name.strip()
            if not folder_name:
                return (
                    gr.update(visible=True, value='<span class="result-error">⚠️ Please enter a folder name.</span>'),
                    gr.update(visible=False),
                    gr.update(choices=[], value=[]),
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
                    )

                # Build choices: "All" + numbered image names
                choices = ["All"] + images
                
                GLOBAL_STATE["folder_name"] = folder_name
                GLOBAL_STATE["loaded_images"] = images
                GLOBAL_STATE["selected_images"] = choices  # Select all by default
                
                return (
                    gr.update(
                        visible=True,
                        value=f'<span class="result-success">✅ Found {len(images)} image(s) in "<b>{folder_name}</b>"</span>',
                    ),
                    gr.update(visible=True),
                    gr.update(choices=choices, value=GLOBAL_STATE["selected_images"]),
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
                )

        def on_image_selection_change(selected: list[str]):
            """Handle 'All' toggle logic in the checkbox group."""
            all_images = GLOBAL_STATE.get("loaded_images", [])
            prev_selected = GLOBAL_STATE.get("selected_images", [])
            
            if not all_images:
                GLOBAL_STATE["selected_images"] = selected
                return gr.update()

            prev_had_all = "All" in (prev_selected or [])
            curr_has_all = "All" in selected
            images_only = [s for s in selected if s != "All"]
            all_with_all = ["All"] + all_images

            # Case 1: User just checked "All"
            if curr_has_all and not prev_had_all:
                GLOBAL_STATE["selected_images"] = all_with_all
                return gr.update(value=all_with_all)

            # Case 2: User just unchecked "All"
            if not curr_has_all and prev_had_all:
                GLOBAL_STATE["selected_images"] = []
                return gr.update(value=[])

            # Case 3: User deselected an individual image (All was on)
            if prev_had_all and curr_has_all and len(images_only) < len(all_images):
                new_val = images_only  # drop "All"
                GLOBAL_STATE["selected_images"] = new_val
                return gr.update(value=new_val)

            # Case 4: User selected the last remaining image → auto-add "All"
            if not curr_has_all and set(images_only) == set(all_images):
                GLOBAL_STATE["selected_images"] = all_with_all
                return gr.update(value=all_with_all)

            # Default: no change needed
            GLOBAL_STATE["selected_images"] = selected
            return gr.update()

        def run_pipeline():
            """Launch translation pipeline in a background thread."""
            folder_name = GLOBAL_STATE["folder_name"]
            selected_images = [s for s in GLOBAL_STATE["selected_images"] if s != "All"]

            if not selected_images:
                GLOBAL_STATE["log_text"] = "⚠️ No images selected."
                return sync_state_from_global()

            # Lock global state
            GLOBAL_STATE["is_running"] = True
            GLOBAL_STATE["log_text"] = f"🚀 Starting translation for {len(selected_images)} image(s) from '{folder_name}'...\n"
            GLOBAL_STATE["failed_text"] = ""
            GLOBAL_STATE["download_url"] = ""

            def _background_task():
                log_lines = [GLOBAL_STATE["log_text"]]
                final_result = None
                
                try:
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
                        GLOBAL_STATE["log_text"] = "\n".join(log_lines)
                        
                    if final_result is None:
                        final_result = TranslationResult()

                    # Summary line
                    log_lines.append(
                        f"\n{'═' * 40}\n"
                        f"🏁 Pipeline complete: {final_result.succeeded}/{final_result.total} images translated\n"
                        f"{'═' * 40}"
                    )
                    GLOBAL_STATE["log_text"] = "\n".join(log_lines)

                    # Failed images
                    failed_visible = len(final_result.failed_images) > 0
                    GLOBAL_STATE["failed_text"] = "\n".join(final_result.failed_images) if failed_visible else ""

                    # Download link
                    if final_result.download_url:
                        GLOBAL_STATE["download_url"] = final_result.download_url
                except Exception as e:
                    logger.error("Background task error: %s", e)
                    log_lines.append(f"\n❌ Pipeline crashed: {e}")
                    GLOBAL_STATE["log_text"] = "\n".join(log_lines)
                finally:
                    # Always unlock the global state when thread finishes
                    GLOBAL_STATE["is_running"] = False

            # Start thread and let Gradio handler exit, preventing websocket tie-up
            thread = threading.Thread(target=_background_task, daemon=True)
            thread.start()
            return sync_state_from_global()

        def authorize_action(password: str, action: str):
            action = (action or "").strip()
            is_valid, err_message = _validate_action_password(password)

            if not action:
                is_valid = False
                err_message = "⚠️ No action selected. Please try again."

            if not is_valid:
                return (
                    action,
                    gr.update(visible=True),
                    gr.update(value=_action_prompt(action or "translate")),
                    gr.update(value=""),
                    gr.update(
                        visible=True,
                        value=f'<span class="result-error">{err_message}</span>',
                    ),
                    *sync_state_from_global(),
                    "",
                    *_close_upload_modal(),
                )

            if action == "translate":
                sync_updates = run_pipeline()
                return (
                    *_close_auth_modal(),
                    *sync_updates,
                    "",
                    *_close_upload_modal(),
                )

            if action == "download":
                download_url = GLOBAL_STATE.get("download_url", "")
                if not download_url:
                    return (
                        action,
                        gr.update(visible=True),
                        gr.update(value=_action_prompt(action)),
                        gr.update(value=""),
                        gr.update(
                            visible=True,
                            value='<span class="result-error">❌ Download link is not ready yet.</span>',
                        ),
                        *sync_state_from_global(),
                        "",
                        *_close_upload_modal(),
                    )
                return (
                    *_close_auth_modal(),
                    *sync_state_from_global(),
                    download_url,
                    *_close_upload_modal(),
                )

            if action == "upload_zip":
                return (
                    *_close_auth_modal(),
                    *sync_state_from_global(),
                    "",
                    *_open_upload_modal(),
                )

            return (
                *_close_auth_modal(),
                *sync_state_from_global(),
                "",
                *_close_upload_modal(),
            )

        def confirm_zip_upload(zip_file_path: str | None):
            if not zip_file_path:
                return (
                    gr.update(visible=True),
                    gr.update(),
                    gr.update(visible=False, value=""),
                    gr.update(
                        visible=True,
                        value='<span class="result-error">⚠️ Please select a .zip file.</span>',
                    ),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                )

            try:
                folder_name, uploaded_count = upload_raw_manga_zip(
                    zip_path=Path(zip_file_path),
                    local_dir=settings.output_dir,
                )
                folder_status_update, image_section_update, image_selector_update = load_folder(
                    folder_name
                )

                return (
                    gr.update(visible=True),
                    gr.update(value=None),
                    gr.update(
                        visible=True,
                        value=(
                            '<span class="result-success">✅ Uploaded '
                            f'{uploaded_count} file(s) to raw-manga/<b>{folder_name}</b>/.</span>'
                        ),
                    ),
                    gr.update(visible=False, value=""),
                    gr.update(value=folder_name),
                    folder_status_update,
                    image_section_update,
                    image_selector_update,
                )
            except Exception as e:
                logger.error("Error uploading ZIP to raw-manga: %s", e)
                return (
                    gr.update(visible=True),
                    gr.update(value=None),
                    gr.update(visible=False, value=""),
                    gr.update(
                        visible=True,
                        value=f'<span class="result-error">❌ Upload failed: {e}</span>',
                    ),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                )

        # ── Wire Events ────────────────────────────────

        load_btn.click(
            fn=load_folder,
            inputs=[folder_input],
            outputs=[
                folder_status,
                image_section,
                image_selector,
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
            ],
        )

        image_selector.change(
            fn=on_image_selection_change,
            inputs=[image_selector],
            outputs=[image_selector],
        )
        
        # When app loads or timer tickets, sync the UI bounds to the backend global state.
        sync_outputs = [
            folder_input,
            load_btn,
            image_section,
            image_selector,
            translate_btn,
            results_section,
            progress_log,
            failed_images_box,
            download_btn,
            upload_zip_btn,
            timer,
        ]
        auth_outputs = [
            auth_action,
            auth_modal,
            auth_prompt,
            auth_password,
            auth_error,
        ]
        upload_modal_outputs = [
            upload_modal,
            upload_zip_file,
            upload_status,
            upload_error,
        ]
        
        app.load(
            fn=sync_state_from_global,
            inputs=None,
            outputs=sync_outputs
        )
        timer.tick(
            fn=sync_state_from_global,
            inputs=None,
            outputs=sync_outputs
        )

        translate_btn.click(
            fn=lambda: _open_auth_modal("translate"),
            inputs=[],
            outputs=auth_outputs,
            queue=False,
        )

        upload_zip_btn.click(
            fn=lambda: _open_auth_modal("upload_zip"),
            inputs=[],
            outputs=auth_outputs,
            queue=False,
        )

        download_btn.click(
            fn=lambda: _open_auth_modal("download"),
            inputs=[],
            outputs=auth_outputs,
            queue=False,
        )

        auth_cancel_btn.click(
            fn=_close_auth_modal,
            inputs=[],
            outputs=auth_outputs,
            queue=False,
        )

        auth_submit_outputs = [
            *auth_outputs,
            *sync_outputs,
            download_launch_url,
            *upload_modal_outputs,
        ]

        auth_submit_btn.click(
            fn=authorize_action,
            inputs=[auth_password, auth_action],
            outputs=auth_submit_outputs,
        )
        auth_password.submit(
            fn=authorize_action,
            inputs=[auth_password, auth_action],
            outputs=auth_submit_outputs,
        )

        upload_close_btn.click(
            fn=_close_upload_modal,
            inputs=[],
            outputs=upload_modal_outputs,
            queue=False,
        )

        upload_confirm_btn.click(
            fn=confirm_zip_upload,
            inputs=[upload_zip_file],
            outputs=[
                *upload_modal_outputs,
                folder_input,
                folder_status,
                image_section,
                image_selector,
            ],
        )

        download_launch_url.change(
            fn=lambda _url: "",
            inputs=[download_launch_url],
            outputs=[download_launch_url],
            js="""
                (url) => {
                    if (url) {
                        window.open(url, "_blank", "noopener");
                    }
                    return "";
                }
            """,
            queue=False,
        )

    return app, theme, CUSTOM_CSS
