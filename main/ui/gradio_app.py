"""
Gradio UI for the Manga Translator.

Provides a web interface to:
1. Load manga folders from GCP Cloud Storage
2. Select images for translation
3. Run the translation pipeline with live progress
4. Download translated results
"""

from __future__ import annotations

import hashlib
import hmac
import html
import inspect
import logging
import threading
import time
from pathlib import Path

import gradio as gr
from gradio.data_classes import FileData

from ..config import settings
from ..gcp_storage import (
    delete_manga_folder,
    folder_exists,
    list_images_in_folder,
    upload_raw_manga_zip,
)
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


def _keep_original_filename(component: gr.File) -> gr.File:
    """Make a single-file `gr.File` hand back the filename the user picked.

    Gradio strips characters such as `[`, `]` and `&` from the temp file it
    stores, but the browser still sends the untouched name as `orig_name`.
    It is attached to the returned path as `.orig_name`.

    This wraps the instance's `preprocess` instead of subclassing `gr.File`,
    because every Component subclass makes Gradio write a `.pyi` stub of the
    whole defining module at import time.
    """
    preprocess = component.preprocess

    def preprocess_with_original_name(payload):
        processed = preprocess(payload)
        if isinstance(payload, FileData) and isinstance(processed, str):
            processed.orig_name = payload.orig_name  # type: ignore[attr-defined]
        return processed

    component.preprocess = preprocess_with_original_name  # type: ignore[method-assign]
    return component


# ── Access control ─────────────────────────────────────
# The whole UI sits behind one password (GRADIO_ACTION_PASSWORD). Unlocking
# marks the session as unlocked server-side and stores a device token in the
# browser's localStorage, so the same device skips the lock screen next time.
#
# The token is an HMAC of the password, never the password itself. It changes
# when the password changes, which logs every remembered device out.
# BrowserState encrypts in the browser, so its secret is public. It is fixed
# only so stored tokens stay readable across app restarts.
# The stored value is {"token": ...}. BrowserState skips saving falsy values,
# so logging out has to write {"token": ""} rather than "".
_DEVICE_TOKEN_STORAGE_KEY = "manga-translator-ocr-device-token"
_DEVICE_TOKEN_STORAGE_SECRET = "manga-translator-ocr-browser-state-v1"
_DEVICE_TOKEN_CONTEXT = b"manga-translator-ocr/device-token/v1"
_FAILED_UNLOCK_DELAY_SECONDS = 1.0
_MISSING_PASSWORD_MESSAGE = (
    "⚠️ GRADIO_ACTION_PASSWORD is not configured in .env. Set it and restart the app."
)


def _configured_password() -> str:
    return settings.gradio_action_password.strip()


def _device_token(password: str) -> str:
    return hmac.new(
        password.encode("utf-8"), _DEVICE_TOKEN_CONTEXT, hashlib.sha256
    ).hexdigest()


def _check_password(password: str) -> tuple[bool, str]:
    configured_password = _configured_password()
    if not configured_password:
        return False, _MISSING_PASSWORD_MESSAGE
    if not password:
        return False, "⚠️ Password is required."
    if not hmac.compare_digest(password.encode("utf-8"), configured_password.encode("utf-8")):
        return False, "❌ Incorrect password."
    return True, ""


def _stored_device(token: str = "") -> dict[str, str]:
    return {"token": token}


def _device_token_is_valid(stored: object) -> bool:
    token = stored.get("token") if isinstance(stored, dict) else None
    configured_password = _configured_password()
    if not configured_password or not isinstance(token, str) or not token:
        return False
    return hmac.compare_digest(token, _device_token(configured_password))


def _ensure_unlocked(is_unlocked: object) -> None:
    if is_unlocked is not True:
        raise gr.Error("🔒 This session is locked. Reload the page and enter the password.")


def _require_unlocked(fn):
    """Wrap an event handler so it only runs for an unlocked session.

    The wrapped handler takes the session's `unlocked` State as its first
    input, followed by the handler's own inputs. Hiding the UI is not enough on
    its own, because Gradio event endpoints can be called directly.
    """
    if inspect.isgeneratorfunction(fn):

        def guarded_generator(is_unlocked, *args):
            _ensure_unlocked(is_unlocked)
            yield from fn(*args)

        return guarded_generator

    def guarded(is_unlocked, *args):
        _ensure_unlocked(is_unlocked)
        return fn(*args)

    return guarded


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
        # Per-session flag, set only after the password or device token checks out.
        unlocked = gr.State(False)
        remembered_token = gr.BrowserState(
            _stored_device(),
            storage_key=_DEVICE_TOKEN_STORAGE_KEY,
            secret=_DEVICE_TOKEN_STORAGE_SECRET,
        )

        # ── Header ─────────────────────────────────────
        gr.HTML(
            """
            <div class="app-header">
                <h1>🎌 Manga Translator OCR</h1>
                <p>Translate manga pages from Japanese to English via GCP Cloud Storage</p>
            </div>
            """
        )

        # ── Lock screen ────────────────────────────────
        # Both views start hidden; app.load shows one of them once the
        # remembered device token (if any) has been checked.
        checking_access = gr.HTML('<div class="lock-checking">🔒 Checking access…</div>')
        with gr.Column(visible=False, elem_classes=["lock-screen"]) as lock_screen:
            with gr.Group(elem_classes=["section-card", "lock-card"]):
                gr.HTML('<div class="section-title">🔒 Enter Password</div>')
                gr.HTML(
                    '<div class="auth-modal-message">Unlock once and this device '
                    "is remembered.</div>"
                )
                unlock_password = gr.Textbox(
                    label="Password",
                    type="password",
                    placeholder="Enter password",
                    interactive=True,
                )
                unlock_error = gr.HTML(visible=False)
                unlock_btn = gr.Button(
                    "🔓 Unlock",
                    variant="primary",
                    elem_classes=["primary-btn"],
                )

        # ── Main view (only visible once unlocked) ─────
        with gr.Column(visible=False) as main_view:
            with gr.Row(elem_classes=["top-actions-row"]):
                upload_zip_btn = gr.Button(
                    "📤 Upload ZIP to raw-manga",
                    variant="secondary",
                    elem_classes=["upload-zip-btn"],
                )
                delete_folder_btn = gr.Button(
                    "🗑️ Delete Folder",
                    variant="stop",
                    elem_classes=["delete-folder-btn"],
                )
                clear_btn = gr.Button(
                    "🧹 Clear",
                    variant="secondary",
                    elem_classes=["clear-btn"],
                    interactive=True,
                )
                lock_btn = gr.Button(
                    "🔒 Log out",
                    variant="secondary",
                    elem_classes=["lock-btn"],
                    scale=0,
                    min_width=120,
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

            # ZIP upload popup.
            with gr.Group(visible=False, elem_classes=["auth-modal-overlay"]) as upload_modal:
                with gr.Group(elem_classes=["auth-modal-card"]):
                    gr.HTML('<div class="auth-modal-title">📦 Upload Manga ZIP</div>')
                    gr.HTML(
                        '<div class="auth-modal-message">Select a .zip file. '
                        'It will be uploaded to raw-manga/&lt;zip-name&gt;/.</div>'
                    )
                    upload_zip_file = _keep_original_filename(
                        gr.File(
                            label="ZIP file",
                            file_types=[".zip"],
                            type="filepath",
                        )
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

            # Folder deletion popup.
            with gr.Group(visible=False, elem_classes=["auth-modal-overlay"]) as delete_modal:
                with gr.Group(elem_classes=["auth-modal-card"]):
                    gr.HTML('<div class="auth-modal-title">🗑️ Delete Manga Folder</div>')
                    gr.HTML(
                        '<div class="auth-modal-message">Delete folder from both '
                        "raw-manga/ and translated-manga/.</div>"
                    )
                    delete_folder_name = gr.Textbox(
                        label="Folder name",
                        placeholder="Enter folder name to delete",
                        interactive=True,
                    )
                    delete_status = gr.HTML(visible=False)
                    delete_error = gr.HTML(visible=False)
                    with gr.Row():
                        delete_confirm_btn = gr.Button(
                            "Confirm",
                            variant="stop",
                            elem_classes=["delete-folder-btn"],
                        )
                        delete_close_btn = gr.Button("Close")


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

        def _close_delete_modal():
            return (
                gr.update(visible=False),
                gr.update(value=""),
                gr.update(visible=False, value=""),
                gr.update(visible=False, value=""),
            )

        def _open_delete_modal():
            default_folder = GLOBAL_STATE.get("folder_name", "")
            return (
                gr.update(visible=True),
                gr.update(value=default_folder),
                gr.update(visible=False, value=""),
                gr.update(visible=False, value=""),
            )

        def _operation_controls(is_busy: bool):
            return (
                gr.update(interactive=not is_busy),  # upload_zip_btn
                gr.update(interactive=not is_busy),  # delete_folder_btn
                gr.update(interactive=not is_busy),  # upload_confirm_btn
                gr.update(interactive=not is_busy),  # delete_confirm_btn
                gr.update(interactive=not is_busy),  # clear_btn
            )

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
                gr.update(interactive=not is_running), # delete_folder_btn
                gr.update(interactive=not is_running), # clear_btn
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
                            value=f'<span class="result-error">❌ Folder "<b>{html.escape(folder_name)}</b>" not found in raw-manga/.</span>',
                        ),
                        gr.update(visible=False),
                        gr.update(choices=[], value=[]),
                    )

                images = list_images_in_folder(folder_name)
                if not images:
                    return (
                        gr.update(
                            visible=True,
                            value=f'<span class="result-error">⚠️ Folder "<b>{html.escape(folder_name)}</b>" has no images.</span>',
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
                        value=f'<span class="result-success">✅ Found {len(images)} image(s) in "<b>{html.escape(folder_name)}</b>"</span>',
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

            if GLOBAL_STATE.get("is_running"):
                return sync_state_from_global()

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

        def confirm_zip_upload(zip_file_path: str | None):
            # Disable upload/delete/clear actions while upload is in progress.
            yield (
                gr.update(visible=True),
                gr.update(),
                gr.update(visible=True, value="⏳ Uploading ZIP..."),
                gr.update(visible=False, value=""),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                *_operation_controls(True),
            )

            if not zip_file_path:
                yield (
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
                    *_operation_controls(False),
                )
                return

            try:
                folder_name, uploaded_count = upload_raw_manga_zip(
                    zip_path=Path(zip_file_path),
                    local_dir=settings.output_dir,
                    original_filename=getattr(zip_file_path, "orig_name", None),
                )
                folder_status_update, image_section_update, image_selector_update = load_folder(
                    folder_name
                )

                yield (
                    gr.update(visible=True),
                    gr.update(value=None),
                    gr.update(
                        visible=True,
                        value=(
                            '<span class="result-success">✅ Uploaded '
                            f'{uploaded_count} file(s) to raw-manga/<b>{html.escape(folder_name)}</b>/.</span>'
                        ),
                    ),
                    gr.update(visible=False, value=""),
                    gr.update(value=folder_name),
                    folder_status_update,
                    image_section_update,
                    image_selector_update,
                    *_operation_controls(False),
                )
            except Exception as e:
                logger.error("Error uploading ZIP to raw-manga: %s", e)
                yield (
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
                    *_operation_controls(False),
                )

        def confirm_delete_folder(folder_name: str):
            # Disable upload/delete/clear actions while delete is in progress.
            yield (
                gr.update(visible=True),
                gr.update(),
                gr.update(visible=True, value="⏳ Deleting folder..."),
                gr.update(visible=False, value=""),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                *_operation_controls(True),
            )

            normalized = (folder_name or "").strip().strip("/")
            if not normalized:
                yield (
                    gr.update(visible=True),
                    gr.update(),
                    gr.update(visible=False, value=""),
                    gr.update(
                        visible=True,
                        value='<span class="result-error">⚠️ Please enter a folder name.</span>',
                    ),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    *_operation_controls(False),
                )
                return

            try:
                raw_deleted, translated_deleted = delete_manga_folder(normalized)
                total_deleted = raw_deleted + translated_deleted
                if total_deleted == 0:
                    yield (
                        gr.update(visible=True),
                        gr.update(value=normalized),
                        gr.update(visible=False, value=""),
                        gr.update(
                            visible=True,
                            value=(
                                '<span class="result-error">⚠️ Folder '
                                f'"<b>{html.escape(normalized)}</b>" was not found in raw-manga/ '
                                "or translated-manga/.</span>"
                            ),
                        ),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        *_operation_controls(False),
                    )
                    return

                if GLOBAL_STATE.get("folder_name") == normalized:
                    GLOBAL_STATE["folder_name"] = ""
                    GLOBAL_STATE["loaded_images"] = []
                    GLOBAL_STATE["selected_images"] = []
                    folder_input_update = gr.update(value="")
                    folder_status_update = gr.update(visible=False, value="")
                    image_section_update = gr.update(visible=False)
                    image_selector_update = gr.update(choices=[], value=[])
                else:
                    folder_input_update = gr.update()
                    folder_status_update = gr.update()
                    image_section_update = gr.update()
                    image_selector_update = gr.update()

                yield (
                    gr.update(visible=True),
                    gr.update(value=normalized),
                    gr.update(
                        visible=True,
                        value=(
                            '<span class="result-success">✅ Deleted '
                            f'{raw_deleted} file(s) from raw-manga/<b>{html.escape(normalized)}</b>/ and '
                            f'{translated_deleted} file(s) from translated-manga/<b>{html.escape(normalized)}</b>/.</span>'
                        ),
                    ),
                    gr.update(visible=False, value=""),
                    folder_input_update,
                    folder_status_update,
                    image_section_update,
                    image_selector_update,
                    *_operation_controls(False),
                )
            except Exception as e:
                logger.error("Error deleting folder from GCS: %s", e)
                yield (
                    gr.update(visible=True),
                    gr.update(value=normalized),
                    gr.update(visible=False, value=""),
                    gr.update(
                        visible=True,
                        value=f'<span class="result-error">❌ Delete failed: {e}</span>',
                    ),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    *_operation_controls(False),
                )

        def get_download_url() -> str:
            download_url = GLOBAL_STATE.get("download_url", "")
            if not download_url:
                raise gr.Error("Download link is not ready yet.")
            return download_url

        def clear_all_ui():
            GLOBAL_STATE["folder_name"] = ""
            GLOBAL_STATE["loaded_images"] = []
            GLOBAL_STATE["selected_images"] = []
            GLOBAL_STATE["is_running"] = False
            GLOBAL_STATE["log_text"] = ""
            GLOBAL_STATE["failed_text"] = ""
            GLOBAL_STATE["download_url"] = ""

            return (
                gr.update(visible=False, value=""),  # folder_status
                *_close_upload_modal(),
                *_close_delete_modal(),
                "",  # download_launch_url
                *sync_state_from_global(),
            )

        # ── Lock screen handlers ───────────────────────

        def _access_views(is_unlocked: bool, error: str = ""):
            return (
                gr.update(visible=False),  # checking_access
                gr.update(visible=not is_unlocked),  # lock_screen
                gr.update(visible=is_unlocked),  # main_view
                gr.update(
                    visible=bool(error),
                    value=f'<span class="result-error">{error}</span>' if error else "",
                ),  # unlock_error
            )

        def restore_access(stored: dict):
            """On page load, unlock straight away if this device is remembered."""
            if _device_token_is_valid(stored):
                return (True, *_access_views(True))
            error = "" if _configured_password() else _MISSING_PASSWORD_MESSAGE
            return (False, *_access_views(False, error))

        def unlock(password: str):
            is_valid, error = _check_password(password)
            if not is_valid:
                if password and _configured_password():
                    # Slow down password guessing.
                    time.sleep(_FAILED_UNLOCK_DELAY_SECONDS)
                return (
                    False,
                    _stored_device(),
                    *_access_views(False, error),
                    gr.update(value=""),
                )
            token = _device_token(_configured_password())
            return (True, _stored_device(token), *_access_views(True), gr.update(value=""))

        def lock():
            """Lock this session and forget the device."""
            return (
                False,
                _stored_device(),
                *_access_views(False),
                gr.update(value=""),
                *_close_upload_modal(),
                *_close_delete_modal(),
            )

        def sync_for_session(is_unlocked: bool):
            """Sync the UI from global state, but only for unlocked sessions."""
            if is_unlocked is not True:
                return tuple(gr.update() for _ in sync_outputs)
            return sync_state_from_global()

        def select_images_for_session(is_unlocked: bool, selected: list[str]):
            # Value changes made by syncs also fire `.change`, so a locked
            # session is skipped quietly here instead of raising.
            if is_unlocked is not True:
                return gr.update()
            return on_image_selection_change(selected)

        # ── Wire Events ────────────────────────────────

        # When the app loads or the timer ticks, sync the UI to the global state.
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
            delete_folder_btn,
            clear_btn,
            timer,
        ]
        upload_modal_outputs = [
            upload_modal,
            upload_zip_file,
            upload_status,
            upload_error,
        ]
        delete_modal_outputs = [
            delete_modal,
            delete_folder_name,
            delete_status,
            delete_error,
        ]
        access_view_outputs = [
            checking_access,
            lock_screen,
            main_view,
            unlock_error,
        ]
        clear_outputs = [
            folder_status,
            *upload_modal_outputs,
            *delete_modal_outputs,
            download_launch_url,
            *sync_outputs,
        ]
        folder_load_outputs = [
            folder_status,
            image_section,
            image_selector,
        ]

        app.load(
            fn=restore_access,
            inputs=[remembered_token],
            outputs=[unlocked, *access_view_outputs],
            queue=False,
        ).then(
            fn=sync_for_session,
            inputs=[unlocked],
            outputs=sync_outputs,
        )
        timer.tick(
            fn=sync_for_session,
            inputs=[unlocked],
            outputs=sync_outputs,
        )

        unlock_outputs = [
            unlocked,
            remembered_token,
            *access_view_outputs,
            unlock_password,
        ]
        for unlock_trigger in (unlock_btn.click, unlock_password.submit):
            unlock_trigger(
                fn=unlock,
                inputs=[unlock_password],
                outputs=unlock_outputs,
            ).then(
                fn=sync_for_session,
                inputs=[unlocked],
                outputs=sync_outputs,
            )

        lock_btn.click(
            fn=lock,
            inputs=[],
            outputs=[*unlock_outputs, *upload_modal_outputs, *delete_modal_outputs],
            queue=False,
        )

        # Also trigger load on Enter key
        for load_trigger in (load_btn.click, folder_input.submit):
            load_trigger(
                fn=_require_unlocked(load_folder),
                inputs=[unlocked, folder_input],
                outputs=folder_load_outputs,
            )

        image_selector.change(
            fn=select_images_for_session,
            inputs=[unlocked, image_selector],
            outputs=[image_selector],
        )

        translate_btn.click(
            fn=_require_unlocked(run_pipeline),
            inputs=[unlocked],
            outputs=sync_outputs,
        )

        upload_zip_btn.click(
            fn=_require_unlocked(_open_upload_modal),
            inputs=[unlocked],
            outputs=upload_modal_outputs,
            queue=False,
        )

        delete_folder_btn.click(
            fn=_require_unlocked(_open_delete_modal),
            inputs=[unlocked],
            outputs=delete_modal_outputs,
            queue=False,
        )

        download_btn.click(
            fn=_require_unlocked(get_download_url),
            inputs=[unlocked],
            outputs=[download_launch_url],
            queue=False,
        )

        upload_close_btn.click(
            fn=_close_upload_modal,
            inputs=[],
            outputs=upload_modal_outputs,
            queue=False,
        )

        upload_confirm_btn.click(
            fn=_require_unlocked(confirm_zip_upload),
            inputs=[unlocked, upload_zip_file],
            outputs=[
                *upload_modal_outputs,
                folder_input,
                folder_status,
                image_section,
                image_selector,
                upload_zip_btn,
                delete_folder_btn,
                upload_confirm_btn,
                delete_confirm_btn,
                clear_btn,
            ],
        )

        delete_close_btn.click(
            fn=_close_delete_modal,
            inputs=[],
            outputs=delete_modal_outputs,
            queue=False,
        )

        delete_confirm_btn.click(
            fn=_require_unlocked(confirm_delete_folder),
            inputs=[unlocked, delete_folder_name],
            outputs=[
                *delete_modal_outputs,
                folder_input,
                folder_status,
                image_section,
                image_selector,
                upload_zip_btn,
                delete_folder_btn,
                upload_confirm_btn,
                delete_confirm_btn,
                clear_btn,
            ],
        )

        clear_btn.click(
            fn=_require_unlocked(clear_all_ui),
            inputs=[unlocked],
            outputs=clear_outputs,
            queue=False,
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
