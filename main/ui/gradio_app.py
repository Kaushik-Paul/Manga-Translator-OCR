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
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
import pandas as pd
from gradio.data_classes import FileData

from ..config import settings
from ..gcp_storage import (
    TranslatedArchive,
    archive_download_url,
    delete_manga_folder,
    folder_exists,
    list_images_in_folder,
    list_translated_archives,
    mark_archive_downloaded,
    translated_page_urls,
    upload_raw_manga_zip,
)
from .pipeline_runner import TranslationResult, run_translation_pipeline
from .styles import CUSTOM_CSS, build_theme

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
    # Folder the current download_url belongs to.
    "download_folder": "",
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

        return _copy_signature(guarded_generator, fn)

    def guarded(is_unlocked, *args):
        _ensure_unlocked(is_unlocked)
        return fn(*args)

    return _copy_signature(guarded, fn)


def _copy_signature(guarded, fn):
    """Give `guarded` the signature of `fn` with `is_unlocked` in front.

    Gradio reads a handler's signature and type hints to inject special
    arguments such as `gr.SelectData`. Without this, a guarded handler would
    never receive them.
    """
    signature = inspect.signature(fn)
    unlocked_param = inspect.Parameter("is_unlocked", inspect.Parameter.POSITIONAL_OR_KEYWORD)
    guarded.__signature__ = signature.replace(
        parameters=[unlocked_param, *signature.parameters.values()]
    )
    guarded.__annotations__ = dict(getattr(fn, "__annotations__", {}))
    return guarded


_HEADER_HTML = """
<header class="mt-hero">
    <div class="mt-brand">
        <div class="mt-logo" aria-hidden="true">訳</div>
        <div>
            <h1>Manga Translator <span>OCR</span></h1>
            <p>Japanese manga pages to clean English lettering, straight from Cloud Storage.</p>
        </div>
    </div>
    <ol class="mt-pipeline" aria-label="Pipeline stages">
        <li>Detect</li>
        <li>OCR</li>
        <li>Translate</li>
        <li>Inpaint</li>
        <li>Render</li>
    </ol>
</header>
"""


def _card_header(step: str, title: str, subtitle: str) -> str:
    """HTML for a numbered step card's heading."""
    return (
        '<div class="mt-card-head">'
        f'<span class="mt-step" aria-hidden="true">{step}</span>'
        f"<div><h3>{title}</h3><p>{subtitle}</p></div>"
        "</div>"
    )


# Run status shown next to the workspace title: (css modifier, label).
_RUN_STATUSES = {
    "idle": ("idle", "Ready"),
    "running": ("running", "Translating…"),
    "done": ("done", "Download ready"),
    "finished": ("finished", "Finished"),
}


def _run_status() -> str:
    """Pick the run status key from the shared global state."""
    if GLOBAL_STATE.get("is_running"):
        return "running"
    if GLOBAL_STATE.get("download_url"):
        return "done"
    if GLOBAL_STATE.get("log_text"):
        return "finished"
    return "idle"


def _workspace_header(status: str = "idle") -> str:
    """HTML for the toolbar's title, description and run status badge."""
    modifier, label = _RUN_STATUSES[status]
    return (
        '<div class="mt-toolbar-title">'
        '<div class="mt-toolbar-row"><h2>Workspace</h2>'
        f'<span class="mt-badge mt-badge-{modifier}" role="status">'
        f'<span class="mt-badge-dot"></span>{label}</span></div>'
        "<p>Load a raw-manga folder, pick pages, translate.</p>"
        "</div>"
    )


# ── Translated library ─────────────────────────────────
_LIBRARY_COLUMNS = ("Chapter", "Pages", "Size", "Translated", "Downloaded")


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _relative_time(moment: datetime | None) -> str:
    """Describe a timestamp as "5 min ago", "3 days ago" or a date."""
    if moment is None:
        return "—"
    seconds = (datetime.now(timezone.utc) - moment).total_seconds()
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)} min ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)} h ago"
    if seconds < 30 * 86400:
        days = int(seconds // 86400)
        return f"{days} day{'s' if days != 1 else ''} ago"
    return moment.strftime("%d %b %Y")


def _library_frame(archives: list[TranslatedArchive]) -> pd.DataFrame:
    rows = [
        [
            archive.folder_name,
            archive.page_count,
            _human_size(archive.archive_size),
            _relative_time(archive.translated_at),
            f"✅ {_relative_time(archive.downloaded_at)}" if archive.downloaded_at else "Not yet",
        ]
        for archive in archives
    ]
    return pd.DataFrame(rows, columns=list(_LIBRARY_COLUMNS))


def _library_summary(archives: list[TranslatedArchive] | None) -> str:
    if archives is None:
        return '<p class="mt-library-count">Open this tab to load the library.</p>'
    total = sum(archive.archive_size for archive in archives)
    pending = sum(1 for archive in archives if archive.downloaded_at is None)
    count = f"{len(archives)} archive{'s' if len(archives) != 1 else ''}"
    return (
        '<p class="mt-library-count">'
        f"<b>{count}</b> · {_human_size(total)} · "
        f"<b>{pending}</b> not downloaded yet</p>"
    )


def _preview_header(archive: TranslatedArchive) -> str:
    """HTML for the preview card: name, facts and the downloaded badge."""
    if archive.downloaded_at:
        badge = (
            '<span class="mt-badge mt-badge-done"><span class="mt-badge-dot"></span>'
            f"Downloaded {_relative_time(archive.downloaded_at)}</span>"
        )
    else:
        badge = (
            '<span class="mt-badge mt-badge-finished"><span class="mt-badge-dot"></span>'
            "Not downloaded yet</span>"
        )
    return (
        '<div class="mt-preview-head">'
        f"<h3>{html.escape(archive.folder_name)}</h3>"
        '<div class="mt-meta">'
        f"<span>{archive.page_count} pages</span>"
        f"<span>{_human_size(archive.archive_size)}</span>"
        f"<span>Translated {_relative_time(archive.translated_at)}</span>"
        f"{badge}</div></div>"
    )


def _modal_header(icon: str, title: str, message: str, danger: bool = False) -> str:
    """HTML for a popup's icon, title and explanation."""
    tone = " mt-modal-icon-danger" if danger else ""
    return (
        '<div class="mt-modal-head">'
        f'<span class="mt-modal-icon{tone}" aria-hidden="true">{icon}</span>'
        f'<div><div class="auth-modal-title">{title}</div>'
        f'<div class="auth-modal-message">{message}</div></div>'
        "</div>"
    )


def create_app() -> tuple[gr.Blocks, gr.themes.Base, str]:
    """Build and return the Gradio Blocks application, theme, and CSS."""

    theme = build_theme()

    with gr.Blocks(
        title="Manga Translator OCR",
    ) as app:

        # Hidden state component to trigger auto-updates
        timer = gr.Timer(value=1.0, active=False)
        # Per-session flag, set only after the password or device token checks out.
        unlocked = gr.State(False)
        # Library rows for this session, and the folder picked for preview.
        library_archives = gr.State([])
        library_selected = gr.State("")
        remembered_token = gr.BrowserState(
            _stored_device(),
            storage_key=_DEVICE_TOKEN_STORAGE_KEY,
            secret=_DEVICE_TOKEN_STORAGE_SECRET,
        )

        # ── Header ─────────────────────────────────────
        gr.HTML(_HEADER_HTML, elem_classes=["mt-bare"])

        # ── Lock screen ────────────────────────────────
        # Both views start hidden; app.load shows one of them once the
        # remembered device token (if any) has been checked.
        checking_access = gr.HTML(
            '<div class="lock-checking"><span class="mt-spinner"></span>Checking access…</div>'
        )
        with gr.Column(visible=False, elem_classes=["lock-screen"]) as lock_screen:
            with gr.Column(elem_classes=["lock-card"]):
                gr.HTML(
                    """
                    <div class="lock-head">
                        <div class="lock-badge" aria-hidden="true">🔒</div>
                        <h2>Unlock the workspace</h2>
                        <p>Enter the access password. This device is remembered after you unlock.</p>
                    </div>
                    """,
                    elem_classes=["mt-bare"],
                )
                unlock_password = gr.Textbox(
                    label="Password",
                    type="password",
                    placeholder="Enter password",
                    interactive=True,
                )
                unlock_error = gr.HTML(visible=False)
                unlock_btn = gr.Button(
                    "Unlock",
                    variant="primary",
                    size="lg",
                )

        # ── Main view (only visible once unlocked) ─────
        with gr.Column(visible=False, elem_classes=["mt-main"]) as main_view:
            # ── Toolbar ────────────────────────────────────
            with gr.Row(elem_classes=["mt-toolbar"], equal_height=True):
                workspace_header = gr.HTML(
                    _workspace_header(),
                    elem_classes=["mt-bare", "mt-toolbar-head"],
                )
                with gr.Row(elem_classes=["mt-toolbar-actions"]):
                    upload_zip_btn = gr.Button(
                        "📤 Upload ZIP",
                        variant="secondary",
                        size="md",
                        scale=0,
                        min_width=0,
                    )
                    delete_folder_btn = gr.Button(
                        "🗑️ Delete folder",
                        variant="stop",
                        size="md",
                        scale=0,
                        min_width=0,
                    )
                    clear_btn = gr.Button(
                        "🧹 Clear",
                        variant="secondary",
                        size="md",
                        scale=0,
                        min_width=0,
                        interactive=True,
                    )
                    lock_btn = gr.Button(
                        "🔒 Log out",
                        variant="secondary",
                        size="md",
                        scale=0,
                        min_width=0,
                        elem_classes=["lock-btn"],
                    )

            with gr.Tabs(elem_classes=["mt-tabs"]):
                with gr.Tab("✨ Translate", id="translate", elem_classes=["mt-tab"]):
                    with gr.Row(elem_classes=["mt-columns"]):
                        with gr.Column(scale=5, min_width=320, elem_classes=["mt-stack"]):
                            # ── Step 1: Folder ─────────────────────────────
                            with gr.Column(elem_classes=["mt-card"]):
                                gr.HTML(
                                    _card_header(
                                        "1",
                                        "Choose a folder",
                                        "Name of a folder in the raw-manga bucket.",
                                    ),
                                    elem_classes=["mt-bare"],
                                )
                                with gr.Row(elem_classes=["mt-folder-row"], equal_height=True):
                                    folder_input = gr.Textbox(
                                        label="Folder name",
                                        show_label=False,
                                        placeholder="e.g. my-manga-chapter-01",
                                        scale=4,
                                        min_width=180,
                                        interactive=True,
                                        container=False,
                                    )
                                    load_btn = gr.Button(
                                        "🔍 Load",
                                        variant="primary",
                                        scale=0,
                                        min_width=110,
                                        interactive=True,
                                    )
                                folder_status = gr.HTML(visible=False, elem_classes=["mt-status"])

                            # ── Step 2: Pages ──────────────────────────────
                            with gr.Column(elem_classes=["mt-card"], visible=False) as image_section:
                                gr.HTML(
                                    _card_header(
                                        "2",
                                        "Pick pages",
                                        "Every page is selected. Untick the ones to skip.",
                                    ),
                                    elem_classes=["mt-bare"],
                                )
                                image_selector = gr.CheckboxGroup(
                                    label="Images to translate",
                                    show_label=False,
                                    choices=[],
                                    value=[],
                                    elem_classes=["image-selector"],
                                    interactive=True,
                                )
                                translate_btn = gr.Button(
                                    "🚀 Translate selected pages",
                                    variant="primary",
                                    size="lg",
                                    elem_classes=["translate-btn"],
                                    interactive=True,
                                )

                        # ── Step 3: Progress & results ─────────────────
                        with gr.Column(
                            scale=6,
                            min_width=320,
                            visible=False,
                            elem_classes=["mt-card", "mt-results"],
                        ) as results_section:
                            gr.HTML(
                                _card_header(
                                    "3",
                                    "Progress & results",
                                    "Live pipeline log. The download link appears when it finishes.",
                                ),
                                elem_classes=["mt-bare"],
                            )
                            progress_log = gr.Textbox(
                                label="Pipeline log",
                                show_label=False,
                                lines=14,
                                max_lines=30,
                                interactive=False,
                                autoscroll=True,
                                elem_classes=["progress-box"],
                            )
                            failed_images_box = gr.Textbox(
                                label="Failed images",
                                lines=3,
                                interactive=False,
                                visible=False,
                                elem_classes=["failed-box"],
                            )
                            download_btn = gr.Button(
                                "📥 Download translated manga (CBZ) · link valid 24 h",
                                variant="primary",
                                size="lg",
                                elem_classes=["download-btn"],
                                interactive=True,
                                visible=False,
                            )

                with gr.Tab("📚 Library", id="library", elem_classes=["mt-tab"]) as library_tab:
                    with gr.Column(elem_classes=["mt-stack"]):
                        with gr.Column(elem_classes=["mt-card", "mt-library"]):
                            with gr.Row(elem_classes=["mt-library-bar"], equal_height=True):
                                gr.HTML(
                                    _card_header(
                                        "📚",
                                        "Translated library",
                                        "Every converted CBZ in translated-manga/. "
                                        "Click a row to preview it.",
                                    ),
                                    elem_classes=["mt-bare", "mt-library-head"],
                                )
                                library_refresh_btn = gr.Button(
                                    "🔄 Refresh",
                                    variant="secondary",
                                    size="sm",
                                    scale=0,
                                    min_width=0,
                                )
                            library_summary = gr.HTML(
                                _library_summary(None),
                                elem_classes=["mt-bare"],
                            )
                            library_table = gr.Dataframe(
                                value=_library_frame([]),
                                headers=list(_LIBRARY_COLUMNS),
                                datatype=["str", "number", "str", "str", "str"],
                                interactive=False,
                                show_label=False,
                                show_search="search",
                                wrap=True,
                                max_height=440,
                                column_widths=["44%", "10%", "13%", "16%", "17%"],
                                elem_classes=["mt-library-table"],
                            )
                            library_status = gr.HTML(visible=False, elem_classes=["mt-status"])

                        with gr.Column(
                            visible=False,
                            elem_classes=["mt-card", "mt-preview"],
                        ) as preview_section:
                            with gr.Row(elem_classes=["mt-preview-bar"], equal_height=True):
                                preview_header = gr.HTML(
                                    elem_classes=["mt-bare", "mt-preview-headblock"],
                                )
                                preview_download_btn = gr.Button(
                                    "📥 Download CBZ",
                                    variant="primary",
                                    scale=0,
                                    min_width=190,
                                    elem_classes=["download-btn"],
                                )
                            preview_gallery = gr.Gallery(
                                label="Pages",
                                show_label=False,
                                columns=6,
                                height=620,
                                object_fit="contain",
                                allow_preview=True,
                                interactive=False,
                                elem_classes=["mt-preview-gallery"],
                            )

            # Hidden trigger value. JS listener opens this URL in a new tab.
            download_launch_url = gr.Textbox(value="", visible=False, interactive=False)

            # ZIP upload popup.
            with gr.Column(visible=False, elem_classes=["auth-modal-overlay"]) as upload_modal:
                with gr.Column(elem_classes=["auth-modal-card"]):
                    gr.HTML(
                        _modal_header(
                            "📦",
                            "Upload a manga ZIP",
                            "The pages are uploaded to raw-manga/&lt;zip-name&gt;/ and "
                            "loaded straight away.",
                        ),
                        elem_classes=["mt-bare"],
                    )
                    upload_zip_file = _keep_original_filename(
                        gr.File(
                            label="ZIP file",
                            file_types=[".zip"],
                            type="filepath",
                            height=180,
                            elem_classes=["mt-dropzone"],
                        )
                    )
                    upload_status = gr.HTML(visible=False, elem_classes=["mt-status"])
                    upload_error = gr.HTML(visible=False, elem_classes=["mt-status"])
                    with gr.Row(elem_classes=["mt-modal-actions"]):
                        upload_close_btn = gr.Button("Close", variant="secondary")
                        upload_confirm_btn = gr.Button("Upload", variant="primary")

            # Folder deletion popup.
            with gr.Column(visible=False, elem_classes=["auth-modal-overlay"]) as delete_modal:
                with gr.Column(elem_classes=["auth-modal-card"]):
                    gr.HTML(
                        _modal_header(
                            "🗑️",
                            "Delete a manga folder",
                            "Removes the folder from both raw-manga/ and "
                            "translated-manga/. This cannot be undone.",
                            danger=True,
                        ),
                        elem_classes=["mt-bare"],
                    )
                    delete_folder_name = gr.Textbox(
                        label="Folder name",
                        placeholder="Enter folder name to delete",
                        interactive=True,
                    )
                    delete_status = gr.HTML(visible=False, elem_classes=["mt-status"])
                    delete_error = gr.HTML(visible=False, elem_classes=["mt-status"])
                    with gr.Row(elem_classes=["mt-modal-actions"]):
                        delete_close_btn = gr.Button("Close", variant="secondary")
                        delete_confirm_btn = gr.Button("Delete", variant="stop")


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
                gr.update(active=is_running), # timer active while running
                _workspace_header(_run_status()), # workspace_header
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
            GLOBAL_STATE["download_folder"] = ""

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
                        GLOBAL_STATE["download_folder"] = folder_name
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
            download_folder = GLOBAL_STATE.get("download_folder", "")
            if download_folder:
                # Marked in the background so the link opens straight away.
                threading.Thread(
                    target=mark_archive_downloaded, args=(download_folder,), daemon=True
                ).start()
            return download_url

        # ── Library handlers ───────────────────────────

        def _library_error(message: str):
            return gr.update(
                visible=True,
                value=f'<span class="result-error">{html.escape(message)}</span>',
            )

        def load_library():
            """List the converted CBZs in the bucket."""
            try:
                archives = list_translated_archives()
            except Exception as e:
                logger.error("Error listing translated archives: %s", e)
                return (
                    gr.update(),
                    gr.update(),
                    _library_error(f"❌ Could not load the library: {e}"),
                    gr.update(),
                )
            if archives:
                status = gr.update(visible=False, value="")
            else:
                status = gr.update(
                    visible=True,
                    value=(
                        '<span class="result-error">No converted CBZ in translated-manga/ '
                        "yet. Translate a folder and it shows up here.</span>"
                    ),
                )
            return _library_frame(archives), _library_summary(archives), status, archives

        def select_archive(archives: list, evt: gr.SelectData):
            """Show the preview of the clicked row's archive."""
            row = evt.row_value or []
            folder_name = str(row[0]) if row else ""
            archive = next((a for a in archives or [] if a.folder_name == folder_name), None)
            if archive is None:
                return (
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    _library_error("⚠️ That archive is gone. Refresh the library."),
                )
            try:
                pages = translated_page_urls(folder_name)
            except Exception as e:
                logger.error("Error loading preview for %s: %s", folder_name, e)
                return (
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    _library_error(f"❌ Could not load the preview: {e}"),
                )
            return (
                folder_name,
                gr.update(visible=True),
                _preview_header(archive),
                [(url, name) for name, url in pages],
                gr.update(visible=False, value=""),
            )

        def download_archive(folder_name: str):
            """Open the selected CBZ and refresh the library's downloaded marks."""
            if not folder_name:
                raise gr.Error("Select an archive first.")
            try:
                url = archive_download_url(folder_name)
            except Exception as e:
                logger.error("Error creating download link for %s: %s", folder_name, e)
                raise gr.Error(f"Could not create the download link: {e}") from e

            frame, summary, status, archives = load_library()
            archive = next(
                (a for a in archives if a.folder_name == folder_name), None
            ) if isinstance(archives, list) else None
            header = _preview_header(archive) if archive else gr.update()
            return url, frame, summary, status, archives, header

        def clear_all_ui():
            GLOBAL_STATE["folder_name"] = ""
            GLOBAL_STATE["loaded_images"] = []
            GLOBAL_STATE["selected_images"] = []
            GLOBAL_STATE["is_running"] = False
            GLOBAL_STATE["log_text"] = ""
            GLOBAL_STATE["failed_text"] = ""
            GLOBAL_STATE["download_url"] = ""
            GLOBAL_STATE["download_folder"] = ""

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
            workspace_header,
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

        library_outputs = [library_table, library_summary, library_status, library_archives]
        for library_trigger in (library_tab.select, library_refresh_btn.click):
            library_trigger(
                fn=_require_unlocked(load_library),
                inputs=[unlocked],
                outputs=library_outputs,
            )

        library_table.select(
            fn=_require_unlocked(select_archive),
            inputs=[unlocked, library_archives],
            outputs=[
                library_selected,
                preview_section,
                preview_header,
                preview_gallery,
                library_status,
            ],
        ).then(
            fn=None,
            js="""
                () => document.querySelector(".mt-preview")
                    ?.scrollIntoView({ behavior: "smooth", block: "start" })
            """,
        )

        preview_download_btn.click(
            fn=_require_unlocked(download_archive),
            inputs=[unlocked, library_selected],
            outputs=[download_launch_url, *library_outputs, preview_header],
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
