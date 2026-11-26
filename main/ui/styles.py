"""
Theme and custom CSS for the Manga Translator Gradio UI.

The look is "sumi ink and vermilion": deep ink neutrals with a hanko-seal
red-orange accent and an indigo secondary. Colours that Gradio owns (blocks,
inputs, buttons) are set on the theme so light and dark mode both work. The
CSS below only adds layout and the pieces Gradio has no theme variable for.
"""

from __future__ import annotations

import gradio as gr

# Hanko-seal vermilion, used as the primary hue.
VERMILION = gr.themes.Color(
    name="vermilion",
    c50="#fff4ed",
    c100="#ffe6d5",
    c200="#fecaaa",
    c300="#fda574",
    c400="#fb743c",
    c500="#f94f16",
    c600="#ea350c",
    c700="#c2250c",
    c800="#9a2012",
    c900="#7c1d12",
    c950="#430b07",
)


def build_theme() -> gr.themes.Base:
    """Return the app's Gradio theme."""
    theme = gr.themes.Base(
        primary_hue=VERMILION,
        secondary_hue=gr.themes.colors.indigo,
        neutral_hue=gr.themes.colors.slate,
        radius_size=gr.themes.sizes.radius_lg,
        spacing_size=gr.themes.sizes.spacing_md,
        text_size=gr.themes.sizes.text_md,
        font=[
            gr.themes.GoogleFont("Plus Jakarta Sans", weights=(400, 500, 600, 700, 800)),
            "ui-sans-serif",
            "system-ui",
            "sans-serif",
        ],
        font_mono=[
            gr.themes.GoogleFont("JetBrains Mono", weights=(400, 600)),
            "ui-monospace",
            "Consolas",
            "monospace",
        ],
    )
    return theme.set(
        # Page and surfaces
        body_background_fill="#f5f6fa",
        body_background_fill_dark="#0a0c12",
        body_text_color="#151826",
        body_text_color_dark="#e7e9f2",
        body_text_color_subdued="#5d6378",
        body_text_color_subdued_dark="#9097ad",
        background_fill_primary="#ffffff",
        background_fill_primary_dark="#12151f",
        background_fill_secondary="#f1f3f8",
        background_fill_secondary_dark="#171b27",
        border_color_primary="#e2e5ee",
        border_color_primary_dark="#252a3a",
        color_accent_soft="*primary_50",
        color_accent_soft_dark="rgba(249, 79, 22, 0.14)",
        # Blocks
        block_background_fill="*background_fill_primary",
        block_background_fill_dark="*background_fill_primary",
        block_border_width="1px",
        block_border_color="*border_color_primary",
        block_border_color_dark="*border_color_primary",
        block_shadow="none",
        block_shadow_dark="none",
        block_radius="14px",
        block_label_background_fill="transparent",
        block_label_background_fill_dark="transparent",
        block_label_border_width="0px",
        block_label_text_color="*body_text_color_subdued",
        block_label_text_color_dark="*body_text_color_subdued",
        block_label_text_weight="600",
        block_label_padding="0 0 6px 0",
        block_title_background_fill="transparent",
        block_title_background_fill_dark="transparent",
        block_title_text_color="*body_text_color_subdued",
        block_title_text_color_dark="*body_text_color_subdued",
        block_title_text_weight="600",
        block_title_padding="0 0 6px 0",
        panel_background_fill="*background_fill_primary",
        panel_background_fill_dark="*background_fill_primary",
        # Inputs
        input_background_fill="#f8f9fc",
        input_background_fill_dark="#0e111a",
        input_background_fill_focus="#ffffff",
        input_background_fill_focus_dark="#0e111a",
        input_border_color="#dfe3ec",
        input_border_color_dark="#2a3042",
        input_border_color_focus="*primary_400",
        input_border_color_focus_dark="*primary_500",
        input_border_width="1px",
        input_shadow="none",
        input_shadow_dark="none",
        input_shadow_focus="0 0 0 4px rgba(249, 79, 22, 0.14)",
        input_shadow_focus_dark="0 0 0 4px rgba(249, 79, 22, 0.22)",
        input_radius="10px",
        input_padding="12px 14px",
        # Buttons
        button_large_radius="12px",
        button_medium_radius="10px",
        button_small_radius="8px",
        button_large_text_weight="700",
        button_medium_text_weight="600",
        button_transition="all 0.18s ease",
        button_border_width="1px",
        button_primary_background_fill="linear-gradient(135deg, *primary_500, #ff8a3d)",
        button_primary_background_fill_dark="linear-gradient(135deg, *primary_500, #ff8a3d)",
        button_primary_background_fill_hover="linear-gradient(135deg, *primary_600, *primary_500)",
        button_primary_background_fill_hover_dark="linear-gradient(135deg, *primary_600, *primary_500)",
        button_primary_border_color="transparent",
        button_primary_border_color_dark="transparent",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        button_primary_shadow="0 6px 18px -6px rgba(249, 79, 22, 0.55)",
        button_primary_shadow_dark="0 6px 18px -6px rgba(249, 79, 22, 0.55)",
        button_primary_shadow_hover="0 10px 24px -8px rgba(249, 79, 22, 0.65)",
        button_primary_shadow_hover_dark="0 10px 24px -8px rgba(249, 79, 22, 0.65)",
        button_secondary_background_fill="*background_fill_primary",
        button_secondary_background_fill_dark="#181c29",
        button_secondary_background_fill_hover="*background_fill_secondary",
        button_secondary_background_fill_hover_dark="#20263a",
        button_secondary_border_color="*border_color_primary",
        button_secondary_border_color_dark="*border_color_primary",
        button_secondary_text_color="*body_text_color",
        button_secondary_text_color_dark="*body_text_color",
        button_secondary_shadow="none",
        button_secondary_shadow_dark="none",
        button_cancel_background_fill="#fff1f0",
        button_cancel_background_fill_dark="rgba(239, 68, 68, 0.12)",
        button_cancel_background_fill_hover="#ffe2df",
        button_cancel_background_fill_hover_dark="rgba(239, 68, 68, 0.2)",
        button_cancel_border_color="#fecaca",
        button_cancel_border_color_dark="rgba(239, 68, 68, 0.35)",
        button_cancel_text_color="#c81e1e",
        button_cancel_text_color_dark="#fca5a5",
        # Checkboxes (image chips)
        checkbox_label_background_fill="*background_fill_primary",
        checkbox_label_background_fill_dark="#151926",
        checkbox_label_background_fill_hover="*background_fill_secondary",
        checkbox_label_background_fill_hover_dark="#1c2131",
        checkbox_label_background_fill_selected="*primary_50",
        checkbox_label_background_fill_selected_dark="rgba(249, 79, 22, 0.14)",
        checkbox_label_border_color="*border_color_primary",
        checkbox_label_border_color_dark="*border_color_primary",
        checkbox_label_border_color_selected="*primary_300",
        checkbox_label_border_color_selected_dark="rgba(249, 79, 22, 0.55)",
        checkbox_label_border_width="1px",
        checkbox_label_text_color_selected="*primary_700",
        checkbox_label_text_color_selected_dark="*primary_200",
        checkbox_background_color_selected="*primary_500",
        checkbox_background_color_selected_dark="*primary_500",
        checkbox_border_color_selected="*primary_500",
        checkbox_border_color_selected_dark="*primary_500",
        checkbox_border_radius="6px",
        loader_color="*primary_500",
        loader_color_dark="*primary_400",
    )


CUSTOM_CSS = """
/* ── Tokens ─────────────────────────────────────────── */
:root {
    --mt-accent: #f94f16;
    --mt-accent-2: #ff8a3d;
    --mt-indigo: #6366f1;
    --mt-success: #059669;
    --mt-danger: #dc2626;
    --mt-hero-ink: rgba(21, 24, 38, 0.07);
    --mt-card-shadow: 0 1px 2px rgba(16, 24, 40, 0.04), 0 8px 24px -12px rgba(16, 24, 40, 0.12);
    --mt-glow: radial-gradient(60% 120% at 0% 0%, rgba(249, 79, 22, 0.12), transparent 60%),
               radial-gradient(50% 100% at 100% 0%, rgba(99, 102, 241, 0.12), transparent 60%);
}

.dark {
    --mt-success: #34d399;
    --mt-danger: #f87171;
    --mt-hero-ink: rgba(255, 255, 255, 0.06);
    --mt-card-shadow: 0 1px 2px rgba(0, 0, 0, 0.3), 0 12px 32px -16px rgba(0, 0, 0, 0.6);
    --mt-glow: radial-gradient(60% 120% at 0% 0%, rgba(249, 79, 22, 0.18), transparent 60%),
               radial-gradient(50% 100% at 100% 0%, rgba(99, 102, 241, 0.18), transparent 60%);
}

/* ── Global ─────────────────────────────────────────── */
.gradio-container {
    max-width: 1120px !important;
    margin: 0 auto !important;
    padding-left: 20px !important;
    padding-right: 20px !important;
}

/* Let the input focus ring show instead of being clipped by its block. */
.gradio-container .form,
.gradio-container .form > .block {
    overflow: visible !important;
}

/* gr.HTML pads its content; blocks marked mt-bare draw their own box. */
.mt-bare .html-container {
    padding: 0 !important;
}

/* ── Hero header ────────────────────────────────────── */
.mt-hero {
    position: relative;
    overflow: hidden;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 1rem 1.5rem;
    margin-top: 0.75rem;
    padding: 1.35rem 1.5rem;
    border: 1px solid var(--border-color-primary);
    border-radius: 20px;
    background-color: var(--background-fill-primary);
    background-image: var(--mt-glow);
    box-shadow: var(--mt-card-shadow);
}

/* Halftone dots, a nod to screentone. */
.mt-hero::after {
    content: "";
    position: absolute;
    inset: 0 0 0 auto;
    width: 45%;
    background-image: radial-gradient(var(--mt-hero-ink) 1.2px, transparent 1.4px);
    background-size: 10px 10px;
    -webkit-mask-image: linear-gradient(to left, #000, transparent);
    mask-image: linear-gradient(to left, #000, transparent);
    pointer-events: none;
}

.mt-brand {
    position: relative;
    z-index: 1;
    display: flex;
    align-items: center;
    gap: 0.9rem;
    min-width: 0;
}

.mt-logo {
    flex: none;
    display: grid;
    place-items: center;
    width: 52px;
    height: 52px;
    border-radius: 14px;
    background: linear-gradient(140deg, var(--mt-accent), var(--mt-accent-2));
    color: #fff;
    font-size: 1.6rem;
    font-weight: 800;
    line-height: 1;
    box-shadow: 0 10px 24px -10px rgba(249, 79, 22, 0.8), inset 0 -2px 0 rgba(0, 0, 0, 0.12);
    transform: rotate(-4deg);
}

.mt-brand h1 {
    margin: 0 !important;
    font-size: clamp(1.35rem, 1.1rem + 1vw, 1.75rem) !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em;
    line-height: 1.15 !important;
    color: var(--body-text-color) !important;
}

.mt-brand h1 span {
    background-image: linear-gradient(135deg, var(--mt-accent), var(--mt-accent-2)) !important;
    -webkit-background-clip: text !important;
    background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
}

.mt-brand p {
    margin: 0.2rem 0 0 0 !important;
    font-size: 0.92rem;
    color: var(--body-text-color-subdued);
}

.mt-pipeline {
    position: relative;
    z-index: 1;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.35rem;
    margin: 0 !important;
    padding: 0 !important;
    list-style: none !important;
}

.mt-pipeline li {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    margin: 0 !important;
    padding: 0.3rem 0.65rem;
    border: 1px solid var(--border-color-primary);
    border-radius: 999px;
    background: var(--background-fill-secondary);
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--body-text-color-subdued);
    white-space: nowrap;
}

.mt-pipeline li::before {
    content: "";
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--mt-accent);
    opacity: 0.85;
}

/* ── Sections ───────────────────────────────────────── */
.section-card {
    border: 1px solid var(--border-color-primary) !important;
    border-radius: 16px !important;
    background: var(--background-fill-primary) !important;
    box-shadow: var(--mt-card-shadow) !important;
    padding: 1.25rem !important;
    margin-bottom: 0.75rem !important;
}

.section-title {
    font-size: 1.05rem;
    font-weight: 700;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.top-actions-row {
    justify-content: flex-end !important;
    margin-bottom: 0.75rem !important;
}

/* ── Buttons ────────────────────────────────────────── */
button.lg, button.md, button.sm {
    letter-spacing: -0.005em;
}

button.primary:hover:not(:disabled),
button.secondary:hover:not(:disabled),
button.stop:hover:not(:disabled) {
    transform: translateY(-1px);
}

button:disabled {
    opacity: 0.5 !important;
    transform: none !important;
    box-shadow: none !important;
    cursor: not-allowed !important;
}

.translate-btn {
    background: linear-gradient(135deg, var(--mt-indigo), #8b5cf6) !important;
    box-shadow: 0 8px 22px -8px rgba(99, 102, 241, 0.6) !important;
    font-size: 1.02rem !important;
}

.download-btn {
    background: linear-gradient(135deg, #0d9488, #059669) !important;
    box-shadow: 0 8px 22px -8px rgba(5, 150, 105, 0.6) !important;
}

.lock-btn {
    white-space: nowrap !important;
}

/* ── Progress ───────────────────────────────────────── */
.progress-box textarea {
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
    line-height: 1.65 !important;
}

/* ── Results ────────────────────────────────────────── */
.result-success {
    color: var(--mt-success);
    font-weight: 600;
}

.result-error {
    color: var(--mt-danger);
    font-weight: 600;
}

/* ── Upload / delete modals ────────────────────────── */
.auth-modal-overlay {
    position: fixed !important;
    inset: 0 !important;
    z-index: 9999 !important;
    background: rgba(10, 12, 18, 0.55) !important;
    -webkit-backdrop-filter: blur(6px);
    backdrop-filter: blur(6px);
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 1rem !important;
}

.auth-modal-card {
    width: min(480px, 100%) !important;
    border: 1px solid var(--border-color-primary) !important;
    border-radius: 18px !important;
    background: var(--background-fill-primary) !important;
    padding: 1.25rem !important;
    box-shadow: 0 24px 64px -16px rgba(0, 0, 0, 0.45) !important;
}

.auth-modal-title {
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 0.4rem;
}

.auth-modal-message {
    margin-bottom: 0.7rem;
    color: var(--body-text-color-subdued);
}

/* ── Lock screen ───────────────────────────────────── */
.lock-checking {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.6rem;
    padding: 3rem 0;
    color: var(--body-text-color-subdued);
    font-weight: 500;
}

.mt-spinner {
    width: 18px;
    height: 18px;
    border-radius: 50%;
    border: 2px solid var(--border-color-primary);
    border-top-color: var(--mt-accent);
    animation: mt-spin 0.8s linear infinite;
}

@keyframes mt-spin {
    to { transform: rotate(360deg); }
}

.lock-screen {
    width: min(440px, 100%) !important;
    max-width: 440px !important;
    margin: clamp(1.5rem, 6vh, 4rem) auto 0 auto !important;
}

.lock-card {
    gap: 1rem !important;
    padding: 1.75rem !important;
    border: 1px solid var(--border-color-primary) !important;
    border-radius: 22px !important;
    background-color: var(--background-fill-primary) !important;
    background-image: var(--mt-glow) !important;
    box-shadow: var(--mt-card-shadow) !important;
}

.lock-head {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    gap: 0.35rem;
}

.lock-badge {
    display: grid;
    place-items: center;
    width: 56px;
    height: 56px;
    margin-bottom: 0.4rem;
    border-radius: 18px;
    background: var(--color-accent-soft);
    border: 1px solid var(--border-color-primary);
    font-size: 1.6rem;
}

.lock-head h2 {
    margin: 0 !important;
    font-size: 1.35rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em;
}

.lock-head p {
    margin: 0 !important;
    font-size: 0.92rem;
    color: var(--body-text-color-subdued);
}

.lock-card .form {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}

.lock-card .block {
    padding: 0 !important;
    border: none !important;
    background: transparent !important;
}

/* ── Checkbox group (image selector) ────────────────── */
.image-selector label {
    transition: background 0.15s ease, border-color 0.15s ease;
}
"""
