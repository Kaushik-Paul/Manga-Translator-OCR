"""
Custom CSS styling for the Manga Translator Gradio UI.
"""

CUSTOM_CSS = """
/* ── Global ─────────────────────────────────────────── */
.gradio-container {
    max-width: 900px !important;
    margin: 0 auto !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}

/* ── Header ─────────────────────────────────────────── */
.app-header {
    text-align: center;
    padding: 1.5rem 0 0.5rem 0;
}

.app-header h1 {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #ff6b6b, #ee5a24, #f9ca24);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}

.app-header p {
    opacity: 0.7;
    font-size: 0.95rem;
    margin-top: 0.25rem;
}

/* ── Sections ───────────────────────────────────────── */
.section-card {
    border: 1px solid var(--border-color-primary) !important;
    border-radius: 12px !important;
    padding: 1.25rem !important;
    margin-bottom: 0.75rem !important;
}

.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Buttons ────────────────────────────────────────── */
.primary-btn {
    background: linear-gradient(135deg, #ee5a24, #f0932b) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}

.primary-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(238, 90, 36, 0.35) !important;
}

.primary-btn:disabled {
    opacity: 0.5 !important;
    transform: none !important;
    box-shadow: none !important;
    cursor: not-allowed !important;
}

.translate-btn {
    background: linear-gradient(135deg, #6c5ce7, #a29bfe) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    border-radius: 10px !important;
    padding: 0.75rem 2rem !important;
    transition: all 0.2s ease !important;
}

.translate-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 16px rgba(108, 92, 231, 0.4) !important;
}

.translate-btn:disabled {
    opacity: 0.5 !important;
    transform: none !important;
    box-shadow: none !important;
    cursor: not-allowed !important;
}

/* ── Progress ───────────────────────────────────────── */
.progress-box textarea {
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    font-size: 0.85rem !important;
    line-height: 1.6 !important;
}

/* ── Results ────────────────────────────────────────── */
.result-success {
    color: #00b894;
    font-weight: 600;
}

.result-error {
    color: #e17055;
    font-weight: 600;
}

.download-link a {
    color: #6c5ce7 !important;
    font-weight: 600;
    text-decoration: underline;
}

/* ── Checkbox group (image selector) ────────────────── */
.image-selector label {
    transition: background 0.15s ease;
}

.image-selector label:hover {
    background: var(--background-fill-secondary);
    border-radius: 6px;
}
"""
