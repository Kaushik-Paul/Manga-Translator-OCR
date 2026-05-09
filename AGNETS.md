# Product Overview

Manga Translator OCR is an end-to-end pipeline that translates Japanese manga/doujinshi pages into English. It detects speech bubbles, extracts Japanese text via OCR, translates it using an LLM, inpaints the original text, and renders the English translation back onto the page.

## Core Pipeline (in order)
1. **Detection** — `comic-text-detector` ONNX model finds text regions and produces a pixel-level mask
2. **OCR** — `manga-ocr` extracts Japanese text from each region (supports vertical text, furigana, varied fonts)
3. **Translation** — Batch call to OpenRouter LLM (default: `qwen/qwen3-235b-a22b-2507`); includes a repair pass for broken outputs
4. **Inpainting** — OpenCV Telea algorithm erases original Japanese text using the ML mask
5. **Rendering** — PIL renders translated English text into the cleaned bubble using comic fonts

## Entry Points
- **Gradio Web UI** — `main/app.py` — interactive single-page translation with GCP Cloud Storage integration
- **CLI** — `main/cli.py` — batch translation of files or directories

## Deployment Targets
- Local (CPU-only, no Modal)
- Hugging Face Spaces (Gradio deploy)
- Modal.com for GPU-accelerated OCR and detection offloading

## Key Constraints
- Source language is Japanese only (`source_lang = "ja"`)
- Translation output must be concise (speech bubble space limits)
- Translated text must use only basic ASCII + standard punctuation (font rendering constraints)
- All translated dialogue is rendered in ALL CAPS (standard manga lettering convention)

# Project Structure

```
manga-translator-ocr/
├── main/                        # All application source code
│   ├── app.py                   # Gradio UI launcher (entry point)
│   ├── cli.py                   # CLI entry point (python -m main.cli)
│   ├── config.py                # Settings dataclass, .env loading, singleton `settings`
│   ├── pipeline.py              # Core orchestrator: detection → OCR → translate → inpaint → render
│   ├── detector.py              # ComicTextDetector (ONNX), ModalTextDetector, TextRegion dataclass
│   ├── ocr.py                   # MangaOCREngine, ModalOCREngine, ModalCPUOCREngine
│   ├── translator.py            # OpenRouter API calls, batch translation, repair pass
│   ├── renderer.py              # Inpainting (cv2), text rendering (PIL), font/bubble logic
│   ├── gcp_storage.py           # GCS download/upload, presigned URLs, ZIP handling
│   ├── fonts/                   # Bundled comic fonts (animeace2.otf, CCWildWordsRoman.ttf, CJK fonts)
│   ├── weights/                 # Model weights (gitignored, must be populated manually)
│   │   ├── comic-text-detector/ # comic-text-detector.onnx
│   │   └── manga-ocr-base/      # HuggingFace model files
│   ├── scripts/                 # Deployment and utility scripts
│   │   ├── modal_ocr_service.py     # Modal GPU/CPU OCR service definition
│   │   ├── modal_detector_service.py # Modal detector service definition
│   │   └── upload_assets.py         # Upload weights/fonts to HF Space
│   └── ui/                      # Gradio UI components
│       ├── gradio_app.py        # Full Gradio Blocks app, event handlers, global state
│       ├── pipeline_runner.py   # Generator-based pipeline wrapper for UI progress reporting
│       └── styles.py            # Custom CSS
├── manga_testing/               # Test images (gitignored)
├── pyproject.toml               # Project metadata and dependencies
├── .env                         # Local secrets (gitignored in production)
└── .python-version              # Python 3.12
```

## Architecture Patterns

### Singleton Models
`ComicTextDetector`, `MangaOCREngine`, `ModalOCREngine`, and `ModalCPUOCREngine` all use `__new__`-based singletons. Models are lazy-loaded on first use via `_ensure_model()` / `_ensure_modal()`. Never instantiate these directly in loops.

### Factory Functions
Use factory functions to get the right backend based on config:
- `get_text_detector()` → returns `ModalTextDetector` or `ComicTextDetector`
- `get_ocr_engine(lang)` → returns `ModalOCREngine`, `ModalCPUOCREngine`, or `MangaOCREngine`

### Settings Singleton
`from main.config import settings` — always import the singleton, never instantiate `Settings()` directly except in `pipeline_runner.py` where a config override is needed.

### Pipeline Data Flow
`pipeline.py` is the single orchestrator. It uses `RenderTextUnit` dataclass to carry per-region state (source text, translated text, style hint, text color, context region) through the pipeline stages.

### Gradio Global State
`GLOBAL_STATE` dict in `gradio_app.py` is the shared mutable state across Gradio event handlers. The translation pipeline runs in a background `threading.Thread`; the UI polls via a `gr.Timer`. Do not block Gradio event handlers with long-running work.

### Parallelism
- Multiple pages: `ThreadPoolExecutor` with batch processing (configurable via `MODAL_MAX_PARALLEL_PAGES` / `LOCAL_MAX_PARALLEL_PAGES`)
- Multiple OCR sub-regions per page: parallel only when `USE_MODAL=true` (3 workers max)
- OpenRouter API: bounded by `BoundedSemaphore` (`OPENROUTER_MAX_CONCURRENT_CALLS`)

## Code Style Conventions
- `from __future__ import annotations` at the top of every module
- Type hints throughout; use `NDArray` from `numpy.typing` for numpy arrays
- Dataclasses (`@dataclass`) for structured data (e.g. `TextRegion`, `RenderTextUnit`, `Settings`)
- Logging via `logging.getLogger(__name__)` — never use `print()`
- Image arrays are BGR (OpenCV convention) internally; convert to RGB only for PIL operations
- All font paths resolved at module load time via `_find_font()` — never hardcode absolute paths

# Tech Stack

## Language & Runtime
- Python 3.12 (see `.python-version`)
- Package manager: `uv` (preferred over pip)

## Core Dependencies
| Library | Purpose |
|---|---|
| `manga-ocr` | Japanese OCR (HuggingFace model `kha-white/manga-ocr-base`) |
| `onnxruntime` | Local inference for `comic-text-detector` ONNX model |
| `opencv-python-headless` | Image processing, inpainting, morphological ops |
| `Pillow` | Text rendering onto images |
| `httpx` | OpenRouter API calls (sync, with timeout config) |
| `python-dotenv` | `.env` loading via `main/config.py` |
| `gradio>=5.0` | Web UI |
| `modal>=0.73` | Cloud GPU offloading for OCR and detection |
| `google-cloud-storage` | GCP bucket integration (raw/translated manga storage) |
| `huggingface-hub` | Model weight downloads (fallback if local weights missing) |

## Configuration
All settings are loaded from `.env` via `main/config.py` into a `Settings` dataclass singleton (`settings`). Key env vars:

```
OPENROUTER_API_KEY        # Required
TRANSLATION_MODEL         # Default: qwen/qwen3-235b-a22b-2507
USE_MODAL                 # true/false — enables Modal GPU offloading
USE_MANGAOCR_CPU          # true/false — Modal CPU vs GPU OCR
USE_DETECTION_MODEL       # true/false — Modal vs local ONNX detector
MODAL_MAX_PARALLEL_PAGES  # Parallelism when USE_MODAL=true
LOCAL_MAX_PARALLEL_PAGES  # Parallelism when USE_MODAL=false
OPENROUTER_MAX_CONCURRENT_CALLS
GCP_SERVICE_ACCOUNT_BASE64
GCP_BUCKET_NAME
GRADIO_ACTION_PASSWORD
HF_TOKEN
```

## Common Commands

```bash
# Install dependencies
uv sync

# Run Gradio web UI (localhost:7860)
uv run python main/app.py

# Translate a single image via CLI
uv run python -m main.cli --input page01.jpg

# Translate a directory
uv run python -m main.cli --input ./manga_chapter/ --output ./output/ -v

# Deploy Modal services (GPU OCR + detector)
uv run modal setup
uv run modal deploy main/scripts/modal_ocr_service.py
uv run modal deploy main/scripts/modal_detector_service.py

# Deploy to Hugging Face Spaces
uv run gradio deploy

# Upload weights/fonts to HF Space (after deploy)
uv run python main/scripts/upload_assets.py
```

## Linting
- Ruff is configured (`.ruff_cache/` present). Run with: `uv run ruff check .`

## Image Formats
Supported: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tiff`
Output quality: WebP at quality=82, JPEG at quality=85 (via PIL, not cv2, to avoid bloat)
