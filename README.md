# 🎌 Manga Translator OCR 🎌

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](http://projects.kaushikpaul.co.in/manga-ocr)
[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange)](https://gradio.app/)
[![Modal](https://img.shields.io/badge/Acceleration-Modal-green)](https://modal.com/)

Manga Translator OCR is an end-to-end, state-of-the-art pipeline for seamlessly extracting, translating, and rendering Japanese manga and doujinshi pages into English.

Whether you're looking for an interactive reading experience through a clean Web UI or you need to bulk-translate entire chapters via the CLI, this tool provides an automated, high-quality translation pipeline tailored specifically for graphic novels. This project uses parallel processing to dramatically speed up page operations and supports cloud GPU offloading for resource-heavy models to supercharge the translation speed.

## 🙌 Acknowledgements & Shoutouts

This project wouldn't be possible without the incredible work from the open-source community. A massive shoutout to the models that power the core OCR and text detection capabilities:

* **[manga-ocr-base by kha-white](https://huggingface.co/kha-white/manga-ocr-base)** - For flawless Japanese vertical/horizontal OCR and furigana handling.
* **[comic-text-detector-onnx by mayocream](https://huggingface.co/mayocream/comic-text-detector-onnx)** - For highly accurate text bubble object detection model parsing.

---

## ✨ Why It's Awesome (Features)

* **Robust Speech Bubble Detection**: Powered by `comic-text-detector` to accurately find text regions, even in complex panels.
* **Cutting-Edge Japanese OCR**: Utilizes `manga-ocr` to handle vertical text, furigana, and varied manga fonts with incredibly high accuracy.
* **Smart LLM Translations**: Context-aware translations powered by your choice of LLM (via OpenRouter)! Defaulting to powerful models like Qwen (`qwen/qwen3-235b-a22b-2507`) or Mistral to preserve the tone and nuance of the original Japanese.
* **Seamless Inpainting & Rendering**: Dynamically removes original Japanese text from the image and smartly renders the translated English text to fit perfectly inside the speech bubbles.
* **Parallel Page Processing**: Processes multiple pages concurrently out of the box — overlapping I/O-bound API calls with CPU-bound detection/OCR for faster batch translations, even without a local GPU.
* **Cloud GPU Acceleration**: Need more speed? Toggle Modal.com integration to offload OCR and Detection to remote GPUs and process entire chapters in parallel!
* **Flexibility & Speed**: Use the interactive Gradio Web UI for single page experimentation, or the robust CLI for automated, folder-wide batch processing.

## 🚀 Try It Live

Don't want to install anything? **[Try the live demo on Hugging Face Spaces!](http://projects.kaushikpaul.co.in/manga-ocr)**

---

## 🛠️ Installation & Setup

We recommend using [`uv`](https://github.com/astral-sh/uv) for lightning-fast dependency management.

### 1. Clone the repository

```bash
git clone https://github.com/kaushikpaul/Manga-Translator-OCR.git
cd Manga-Translator-OCR
```

### 2. Prepare Weights and Fonts

Both the `main/weights` and `main/fonts` folders are intentionally placed in `.gitignore` to keep the repository lightweight. You must populate them locally before running the app.

* **Weights (`main/weights/`)**: Download the necessary model weights.
  1. Download the Manga OCR models from: [kha-white/manga-ocr-base](https://huggingface.co/kha-white/manga-ocr-base)
  2. Download the Comic Text Detector models from: [mayocream/comic-text-detector-onnx](https://huggingface.co/mayocream/comic-text-detector-onnx)
  3. Place the downloaded assets directly inside `main/weights/`.
* **Fonts (`main/fonts/`)**: Add your favorite custom `.ttf` or `.otf` fonts (e.g., WildWords, AnimeAce) into the `main/fonts/` folder. These will be used to render the translated English text cleanly onto the speech bubbles.
  * *Important:* After adding your fonts, you must update the font search lists at the top of `main/renderer.py` (e.g., `_DIALOGUE_FONT_SEARCH_PATHS`) to include your new font filenames so the pipeline knows which ones to use.

### 3. Install dependencies

```bash
uv sync
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory to configure the pipeline:

```env
# Required for LLM translations.
# OpenCode Go is the default provider. Set USE_OPENROUTER=true to use OpenRouter.
OPENCODE_GO_API_KEY="your-opencode-go-api-key-here"
OPENCODE_GO_MODEL="deepseek-v4-flash"
USE_OPENROUTER="false"
OPENROUTER_API_KEY="your-openrouter-api-key-here"
OPENROUTER_MODEL="deepseek/deepseek-chat"

# Optional: auto, openai, or anthropic. Auto handles MiniMax/Qwen Anthropic-style
# OpenCode Go models and OpenAI-compatible models like deepseek-v4-flash.
OPENCODE_GO_API_STYLE="auto"

# Optional: Enable Modal OCR offloading
USE_MODAL="false"

# Optional (only when USE_MODAL=true):
# false -> Modal T4 GPU OCR (faster, higher cost)
# true  -> Modal CPU OCR (slower, lower cost)
USE_MANGAOCR_CPU="false"

# Optional: Hugging Face token (helps authenticated/faster model downloads)
HF_TOKEN="hf_xxx"

# Optional: Max parallel LLM translation requests (set to 1 to fully serialize)
OPENROUTER_MAX_CONCURRENT_CALLS="3"

# Optional: Parallel pages when running locally (USE_MODAL=false).
# 2 is ideal for dual-core CPU + 16 GB RAM setups (e.g. HF Spaces).
LOCAL_MAX_PARALLEL_PAGES="2"

# Optional (only when USE_MODAL=true): max parallel pages per batch
MODAL_MAX_PARALLEL_PAGES="2"

# Optional but recommended for Gradio abuse protection:
# required to run "Translate Selected" and to open the download link
GRADIO_ACTION_PASSWORD="change-this-password"
```

---

## ☁️ Deployments

### Deploying Models to Modal (Optional GPU Acceleration)

If you want to speed up inference by offloading the heavy OCR and Detection workloads to Modal's serverless GPUs, you can deploy the included scripts:

1. Setting up Modal:

   ```bash
   uv run modal setup
   ```

2. Deploy the services:

   ```bash
   uv run modal deploy main/scripts/modal_detector_service.py
   uv run modal deploy main/scripts/modal_ocr_service.py
   ```

3. Update `.env`: Set `USE_MODAL="true"` so your local pipeline route requests to Modal.

### Deploying to Hugging Face (Gradio Spaces)

To host the app completely on Hugging Face Spaces, use the bundled deploy helper instead of `gradio deploy`. It uses the Hugging Face API directly and respects Git's ignore rules, so local folders like `.venv/`, `output/`, `manga_testing/`, `temp_repo/`, `main/weights/`, and `main/fonts/` are not uploaded accidentally.

1. **Deploy using the safe project command**:
   Ensure your Hugging Face CLI is authenticated and run:

   ```bash
   uv run python -m main.scripts.deploy_space --repo-id kaushikpaul/Manga-Translator-OCR
   ```

   You can preview the exact upload set first:

   ```bash
   uv run python -m main.scripts.deploy_space --dry-run
   ```

2. **Configure Secrets**: In your new HF Space's settings tab, specify your environment variables (like `OPENROUTER_API_KEY`, etc.).

3. **Upload Heavy Assets**: Since `weights` and `fonts` are intentionally skipped by the safe deploy command, use our bundled upload script to sync these heavy assets automatically up to HF without straining your Git history.

   Ensure your Hugging Face CLI is authenticated, edit `main/scripts/upload_assets.py` to match your Space's `REPO_ID`, and then run:

   ```bash
   uv run python main/scripts/upload_assets.py
   ```

   *Note: This script dynamically allows Large Files in the remote repository's `.gitignore` and pushes your local `main/weights/` and `main/fonts/` directly to Gradio via the Hugging Face API.*

---

## 💻 Usage Guide

### 🎨 Gradio Web Interface

The Gradio UI provides a beautiful, interactive way to upload pages, inspect detected text, and view side-by-side translations.

```bash
uv run python main/app.py
```

*The UI will be accessible locally at `http://0.0.0.0:7860`.*

### ⚡ Command Line Interface (CLI)

Built for speed and automation. Process entire chapters automatically and watch your output folder populate with translated pages.

**Translate a single page:**

```bash
uv run python -m main.cli --input page01.jpg
```

**Translate multiple specific pages:**

```bash
uv run python -m main.cli --input page01.jpg page02.jpg page03.webp
```

**Translate an entire directory:**

```bash
uv run python -m main.cli --input ./raw_manga_chapter/ --output output/translated -v
```

**Change the LLM dynamically:**

```bash
uv run python -m main.cli --input page.png --model mistralai/mistral-large-latest
```

---

## 🏗️ How It Works (The Pipeline)

1. **Detection**: `comic-text-detector` analyzes the image and creates precise masks for every speech bubble and text block.
2. **Text Extraction (OCR)**: The masked regions are passed to `manga-ocr` (either locally or optionally distributed via Modal) to extract raw Japanese text.
3. **Translation**: The extracted Japanese text is sent to an LLM via OpenRouter, where it is translated into natural-sounding English.
4. **Inpainting**: The original Japanese text is erased from the background image using advanced inpainting techniques.
5. **Rendering**: The translated English text is algorithmically fitted, word-wrapped, and rendered back onto the clean image bubbles.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

### 🎉 Enjoy Reading

---
