# 🎌 Manga Translator OCR 🎌

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/kaushikpaul/Manga-Translator-OCR)
[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange)](https://gradio.app/)
[![Modal](https://img.shields.io/badge/Acceleration-Modal-green)](https://modal.com/)

Manga Translator OCR is an end-to-end, state-of-the-art pipeline for seamlessly extracting, translating, and rendering Japanese manga and doujinshi pages into English.

Whether you're looking for an interactive reading experience through a clean Web UI or you need to bulk-translate entire chapters via the CLI, this tool provides an automated, high-quality translation pipeline tailored specifically for graphic novels.

---

## ✨ Why It's Awesome (Features)

* **Robust Speech Bubble Detection**: Powered by `comic-text-detector` to accurately find text regions, even in complex panels.
* **Cutting-Edge Japanese OCR**: Utilizes `manga-ocr` to handle vertical text, furigana, and varied manga fonts with incredibly high accuracy.
* **Smart LLM Translations**: Context-aware translations powered by your choice of LLM (via OpenRouter)! Defaulting to powerful models like DeepSeek or Mistral to preserve the tone and nuance of the original Japanese.
* **Seamless Inpainting & Rendering**: Dynamically removes original Japanese text from the image and smartly renders the translated English text to fit perfectly inside the speech bubbles.
* **Cloud GPU Acceleration**: Need speed? Toggle Modal.com integration to offload OCR to remote GPUs and process entire chapters in parallel!
* **Flexibility & Speed**: Use the interactive Gradio Web UI for single page experimentation, or the robust CLI for automated, folder-wide batch processing.

## 🚀 Try It Live

Don't want to install anything? **[Try the live demo on Hugging Face Spaces!](https://huggingface.co/spaces/kaushikpaul/Manga-Translator-OCR)**

---

## 🛠️ Installation & Setup

We recommend using [`uv`](https://github.com/astral-sh/uv) for lightning-fast dependency management.

1. **Clone the repository:**

   ```bash
   git clone https://github.com/kaushikpaul/Manga-Translator-OCR.git
   cd Manga-Translator-OCR
   ```

2. **Install dependencies:**

   ```bash
   uv sync
   ```

3. **Configure Environment Variables:**
   Create a `.env` file in the root directory to configure the pipeline:

   ```env
   # Required for LLM translations
   OPENROUTER_API_KEY="your-openrouter-api-key-here"

   # Optional: Enable Modal for GPU-accelerated OCR
   USE_MODAL="false" 
   
   # Optional: Override the translation model
   TRANSLATION_MODEL="deepseek/deepseek-chat"
   ```

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
uv run python -m main.cli --input ./raw_manga_chapter/ --output ./translated_chapter/
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
