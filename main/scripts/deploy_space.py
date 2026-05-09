"""Deploy the Gradio app to Hugging Face Spaces without uploading ignored files.

The Gradio CLI uses huggingface_hub.upload_folder, which does not read
.gitignore by itself. This helper keeps the deploy rule in the project instead
of patching the installed Gradio package in .venv.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import gradio as gr
from huggingface_hub import HfApi, metadata_load, metadata_save
from huggingface_hub.utils import filter_repo_objects


PROJECT_ROOT = Path(__file__).resolve().parents[2]
README_PATH = PROJECT_ROOT / "README.md"

DEPLOY_IGNORE_PATTERNS = [
    ".venv/**",
    "venv/**",
    "__pycache__/**",
    "**/__pycache__/**",
    "*.pyc",
    ".env",
    ".env.*",
    ".idea/**",
    ".kilo/**",
    ".kiro/**",
    ".ruff_cache/**",
    ".codex",
    ".codex/**",
    "gradio_sdk.txt",
    "output/**",
    "manga_testing/**",
    "temp_repo/**",
    "main/weights/**",
    "main/fonts/**",
]


def git_visible_files() -> list[str]:
    """Return files Git would include, respecting .gitignore."""
    result = subprocess.run(
        [
            "git",
            "-C",
            str(PROJECT_ROOT),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def load_readme_metadata() -> dict:
    if not README_PATH.exists():
        return {}
    try:
        return metadata_load(str(README_PATH))
    except ValueError:
        return {}


def resolve_space_id(api: HfApi, explicit_repo_id: str | None, metadata: dict) -> str:
    if explicit_repo_id:
        return explicit_repo_id

    env_repo_id = os.getenv("HF_SPACE_ID")
    if env_repo_id:
        return env_repo_id

    title = metadata.get("title")
    if not title:
        raise SystemExit(
            "Could not infer Space id. Pass --repo-id USER_OR_ORG/SPACE_NAME or set HF_SPACE_ID."
        )

    if "/" in title:
        return title

    username = api.whoami()["name"]
    return f"{username}/{title}"


def ensure_readme_metadata(metadata: dict, app_file: str) -> None:
    metadata["title"] = metadata.get("title") or PROJECT_ROOT.name
    metadata["app_file"] = app_file
    metadata["sdk"] = "gradio"
    metadata["sdk_version"] = gr.__version__
    metadata_save(str(README_PATH), metadata)


def filtered_upload_files(allow_patterns: list[str]) -> list[str]:
    return list(
        filter_repo_objects(
            allow_patterns,
            allow_patterns=allow_patterns,
            ignore_patterns=DEPLOY_IGNORE_PATTERNS,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deploy this project to Hugging Face Spaces while respecting .gitignore."
    )
    parser.add_argument(
        "--repo-id",
        help="Space id, for example kaushikpaul/Manga-Translator-OCR_Copy. Defaults to HF_SPACE_ID or README title under your HF user.",
    )
    parser.add_argument(
        "--app-file",
        default=None,
        help="Gradio app file stored in README metadata. Defaults to existing metadata or main/app.py.",
    )
    parser.add_argument(
        "--hardware",
        default=None,
        help="Optional Hugging Face Space hardware, for example cpu-basic.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the Space as private if it does not already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the files that would be uploaded without creating or updating the Space.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api = HfApi()
    metadata = load_readme_metadata()
    app_file = args.app_file or metadata.get("app_file") or "main/app.py"
    ensure_readme_metadata(metadata, app_file)

    allow_patterns = git_visible_files()
    upload_files = filtered_upload_files(allow_patterns)
    upload_bytes = sum((PROJECT_ROOT / path).stat().st_size for path in upload_files)

    print(f"Upload set: {len(upload_files)} files, {upload_bytes / 1024:.1f} KiB")
    if args.dry_run:
        for path in upload_files:
            print(path)
        return

    space_id = resolve_space_id(api, args.repo_id, metadata)
    print(f"Deploying to https://huggingface.co/spaces/{space_id}")
    api.create_repo(
        repo_id=space_id,
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True,
        private=args.private,
        space_hardware=args.hardware,
    )
    api.upload_folder(
        repo_id=space_id,
        repo_type="space",
        folder_path=PROJECT_ROOT,
        allow_patterns=allow_patterns,
        ignore_patterns=DEPLOY_IGNORE_PATTERNS,
        commit_message="Deploy Gradio Space",
    )
    print(f"Space available at https://huggingface.co/spaces/{space_id}")


if __name__ == "__main__":
    main()
