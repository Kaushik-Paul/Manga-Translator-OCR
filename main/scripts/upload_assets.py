import os
from huggingface_hub import HfApi

# Configuration
REPO_ID = "kaushikpaul/Manga-Translator-OCR_Copy"
REPO_TYPE = "space"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR) # main/
PROJECT_ROOT = os.path.dirname(BASE_DIR)
GITIGNORE_PATH = os.path.join(PROJECT_ROOT, ".gitignore")

FOLDERS_TO_UPLOAD = ["weights", "fonts"]


def _read_gitignore() -> str | None:
    if not os.path.exists(GITIGNORE_PATH):
        return None

    with open(GITIGNORE_PATH, "r", encoding="utf-8") as f:
        return f.read()


def _gitignore_without_asset_rules(content: str) -> str:
    lines = content.splitlines(keepends=True)
    return "".join(
        line
        for line in lines
        if "main/weights" not in line and "main/fonts" not in line
    )


def upload():
    api = HfApi()

    strict_gitignore = _read_gitignore()
    if strict_gitignore is not None:
        # 1. Upload a permissive remote .gitignore so asset uploads are allowed.
        permissive_gitignore = _gitignore_without_asset_rules(strict_gitignore)
        print("🔓 Briefly updating .gitignore on Hugging Face to permit large files...")
        api.upload_file(
            path_or_fileobj=permissive_gitignore.encode("utf-8"),
            path_in_repo=".gitignore",
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            commit_message="Temporarily allow weights and fonts"
        )
    
    try:
        # 4. Sync fonts: delete remote files that no longer exist locally
        fonts_local = os.path.join(BASE_DIR, "fonts")
        fonts_remote = "main/fonts"
        if os.path.isdir(fonts_local):
            local_files = {
                f for f in os.listdir(fonts_local)
                if os.path.isfile(os.path.join(fonts_local, f)) and not f.startswith(".")
            }
            try:
                remote_entries = api.list_repo_tree(
                    repo_id=REPO_ID,
                    repo_type=REPO_TYPE,
                    path_in_repo=fonts_remote,
                )
                for entry in remote_entries:
                    # entry.rfilename is the full path like "main/fonts/OldFont.ttf"
                    basename = os.path.basename(entry.rfilename)
                    if basename not in local_files:
                        print(f"🗑️  Deleting stale remote font: {entry.rfilename}")
                        api.delete_file(
                            path_in_repo=entry.rfilename,
                            repo_id=REPO_ID,
                            repo_type=REPO_TYPE,
                            commit_message=f"Remove stale font {basename}",
                        )
            except Exception as e:
                print(f"⚠️ Could not sync remote fonts (non-fatal): {e}")

        # 5. Upload local folders
        for folder_name in FOLDERS_TO_UPLOAD:
            local_folder = os.path.join(BASE_DIR, folder_name)
            path_in_repo = f"main/{folder_name}"
            
            if not os.path.exists(local_folder):
                print(f"⚠️ Warning: Folder not found at {local_folder}. Skipping.")
                continue

            print(f"🚀 Uploading {local_folder} to {REPO_ID} (Space) at {path_in_repo}...")
            api.upload_folder(
                folder_path=local_folder,
                path_in_repo=path_in_repo,
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
            )
            print(f"✅ Upload of {folder_name} successful!")
    except Exception as e:
        print(f"❌ Upload failed: {e}")
    finally:
        # Restore the strict remote .gitignore so normal deploys stay lightweight.
        if strict_gitignore is not None:
            try:
                print("🔒 Restoring remote .gitignore to protect normal deploys...")
                api.upload_file(
                    path_or_fileobj=strict_gitignore.encode("utf-8"),
                    path_in_repo=".gitignore",
                    repo_id=REPO_ID,
                    repo_type=REPO_TYPE,
                    commit_message="Restore asset ignore rules"
                )
            except Exception as e:
                print(f"⚠️ Could not restore remote .gitignore: {e}")
            
    print(f"\n✨ View files at: https://huggingface.co/spaces/{REPO_ID}/tree/main")

if __name__ == "__main__":
    upload()
