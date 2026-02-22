import os
import shutil
from huggingface_hub import HfApi

# Configuration
REPO_ID = "kaushikpaul/Manga-Translator-OCR" 
REPO_TYPE = "space"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR) # main/
PROJECT_ROOT = os.path.dirname(BASE_DIR)
GITIGNORE_PATH = os.path.join(PROJECT_ROOT, ".gitignore")

FOLDERS_TO_UPLOAD = ["weights", "fonts"]

def upload():
    api = HfApi()
    
    backup_path = GITIGNORE_PATH + ".bak"
    if os.path.exists(GITIGNORE_PATH):
        # 1. Backup local .gitignore
        shutil.copy(GITIGNORE_PATH, backup_path)
        
        # 2. Modify local .gitignore to remove weights/fonts ignores
        with open(GITIGNORE_PATH, "r") as f:
            lines = f.readlines()
            
        with open(GITIGNORE_PATH, "w") as f:
            for line in lines:
                if "main/weights" in line or "main/fonts" in line:
                    continue
                f.write(line)
        
        # 3. Upload the permissive .gitignore to HF Space first
        print("🔓 Briefly updating .gitignore on Hugging Face to permit large files...")
        api.upload_file(
            path_or_fileobj=GITIGNORE_PATH,
            path_in_repo=".gitignore",
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            commit_message="Temporarily allow weights and fonts"
        )
    
    try:
        # 4. Use the simple `upload_folder` to upload everything
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
        # 5. Restore the strict .gitignore locally so git stays clean
        if os.path.exists(backup_path):
            shutil.move(backup_path, GITIGNORE_PATH)
            print("🔒 Restored local .gitignore to protect your Git repository.")
            
    print(f"\n✨ View files at: https://huggingface.co/spaces/{REPO_ID}/tree/main")

if __name__ == "__main__":
    upload()
