import os
from huggingface_hub import HfApi

# Configuration
# Replace this with your actual Hugging Face Space Repo ID if different
REPO_ID = "kaushikpaul/Manga-Translator-OCR" 
REPO_TYPE = "space"

# Path to this script's directory (main/scripts)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Base directory (main/)
BASE_DIR = os.path.dirname(SCRIPT_DIR)

# Folders to upload (relative to main/)
FOLDERS_TO_UPLOAD = ["weights", "fonts"]

def upload():
    api = HfApi()
    
    for folder_name in FOLDERS_TO_UPLOAD:
        local_folder = os.path.join(BASE_DIR, folder_name)
        path_in_repo = f"main/{folder_name}"
        
        if not os.path.exists(local_folder):
            print(f"⚠️ Warning: Folder not found at {local_folder}. Skipping.")
            continue

        print(f"Uploading {local_folder} to {REPO_ID} (Space) at {path_in_repo}...")
        
        try:
            api.upload_folder(
                folder_path=local_folder,
                path_in_repo=path_in_repo,
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                ignore_patterns=[], # This is crucial to bypass .gitignore
            )
            print(f"✅ Upload of {folder_name} successful!")
        except Exception as e:
            print(f"❌ Upload failed for {folder_name}: {e}")
            
    print(f"\nView files at: https://huggingface.co/spaces/{REPO_ID}/tree/main")

if __name__ == "__main__":
    upload()
