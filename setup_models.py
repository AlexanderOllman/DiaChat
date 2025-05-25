"""Download required checkpoints into /models if missing.
Run this during container build (or manually) to prime the volume.
"""
import os
from huggingface_hub import hf_hub_download, whoami, login
from huggingface_hub.utils import LocalTokenNotFoundError
from dia.model import Dia

def check_hf_authentication():
    """Check if user is logged into Hugging Face and prompt login if needed."""
    try:
        user_info = whoami()
        print(f"[setup] Logged in as: {user_info['name']} ({user_info.get('email', 'no email')})")
        return True
    except LocalTokenNotFoundError:
        print("[setup] No Hugging Face authentication found.")
        print("[setup] Some models may require authentication.")
        response = input("[setup] Would you like to log in now? (y/n): ").lower().strip()
        
        if response in ['y', 'yes']:
            try:
                login()
                user_info = whoami()
                print(f"[setup] Successfully logged in as: {user_info['name']}")
                return True
            except Exception as e:
                print(f"[setup] Login failed: {e}")
                print("[setup] Continuing without authentication - some models may not be accessible.")
                return False
        else:
            print("[setup] Continuing without authentication - some models may not be accessible.")
            return False

# Use MODELS_DIR from environment if set (e.g., in Docker), otherwise default to ./models
# Ensures models are downloaded to the location specified by the ENV, or a local default.
_MODELS_DIR_ENV = os.getenv("MODELS_DIR")
if _MODELS_DIR_ENV:
    # If MODELS_DIR is set (e.g. "/models" in Docker), use it directly.
    # Ensure it's treated as an absolute path.
    MODELS_DIR = os.path.abspath(_MODELS_DIR_ENV)
else:
    # Default to "./models" relative to this script if ENV is not set.
    # This keeps behavior consistent for local runs outside Docker.
    MODELS_DIR = os.path.abspath("./models")

print(f"[setup] Using MODELS_DIR: {MODELS_DIR}")
os.makedirs(MODELS_DIR, exist_ok=True)

# Set Hugging Face cache directory to this MODELS_DIR
# This ensures Dia and hf_hub_download use the same resolved path.
os.environ["HF_HUB_CACHE"] = MODELS_DIR # MODELS_DIR is already absolute

# Check authentication before downloading models
check_hf_authentication()

# Download Whisper GGML model from huggingface for pywhispercpp
WHISPER_FILE = os.path.join(MODELS_DIR, "ggml-base.en.bin")
if not os.path.exists(WHISPER_FILE):
    print("[setup] Fetching Whisper base.en model…")
    hf_hub_download(
        repo_id="ggerganov/whisper.cpp",
        filename="ggml-base.en.bin",
        local_dir=MODELS_DIR,
        local_dir_use_symlinks=False
    )
else:
    print("[setup] Whisper model already present.")

print("[setup] Ensuring Dia weights cached…")
Dia.from_pretrained("nari-labs/Dia-1.6B", device="cpu")
print("[setup] Done.")