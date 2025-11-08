# app.py — HF Spaces entrypoint
import os, sys
sys.path.append(os.getcwd())
from huggingface_hub import hf_hub_download

REPO_ID  = os.getenv("HF_WEIGHTS_REPO", "Mohamedhiba/focusedAI-engage-v2")
FILENAME = os.getenv("HF_WEIGHTS_FILE", "best.pt")

ckpt_path = os.getenv("CKPT_PATH") or hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
os.environ["CKPT_PATH"] = ckpt_path

# your preferred demo flags
sys.argv += [
    "--ckpt", ckpt_path,
    "--crop", "none", "--preproc", "center",
    "--policy", "argmax", "--smooth", "0.0",
    "--temp", "0.60", "--bias_f", "0.50", "--bias_d", "0.10", "--bias_n", "-0.40",
]

from demo.app import main as run_demo
if __name__ == "__main__":
    run_demo()
