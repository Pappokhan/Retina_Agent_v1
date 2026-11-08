"""Central settings. Everything tunable can be overridden with environment variables."""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

try:  # optional: read secrets such as ANTHROPIC_API_KEY from .env
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


# --- Paths -------------------------------------------------------------------
WEIGHTS_DIR = Path(os.getenv("WEIGHTS_DIR", BASE_DIR / "weights"))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", BASE_DIR / "uploads"))
KNOWLEDGE_DIR = Path(os.getenv("KNOWLEDGE_DIR", BASE_DIR / "knowledge"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", BASE_DIR / "faiss_index"))

# --- Model / preprocessing: identical in all four training notebooks ---------
IMG_SIZE = 512
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
THRESHOLD = 0.5                      # notebooks binarise at 0.5
UNCERTAIN_BAND = (0.3, 0.7)          # pixels in this probability band count as "uncertain"
MODEL_CACHE_SIZE = _int("MODEL_CACHE_SIZE", 4)   # ~55 MB per model; all 4 variants stay loaded

# --- Checkpoints: the Dataset A model only, in 4 variants ---------------------
# The app serves only the notebook-A model (trained on 800 images). "folder" is the data folder
# name used inside that notebook; it does not match the letter.
DATASETS = {
    "A": {"label": "Dataset A", "images": 800, "split": "640 / 80 / 80", "folder": "Dataset D"},
}
DEFAULT_DATASET = "A"

VARIANTS = {
    "best": {
        "label": "Best validation",
        "file": "best.pth",
        "note": "best_model_AttentionUNet.pth: epoch with the highest validation Dice",
    },
    "standard": {
        "label": "Final epoch",
        "file": "standard.pth",
        "note": "standard_attention_unet.pth: weights at the end of training",
    },
    "fedavg": {
        "label": "FedAvg",
        "file": "fedavg.pth",
        "note": "fedavg_attention_unet.pth from the notebook's federated run",
    },
    "fedprox": {
        "label": "FedProx",
        "file": "fedprox.pth",
        "note": "fedprox_attention_unet.pth from the notebook's federated run",
    },
}

# --- Uploads -----------------------------------------------------------------
MAX_UPLOAD_MB = _int("MAX_UPLOAD_MB", 20)
MAX_IMAGE_PIXELS = _int("MAX_IMAGE_PIXELS", 40_000_000)
UPLOAD_TTL_HOURS = _float("UPLOAD_TTL_HOURS", 6.0)   # uploads are deleted after this
ALLOWED_FORMATS = {"PNG", "JPEG", "TIFF", "BMP", "GIF", "WEBP", "PPM"}
OVERLAY_RGB = (0, 229, 208)          # cyan: stays visible on red/orange retinal images

# --- VLM / agent / RAG -------------------------------------------------------
# Models are written "provider:model" (see llm.py). A bare name such as claude-sonnet-5-5 means Anthropic.
VLM_MODEL = os.getenv("VLM_MODEL", "google_genai:gemini-3.5-flash-lite")  # looks at images: report, ask, compare, XAI
VLM_FALLBACK_MODEL = os.getenv("VLM_FALLBACK_MODEL", "")             # optional: used when the main model is busy
VLM_THINKING = os.getenv("VLM_THINKING", "low")                       # Gemini 3 thinking level: minimal|low|medium|high
AGENT_MODEL = os.getenv("AGENT_MODEL") or VLM_MODEL                    # drives the chat agent; must support tool calling
LANGUAGES = ("English", "Bangla")
DEFAULT_LANG = os.getenv("VLM_LANG", "English")
RAG_BACKEND = os.getenv("RAG_BACKEND", "bm25")      # "bm25" (no downloads) or "faiss"
RAG_EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RAG_K = _int("RAG_K", 4)
AGENT_RECURSION_LIMIT = _int("AGENT_RECURSION_LIMIT", 14)

# --- Realtime / caching -------------------------------------------------------
# Live frames can run below 512 px: the U-Net is fully convolutional (sizes must be multiples of 16).
LIVE_SIZES = (256, 320, 384, 448, 512)
PRED_CACHE_SIZE = _int("PRED_CACHE_SIZE", 24)        # cached predictions: re-threshold / XAI without re-running
VLM_CACHE_SIZE = _int("VLM_CACHE_SIZE", 64)          # cached VLM answers (saves free-tier quota)
ASK_HISTORY_TURNS = _int("ASK_HISTORY_TURNS", 4)     # follow-up memory for the Ask tab

# --- Explainability (XAI) -----------------------------------------------------
GRADCAM_LAYERS = ("down4", "up1", "up2", "up3")      # bottleneck -> finer decoder stages
GRADCAM_DEFAULT_LAYER = "up2"
MC_SAMPLES = _int("MC_SAMPLES", 8)                   # Monte-Carlo dropout passes (the model has Dropout2d p=0.1)
MC_MAX_SAMPLES = 24

# --- Server ------------------------------------------------------------------
HOST = os.getenv("HOST", "127.0.0.1")
PORT = _int("PORT", 8000)
APP_TOKEN = os.getenv("APP_TOKEN", "")               # set to require X-App-Token on /api/*
RATE_LIMIT_PER_MIN = _int("RATE_LIMIT_PER_MIN", 30)  # per client, for VLM and chat routes
