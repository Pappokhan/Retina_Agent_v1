from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

try:  
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


WEIGHTS_DIR = Path(os.getenv("WEIGHTS_DIR", BASE_DIR / "weights"))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", BASE_DIR / "uploads"))
KNOWLEDGE_DIR = Path(os.getenv("KNOWLEDGE_DIR", BASE_DIR / "knowledge"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", BASE_DIR / "faiss_index"))

IMG_SIZE = 512
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
THRESHOLD = 0.5                      # notebooks binarise at 0.5
UNCERTAIN_BAND = (0.3, 0.7)          # pixels in this probability band count as "uncertain"
MODEL_CACHE_SIZE = _int("MODEL_CACHE_SIZE", 4)   # ~55 MB per model; all 4 variants stay loaded

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

MAX_UPLOAD_MB = _int("MAX_UPLOAD_MB", 20)
MAX_IMAGE_PIXELS = _int("MAX_IMAGE_PIXELS", 40_000_000)
UPLOAD_TTL_HOURS = _float("UPLOAD_TTL_HOURS", 6.0)   
ALLOWED_FORMATS = {"PNG", "JPEG", "TIFF", "BMP", "GIF", "WEBP", "PPM"}
OVERLAY_RGB = (0, 229, 208)  

VLM_MODEL = os.getenv("VLM_MODEL", "google_genai:gemini-3.5-flash-lite")  
VLM_FALLBACK_MODEL = os.getenv("VLM_FALLBACK_MODEL", "")          
VLM_THINKING = os.getenv("VLM_THINKING", "low")                      
AGENT_MODEL = os.getenv("AGENT_MODEL") or VLM_MODEL                  
LANGUAGES = ("English", "Bangla")
DEFAULT_LANG = os.getenv("VLM_LANG", "English")
RAG_BACKEND = os.getenv("RAG_BACKEND", "bm25")      
RAG_EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RAG_K = _int("RAG_K", 4)
AGENT_RECURSION_LIMIT = _int("AGENT_RECURSION_LIMIT", 14)

LIVE_SIZES = (256, 320, 384, 448, 512)
PRED_CACHE_SIZE = _int("PRED_CACHE_SIZE", 24)       
VLM_CACHE_SIZE = _int("VLM_CACHE_SIZE", 64)        
ASK_HISTORY_TURNS = _int("ASK_HISTORY_TURNS", 4)     

GRADCAM_LAYERS = ("down4", "up1", "up2", "up3")     
GRADCAM_DEFAULT_LAYER = "up2"
MC_SAMPLES = _int("MC_SAMPLES", 8)
MC_MAX_SAMPLES = 24

HOST = os.getenv("HOST", "127.0.0.1")
PORT = _int("PORT", 8000)
APP_TOKEN = os.getenv("APP_TOKEN", "")               
RATE_LIMIT_PER_MIN = _int("RATE_LIMIT_PER_MIN", 30)  # per client, for VLM and chat routes
