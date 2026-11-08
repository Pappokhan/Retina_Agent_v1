"""Image I/O, overlays and small numeric helpers shared by the routes and the agent tools."""
from __future__ import annotations

import base64
import io
import re
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

import config

ID_RE = re.compile(r"^[A-Za-z0-9_]{1,64}$")


class BadImage(ValueError):
    """The uploaded bytes are not a usable image."""


# --- ids and paths -------------------------------------------------------------
def check_id(value: str) -> str:
    """Only letters, digits and underscores: stops path traversal through ids."""
    if not isinstance(value, str) or not ID_RE.match(value):
        raise ValueError("Invalid id. Use the id returned by the upload step.")
    return value


def upload_path(image_id: str) -> Path:
    root = Path(config.UPLOAD_DIR).resolve()
    path = (root / f"{check_id(image_id)}.png").resolve()
    if path.parent != root:
        raise ValueError("Invalid id.")
    return path


# --- decoding / saving ---------------------------------------------------------
def decode_image(data: bytes) -> Image.Image:
    """Validate and decode untrusted bytes into an RGB image."""
    if not data:
        raise BadImage("The file is empty.")
    try:
        probe = Image.open(io.BytesIO(data))
        fmt = probe.format
        if fmt not in config.ALLOWED_FORMATS:
            raise BadImage(f"Unsupported image format ({fmt}). Use PNG, JPEG, TIFF, BMP, GIF or WEBP.")
        if probe.width * probe.height > config.MAX_IMAGE_PIXELS:
            raise BadImage("The image is too large. Resize it below "
                           f"{config.MAX_IMAGE_PIXELS // 1_000_000} megapixels and try again.")
        probe.verify()                              # verify() invalidates the object, so reopen
        img = Image.open(io.BytesIO(data))
        img.load()
    except BadImage:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise BadImage("The image is too large to open safely.") from None
    except Exception as exc:  # noqa: BLE001
        raise BadImage("This file is not a readable image.") from exc
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def save_upload(data: bytes) -> tuple[str, int, int]:
    img = decode_image(data)
    image_id = uuid.uuid4().hex[:16]
    Path(config.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    img.save(upload_path(image_id), "PNG")
    return image_id, img.width, img.height


def load_upload(image_id: str) -> Image.Image:
    path = upload_path(image_id)
    if not path.is_file():
        raise FileNotFoundError("Unknown image id. It may have expired; upload the image again.")
    with Image.open(path) as im:
        return im.convert("RGB")


def save_png(image: Image.Image, name: str) -> str:
    check_id(name)
    Path(config.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    image.save(upload_path(name), "PNG")
    return name


_last_purge = 0.0


def purge_uploads(force: bool = False) -> int:
    """Delete uploads older than UPLOAD_TTL_HOURS (retinal images can be sensitive)."""
    global _last_purge
    now = time.time()
    if not force and now - _last_purge < 600:
        return 0
    _last_purge = now
    root = Path(config.UPLOAD_DIR)
    if not root.is_dir():
        return 0
    cutoff = now - config.UPLOAD_TTL_HOURS * 3600
    removed = 0
    for f in root.glob("*.png"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
                removed += 1
        except OSError:
            pass
    return removed


# --- rendering -----------------------------------------------------------------
def resized(img: Image.Image, size: int = config.IMG_SIZE) -> Image.Image:
    img = img.convert("RGB")
    return img if img.size == (size, size) else img.resize((size, size), Image.Resampling.BILINEAR)


def overlay(img: Image.Image, mask: np.ndarray, alpha: float = 0.6,
            rgb: tuple[int, int, int] = config.OVERLAY_RGB, size: int | None = None) -> Image.Image:
    size = size or mask.shape[0]
    base = np.asarray(resized(img, size)).astype(np.float32)
    out = base.copy()
    m = mask.astype(bool)
    out[m] = (1.0 - alpha) * base[m] + alpha * np.asarray(rgb, dtype=np.float32)
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def mask_rgba(mask: np.ndarray, rgb: tuple[int, int, int] = config.OVERLAY_RGB) -> Image.Image:
    """Transparent PNG layer: mask colour where vessel, fully transparent elsewhere (for live video)."""
    h, w = mask.shape
    out = np.zeros((h, w, 4), dtype=np.uint8)
    m = mask.astype(bool)
    out[m, :3] = rgb
    out[m, 3] = 255
    return Image.fromarray(out, "RGBA")


def prob_png(prob: np.ndarray) -> Image.Image:
    """8-bit greyscale probability map, so the browser can re-threshold instantly."""
    return Image.fromarray((np.clip(prob, 0, 1) * 255).round().astype(np.uint8), "L")


_STOPS = np.array([0.0, 0.35, 0.65, 1.0])
_COLORS = np.array([[12, 18, 38], [24, 84, 140], [0, 200, 180], [250, 246, 170]], dtype=np.float32)


def heatmap(prob: np.ndarray) -> Image.Image:
    p = np.clip(prob, 0.0, 1.0)
    rgb = np.stack([np.interp(p, _STOPS, _COLORS[:, c]) for c in range(3)], axis=-1)
    return Image.fromarray(rgb.astype(np.uint8))


def to_b64(img: Image.Image, fmt: str = "PNG", quality: int = 85) -> str:
    buf = io.BytesIO()
    if fmt.upper() == "JPEG":
        img.convert("RGB").save(buf, "JPEG", quality=quality)
    else:
        img.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# --- numbers --------------------------------------------------------------------
def mask_stats(prob: np.ndarray, mask: np.ndarray) -> dict:
    lo, hi = config.UNCERTAIN_BAND
    fg = mask.astype(bool)
    return {
        "area_percent": round(float(fg.mean()) * 100, 2),
        "mean_confidence": round(float(prob[fg].mean()) if fg.any() else 0.0, 3),
        "uncertain_percent": round(float(((prob > lo) & (prob < hi)).mean()) * 100, 2),
    }


def pairwise(masks: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """Pairwise Dice (agreement) and disagreement percent between masks."""
    names = list(masks)
    out: dict[str, dict[str, float]] = {"dice": {}, "disagreement_percent": {}}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ma, mb = masks[a].astype(bool), masks[b].astype(bool)
            denom = ma.sum() + mb.sum()
            dice = 1.0 if denom == 0 else 2.0 * float((ma & mb).sum()) / float(denom)
            out["dice"][f"{a}_vs_{b}"] = round(dice, 4)
            out["disagreement_percent"][f"{a}_vs_{b}"] = round(float((ma ^ mb).mean()) * 100, 2)
    return out
