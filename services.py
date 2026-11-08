"""Operations shared by the Flask routes and the LangChain tools.

Both entry points call these functions, so the numbers the UI shows and the numbers the
agent quotes always come from the same code path (the server computes them, never the client).

Predictions of uploaded images are cached by (image, dataset, variant): changing the threshold,
opening an XAI view or asking the VLM about the same result does not run the network again.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from PIL import Image

import app_utils as U
import config
import xai
from inference import ModelNotAvailable, ModelRegistry, Prediction, validate_choice

_registry: ModelRegistry | None = None
_lock = threading.Lock()
_pred_cache: OrderedDict[tuple, Prediction] = OrderedDict()
_xai_cache: OrderedDict[tuple, dict] = OrderedDict()
_cache_lock = threading.Lock()


def get_registry() -> ModelRegistry:
    global _registry
    with _lock:
        if _registry is None:
            _registry = ModelRegistry()
        return _registry


def reset_registry() -> None:
    """Used by tests and after replacing weight files."""
    global _registry
    with _lock:
        _registry = None
    with _cache_lock:
        _pred_cache.clear()
        _xai_cache.clear()


def _cache_get(cache: OrderedDict, key):
    with _cache_lock:
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
    return None


def _cache_put(cache: OrderedDict, key, value, limit: int):
    with _cache_lock:
        cache[key] = value
        while len(cache) > max(1, limit):
            cache.popitem(last=False)


@dataclass
class Segmentation:
    image: Image.Image          # original upload (full resolution)
    pred: Prediction
    stats: dict
    overlay: Image.Image        # S x S with the mask drawn on top
    cached: bool = False


def _stats(pred: Prediction, mask: np.ndarray) -> dict:
    stats = U.mask_stats(pred.prob, mask)
    stats["model_ms"] = round(pred.ms, 1)
    return stats


def segment(image: Image.Image, dataset: str, variant: str, threshold: float | None = None,
            size: int | None = None, attention: bool = True) -> Segmentation:
    validate_choice(dataset, variant)
    pred = get_registry().predict(image, dataset, variant, threshold, size=size, attention=attention)
    return Segmentation(image=image, pred=pred, stats=_stats(pred, pred.mask),
                        overlay=U.overlay(image, pred.mask, size=pred.prob.shape[0]))


def run_segmentation(image_id: str, dataset: str, variant: str, threshold: float | None = None) -> Segmentation:
    validate_choice(dataset, variant)
    image = U.load_upload(image_id)
    key = (image_id, dataset, variant)
    base = _cache_get(_pred_cache, key)
    cached = base is not None
    if base is None:
        base = get_registry().predict(image, dataset, variant, attention=True)
        _cache_put(_pred_cache, key, base, config.PRED_CACHE_SIZE)
    thr = config.THRESHOLD if threshold is None else float(threshold)
    mask = base.prob >= thr
    pred = Prediction(prob=base.prob, mask=mask, ms=base.ms, attention=base.attention)
    return Segmentation(image=image, pred=pred, stats=_stats(pred, mask), overlay=U.overlay(image, mask), cached=cached)


@dataclass
class Comparison:
    image: Image.Image
    results: dict[str, Segmentation]
    agreement: dict
    missing: list[str]


def run_compare(image_id: str, dataset: str = config.DEFAULT_DATASET) -> Comparison:
    validate_choice(dataset, "best")
    reg = get_registry()
    have = reg.available_variants(dataset)
    missing = [v for v in config.VARIANTS if v not in have]
    if len(have) < 2:
        raise ModelNotAvailable(
            f"Comparison needs at least two checkpoints for dataset {dataset}; found {len(have)}."
        )
    results = {v: run_segmentation(image_id, dataset, v) for v in have}
    agreement = U.pairwise({v: r.pred.mask for v, r in results.items()})
    image = next(iter(results.values())).image
    return Comparison(image=image, results=results, agreement=agreement, missing=missing)


# --- explainability -------------------------------------------------------------
XAI_METHODS = ("attention", "gradcam", "uncertainty", "mc_dropout")


def run_xai(image_id: str, dataset: str, variant: str, method: str, threshold: float | None = None,
            roi: dict | None = None, layer: str | None = None, samples: int | None = None) -> dict:
    """Returns {"map": (512,512) float in [0,1], "summary": {...}, "method": ...}; cached."""
    if method not in XAI_METHODS:
        raise ValueError(f"method must be one of {', '.join(XAI_METHODS)}.")
    seg = run_segmentation(image_id, dataset, variant, threshold)
    thr = config.THRESHOLD if threshold is None else float(threshold)
    roi_key = None if not roi else (round(float(roi["x"])), round(float(roi["y"])), round(float(roi.get("r", 48))))
    key = (image_id, dataset, variant, method, round(thr, 3), roi_key, layer, samples)
    hit = _cache_get(_xai_cache, key)
    if hit is not None:
        return hit

    if method == "attention":
        amap, gates = xai.attention_summary(seg.pred.attention, seg.pred.mask)
        out = {"map": amap, "summary": {"gates": gates, "combined": "mean of the 4 gates, upsampled"},
               "layers": {k: xai.normalize(xai.upsample(v)) for k, v in seg.pred.attention.items()}}
    elif method == "uncertainty":
        ent = xai.entropy(seg.pred.prob)
        out = {"map": ent, "summary": {
            "mean_entropy_bits": round(float(ent.mean()), 4),
            "high_entropy_percent": round(float((ent > 0.5).mean()) * 100, 2),
            "mean_entropy_on_vessels": round(float(ent[seg.pred.mask].mean()), 4) if seg.pred.mask.any() else 0.0,
        }}
    elif method == "gradcam":
        roi_arr = xai.roi_mask(*roi_key) if roi_key else None
        cam, info = get_registry().gradcam(seg.image, dataset, variant, layer or config.GRADCAM_DEFAULT_LAYER,
                                           roi=roi_arr, threshold=thr)
        summary = {**info, **xai.cam_summary(cam, seg.pred.mask, roi_arr)}
        if roi_key:
            summary["region"] = {"x": roi_key[0], "y": roi_key[1], "r": roi_key[2]}
        out = {"map": cam, "summary": summary}
    else:  # mc_dropout
        n = samples or config.MC_SAMPLES
        mean, std, ms = get_registry().mc_dropout(seg.image, dataset, variant, n)
        out = {"map": xai.normalize(std), "summary": {"samples": int(max(2, min(config.MC_MAX_SAMPLES, n))),
                                                       "ms": round(ms, 1),
                                                       "max_std": round(float(std.max()), 4),
                                                       **xai.mc_summary(mean, std, seg.pred.mask, thr)}}
    out["method"] = method
    _cache_put(_xai_cache, key, out, config.PRED_CACHE_SIZE)
    return out


def context_for_vlm(seg: Segmentation) -> dict:
    """The grounded numbers the VLM may quote: stats, quality, regions, attention focus."""
    _amap, gates = xai.attention_summary(seg.pred.attention, seg.pred.mask) if seg.pred.attention else (None, {})
    ent = xai.entropy(seg.pred.prob)
    return {
        "segmentation": seg.stats,
        "image_quality": xai.image_quality(seg.image),
        "regions": xai.region_stats(seg.pred.prob, seg.pred.mask),
        "attention_gates": gates,
        "uncertainty": {"mean_entropy_bits": round(float(ent.mean()), 4),
                        "high_entropy_percent": round(float((ent > 0.5).mean()) * 100, 2)},
    }


# --- overlay ids (so a tool or the VLM can refer to a saved overlay by id) -----
def save_overlay(image_id: str, dataset: str, variant: str, overlay_img: Image.Image) -> str:
    return U.save_png(overlay_img, f"{image_id}_{dataset}_{variant}_ov")


def load_overlay(overlay_id: str) -> Image.Image:
    if not overlay_id.endswith("_ov"):
        raise ValueError("That is not an overlay id.")
    return U.load_upload(overlay_id)
