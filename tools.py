from __future__ import annotations

import functools
import json

from langchain_core.tools import tool

import app_utils as U
import config
import rag
import services
import vlm
from inference import BadCheckpoint, ModelNotAvailable

_EXPECTED = (ValueError, FileNotFoundError, ModelNotAvailable, BadCheckpoint, vlm.VLMUnavailable, vlm.VLMError)


def _safe(fn):
    @functools.wraps(fn)
    def inner(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except _EXPECTED as exc:
            return f"ERROR: {exc}"
    return inner


@tool
@_safe
def list_models() -> str:
    """List which checkpoints of the Dataset A model are installed (variants best, standard, fedavg, fedprox)."""
    avail = services.get_registry().available()
    lines = [f"Dataset {ds}: " + (", ".join(v for v, ok in row.items() if ok) or "no checkpoints")
             for ds, row in avail.items()]
    return "\n".join(lines)


@tool
@_safe
def segment_image(image_id: str, variant: str = "best", dataset: str = config.DEFAULT_DATASET) -> str:
    """Segment an uploaded image with the Dataset A model. variant is best, standard, fedavg or fedprox.
    Returns the measured stats and an overlay_id that inspect_overlay can look at."""
    seg = services.run_segmentation(image_id, dataset, variant)
    overlay_id = services.save_overlay(image_id, dataset, variant, seg.overlay)
    return json.dumps({"dataset": dataset, "variant": variant, "overlay_id": overlay_id, **seg.stats})


@tool
@_safe
def compare_models(image_id: str, dataset: str = config.DEFAULT_DATASET) -> str:
    """Run every installed variant of the Dataset A model on the same image. Returns each variant's vessel
    area percent plus pairwise Dice agreement and disagreement percent between the variants."""
    cmp = services.run_compare(image_id, dataset)
    return json.dumps({
        "dataset": dataset,
        "area_percent": {v: r.stats["area_percent"] for v, r in cmp.results.items()},
        "pairwise_dice": cmp.agreement["dice"],
        "disagreement_percent": cmp.agreement["disagreement_percent"],
        "variants_not_installed": cmp.missing,
    })


@tool
@_safe
def qc_check(image_id: str, variant: str = "best", dataset: str = config.DEFAULT_DATASET) -> str:
    """Segment the image, then have a vision model do a visual sanity check of the mask (mask on
    background, obvious misses, over-segmentation). This is NOT an accuracy measurement."""
    seg = services.run_segmentation(image_id, dataset, variant)
    rep = vlm.report(seg.image, seg.overlay, seg.stats, context=services.context_for_vlm(seg))
    return json.dumps({"dataset": dataset, "variant": variant, "stats": seg.stats, "report": rep})


@tool
@_safe
def inspect_overlay(overlay_id: str, question: str) -> str:
    """Answer a visual question about a segmentation overlay produced by segment_image (pass its overlay_id)."""
    overlay = services.load_overlay(overlay_id)
    original = None
    try:
        original = U.load_upload(overlay_id.split("_")[0])
    except (ValueError, FileNotFoundError):
        pass
    return vlm.ask(overlay, question, original=original)


@tool
@_safe
def search_knowledge(query: str) -> str:
    """Search the project notes (Dataset A, model card, training, reported metrics, federated runs,
    explanation maps, measurements, live mode, limitations and privacy). Returns numbered notes tagged
    'file > section'; cite those labels. Use it for every factual question about the model or the app."""
    return rag.search(query)


@tool
@_safe
def explain_prediction(image_id: str, method: str = "attention", variant: str = "best",
                       dataset: str = config.DEFAULT_DATASET) -> str:
    """Explainability (XAI) numbers for the model's prediction on an uploaded image.
    method: attention (attention-gate focus on vessels), gradcam (Seg-Grad-CAM evidence),
    uncertainty (entropy of the probability map) or mc_dropout (variation over dropout passes; slower).
    Returns measured numbers only; quote them, do not reinterpret them as accuracy."""
    res = services.run_xai(image_id, dataset, variant, method)
    return json.dumps({"method": method, "variant": variant, "summary": res["summary"]}, default=str)


TOOLS = [list_models, segment_image, compare_models, qc_check, inspect_overlay, explain_prediction, search_knowledge]
