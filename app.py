from __future__ import annotations

import hmac
import itertools
import json
import sys
import logging
import threading
import time
from collections import defaultdict, deque
from functools import wraps
from urllib.parse import quote

import torch
from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

import agent
import app_utils as U
import config
import llm
import rag
import services
import vlm
import xai
from inference import BadCheckpoint, ModelNotAvailable

log = logging.getLogger("retina-agent")


def _body() -> dict:
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


def _choice(body: dict) -> tuple[str, str]:
    return str(body.get("dataset") or config.DEFAULT_DATASET), str(body.get("variant") or "best")


def _roi(body: dict) -> dict | None:
    roi = body.get("roi")
    if not roi:
        return None
    try:
        out = {"x": float(roi["x"]), "y": float(roi["y"]), "r": float(roi.get("r", 48))}
    except (TypeError, KeyError, ValueError):
        raise ValueError("roi must be {x, y, r} in 512-pixel image coordinates.") from None
    if not (0 <= out["x"] < config.IMG_SIZE and 0 <= out["y"] < config.IMG_SIZE and 4 <= out["r"] <= 256):
        raise ValueError("roi must lie inside the 512 x 512 image, with r between 4 and 256.")
    return out


def _client_key(body: dict) -> str:
    sid = str(body.get("session_id") or "")
    return sid if sid and U.ID_RE.match(sid) else (request.remote_addr or "?")


def _threshold(body: dict) -> float | None:
    raw = body.get("threshold")
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError("threshold must be a number between 0.05 and 0.95.") from None
    if not 0.05 <= value <= 0.95:
        raise ValueError("threshold must be between 0.05 and 0.95.")
    return value


def _primed(gen):
    """Pull the first chunk so errors become a proper HTTP error, not a half-sent response."""
    try:
        first = next(gen)
    except StopIteration:
        return iter(())
    return itertools.chain([first], gen)


def _safe_stream(chunks):
    try:
        yield from chunks
    except Exception:  # noqa: BLE001 - headers are already sent; end the text cleanly
        log.exception("stream interrupted")
        yield "\n[The answer was interrupted. Ask again.]"


def _check_environment() -> None:
    """Warn loudly at start-up when this interpreter lacks packages (common with a wrong IDE interpreter)."""
    missing = llm.missing_packages(config.VLM_MODEL, config.AGENT_MODEL)
    if not missing:
        return
    line = "=" * 78
    print(f"\n{line}\nWARNING: this Python is missing packages: {', '.join(missing)}\n"
          f"  Python in use : {sys.executable}\n"
          f"  Segmentation and XAI still work; Report, Ask and Chat will not.\n"
          f"  Fix (pick one):\n"
          f"   1. Use this project's environment: .venv\\Scripts\\python.exe app.py\n"
          f"      (PyCharm: Settings > Project > Python Interpreter > Add > Existing > .venv\\Scripts\\python.exe)\n"
          f"   2. Or install them here: \"{sys.executable}\" -m pip install -r requirements.txt\n{line}\n",
          flush=True)


def recommended_live_size() -> int:
    return 512 if torch.cuda.is_available() else 256


def _warmup() -> None:
    reg = services.get_registry()
    for v in reg.available_variants(config.DEFAULT_DATASET):
        reg.get(config.DEFAULT_DATASET, v)
    reg.warmup(config.DEFAULT_DATASET, "best", sizes=(recommended_live_size(), config.IMG_SIZE))
    log.info("warm-up done: %s", reg.loaded())


def create_app(warmup: bool = False) -> Flask:
    logging.basicConfig(level=logging.INFO)
    app = Flask(__name__)
    app.json.sort_keys = False      # keep dataset/variant order as defined in config.py
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024
    U.purge_uploads(force=True)
    if warmup:   # load the models and run one pass in the background, so the first click is fast
        threading.Thread(target=_warmup, daemon=True, name="warmup").start()
    hits: dict[str, deque] = defaultdict(deque)

    @app.before_request
    def require_token():
        if config.APP_TOKEN and request.path.startswith("/api/") and request.path != "/api/health":
            sent = request.headers.get("X-App-Token", "")
            if not hmac.compare_digest(sent.encode(), config.APP_TOKEN.encode()):
                return jsonify(error="This server needs an access token. Enter it in the header.",
                               code="unauthorized"), 401

    def limited(fn):
        """Per-client rate limit for routes that cost money (VLM and chat)."""
        @wraps(fn)
        def inner(*args, **kwargs):
            q = hits[request.remote_addr or "?"]
            now = time.monotonic()
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= config.RATE_LIMIT_PER_MIN:
                return jsonify(error="Too many requests. Wait a few seconds and try again.",
                               code="rate_limited"), 429
            q.append(now)
            return fn(*args, **kwargs)
        return inner

    def fail(message: str, status: int, code: str):
        return jsonify(error=message, code=code), status

    @app.errorhandler(ModelNotAvailable)
    def _missing_model(e):
        return fail(str(e) + " Copy the checkpoint into the weights folder (see README).", 409, "model_missing")

    @app.errorhandler(BadCheckpoint)
    def _bad_checkpoint(e):
        return fail(f"A checkpoint could not be loaded: {e}", 500, "bad_checkpoint")

    @app.errorhandler(U.BadImage)
    def _bad_image(e):
        return fail(str(e), 400, "bad_image")

    @app.errorhandler(ValueError)
    def _bad_request(e):
        return fail(str(e), 400, "bad_request")

    @app.errorhandler(FileNotFoundError)
    def _not_found(e):
        return fail(str(e), 404, "not_found")

    @app.errorhandler(vlm.VLMUnavailable)
    def _no_key(e):
        return fail(str(e), 503, "vlm_unavailable")

    @app.errorhandler(vlm.VLMRateLimited)
    def _rate_limited(e):
        return fail(str(e), 429, "provider_rate_limited")

    @app.errorhandler(vlm.VLMError)
    def _vlm_failed(e):
        return fail(str(e), 502, "vlm_error")

    @app.errorhandler(RequestEntityTooLarge)
    def _too_large(_e):
        return fail(f"The file is larger than {config.MAX_UPLOAD_MB} MB. Resize it and try again.", 413, "too_large")

    @app.errorhandler(Exception)
    def _unexpected(e):
        if isinstance(e, HTTPException):
            return e
        log.exception("unhandled error")
        return fail("Something went wrong on the server. Check the server log for details.", 500, "server_error")

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        reg = services.get_registry()
        installed = sum(ok for row in reg.available().values() for ok in row.values())
        return jsonify(
            ok=True, device=str(reg.device), cuda=torch.cuda.is_available(),
            models_installed=installed, models_total=len(config.DATASETS) * len(config.VARIANTS),
            models_loaded=reg.loaded(), dataset=config.DEFAULT_DATASET,
            vlm_configured=llm.is_configured(config.VLM_MODEL),
            vlm_model=config.VLM_MODEL, vlm_label=llm.info(config.VLM_MODEL)["label"],
            vlm_privacy=llm.info(config.VLM_MODEL)["privacy"],
            agent_model=config.AGENT_MODEL, agent_configured=llm.is_configured(config.AGENT_MODEL),
            missing_packages=llm.missing_packages(config.VLM_MODEL, config.AGENT_MODEL),
            python=sys.executable,
            auth_required=bool(config.APP_TOKEN), rag_backend=config.RAG_BACKEND,
        )

    @app.get("/api/models")
    def models():
        return jsonify(
            datasets=config.DATASETS, dataset=config.DEFAULT_DATASET, variants=config.VARIANTS,
            available=services.get_registry().available(), default_threshold=config.THRESHOLD,
            languages=list(config.LANGUAGES), live_sizes=list(config.LIVE_SIZES),
            live_size=recommended_live_size(), xai_methods=list(services.XAI_METHODS),
            gradcam_layers=list(config.GRADCAM_LAYERS), gradcam_layer=config.GRADCAM_DEFAULT_LAYER,
            mc_samples=config.MC_SAMPLES,
        )

    @app.post("/api/upload")
    def upload():
        file = request.files.get("image")
        if file is None:
            raise ValueError("Choose an image file to upload.")
        U.purge_uploads()
        image_id, width, height = U.save_upload(file.read())
        return jsonify(image_id=image_id, width=width, height=height)

    @app.post("/api/predict")
    def predict():
        body = _body()
        dataset, variant = _choice(body)
        threshold = _threshold(body)
        seg = services.run_segmentation(str(body.get("image_id", "")), dataset, variant, threshold)
        out = dict(dataset=dataset, variant=variant, threshold=threshold or config.THRESHOLD,
                   stats=seg.stats, cached=seg.cached)
        if not body.get("stats_only"):   # the browser re-thresholds locally and then only needs new stats
            out.update(original=U.to_b64(U.resized(seg.image)), overlay=U.to_b64(seg.overlay),
                       heatmap=U.to_b64(U.heatmap(seg.pred.prob)), prob=U.to_b64(U.prob_png(seg.pred.prob)))
        return jsonify(out)

    @app.post("/api/live")
    def live():
        """Per-frame endpoint for the camera: nothing is saved, no VLM, no agent (kept fast).

        Returns a transparent mask layer that the browser draws over the live video, so the video
        itself never waits for the model. Optional xai=uncertainty|attention adds a small heat layer.
        """
        t0 = time.perf_counter()
        file = request.files.get("image")
        if file is None:
            raise ValueError("No frame received.")
        dataset = request.form.get("dataset") or config.DEFAULT_DATASET
        variant = request.form.get("variant") or "best"
        try:
            size = int(request.form.get("size") or recommended_live_size())
        except ValueError:
            raise ValueError("size must be a number.") from None
        raw_thr = request.form.get("threshold")
        threshold = _threshold({"threshold": raw_thr}) if raw_thr else None
        mode = request.form.get("xai", "none")
        if mode not in ("none", "uncertainty", "attention"):
            raise ValueError("xai must be none, uncertainty or attention.")
        frame = U.decode_image(file.read())
        t1 = time.perf_counter()
        seg = services.segment(frame, dataset, variant, threshold, size=size, attention=mode == "attention")
        out = dict(mask=U.to_b64(U.mask_rgba(seg.pred.mask)), size=size, stats=seg.stats)
        if mode == "uncertainty":
            out["xai"] = U.to_b64(xai.colorize(xai.entropy(seg.pred.prob), size), "JPEG", 70)
        elif mode == "attention":
            amap, _ = xai.attention_summary(seg.pred.attention, seg.pred.mask)
            out["xai"] = U.to_b64(xai.colorize(amap, size), "JPEG", 70)
        out["timing"] = {"decode_ms": round((t1 - t0) * 1000, 1), "model_ms": seg.stats["model_ms"],
                         "server_ms": round((time.perf_counter() - t0) * 1000, 1)}
        return jsonify(out)

    @app.post("/api/compare")
    def compare():
        body = _body()
        cmp = services.run_compare(str(body.get("image_id", "")), str(body.get("dataset") or config.DEFAULT_DATASET))
        return jsonify(
            original=U.to_b64(U.resized(cmp.image)),
            variants={v: {"label": config.VARIANTS[v]["label"], "overlay": U.to_b64(r.overlay), "stats": r.stats}
                      for v, r in cmp.results.items()},
            agreement=cmp.agreement, missing=cmp.missing,
        )

    def _xai_result(body: dict):
        dataset, variant = _choice(body)
        image_id = str(body.get("image_id", ""))
        method = str(body.get("method") or "attention")
        samples = body.get("samples")
        try:
            samples = int(samples) if samples is not None else None
        except (TypeError, ValueError):
            raise ValueError("samples must be a whole number.") from None
        threshold = _threshold(body)
        res = services.run_xai(image_id, dataset, variant, method, threshold, _roi(body),
                               body.get("layer") or None, samples)
        seg = services.run_segmentation(image_id, dataset, variant, threshold)
        return dataset, variant, seg, res

    @app.post("/api/xai")
    def explain():
        _dataset, _variant, seg, res = _xai_result(_body())
        heat = xai.colorize(res["map"])
        out = dict(method=res["method"], summary=res["summary"], heat=U.to_b64(heat, "JPEG", 88),
                   blend=U.to_b64(xai.blend(U.resized(seg.image), heat), "JPEG", 88))
        if "layers" in res:
            out["layers"] = {k: U.to_b64(xai.colorize(v), "JPEG", 80) for k, v in res["layers"].items()}
        return jsonify(out)

    def _vlm_maps(seg) -> dict:
        maps = {"uncertainty (entropy)": xai.colorize(xai.entropy(seg.pred.prob))}
        if seg.pred.attention:
            amap, _ = xai.attention_summary(seg.pred.attention, seg.pred.mask)
            maps["attention-gate"] = xai.colorize(amap)
        return maps

    @app.post("/api/vlm/report")
    @limited
    def vlm_report():
        body = _body()
        dataset, variant = _choice(body)
        threshold = _threshold(body)
        image_id = str(body.get("image_id", ""))
        seg = services.run_segmentation(image_id, dataset, variant, threshold)
        lang = vlm._lang(body.get("lang"))
        context = services.context_for_vlm(seg)
        notes = rag.retrieve(rag.query_for("report", variant), k=5)
        rep = vlm.cached(("report", image_id, dataset, variant, seg.stats["area_percent"], lang),
                         lambda: vlm.report(seg.image, seg.overlay, seg.stats, lang, context, _vlm_maps(seg), notes))
        return jsonify(report=rep, stats=seg.stats, context=context, sources=rag.sources(notes))

    @app.post("/api/vlm/ask")
    @limited
    def vlm_ask():
        body = _body()
        dataset, variant = _choice(body)
        image_id = str(body.get("image_id", ""))
        seg = services.run_segmentation(image_id, dataset, variant, _threshold(body))
        question = str(body.get("question", ""))
        key = (_client_key(body), image_id, variant)
        maps = _vlm_maps(seg) if body.get("with_xai", True) else None
        past = vlm.history(key)
        notes = rag.retrieve(rag.query_for("ask", variant, " ".join([q for q, _ in past[-1:]] + [question])))
        gen = vlm.ask_stream(seg.overlay, question, original=seg.image, lang=body.get("lang"),
                             context=services.context_for_vlm(seg), maps=maps, past=past, notes=notes)

        def recorded(chunks):
            parts = []
            for c in chunks:
                parts.append(c)
                yield c
            vlm.remember(key, question.strip()[:500], "".join(parts))

        return Response(stream_with_context(_safe_stream(recorded(_primed(gen)))),
                        mimetype="text/plain; charset=utf-8",
                        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no",
                                 "X-Sources": quote(json.dumps(rag.sources(notes)))})

    @app.post("/api/vlm/ask/reset")
    def vlm_ask_reset():
        body = _body()
        _dataset, variant = _choice(body)
        vlm.forget((_client_key(body), str(body.get("image_id", "")), variant))
        return jsonify(ok=True)

    @app.post("/api/vlm/xai")
    @limited
    def vlm_xai():
        body = _body()
        dataset, variant, seg, res = _xai_result(body)
        lang = vlm._lang(body.get("lang"))
        key = ("xai", str(body.get("image_id")), dataset, variant, res["method"],
               json.dumps(res["summary"], sort_keys=True, default=str), lang)
        notes = rag.retrieve(rag.query_for("xai", variant, res["method"]))
        text = vlm.cached(key, lambda: vlm.explain_xai(
            seg.image, seg.overlay, xai.blend(U.resized(seg.image), xai.colorize(res["map"])),
            res["method"], res["summary"], seg.stats, lang, notes))
        return jsonify(text=text, summary=res["summary"], method=res["method"], sources=rag.sources(notes))

    @app.post("/api/vlm/compare")
    @limited
    def vlm_compare():
        body = _body()
        cmp = services.run_compare(str(body.get("image_id", "")), str(body.get("dataset") or config.DEFAULT_DATASET))
        stats = {"area_percent": {v: r.stats["area_percent"] for v, r in cmp.results.items()},
                 "pairwise_dice": cmp.agreement["dice"],
                 "disagreement_percent": cmp.agreement["disagreement_percent"]}
        notes = rag.retrieve(rag.query_for("compare"))
        text = vlm.compare(cmp.image, {v: r.overlay for v, r in cmp.results.items()}, stats, body.get("lang"), notes)
        return jsonify(text=text, stats=stats, sources=rag.sources(notes))

    @app.get("/api/knowledge/search")
    def knowledge_search():
        q = request.args.get("q", "").strip()
        if not q:
            raise ValueError("Type something to search the notes for.")
        if len(q) > 300:
            raise ValueError("Keep the search under 300 characters.")
        return jsonify(query=q, results=rag.sources(rag.retrieve(q, k=6)))

    @app.post("/api/chat")
    @limited
    def chat():
        body = _body()
        session_id = U.check_id(str(body.get("session_id", "")))
        message = str(body.get("message", "")).strip()
        if not message:
            raise ValueError("Type a message first.")
        if len(message) > 2000:
            raise ValueError("That message is too long. Keep it under 2000 characters.")
        ctx = [f"{k}={U.check_id(str(body[k]))}" for k in ("image_id", "dataset", "variant") if body.get(k)]
        prefix = f"[ui_context {' '.join(ctx)}]\n" if ctx else ""
        return jsonify(**agent.chat(prefix + message, session_id))

    return app


if __name__ == "__main__":
    from werkzeug.serving import WSGIRequestHandler

    WSGIRequestHandler.protocol_version = "HTTP/1.1" 
    _check_environment()
    create_app(warmup=True).run(host=config.HOST, port=config.PORT, threaded=True)
