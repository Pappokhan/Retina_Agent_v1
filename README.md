# Retina Agent

A realtime web app for the **Dataset A** Attention U-Net (4 variants: best, final epoch, FedAvg, FedProx)
with explainability (XAI), vision-language explanations (default `google_genai:gemini-3.5-flash-lite`),
and a LangChain agent with RAG over your own notes. Datasets B, C and D are no longer served
(their weight folders are left on disk, unused).

**Realtime**
- Live camera: the video plays at full frame rate; frames go to the server one at a time (latest frame
  wins, no queue) and the returned transparent mask is drawn over the video. Input size 256–512 px,
  with *Auto* adapting to the measured latency. A HUD shows model fps and latency. *Capture frame*
  sends the current frame to the full analysis.
- Uploaded images: the threshold slider re-thresholds the probability map in the browser (instant);
  predictions are cached server-side, so re-thresholding, XAI and VLM calls never re-run the network.
- Models are loaded and warmed up at start; `channels_last` gives ~20 % faster CPU inference (identical masks).

**XAI** (all computed from the real model)
- *Attention gates*: the U-Net's own psi coefficients from its 4 gates, with per-gate "focus ratio" on vessels.
- *Grad-CAM (HiResCAM)*: gradient × activation for the vessel score at a chosen layer; click the image to
  explain one region.
- *Uncertainty*: entropy of the probability map. *MC dropout*: std over N dropout-on passes.
- *Explain this map in words*: the VLM narrates the current XAI map using the measured numbers.

**VLM**: reports get the original, mask overlay, uncertainty and attention maps plus server-measured image
quality, per-region vessel density/uncertainty and attention focus; native JSON output; answers are cached;
Ask remembers follow-ups; optional `VLM_FALLBACK_MODEL` when the main model is busy.

```
Quick segment / live camera : browser → /api/predict | /api/live → inference.py        (no API calls, fast)
Explain / Ask / Differences : browser → /api/vlm/*  → server runs the model → vlm.py → your chosen model (llm.py)
Chat agent                  : browser → /api/chat   → agent.py → tools.py → inference / vlm / rag
```

The server always computes the numbers (vessel area, confidence, agreement). The browser never sends
them, and the language models are told to quote them, not invent them.

## Install

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                     # then put a free GOOGLE_API_KEY in .env (see section 7a)
```

For a CPU-only machine you can save a lot of disk with
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` before the line above.


## Run

```bash
python app.py                          # http://127.0.0.1:8000
# production
gunicorn -w 1 --threads 4 -b 0.0.0.0:8000 "app:create_app()"
```

Measure speed on your own hardware, and do not assume a frame rate:

```bash
python scripts/benchmark.py --dataset A --variant best
```

Docker (CPU image; mount the weights, do not bake them in):

```bash
docker build -t retina-agent .
docker run --rm -p 8000:8000 --env-file .env -v "$PWD/weights:/app/weights" retina-agent
```

## What is in the app

| Area | What it does |
|---|---|
| Viewer | Overlay with a strength slider, a drag-to-reveal wipe, probability heatmap, original. Cyan mask, because red disappears on retinal images. |
| Stats | Vessel area, mean confidence, uncertain pixels (probability 0.3 to 0.7), model time, all measured on the server. |
| Compare variants | Every installed variant on one image, plus pairwise Dice between their masks. |
| Live camera | Frames go to `/api/live` one at a time; shows measured frames per second. Needs localhost or HTTPS. |
| Report / Ask / Differences | The vision model looks at the original and overlay and writes a short description, a visual sanity check, or an answer. English or Bangla. |
| Chat agent | LangChain agent with six tools: `list_models`, `segment_image`, `compare_models`, `qc_check`, `inspect_overlay`, `search_knowledge`. |

The agent's `search_knowledge` tool searches `knowledge/*.md`. Those notes are written from your four
notebooks (dataset sizes, splits, reported metrics, caveats). Edit them whenever your results change.
Retrieval is BM25 by default (no model download). For embedding search install the optional packages in
`requirements.txt`, set `RAG_BACKEND=faiss`, and run `python scripts/build_index.py`.

## Tests

```bash
pytest -q
```

The tests use randomly initialised weights with the real key layout, a fake chat model and a scripted
fake agent model, so they need no API key and no real data. If `standard_attention_unet.pth` is present
(or `REAL_CKPT` points at a real checkpoint), one test also loads it strictly and checks the parameter
count against the notebooks (13,746,861).

## Layout

```
app.py            Flask routes (JSON errors, rate limit, optional token)
services.py       shared operations used by both the routes and the agent tools
inference.py      checkpoint loading, LRU model cache, predict()
models.py         AttentionUNet, copied unchanged from the notebooks
app_utils.py      upload validation, overlay/heatmap, stats, agreement
vlm.py            vision-model calls: report, ask (streaming), compare
llm.py            provider table (Gemini, Ollama, Claude) and friendly errors
tools.py          the agent's six tools        agent.py   create_agent + system prompt + memory
rag.py            retrieval over knowledge/    config.py  all settings (env overridable)
knowledge/        notes the agent searches     templates/ static/   the UI
scripts/          organize_weights, check_weights, eval_folder, benchmark, build_index
tests/            pytest suite
```
