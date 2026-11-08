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

## 1. Install

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                     # then put a free GOOGLE_API_KEY in .env (see section 7a)
```

For a CPU-only machine you can save a lot of disk with
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` before the line above.

## 2. Put the 16 checkpoints in place

Each notebook saved four files with the same names (`best_model_AttentionUNet.pth`,
`standard_attention_unet.pth`, `fedavg_attention_unet.pth`, `fedprox_attention_unet.pth`), so collect
one folder per notebook and let the script sort them:

```bash
python scripts/organize_weights.py --A /path/to/notebook_A_files --B /path/to/notebook_B_files \
                                   --C /path/to/notebook_C_files --D /path/to/notebook_D_files
python scripts/check_weights.py
```

Result:

```
weights/
├── A/  best.pth  standard.pth  fedavg.pth  fedprox.pth
├── B/  ...
├── C/  ...
└── D/  ...
```

`check_weights.py` loads every file strictly into `AttentionUNet` and runs a 512×512 forward pass.
The app also works with only some files present; missing ones are greyed out in the UI.

## 3. Check that deployment reproduces your notebook numbers

Before trusting the app, score a checkpoint on the notebook's own test split with the app's own
inference code. It rebuilds the notebook split (`random_split`, seed 42) and prints the difference from
the numbers the notebook reported:

```bash
python scripts/eval_folder.py --dataset A --variant standard \
    --images "/path/to/Dataset D/images" --masks "/path/to/Dataset D/masks"
```

Dataset B's notebook joined a train and a test folder, so pass `--images` and `--masks` twice, train
first. Small differences (about 0.001 to 0.003 Dice) can come from GPU versus CPU and float16; a large
gap means the preprocessing or the dataset letter is wrong.

## 4. Run

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

## 5. What is in the app

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

## 6. Things you should know before showing results

These come from reading the four notebooks, and they are also written into `knowledge/` so the agent
tells users the same thing.

1. **The federated code does not do federated averaging.** In all four notebooks the five `FLClient`
   objects share one model object, so each round trains the same model on five shards in sequence and
   FedAvg averages five identical copies. FedProx's anchor is the previous client's result, not the
   round's starting weights. The `fedavg` and `fedprox` checkpoints are therefore sequential shard
   training with those labels, and the difference between them is not evidence about either algorithm.
   Fix: one `copy.deepcopy(global_model)` per client and a snapshot of the global weights at the start
   of each round, then retrain and re-save.
2. **The notebooks' clDice is not clDice.** It uses a 3×3 erosion instead of skeletonisation. Do not
   report those numbers (0.5172, 0.5356, 0.2801, 0.6276) as clDice.
3. **Datasets B, C and D are tiny** (28 to 100 images; test sets of 15, 8 and 6). Differences of a few
   Dice points are noise.
4. **Dataset letters and folder names disagree** inside the notebooks (notebook A reads "Dataset D",
   B reads "Dataset A", C reads "Dataset XX", D reads "Dataset B"). The app uses the notebook letter.
   If you meant the folder names, swap the labels in `config.py` and `knowledge/datasets.md`.
5. **The notebooks' test metrics are from the final-epoch model**, which is the `standard` checkpoint,
   not the `best` one.
6. **Each model only knows its own dataset.** An image from another camera or population can give a poor
   mask with no warning. The VLM check is a sanity check, not accuracy.

## 7. Privacy and security

- Research software, not a medical device. The UI and every prompt say so; the models are told never to
  diagnose.
- Uploads are stored as PNG under `uploads/` and deleted after `UPLOAD_TTL_HOURS` (default 6).
- **Report, Ask, Differences and the chat agent send the image and overlay to the provider named in
  `VLM_MODEL`** (the UI footer says which, and what that means). Do not use them on patient data unless
  your consent and data agreements allow that. Segmentation and the live camera never call a model API.
  With `ollama:` models nothing leaves your machine.
- Set `APP_TOKEN` before exposing the app beyond your own machine; the UI then asks for the token. Put it
  behind HTTPS (also required for the camera). VLM and chat routes are rate limited per client.
- Checkpoints are loaded with `torch.load(..., weights_only=True)`, so a tampered `.pth` cannot run code.
- Chat memory (`InMemorySaver`) lives in the server process and is lost on restart.

### 7a. Free models

Set `VLM_MODEL` (images: Report, Ask, Differences) and `AGENT_MODEL` (the chat agent) in `.env`:

| Option | Cost | Images go to | Notes |
|---|---|---|---|
| `google_genai:gemini-2.5-flash` + `GOOGLE_API_KEY` from Google AI Studio | free tier | Google | Rate limits apply and are not fixed; when reached the app shows "wait a minute". Google says free-tier content may be used to improve its products and reviewed by humans, so use no patient data. EEA, Swiss and UK users get paid-service data terms. |
| `ollama:gemma4` (any local vision model) | free | stays on your machine | Needs RAM or a GPU. The agent needs a model that supports tool calling; if the chat agent fails, set `AGENT_MODEL` to a tool-capable model. |
| `anthropic:claude-sonnet-5-5` + `ANTHROPIC_API_KEY` | paid | Anthropic | |

Model names change often; check the provider's current list if one is rejected. Small free or local
models describe images less reliably than large ones, and the prompts forbid diagnosis either way.

## 8. Tests

```bash
pytest -q
```

The tests use randomly initialised weights with the real key layout, a fake chat model and a scripted
fake agent model, so they need no API key and no real data. If `standard_attention_unet.pth` is present
(or `REAL_CKPT` points at a real checkpoint), one test also loads it strictly and checks the parameter
count against the notebooks (13,746,861).

## 9. What was and was not verified

Verified here (CPU, Python 3.13, versions in `requirements-tested.txt`): 99 automated tests; a real
checkpoint loads strictly with the notebooks' parameter count; the UI driven in headless Chromium
(upload, segment, wipe, compare, error messages, mobile width); the `gunicorn` command from section 4.

Not verified, because it needs things that were not available: calls to any real model API: Gemini, Ollama or Claude (the
tests use a fake model, so the prompts and the report quality are unchecked), the camera mode (needs a
camera and HTTPS or localhost), a GPU (all timings above are CPU; a 512×512 image took about 1.6 s on a
2-core CPU), `RAG_BACKEND=faiss`, the Dockerfile build, and your other 15 checkpoints (only
`standard_attention_unet.pth` was available; the rest load through the same code path). Run
`check_weights.py` and `eval_folder.py` on your machine to cover the checkpoint part.

## 10. Layout

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
