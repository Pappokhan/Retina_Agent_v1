/* Retina Agent front end. No dependencies. All server text is rendered with textContent.
 *
 * Realtime design:
 *  - Uploaded images: the server sends the probability map once; the browser re-thresholds it on a
 *    canvas, so the threshold slider is instant. Stats are refreshed from the server (cached, fast).
 *  - Live camera: the <video> plays natively at full frame rate. Frames go to the server one at a
 *    time (latest frame wins, never a queue) and the returned transparent mask is drawn on a canvas
 *    over the video. With "Auto" resolution the input size adapts to the measured latency.
 */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const el = (tag, props = {}, ...kids) => {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(props)) {
      if (k === "class") n.className = v;
      else if (k === "text") n.textContent = v;
      else if (k.startsWith("data-") || k.startsWith("aria-")) n.setAttribute(k, v);
      else n[k] = v;
    }
    n.append(...kids);
    return n;
  };
  const b64 = (data, mime = "image/png") => `data:${mime};base64,${data}`;
  const jpg = (data) => b64(data, "image/jpeg");
  const SIZE = 512;
  const MASK_RGB = [0, 229, 208];
  const XAI_VIEWS = ["attention", "gradcam", "uncertainty", "mc_dropout"];

  const state = {
    models: null, dataset: "A", variant: "best", imageId: null, threshold: 0.5,
    view: "overlay", last: null, prob: null, heatImg: null, maskLayer: null,
    xai: new Map(), xaiImgs: new Map(), roi: null, live: null, session: makeSession(),
  };

  function makeSession() {
    const fresh = () => "s" + Math.random().toString(36).slice(2, 12) + Date.now().toString(36);
    try {
      let s = localStorage.getItem("retina-session");
      if (!s) { s = fresh(); localStorage.setItem("retina-session", s); }
      return s;
    } catch { return fresh(); }
  }

  // ---------- helpers ----------
  function setMsg(text, kind = "") {
    const m = $("msg");
    m.textContent = text;
    if (kind) m.dataset.kind = kind; else delete m.dataset.kind;
  }

  async function api(path, opts = {}) {
    const headers = { ...(opts.headers || {}) };
    const token = $("token").value;
    if (token) headers["X-App-Token"] = token;
    let body = opts.body;
    if (opts.json !== undefined) { headers["Content-Type"] = "application/json"; body = JSON.stringify(opts.json); }
    const res = await fetch(path, { method: opts.method || "GET", headers, body, signal: opts.signal });
    if (!res.ok) {
      let msg = `The server returned an error (${res.status}).`;
      try { const j = await res.json(); if (j.error) msg = j.error; } catch { /* keep default */ }
      throw new Error(msg);
    }
    return opts.raw ? res : res.json();
  }

  async function busy(btn, fn) {
    btn.disabled = true;
    btn.setAttribute("aria-busy", "true");
    try { return await fn(); }
    catch (e) { setMsg(e.message, "error"); }
    finally { btn.disabled = false; btn.removeAttribute("aria-busy"); }
  }

  const loadImg = (src) => new Promise((resolve, reject) => {
    const i = new Image();
    i.onload = () => resolve(i);
    i.onerror = () => reject(new Error("Could not decode an image from the server."));
    i.src = src;
  });

  const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };
  const needImage = () => {
    if (state.imageId) return true;
    setMsg("Upload an image first.", "error");
    return false;
  };
  const lang = () => $("lang").value;
  const basePayload = () => ({ image_id: state.imageId, dataset: state.dataset, variant: state.variant,
    threshold: state.threshold, session_id: state.session });
  const spinner = (on) => { $("spinner").hidden = !on; };

  // ---------- boot ----------
  async function boot() {
    try {
      const h = await api("/api/health");
      renderStatus(h);
      if (h.auth_required) {
        $("token-wrap").hidden = false;
        try { $("token").value = sessionStorage.getItem("retina-token") || ""; } catch { /* private mode */ }
      }
    } catch {
      $("st-device").textContent = "Server unreachable";
      $("st-device").dataset.ok = "false";
      setMsg("Cannot reach the server. Check that it is running and reload the page.", "error");
      return;
    }
    await loadModels();
  }

  function renderStatus(h) {
    const dev = $("st-device");
    dev.textContent = h.cuda ? "Running on GPU" : "Running on CPU";
    dev.dataset.ok = "true";
    const mod = $("st-models");
    mod.textContent = `${h.models_installed} of ${h.models_total} variants installed`;
    mod.dataset.ok = String(h.models_installed === h.models_total);
    const v = $("st-vlm");
    const name = (h.vlm_model || "").split(":").pop();
    const missing = h.missing_packages || [];
    v.textContent = missing.length ? "Explanations off: packages missing"
      : h.vlm_configured ? `Explanations: ${name}` : "Explanations off: set a model key in .env";
    v.dataset.ok = String(h.vlm_configured && !missing.length);
    if (missing.length) {
      setMsg(`The server's Python (${h.python}) is missing: ${missing.join(", ")}. Segmentation and XAI work; ` +
        "Report, Ask and Chat need them. Run the app with this project's .venv, or pip install -r requirements.txt into that Python.", "error");
    }
    $("disclaimer").textContent = "Research software. Not a medical device and not for diagnosis. " + h.vlm_privacy;
  }

  async function loadModels() {
    try {
      state.models = await api("/api/models");
    } catch (e) {
      setMsg(e.message, "error");
      return;
    }
    const m = state.models;
    state.dataset = m.dataset;
    $("lang").replaceChildren(...m.languages.map((l) => el("option", { value: l, text: l })));
    $("layer").replaceChildren(...m.gradcam_layers.map((l) => el("option", { value: l, text: layerLabel(l) })));
    $("layer").value = m.gradcam_layer;
    $("live-size").replaceChildren(el("option", { value: "auto", text: `Auto (starts at ${m.live_size} px)` }),
      ...m.live_sizes.map((s) => el("option", { value: String(s), text: `${s} px${s === 512 ? " (training size)" : ""}` })));
    renderVariants();
    if (!Object.values(m.available[m.dataset]).some(Boolean)) {
      setMsg("No checkpoints found. Copy the .pth files into the weights/A folder (see the README).", "error");
    }
  }

  const layerLabel = (l) => ({ down4: "Bottleneck (32 px)", up1: "Decoder 1 (64 px)", up2: "Decoder 2 (128 px)", up3: "Decoder 3 (256 px)" }[l] || l);

  function renderVariants() {
    const wrap = $("variant-list");
    const avail = state.models.available[state.dataset];
    if (!avail[state.variant]) state.variant = Object.keys(avail).find((k) => avail[k]) || "best";
    wrap.replaceChildren(...Object.entries(state.models.variants).map(([k, v]) => {
      const input = el("input", { type: "radio", name: "variant", value: k, checked: k === state.variant, disabled: !avail[k] });
      input.addEventListener("change", () => { state.variant = k; onVariantChange(); });
      const short = { best: "Best", standard: "Final", fedavg: "FedAvg", fedprox: "FedProx" }[k] || v.label;
      return el("label", { class: "seg-opt", title: avail[k] ? v.note : `${v.label}: not installed` },
        input, el("span", { text: short }));
    }));
    $("variant-note").textContent = state.models.variants[state.variant]?.note || "";
  }

  function onVariantChange() {
    renderVariants();
    $("compare").hidden = true;
    resetXai();
    if (state.imageId && !state.live) predict();
  }

  // ---------- upload ----------
  async function upload(file, label) {
    if (!file) return;
    setMsg("Uploading…");
    const fd = new FormData();
    fd.append("image", file, file.name || "image.png");
    try {
      const r = await api("/api/upload", { method: "POST", body: fd });
      state.imageId = r.image_id;
      $("drop-text").textContent = `${label || file.name} (${r.width} × ${r.height})`;
      $("compare").hidden = true;
      resetXai();
      resetAskLog();
      setMsg("");
      await predict();
    } catch (e) { setMsg(e.message, "error"); }
  }

  $("file").addEventListener("change", (e) => upload(e.target.files[0]));
  const drop = $("drop");
  for (const t of ["dragenter", "dragover"]) drop.addEventListener(t, (e) => { e.preventDefault(); drop.dataset.over = "true"; });
  for (const t of ["dragleave", "drop"]) drop.addEventListener(t, (e) => { e.preventDefault(); delete drop.dataset.over; });
  drop.addEventListener("drop", (e) => upload(e.dataTransfer.files[0]));

  // ---------- segmentation ----------
  async function predict() {
    if (!needImage()) return;
    setMsg("Segmenting…");
    spinner(true);
    try {
      const r = await api("/api/predict", { method: "POST", json: basePayload() });
      const [orig, heat, probImg] = await Promise.all([loadImg(b64(r.original)), loadImg(b64(r.heatmap)), loadImg(b64(r.prob))]);
      state.last = r;
      state.heatImg = heat;
      state.prob = readProb(probImg);
      $("img-base").src = orig.src;
      buildMask();
      showStats(r.stats);
      $("stage").hidden = false;
      $("empty").hidden = true;
      render();
      setMsg(`${state.models.variants[state.variant].label}: done${r.cached ? " (cached)" : ` in ${Math.round(r.stats.model_ms)} ms`}.`, "ok");
    } catch (e) { setMsg(e.message, "error"); }
    finally { spinner(false); }
  }

  function readProb(img) {
    const c = el("canvas", { width: SIZE, height: SIZE });
    const ctx = c.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(img, 0, 0, SIZE, SIZE);
    const px = ctx.getImageData(0, 0, SIZE, SIZE).data;
    const out = new Uint8ClampedArray(SIZE * SIZE);
    for (let i = 0, j = 0; i < out.length; i++, j += 4) out[i] = px[j];
    return out;
  }

  /** Re-threshold the probability map in the browser (instant). */
  function buildMask() {
    if (!state.prob) return;
    const c = state.maskLayer || (state.maskLayer = el("canvas", { width: SIZE, height: SIZE }));
    const ctx = c.getContext("2d");
    const img = ctx.createImageData(SIZE, SIZE);
    const d = img.data, p = state.prob, cut = Math.round(state.threshold * 255);
    for (let i = 0, j = 0; i < p.length; i++, j += 4) {
      if (p[i] >= cut) { d[j] = MASK_RGB[0]; d[j + 1] = MASK_RGB[1]; d[j + 2] = MASK_RGB[2]; d[j + 3] = 160; }
    }
    ctx.putImageData(img, 0, 0);
  }

  function showStats(s) {
    $("s-area").textContent = `${s.area_percent.toFixed(1)}%`;
    $("s-conf").textContent = s.mean_confidence.toFixed(2);
    $("s-unc").textContent = `${s.uncertain_percent.toFixed(1)}%`;
    $("s-ms").textContent = `${Math.round(s.model_ms)} ms`;
  }

  // ---------- threshold (instant mask, debounced numbers) ----------
  const refreshStats = debounce(async () => {
    if (!state.imageId || state.live) return;
    try {
      const r = await api("/api/predict", { method: "POST", json: { ...basePayload(), stats_only: true } });
      showStats(r.stats);
      if (XAI_VIEWS.includes(state.view) && state.view !== "mc_dropout") loadXai();
    } catch (e) { setMsg(e.message, "error"); }
  }, 250);

  $("thr").addEventListener("input", () => {
    state.threshold = Number($("thr").value);
    $("thr-out").textContent = state.threshold.toFixed(2);
    if (!state.live) { buildMask(); render(); refreshStats(); }
  });

  // ---------- viewer ----------
  function render() {
    const v = state.view;
    document.querySelectorAll(".view").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.view === v)));
    const isXai = XAI_VIEWS.includes(v);
    $("opacity-wrap").hidden = v === "orig" || v === "wipe";
    $("wipe-wrap").hidden = v !== "wipe";
    $("wipe-handle").hidden = v !== "wipe" || !!state.live;
    $("gate-wrap").hidden = v !== "attention";
    $("layer-wrap").hidden = $("radius-wrap").hidden = v !== "gradcam";
    $("btn-clear-roi").hidden = v !== "gradcam" || !state.roi;
    $("samples-wrap").hidden = $("btn-mc").hidden = v !== "mc_dropout";
    $("stage-tip").hidden = v !== "gradcam" || !!state.live || !state.last;
    $("xai-panel").hidden = !isXai || !!state.live || !state.last;
    renderRoi();
    if (state.live || !state.last) return;

    const canvas = $("layer-canvas");
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, SIZE, SIZE);
    canvas.style.clipPath = "";
    canvas.hidden = v === "orig";
    const strength = Number($("opacity").value) / 100;
    canvas.style.opacity = String(v === "wipe" ? 1 : strength);
    if (v === "overlay" || v === "wipe") {
      if (state.maskLayer) ctx.drawImage(state.maskLayer, 0, 0);
      if (v === "wipe") {
        const w = Number($("wipe").value);
        canvas.style.clipPath = `inset(0 ${100 - w}% 0 0)`;
        $("wipe-handle").style.left = `${w}%`;
      }
    } else if (v === "heat") {
      if (state.heatImg) ctx.drawImage(state.heatImg, 0, 0, SIZE, SIZE);
    } else if (isXai) {
      const img = state.xaiImgs.get(xaiImageKey());
      if (img) ctx.drawImage(img, 0, 0, SIZE, SIZE);
      renderXaiPanel();
    }
  }

  document.querySelectorAll(".view").forEach((b) => b.addEventListener("click", () => {
    state.view = b.dataset.view;
    render();
    if (XAI_VIEWS.includes(state.view) && state.view !== "mc_dropout" && state.last && !state.live) loadXai();
  }));
  $("opacity").addEventListener("input", render);
  $("wipe").addEventListener("input", render);

  const stage = $("stage");
  const toImageXY = (ev) => {
    const r = stage.getBoundingClientRect();
    return { x: ((ev.clientX - r.left) / r.width) * SIZE, y: ((ev.clientY - r.top) / r.height) * SIZE, fx: (ev.clientX - r.left) / r.width };
  };
  stage.addEventListener("pointerdown", (ev) => {
    if (state.live) return;
    if (state.view === "wipe") {
      stage.setPointerCapture(ev.pointerId);
      $("wipe").value = Math.min(100, Math.max(0, toImageXY(ev).fx * 100));
      render();
    } else if (state.view === "gradcam" && state.last) {
      const { x, y } = toImageXY(ev);
      state.roi = { x: Math.round(Math.min(SIZE - 1, Math.max(0, x))), y: Math.round(Math.min(SIZE - 1, Math.max(0, y))), r: Number($("radius").value) };
      render();
      loadXai();
    }
  });
  stage.addEventListener("pointermove", (ev) => {
    if (state.view === "wipe" && (ev.buttons & 1) && !state.live) {
      $("wipe").value = Math.min(100, Math.max(0, toImageXY(ev).fx * 100));
      render();
    }
  });

  $("btn-run").addEventListener("click", (e) => busy(e.currentTarget, predict));

  // ---------- XAI ----------
  const XAI_INFO = {
    attention: ["Attention gates", "The U-Net's own attention coefficients. Bright areas are where the decoder let skip-connection features through. A focus ratio above 1 means a gate concentrates on vessels more than their area alone explains."],
    gradcam: ["Grad-CAM (HiResCAM)", "Which pixels of the chosen layer most increased the vessel score (gradient × activation). Click the image to explain just one region; the layer menu moves from coarse (bottleneck) to fine."],
    uncertainty: ["Uncertainty (entropy)", "Bright where the predicted probability is close to 50/50. Thin vessels and vessel edges are usually the least certain."],
    mc_dropout: ["Monte-Carlo dropout", "Runs the model several times with dropout switched on. Bright where the prediction changes between runs (model uncertainty). Slower: one full pass per run."],
  };

  function xaiRequest() {
    const m = state.view;
    const req = { ...basePayload(), method: m };
    if (m === "gradcam") { req.layer = $("layer").value; if (state.roi) req.roi = state.roi; }
    if (m === "mc_dropout") req.samples = Number($("samples").value);
    return req;
  }
  const xaiKey = () => JSON.stringify(xaiRequest());
  const xaiImageKey = () => xaiKey() + (state.view === "attention" ? `|${$("gate").value}` : "");

  let xaiCtl = null;
  async function loadXai() {
    if (!state.imageId || !XAI_VIEWS.includes(state.view)) return;
    const key = xaiKey();
    if (state.xai.has(key)) { await ensureXaiImage(key); render(); return; }
    if (xaiCtl) xaiCtl.abort();
    const ctl = xaiCtl = new AbortController();
    spinner(true);
    $("xai-text").replaceChildren();
    setMsg(state.view === "mc_dropout" ? `Running ${$("samples").value} dropout passes…` : "Computing explanation…");
    try {
      const r = await api("/api/xai", { method: "POST", json: xaiRequest(), signal: ctl.signal });
      state.xai.set(key, r);
      await ensureXaiImage(key);
      if (ctl === xaiCtl) { render(); setMsg(`${XAI_INFO[r.method][0]} ready.`, "ok"); }
    } catch (e) { if (e.name !== "AbortError") setMsg(e.message, "error"); }
    finally { if (ctl === xaiCtl) { spinner(false); xaiCtl = null; } }
  }

  async function ensureXaiImage(key) {
    const r = state.xai.get(key);
    if (!r) return;
    const gate = state.view === "attention" ? $("gate").value : null;
    const ik = key + (gate ? `|${gate}` : "");
    if (state.xaiImgs.has(ik)) return;
    const data = gate && gate !== "combined" && r.layers?.[gate] ? r.layers[gate] : r.heat;
    state.xaiImgs.set(ik, await loadImg(jpg(data)));
  }

  function resetXai() {
    state.xai.clear();
    state.xaiImgs.clear();
    state.roi = null;
    $("xai-text").replaceChildren();
  }

  $("gate").addEventListener("change", async () => { await ensureXaiImage(xaiKey()); render(); });
  $("layer").addEventListener("change", loadXai);
  $("radius").addEventListener("input", () => { $("radius-out").textContent = $("radius").value; if (state.roi) { state.roi.r = Number($("radius").value); renderRoi(); } });
  $("radius").addEventListener("change", () => { if (state.roi) loadXai(); });
  $("btn-clear-roi").addEventListener("click", () => { state.roi = null; render(); loadXai(); });
  $("btn-mc").addEventListener("click", (e) => busy(e.currentTarget, loadXai));
  $("samples").addEventListener("change", () => render());

  function renderRoi() {
    const box = $("roi");
    const show = state.view === "gradcam" && state.roi && !state.live && state.last;
    box.hidden = !show;
    if (!show) return;
    const { x, y, r } = state.roi;
    Object.assign(box.style, { left: `${((x - r) / SIZE) * 100}%`, top: `${((y - r) / SIZE) * 100}%`, width: `${(2 * r / SIZE) * 100}%`, height: `${(2 * r / SIZE) * 100}%` });
  }

  const LABELS = {
    mean_entropy_bits: "Mean entropy (bits)", high_entropy_percent: "Pixels with entropy above 0.5",
    mean_entropy_on_vessels: "Mean entropy on vessels", layer: "Layer", target: "Explained target",
    target_pixels: "Target pixels", ms: "Compute time", energy_on_vessels: "Evidence on vessels",
    hot_area_percent: "Hot area (CAM ≥ 0.5)", energy_inside_region: "Evidence inside region",
    region_area_percent: "Region size", peak_center_xy: "Peak location (x, y)", region: "Region",
    samples: "Passes", max_std: "Largest std", mean_std: "Mean std", mean_std_on_vessels: "Mean std on vessels",
    high_variance_percent: "Pixels with std above 0.1", dice_mc_vs_single_pass: "Dice, MC mean vs single pass",
  };
  function fmt(k, v) {
    if (v === null || v === undefined) return "–";
    if (k === "ms") return `${Math.round(v)} ms`;
    if (k === "layer") return layerLabel(v);
    if (k === "region") return `x ${v.x}, y ${v.y}, r ${v.r}`;
    if (Array.isArray(v)) return v.join(", ");
    if (k.startsWith("energy")) return `${(v * 100).toFixed(1)}%`;
    if (k.endsWith("percent")) return `${Number(v).toFixed(1)}%`;
    return typeof v === "number" ? String(v) : String(v);
  }

  function renderXaiPanel() {
    const m = state.view;
    $("xai-title").textContent = XAI_INFO[m][0];
    $("xai-help").textContent = XAI_INFO[m][1];
    const r = state.xai.get(xaiKey());
    const box = $("xai-summary");
    $("btn-xai-explain").disabled = !r;
    if (!r) {
      box.replaceChildren(el("p", { class: "placeholder", text: m === "mc_dropout" ? "Choose the number of passes and select Run MC dropout." : "Computing…" }));
      return;
    }
    const s = r.summary;
    if (m === "attention") {
      const rows = Object.entries(s.gates).map(([g, v]) => el("tr", {}, el("th", { scope: "row", text: g }),
        el("td", { text: `${v.resolution} px` }), el("td", { text: v.mean.toFixed(3) }),
        el("td", { text: `${(v.on_vessel_share * 100).toFixed(1)}%` }),
        el("td", { text: v.vessel_focus_ratio == null ? "–" : `${v.vessel_focus_ratio.toFixed(2)}×` })));
      box.replaceChildren(el("table", { class: "agreement" },
        el("thead", {}, el("tr", {}, ...["Gate", "Resolution", "Mean", "Attention on vessels", "Focus ratio"].map((t) => el("th", { text: t })))),
        el("tbody", {}, ...rows)));
    } else {
      box.replaceChildren(el("dl", { class: "kv" }, ...Object.entries(s).filter(([k]) => LABELS[k])
        .map(([k, v]) => el("div", {}, el("dt", { text: LABELS[k] }), el("dd", { text: fmt(k, v) })))));
    }
  }

  $("btn-xai-explain").addEventListener("click", (e) => busy(e.currentTarget, async () => {
    if (!needImage()) return;
    const out = $("xai-text");
    out.replaceChildren(el("p", { class: "placeholder", text: "The vision model is reading the map…" }));
    const r = await api("/api/vlm/xai", { method: "POST", json: { ...xaiRequest(), lang: lang() } });
    out.replaceChildren(el("p", { text: r.text }),
      el("p", { class: "fineprint", text: "An explanation of the model's behaviour, not a measure of accuracy." }),
      renderSources(r.sources));
  }));

  // ---------- compare ----------
  $("btn-compare").addEventListener("click", (e) => busy(e.currentTarget, async () => {
    if (!needImage()) return;
    setMsg("Running every installed variant…");
    const r = await api("/api/compare", { method: "POST", json: { image_id: state.imageId, dataset: state.dataset } });
    $("compare-grid").replaceChildren(...Object.entries(r.variants).map(([, v]) => el("figure", {},
      el("img", { src: b64(v.overlay), alt: `${v.label} overlay` }),
      el("figcaption", {}, el("strong", { text: v.label }), el("span", { text: `${v.stats.area_percent.toFixed(1)}% area` })))));
    $("compare-note").textContent = r.missing.length
      ? `Not installed: ${r.missing.map((k) => state.models.variants[k].label).join(", ")}.` : "";
    renderAgreement(r);
    $("compare").hidden = false;
    $("compare").scrollIntoView({ behavior: "smooth", block: "nearest" });
    setMsg("Comparison ready.", "ok");
  }));

  function renderAgreement(r) {
    const names = Object.keys(r.variants);
    const dice = (a, b) => (a === b ? 1 : (r.agreement.dice[`${a}_vs_${b}`] ?? r.agreement.dice[`${b}_vs_${a}`]));
    const t = $("agreement");
    t.tHead.replaceChildren(el("tr", {}, el("th", { text: "" }), ...names.map((n) => el("th", { text: r.variants[n].label }))));
    t.tBodies[0].replaceChildren(...names.map((a) => el("tr", {}, el("th", { scope: "row", text: r.variants[a].label }),
      ...names.map((b) => {
        const v = dice(a, b);
        const td = el("td", { text: v.toFixed(2), "data-dice": "1" });
        td.style.setProperty("--v", String(v));
        return td;
      }))));
  }

  // ---------- assistant tabs ----------
  const tabs = ["report", "ask", "chat", "notes"];

  // ---------- RAG sources ----------
  /** Collapsible list of the project notes an answer used. Accepts {id, source, section, snippet} or labels. */
  function renderSources(list, open = false) {
    if (!list || !list.length) return el("p", { class: "fineprint", text: "No project notes matched this question." });
    const items = list.map((s, i) => typeof s === "string"
      ? el("li", {}, el("b", { text: s }))
      : el("li", {}, el("b", { text: `[${s.id ?? i + 1}] ${s.source}${s.section ? " › " + s.section : ""}` }),
          el("span", { class: "snippet", text: s.snippet || "" })));
    return el("details", { class: "sources", open }, el("summary", { text: `Sources from project notes (${list.length})` }), el("ol", {}, ...items));
  }

  $("notes-form").addEventListener("submit", (e) => {
    e.preventDefault();
    busy($("btn-notes"), async () => {
      const q = $("notes-q").value.trim();
      if (!q) { setMsg("Type something to search for.", "error"); return; }
      const r = await api(`/api/knowledge/search?q=${encodeURIComponent(q)}`);
      $("notes-out").replaceChildren(r.results.length ? renderSources(r.results, true)
        : el("p", { class: "placeholder", text: "Nothing in the notes matches that." }));
    });
  });
  function activateTab(name) {
    for (const t of tabs) {
      const sel = t === name;
      const tab = $(`tab-${t}`);
      tab.setAttribute("aria-selected", String(sel));
      tab.tabIndex = sel ? 0 : -1;
      $(`panel-${t}`).hidden = !sel;
    }
  }
  for (const t of tabs) {
    $(`tab-${t}`).addEventListener("click", () => activateTab(t));
    $(`tab-${t}`).addEventListener("keydown", (e) => {
      if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
      const next = tabs[(tabs.indexOf(t) + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length];
      activateTab(next);
      $(`tab-${next}`).focus();
    });
  }

  // ---------- report ----------
  $("btn-report").addEventListener("click", (e) => busy(e.currentTarget, async () => {
    if (!needImage()) return;
    setMsg("The vision model is writing the report…");
    $("report-out").replaceChildren(el("p", { class: "placeholder", text: "Reading the image, mask, uncertainty and attention maps…" }));
    const t0 = performance.now();
    const { report, context, sources } = await api("/api/vlm/report", { method: "POST", json: { ...basePayload(), lang: lang() } });
    renderReport(report, context);
    $("report-out").append(renderSources(sources));
    setMsg(`Report ready in ${((performance.now() - t0) / 1000).toFixed(1)} s.`, "ok");
  }));

  function renderReport(rep, ctx) {
    const out = $("report-out");
    const v = rep.qc.looks_plausible;
    const [kind, label] = v === true ? ["ok", "Mask looks plausible"] : v === false ? ["check", "Mask needs a look"] : ["none", "Not assessed"];
    const q = rep.image_quality?.rating;
    const list = (items) => el("ul", {}, ...items.map((i) => el("li", { text: i })));
    const section = (title, ...kids) => (kids.length ? [el("h3", { text: title }), ...kids] : []);
    const iq = ctx?.image_quality;
    out.replaceChildren(
      el("p", { class: "chips-row" },
        el("span", { class: "verdict", "data-v": kind, text: label }),
        ...(q ? [el("span", { class: "verdict", "data-v": q === "good" ? "ok" : q === "poor" ? "check" : "none", text: `Image quality: ${q}` })] : [])),
      el("p", { text: rep.summary || rep.description || "No summary was returned." }),
      ...section("Visual check", ...(rep.qc.issues.length ? [list(rep.qc.issues)] : [el("p", { class: "fineprint", text: "No issues listed." })])),
      ...section("Regions", ...(rep.regions?.length ? [el("ul", {}, ...rep.regions.map((r) => el("li", {}, el("b", { text: `${r.region.replace("_", " ")}: ` }), r.note)))] : [])),
      ...section("What the model focused on", ...(rep.model_focus ? [el("p", { text: rep.model_focus })] : [])),
      ...section("Uncertainty", ...(rep.uncertainty ? [el("p", { text: rep.uncertainty })] : [])),
      ...section("Image quality notes", ...(rep.image_quality?.notes?.length ? [list(rep.image_quality.notes)] : [])),
      ...section("Caveats", ...(rep.caveats.length ? [list(rep.caveats)] : [])),
      ...section("Try next", ...(rep.next_steps?.length ? [list(rep.next_steps)] : [])),
      ...(iq ? [el("p", { class: "fineprint", text: `Measured: brightness ${iq.brightness}, contrast ${iq.contrast}, sharpness ${iq.sharpness}, field of view ${iq.field_of_view_percent}%.` })] : []),
      el("p", { class: "fineprint", text: "A visual sanity check only. It does not measure accuracy; use Dice and IoU on the test set for that." }),
    );
  }

  $("btn-diff").addEventListener("click", (e) => busy(e.currentTarget, async () => {
    if (!needImage()) return;
    setMsg("Comparing every variant…");
    const r = await api("/api/vlm/compare", { method: "POST", json: { image_id: state.imageId, dataset: state.dataset, lang: lang() } });
    $("report-out").replaceChildren(el("h3", { text: "How the variants differ" }), el("p", { text: r.text }),
      el("p", { class: "fineprint", text: "Describes agreement between masks; it does not say which one is correct." }),
      renderSources(r.sources));
    setMsg("");
  }));

  // ---------- ask (streaming, with follow-ups) ----------
  const askLog = $("ask-log");
  function resetAskLog() {
    askLog.replaceChildren(el("p", { class: "placeholder", text: "Ask about the current result. Follow-up questions remember the last few answers." }));
  }
  function addBubble(logEl, role, text) {
    if (logEl.querySelector(".placeholder")) logEl.replaceChildren();
    const p = el("p", { class: role === "user" ? "msg-user" : "msg-bot", text });
    logEl.append(p);
    logEl.scrollTop = logEl.scrollHeight;
    return p;
  }

  async function ask(q) {
    if (!needImage()) return;
    if (!q) { setMsg("Type a question first.", "error"); return; }
    $("ask-q").value = "";
    addBubble(askLog, "user", q);
    const p = addBubble(askLog, "bot", "…");
    try {
      const res = await api("/api/vlm/ask", { method: "POST", raw: true, json: { ...basePayload(), question: q, lang: lang() } });
      let sources = [];
      try { sources = JSON.parse(decodeURIComponent(res.headers.get("X-Sources") || "[]")); } catch { /* none */ }
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      p.textContent = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        p.textContent += dec.decode(value, { stream: true });
        askLog.scrollTop = askLog.scrollHeight;
      }
      p.append(renderSources(sources));
      askLog.scrollTop = askLog.scrollHeight;
    } catch (err) {
      p.textContent = err.message;
      p.style.color = "var(--bad)";
    }
  }
  $("ask-form").addEventListener("submit", (e) => { e.preventDefault(); busy($("btn-ask"), () => ask($("ask-q").value.trim())); });
  $("ask-chips").addEventListener("click", (e) => {
    const chip = e.target.closest(".chip");
    if (chip) busy($("btn-ask"), () => ask(chip.textContent));
  });
  $("btn-ask-reset").addEventListener("click", async () => {
    resetAskLog();
    if (state.imageId) { try { await api("/api/vlm/ask/reset", { method: "POST", json: basePayload() }); } catch { /* ignore */ } }
  });

  // ---------- chat agent ----------
  const log = $("chat-log");
  $("chat-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const q = $("chat-q").value.trim();
    if (!q) return;
    $("chat-q").value = "";
    addBubble(log, "user", q);
    const pending = addBubble(log, "bot", "Thinking…");
    busy($("btn-chat"), async () => {
      try {
        const r = await api("/api/chat", { method: "POST", json: {
          session_id: state.session, message: q,
          ...(state.imageId ? { image_id: state.imageId } : {}), dataset: state.dataset, variant: state.variant } });
        pending.textContent = r.reply || "The agent returned no text.";
        if (r.tools_used?.length) pending.append(el("span", { class: "tools-used", text: `Used: ${r.tools_used.join(", ")}` }));
        if (r.sources?.length) pending.append(renderSources(r.sources));
      } catch (err) {
        pending.textContent = err.message;
        pending.style.color = "var(--bad)";
      }
      log.scrollTop = log.scrollHeight;
    });
  });

  // ---------- live camera (realtime) ----------
  const cap = el("canvas");
  const capCtx = cap.getContext("2d");

  function liveSizes() { return state.models.live_sizes; }

  async function startLive() {
    if (!navigator.mediaDevices?.getUserMedia) {
      setMsg("The camera needs a secure page. Open the app on localhost or over HTTPS.", "error");
      return;
    }
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment", width: { ideal: 1280 }, frameRate: { ideal: 30 } }, audio: false });
    } catch {
      setMsg("Camera access was blocked. Allow the camera for this site and try again.", "error");
      return;
    }
    const video = $("cam");
    video.srcObject = stream;
    await video.play();
    const choice = $("live-size").value;
    state.live = { stream, auto: choice === "auto", size: choice === "auto" ? state.models.live_size : Number(choice),
      lat: [], done: [], ema: null, cool: 0 };
    document.body.dataset.live = "true";
    $("btn-live-text").textContent = "Stop live camera";
    for (const id of ["btn-run", "btn-compare", "btn-report", "btn-diff", "btn-ask"]) $(id).disabled = true;
    $("file").disabled = true;
    $("btn-snap").disabled = false;
    $("stage").hidden = false;
    $("empty").hidden = true;
    video.hidden = false;
    $("img-base").hidden = true;
    $("layer-canvas").hidden = true;
    $("live-canvas").hidden = false;
    $("live-xai-canvas").hidden = false;
    $("hud").hidden = false;
    render();
    setMsg("Live: the video runs at full speed; the mask updates as fast as the model allows.", "ok");
    liveLoop(state.live);
  }

  function stopLive(message = "Live camera stopped.") {
    if (!state.live) return;
    state.live.stream.getTracks().forEach((t) => t.stop());
    $("cam").srcObject = null;
    state.live = null;
    delete document.body.dataset.live;
    $("btn-live-text").textContent = "Start live camera";
    for (const id of ["btn-run", "btn-compare", "btn-report", "btn-diff", "btn-ask"]) $(id).disabled = false;
    $("file").disabled = false;
    $("btn-snap").disabled = true;
    $("cam").hidden = true;
    $("live-canvas").hidden = true;
    $("live-xai-canvas").hidden = true;
    $("hud").hidden = true;
    $("img-base").hidden = false;
    for (const id of ["live-canvas", "live-xai-canvas"]) $(id).getContext("2d").clearRect(0, 0, SIZE, SIZE);
    if (state.last) render(); else { $("stage").hidden = true; $("empty").hidden = false; }
    setMsg(message);
  }

  function grabFrame(size) {
    const video = $("cam");
    const vw = video.videoWidth, vh = video.videoHeight;
    if (!vw || !vh) return null;
    const s = Math.min(vw, vh);
    cap.width = cap.height = size;
    capCtx.drawImage(video, (vw - s) / 2, (vh - s) / 2, s, s, 0, 0, size, size);
    return cap;
  }

  function adapt(session, ms) {
    session.ema = session.ema == null ? ms : 0.7 * session.ema + 0.3 * ms;
    if (!session.auto || --session.cool > 0) return;
    const sizes = liveSizes();
    const i = sizes.indexOf(session.size);
    if (session.ema > 650 && i > 0) { session.size = sizes[i - 1]; session.cool = 4; session.ema = null; }
    else if (session.ema < 180 && i < sizes.length - 1) { session.size = sizes[i + 1]; session.cool = 6; session.ema = null; }
  }

  async function liveLoop(session) {
    const maskCtx = $("live-canvas").getContext("2d");
    const xaiCtx = $("live-xai-canvas").getContext("2d");
    while (state.live === session) {
      const frame = grabFrame(session.size);
      if (!frame) { await new Promise((r) => setTimeout(r, 30)); continue; }
      const blob = await new Promise((r) => frame.toBlob(r, "image/jpeg", 0.82));
      const fd = new FormData();
      fd.append("image", blob, "frame.jpg");
      fd.append("dataset", state.dataset);
      fd.append("variant", state.variant);
      fd.append("size", String(session.size));
      fd.append("threshold", String(state.threshold));
      fd.append("xai", $("live-xai").value);
      const t0 = performance.now();
      try {
        const r = await api("/api/live", { method: "POST", body: fd });
        if (state.live !== session) return;
        const [mask, heat] = await Promise.all([loadImg(b64(r.mask)), r.xai ? loadImg(jpg(r.xai)) : null]);
        maskCtx.clearRect(0, 0, SIZE, SIZE);
        maskCtx.drawImage(mask, 0, 0, SIZE, SIZE);
        xaiCtx.clearRect(0, 0, SIZE, SIZE);
        if (heat) xaiCtx.drawImage(heat, 0, 0, SIZE, SIZE);
        $("live-canvas").style.opacity = String(Number($("opacity").value) / 100 * 0.85);
        showStats(r.stats);
        const ms = performance.now() - t0;
        const now = performance.now();
        session.done.push(now);
        while (session.done.length && now - session.done[0] > 3000) session.done.shift();
        const fps = session.done.length > 1 ? (session.done.length - 1) / ((now - session.done[0]) / 1000) : 1000 / ms;
        $("h-fps").textContent = fps.toFixed(1);
        $("h-lat").textContent = String(Math.round(ms));
        $("h-size").textContent = String(r.size);
        adapt(session, ms);
      } catch (e) {
        stopLive(e.message);
        setMsg(e.message, "error");
        return;
      }
    }
  }

  $("btn-live").addEventListener("click", () => { if (state.live) stopLive(); else startLive(); });
  $("btn-snap").addEventListener("click", () => {
    if (!state.live) return;
    const frame = grabFrame(SIZE);
    if (!frame) return;
    frame.toBlob((blob) => {
      stopLive("Frame captured. Running the full analysis…");
      upload(new File([blob], "camera-frame.png", { type: "image/png" }), "Camera frame");
    }, "image/png");
  });
  $("live-size").addEventListener("change", () => {
    if (!state.live) return;
    const v = $("live-size").value;
    state.live.auto = v === "auto";
    if (v !== "auto") state.live.size = Number(v);
  });
  document.addEventListener("visibilitychange", () => { if (document.hidden) stopLive("Live camera paused because the tab was hidden."); });

  // ---------- access token ----------
  $("token").addEventListener("change", async () => {
    try { sessionStorage.setItem("retina-token", $("token").value); } catch { /* private mode */ }
    setMsg("");
    await loadModels();
  });

  boot();
})();
