import base64
import io
import json
from urllib.parse import unquote

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from PIL import Image

import agent
import config
import vlm
from conftest import png_bytes


def decode(b64):
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def post(client, path, **body):
    return client.post(path, json=body)


def test_index_and_static(client):
    r = client.get("/")
    assert r.status_code == 200 and b"Retina Agent" in r.data
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


def test_health_and_models(client):
    h = client.get("/api/health").get_json()
    assert h["ok"] and h["models_installed"] == 3 and h["models_total"] == 4 and h["dataset"] == "A"
    m = client.get("/api/models").get_json()
    assert set(m["datasets"]) == {"A"} and m["dataset"] == "A"          # only the Dataset A model is served
    assert list(m["variants"]) == ["best", "standard", "fedavg", "fedprox"]
    assert m["available"]["A"]["fedavg"] is True and m["available"]["A"]["fedprox"] is False
    assert 512 in m["live_sizes"] and m["live_size"] in m["live_sizes"]
    assert m["xai_methods"] == ["attention", "gradcam", "uncertainty", "mc_dropout"]


def test_upload_rejects_bad_input(client):
    r = client.post("/api/upload", data={}, content_type="multipart/form-data")
    assert r.status_code == 400 and "error" in r.get_json()
    r = client.post("/api/upload", data={"image": (io.BytesIO(b"not an image"), "x.png")},
                    content_type="multipart/form-data")
    assert r.status_code == 400 and r.get_json()["code"] == "bad_image"


def test_upload_too_large(client):
    client.application.config["MAX_CONTENT_LENGTH"] = 100
    r = client.post("/api/upload", data={"image": (io.BytesIO(png_bytes()), "x.png")},
                    content_type="multipart/form-data")
    assert r.status_code == 413 and r.get_json()["code"] == "too_large"


def test_predict_flow(client, uploaded):
    r = post(client, "/api/predict", image_id=uploaded, dataset="A", variant="best")
    assert r.status_code == 200, r.get_json()
    d = r.get_json()
    assert set(d["stats"]) == {"area_percent", "mean_confidence", "uncertain_percent", "model_ms"}
    for key in ("original", "overlay", "heatmap", "prob"):
        assert decode(d[key]).size == (512, 512)
    assert decode(d["prob"]).mode == "L" and d["cached"] is False

    # same image again: served from the prediction cache; a new threshold changes only the numbers
    again = post(client, "/api/predict", image_id=uploaded, variant="best", threshold=0.9, stats_only=True).get_json()
    assert again["cached"] is True and "original" not in again
    assert again["stats"]["area_percent"] <= d["stats"]["area_percent"]


@pytest.mark.parametrize("payload, status, code", [
    (dict(dataset="B", variant="best"), 400, "bad_request"),         # datasets B-D were removed
    (dict(dataset="A", variant="fedprox"), 409, "model_missing"),
    (dict(dataset="Z", variant="best"), 400, "bad_request"),
    (dict(dataset="A", variant="nope"), 400, "bad_request"),
    (dict(dataset="A", variant="best", threshold=2), 400, "bad_request"),
    (dict(dataset="A", variant="best", threshold="x"), 400, "bad_request"),
])
def test_predict_errors(client, uploaded, payload, status, code):
    r = post(client, "/api/predict", image_id=uploaded, **payload)
    assert r.status_code == status and r.get_json()["code"] == code


def test_predict_bad_ids(client):
    assert post(client, "/api/predict", image_id="../../etc/passwd", dataset="A").status_code == 400
    assert post(client, "/api/predict", image_id="", dataset="A").status_code == 400
    assert post(client, "/api/predict", image_id="deadbeef", dataset="A").status_code == 404


def test_compare(client, uploaded):
    d = post(client, "/api/compare", image_id=uploaded, dataset="A").get_json()
    assert set(d["variants"]) == {"best", "standard", "fedavg"} and d["missing"] == ["fedprox"]
    assert len(d["agreement"]["dice"]) == 3


def live(client, **form):
    data = {"image": (io.BytesIO(png_bytes((200, 200), fmt="JPEG")), "f.jpg"), "variant": "best", **form}
    return client.post("/api/live", data=data, content_type="multipart/form-data")


def test_live_endpoint(client):
    r = live(client, size="256")
    assert r.status_code == 200, r.get_json()
    d = r.get_json()
    mask = decode(d["mask"])
    assert d["size"] == 256 and mask.size == (256, 256) and mask.mode == "RGBA"   # transparent layer
    assert "xai" not in d and {"model_ms", "server_ms"} <= set(d["timing"])
    d = live(client, size="384", xai="uncertainty", threshold="0.3").get_json()
    assert decode(d["xai"]).size == (384, 384)
    assert decode(live(client, size="256", xai="attention").get_json()["xai"]).size == (256, 256)
    assert live(client, size="300").status_code == 400                     # not a multiple of 16 we allow
    assert live(client, xai="nope").status_code == 400
    assert client.post("/api/live", data={}, content_type="multipart/form-data").status_code == 400


@pytest.mark.parametrize("method", ["attention", "uncertainty", "gradcam", "mc_dropout"])
def test_xai_endpoint(client, uploaded, method):
    body = dict(image_id=uploaded, variant="best", method=method)
    if method == "mc_dropout":
        body["samples"] = 2
    r = post(client, "/api/xai", **body)
    assert r.status_code == 200, r.get_json()
    d = r.get_json()
    assert d["method"] == method and decode(d["heat"]).size == (512, 512) and decode(d["blend"]).size == (512, 512)
    if method == "attention":
        assert set(d["layers"]) == {"up1", "up2", "up3", "up4"}
        assert set(d["summary"]["gates"]["up4"]) >= {"on_vessel_share", "vessel_focus_ratio"}
    if method == "mc_dropout":
        assert d["summary"]["samples"] == 2 and "dice_mc_vs_single_pass" in d["summary"]


def test_xai_gradcam_region_and_errors(client, uploaded):
    d = post(client, "/api/xai", image_id=uploaded, method="gradcam", layer="up3",
             roi={"x": 256, "y": 256, "r": 40}).get_json()
    assert d["summary"]["layer"] == "up3" and d["summary"]["region"] == {"x": 256, "y": 256, "r": 40}
    assert "energy_inside_region" in d["summary"]
    assert post(client, "/api/xai", image_id=uploaded, method="nope").status_code == 400
    assert post(client, "/api/xai", image_id=uploaded, method="gradcam", layer="bad").status_code == 400
    assert post(client, "/api/xai", image_id=uploaded, method="gradcam", roi={"x": 900, "y": 1}).status_code == 400


def test_vlm_routes(client, uploaded, fake_vlm):
    rep = post(client, "/api/vlm/report", image_id=uploaded, dataset="A", variant="best", lang="Bangla").get_json()
    assert rep["report"]["qc"]["looks_plausible"] is True and "area_percent" in rep["stats"]
    system, content = fake_vlm.last
    assert "diagnose" in system
    images = [b for b in content if b["type"] == "image_url"]
    assert len(images) == 4 and "Bangla" in content[-1]["text"]      # original, overlay, uncertainty, attention
    assert "image_quality" in content[-1]["text"] and "attention_gates" in content[-1]["text"]
    assert {"image_quality", "regions", "attention_gates"} <= set(rep["context"])
    # RAG: the prompt carries numbered project notes and the response lists them as sources
    assert "Project notes" in content[-1]["text"] and "[1] (" in content[-1]["text"]
    assert rep["sources"] and {"id", "source", "section", "snippet"} <= set(rep["sources"][0])
    calls = len(fake_vlm.calls)
    post(client, "/api/vlm/report", image_id=uploaded, dataset="A", variant="best", lang="Bangla")
    assert len(fake_vlm.calls) == calls                                   # same request: answered from cache

    r = post(client, "/api/vlm/ask", image_id=uploaded, variant="best", question="Where is it?", session_id="s1")
    assert r.status_code == 200 and r.get_data(as_text=True) == "The mask covers the vessels."
    r = post(client, "/api/vlm/ask", image_id=uploaded, variant="fedavg", question="Is FedAvg real federated averaging?")
    r.get_data()
    sources = json.loads(unquote(r.headers["X-Sources"]))
    assert sources[0]["source"] == "federated_setup.md"
    assert "federated_setup.md" in fake_vlm.calls[-1][-1].content[-1]["text"]
    post(client, "/api/vlm/ask", image_id=uploaded, variant="best", question="And the edge?", session_id="s1").get_data()
    msgs = fake_vlm.calls[-1]
    assert len(msgs) == 4 and msgs[1].content == "Where is it?"            # follow-up sees the previous turn
    assert post(client, "/api/vlm/ask/reset", image_id=uploaded, session_id="s1").status_code == 200
    post(client, "/api/vlm/ask", image_id=uploaded, variant="best", question="Fresh?", session_id="s1").get_data()
    assert len(fake_vlm.calls[-1]) == 2
    assert post(client, "/api/vlm/ask", image_id=uploaded, dataset="A", question="  ").status_code == 400

    fake_vlm.reply = "The bright areas follow the main vessels."
    x = post(client, "/api/vlm/xai", image_id=uploaded, method="uncertainty").get_json()
    assert x["text"] == "The bright areas follow the main vessels." and "mean_entropy_bits" in x["summary"]
    assert x["sources"][0]["source"] == "xai.md"

    cmp = post(client, "/api/vlm/compare", image_id=uploaded, dataset="A").get_json()
    assert cmp["text"] and set(cmp["stats"]["area_percent"]) == {"best", "standard", "fedavg"}
    assert any(s["source"] == "federated_setup.md" for s in cmp["sources"])


def test_knowledge_search(client):
    d = client.get("/api/knowledge/search?q=focus ratio").get_json()
    assert d["results"][0]["source"] == "xai.md" and d["results"][0]["section"] == "Attention gate map"
    assert client.get("/api/knowledge/search?q=").status_code == 400
    assert client.get("/api/knowledge/search?q=" + "x" * 301).status_code == 400


def test_vlm_without_key_is_a_clear_503(client, uploaded, monkeypatch):
    monkeypatch.setattr(vlm, "_model", None)
    monkeypatch.setattr(config, "VLM_MODEL", "anthropic:claude-sonnet-5-5")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = post(client, "/api/vlm/report", image_id=uploaded, dataset="A", variant="best")
    assert r.status_code == 503 and "ANTHROPIC_API_KEY" in r.get_json()["error"]


def test_vlm_api_failure_is_a_502(client, uploaded, fake_vlm):
    fake_vlm.reply = RuntimeError("boom with secret details")
    r = post(client, "/api/vlm/report", image_id=uploaded, dataset="A", variant="best")
    assert r.status_code == 502 and "secret" not in r.get_json()["error"]


def test_provider_rate_limit_is_a_429(client, uploaded, fake_vlm):
    fake_vlm.reply = RuntimeError("429 quota exceeded")
    r = post(client, "/api/vlm/report", image_id=uploaded, dataset="A", variant="best")
    assert r.status_code == 429 and r.get_json()["code"] == "provider_rate_limited"


def test_health_reports_provider(client, monkeypatch):
    monkeypatch.setattr(config, "VLM_MODEL", "ollama:gemma4")
    h = client.get("/api/health").get_json()
    assert h["vlm_configured"] is True and h["vlm_label"] == "local Ollama" and "not sent" in h["vlm_privacy"]
    monkeypatch.setattr(config, "VLM_MODEL", "google_genai:gemini-2.5-flash")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    h = client.get("/api/health").get_json()
    assert h["vlm_configured"] is False and "do not send patient data" in h["vlm_privacy"]


class StubAgent:
    def __init__(self):
        self.seen = []

    def invoke(self, inputs, config=None):
        text = inputs["messages"][0]["content"]
        self.seen.append((text, config["configurable"]["thread_id"]))
        return {"messages": [HumanMessage(content=text), AIMessage(content=[{"type": "text", "text": "hello"}])]}


def test_chat_passes_ui_context(client, uploaded):
    stub = StubAgent()
    agent.set_agent(stub)
    try:
        d = post(client, "/api/chat", session_id="sess1", message="hi", image_id=uploaded,
                 dataset="A", variant="best").get_json()
        assert d == {"reply": "hello", "tools_used": [], "sources": []}
        text, thread = stub.seen[0]
        assert thread == "sess1" and text.startswith(f"[ui_context image_id={uploaded} dataset=A variant=best]\nhi")
        assert post(client, "/api/chat", session_id="sess1", message="").status_code == 400
        assert post(client, "/api/chat", session_id="bad id!", message="x").status_code == 400
        assert post(client, "/api/chat", session_id="s", message="x" * 2001).status_code == 400
        assert post(client, "/api/chat", session_id="s", message="x", image_id="../x").status_code == 400
    finally:
        agent.set_agent(None)


def test_rate_limit(client, monkeypatch):
    monkeypatch.setattr(config, "RATE_LIMIT_PER_MIN", 2)
    agent.set_agent(StubAgent())
    try:
        codes = [post(client, "/api/chat", session_id="s", message="hi").status_code for _ in range(3)]
        assert codes == [200, 200, 429]
    finally:
        agent.set_agent(None)


def test_access_token(client, monkeypatch):
    monkeypatch.setattr(config, "APP_TOKEN", "s3cret")
    assert client.get("/api/health").status_code == 200                      # health stays open
    assert client.get("/api/models").status_code == 401
    assert client.get("/api/models", headers={"X-App-Token": "wrong"}).status_code == 401
    assert client.get("/api/models", headers={"X-App-Token": "s3cret"}).status_code == 200
    assert client.get("/").status_code == 200
