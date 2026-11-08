import json
import os

import tools
import app_utils as U
from conftest import png_bytes


def run(tool, **kw):
    return tool.invoke(kw)


def test_all_tools_are_registered_with_descriptions():
    names = {t.name for t in tools.TOOLS}
    assert names == {"list_models", "segment_image", "compare_models", "qc_check", "inspect_overlay",
                     "explain_prediction", "search_knowledge"}
    assert all(t.description.strip() for t in tools.TOOLS)


def test_list_models(env):
    out = run(tools.list_models)
    assert out == "Dataset A: best, standard, fedavg"


def test_segment_image_saves_overlay(env):
    image_id, _, _ = U.save_upload(png_bytes())
    d = json.loads(run(tools.segment_image, image_id=image_id, dataset="A", variant="best"))
    assert d["overlay_id"] == f"{image_id}_A_best_ov" and os.path.isfile(U.upload_path(d["overlay_id"]))
    assert {"area_percent", "mean_confidence", "uncertain_percent", "model_ms"} <= set(d)


def test_compare_models(env):
    image_id, _, _ = U.save_upload(png_bytes())
    d = json.loads(run(tools.compare_models, image_id=image_id, dataset="A"))
    assert set(d["area_percent"]) == {"best", "standard", "fedavg"}
    assert d["variants_not_installed"] == ["fedprox"] and len(d["pairwise_dice"]) == 3


def test_expected_failures_come_back_as_text(env):
    image_id, _, _ = U.save_upload(png_bytes())
    assert run(tools.segment_image, image_id=image_id, dataset="Z").startswith("ERROR:")
    assert "only" in run(tools.segment_image, image_id=image_id, dataset="C")
    assert "No weights" in run(tools.segment_image, image_id=image_id, variant="fedprox")
    assert run(tools.segment_image, image_id="nope", dataset="A").startswith("ERROR:")
    assert run(tools.segment_image, image_id="../x", dataset="A").startswith("ERROR:")
    assert run(tools.compare_models, image_id=image_id, dataset="B").startswith("ERROR:")
    d = json.loads(run(tools.explain_prediction, image_id=image_id, method="uncertainty"))
    assert d["method"] == "uncertainty" and "mean_entropy_bits" in d["summary"]
    assert run(tools.explain_prediction, image_id=image_id, method="magic").startswith("ERROR:")


def test_vlm_tools(env, fake_vlm):
    image_id, _, _ = U.save_upload(png_bytes())
    seg = json.loads(run(tools.segment_image, image_id=image_id, dataset="A", variant="best"))
    fake_vlm.reply = "The mask covers the vessels."
    out = run(tools.inspect_overlay, overlay_id=seg["overlay_id"], question="Any gaps?")
    assert out == "The mask covers the vessels."
    content = fake_vlm.last[1]
    assert sum(b["type"] == "image_url" for b in content) == 2          # original + overlay
    assert run(tools.inspect_overlay, overlay_id=image_id, question="x").startswith("ERROR:")   # not an overlay id

    fake_vlm.reply = ('{"description": "d", "qc": {"looks_plausible": true, "issues": []}, "caveats": []}')
    qc = json.loads(run(tools.qc_check, image_id=image_id, dataset="A"))
    assert qc["report"]["qc"]["looks_plausible"] is True and "area_percent" in qc["stats"]


def test_vlm_tool_without_key_is_an_error_string(env, monkeypatch):
    import vlm
    import config
    monkeypatch.setattr(vlm, "_model", None)
    monkeypatch.setattr(config, "VLM_MODEL", "anthropic:claude-sonnet-5-5")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    image_id, _, _ = U.save_upload(png_bytes())
    assert "ANTHROPIC_API_KEY" in run(tools.qc_check, image_id=image_id, dataset="A")


def test_search_knowledge():
    out = run(tools.search_knowledge, query="FedProx caveat")
    assert out.startswith("[1] federated_setup.md > ")
