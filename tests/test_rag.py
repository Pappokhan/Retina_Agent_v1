import shutil

import pytest

import config
import rag


@pytest.fixture(autouse=True)
def fresh_retriever():
    rag.clear_cache()
    yield
    rag.clear_cache()


@pytest.mark.parametrize("query, expected", [
    ("Is FedAvg real federated averaging", "federated_setup.md > Important caveat: not true federated averaging"),
    ("What does uncertain pixels mean", "measurements.md > Segmentation numbers"),
    ("test Dice and IoU", "model_card.md > Reported test results"),
    ("how many images are in the dataset", "datasets.md > Size and split"),
    ("are uploaded images deleted privacy", "limitations.md > Privacy"),
    ("what is the attention focus ratio", "xai.md > Attention gate map"),
    ("Grad-CAM region evidence", "xai.md > Grad-CAM (HiResCAM)"),
    ("what does sharpness mean", "measurements.md > Image quality cues"),
    ("live camera resolution auto speed", "realtime.md > Resolution and speed"),
    ("how was the model trained", "model_card.md > Training setup"),
])
def test_retrieval_finds_the_right_section(query, expected):
    labels = [h.label for h in rag.retrieve(query, k=4)]
    assert expected in labels, labels


def test_hits_are_ranked_deduplicated_and_trimmed():
    hits = rag.retrieve("FedProx proximal anchor caveat", k=6)
    assert hits[0].source == "federated_setup.md"
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert all(h.score >= 0.35 * hits[0].score for h in hits)            # weak tail dropped
    assert len({(h.source, h.text[:80]) for h in hits}) == len(hits)
    assert rag.retrieve("   ") == [] and rag.retrieve("the of and") == []


def test_chunks_keep_file_and_section():
    chunks = rag.load_chunks()
    assert {c.metadata["source"] for c in chunks} == {
        "datasets.md", "model_card.md", "federated_setup.md", "limitations.md", "usage.md",
        "xai.md", "measurements.md", "realtime.md"}
    assert all(c.metadata["section"] for c in chunks)
    assert all(len(c.page_content) <= rag.CHUNK_SIZE for c in chunks)
    assert not any(c.page_content.lstrip().startswith("#") for c in chunks)   # headers live in metadata


def test_search_output_is_numbered_and_labelled():
    out = rag.search("federated caveat")
    assert out.startswith("[1] federated_setup.md > ")
    assert rag.search("zzzz qqqq") == "No matching notes found."


def test_context_block_and_sources_for_citations():
    hits = rag.retrieve("attention focus ratio")
    block = rag.context_block(hits)
    assert block.startswith("Project notes") and "[1] (xai.md > Attention gate map)" in block and "cite" in block
    assert rag.context_block([]) == ""
    src = rag.sources(hits)
    assert src[0]["id"] == 1 and src[0]["source"] == "xai.md" and len(src[0]["snippet"]) <= 221
    assert len(rag.context_block(hits * 10, max_chars=1000)) < 1400              # bounded prompt size


def test_index_refreshes_when_notes_change(tmp_path, monkeypatch):
    kdir = tmp_path / "knowledge"
    shutil.copytree(config.KNOWLEDGE_DIR, kdir)
    monkeypatch.setattr(config, "KNOWLEDGE_DIR", kdir)
    assert not rag.retrieve("zebrafish retina")
    (kdir / "extra.md").write_text("# Extra\n\n## Zebrafish\n\nZebrafish retina images are not supported.\n",
                                   encoding="utf-8")
    assert rag.retrieve("zebrafish retina")[0].label == "extra.md > Zebrafish"   # no restart needed


def test_queries_for_features_hit_relevant_notes():
    assert any(h.source == "federated_setup.md" for h in rag.retrieve(rag.query_for("report", "fedavg")))
    assert rag.retrieve(rag.query_for("xai", "best", "mc_dropout"))[0].label == "xai.md > Monte-Carlo dropout"
    assert any("federated" in h.source for h in rag.retrieve(rag.query_for("compare")))
