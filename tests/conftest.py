import io
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

os.environ.setdefault("RAG_BACKEND", "bm25")

LAYOUT = {"A": ["best", "standard", "fedavg"]}


def png_bytes(size=(640, 480), seed=0, fmt="PNG") -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, fmt)
    return buf.getvalue()


@pytest.fixture(scope="session")
def random_ckpt(tmp_path_factory) -> Path:
    """A randomly initialised AttentionUNet state_dict (same keys as the real checkpoints)."""
    import torch
    from inference import build_model

    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("ckpt") / "random.pth"
    torch.save(build_model().state_dict(), path)
    return path


@pytest.fixture(scope="session")
def weights_dir(tmp_path_factory, random_ckpt) -> Path:
    root = tmp_path_factory.mktemp("weights")
    for ds, variants in LAYOUT.items():
        (root / ds).mkdir()
        for v in variants:
            os.link(random_ckpt, root / ds / config.VARIANTS[v]["file"])   # hard link: no 55 MB copies
    return root


@pytest.fixture
def env(tmp_path, monkeypatch, weights_dir):
    """Isolated weights + uploads for one test."""
    import services

    monkeypatch.setattr(config, "WEIGHTS_DIR", weights_dir)
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path / "uploads")
    services.reset_registry()
    yield tmp_path
    services.reset_registry()


@pytest.fixture
def client(env):
    from app import create_app

    app = create_app()
    app.testing = True
    return app.test_client()


@pytest.fixture
def uploaded(client):
    r = client.post("/api/upload", data={"image": (io.BytesIO(png_bytes()), "x.png")},
                    content_type="multipart/form-data")
    assert r.status_code == 200, r.get_json()
    return r.get_json()["image_id"]

class _Msg:
    def __init__(self, text):
        self.content = text


class FakeChat:
    """Records every call; `reply` may be text or an Exception to raise."""

    def __init__(self, reply='{"description": "Thin branching vessels.", "qc": {"looks_plausible": true, "issues": []}, "caveats": ["one image"]}',
                 parts=("The mask ", "covers the vessels.")):
        self.reply, self.parts, self.calls = reply, list(parts), []

    def invoke(self, messages):
        self.calls.append(messages)
        if isinstance(self.reply, Exception):
            raise self.reply
        return _Msg(self.reply)

    def stream(self, messages):
        self.calls.append(messages)
        if isinstance(self.reply, Exception):
            raise self.reply
        return iter(_Msg(p) for p in self.parts)

    @property
    def last(self):
        """(system text, human content blocks) of the latest call."""
        system, human = self.calls[-1]
        return system.content, human.content


@pytest.fixture
def fake_vlm(monkeypatch):
    import vlm

    model = FakeChat()
    monkeypatch.setattr(vlm, "_model", model)
    monkeypatch.setattr(vlm, "_json_model", model)
    vlm.clear_cache()
    yield model
    vlm.clear_cache()
