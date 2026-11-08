import os

import numpy as np
import pytest
import torch
from PIL import Image

import config
import services
from inference import BadCheckpoint, ModelNotAvailable, ModelRegistry, build_transform, load_weights

REAL = os.getenv("REAL_CKPT", "/mnt/user-data/uploads/standard_attention_unet.pth")


def img(size=(300, 200)):
    return Image.fromarray(np.random.default_rng(1).integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))


@pytest.mark.skipif(not os.path.isfile(REAL), reason="real checkpoint not available")
def test_real_checkpoint_loads_strictly_and_predicts():
    model = load_weights(REAL, torch.device("cpu"))          # strict=True inside
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 13_746_861   # as printed by the notebooks
    assert not model.training
    with torch.inference_mode():
        out = model(torch.zeros(1, 3, 512, 512))
    assert out.shape == (1, 1, 512, 512) and torch.isfinite(out).all()


def test_preprocessing_is_the_notebook_pipeline():
    x = build_transform()(Image.new("RGB", (123, 77), (255, 255, 255)))
    assert x.shape == (3, 512, 512)
    for c in range(3):    # white pixel -> (1 - mean) / std
        expected = (1 - config.IMAGENET_MEAN[c]) / config.IMAGENET_STD[c]
        assert torch.allclose(x[c], torch.full((512, 512), expected), atol=1e-5)


def test_predict_shapes_and_threshold(env):
    reg = services.get_registry()
    p = reg.predict(img(), "A", "best")
    assert p.prob.shape == (512, 512) and p.prob.dtype == np.float32
    assert 0.0 <= p.prob.min() and p.prob.max() <= 1.0
    assert np.array_equal(p.mask, p.prob >= 0.5)
    assert reg.predict(img(), "A", "best", threshold=0.9).mask.sum() <= p.mask.sum()
    assert p.ms > 0


def test_missing_and_invalid_choices(env):
    reg = services.get_registry()
    with pytest.raises(ModelNotAvailable):
        reg.predict(img(), "A", "fedprox")
    for ds in ("B", "C", "D", "Z"):                 # only the Dataset A model is served
        with pytest.raises(ValueError):
            reg.predict(img(), ds, "best")
    with pytest.raises(ValueError):
        reg.predict(img(), "A", "nope")
    assert reg.available() == {"A": {"best": True, "standard": True, "fedavg": True, "fedprox": False}}


def test_live_sizes_and_attention_capture(env):
    reg = services.get_registry()
    p = reg.predict(img(), "A", "best", size=256, attention=True)
    assert p.prob.shape == (256, 256)
    assert {k: v.shape for k, v in p.attention.items()} == {"up1": (32, 32), "up2": (64, 64), "up3": (128, 128), "up4": (256, 256)}
    assert all(0 <= v.min() and v.max() <= 1 for v in p.attention.values())
    assert reg.predict(img(), "A", "best").attention == {}           # hooks only when asked, and removed after
    with pytest.raises(ValueError):
        reg.predict(img(), "A", "best", size=300)


def test_gradcam_and_mc_dropout(env):
    import xai
    reg = services.get_registry()
    cam, info = reg.gradcam(img(), "A", "best", "up2")
    assert cam.shape == (512, 512) and 0 <= cam.min() and cam.max() <= 1 and info["layer"] == "up2"
    roi = xai.roi_mask(100, 100, 30)
    _, info = reg.gradcam(img(), "A", "best", "down4", roi=roi)
    assert info["target_pixels"] <= int(roi.sum())
    model = reg.get("A", "best")
    assert not model.training and all(p.grad is None for p in model.parameters())   # left clean
    mean, std, ms = reg.mc_dropout(img(), "A", "best", samples=3)
    assert mean.shape == std.shape == (512, 512) and std.max() > 0 and ms > 0
    assert not any(m.training for m in model.modules())                           # dropout switched back off
    with pytest.raises(ValueError):
        reg.gradcam(img(), "A", "best", "outc")


def test_bad_checkpoints_are_rejected(tmp_path):
    wrong = tmp_path / "wrong.pth"
    torch.save({"conv.weight": torch.zeros(1)}, wrong)
    with pytest.raises(BadCheckpoint):
        load_weights(wrong, torch.device("cpu"))
    junk = tmp_path / "junk.pth"
    junk.write_bytes(b"not a checkpoint")
    with pytest.raises(BadCheckpoint):
        load_weights(junk, torch.device("cpu"))


def test_wrapped_and_dataparallel_checkpoints_load(tmp_path, random_ckpt):
    state = torch.load(random_ckpt, weights_only=True)
    path = tmp_path / "wrapped.pth"
    torch.save({"state_dict": {"module." + k: v for k, v in state.items()}}, path)
    load_weights(path, torch.device("cpu"))


def test_lru_cache_evicts_oldest(env, weights_dir):
    reg = ModelRegistry(weights_dir=weights_dir, device="cpu", cache_size=1)
    reg.get("A", "best")
    reg.get("A", "standard")
    assert reg.loaded() == ["A/standard"]
