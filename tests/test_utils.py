import os
import time

import numpy as np
import pytest
from PIL import Image

import app_utils as U
import config
from conftest import png_bytes


@pytest.mark.parametrize("bad", ["", "../x", "a/b", "a b", "x" * 65, "a.png", None, 5])
def test_check_id_rejects(bad):
    with pytest.raises(ValueError):
        U.check_id(bad)


def test_check_id_accepts():
    assert U.check_id("abc123_DEF") == "abc123_DEF"


def test_upload_path_cannot_escape(env):
    with pytest.raises(ValueError):
        U.upload_path("../../etc/passwd")


def test_decode_rejects_non_images(env):
    for data in (b"", b"hello world", b"%PDF-1.4 fake"):
        with pytest.raises(U.BadImage):
            U.decode_image(data)


def test_decode_limits_size(env, monkeypatch):
    monkeypatch.setattr(config, "MAX_IMAGE_PIXELS", 1000)
    with pytest.raises(U.BadImage):
        U.decode_image(png_bytes((100, 100)))


def test_decode_converts_modes(env):
    import io
    buf = io.BytesIO()
    Image.new("RGBA", (20, 10), (255, 0, 0, 128)).save(buf, "PNG")
    assert U.decode_image(buf.getvalue()).mode == "RGB"
    buf = io.BytesIO()
    Image.new("L", (20, 10), 7).save(buf, "TIFF")
    assert U.decode_image(buf.getvalue()).size == (20, 10)


def test_upload_roundtrip_and_unknown_id(env):
    image_id, w, h = U.save_upload(png_bytes((321, 123)))
    assert (w, h) == (321, 123)
    assert U.load_upload(image_id).size == (321, 123)
    with pytest.raises(FileNotFoundError):
        U.load_upload("doesnotexist")


def test_overlay_only_recolors_the_mask(env):
    img = Image.new("RGB", (64, 64), (200, 40, 40))
    mask = np.zeros((512, 512), bool)
    mask[100:120, 100:120] = True
    out = np.asarray(U.overlay(img, mask))
    assert out.shape == (512, 512, 3)
    assert tuple(out[0, 0]) == (200, 40, 40)                         # untouched background
    assert out[110, 110, 1] > 100 and out[110, 110, 0] < 200       # moved toward cyan


def test_heatmap_range():
    h = np.asarray(U.heatmap(np.array([[0.0, 1.0], [0.5, 2.0]])))
    assert h.shape == (2, 2, 3)
    assert not np.array_equal(h[0, 0], h[0, 1])


def test_mask_stats_known_values():
    prob = np.zeros((10, 10), np.float32)
    prob[:5] = 0.9
    prob[5:7] = 0.5
    s = U.mask_stats(prob, prob >= 0.5)
    assert s["area_percent"] == 70.0
    assert s["uncertain_percent"] == 20.0
    assert s["mean_confidence"] == pytest.approx((0.9 * 50 + 0.5 * 20) / 70, abs=1e-3)


def test_pairwise_dice():
    a = np.zeros((4, 4), bool); a[:2] = True
    b = a.copy()
    c = np.zeros((4, 4), bool); c[2:] = True
    empty = np.zeros((4, 4), bool)
    r = U.pairwise({"a": a, "b": b, "c": c, "e": empty, "f": empty.copy()})
    assert r["dice"]["a_vs_b"] == 1.0 and r["dice"]["a_vs_c"] == 0.0
    assert r["dice"]["e_vs_f"] == 1.0                                # both empty: identical
    assert r["disagreement_percent"]["a_vs_c"] == 100.0


def test_purge_removes_only_old_files(env):
    keep, _, _ = U.save_upload(png_bytes((8, 8)))
    old, _, _ = U.save_upload(png_bytes((8, 8), seed=2))
    past = time.time() - (config.UPLOAD_TTL_HOURS * 3600 + 60)
    os.utime(U.upload_path(old), (past, past))
    assert U.purge_uploads(force=True) == 1
    assert U.upload_path(keep).exists() and not U.upload_path(old).exists()
