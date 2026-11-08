import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import Dataset, random_split

ROOT = Path(__file__).resolve().parents[1]


def run(script, *args, env=None):
    e = {**os.environ, **(env or {})}
    return subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args],
                          capture_output=True, text=True, env=e, timeout=600)


def test_check_weights_ok_and_bad(weights_dir, tmp_path):
    ok = run("check_weights.py", "--weights", str(weights_dir), "--allow-missing", "--no-forward")
    assert ok.returncode == 0 and "3 ok, 0 bad, 1 missing (of 4)" in ok.stdout
    strict = run("check_weights.py", "--weights", str(weights_dir), "--no-forward")
    assert strict.returncode == 1                                         # missing files fail by default
    (tmp_path / "A").mkdir()
    (tmp_path / "A" / "best.pth").write_bytes(b"junk")
    bad = run("check_weights.py", "--weights", str(tmp_path), "--allow-missing", "--no-forward")
    assert bad.returncode == 1 and "BAD" in bad.stdout


def test_organize_weights(tmp_path, random_ckpt):
    src = tmp_path / "kaggle_A" / "results"
    src.mkdir(parents=True)
    for name in ("standard_attention_unet.pth", "fedavg_attention_unet.pth", "fedprox_attention_unet.pth"):
        os.link(random_ckpt, src / name)
    os.link(random_ckpt, tmp_path / "kaggle_A" / "best_model_AttentionUNet.pth")   # saved outside results/
    dest = tmp_path / "w"
    r = run("organize_weights.py", "--A", str(tmp_path / "kaggle_A"), "--weights", str(dest))
    assert r.returncode == 0, r.stdout + r.stderr
    assert sorted(p.name for p in (dest / "A").iterdir()) == ["best.pth", "fedavg.pth", "fedprox.pth", "standard.pth"]
    # a missing file is reported, not silently skipped
    r = run("organize_weights.py", "--A", str(tmp_path / "kaggle_A" / "results"), "--weights", str(tmp_path / "w2"))
    assert r.returncode == 1 and "best_model_AttentionUNet.pth" in r.stdout


def test_eval_split_equals_the_notebooks_random_split():
    """The notebooks call random_split(dataset, [train, val, test], generator=seed 42) on the dataset."""
    class Dummy(Dataset):
        def __init__(self, n): self.n = n
        def __len__(self): return self.n
        def __getitem__(self, i): return i

    for n, sizes in {800: (640, 80, 80), 100: (70, 15, 15), 45: (31, 6, 8), 28: (18, 4, 6)}.items():
        nb = random_split(Dummy(n), list(sizes), generator=torch.Generator().manual_seed(42))
        mine = random_split(range(n), list(sizes), generator=torch.Generator().manual_seed(42))
        assert [s.indices for s in nb] == [s.indices for s in mine]
    # and the notebook's own size arithmetic for the 28-image dataset (0.643 / 0.1785 ratios)
    assert (int(28 * 0.643), int(28 * 0.1785), 28 - int(28 * 0.643) - int(28 * 0.1785)) == (18, 4, 6)
    assert (int(45 * 0.70), int(45 * 0.15), 45 - int(45 * 0.70) - int(45 * 0.15)) == (31, 6, 8)


def test_eval_folder_checks_dataset(tmp_path):
    imgs, masks = tmp_path / "img", tmp_path / "msk"
    imgs.mkdir(); masks.mkdir()
    rng = np.random.default_rng(0)
    for i in range(28):
        Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(imgs / f"{i:02d}.png")
        Image.fromarray((rng.random((64, 64)) > 0.8).astype(np.uint8) * 255).save(masks / f"{i:02d}.png")
    removed = run("eval_folder.py", "--dataset", "D", "--images", str(imgs), "--masks", str(masks))
    assert removed.returncode != 0 and "invalid choice" in removed.stderr       # only Dataset A is served
    short = run("eval_folder.py", "--dataset", "A", "--images", str(imgs), "--masks", str(masks))
    assert short.returncode != 0 and "should have 800" in (short.stdout + short.stderr)
