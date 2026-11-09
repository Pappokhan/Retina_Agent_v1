#!/usr/bin/env python
import argparse
import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import random_split  # noqa: E402
from torchvision import transforms  # noqa: E402

import config  # noqa: E402
from inference import ModelRegistry  # noqa: E402

SPLITS = {"A": (640, 80, 80), "B": (70, 15, 15), "C": (31, 6, 8), "D": (18, 4, 6)}   # train/val/test in the notebooks
REPORTED = {   # test metrics printed by each notebook (final-epoch model)
    "A": {"Dice": 0.9295, "IoU": 0.8682, "Sensitivity": 0.9261, "Specificity": 0.9948},
    "B": {"Dice": 0.8236, "IoU": 0.7001, "Sensitivity": 0.8485, "Specificity": 0.9798},
    "C": {"Dice": 0.8299, "IoU": 0.7093, "Sensitivity": 0.8351, "Specificity": 0.9845},
    "D": {"Dice": 0.7935, "IoU": 0.6577, "Sensitivity": 0.8543, "Specificity": 0.9774},
}


def collect(dirs):
    return [p for d in dirs for p in sorted(glob.glob(os.path.join(d, "*.*")))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=list(config.DATASETS))
    ap.add_argument("--variant", default="standard", choices=list(config.VARIANTS))
    ap.add_argument("--images", action="append", required=True)
    ap.add_argument("--masks", action="append", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    args = ap.parse_args()

    imgs, masks = collect(args.images), collect(args.masks)
    if len(imgs) != len(masks):
        sys.exit(f"{len(imgs)} images but {len(masks)} masks.")
    tr, va, te = SPLITS[args.dataset]
    chosen = list(range(len(imgs)))
    if args.split != "all":
        if len(imgs) != tr + va + te:
            sys.exit(f"Dataset {args.dataset} should have {tr + va + te} pairs, found {len(imgs)}; "
                     "check the folders (or use --split all).")
        parts = random_split(range(len(imgs)), [tr, va, te], generator=torch.Generator().manual_seed(42))
        chosen = list(parts[("train", "val", "test").index(args.split)].indices)

    reg = ModelRegistry()
    mask_tf = transforms.Compose([transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)), transforms.ToTensor()])
    tp = fp = fn = tn = 0
    per_image = []
    for i in chosen:
        pred = reg.predict(Image.open(imgs[i]).convert("RGB"), args.dataset, args.variant, 0.5).mask
        gt = (mask_tf(Image.open(masks[i]).convert("L")) > 0.5).squeeze(0).numpy()
        t, f_p, f_n = int((pred & gt).sum()), int((pred & ~gt).sum()), int((~pred & gt).sum())
        tp, fp, fn, tn = tp + t, fp + f_p, fn + f_n, tn + int((~pred & ~gt).sum())
        per_image.append(2 * t / max(1, 2 * t + f_p + f_n))

    eps = 1e-6
    got = {"Dice": (2 * tp + eps) / (2 * tp + fp + fn + eps), "IoU": (tp + eps) / (tp + fp + fn + eps),
           "Sensitivity": (tp + eps) / (tp + fn + eps), "Specificity": (tn + eps) / (tn + fp + eps)}
    print(f"dataset {args.dataset} | variant {args.variant} | split {args.split} | {len(chosen)} images | device {reg.device}")
    print("(pooled over all pixels, like the notebooks)")
    for k, v in got.items():
        ref = REPORTED[args.dataset][k] if args.variant == "standard" and args.split == "test" else None
        extra = f"   notebook {ref:.4f}   diff {v - ref:+.4f}" if ref is not None else ""
        print(f"  {k:<12}{v:.4f}{extra}")
    print(f"  per-image Dice: mean {np.mean(per_image):.4f}, std {np.std(per_image):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
