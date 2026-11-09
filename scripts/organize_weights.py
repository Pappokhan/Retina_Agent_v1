#!/usr/bin/env python
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402

SOURCE_NAMES = {
    "best": "best_model_AttentionUNet.pth",
    "standard": "standard_attention_unet.pth",
    "fedavg": "fedavg_attention_unet.pth",
    "fedprox": "fedprox_attention_unet.pth",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for ds in config.DATASETS:
        ap.add_argument(f"--{ds}", type=Path, help=f"folder holding the four files saved by notebook {ds}")
    ap.add_argument("--weights", type=Path, default=config.WEIGHTS_DIR, help="destination (default: %(default)s)")
    ap.add_argument("--move", action="store_true", help="move instead of copy")
    args = ap.parse_args()

    problems = copied = 0
    for ds in config.DATASETS:
        src_root = getattr(args, ds)
        if src_root is None:
            continue
        for var, name in SOURCE_NAMES.items():
            hits = sorted(src_root.expanduser().rglob(name))
            if len(hits) != 1:
                problems += 1
                print(f"[{ds}/{var}] " + (f"not found: {name}" if not hits else
                      f"{len(hits)} files named {name}; keep one:\n    " + "\n    ".join(map(str, hits))))
                continue
            dest = args.weights / ds / config.VARIANTS[var]["file"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            (shutil.move if args.move else shutil.copy2)(hits[0], dest)
            copied += 1
            print(f"[{ds}/{var}] {hits[0]} -> {dest}")
    print(f"\n{copied} file(s) placed, {problems} problem(s). Next: python scripts/check_weights.py")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
