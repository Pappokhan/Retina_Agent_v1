#!/usr/bin/env python
"""Verify the 16 checkpoints: present, loadable into AttentionUNet (strict), finite output.

    python scripts/check_weights.py
    python scripts/check_weights.py --allow-missing --no-forward
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

import config  # noqa: E402
from inference import BadCheckpoint, load_weights  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=str(config.WEIGHTS_DIR), help="weights folder (default: %(default)s)")
    ap.add_argument("--allow-missing", action="store_true", help="do not fail when some files are missing")
    ap.add_argument("--no-forward", action="store_true", help="skip the 512x512 smoke forward pass")
    args = ap.parse_args()

    root = Path(args.weights)
    dev = torch.device("cpu")
    ok = bad = missing = 0
    print(f"Weights folder: {root}\n")
    print(f"{'dataset':<8}{'variant':<10}{'status':<9}detail")
    for ds in config.DATASETS:
        for var, meta in config.VARIANTS.items():
            path = root / ds / meta["file"]
            if not path.is_file():
                missing += 1
                print(f"{ds:<8}{var:<10}{'MISSING':<9}{path}")
                continue
            try:
                model = load_weights(path, dev)
                detail = f"{sum(p.numel() for p in model.parameters()):,} params, {path.stat().st_size / 1e6:.1f} MB"
                if not args.no_forward:
                    with torch.inference_mode():
                        out = model(torch.zeros(1, 3, config.IMG_SIZE, config.IMG_SIZE))
                    if out.shape != (1, 1, config.IMG_SIZE, config.IMG_SIZE) or not torch.isfinite(out).all():
                        raise BadCheckpoint("forward pass gave a wrong shape or non-finite values")
                ok += 1
                print(f"{ds:<8}{var:<10}{'OK':<9}{detail}")
            except BadCheckpoint as exc:
                bad += 1
                print(f"{ds:<8}{var:<10}{'BAD':<9}{exc}")
    print(f"\n{ok} ok, {bad} bad, {missing} missing (of {len(config.DATASETS) * len(config.VARIANTS)})")
    return 1 if bad or (missing and not args.allow_missing) else 0


if __name__ == "__main__":
    sys.exit(main())
