from __future__ import annotations

import numpy as np
from PIL import Image

import config

_STOPS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
_INFERNO = np.array([[0, 0, 4], [87, 16, 110], [188, 55, 84], [249, 142, 9], [252, 255, 164]], dtype=np.float32)


def colorize(values: np.ndarray, size: int = config.IMG_SIZE) -> Image.Image:
    """Map an array in [0, 1] to an RGB heat image of size x size."""
    v = np.clip(np.nan_to_num(values.astype(np.float32)), 0.0, 1.0)
    rgb = np.stack([np.interp(v, _STOPS, _INFERNO[:, c]) for c in range(3)], axis=-1).astype(np.uint8)
    img = Image.fromarray(rgb)
    return img if img.size == (size, size) else img.resize((size, size), Image.Resampling.BILINEAR)


def blend(base: Image.Image, heat: Image.Image, alpha: float = 0.55) -> Image.Image:
    base = base.convert("RGB").resize(heat.size, Image.Resampling.BILINEAR)
    return Image.blend(base, heat.convert("RGB"), alpha)


def upsample(a: np.ndarray, size: int = config.IMG_SIZE) -> np.ndarray:
    if a.shape == (size, size):
        return a.astype(np.float32)
    img = Image.fromarray(a.astype(np.float32), mode="F").resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.float32)


def normalize(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return np.zeros_like(a, dtype=np.float32) if hi - lo < 1e-8 else ((a - lo) / (hi - lo)).astype(np.float32)


def entropy(prob: np.ndarray) -> np.ndarray:
    """Binary entropy in bits (0 = certain, 1 = p of 0.5)."""
    p = np.clip(prob.astype(np.float64), 1e-6, 1 - 1e-6)
    return (-(p * np.log2(p) + (1 - p) * np.log2(1 - p))).astype(np.float32)


def roi_mask(x: float, y: float, r: float, size: int = config.IMG_SIZE) -> np.ndarray:
    """Disc of radius r (pixels, 512 space) around (x, y)."""
    if not (0 <= x < size and 0 <= y < size):
        raise ValueError("The region must be inside the image.")
    r = float(min(max(r, 8), size / 2))
    yy, xx = np.mgrid[0:size, 0:size]
    return (xx - x) ** 2 + (yy - y) ** 2 <= r * r


def image_quality(image: Image.Image) -> dict:
    """Simple, objective acquisition-quality cues (not a clinical gradeability score)."""
    g = np.asarray(image.convert("L").resize((config.IMG_SIZE, config.IMG_SIZE)), dtype=np.float32)
    fov = g > 12                                         # fundus field of view (non-black area)
    inside = g[fov] if fov.any() else g.ravel()
    lap = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
    lap_in = lap[fov[1:-1, 1:-1]] if fov[1:-1, 1:-1].any() else lap.ravel()
    return {
        "brightness": round(float(inside.mean()) / 255, 3),
        "contrast": round(float(inside.std()) / 255, 3),
        "sharpness": round(float(lap_in.var()), 1),       # variance of Laplacian; low = blurry
        "field_of_view_percent": round(float(fov.mean()) * 100, 1),
        "overexposed_percent": round(float((g > 245).mean()) * 100, 2),
    }


_REGIONS = {
    "top_left": (slice(0, 256), slice(0, 256)),
    "top_right": (slice(0, 256), slice(256, 512)),
    "bottom_left": (slice(256, 512), slice(0, 256)),
    "bottom_right": (slice(256, 512), slice(256, 512)),
    "center": (slice(170, 342), slice(170, 342)),
}


def region_stats(prob: np.ndarray, mask: np.ndarray) -> dict:
    """Vessel density and uncertain-pixel share per image region (image coordinates, not anatomy)."""
    prob, mask = upsample(prob), upsample(mask.astype(np.float32)) >= 0.5
    lo, hi = config.UNCERTAIN_BAND
    out = {}
    for name, (ys, xs) in _REGIONS.items():
        p, m = prob[ys, xs], mask[ys, xs]
        out[name] = {"vessel_percent": round(float(m.mean()) * 100, 2),
                     "uncertain_percent": round(float(((p > lo) & (p < hi)).mean()) * 100, 2)}
    return out


def attention_summary(maps: dict[str, np.ndarray], mask: np.ndarray) -> tuple[np.ndarray, dict]:
    """Combined attention map (512) and, per gate, how much of the attention mass lies on vessels."""
    m = upsample(mask.astype(np.float32)) >= 0.5
    vessel_share = float(m.mean())
    gates, ups = {}, []
    for name, a in maps.items():
        u = upsample(a)
        ups.append(u)
        total = float(u.sum())
        on = float(u[m].sum()) / total if total > 0 else 0.0
        gates[name] = {
            "resolution": int(a.shape[0]),
            "mean": round(float(a.mean()), 3),
            "on_vessel_share": round(on, 3),
            "vessel_focus_ratio": round(on / vessel_share, 2) if vessel_share > 0 else None,
        }
    combined = normalize(np.mean(ups, axis=0)) if ups else np.zeros((config.IMG_SIZE,) * 2, np.float32)
    return combined, gates


def cam_summary(cam: np.ndarray, mask: np.ndarray, roi: np.ndarray | None) -> dict:
    m = upsample(mask.astype(np.float32)) >= 0.5
    total = float(cam.sum())
    hot = cam >= 0.5
    out = {
        "energy_on_vessels": round(float(cam[m].sum()) / total, 3) if total > 0 else 0.0,
        "hot_area_percent": round(float(hot.mean()) * 100, 2),
    }
    if roi is not None:
        out["energy_inside_region"] = round(float(cam[roi].sum()) / total, 3) if total > 0 else 0.0
        out["region_area_percent"] = round(float(roi.mean()) * 100, 2)
    ys, xs = np.nonzero(cam >= 0.8)
    if len(xs):
        out["peak_center_xy"] = [int(xs.mean()), int(ys.mean())]
    return out


def mc_summary(mean: np.ndarray, std: np.ndarray, base_mask: np.ndarray, thr: float) -> dict:
    mc_mask = mean >= thr
    base = upsample(base_mask.astype(np.float32)) >= 0.5
    denom = mc_mask.sum() + base.sum()
    dice = 1.0 if denom == 0 else 2.0 * float((mc_mask & base).sum()) / float(denom)
    vessels = base | mc_mask
    return {
        "mean_std": round(float(std.mean()), 4),
        "mean_std_on_vessels": round(float(std[vessels].mean()), 4) if vessels.any() else 0.0,
        "high_variance_percent": round(float((std > 0.1).mean()) * 100, 2),
        "dice_mc_vs_single_pass": round(dice, 4),
    }
