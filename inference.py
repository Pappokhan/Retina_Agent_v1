from __future__ import annotations

import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import config
from models import AttentionUNet


class ModelNotAvailable(FileNotFoundError):

class BadCheckpoint(RuntimeError):

GATES = ("up1", "up2", "up3", "up4") 


@dataclass
class Prediction:
    prob: np.ndarray   
    mask: np.ndarray   
    ms: float      
    attention: dict[str, np.ndarray] = field(default_factory=dict) 


def validate_choice(dataset: str, variant: str) -> None:
    if dataset not in config.DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. This app serves only: {', '.join(config.DATASETS)}.")
    if variant not in config.VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Choose one of: {', '.join(config.VARIANTS)}.")


def build_transform(size: int = config.IMG_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])


def check_size(size: int | None) -> int:
    if size is None:
        return config.IMG_SIZE
    size = int(size)
    if size not in config.LIVE_SIZES:
        raise ValueError(f"size must be one of {', '.join(map(str, config.LIVE_SIZES))}.")
    return size


def build_model() -> nn.Module:
    return AttentionUNet(n_channels=3, n_classes=1, bilinear=True, dropout=0.1)


def _extract_state_dict(obj) -> dict:
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if isinstance(obj.get(key), dict):
                obj = obj[key]
                break
    if not isinstance(obj, dict) or not obj:
        raise BadCheckpoint("the file does not contain a state_dict")
    return {k.removeprefix("module."): v for k, v in obj.items()}


def load_weights(path: Path, device: torch.device) -> nn.Module:
    path = Path(path)
    try:
        raw = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc: 
        raise BadCheckpoint(f"{path.name}: cannot be read ({type(exc).__name__}: {exc})") from exc
    state = _extract_state_dict(raw)
    model = build_model()
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise BadCheckpoint(
            f"{path.name}: weights do not match AttentionUNet (wrong architecture or renamed layers). "
            f"{str(exc).splitlines()[0]}"
        ) from exc
    return model.to(device).to(memory_format=torch.channels_last).eval()


@contextmanager
def _capture_attention(model: nn.Module, out: dict):
    """Record each attention gate's psi map (B, 1, h, w) during one forward pass."""
    handles = []
    for name in GATES:
        gate = getattr(model, name).att.psi
        handles.append(gate.register_forward_hook(
            lambda _m, _i, o, n=name: out.__setitem__(n, o[0, 0].detach().float().cpu().numpy())))
    try:
        yield out
    finally:
        for h in handles:
            h.remove()


@contextmanager
def _dropout_active(model: nn.Module):
    """Turn on only the Dropout2d layers (BatchNorm stays in eval mode) for MC dropout."""
    drops = [m for m in model.modules() if isinstance(m, nn.Dropout2d)]
    for m in drops:
        m.train()
    try:
        yield len(drops)
    finally:
        for m in drops:
            m.eval()


class ModelRegistry:
    """Lazy-loading LRU cache of the checkpoints."""

    def __init__(self, weights_dir: Path | None = None, device: str | None = None,
                 cache_size: int | None = None):
        self.weights_dir = Path(weights_dir or config.WEIGHTS_DIR)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.cache_size = max(1, cache_size or config.MODEL_CACHE_SIZE)
        self._cache: OrderedDict[tuple[str, str], nn.Module] = OrderedDict()
        self._load_lock = threading.Lock()
        self._run_lock = threading.Lock()   # one forward pass at a time
        self._transforms: dict[int, transforms.Compose] = {}

    def path(self, dataset: str, variant: str) -> Path:
        validate_choice(dataset, variant)
        return self.weights_dir / dataset / config.VARIANTS[variant]["file"]

    def available(self) -> dict[str, dict[str, bool]]:
        return {
            ds: {var: self.path(ds, var).is_file() for var in config.VARIANTS}
            for ds in config.DATASETS
        }

    def available_variants(self, dataset: str) -> list[str]:
        return [v for v, ok in self.available()[dataset].items() if ok]

    def loaded(self) -> list[str]:
        return [f"{d}/{v}" for d, v in self._cache]

    def _x(self, image: Image.Image, size: int) -> torch.Tensor:
        t = self._transforms.get(size)
        if t is None:
            t = self._transforms[size] = build_transform(size)
        return t(image.convert("RGB")).unsqueeze(0).to(self.device).contiguous(memory_format=torch.channels_last)

    def get(self, dataset: str, variant: str) -> nn.Module:
        key = (dataset, variant)
        with self._load_lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            path = self.path(dataset, variant)
            if not path.is_file():
                raise ModelNotAvailable(
                    f"No weights for dataset {dataset} / {variant}. Expected the file {path}."
                )
            model = load_weights(path, self.device)
            self._cache[key] = model
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
            return model

    def warmup(self, dataset: str, variant: str = "best", sizes=(config.IMG_SIZE,)) -> None:
        """Load a model and run one pass per size so the first real request is not slow."""
        try:
            for s in sizes:
                self.predict(Image.new("RGB", (s, s)), dataset, variant, size=s)
        except (ModelNotAvailable, BadCheckpoint):
            pass

    def predict(self, image: Image.Image, dataset: str, variant: str,
                threshold: float | None = None, size: int | None = None,
                attention: bool = False) -> Prediction:
        thr = config.THRESHOLD if threshold is None else float(threshold)
        size = check_size(size)
        model = self.get(dataset, variant)
        x = self._x(image, size)
        use_cuda = self.device.type == "cuda"
        maps: dict[str, np.ndarray] = {}
        with self._run_lock, torch.inference_mode():
            hooks = _capture_attention(model, maps) if attention else _null()
            with hooks:
                t0 = time.perf_counter()
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):
                    logits = model(x)
                prob = torch.sigmoid(logits.float())[0, 0]
                if use_cuda:
                    torch.cuda.synchronize()
                ms = (time.perf_counter() - t0) * 1000.0
        prob_np = prob.cpu().numpy().astype(np.float32)
        return Prediction(prob=prob_np, mask=prob_np >= thr, ms=ms, attention=maps)

    def gradcam(self, image: Image.Image, dataset: str, variant: str,
                layer: str = config.GRADCAM_DEFAULT_LAYER, roi: np.ndarray | None = None,
                threshold: float | None = None) -> tuple[np.ndarray, dict]:
        if layer not in config.GRADCAM_LAYERS:
            raise ValueError(f"layer must be one of {', '.join(config.GRADCAM_LAYERS)}.")
        thr = config.THRESHOLD if threshold is None else float(threshold)
        model = self.get(dataset, variant)
        x = self._x(image, config.IMG_SIZE)
        module = getattr(model, layer)
        store: dict[str, torch.Tensor] = {}

        def fwd(_m, _i, o):
            o.retain_grad()
            store["act"] = o

        with self._run_lock, torch.enable_grad():
            handle = module.register_forward_hook(fwd)
            try:
                t0 = time.perf_counter()
                logits = model(x)[0, 0]
                prob = torch.sigmoid(logits.detach())
                vessel = prob >= thr
                region = torch.ones_like(vessel) if roi is None else torch.from_numpy(roi.astype(bool)).to(vessel.device)
                target_mask = vessel & region
                used_vessels = bool(target_mask.any())
                if not used_vessels:
                    target_mask = region
                target = (logits * target_mask.float()).sum()
                model.zero_grad(set_to_none=True)
                target.backward()
                act, grad = store["act"], store["act"].grad
                cam = F.relu((grad * act).sum(dim=1, keepdim=True))     # (1, 1, h, w)
                cam = F.interpolate(cam, size=(config.IMG_SIZE, config.IMG_SIZE), mode="bilinear",
                                    align_corners=False)[0, 0].detach()
                ms = (time.perf_counter() - t0) * 1000.0
            finally:
                handle.remove()
                model.zero_grad(set_to_none=True)
        cam_np = cam.float().cpu().numpy()
        peak = float(cam_np.max())
        cam_np = cam_np / peak if peak > 0 else cam_np
        info = {"layer": layer, "target": "vessel pixels" if used_vessels else "whole region",
                "target_pixels": int(target_mask.sum().item()), "ms": round(ms, 1)}
        return cam_np.astype(np.float32), info

    def mc_dropout(self, image: Image.Image, dataset: str, variant: str,
                   samples: int = config.MC_SAMPLES) -> tuple[np.ndarray, np.ndarray, float]:
        """Mean and std of the vessel probability over `samples` dropout-on passes (512 x 512)."""
        samples = int(max(2, min(config.MC_MAX_SAMPLES, samples)))
        model = self.get(dataset, variant)
        x = self._x(image, config.IMG_SIZE)
        runs = []
        with self._run_lock, torch.inference_mode(), _dropout_active(model):
            t0 = time.perf_counter()
            for _ in range(samples):
                runs.append(torch.sigmoid(model(x).float())[0, 0].cpu())
            ms = (time.perf_counter() - t0) * 1000.0
        stack = torch.stack(runs).numpy()
        return stack.mean(0).astype(np.float32), stack.std(0).astype(np.float32), ms


@contextmanager
def _null():
    yield
