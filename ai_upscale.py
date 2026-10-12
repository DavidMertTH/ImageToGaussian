"""AI upscaling of SHARP inputs with Real-ESRGAN x4plus (via spandrel).

SHARP resizes every input to its square network resolution. When a photo is
smaller than that, plain bilinear enlargement adds no detail; Real-ESRGAN
synthesises plausible fine texture first. New detail is invented, not recovered.

Setup: ``setup-upscale.ps1`` (installs spandrel and downloads the weights).
Manual use::

    python ai_upscale.py --input photo.jpg --output photo_2048.png --target 2048
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
MODEL_SHA256 = "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1"
MODEL_PATH = ROOT / "models" / "upscale" / "RealESRGAN_x4plus.pth"

# Inputs closer than this to the target are only resized, not AI-upscaled.
MIN_UPSCALE_FACTOR = 1.05
TILE = 512
TILE_PAD = 16


def status() -> dict:
    ready = MODEL_PATH.is_file() and importlib.util.find_spec("spandrel") is not None
    return {
        "ready": ready,
        "message": "KI-Upscaler Real-ESRGAN ist bereit" if ready else
        "KI-Upscaler fehlt. Einrichten: setup-upscale.ps1 ausführen.",
    }


def download_model() -> Path:
    if MODEL_PATH.is_file():
        return MODEL_PATH
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    partial = MODEL_PATH.with_suffix(".partial")
    digest = hashlib.sha256()
    with urllib.request.urlopen(MODEL_URL) as response, partial.open("wb") as file:
        while chunk := response.read(1 << 20):
            digest.update(chunk)
            file.write(chunk)
    if digest.hexdigest() != MODEL_SHA256:
        partial.unlink()
        raise RuntimeError(f"Prüfsumme des Upscaler-Modells stimmt nicht: {digest.hexdigest()}")
    partial.replace(MODEL_PATH)
    return MODEL_PATH


def load_model(device: torch.device):
    from spandrel import ImageModelDescriptor, ModelLoader

    if not MODEL_PATH.is_file():
        raise RuntimeError(status()["message"])
    model = ModelLoader(device=device).load_from_file(MODEL_PATH)
    if not isinstance(model, ImageModelDescriptor):
        raise RuntimeError("Das Upscaler-Modell ist kein Bildmodell.")
    model.eval()
    if device.type == "cuda" and model.supports_half:
        model.half()
    return model


@torch.inference_mode()
def upscale(image: np.ndarray, model, device: torch.device) -> np.ndarray:
    """Upscale an HxWx3 uint8 image by the model scale, tile by tile to bound memory."""
    scale = model.scale
    dtype = torch.float16 if device.type == "cuda" and model.supports_half else torch.float32
    source = torch.from_numpy(np.ascontiguousarray(image).copy()).to(device).permute(2, 0, 1)[None].to(dtype) / 255
    _, _, height, width = source.shape
    output = torch.empty((1, 3, height * scale, width * scale), dtype=torch.uint8, device=device)
    for top in range(0, height, TILE):
        for left in range(0, width, TILE):
            bottom, right = min(top + TILE, height), min(left + TILE, width)
            # Pad each tile with context so tile borders do not show.
            pad_top, pad_left = max(top - TILE_PAD, 0), max(left - TILE_PAD, 0)
            pad_bottom, pad_right = min(bottom + TILE_PAD, height), min(right + TILE_PAD, width)
            result = model(source[..., pad_top:pad_bottom, pad_left:pad_right])
            y0, x0 = (top - pad_top) * scale, (left - pad_left) * scale
            output[..., top * scale:bottom * scale, left * scale:right * scale] = (
                result[..., y0:y0 + (bottom - top) * scale, x0:x0 + (right - left) * scale]
                .clamp(0, 1).mul(255).round().to(torch.uint8)
            )
    return output[0].permute(1, 2, 0).cpu().numpy()


def upscale_to(image: np.ndarray, target_long_side: int, device: torch.device, model=None) -> tuple[np.ndarray, float]:
    """Enlarge so the long side reaches target_long_side; returns image and scale factor.

    Images that are already large enough are returned unchanged with factor 1.
    """
    height, width = image.shape[:2]
    factor = target_long_side / max(height, width)
    if factor < MIN_UPSCALE_FACTOR:
        return image, 1.0
    own_model = model is None
    if own_model:
        model = load_model(device)
    try:
        enlarged = image
        while enlarged.shape[0] < height * factor:
            enlarged = upscale(enlarged, model, device)
    finally:
        if own_model:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    size = (round(width * factor), round(height * factor))
    resized = Image.fromarray(enlarged).resize(size, Image.Resampling.LANCZOS)
    return np.asarray(resized), size[0] / width


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--download-model", action="store_true", help="Download the Real-ESRGAN weights and exit.")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--target", type=int, default=2048, help="Long side of the result in pixels.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.download_model:
        print(f"Upscaler-Modell: {download_model()}")
        return 0
    if not args.input or not args.output:
        parser.error("--input and --output are required unless --download-model is given")
    image = np.asarray(Image.open(args.input).convert("RGB"))
    result, factor = upscale_to(image, args.target, torch.device(args.device))
    Image.fromarray(result).save(args.output)
    print(f"{image.shape[1]}x{image.shape[0]} -> {result.shape[1]}x{result.shape[0]} (x{factor:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
