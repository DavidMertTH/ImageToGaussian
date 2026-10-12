"""Predict a SHARP Gaussian scene from one photo at a selectable resolution.

SHARP was trained at 1536 x 1536. Higher values produce more and finer
Gaussians (2048: 2.1 M instead of 1.2 M) and help when the photo itself has
more detail than 1536 px. With ``--upscale auto`` smaller photos are first
enlarged with the Real-ESRGAN AI upscaler instead of plain bilinear resizing.

Example (use the project's Python environment)::

    python sharp_predict.py --input photo.jpg --output scene.ply --resolution 2048 --upscale auto

Progress is emitted as one UTF-8 JSON object per stdout line.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from PIL import Image

import ai_upscale
from sharp.cli.predict import (
    DEFAULT_INTERNAL_RESOLUTION,
    DEFAULT_MODEL_URL,
    predict_image,
    validate_internal_resolution,
)
from sharp.models import PredictorParams, create_predictor
from sharp.utils import io
from sharp.utils.gaussians import save_ply


def emit(stage: str, **details: Any) -> None:
    if "progress" in details and "progressPercent" not in details:
        details["progressPercent"] = round(float(details["progress"]) * 100)
    print(json.dumps({"stage": stage, **details}, ensure_ascii=False), flush=True)


def load_predictor(device: torch.device):
    state_dict = torch.hub.load_state_dict_from_url(
        DEFAULT_MODEL_URL, map_location="cpu", progress=False, weights_only=True
    )
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    return predictor.eval().to(device)


def run(args: argparse.Namespace) -> None:
    started = time.monotonic()
    device = torch.device(args.device)
    emit("Bild wird vorbereitet", progress=0, status="processing")
    image, _, f_px = io.load_rgb(args.input)
    source_height, source_width = image.shape[:2]
    upscale_factor = 1.0
    if args.upscale == "auto" and args.resolution / max(source_width, source_height) >= ai_upscale.MIN_UPSCALE_FACTOR:
        emit("KI-Upscaling läuft", progress=0.05)
        image, upscale_factor = ai_upscale.upscale_to(image, args.resolution, device)
        # The focal length in pixels grows with the image.
        f_px *= upscale_factor
        if args.save_upscaled:
            Image.fromarray(image).save(args.save_upscaled)
    height, width = image.shape[:2]
    emit("SHARP-Modell wird geladen", progress=0.25)
    predictor = load_predictor(device)
    emit(f"3D-Szene wird berechnet ({args.resolution} px)", progress=0.35)
    try:
        gaussians = predict_image(predictor, image, f_px, device, args.resolution)
    except torch.OutOfMemoryError as error:
        raise RuntimeError(
            f"Nicht genügend GPU-Speicher für {args.resolution} px. "
            "Eine kleinere SHARP-Auflösung wählen oder andere GPU-Anwendungen schließen."
        ) from error
    emit("Szene wird gespeichert", progress=0.9)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_ply(gaussians, f_px, (height, width), args.output)
    emit(
        "Fertig",
        status="ready",
        progress=1,
        output=str(args.output),
        resolution=args.resolution,
        upscaleFactor=round(upscale_factor, 3),
        sourceSize=[source_width, source_height],
        gaussianCount=int(gaussians.mean_vectors.shape[1]),
        durationSeconds=round(time.monotonic() - started, 1),
    )


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output .ply file.")
    parser.add_argument("--resolution", type=int, default=DEFAULT_INTERNAL_RESOLUTION,
                        help="Square SHARP network resolution (multiple of 64, 768-4096; trained: 1536).")
    parser.add_argument("--upscale", choices=("off", "auto"), default="off",
                        help="auto: AI-upscale photos smaller than --resolution before SHARP.")
    parser.add_argument("--save-upscaled", type=Path, help="Also save the upscaled image.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    try:
        validate_internal_resolution(args.resolution)
    except ValueError as error:
        parser.error(str(error))
    try:
        run(args)
        return 0
    except Exception as error:
        emit("SHARP-Verarbeitung fehlgeschlagen", status="error", error=str(error))
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
