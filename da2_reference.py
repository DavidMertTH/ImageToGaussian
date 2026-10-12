"""Predict one seamless DA² distance map for a 180- or 360-degree ERP image.

Run with .venv-depth. DA² (Depth Anything in Any Direction) infers the whole
panorama in a single pass, so its distance has no seams between perspective
views. Its output is scale-invariant: the absolute scale is unknown and is
calibrated later against SHARP (panorama_splats.py --reference-depth).

The saved .npy holds float32 radial distance per ERP pixel at the input
resolution, in the same layout and OpenCV axes as the input (+X right, +Y
down, +Z forward; longitude zero is the image center).

A square 180-degree hemisphere can be inferred in two ways:

direct  DA²'s spherical ray embedding is told the true 180 x 180 degree field
        of view. No invented content is shown to the model.
embed   The hemisphere is pasted into the center of a black 2:1 panorama, which
        matches DA²'s training layout; the black back half is cropped away.

Experimental: DA² was trained on full 360-degree panoramas.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parent
DA2_MODEL = ROOT / "models" / "da2"
PROTOCOL_STDOUT = sys.stdout
# configs/infer.json of EnVision-Research/DA-2 at d659838585f2bc9967c7e9367af573795271dbbb.
DA2_CONFIG: dict[str, Any] = {
    "env": {"seed": 42, "verbose": False},
    "inference": {"min_pixels": 580000, "max_pixels": 620000},
    "spherevit": {
        "vit_w_esphere": {
            "input_dims": [1024, 1024, 1024, 1024],
            "hidden_dim": 512,
            "num_heads": 8,
            "expansion": 4,
            "num_layers_head": [2, 2, 2],
            "dropout": 0.0,
            "layer_scale": 0.0001,
            "out_dim": 64,
            "kernel_size": 3,
            "num_prompt_blocks": 1,
            "use_norm": False,
        },
        "sphere": {"width": 1092, "height": 546, "hfov": 2 * math.pi, "vfov": math.pi},
    },
}


def emit(stage: str, progress: float, **details: Any) -> None:
    """JSON Lines only on stdout, as in the other pipeline scripts."""
    print(json.dumps({"stage": stage, "progressPercent": round(progress * 100), **details}, ensure_ascii=False),
          file=PROTOCOL_STDOUT, flush=True)


def load_model(device):
    import torch

    # Importing the model module only; DA²'s package utilities pull in training tools.
    from da2.model.spherevit import SphereViT

    if not (DA2_MODEL / "model.safetensors").is_file():
        raise FileNotFoundError("DA²-Gewichte fehlen. Zuerst setup-da2.ps1 ausführen.")
    model = SphereViT.from_pretrained(str(DA2_MODEL), config=json.loads(json.dumps(DA2_CONFIG)))
    model = model.to(device).eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return model


def predict_distance(model, image: np.ndarray, hfov: float, vfov: float, pixels: int | None) -> np.ndarray:
    """Run DA² once and return distance at the input resolution."""
    import torch

    height, width = image.shape[:2]
    model.config["spherevit"]["sphere"] = {"width": width, "height": height, "hfov": hfov, "vfov": vfov}
    if pixels:
        model.config["inference"]["min_pixels"] = model.config["inference"]["max_pixels"] = pixels
    tensor = torch.from_numpy(image).permute(2, 0, 1)[None].float().div(255).to(model.device)
    autocast = torch.autocast("cuda") if model.device.type == "cuda" else torch.autocast("cpu", enabled=False)
    with torch.inference_mode(), autocast:
        distance = model(tensor)
    return distance[0].float().cpu().numpy()


def colorize(distance: np.ndarray) -> Image.Image:
    """Near is bright; inverse distance keeps far backgrounds readable."""
    inverse = 1 / np.maximum(distance, 1e-6)
    low, high = np.quantile(inverse, [0.02, 0.98])
    value = np.clip((inverse - low) / max(high - low, 1e-8), 0, 1)
    return Image.fromarray((value * 255).astype(np.uint8))


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    started = time.monotonic()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    emit("DA²-Panorama wird vorbereitet", 0)
    with Image.open(args.input) as source:
        image = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
    height, width = image.shape[:2]
    expected_ratio = args.coverage / 180
    if abs(width / height - expected_ratio) > 0.02:
        raise ValueError(f"Für {args.coverage}° ist ein equirectangulares Bild im Format {expected_ratio:g}:1 nötig.")
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    emit("DA²-Modell wird geladen", 0.1, device=str(device))
    model = load_model(device)
    emit("DA² schätzt die Entfernung", 0.4)
    if args.coverage == 360:
        distance = predict_distance(model, image, 2 * math.pi, math.pi, args.pixels)
    elif args.hemisphere == "direct":
        distance = predict_distance(model, image, math.pi, math.pi, args.pixels)
    else:
        canvas = np.zeros((height, 2 * height, 3), dtype=np.uint8)
        left = height // 2
        canvas[:, left:left + width] = image
        distance = predict_distance(model, canvas, 2 * math.pi, math.pi, args.pixels)[:, left:left + width]
    distance = np.ascontiguousarray(distance, dtype=np.float32)
    if not np.isfinite(distance).all() or (distance <= 0).any():
        raise RuntimeError("DA² hat ungültige Entfernungen geliefert.")
    temporary = output.with_name(output.stem + ".partial.npy")
    np.save(temporary, distance)
    temporary.replace(output)
    colorize(distance).save(output.with_suffix(".png"))
    metadata = {
        "model": "haodongli/DA-2",
        "coverageDegrees": args.coverage,
        "hemisphereMode": args.hemisphere if args.coverage == 180 else None,
        "shape": list(distance.shape),
        "distanceQuantiles": np.quantile(distance, [0.02, 0.5, 0.98]).round(4).tolist(),
        "scale": "scale-invariant radial distance",
        "durationSeconds": round(time.monotonic() - started, 2),
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    emit("DA²-Entfernung ist fertig", 1, status="ready", output=str(output), **metadata)
    return metadata


def main() -> int:
    if hasattr(PROTOCOL_STDOUT, "reconfigure"):
        PROTOCOL_STDOUT.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="Square 180-degree or 2:1 360-degree ERP image.")
    parser.add_argument("--output", type=Path, required=True, help="Output .npy distance map.")
    parser.add_argument("--coverage", type=int, choices=(180, 360), default=180)
    parser.add_argument("--hemisphere", choices=("direct", "embed"), default="direct",
                        help="How a 180-degree hemisphere is shown to DA².")
    parser.add_argument("--pixels", type=int, help="Network input pixel count; DA² default is about 600000.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".npy":
        parser.error("--output must end in .npy")
    try:
        run(args)
        return 0
    except Exception as error:
        emit("DA²-Verarbeitung fehlgeschlagen", 1, status="error", error=str(error))
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
