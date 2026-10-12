"""Extend a photo to a 360 degree ERP panorama with Qwen-Image-2512 + Qwen 360 Diffusion.

The 360 LoRA (ProGamerGov/qwen-360-diffusion, MIT, trained on ~35k ERP images)
is a text-to-panorama LoRA. Here it runs inside QwenImageInpaintPipeline: the
photo is projected into the ERP canvas, the rest is masked and generated from
the scene description. Runs in .venv-outpaint (diffusers 0.41). Base model:
Apache 2.0.
"""

from __future__ import annotations

import argparse
import secrets
import time
from pathlib import Path

import numpy as np
from PIL import Image

from common import (MODELS, emit, load_photo, marker_ready, photo_fov, project_photo, quiet, run_cli, save_png,
                    smooth_extension, snapshot, write_marker)

MODEL_ID = "Qwen/Qwen-Image-2512"
MODEL_REVISION = "25468b98e3276ca6700de15c6628e51b7de54a26"
LORA_ID = "ProGamerGov/qwen-360-diffusion"
LORA_REVISION = "f63e5de60bd27b7d9af9c64f8c591381775ea9aa"
LORA_FILE = "qwen-360-diffusion-2512-int8-bf16-v2.safetensors"
CACHE = MODELS / "qwen360"
TRIGGER = "equirectangular 360 degree panorama with equirectangular projection"
NEGATIVE = "blurry, distorted horizon, seam, border, frame, text, watermark"


def download() -> None:
    emit("download", 0, message="Qwen-Image-2512 und 360°-LoRA werden geladen")
    snapshot(MODEL_ID, MODEL_REVISION, CACHE / "base",
             ["model_index.json", "scheduler/*", "text_encoder/*", "tokenizer/*", "transformer/*", "vae/*"])
    snapshot(LORA_ID, LORA_REVISION, CACHE / "lora", [LORA_FILE])
    write_marker(CACHE, modelId=MODEL_ID, modelRevision=MODEL_REVISION, loraId=LORA_ID, loraRevision=LORA_REVISION)
    emit("ready", 100, message="Qwen-Image ist bereit")


def generate(args) -> None:
    import torch
    from diffusers import QwenImageInpaintPipeline
    from scipy.ndimage import binary_erosion

    if not marker_ready(CACHE):
        raise RuntimeError("Qwen-Image fehlt. setup.ps1 ausführen.")
    started = time.perf_counter()
    seed = args.seed if args.seed is not None else secrets.randbelow(2**31)
    fov = args.fov or photo_fov(args.input)[0]
    emit("prepare", 2, message="Foto wird auf die 360°-Kugel projiziert", fov=fov, seed=seed)
    projected, known = project_photo(load_photo(args.input), args.width, args.height, 360, fov)
    canvas = smooth_extension(projected, known)
    # Let the model repaint a thin border of the photo so it can blend into it.
    keep = binary_erosion(known, iterations=8)
    save_png(canvas, args.output.with_name(args.output.stem + ".control.png"))
    emit("load", 5, message="Qwen-Image-2512 wird geladen")
    with quiet():
        pipe = QwenImageInpaintPipeline.from_pretrained(str(CACHE / "base"), torch_dtype=torch.bfloat16,
                                                        local_files_only=True)
        pipe.load_lora_weights(str(CACHE / "lora"), weight_name=LORA_FILE, adapter_name="pano")
        pipe.set_adapters(["pano"], adapter_weights=[args.lora_strength])
        pipe.vae.enable_tiling()
        pipe.set_progress_bar_config(disable=True)
        if args.offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")

    def on_step(_pipe, step, _t, kwargs):
        emit("generate", 10 + 85 * (step + 1) / args.steps, message=f"Qwen-Image · {step + 1}/{args.steps}")
        return kwargs

    prompt = f"{TRIGGER}, {args.prompt.strip()}" if args.prompt else TRIGGER
    with torch.inference_mode(), quiet():
        image = pipe(prompt=prompt, negative_prompt=NEGATIVE, true_cfg_scale=args.cfg,
                     image=Image.fromarray(np.rint(canvas).astype(np.uint8)),
                     mask_image=Image.fromarray((~keep).astype(np.uint8) * 255),
                     width=args.width, height=args.height, strength=1.0, num_inference_steps=args.steps,
                     generator=torch.Generator("cuda").manual_seed(seed),
                     callback_on_step_end=on_step, callback_on_step_end_tensor_inputs=[]).images[0]
    save_png(np.asarray(image.convert("RGB")), args.output)
    emit("ready", 100, message="Qwen-Panorama ist fertig", output=str(args.output), seed=seed,
         durationSeconds=round(time.perf_counter() - started, 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, help="2:1 ERP PNG")
    parser.add_argument("--download-models", action="store_true")
    parser.add_argument("--fov", type=float, help="Horizontal FOV of the photo; default from EXIF or 65")
    parser.add_argument("--prompt", default="", help="Scene description (from outpaint/caption.py)")
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--cfg", type=float, default=4.0)
    parser.add_argument("--lora-strength", type=float, default=1.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--offload", action="store_true", help="CPU offload for GPUs below ~64 GB")
    args = parser.parse_args()
    if args.download_models:
        download()
        return
    if not args.input or not args.output:
        parser.error("--input and --output are required")
    generate(args)


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
