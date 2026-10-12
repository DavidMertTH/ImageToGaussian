"""Extend a photo to a 360 degree ERP panorama with DiT360 (FLUX.1-dev + panorama LoRA).

DiT360 (Insta360 Research, CVPR 2026, code and LoRA MIT) outpaints by RF
inversion: the canvas is inverted, then regenerated with the DiT360 LoRA while
the attention of the known region is tied to the inverted photo
("Personalize Anything"). Its code is cloned to outpaint/vendor/DiT360 by
setup.ps1. Runs in .venv-dit360 (diffusers 0.35.2, transformers 4.52.4).
The model was trained at 2048 x 1024 only.
"""

from __future__ import annotations

import argparse
import secrets
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from common import (MODELS, TOOL_DIR, emit, load_photo, marker_ready, photo_fov, project_photo, quiet, run_cli,
                    save_png, smooth_extension, snapshot, write_marker)

BASE_ID = "black-forest-labs/FLUX.1-dev"
BASE_REVISION = "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
LORA_ID = "Insta360-Research/DiT360-Panorama-Image-Generation"
LORA_REVISION = "88b5915693607d3377e1b5385d7abbe6c671d6ce"
CODE_COMMIT = "3779fe7965473f6824994c663a0ae7a76bc7aafa"
VENDOR = TOOL_DIR / "vendor" / "DiT360"
CACHE = MODELS / "dit360"


def download() -> None:
    emit("download", 0, message="FLUX.1-dev und DiT360-LoRA werden geladen")
    snapshot(BASE_ID, BASE_REVISION, CACHE / "base",
             ["model_index.json", "scheduler/*", "text_encoder/*", "text_encoder_2/*", "tokenizer/*",
              "tokenizer_2/*", "transformer/*", "vae/*"])
    snapshot(LORA_ID, LORA_REVISION, CACHE / "lora", ["adapter_config.json", "adapter_model.safetensors"])
    write_marker(CACHE, baseId=BASE_ID, baseRevision=BASE_REVISION, loraId=LORA_ID, loraRevision=LORA_REVISION,
                 codeCommit=CODE_COMMIT)
    emit("ready", 100, message="DiT360 ist bereit")


def generate(args) -> None:
    import torch
    from scipy.ndimage import binary_erosion

    if not marker_ready(CACHE) or not (VENDOR / "pa_src" / "pipeline.py").is_file():
        raise RuntimeError("DiT360 fehlt. setup.ps1 ausführen.")
    sys.path.insert(0, str(VENDOR))
    from pa_src.attn_processor import PersonalizeAnythingAttnProcessor, set_flux_transformer_attn_processor
    from pa_src.pipeline import RFPanoInversionParallelFluxPipeline

    started = time.perf_counter()
    seed = args.seed if args.seed is not None else secrets.randbelow(2**31)
    fov = args.fov or photo_fov(args.input)[0]
    emit("prepare", 2, message="Foto wird auf die 360°-Kugel projiziert", fov=fov, seed=seed)
    projected, known = project_photo(load_photo(args.input), args.width, args.height, 360, fov)
    canvas = smooth_extension(projected, known)
    save_png(canvas, args.output.with_name(args.output.stem + ".control.png"))
    emit("load", 5, message="FLUX.1-dev und DiT360 werden geladen")
    device = torch.device("cuda")
    dtype = torch.float16
    with quiet():
        pipe = RFPanoInversionParallelFluxPipeline.from_pretrained(str(CACHE / "base"), torch_dtype=dtype,
                                                                   low_cpu_mem_usage=True, local_files_only=True)
        pipe.load_lora_weights(str(CACHE / "lora"), weight_name="adapter_model.safetensors")
        pipe.set_progress_bar_config(disable=True)
        if args.offload == "model":
            pipe.enable_model_cpu_offload()
        elif args.offload == "sequential":
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.to(device)

    latent_h = args.height // (pipe.vae_scale_factor * 2)
    latent_w = args.width // (pipe.vae_scale_factor * 2)
    # 1 = keep (tied to the inverted photo). Erode so the border can blend.
    keep_px = binary_erosion(known, iterations=12)
    keep = np.asarray(Image.fromarray(keep_px.astype(np.uint8) * 255).resize((latent_w, latent_h), Image.NEAREST)) > 127
    mask = torch.from_numpy(keep.astype(np.float32))
    mask = torch.cat([mask[:, 0:1], mask, mask[:, -1:]], dim=-1).view(-1, 1)  # circular padding columns
    image = Image.fromarray(np.rint(canvas).astype(np.uint8))

    emit("invert", 10, message="DiT360 · Bild wird invertiert")
    with torch.inference_mode(), quiet():
        inverted, image_latents, image_ids = pipe.invert(source_prompt="", image=image, height=args.height,
                                                         width=args.width, num_inversion_steps=args.steps, gamma=1.0)
    set_flux_transformer_attn_processor(
        pipe.transformer,
        set_attn_proc_func=lambda name, dh, nh, ap: PersonalizeAnythingAttnProcessor(
            name=name, tau=args.tau / 100, mask=mask, device=device, img_dims=latent_h * (latent_w + 2)))

    def on_step(_pipe, step, _t, kwargs):
        emit("generate", 40 + 55 * (step + 1) / args.steps, message=f"DiT360 · {step + 1}/{args.steps}")
        return kwargs

    prompt = "This is a panorama image. " + (f"The image depicts {args.prompt.strip()}" if args.prompt else "")
    with torch.inference_mode(), quiet():
        result = pipe([prompt, prompt], inverted_latents=inverted, image_latents=image_latents,
                      latent_image_ids=image_ids, height=args.height, width=args.width, start_timestep=0.0,
                      stop_timestep=0.99, num_inference_steps=args.steps, eta=1.0, guidance_scale=args.guidance,
                      generator=torch.Generator(device=device).manual_seed(seed), mask=mask, use_timestep=True,
                      callback_on_step_end=on_step, callback_on_step_end_tensor_inputs=["latents"])
    save_png(np.asarray(result.images[1].convert("RGB")), args.output)
    emit("ready", 100, message="DiT360-Panorama ist fertig", output=str(args.output), seed=seed,
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
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=2.8)
    parser.add_argument("--tau", type=float, default=50, help="0-100; lower keeps the photo more strictly")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--offload", choices=("none", "model", "sequential"), default="none",
                        help="model: GPUs from ~28 GB; sequential: 16 GB cards (very slow)")
    args = parser.parse_args()
    if args.download_models:
        download()
        return
    if not args.input or not args.output:
        parser.error("--input and --output are required")
    generate(args)


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
