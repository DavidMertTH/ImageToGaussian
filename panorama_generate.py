"""Generate a 360 degree ERP panorama from a perspective photo.

Run with the isolated .venv-panorama interpreter. Models are downloaded only by
--download-models; normal image jobs use the local cache exclusively. No model
repository Python code is executed.

The source photo conditions generation. By default the generated reference
region is kept so that lighting and colors remain consistent across the entire
panorama. --preserve-original restores projected source pixels with a soft edge,
which can introduce visible brightness or geometry differences at that border.
The unmodified generated RGB image is always saved as <output>.generated.png.

The LoRA requires the BASE Klein model and a green, 2:1 ERP reference canvas:
https://huggingface.co/nomadoor/flux-2-klein-4B-360-erp-outpaint-lora
https://huggingface.co/docs/diffusers/v0.37.1/en/api/pipelines/flux2
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import os
from pathlib import Path
import secrets
import sys
import time


ROOT = Path(__file__).resolve().parent
PROTOCOL_STDOUT = sys.stdout
# 4B fits 16 GB cards with offloading; 9B (same LoRA recipe) needs a large GPU.
VARIANTS = {
    "4b": {
        "modelId": "black-forest-labs/FLUX.2-klein-base-4B",
        "modelRevision": "a3b4f4849157f664bdbc776fd7453c2783562f4d",
        "loraId": "nomadoor/flux-2-klein-4B-360-erp-outpaint-lora",
        "loraRevision": "124129f3ebce357c007ffca177d19567246f8bff",
        "loraFilename": "flux-2-klein-4B-360-erp-outpaint-lora_V1.safetensors",
        "cacheDir": ROOT / "models" / "panorama",
    },
    "9b": {
        "modelId": "black-forest-labs/FLUX.2-klein-base-9B",
        "modelRevision": "32773329fbe7e81a90ef971740e8ba4b0364ecf3",
        "loraId": "nomadoor/flux-2-klein-9B-360-erp-outpaint-lora",
        "loraRevision": "64b1a5d1e2dea2d9173a5ea46e09645672e2acfb",
        "loraFilename": "flux-2-klein-9B-360-erp-outpaint-lora_V1.safetensors",
        "cacheDir": ROOT / "models" / "panorama-9b",
    },
}
MODEL_ID = MODEL_REVISION = LORA_ID = LORA_REVISION = LORA_FILENAME = None


def select_variant(name: str) -> None:
    global MODEL_ID, MODEL_REVISION, LORA_ID, LORA_REVISION, LORA_FILENAME
    variant = VARIANTS[name]
    MODEL_ID, MODEL_REVISION = variant["modelId"], variant["modelRevision"]
    LORA_ID, LORA_REVISION, LORA_FILENAME = variant["loraId"], variant["loraRevision"], variant["loraFilename"]


select_variant("4b")
TRIGGER = (
    "Fill the green spaces according to the image. "
    "Outpaint as a seamless 360 equirectangular panorama (2:1). "
    "Keep the horizon level. Match left and right edges."
)
MODEL_FILES = [
    "model_index.json", "scheduler/*", "text_encoder/*.json",
    "text_encoder/*.safetensors", "tokenizer/*", "transformer/*.json",
    "transformer/*.safetensors", "vae/*.json", "vae/*.safetensors",
]


def emit(stage: str, progress: float, **fields) -> None:
    print(json.dumps({
        "stage": stage,
        "progressPercent": round(max(0.0, min(100.0, progress)), 1),
        **fields,
    }, ensure_ascii=False), file=PROTOCOL_STDOUT, flush=True)


def load_runtime():
    try:
        # Keep dependency logs on stderr; stdout is a JSON-lines protocol.
        with contextlib.redirect_stdout(sys.stderr):
            import numpy as np
            import torch
            import diffusers
            import transformers
            import peft
            from PIL import Image, ImageOps
            from diffusers import Flux2KleinPipeline
            from huggingface_hub import hf_hub_download, snapshot_download
            from safetensors import safe_open
    except Exception as exc:
        raise RuntimeError(
            "Panorama-Pakete fehlen oder sind inkompatibel. Bitte setup-panorama.ps1 -FillTool klein-lora "
            "ausführen; requirements-panorama.txt gehört in .venv-panorama. "
            f"({type(exc).__name__}: {exc})"
        ) from exc
    if int(transformers.__version__.split(".", 1)[0]) >= 5:
        raise RuntimeError("Diese Panorama-Pipeline benötigt transformers==4.57.6.")
    return {
        "np": np, "torch": torch, "Image": Image, "ImageOps": ImageOps,
        "Pipeline": Flux2KleinPipeline, "snapshot_download": snapshot_download,
        "hf_hub_download": hf_hub_download, "safe_open": safe_open,
        "versions": {
            "torch": torch.__version__, "diffusers": diffusers.__version__,
            "transformers": transformers.__version__, "peft": peft.__version__,
        },
    }


def resolve_weights(cache_dir: Path, runtime: dict, download: bool = False):
    """Use local folders, avoiding Windows symlink privileges and network in jobs."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / "base"
    lora_directory = cache_dir / "lora"
    lora_path = lora_directory / LORA_FILENAME
    try:
        if download:
            with contextlib.redirect_stdout(sys.stderr):
                runtime["snapshot_download"](
                    repo_id=MODEL_ID, revision=MODEL_REVISION,
                    local_dir=str(model_path), allow_patterns=MODEL_FILES,
                    max_workers=2,
                )
                runtime["hf_hub_download"](
                    repo_id=LORA_ID, filename=LORA_FILENAME,
                    revision=LORA_REVISION, local_dir=str(lora_directory),
                )
    except Exception as exc:
        action = "Download fehlgeschlagen" if download else "Modellgewichte fehlen im lokalen Cache"
        raise RuntimeError(
            f"{action}. Bitte setup-panorama.ps1 -FillTool klein-lora ausführen oder "
            "'.venv-panorama\\Scripts\\python.exe panorama_generate.py --download-models' "
            f"starten. ({type(exc).__name__}: {exc})"
        ) from exc
    required = [
        "model_index.json", "scheduler/scheduler_config.json",
        "text_encoder/config.json", "text_encoder/model.safetensors.index.json",
        "tokenizer/tokenizer.json", "transformer/config.json", "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    ]
    missing = [name for name in required if not (model_path / name).is_file()]
    if not missing:
        # The 4B transformer is one file, the 9B transformer is sharded.
        for folder, single in (("text_encoder", "model.safetensors"), ("transformer", "diffusion_pytorch_model.safetensors")):
            indexes = list((model_path / folder).glob("*.safetensors.index.json"))
            if not indexes:
                if not (model_path / folder / single).is_file():
                    missing.append(f"{folder}/{single}")
                continue
            shards = set(json.loads(indexes[0].read_text()).get("weight_map", {}).values())
            missing.extend(f"{folder}/{name}" for name in shards if not (model_path / folder / name).is_file())
    if not lora_path.is_file():
        missing.append(f"lora/{LORA_FILENAME}")
    if missing:
        raise RuntimeError("Modell-Download unvollständig: " + ", ".join(missing) + ". setup-panorama.ps1 -FillTool klein-lora erneut ausführen.")
    return model_path, lora_path


def download_models(args, runtime: dict) -> None:
    emit("download", 0, message=f"FLUX.2 Klein Base {args.model.upper()} und Panorama-LoRA werden geladen")
    model_path, lora_path = resolve_weights(args.cache_dir, runtime, download=True)
    emit("validate", 95, message="Modelldateien werden geprüft")
    # Opening headers validates safetensors without allocating the full models.
    weight_files = list(model_path.glob("*/*.safetensors")) + [lora_path]
    for path in weight_files:
        with runtime["safe_open"](str(path), framework="pt", device="cpu") as handle:
            if not handle.keys():
                raise RuntimeError(f"Leere Modelldatei: {path.name}")
    marker = {
        "modelId": MODEL_ID, "modelRevision": MODEL_REVISION,
        "loraId": LORA_ID, "loraRevision": LORA_REVISION,
        "loraFilename": LORA_FILENAME,
        "modelPath": str(model_path), "loraPath": str(lora_path),
        "versions": runtime["versions"], "ready": True,
    }
    marker_path = args.cache_dir / "ready.json"
    temporary = marker_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(marker, indent=2), encoding="utf-8")
    temporary.replace(marker_path)
    emit("ready", 100, message="Panorama-Modelle sind lokal verfügbar", cacheDir=str(args.cache_dir))


def project_reference(image, width: int, height: int, hfov: float, np):
    """Inverse-project ERP rays into a pinhole photo with square pixel intrinsics.

    ERP center is yaw 0 (forward), latitude 0 (horizon). This is a geometrically
    projected reference, not a rectangle stretched over longitude/latitude.
    """
    source = np.asarray(image, dtype=np.uint8)
    ih, iw = source.shape[:2]
    focal = iw / (2.0 * math.tan(math.radians(hfov) / 2.0))
    longitude = ((np.arange(width, dtype=np.float32) + 0.5) / width - 0.5) * (2 * math.pi)
    dx = np.sin(longitude)[None, :]
    dz = np.cos(longitude)[None, :]
    projected = np.zeros((height, width, 3), dtype=np.uint8)
    alpha = np.zeros((height, width), dtype=np.float32)
    valid_mask = np.zeros((height, width), dtype=bool)
    # Feather into the generated pixels only; never mix green into the reference.
    feather = max(2.0, min(iw, ih) * 0.025)
    for row in range(0, height, 128):
        stop = min(row + 128, height)
        latitude = (0.5 - (np.arange(row, stop, dtype=np.float32) + 0.5) / height) * math.pi
        cos_lat = np.cos(latitude)[:, None]
        z = dz * cos_lat
        safe_z = np.where(z > 1e-6, z, 1.0)
        u = focal * (dx * cos_lat) / safe_z + (iw - 1) / 2.0
        v = focal * (-np.sin(latitude)[:, None]) / safe_z + (ih - 1) / 2.0
        valid = (z > 1e-6) & (u >= 0) & (u <= iw - 1) & (v >= 0) & (v <= ih - 1)
        x = np.clip(u, 0, iw - 1)
        y = np.clip(v, 0, ih - 1)
        x0, y0 = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
        x1, y1 = np.minimum(x0 + 1, iw - 1), np.minimum(y0 + 1, ih - 1)
        wx, wy = (x - x0)[..., None], (y - y0)[..., None]
        sample = (
            source[y0, x0] * (1 - wx) * (1 - wy)
            + source[y0, x1] * wx * (1 - wy)
            + source[y1, x0] * (1 - wx) * wy
            + source[y1, x1] * wx * wy
        )
        projected[row:stop] = np.where(valid[..., None], np.rint(sample), 0).astype(np.uint8)
        distance = np.minimum(np.minimum(u, iw - 1 - u), np.minimum(v, ih - 1 - v))
        blend = np.clip(distance / feather, 0, 1)
        alpha[row:stop] = np.where(valid, blend * blend * (3 - 2 * blend), 0)
        valid_mask[row:stop] = valid
    control = np.empty_like(projected)
    control[:] = (0, 255, 0)
    control[valid_mask] = projected[valid_mask]
    return control, projected, alpha


def soften_seam(rgb, np, band: int = 16):
    """Small wraparound blend, outside the reference at the front of the panorama."""
    output = rgb.copy()
    band = min(band, rgb.shape[1] // 32)
    for i in range(band):
        weight = 0.5 * (1.0 - i / band) ** 2
        left, right = rgb[:, i].astype(np.float32), rgb[:, -1 - i].astype(np.float32)
        output[:, i] = np.rint(left * (1 - weight) + right * weight).astype(np.uint8)
        output[:, -1 - i] = np.rint(right * (1 - weight) + left * weight).astype(np.uint8)
    return output


def generate(args, runtime: dict) -> None:
    np, torch = runtime["np"], runtime["torch"]
    Image, ImageOps = runtime["Image"], runtime["ImageOps"]
    if not torch.cuda.is_available():
        raise RuntimeError("Für diese Panorama-Pipeline wird CUDA benötigt. Bitte .venv-panorama mit CUDA-PyTorch verwenden.")
    started = time.perf_counter()
    seed = args.seed if args.seed is not None else secrets.randbelow(2**31)
    emit("prepare", 2, message="Foto wird auf die 360°-Kugel projiziert", seed=seed)
    with Image.open(args.input) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
    control, reference, alpha = project_reference(image, args.width, args.height, args.fov, np)
    control_image = Image.fromarray(control)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    control_path = args.output.with_name(args.output.stem + ".control.png")
    control_image.save(control_path)
    model_path, lora_path = resolve_weights(args.cache_dir, runtime)

    emit("load", 5, message="FLUX.2 Klein und Panorama-LoRA werden geladen")
    with contextlib.redirect_stdout(sys.stderr):
        pipe = runtime["Pipeline"].from_pretrained(
            str(model_path), torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, local_files_only=True, use_safetensors=True,
        )
        pipe.load_lora_weights(
            str(lora_path.parent), weight_name=lora_path.name,
            adapter_name="panorama", local_files_only=True, use_safetensors=True,
        )
        pipe.set_adapters(["panorama"], adapter_weights=[args.lora_strength])
        pipe.vae.enable_tiling()
        pipe.set_progress_bar_config(disable=True)
        if args.offload == "none":
            pipe.to("cuda")
        elif args.offload == "model":
            pipe.enable_model_cpu_offload(device="cuda")
        else:
            # Layer-level transfers make the default suitable for 16 GB cards
            # while another application may already occupy several GB of VRAM.
            pipe.enable_sequential_cpu_offload(device="cuda")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    emit("generate", 12, message="Fehlende Umgebung wird in 360° ergänzt", step=0, totalSteps=args.steps)

    def on_step(_pipeline, step, _timestep, callback_kwargs):
        emit("generate", 12 + 81 * ((step + 1) / args.steps),
             message=f"Panorama wird erzeugt · {step + 1}/{args.steps}",
             step=step + 1, totalSteps=args.steps)
        return callback_kwargs

    prompt = TRIGGER
    if args.prompt:
        prompt += " " + args.prompt.strip()
    prompt += " Preserve the reference scene. Remove all green placeholder areas."
    try:
        with torch.inference_mode(), contextlib.redirect_stdout(sys.stderr):
            result = pipe(
                prompt=prompt, image=control_image,
                width=args.width, height=args.height,
                num_inference_steps=args.steps, guidance_scale=args.guidance,
                generator=torch.Generator(device="cuda").manual_seed(seed),
                callback_on_step_end=on_step,
                callback_on_step_end_tensor_inputs=[],
            )
    except torch.cuda.OutOfMemoryError as exc:
        extra = " --offload sequential verwenden." if args.offload == "model" else " Andere GPU-Anwendungen schließen oder --width 1536 --height 768 verwenden."
        raise RuntimeError("Nicht genügend GPU-Speicher für die Panorama-Erzeugung." + extra) from exc
    final_message = (
        "Originalausschnitt wird zurückgesetzt und Übergänge werden geglättet"
        if args.preserve_original else "Panorama-Übergang wird geglättet"
    )
    emit("finalize", 95, message=final_message)
    generated = result.images[0].convert("RGB")
    generated_path = args.output.with_name(args.output.stem + ".generated.png")
    generated.save(generated_path)
    if generated.size != (args.width, args.height):
        generated = generated.resize((args.width, args.height), Image.Resampling.LANCZOS)
    if args.preserve_original:
        pixels = np.asarray(generated, dtype=np.float32)
        blend = alpha[..., None]
        combined = np.rint(pixels * (1 - blend) + reference.astype(np.float32) * blend).clip(0, 255).astype(np.uint8)
    else:
        combined = np.asarray(generated, dtype=np.uint8)
    combined = soften_seam(combined, np)
    Image.fromarray(combined).save(args.output)
    duration = time.perf_counter() - started
    metadata = {
        "projection": "equirectangular", "width": args.width, "height": args.height,
        "sourceHorizontalFov": args.fov, "sourceYaw": 0, "sourcePitch": 0,
        "seed": seed, "steps": args.steps, "guidance": args.guidance,
        "loraStrength": args.lora_strength, "offload": args.offload,
        "originalPreserved": args.preserve_original,
        "generatedPath": str(generated_path),
        "modelId": MODEL_ID, "modelRevision": MODEL_REVISION,
        "loraId": LORA_ID, "loraRevision": LORA_REVISION,
        "durationSeconds": round(duration, 3),
        "peakCudaMemoryGB": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "versions": runtime["versions"],
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    emit("ready", 100, message="360°-Panorama ist fertig", output=str(args.output),
         durationSeconds=round(duration, 1), seed=seed,
         peakCudaMemoryGB=metadata["peakCudaMemoryGB"])
    del pipe, result
    gc.collect()
    torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Perspective JPG/PNG/WebP photo")
    parser.add_argument("--output", type=Path, help="Output panorama PNG")
    parser.add_argument("--download-models", action="store_true", help="Only download models and write cache ready.json")
    parser.add_argument("--model", choices=tuple(VARIANTS), default="4b", help="FLUX.2 Klein base size")
    parser.add_argument("--cache-dir", type=Path, help="Default: models/panorama (4b) or models/panorama-9b (9b)")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--fov", type=float, default=65.0, help="Assumed source horizontal FOV in degrees")
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--lora-strength", type=float, default=0.9)
    parser.add_argument("--offload", choices=("sequential", "model", "none"), default="sequential")
    parser.add_argument("--prompt", default="", help="Optional additional panorama description")
    parser.add_argument("--preserve-original", action="store_true", help="Restore projected source pixels; may create visible lighting differences")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    select_variant(args.model)
    args.cache_dir = (args.cache_dir or VARIANTS[args.model]["cacheDir"]).resolve()
    if not args.download_models:
        if args.input is None or args.output is None:
            parser.error("--input and --output are required unless --download-models is used")
        args.input, args.output = args.input.resolve(), args.output.resolve()
        if not args.input.is_file():
            parser.error(f"Input image does not exist: {args.input}")
        if args.output == args.input:
            parser.error("Output must be different from the original input photo")
        if args.output.suffix.lower() != ".png":
            parser.error("Output panorama must use a .png extension")
        if args.width != 2 * args.height or args.height < 256 or args.width > 4096:
            parser.error("ERP dimensions must be 2:1, height >= 256, width <= 4096")
        if args.width % 32 or args.height % 32:
            parser.error("ERP width and height must be multiples of 32")
        if not 1 <= args.steps <= 60:
            parser.error("--steps must be between 1 and 60")
        if not 10 <= args.fov <= 160:
            parser.error("--fov must be between 10 and 160 degrees")
        if not 1 <= args.guidance <= 20 or not 0 <= args.lora_strength <= 2:
            parser.error("Guidance must be 1..20 and LoRA strength 0..2")
        if args.seed is not None and not 0 <= args.seed < 2**63:
            parser.error("Seed must be in 0..2^63-1")
    return args


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    # Keep HF progress on stderr and memory allocation fragmentation down.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    try:
        runtime = load_runtime()
        if args.download_models:
            download_models(args, runtime)
        else:
            generate(args, runtime)
    except Exception as exc:
        emit("error", 0, message=str(exc)[:2000], error=str(exc)[:2000], exception=type(exc).__name__)
        if args.debug:
            import traceback
            traceback.print_exc(file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
