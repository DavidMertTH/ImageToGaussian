"""Outpaint a photo to a square 180 x 180 degree front-hemisphere ERP image.

FLUX.1 Fill NF4 generates overlapping perspective patches, which are projected
back onto the front hemisphere. Original pixels and earlier patches are retained
except for a narrow cross-fade band that hides the seams between them.
Weights are downloaded only with --download-models, never during image jobs.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
from pathlib import Path
import secrets
import sys
import time

from hemisphere_geometry import (
    FILL_VIEWS,
    PATCH_FOV,
    camera_rotation,
    erp_rays,
    pinhole_rays,
    sample_erp,
    sample_pinhole,
)
from panorama_generate import emit

ROOT = Path(__file__).resolve().parent
MODEL_ID = "black-forest-labs/FLUX.1-Fill-dev"
MODEL_REVISION = "358293da0354175698b67ec8299acf928313a78a"
QUANT_ID = "diffusers/FLUX.1-Fill-dev-nf4"
QUANT_REVISION = "affe813a66938ecfcf20d0ba96007c14ee328537"
# Do not name object classes here: "architecture" made FLUX add buildings to beaches.
PROMPT = "Extend this photographic scene naturally. Match its lighting, colors and perspective. Continue only what plausibly surrounds the existing image."


def load_runtime():
    try:
        with contextlib.redirect_stdout(sys.stderr):
            import numpy as np
            import torch
            import bitsandbytes
            from PIL import Image, ImageOps
            from diffusers import FluxFillPipeline, FluxTransformer2DModel
            from transformers import T5EncoderModel
            from huggingface_hub import snapshot_download
            from safetensors import safe_open
    except Exception as exc:
        raise RuntimeError(
            "FLUX-Fill-Pakete fehlen oder sind inkompatibel. "
            "Einmal setup-panorama.ps1 -FillTool flux-fill ausführen. "
            f"({type(exc).__name__}: {exc})"
        ) from exc
    return {
        "np": np, "torch": torch, "Image": Image, "ImageOps": ImageOps,
        "FluxFillPipeline": FluxFillPipeline, "FluxTransformer2DModel": FluxTransformer2DModel,
        "T5EncoderModel": T5EncoderModel, "snapshot_download": snapshot_download,
        "safe_open": safe_open,
    }


def resolve_weights(cache_dir: Path, runtime: dict, download: bool = False, precision: str = "nf4"):
    """Resolve only pinned local components; download requires explicit setup.

    nf4 uses the 4-bit transformer and T5 (16 GB cards); bf16 uses the full
    FLUX.1 Fill weights from the base repository (about 24 GB more, 40 GB+ cards).
    """
    base, quant = cache_dir / "base", cache_dir / "nf4"
    if download and precision == "bf16":
        try:
            with contextlib.redirect_stdout(sys.stderr):
                runtime["snapshot_download"](
                    MODEL_ID, revision=MODEL_REVISION, local_dir=str(base), max_workers=4,
                    allow_patterns=["model_index.json", "scheduler/*", "text_encoder/*", "text_encoder_2/*",
                                    "tokenizer/*", "tokenizer_2/*", "transformer/*", "vae/*"],
                )
        except Exception as exc:
            raise RuntimeError(
                "FLUX-Fill-Download fehlgeschlagen. Zugriff auf https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev "
                f"prüfen und 'hf auth login' ausführen. ({type(exc).__name__})"
            ) from exc
    elif download:
        try:
            with contextlib.redirect_stdout(sys.stderr):
                # Check gated access before downloading the large NF4 weights.
                runtime["snapshot_download"](
                    MODEL_ID, revision=MODEL_REVISION, local_dir=str(base), max_workers=2,
                    allow_patterns=["model_index.json", "scheduler/*", "text_encoder/*.json", "text_encoder/*.safetensors", "tokenizer/*", "tokenizer_2/*", "vae/*.json", "vae/*.safetensors"],
                )
                runtime["snapshot_download"](
                    QUANT_ID, revision=QUANT_REVISION, local_dir=str(quant), max_workers=2,
                    allow_patterns=["transformer/*.json", "transformer/*.safetensors", "text_encoder_2/*.json", "text_encoder_2/*.safetensors"],
                )
        except Exception as exc:
            raise RuntimeError(
                "FLUX-Fill-Download fehlgeschlagen. Bei fehlendem Zugriff zuerst auf "
                "https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev die Modellbedingungen "
                "annehmen und '.venv-panorama\\Scripts\\hf.exe auth login' ausführen. "
                f"Danach setup-panorama.ps1 -FillTool flux-fill erneut starten. ({type(exc).__name__})"
            ) from exc
    required = [base / "model_index.json", base / "scheduler/scheduler_config.json", base / "tokenizer/tokenizer_config.json", base / "tokenizer_2/tokenizer_config.json"]
    weights = (base / "transformer", base / "text_encoder_2") if precision == "bf16" else (quant / "transformer", quant / "text_encoder_2")
    for directory in (base / "text_encoder", base / "vae", *weights):
        required.append(directory / "config.json")
        files = list(directory.glob("*.safetensors"))
        if not files:
            required.append(directory / "model.safetensors")
        for index in directory.glob("*.safetensors.index.json"):
            mapping = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
            if not mapping:
                raise RuntimeError(f"Ungültiger Modellindex: {index.name}")
            required.extend(directory / name for name in set(mapping.values()))
    missing = [str(path.relative_to(cache_dir)) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("FLUX-Fill-Modelle fehlen: " + ", ".join(missing) + ". Einmal setup-panorama.ps1 -FillTool flux-fill ausführen.")
    return base, quant


def download_models(args, runtime):
    emit("download", 0, message=f"FLUX Fill {args.precision.upper()} für 180° wird heruntergeladen")
    base, quant = resolve_weights(args.cache_dir, runtime, download=True, precision=args.precision)
    emit("validate", 95, message="FLUX-Fill-Dateien werden geprüft")
    for directory in (base, quant):
        if not directory.is_dir():
            continue
        for path in directory.glob("*/*.safetensors"):
            with runtime["safe_open"](str(path), framework="pt", device="cpu") as handle:
                if not handle.keys():
                    raise RuntimeError(f"Leere Modelldatei: {path.name}")
    precisions = set()
    try:
        precisions = set(json.loads((args.cache_dir / "ready.json").read_text(encoding="utf-8")).get("precisions", ["nf4"]))
    except (OSError, ValueError):
        pass
    precisions.add(args.precision)
    marker = {"ready": True, "fillTool": "flux-fill", "coverageDegrees": 180, "precisions": sorted(precisions),
              "modelId": MODEL_ID, "modelRevision": MODEL_REVISION,
              "quantizationId": QUANT_ID, "quantizationRevision": QUANT_REVISION}
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = args.cache_dir / "ready.tmp"
    temporary.write_text(json.dumps(marker, indent=2), encoding="utf-8")
    temporary.replace(args.cache_dir / "ready.json")
    emit("ready", 100, message="FLUX Fill für 180° ist bereit")


def load_pipeline(base, quant, runtime, precision="nf4", offload="model"):
    torch = runtime["torch"]
    if precision == "bf16":
        with contextlib.redirect_stdout(sys.stderr):
            pipe = runtime["FluxFillPipeline"].from_pretrained(
                str(base), torch_dtype=torch.bfloat16, local_files_only=True, use_safetensors=True,
            )
            pipe.to("cuda") if offload == "none" else pipe.enable_model_cpu_offload()
            pipe.set_progress_bar_config(disable=True)
        return pipe
    with contextlib.redirect_stdout(sys.stderr):
        transformer = runtime["FluxTransformer2DModel"].from_pretrained(
            str(quant / "transformer"), torch_dtype=torch.bfloat16,
            local_files_only=True, use_safetensors=True,
        )
        text_encoder = runtime["T5EncoderModel"].from_pretrained(
            str(quant / "text_encoder_2"), torch_dtype=torch.bfloat16,
            local_files_only=True, use_safetensors=True,
        )
        pipe = runtime["FluxFillPipeline"].from_pretrained(
            str(base), transformer=transformer, text_encoder_2=text_encoder,
            torch_dtype=torch.bfloat16, local_files_only=True, use_safetensors=True,
        )
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_tiling()
        pipe.set_progress_bar_config(disable=True)
    return pipe


def blend_weights(missing, original, band, np):
    """Per-pixel weight of the generated patch: 1 in holes, fading out over a band.

    The band reaches into existing content so FLUX can repaint a transition
    instead of butting against a hard edge. It stays narrow over the original
    photo (band // 4) and wider over earlier generated views.
    """
    from scipy.ndimage import distance_transform_edt

    distance = distance_transform_edt(~missing)
    limit = np.where(original, max(band // 4, 1), max(band, 1)).astype(np.float32)
    weight = np.clip(1 - distance / limit, 0, 1)
    weight = weight * weight * (3 - 2 * weight)  # smoothstep
    weight[missing] = 1
    return weight.astype(np.float32)


def match_colors(generated, context, known, np, ring=96, sigma=24.0):
    """Remove FLUX's colour and brightness drift relative to the known context.

    The offset is measured on known pixels near the holes, smoothed locally and
    falls back to the mean offset where no known pixels are nearby.
    """
    from scipy.ndimage import distance_transform_edt, gaussian_filter

    near = known & (distance_transform_edt(known) <= ring)
    if near.sum() < 64:
        return generated
    weight = near.astype(np.float32)
    residual = (context - generated) * weight[..., None]
    global_offset = residual.sum((0, 1)) / weight.sum()
    density = gaussian_filter(weight, sigma)
    local = np.stack([gaussian_filter(residual[..., c], sigma) for c in range(3)], -1)
    local /= np.maximum(density, 1e-6)[..., None]
    confidence = np.clip(density * 4, 0, 1)[..., None]
    offset = confidence * local + (1 - confidence) * global_offset
    return np.clip(generated + offset, 0, 255)


def generate(args, runtime):
    np, torch, Image = runtime["np"], runtime["torch"], runtime["Image"]
    if not torch.cuda.is_available():
        raise RuntimeError("FLUX Fill für 180° benötigt eine CUDA-Grafikkarte.")
    started = time.perf_counter()
    seed = args.seed if args.seed is not None else secrets.randbelow(2**31)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    emit("prepare", 2, message="Foto wird auf die vordere 180°-Halbkugel projiziert")
    with Image.open(args.input) as original:
        source = np.asarray(runtime["ImageOps"].exif_transpose(original).convert("RGB"))
    rays = erp_rays(args.size, args.size)
    projected, known = sample_pinhole(source, rays, args.fov)
    canvas = np.where(known[..., None], projected, 0).astype(np.float32)
    original = known.copy()
    Image.fromarray(np.rint(canvas).astype(np.uint8)).save(args.output.with_name(args.output.stem + ".control.png"))
    base, quant = resolve_weights(args.cache_dir, runtime, precision=args.precision)
    emit("load", 5, message=f"FLUX Fill {args.precision.upper()} wird geladen")
    pipe = load_pipeline(base, quant, runtime, args.precision, args.offload)
    torch.cuda.reset_peak_memory_stats()
    if args.save_steps:
        args.save_steps.mkdir(parents=True, exist_ok=True)

    def save_step(index, name, kind, array, size=None):
        """Keep intermediate images for the studio's pipeline view."""
        if not args.save_steps:
            return
        image = Image.fromarray(np.rint(array).clip(0, 255).astype(np.uint8))
        if size and max(image.size) > size:
            image = image.resize((size, size), runtime["Image"].Resampling.LANCZOS)
        suffix = ".png" if kind == "mask" else ".jpg"
        image.save(args.save_steps / f"{index + 1}-{name}.{kind}{suffix}", quality=92)

    records = []
    for index, (name, yaw, pitch) in enumerate(FILL_VIEWS):
        rotation = camera_rotation(yaw, pitch)
        patch_rays = pinhole_rays(args.patch_size, args.patch_size, PATCH_FOV, rotation)
        patch, inside = sample_erp(canvas, patch_rays)
        coverage, _ = sample_erp(known.astype(np.float32), patch_rays)
        # Also mask directions behind the hemisphere: FLUX output there is
        # discarded, but it keeps the mask interpolation at 1 along the
        # hemisphere edge, which the ERP stretches into large polar gaps.
        missing = (coverage < 0.999) | ~inside
        from_original, _ = sample_erp(original.astype(np.float32), patch_rays)
        emit("generate", 10 + 80 * index / len(FILL_VIEWS), message=f"180° wird ergänzt · {name}", viewIndex=index + 1, viewCount=len(FILL_VIEWS))
        if missing.any():
            # Keep context pixels exact after VAE reconstruction; only holes enter
            # the spherical composite. Pixels outside the hemisphere are ignored.
            patch_image = Image.fromarray(np.rint(patch).clip(0, 255).astype(np.uint8))
            weight = blend_weights(missing, from_original > 0.5, args.blend, np)
            mask = Image.fromarray((weight > 0).astype(np.uint8) * 255)
            save_step(index, name, "input", patch)
            save_step(index, name, "mask", np.asarray(mask))
            def on_step(_pipeline, step, _timestep, callback_kwargs):
                progress = 10 + 80 * (index + (step + 1) / args.steps) / len(FILL_VIEWS)
                emit("generate", progress, message=f"180° · {name} · {step + 1}/{args.steps}")
                return callback_kwargs
            try:
                with torch.inference_mode(), contextlib.redirect_stdout(sys.stderr):
                    result = pipe(
                        prompt=args.prompt or PROMPT, image=patch_image, mask_image=mask,
                        width=args.patch_size, height=args.patch_size,
                        num_inference_steps=args.steps, guidance_scale=args.guidance,
                        generator=torch.Generator("cpu").manual_seed(seed + index),
                        callback_on_step_end=on_step, callback_on_step_end_tensor_inputs=[],
                    ).images[0].convert("RGB")
            except torch.cuda.OutOfMemoryError as exc:
                raise RuntimeError("Nicht genügend GPU-Speicher für FLUX Fill. Andere GPU-Anwendungen schließen oder --patch-size 768 verwenden.") from exc
            generated = match_colors(np.asarray(result, dtype=np.float32), patch, ~missing, np)
            save_step(index, name, "flux", generated)
            back, valid = sample_pinhole(generated, rays, PATCH_FOV, rotation)
            back_weight, _ = sample_pinhole(weight, rays, PATCH_FOV, rotation)
            # Require a fully generated sample so interpolated mask boundaries
            # do not mark unseen directions as filled.
            generated_mask, _ = sample_pinhole(missing.astype(np.float32), rays, PATCH_FOV, rotation)
            write = valid & ~known & (generated_mask > 0.99)
            canvas[write] = back[write]
            # Cross-fade into existing content in ERP space; the known pixels are
            # not resampled, so the original keeps its sharpness outside the band.
            band = valid & known & (back_weight > 0)
            fade = back_weight[band][:, None]
            canvas[band] = canvas[band] * (1 - fade) + back[band] * fade
            known |= write
            save_step(index, name, "canvas", canvas, size=1024)
            records.append({"name": name, "yaw": yaw, "pitch": pitch, "filledPixels": int(write.sum()),
                            "blendedPixels": int(band.sum())})
    if not known.all():
        # Bilinear mask edges may leave a subpixel transition. Fill only these
        # tiny gaps with the nearest valid sample; never accept missing regions.
        from scipy.ndimage import distance_transform_edt
        distances, nearest = distance_transform_edt(~known, return_indices=True)
        if float(distances.max()) > 4:
            raise RuntimeError("Die 180°-Erweiterung enthält noch größere Lücken. Das Ergebnis wurde nicht als fertig gespeichert.")
        canvas[~known] = canvas[nearest[0][~known], nearest[1][~known]]
    emit("finalize", 95, message="180°-Halbkugelbild wird gespeichert")
    Image.fromarray(np.rint(canvas).clip(0, 255).astype(np.uint8)).save(args.output)
    metadata = {"projection": "equirectangular", "coverageDegrees": 180,
                "width": args.size, "height": args.size, "sourceHorizontalFov": args.fov,
                "modelId": MODEL_ID, "quantizationId": QUANT_ID, "fillTool": "flux-fill",
                "modelRevision": MODEL_REVISION, "quantizationRevision": QUANT_REVISION,
                "precision": args.precision, "prompt": args.prompt or PROMPT,
                "seed": seed, "steps": args.steps, "guidance": args.guidance,
                "originalPreserved": args.blend == 0, "blendBand": args.blend,
                "patchFovDegrees": PATCH_FOV,
                "patchResolution": args.patch_size, "views": records,
                "durationSeconds": round(time.perf_counter() - started, 2)}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    emit("ready", 100, message="180°-Halbkugelbild ist fertig", output=str(args.output), seed=seed)
    del pipe
    gc.collect()
    torch.cuda.empty_cache()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--download-models", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "models/fill180")
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument("--patch-size", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=30)
    parser.add_argument("--fov", type=float, default=65)
    parser.add_argument("--blend", type=int, default=48,
                        help="Cross-fade band in patch pixels over earlier views; a quarter of it over the original.")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--precision", choices=("nf4", "bf16"), default="nf4",
                        help="nf4: 4-bit for 16 GB cards; bf16: full weights for 40 GB+ cards")
    parser.add_argument("--offload", choices=("model", "none"), default="model",
                        help="none keeps everything on the GPU (fastest, bf16 needs about 40 GB)")
    parser.add_argument("--save-steps", type=Path,
                        help="Directory for per-patch input, mask, FLUX result and composite progress images.")
    args = parser.parse_args()
    args.cache_dir = args.cache_dir.resolve()
    if not args.download_models:
        if args.input is None or args.output is None:
            parser.error("--input and --output are required")
        args.input, args.output = args.input.resolve(), args.output.resolve()
        if not args.input.is_file() or args.input == args.output or args.output.suffix.lower() != ".png":
            parser.error("Use an existing input and a different PNG output")
        if not 256 <= args.size <= 4096 or args.size % 32:
            parser.error("--size must be 256..4096 and a multiple of 32")
        if not 256 <= args.patch_size <= 1024 or args.patch_size % 32:
            parser.error("--patch-size must be 256..1024 and a multiple of 32")
        if not 0 <= args.blend <= 256:
            parser.error("--blend must be 0..256")
        if not 10 <= args.fov <= 160 or not 1 <= args.steps <= 60 or not 1 <= args.guidance <= 30:
            parser.error("Invalid FOV, step count or guidance")
        if args.seed is not None and not 0 <= args.seed < 2**63 - len(FILL_VIEWS):
            parser.error("Seed is outside the supported range")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        runtime = load_runtime()
        download_models(args, runtime) if args.download_models else generate(args, runtime)
        return 0
    except Exception as exc:
        emit("error", 0, message=str(exc)[:2000], error=str(exc)[:2000], exception=type(exc).__name__)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
