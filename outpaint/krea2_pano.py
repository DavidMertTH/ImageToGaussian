"""Extend a photo to a 360 degree ERP panorama with Krea 2 + the 360 outpaint LoRA.

The LoRA (mickmumpitz/Krea2-360-ERP-LoRAs, MIT) is an ostris ai-toolkit *edit*
LoRA: the green-marked canvas is given to the transformer as an in-context
reference (extra tokens with RoPE index 1, modulated at t = 0) and to the
Qwen3-VL text encoder as an image. That conditioning is not part of
Krea2Pipeline; it is ported here from the reference implementation in
https://huggingface.co/spaces/hugging-apps/krea2-360-panorama-lora (rev 5534b1d).
Runs in .venv-outpaint (diffusers 0.41).
"""

from __future__ import annotations

import argparse
import math
import re
import secrets
import time
from pathlib import Path

import numpy as np

from common import (MODELS, emit, green_canvas, load_photo, marker_ready, photo_fov, project_photo, quiet,
                    run_cli, save_png, snapshot, write_marker)

MODEL_ID = "krea/Krea-2-Turbo"
MODEL_REVISION = "98e0fe118d17c9e3547fbb2e25acdbae2cadf7c7"
LORA_ID = "mickmumpitz/Krea2-360-ERP-LoRAs"
LORA_REVISION = "f2ff4c47c3cd745ef397334ef559d6a9568efbc2"
LORA_FILE = "krea2_oedit_360_erp_outpaint_lora_v1.safetensors"
CACHE = MODELS / "krea2"
PROMPT = ("Fill the green spaces according to the image. Outpaint as a seamless 360 equirectangular "
          "panorama (2:1). Keep the horizon level. Match left and right edges.")
REF_MAX_PIXELS = 1024 * 1024
VLM_MAX_PIXELS = 384 * 384
QWEN_PATCH, QWEN_MERGE, QWEN_TEMPORAL = 16, 2, 2
QWEN_MIN_PIXELS, QWEN_MAX_PIXELS = 3136, 12845056


def download() -> None:
    emit("download", 0, message="Krea 2 Turbo und 360°-LoRA werden geladen")
    snapshot(MODEL_ID, MODEL_REVISION, CACHE / "base",
             ["model_index.json", "scheduler/*", "text_encoder/*", "tokenizer/*", "transformer/*", "vae/*"])
    snapshot(LORA_ID, LORA_REVISION, CACHE / "lora", [LORA_FILE])
    write_marker(CACHE, modelId=MODEL_ID, modelRevision=MODEL_REVISION, loraId=LORA_ID, loraRevision=LORA_REVISION)
    emit("ready", 100, message="Krea 2 ist bereit")


_ATTN = {"wq": "to_q", "wk": "to_k", "wv": "to_v", "wo": "to_out.0", "gate": "to_gate"}
_FF = {"gate": "ff.gate", "up": "ff.up", "down": "ff.down"}


def _module_name(path: str):
    if match := re.match(r"blocks\.(\d+)\.attn\.(\w+)$", path):
        index, sub = match.groups()
        return f"transformer_blocks.{index}.attn.{_ATTN[sub]}" if sub in _ATTN else None
    if match := re.match(r"blocks\.(\d+)\.mlp\.(\w+)$", path):
        index, sub = match.groups()
        return f"transformer_blocks.{index}.{_FF[sub]}" if sub in _FF else None
    if match := re.match(r"txtfusion\.(layerwise_blocks|refiner_blocks)\.(\d+)\.(attn|mlp)\.(\w+)$", path):
        block, index, kind, sub = match.groups()
        table = _ATTN if kind == "attn" else _FF
        return f"text_fusion.{block}.{index}.{table[sub]}" if sub in table else None
    return None


def convert_lora(state_dict: dict) -> dict:
    """ai-toolkit krea2 LoRA layout -> diffusers/PEFT module names."""
    state_dict = {k.removeprefix("base_model.model.").removeprefix("diffusion_model."): v
                  for k, v in state_dict.items()}
    converted = {}
    for key in list(state_dict):
        match = re.search(r"\.(lora_[AB])\.weight$", key)
        if match and (module := _module_name(key[:match.start()])):
            converted[f"transformer.{module}{key[match.start():]}"] = state_dict.pop(key)
    if state_dict:
        raise ValueError(f"Unbekannte LoRA-Tensoren: {list(state_dict)[:4]}")
    return converted


def install_incontext_forward(transformer, torch, F, Output):
    """Append clean reference tokens (RoPE index 1, timestep 0) to the image sequence."""
    original = transformer.forward

    def block_forward(block, hidden, temb_mod, temb_mod0, split, rotary, mask):
        target = (temb_mod.unflatten(-1, (6, -1)) + block.scale_shift_table).unbind(-2)
        reference = (temb_mod0.unflatten(-1, (6, -1)) + block.scale_shift_table).unbind(-2)

        def modulate(h, scale, shift):
            return torch.cat([(1 + target[scale]) * h[:, :split] + target[shift],
                              (1 + reference[scale]) * h[:, split:] + reference[shift]], dim=1)

        def gate(h, index):
            return torch.cat([target[index] * h[:, :split], reference[index] * h[:, split:]], dim=1)

        hidden = hidden + gate(block.attn(modulate(block.norm1(hidden), 0, 1), attention_mask=mask,
                                          image_rotary_emb=rotary), 2)
        return hidden + gate(block.ff(modulate(block.norm2(hidden), 3, 4)), 5)

    def forward(hidden_states, encoder_hidden_states, timestep, position_ids, encoder_attention_mask=None,
                attention_kwargs=None, return_dict=True):
        ref = getattr(transformer, "_edit_ref", None)
        if ref is None:
            return original(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states,
                            timestep=timestep, position_ids=position_ids,
                            encoder_attention_mask=encoder_attention_mask, attention_kwargs=attention_kwargs,
                            return_dict=return_dict)
        self = transformer
        batch, image_len, _ = hidden_states.shape
        text_len = encoder_hidden_states.shape[1]
        temb = self.time_embed(timestep, dtype=hidden_states.dtype)
        temb_mod = self.time_mod_proj(F.gelu(temb, approximate="tanh"))
        temb0 = self.time_embed(torch.zeros_like(timestep), dtype=hidden_states.dtype)
        temb_mod0 = self.time_mod_proj(F.gelu(temb0, approximate="tanh"))
        text_mask = mask = None
        if encoder_attention_mask is not None:
            text_mask = encoder_attention_mask[:, None, None, :]
            mask = torch.cat([encoder_attention_mask, encoder_attention_mask.new_ones((batch, image_len)),
                              encoder_attention_mask.new_ones((batch, ref.shape[1]))], dim=1)[:, None, None, :]
        text = self.txt_in(self.text_fusion(encoder_hidden_states, attention_mask=text_mask))
        ref = ref.to(device=hidden_states.device, dtype=hidden_states.dtype)
        if ref.shape[0] != batch:
            ref = ref[:1].repeat(batch, 1, 1)
        combined = torch.cat([text, self.img_in(torch.cat([hidden_states, ref], dim=1))], dim=1)
        grid_h, grid_w = self._edit_ref_grid
        ref_pos = torch.zeros(grid_h, grid_w, 3, device=combined.device, dtype=position_ids.dtype)
        ref_pos[..., 0] = 1
        ref_pos[..., 1] = torch.arange(grid_h, device=combined.device, dtype=position_ids.dtype)[:, None]
        ref_pos[..., 2] = torch.arange(grid_w, device=combined.device, dtype=position_ids.dtype)[None, :]
        rotary = self.rotary_emb(torch.cat([position_ids, ref_pos.reshape(-1, 3)], dim=0))
        for block in self.transformer_blocks:
            combined = block_forward(block, combined, temb_mod, temb_mod0, text_len + image_len, rotary, mask)
        output = self.final_layer(combined[:, text_len:text_len + image_len], temb)
        return Output(sample=output) if return_dict else (output,)

    transformer.forward = forward


def fit_area(torch, F, samples, max_pixels: int, snap: int = 1):
    height, width = samples.shape[2:]
    scale = min(1.0, math.sqrt(max_pixels / (width * height)))
    size = (max(round(height * scale / snap) * snap, snap), max(round(width * scale / snap) * snap, snap))
    return samples if size == (height, width) else F.interpolate(samples, size=size, mode="area")


def prepare_reference(pipe, torch, F, canvas01):
    ref = fit_area(torch, F, canvas01, REF_MAX_PIXELS, snap=16)
    latents = pipe.vae.encode((ref * 2 - 1).unsqueeze(2).to(pipe.vae.device, pipe.vae.dtype)).latent_dist.mode()[:, :, 0]
    config = pipe.vae.config
    mean = torch.tensor(config.latents_mean).view(1, config.z_dim, 1, 1).to(latents)
    std = 1.0 / torch.tensor(config.latents_std).view(1, config.z_dim, 1, 1).to(latents)
    latents = (latents - mean) / std
    height, width = latents.shape[2:]
    packed = pipe._pack_latents(latents, latents.shape[0], latents.shape[1], height, width)
    return packed, (height // pipe.patch_size, width // pipe.patch_size)


def vision_patches(torch, F, image01):
    image = fit_area(torch, F, image01, VLM_MAX_PIXELS)
    _, channels, height, width = image.shape
    factor = QWEN_PATCH * QWEN_MERGE
    h_bar, w_bar = round(height / factor) * factor, round(width / factor) * factor
    if h_bar * w_bar > QWEN_MAX_PIXELS:
        beta = math.sqrt(height * width / QWEN_MAX_PIXELS)
        h_bar, w_bar = (max(factor, math.floor(height / beta / factor) * factor),
                        max(factor, math.floor(width / beta / factor) * factor))
    elif h_bar * w_bar < QWEN_MIN_PIXELS:
        beta = math.sqrt(QWEN_MIN_PIXELS / (height * width))
        h_bar, w_bar = (max(factor, math.ceil(height * beta / factor) * factor),
                        max(factor, math.ceil(width * beta / factor) * factor))
    image = (F.interpolate(image, size=(h_bar, w_bar), mode="bilinear", align_corners=False) - 0.5) / 0.5
    grid_h, grid_w = h_bar // QWEN_PATCH, w_bar // QWEN_PATCH
    pixels = image[0].unsqueeze(0).repeat(QWEN_TEMPORAL, 1, 1, 1)
    patches = pixels.reshape(1, QWEN_TEMPORAL, channels, grid_h // QWEN_MERGE, QWEN_MERGE, QWEN_PATCH,
                             grid_w // QWEN_MERGE, QWEN_MERGE, QWEN_PATCH).permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flat = patches.reshape(grid_h * grid_w, channels * QWEN_TEMPORAL * QWEN_PATCH * QWEN_PATCH)
    return flat, torch.tensor([[1, grid_h, grid_w]], device=image.device, dtype=torch.long)


def encode_prompt(pipe, torch, F, prompt: str, image01, device):
    """Krea 2 text conditioning with the canvas visible to the Qwen3-VL encoder."""
    pixels, grid = vision_patches(torch, F, image01.to(device))
    tokens = int(grid[0].prod().item()) // QWEN_MERGE**2
    prefix, suffix = pipe.prompt_template_encode_prefix, pipe.prompt_template_encode_suffix
    prefix_len = len(pipe.tokenizer(prefix).input_ids)
    text = prefix + "Picture 1: <|vision_start|>" + "<|image_pad|>" * tokens + "<|vision_end|>" + prompt + suffix
    tok = pipe.tokenizer(text, return_tensors="pt").to(device)
    positions = (tok.attention_mask.long().cumsum(dim=-1) - 1).clamp(min=0).unsqueeze(0).expand(3, -1, -1)
    outputs = pipe.text_encoder(
        input_ids=tok.input_ids, attention_mask=tok.attention_mask, position_ids=positions,
        pixel_values=pixels.to(pipe.text_encoder.dtype), image_grid_thw=grid,
        mm_token_type_ids=(tok.input_ids == pipe.text_encoder.config.image_token_id).long(),
        output_hidden_states=True)
    hidden = torch.stack([outputs.hidden_states[i] for i in pipe.text_encoder_select_layers], dim=2)
    return hidden[:, prefix_len:].to(torch.bfloat16), tok.attention_mask.bool()[:, prefix_len:]


def generate(args) -> None:
    import torch
    import torch.nn.functional as F
    from diffusers import Krea2Pipeline
    from diffusers.models.modeling_outputs import Transformer2DModelOutput
    from safetensors.torch import load_file

    if not marker_ready(CACHE):
        raise RuntimeError("Krea 2 fehlt. setup.ps1 ausführen.")
    started = time.perf_counter()
    seed = args.seed if args.seed is not None else secrets.randbelow(2**31)
    fov = args.fov or photo_fov(args.input)[0]
    emit("prepare", 2, message="Foto wird auf die 360°-Kugel projiziert", fov=fov, seed=seed)
    projected, known = project_photo(load_photo(args.input), args.width, args.height, 360, fov)
    canvas = green_canvas(projected, known)
    canvas.save(args.output.with_name(args.output.stem + ".control.png"))
    emit("load", 5, message="Krea 2 Turbo wird geladen")
    with quiet():
        pipe = Krea2Pipeline.from_pretrained(str(CACHE / "base"), torch_dtype=torch.bfloat16, local_files_only=True)
        pipe.load_lora_weights(convert_lora(load_file(str(CACHE / "lora" / LORA_FILE))), adapter_name="pano")
        pipe.transformer.set_adapters("pano", weights=args.lora_strength)
        pipe.set_progress_bar_config(disable=True)
        if args.offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
    install_incontext_forward(pipe.transformer, torch, F, Transformer2DModelOutput)
    device = torch.device("cuda")
    source = torch.from_numpy(np.asarray(canvas, dtype=np.float32) / 255).permute(2, 0, 1)[None].to(device)

    def on_step(_pipe, step, _t, kwargs):
        emit("generate", 10 + 85 * (step + 1) / args.steps, message=f"Krea 2 · {step + 1}/{args.steps}")
        return kwargs

    prompt = PROMPT + (" " + args.prompt.strip() if args.prompt else "")
    with torch.inference_mode():
        if args.offload:
            pipe.vae.to(device)
            pipe.text_encoder.to(device)
        ref, grid = prepare_reference(pipe, torch, F, source)
        pipe.transformer._edit_ref, pipe.transformer._edit_ref_grid = ref, grid
        embeds, mask = encode_prompt(pipe, torch, F, prompt, source, device)
        with quiet():
            image = pipe(prompt_embeds=embeds, prompt_embeds_mask=mask, width=args.width, height=args.height,
                         num_inference_steps=args.steps, guidance_scale=0.0,
                         generator=torch.Generator("cuda").manual_seed(seed),
                         callback_on_step_end=on_step, callback_on_step_end_tensor_inputs=[]).images[0]
    save_png(np.asarray(image.convert("RGB")), args.output)
    emit("ready", 100, message="Krea-2-Panorama ist fertig", output=str(args.output), seed=seed,
         durationSeconds=round(time.perf_counter() - started, 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, help="2:1 ERP PNG")
    parser.add_argument("--download-models", action="store_true")
    parser.add_argument("--fov", type=float, help="Horizontal FOV of the photo; default from EXIF or 65")
    parser.add_argument("--prompt", default="", help="Optional text appended to the trained instruction")
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--lora-strength", type=float, default=1.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--offload", action="store_true", help="CPU offload for GPUs below ~48 GB")
    args = parser.parse_args()
    if args.download_models:
        download()
        return
    if not args.input or not args.output:
        parser.error("--input and --output are required")
    generate(args)


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
