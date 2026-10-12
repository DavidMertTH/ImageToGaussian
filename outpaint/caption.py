"""Describe a photo and its likely surroundings as a prompt for panorama models.

A fixed prompt makes outpainting models invent unrelated content (buildings on
a beach). Qwen3-VL describes the actual scene instead. Runs in .venv-outpaint.

    python outpaint/caption.py --input photo.jpg --output caption.json
    python outpaint/caption.py --download-models
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import MODELS, emit, load_photo, marker_ready, quiet, run_cli, snapshot, write_marker

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
CACHE = MODELS / "caption"
INSTRUCTION = (
    "You write prompts for an image generator that extends this photo into a full 360-degree panorama "
    "around the camera. Describe only the empty environment in one paragraph of at most 80 words: "
    "type of place, landscape or room, ground surface, vegetation, water, sky or ceiling, materials, "
    "lighting, time of day, weather and colour palette. Never mention people, animals, vehicles or the "
    "main subject of the photo, because the generator would copy them into the surroundings. Do not "
    "add buildings the photo does not suggest. Do not mention the photo, the camera or the frame."
)


def download() -> None:
    emit("download", 0, message="Bildbeschreibung (Qwen3-VL 8B) wird geladen")
    snapshot(MODEL_ID, MODEL_REVISION, CACHE / "model")
    write_marker(CACHE, modelId=MODEL_ID, modelRevision=MODEL_REVISION)
    emit("ready", 100, message="Bildbeschreibung ist bereit")


def describe(path: Path, device: str = "cuda") -> str:
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if not marker_ready(CACHE):
        raise RuntimeError("Bildbeschreibungsmodell fehlt. setup.ps1 ausführen.")
    model_path = str(CACHE / "model")
    with quiet():
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        # device_map="auto" keeps it on a large GPU and spills to RAM on small ones.
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=torch.bfloat16, local_files_only=True, device_map="auto").eval()
    image = Image.fromarray(load_photo(path))
    image.thumbnail((1024, 1024))
    messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                             {"type": "text", "text": INSTRUCTION}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt").to(device)
    with torch.inference_mode(), quiet():
        output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    text = processor.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    del model
    torch.cuda.empty_cache()
    return " ".join(text.split())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--download-models", action="store_true")
    args = parser.parse_args()
    if args.download_models:
        download()
        return
    if not args.input or not args.output:
        parser.error("--input and --output are required")
    emit("caption", 5, message="Bild wird beschrieben")
    caption = describe(args.input)
    args.output.write_text(json.dumps({"caption": caption}, ensure_ascii=False), encoding="utf-8")
    emit("ready", 100, message="Bildbeschreibung ist fertig", caption=caption)


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
