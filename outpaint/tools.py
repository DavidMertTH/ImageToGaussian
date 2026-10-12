"""Registry of the additional panorama models and the job pipeline that runs them.

A job is: optional scene description (Qwen3-VL) -> generator -> finishing
(re-insert the photo; for 180 degrees cut out and AI-upscale the front half).
Every step is a subprocess in the tool's own virtual environment, so CUDA
memory is released between steps and conflicting library versions coexist.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

from common import MODELS, ROOT, TOOL_DIR, marker_ready, photo_fov

ENV_PANORAMA = ".venv-panorama"   # diffusers 0.37 / transformers 4.57 (studio's tested pins)
ENV_OUTPAINT = ".venv-outpaint"       # diffusers 0.41 / transformers 5
ENV_DIT360 = ".venv-dit360"       # diffusers 0.35.2 / transformers 4.52.4


def python(environment: str) -> Path:
    return ROOT / environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _fill_ready(precision: str) -> bool:
    try:
        marker = json.loads((ROOT / "models" / "fill180" / "ready.json").read_text(encoding="utf-8"))
        return marker.get("ready") is True and precision in marker.get("precisions", ["nf4"])
    except (OSError, ValueError):
        return False


TOOLS = {
    "flux-fill-nf4": {
        "label": "FLUX Fill · 4-Bit",
        "description": "Wie im Studio, mit Bildbeschreibung und Bildwinkel aus EXIF. Läuft auch auf 16-GB-Karten.",
        "env": ENV_PANORAMA, "script": ROOT / "fill180_generate.py", "output": "hemisphere",
        "coverages": [180], "caption": True, "ready": lambda: _fill_ready("nf4"),
        "args": lambda c: ["--fov", c["fov"], "--prompt", c["caption"]],
    },
    "flux-fill-bf16": {
        "label": "FLUX Fill · volle Genauigkeit",
        "description": "Fünf Perspektivansichten mit FLUX.1 Fill (bf16), weich überblendet. Nur 180°.",
        "env": ENV_PANORAMA, "script": ROOT / "fill180_generate.py", "output": "hemisphere",
        "coverages": [180], "caption": True, "ready": lambda: _fill_ready("bf16"),
        "args": lambda c: ["--precision", "bf16", "--offload", "none", "--fov", c["fov"], "--prompt", c["caption"]],
    },
    "klein-9b": {
        "label": "FLUX.2 Klein 9B + 360°-LoRA",
        "description": "Ganzes Panorama in einem Durchgang mit dem 360°-Outpaint-LoRA.",
        "env": ENV_PANORAMA, "script": ROOT / "panorama_generate.py", "output": "erp360",
        "coverages": [180, 360], "caption": True,
        "ready": lambda: marker_ready(ROOT / "models" / "panorama-9b"),
        "args": lambda c: ["--model", "9b", "--offload", "none", "--fov", c["fov"], "--prompt", c["caption"]],
    },
    "qwen360": {
        "label": "Qwen-Image-2512 + 360°-LoRA",
        "description": "Qwen-Image mit dem Qwen-360-Diffusion-LoRA, ergänzt um das Foto herum.",
        "env": ENV_OUTPAINT, "script": TOOL_DIR / "qwen360_pano.py", "output": "erp360",
        "coverages": [180, 360], "caption": True,
        "ready": lambda: marker_ready(MODELS / "qwen360"),
        "args": lambda c: ["--fov", c["fov"], "--prompt", c["caption"]],
    },
    "krea2": {
        "label": "Krea 2 + 360°-Outpaint-LoRA",
        "description": "Krea 2 Turbo mit dem 360°-Edit-LoRA; sieht das Foto als Referenzbild.",
        "env": ENV_OUTPAINT, "script": TOOL_DIR / "krea2_pano.py", "output": "erp360",
        "coverages": [180, 360], "caption": False,
        "ready": lambda: marker_ready(MODELS / "krea2"),
        "args": lambda c: ["--fov", c["fov"]],
    },
    "dit360": {
        "label": "DiT360",
        "description": "Panorama-Modell (FLUX.1-dev + DiT360-LoRA), ergänzt per RF-Inversion.",
        "env": ENV_DIT360, "script": TOOL_DIR / "dit360_pano.py", "output": "erp360",
        "coverages": [180, 360], "caption": True,
        "ready": lambda: marker_ready(MODELS / "dit360") and (TOOL_DIR / "vendor" / "DiT360" / "pa_src").is_dir(),
        "args": lambda c: ["--fov", c["fov"], "--prompt", c["caption"]],
    },
}


def tool_status() -> dict:
    status = {}
    for name, tool in TOOLS.items():
        ready = python(tool["env"]).is_file() and tool["ready"]()
        status[name] = {"label": tool["label"], "description": tool["description"], "ready": ready,
                        "coverages": tool["coverages"],
                        "message": "bereit" if ready else "nicht eingerichtet (setup.ps1)"}
    return status


class Cancelled(Exception):
    pass


def run_step(command: list, log, on_event: Callable[[dict], None], register: Callable | None = None) -> None:
    """Run one JSON-lines step; raise with the step's own error message on failure."""
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = subprocess.Popen([str(part) for part in command], cwd=ROOT, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                               creationflags=flags, env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1"})
    if register:
        register(process)
    error, tail = None, []
    for raw in process.stdout:
        log.write(raw)
        log.flush()
        line = raw.strip()
        if not line:
            continue
        tail = (tail + [line])[-12:]
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict):
            if isinstance(event.get("error"), str):
                error = event["error"]
            on_event(event)
    code = process.wait()
    if register:
        register(None)
    if code:
        raise RuntimeError(error or "\n".join(line for line in tail if not line.startswith("{"))[-1500:]
                           or f"Schritt endete mit Code {code}.")


def run_job(tool_name: str, photo: Path, coverage: int, folder: Path, progress: Callable[[str, float], None],
            prompt: str | None = None, register: Callable | None = None, seed: int | None = None) -> Path:
    """Run a complete job and return the finished panorama PNG."""
    tool = TOOLS[tool_name]
    if coverage not in tool["coverages"]:
        raise ValueError(f"{tool['label']} unterstützt {coverage}° nicht.")
    folder.mkdir(parents=True, exist_ok=True)
    fov, fov_source = photo_fov(photo)
    context = {"fov": f"{fov:g}", "caption": prompt or ""}
    with (folder / "outpaint.log").open("a", encoding="utf-8") as log:
        def relay(start: float, end: float):
            def handle(event):
                stage = event.get("message") or event.get("stage")
                percent = event.get("progressPercent")
                if isinstance(percent, (int, float)) and isinstance(stage, str):
                    progress(stage, start + (end - start) * percent / 100)
            return handle

        if tool["caption"] and not prompt:
            progress("Bild wird beschrieben", 1)
            run_step([python(ENV_OUTPAINT), TOOL_DIR / "caption.py", "--input", photo,
                      "--output", folder / "caption.json"], log, relay(0, 10), register)
            context["caption"] = json.loads((folder / "caption.json").read_text(encoding="utf-8"))["caption"]
        raw = folder / ("panorama.png" if tool["output"] == "hemisphere" else "generated360.png")
        command = [python(tool["env"]), tool["script"], "--input", photo, "--output", raw, *tool["args"](context)]
        if seed is not None:
            command += ["--seed", str(seed)]
        run_step(command, log, relay(10, 90), register)
        result = raw
        if tool["output"] == "erp360":
            result = folder / "panorama.png"
            run_step([python(ENV_OUTPAINT), TOOL_DIR / "finish_panorama.py", "--input", raw, "--photo", photo,
                      "--output", result, "--coverage", coverage, "--fov", context["fov"]],
                     log, relay(90, 100), register)
    (folder / "job-info.json").write_text(json.dumps(
        {"tool": tool_name, "coverage": coverage, "fov": fov, "fovSource": fov_source,
         "caption": context["caption"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    if not result.is_file():
        raise RuntimeError("Es wurde kein Panorama erzeugt.")
    return result


if __name__ == "__main__":
    for name, item in tool_status().items():
        print(f"  {'bereit ' if item['ready'] else 'fehlt  '} {name:15} {item['label']}")
