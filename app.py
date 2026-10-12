"""Local SHARP studio: image uploads, background inference, and scene files."""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import mimetypes
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

from PIL import Image, UnidentifiedImageError

import pipeline_view


ROOT = Path(__file__).resolve().parent
DIST = ROOT / "web" / "dist"
DEMO_IMAGES = ROOT / "src" / "Demo Images"
DEMO_OUTPUT = ROOT / "output" / "Demo Images"
JOB_OUTPUT = ROOT / "output" / "ui"
PANORAMA_CACHE = ROOT / "models" / "panorama"
FILL180_CACHE = ROOT / "models" / "fill180"
DEPTH_CACHE = ROOT / "models" / "depth"
# Additional panorama models (DiT360, Qwen, Krea, ...); each runs in its own environment.
sys.path.insert(0, str(ROOT / "outpaint"))
import tools as outpaint_tools  # noqa: E402
UPSCALE_MODEL = ROOT / "models" / "upscale" / "RealESRGAN_x4plus.pth"
DA2_WEIGHTS = ROOT / "models" / "da2" / "model.safetensors"
DA2_WEIGHTS_BYTES = 1378513064
# SHARP network sizes offered in the UI; 1536 is the trained resolution.
SHARP_RESOLUTIONS = (1024, 1536, 2048, 2560)
DEFAULT_SHARP_RESOLUTION = 1536
MAX_UPLOAD = 30 * 1024 * 1024
IMAGE_FORMATS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}

jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
job_queue: queue.Queue[str] = queue.Queue()
# Guarded by jobs_lock: running model process and pending cancel requests per job.
job_processes: dict[str, subprocess.Popen] = {}
cancel_requests: set[str] = set()


class JobCancelled(Exception):
    pass


def terminate_process_tree(process: subprocess.Popen) -> None:
    """Console-script launchers (sharp.exe) run Python as a child process; kill the whole tree."""
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        try:
            os.killpg(process.pid, 9)
        except OSError:
            process.kill()


def check_cancelled(job_id: str) -> None:
    with jobs_lock:
        if job_id in cancel_requests:
            raise JobCancelled()


def python_command(environment: str = ".venv") -> Path:
    return ROOT / environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def panorama_status() -> dict:
    ready = python_command(".venv-panorama").is_file() and (PANORAMA_CACHE / "ready.json").is_file()
    return {
        "ready": ready,
        "message": "360° ist bereit" if ready else "360° benötigt die Panorama-Modelle. Einrichten: setup-panorama.ps1 -FillTool klein-lora.",
    }


def fill_tools_status() -> dict:
    fill_ready = False
    try:
        marker = json.loads((FILL180_CACHE / "ready.json").read_text(encoding="utf-8"))
        fill_ready = (python_command(".venv-panorama").is_file()
                      and marker.get("ready") is True and marker.get("fillTool") == "flux-fill"
                      and (FILL180_CACHE / "nf4/transformer/config.json").is_file()
                      and (FILL180_CACHE / "base/vae/config.json").is_file())
    except (OSError, ValueError):
        pass
    return {
        "flux-fill": {
            "ready": fill_ready, "mode": "hemisphere", "coverageDegrees": 180,
            "message": "FLUX Fill für 180° ist bereit" if fill_ready else
            "FLUX Fill fehlt. Zuerst die Modellbedingungen bei Hugging Face annehmen und anmelden; dann setup-panorama.ps1 -FillTool flux-fill ausführen. Anleitung: STUDIO.md.",
        },
        "klein-lora": {**panorama_status(), "mode": "panorama", "coverageDegrees": 360},
        **outpaint_fill_tools(),
    }


def outpaint_fill_tools() -> dict:
    """Additional panorama models chosen in setup.ps1; only installed ones are offered."""
    tools = {}
    for name, tool in outpaint_tools.tool_status().items():
        if tool["ready"]:
            tools[name] = {"ready": True, "extra": True, "label": tool["label"], "description": tool["description"],
                           "coverages": tool["coverages"], "mode": None, "message": f"{tool['label']} ist bereit"}
    return tools


def fill_tool_matches(capability: dict, mode: str) -> bool:
    """Built-in tools have one mode; additional models can produce 180 and/or 360 degrees."""
    if capability.get("extra"):
        return {"hemisphere": 180, "panorama": 360}.get(mode) in capability["coverages"]
    return capability.get("mode") == mode


def depth_estimators_status() -> dict:
    ready = False
    try:
        marker = json.loads((DEPTH_CACHE / "native-ready.json").read_text(encoding="utf-8"))
        model_path = (DEPTH_CACHE / marker["modelDirectory"]).resolve()
        weights_path = model_path / "model.safetensors"
        ready = (
            python_command(".venv-depth").is_file()
            and (ROOT / "da3_splats.py").is_file()
            and (ROOT / "da3_runtime.py").is_file()
            and marker.get("ready") is True
            and marker.get("pipeline") == "da3-native"
            and model_path.parent == DEPTH_CACHE.resolve()
            and (model_path / "config.json").is_file()
            and weights_path.is_file()
            and weights_path.stat().st_size == marker.get("weightsBytes")
        )
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return {
        "sharp": {"ready": True, "message": "SHARP / Depth Pro ist bereit"},
        "da3": {
            "ready": ready,
            "message": "Depth Anything 3 mit eigener Gaussian-Vorhersage ist bereit" if ready else "Depth Anything 3 benötigt das native Gaussian-Modell. Einmal setup-depth.ps1 ausführen.",
        },
    }


def upscaler_status() -> dict:
    ready = UPSCALE_MODEL.is_file() and importlib.util.find_spec("spandrel") is not None
    return {
        "ready": ready,
        "message": "KI-Upscaler Real-ESRGAN ist bereit" if ready else
        "KI-Upscaler fehlt. Einrichten: setup-upscale.ps1 ausführen.",
    }


def depth_alignment_status() -> dict:
    """DA² reference depth for SHARP panoramas (experimental, setup-da2.ps1)."""
    site_packages = [ROOT / ".venv-depth" / "Lib" / "site-packages",
                     *(ROOT / ".venv-depth" / "lib").glob("python*/site-packages")]
    ready = (
        python_command(".venv-depth").is_file()
        and (ROOT / "da2_reference.py").is_file()
        and any((folder / "da2" / "model" / "spherevit.py").is_file() for folder in site_packages)
        and DA2_WEIGHTS.is_file()
        and DA2_WEIGHTS.stat().st_size == DA2_WEIGHTS_BYTES
    )
    return {
        "ready": ready,
        "message": "DA²-Tiefenangleich ist bereit" if ready else
        "DA² fehlt. Einrichten: setup-da2.ps1 ausführen (benötigt .venv-depth aus setup-depth.ps1).",
    }


def sharp_options(manifest: dict) -> list[str]:
    """SHARP resolution and AI upscaling stored with the job."""
    options = ["--resolution", str(manifest.get("sharpResolution", DEFAULT_SHARP_RESOLUTION))]
    if manifest.get("upscale") == "auto":
        if not upscaler_status()["ready"]:
            raise RuntimeError(upscaler_status()["message"])
        options += ["--upscale", "auto"]
    return options


def sharp_command() -> str:
    local = ROOT / ".venv" / ("Scripts/sharp.exe" if os.name == "nt" else "bin/sharp")
    if local.exists():
        return str(local)
    found = shutil.which("sharp")
    if found:
        return found
    raise FileNotFoundError("SHARP wurde nicht gefunden. Installiere zuerst requirements.txt in .venv.")


def iso_time(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def point_count(path: Path) -> int | None:
    with path.open("rb") as stream:
        for _ in range(80):
            line = stream.readline()
            if line.startswith(b"element vertex "):
                return int(line.split()[-1])
            if line == b"end_header\n":
                break
    return None


def scene_records() -> list[dict]:
    scenes = []
    if DEMO_OUTPUT.exists():
        for ply in DEMO_OUTPUT.glob("*.ply"):
            image = next(
                (candidate for candidate in DEMO_IMAGES.glob(f"{ply.stem}.*")
                 if candidate.suffix.lower() in IMAGE_FORMATS.values()),
                None,
            )
            scenes.append({
                "id": f"demo-{ply.stem}",
                "name": ply.stem,
                "kind": "demo",
                "created": iso_time(ply.stat().st_mtime),
                "ply": ply,
                "image": image,
                "mode": "image",
            })

    if JOB_OUTPUT.exists():
        for manifest_path in JOB_OUTPUT.glob("*/manifest.json"):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                folder = manifest_path.parent
                ply = folder / manifest.get("outputFile", "input.ply")
                image = folder / manifest["inputFile"]
                if ply.is_file() and image.is_file():
                    scenes.append({
                        "id": manifest["id"],
                        "name": manifest["name"],
                        "kind": "upload",
                        "created": manifest["created"],
                        "ply": ply,
                        "image": image,
                        "mode": manifest.get("mode", "image"),
                        "depthEstimator": manifest.get("depthEstimator", "sharp"),
                        "pipeline": manifest.get("pipeline", "da3-correction" if manifest.get("depthEstimator") == "da3" else "sharp"),
                        "fillTool": manifest.get("fillTool", "klein-lora" if manifest.get("mode") == "panorama" else None),
                        "depthAlignment": manifest.get("depthAlignment"),
                        "panorama": folder / "panorama.png" if (folder / "panorama.png").is_file() else None,
                    })
            except (OSError, ValueError, KeyError):
                continue
    return sorted(scenes, key=lambda item: item["created"], reverse=True)


def public_scene(scene: dict) -> dict:
    scene_id = quote(scene["id"], safe="")
    return {
        "id": scene["id"],
        "name": scene["name"],
        "kind": scene["kind"],
        "mode": scene.get("mode", "image"),
        "depthEstimator": scene.get("depthEstimator", "sharp"),
        "pipeline": scene.get("pipeline", "sharp"),
        "fillTool": scene.get("fillTool"),
        "depthAlignment": scene.get("depthAlignment"),
        "created": scene["created"],
        "sizeBytes": scene["ply"].stat().st_size,
        "pointCount": point_count(scene["ply"]),
        "plyUrl": f"/api/scenes/{scene_id}/ply",
        "imageUrl": f"/api/scenes/{scene_id}/image" if scene["image"] else None,
        "panoramaUrl": f"/api/scenes/{scene_id}/panorama" if scene.get("panorama") else None,
    }


def set_job(job_id: str, **fields: object) -> None:
    with jobs_lock:
        jobs[job_id].update(fields)
        snapshot = dict(jobs[job_id])
    folder = JOB_OUTPUT / job_id
    temporary = folder / f"job.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_text(json.dumps(snapshot, ensure_ascii=False), encoding="utf-8")
        # Windows readers and virus scanners can briefly lock the destination.
        # A progress-file lock must never abort a running model calculation.
        for attempt in range(8):
            try:
                temporary.replace(folder / "job.json")
                break
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(0.025 * (attempt + 1))
    except OSError as exc:
        print(f"Statusdatei konnte nicht aktualisiert werden: {exc}", file=sys.stderr, flush=True)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def run_process(job_id: str, command: list[str], log_path: Path, progress_start: float = 0, progress_end: float = 100) -> None:
    """Each model runs in its own process so CUDA memory is released between stages."""
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=flags,
        start_new_session=os.name != "nt",
        env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1"},
    )
    with jobs_lock:
        job_processes[job_id] = process
        cancelled = job_id in cancel_requests
    if cancelled:
        terminate_process_tree(process)
    assert process.stdout is not None
    lines: list[str] = []
    process_error = None
    with log_path.open("a", encoding="utf-8") as log:
        for raw_line in process.stdout:
            log.write(raw_line)
            log.flush()
            line = raw_line.strip()
            if not line:
                continue
            lines.append(line)
            lines = lines[-16:]
            try:
                event = json.loads(line)
            except ValueError:
                event = None
            if isinstance(event, dict):
                fields = {}
                if isinstance(event.get("message"), str):
                    fields["stage"] = event["message"]
                elif isinstance(event.get("stage"), str):
                    fields["stage"] = event["stage"]
                if isinstance(event.get("error"), str):
                    process_error = event["error"]
                if isinstance(event.get("progressPercent"), (int, float)):
                    progress = min(100, max(0, event["progressPercent"]))
                    fields["progressPercent"] = round(progress_start + progress * (progress_end - progress_start) / 100)
                if fields:
                    set_job(job_id, **fields)
                continue
            lower = line.lower()
            stage = None
            if "no checkpoint provided" in lower:
                stage = "Modell wird geladen"
            elif "running preprocessing" in lower:
                stage = "Bild wird vorbereitet"
            elif "running inference" in lower:
                stage = "3D-Szene wird berechnet"
            elif "running postprocessing" in lower:
                stage = "Splats werden aufbereitet"
            elif "saving 3dgs" in lower:
                stage = "Szene wird gespeichert"
            if stage:
                set_job(job_id, stage=stage)
    exit_code = process.wait()
    with jobs_lock:
        job_processes.pop(job_id, None)
    check_cancelled(job_id)
    if exit_code:
        errors = [line for line in lines if not line.startswith('{"stage"')]
        raise RuntimeError(process_error or "\n".join(errors[-5:]) or f"Verarbeitung endete mit Code {exit_code}.")


def restore_jobs() -> None:
    for manifest_path in sorted(JOB_OUTPUT.glob("*/manifest.json"), key=lambda path: path.stat().st_mtime):
        folder = manifest_path.parent
        state_path = folder / "job.json"
        if not state_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state.setdefault("depthEstimator", manifest.get("depthEstimator", "sharp"))
            state.setdefault("pipeline", manifest.get("pipeline", "da3-correction" if manifest.get("depthEstimator") == "da3" else "sharp"))
            state.setdefault("fillTool", manifest.get("fillTool", "klein-lora" if manifest.get("mode") == "panorama" else None))
            job_id = manifest["id"]
            if (folder / manifest.get("outputFile", "input.ply")).is_file():
                state.update(status="ready", stage="Fertig")
            elif state.get("status") in {"queued", "processing"}:
                state.update(status="queued", stage="In Warteschlange", progressPercent=0)
                job_queue.put(job_id)
            jobs[job_id] = state
        except (OSError, ValueError, KeyError):
            continue


def run_jobs() -> None:
    while True:
        job_id = job_queue.get()
        folder = JOB_OUTPUT / job_id
        started = time.monotonic()
        try:
            check_cancelled(job_id)
            manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
            input_path = folder / manifest["inputFile"]
            depth_estimator = manifest.get("depthEstimator", "sharp")
            if depth_estimator not in {"sharp", "da3"}:
                raise RuntimeError("Unbekanntes 3D-Modell.")
            use_da3 = depth_estimator == "da3"
            if use_da3 and not depth_estimators_status()["da3"]["ready"]:
                raise RuntimeError(depth_estimators_status()["da3"]["message"])
            pipeline = "da3-native" if use_da3 else "sharp"
            if manifest.get("pipeline") != pipeline:
                manifest["pipeline"] = pipeline
                (folder / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
            set_job(job_id, status="processing", stage="Verarbeitung startet", progressPercent=0,
                    pipeline=pipeline,
                    startedAt=datetime.now(timezone.utc).isoformat())
            log_path = folder / "pipeline.log"
            if manifest.get("mode") in {"panorama", "hemisphere"}:
                hemisphere = manifest["mode"] == "hemisphere"
                fill_tool = manifest.get("fillTool", "flux-fill" if hemisphere else "klein-lora")
                capability = fill_tools_status().get(fill_tool)
                if not capability or not fill_tool_matches(capability, manifest["mode"]):
                    raise RuntimeError("Ausfülltool und Bildabdeckung passen nicht zusammen.")
                if not capability["ready"]:
                    raise RuntimeError(capability["message"])
                panorama = folder / "panorama.png"
                coverage = 180 if hemisphere else 360
                set_job(job_id, stage=f"{coverage}°-Panorama wird erstellt")
                flux_end = 65
                if capability.get("extra"):
                    def extra_progress(stage: str, percent: float) -> None:
                        set_job(job_id, stage=stage, progressPercent=round(percent * flux_end / 100))

                    def register(process) -> None:
                        with jobs_lock:
                            if process is None:
                                job_processes.pop(job_id, None)
                                return
                            job_processes[job_id] = process
                            cancelled = job_id in cancel_requests
                        if cancelled:
                            terminate_process_tree(process)

                    try:
                        result = outpaint_tools.run_job(fill_tool, input_path, coverage, folder / "outpaint",
                                                        extra_progress, register=register)
                    except RuntimeError:
                        check_cancelled(job_id)
                        raise
                    check_cancelled(job_id)
                    shutil.copyfile(result, panorama)
                else:
                    generator_script = "fill180_generate.py" if hemisphere else "panorama_generate.py"
                    # Intermediate FLUX patches feed the pipeline view.
                    step_options = ["--save-steps", str(folder / "fill_steps")] if hemisphere else []
                    run_process(job_id, [str(python_command(".venv-panorama")), str(ROOT / generator_script),
                                "--input", str(input_path), "--output", str(panorama), *step_options],
                                log_path, 0, flux_end)
                if not panorama.is_file():
                    raise RuntimeError("Die Panorama-Erweiterung hat kein Bild erstellt.")
                reference_options = []
                if not use_da3 and manifest.get("depthAlignment") == "da2":
                    if not depth_alignment_status()["ready"]:
                        raise RuntimeError(depth_alignment_status()["message"])
                    reference = folder / "da2_distance.npy"
                    set_job(job_id, stage="DA² schätzt die Tiefe des Panoramas", progressPercent=flux_end)
                    run_process(job_id, [str(python_command(".venv-depth")), str(ROOT / "da2_reference.py"),
                                "--input", str(panorama), "--output", str(reference),
                                "--coverage", str(coverage)], log_path, flux_end, flux_end + 5)
                    if not reference.is_file():
                        raise RuntimeError("DA² hat keine Tiefenkarte erstellt.")
                    reference_options = ["--reference-depth", str(reference)]
                    flux_end += 5
                if use_da3:
                    command = [str(python_command(".venv-depth")), str(ROOT / "da3_splats.py"),
                               "--input", str(panorama), "--output", str(folder / "scene.ply"),
                               "--projection", "equirectangular", "--coverage", str(coverage)]
                else:
                    command = [str(python_command()), str(ROOT / "panorama_splats.py"),
                               "--input", str(panorama), "--output", str(folder / "scene.ply"),
                               "--coverage", str(coverage), *sharp_options(manifest), *reference_options,
                               "--save-views"]
                set_job(job_id, stage=f"{coverage}°-Szene wird berechnet", progressPercent=flux_end)
                run_process(job_id, command, log_path, flux_end, 100)
            elif use_da3:
                run_process(job_id, [str(python_command(".venv-depth")), str(ROOT / "da3_splats.py"),
                            "--input", str(input_path), "--output", str(folder / "input.ply"),
                            "--projection", "perspective"], log_path)
            else:
                run_process(job_id, [str(python_command()), str(ROOT / "sharp_predict.py"),
                            "--input", str(input_path), "--output", str(folder / "input.ply"),
                            *sharp_options(manifest)], log_path)
            if not (folder / manifest.get("outputFile", "input.ply")).is_file():
                raise RuntimeError("Es wurde keine 3D-Szene erstellt.")
            set_job(job_id, status="ready", stage="Fertig", progressPercent=100,
                    durationSeconds=round(time.monotonic() - started, 1))
        except JobCancelled:
            set_job(job_id, status="cancelled", stage="Abgebrochen", progressPercent=0)
        except Exception as exc:
            set_job(job_id, status="error", stage="Fehlgeschlagen", error=str(exc))
        finally:
            with jobs_lock:
                cancel_requests.discard(job_id)
            job_queue.task_done()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def json_response(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def file_response(self, path: Path, content_type: str | None = None) -> None:
        if not path.is_file():
            self.json_response(404, {"error": "Datei nicht gefunden"})
            return
        browser_types = {".js": "text/javascript", ".css": "text/css", ".wasm": "application/wasm", ".svg": "image/svg+xml"}
        self.send_response(200)
        self.send_header("Content-Type", content_type or browser_types.get(path.suffix.lower()) or mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(path.stat().st_size))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            with path.open("rb") as stream:
                shutil.copyfileobj(stream, self.wfile, length=1024 * 1024)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/health":
            self.json_response(200, {"ok": True, "panorama": panorama_status(),
                                     "fillTools": fill_tools_status(),
                                     "depthEstimators": depth_estimators_status(),
                                     "sharpResolutions": list(SHARP_RESOLUTIONS),
                                     "defaultSharpResolution": DEFAULT_SHARP_RESOLUTION,
                                     "upscaler": upscaler_status(),
                                     "depthAlignment": depth_alignment_status()})
            return
        if path == "/api/scenes":
            self.json_response(200, {"scenes": [public_scene(scene) for scene in scene_records()]})
            return
        if path == "/api/jobs":
            with jobs_lock:
                pending = [dict(job) for job in jobs.values() if job.get("status") in {"queued", "processing"}]
            self.json_response(200, {"jobs": pending})
            return
        if path.startswith("/api/jobs/"):
            job_id = path.removeprefix("/api/jobs/")
            with jobs_lock:
                job = jobs.get(job_id)
                snapshot = dict(job) if job else None
            self.json_response(200, snapshot) if snapshot else self.json_response(404, {"error": "Auftrag nicht gefunden"})
            return
        pipeline = re.fullmatch(r"/api/pipeline/([a-f0-9]{12})(?:/(seams|files/(.+)))?", path)
        if pipeline:
            self.pipeline_get(pipeline.group(1), pipeline.group(2), pipeline.group(3))
            return
        if path.startswith("/api/scenes/"):
            parts = path.split("/")
            if len(parts) == 5 and parts[4] in {"ply", "image", "panorama"}:
                scene_id = unquote(parts[3])
                scene = next((item for item in scene_records() if item["id"] == scene_id), None)
                if scene and scene.get(parts[4]):
                    self.file_response(scene[parts[4]], "application/octet-stream" if parts[4] == "ply" else None)
                    return
            self.json_response(404, {"error": "Szene nicht gefunden"})
            return

        asset = "index.html" if path == "/" else unquote(path).lstrip("/")
        candidate = (DIST / asset).resolve()
        if candidate.is_relative_to(DIST.resolve()) and candidate.is_file():
            self.file_response(candidate)
        else:
            self.json_response(404, {"error": "Datei nicht gefunden. Zuerst npm run build in web ausführen."})

    def pipeline_get(self, job_id: str, action: str | None, relative: str | None) -> None:
        folder = JOB_OUTPUT / job_id
        if not (folder / "manifest.json").is_file():
            self.json_response(404, {"error": "Auftrag nicht gefunden"})
            return
        if action is None:
            with jobs_lock:
                job = dict(jobs[job_id]) if job_id in jobs else None
            self.json_response(200, pipeline_view.pipeline_record(folder, job_id, job))
        elif action == "seams":
            if not (folder / "scene.ply").is_file():
                self.json_response(409, {"error": "Die Szene ist noch nicht fertig."})
                return
            try:
                self.json_response(200, pipeline_view.measure_seams(folder))
            except (OSError, ValueError, KeyError) as exc:
                self.json_response(422, {"error": f"Nähte konnten nicht gemessen werden: {exc}"})
        else:
            relative = unquote(relative)
            wants_thumbnail = "thumb=1" in urlparse(self.path).query
            try:
                target = pipeline_view.thumbnail(folder, relative) if wants_thumbnail else pipeline_view.resolve_file(folder, relative)
            except OSError:
                target = None
            if target is None:
                self.json_response(404, {"error": "Datei nicht gefunden"})
            else:
                self.file_response(target, "application/octet-stream" if target.suffix == ".ply" else None)

    def cancel_job(self, job_id: str) -> None:
        with jobs_lock:
            job = jobs.get(job_id)
            if not job:
                self.json_response(404, {"error": "Auftrag nicht gefunden"})
                return
            status = job.get("status")
            if status not in {"queued", "processing"}:
                self.json_response(409, {"error": "Der Auftrag läuft nicht mehr.", "status": status})
                return
            cancel_requests.add(job_id)
            process = job_processes.get(job_id)
        if status == "queued":
            set_job(job_id, status="cancelled", stage="Abgebrochen", progressPercent=0)
        else:
            set_job(job_id, stage="Wird abgebrochen …")
            if process:
                terminate_process_tree(process)
        self.json_response(202, {"cancelled": job_id})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        cancel = re.fullmatch(r"/api/jobs/([a-f0-9]{12})/cancel", path)
        if cancel:
            self.cancel_job(cancel.group(1))
            return
        if path != "/api/jobs":
            self.json_response(404, {"error": "Endpunkt nicht gefunden"})
            return
        mode = self.headers.get("X-Mode", "image")
        depth_estimator = self.headers.get("X-Depth-Estimator", "sharp")
        if depth_estimator not in {"sharp", "da3"}:
            self.close_connection = True
            self.json_response(400, {"error": "Unbekanntes 3D-Modell."})
            return
        if depth_estimator == "da3" and not depth_estimators_status()["da3"]["ready"]:
            self.close_connection = True
            self.json_response(503, {"error": depth_estimators_status()["da3"]["message"]})
            return
        if mode not in {"image", "panorama", "hemisphere"}:
            self.close_connection = True
            self.json_response(400, {"error": "Unbekannter Verarbeitungsmodus."})
            return
        fill_tool = None
        if mode != "image":
            fill_tool = self.headers.get("X-Fill-Tool", "flux-fill" if mode == "hemisphere" else "klein-lora")
            capability = fill_tools_status().get(fill_tool)
            if not capability or not fill_tool_matches(capability, mode):
                self.close_connection = True
                self.json_response(400, {"error": "Unbekanntes oder unpassendes Bild-Ausfülltool."})
                return
            if not capability["ready"]:
                self.close_connection = True
                self.json_response(503, {"error": capability["message"]})
                return
        sharp_resolution = None
        upscale = None
        depth_alignment = None
        if depth_estimator == "sharp":
            try:
                sharp_resolution = int(self.headers.get("X-Sharp-Resolution", DEFAULT_SHARP_RESOLUTION))
            except ValueError:
                sharp_resolution = 0
            if sharp_resolution not in SHARP_RESOLUTIONS:
                self.close_connection = True
                self.json_response(400, {"error": "Nicht unterstützte SHARP-Auflösung."})
                return
            upscale = self.headers.get("X-Upscale", "off")
            if upscale not in {"off", "auto"}:
                self.close_connection = True
                self.json_response(400, {"error": "Unbekannte Upscaling-Einstellung."})
                return
            if upscale == "auto" and not upscaler_status()["ready"]:
                self.close_connection = True
                self.json_response(503, {"error": upscaler_status()["message"]})
                return
            depth_alignment = self.headers.get("X-Depth-Alignment", "off")
            if depth_alignment not in {"off", "da2"}:
                self.close_connection = True
                self.json_response(400, {"error": "Unbekannter Tiefenangleich."})
                return
            # DA² needs a whole panorama; single photos keep SHARP's own depth.
            if mode == "image":
                depth_alignment = "off"
            if depth_alignment == "da2" and not depth_alignment_status()["ready"]:
                self.close_connection = True
                self.json_response(503, {"error": depth_alignment_status()["message"]})
                return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if not 0 < length <= MAX_UPLOAD:
            self.json_response(413, {"error": "Das Bild muss kleiner als 30 MB sein."})
            return
        body = self.rfile.read(length)
        try:
            with Image.open(io.BytesIO(body)) as image:
                suffix = IMAGE_FORMATS.get(image.format or "")
                if not suffix:
                    raise ValueError("Bitte JPG, PNG oder WebP verwenden.")
                width, height = image.size
                image.verify()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            self.json_response(400, {"error": str(exc) or "Bild konnte nicht gelesen werden."})
            return

        original_name = Path(unquote(self.headers.get("X-Filename", "Bild"))).name[:160]
        job_id = uuid.uuid4().hex[:12]
        folder = JOB_OUTPUT / job_id
        folder.mkdir(parents=True, exist_ok=False)
        input_name = f"input{suffix}"
        (folder / input_name).write_bytes(body)
        manifest = {
            "id": job_id,
            "name": original_name,
            "inputFile": input_name,
            "outputFile": "input.ply" if mode == "image" else "scene.ply",
            "mode": mode,
            "depthEstimator": depth_estimator,
            "pipeline": "da3-native" if depth_estimator == "da3" else "sharp",
            "fillTool": fill_tool,
            "sharpResolution": sharp_resolution,
            "upscale": upscale,
            "depthAlignment": depth_alignment,
            "created": datetime.now(timezone.utc).isoformat(),
            "width": width,
            "height": height,
        }
        (folder / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        with jobs_lock:
            jobs[job_id] = {"id": job_id, "status": "queued", "stage": "In Warteschlange",
                            "name": original_name, "mode": mode, "depthEstimator": depth_estimator,
                            "pipeline": manifest["pipeline"],
                            "fillTool": fill_tool, "sharpResolution": sharp_resolution,
                            "upscale": upscale, "depthAlignment": depth_alignment}
        set_job(job_id, progressPercent=0)
        job_queue.put(job_id)
        self.json_response(202, {"jobId": job_id})

    def do_DELETE(self) -> None:
        parts = urlparse(self.path).path.split("/")
        if len(parts) != 4 or parts[1:3] != ["api", "scenes"]:
            self.json_response(404, {"error": "Endpunkt nicht gefunden"})
            return
        scene_id = unquote(parts[3])
        if not re.fullmatch(r"[a-f0-9]{12}", scene_id):
            self.json_response(400, {"error": "Diese Szene kann nicht gelöscht werden."})
            return
        scene = next((item for item in scene_records() if item["id"] == scene_id), None)
        if not scene or scene["kind"] != "upload":
            self.json_response(404, {"error": "Nur eigene Uploads können gelöscht werden."})
            return

        folder = (JOB_OUTPUT / scene_id).resolve()
        if folder.parent != JOB_OUTPUT.resolve() or not folder.is_dir():
            self.json_response(404, {"error": "Bild nicht gefunden."})
            return
        with jobs_lock:
            job = jobs.get(scene_id)
            if job and job.get("status") in {"queued", "processing"}:
                self.json_response(409, {"error": "Das Bild wird noch verarbeitet und kann gerade nicht gelöscht werden."})
                return
            try:
                shutil.rmtree(folder)
            except OSError as exc:
                self.json_response(500, {"error": f"Bild konnte nicht gelöscht werden: {exc}"})
                return
            jobs.pop(scene_id, None)
        self.json_response(200, {"deleted": scene_id})


class LocalServer(ThreadingHTTPServer):
    allow_reuse_address = False
    daemon_threads = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Lokales Web-UI für Apple SHARP")
    parser.add_argument("--port", type=int, default=43127)
    parser.add_argument("--open", action="store_true", help="Browser beim Start öffnen")
    args = parser.parse_args()
    if not (DIST / "index.html").is_file():
        raise SystemExit("Web-UI fehlt. Im Ordner web zuerst npm install und npm run build ausführen.")
    sharp_command()
    JOB_OUTPUT.mkdir(parents=True, exist_ok=True)
    url = f"http://127.0.0.1:{args.port}"
    server = LocalServer(("127.0.0.1", args.port), Handler)
    restore_jobs()
    threading.Thread(target=run_jobs, name="sharp-worker", daemon=True).start()
    print(f"SHARP Studio: {url}", flush=True)
    if args.open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
