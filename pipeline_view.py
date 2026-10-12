"""Describe a studio job's intermediate results as a timeline of pipeline steps.

The job folder is the single source of truth: a step is done once its result
file exists, so the timeline also works for jobs that are still running, failed
or were created before this view existed (missing intermediates are "skipped").
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from PIL import Image

# File types the pipeline view may serve from a job folder.
SERVABLE = {".jpg", ".jpeg", ".png", ".webp", ".ply", ".json"}
THUMBNAIL_SIZE = 360
VIEW_ORDER = ("front", "right", "back", "left", "up", "down")
VIEW_TITLES = {"front": "Vorne", "right": "Rechts", "back": "Hinten", "left": "Links", "up": "Oben", "down": "Unten"}
FILL_KINDS = (("input", "FLUX-Eingabe"), ("mask", "Maske"), ("flux", "FLUX-Ergebnis"), ("canvas", "Zwischenstand"))
VIEW_KINDS = (("jpg", "Ansicht"), ("sharp-depth.png", "SHARP-Tiefe"), ("reference-depth.png", "DA²-Vorgabe"))


def iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def resolve_file(folder: Path, relative: str) -> Path | None:
    """Return a servable file inside the job folder, or None."""
    try:
        candidate = (folder / relative).resolve()
    except (OSError, ValueError):
        return None
    if not candidate.is_relative_to(folder.resolve()) or candidate.suffix.lower() not in SERVABLE:
        return None
    return candidate if candidate.is_file() else None


def thumbnail(folder: Path, relative: str) -> Path | None:
    """A cached small JPEG of an image artifact; regenerated when the source changes."""
    source = resolve_file(folder, relative)
    if not source or source.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
        return None
    target = folder / "thumbs" / (relative.replace("/", "__").rsplit(".", 1)[0] + ".jpg")
    if target.is_file() and target.stat().st_mtime >= source.stat().st_mtime:
        return target
    target.parent.mkdir(exist_ok=True)
    with Image.open(source) as image:
        image.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE))
        temporary = target.with_suffix(".partial.jpg")
        image.convert("RGB").save(temporary, quality=85)
    temporary.replace(target)
    return target


class Builder:
    def __init__(self, folder: Path, job_id: str):
        self.folder = folder
        self.base = f"/api/pipeline/{quote(job_id, safe='')}/files/"

    def artifact(self, relative: str, label: str) -> dict | None:
        path = self.folder / relative
        if not path.is_file():
            return None
        stat = path.stat()
        url = self.base + quote(relative) + f"?v={int(stat.st_mtime)}"
        kind = "splat" if path.suffix == ".ply" else "image"
        result = {"label": label, "kind": kind, "url": url, "path": relative,
                  "finishedAt": iso(stat.st_mtime), "sizeBytes": stat.st_size}
        if kind == "image":
            result["thumbnailUrl"] = url + "&thumb=1"
        return result

    def artifacts(self, *entries: tuple[str, str]) -> list[dict]:
        return [item for item in (self.artifact(path, label) for path, label in entries) if item]


def fill_groups(builder: Builder, directory: str) -> list[dict]:
    """One group per FLUX patch: '<n>-<Name>.<kind>.<ext>' files."""
    folder = builder.folder / directory
    if not folder.is_dir():
        return []
    names = sorted({path.name.split(".")[0] for path in folder.iterdir() if "-" in path.name},
                   key=lambda name: int(name.split("-")[0]) if name.split("-")[0].isdigit() else 99)
    groups = []
    for name in names:
        artifacts = builder.artifacts(*(
            (f"{directory}/{name}.{kind}{'.png' if kind == 'mask' else '.jpg'}", label) for kind, label in FILL_KINDS
        ))
        if artifacts:
            groups.append({"title": f"Ausschnitt {name.replace('-', ' · ', 1)}", "artifacts": artifacts})
    return groups


def view_groups(builder: Builder, directory: str) -> list[dict]:
    """One group per SHARP view saved by panorama_splats.py --save-views."""
    folder = builder.folder / directory
    if not folder.is_dir():
        return []
    groups = []
    for name in VIEW_ORDER:
        artifacts = builder.artifacts(*((f"{directory}/{name}.{suffix}", label) for suffix, label in VIEW_KINDS))
        gaussians = builder.artifact(f"{directory}/{name}.ply", "Gaussians")
        if gaussians:
            gaussians["view"] = name
            artifacts.append(gaussians)
        if not artifacts:
            continue
        stats = read_json(folder / f"{name}.json") or {}
        meta = {}
        if "kept" in stats:
            meta["Gaussians"] = f"{stats['kept']:,}".replace(",", ".")
        if "sharpToReferenceDepthRatio" in stats:
            meta["SHARP-Tiefe ÷ DA²"] = f"{stats['sharpToReferenceDepthRatio']:.2f}".replace(".", ",")
        groups.append({"title": VIEW_TITLES[name], "artifacts": artifacts, "meta": meta, "done": gaussians is not None})
    return groups


def step(key: str, title: str, description: str, done: bool, artifacts=(), groups=(), meta=None) -> dict:
    return {"key": key, "title": title, "description": description, "done": done,
            "artifacts": list(artifacts), "groups": list(groups), "meta": meta or {}}


def build_steps(folder: Path, manifest: dict, builder: Builder) -> list[dict]:
    mode = manifest.get("mode", "image")
    estimator = manifest.get("depthEstimator", "sharp")
    input_file = manifest.get("inputFile", "input.jpg")
    steps = [step("input", "Foto", "Das hochgeladene Originalbild.",
                  (folder / input_file).is_file(), builder.artifacts((input_file, "Original")))]
    if mode not in {"panorama", "hemisphere"}:
        model = "DA3" if estimator == "da3" else "SHARP"
        steps.append(step("scene", "3D-Szene", f"{model} berechnet die Gaussians direkt aus dem Foto.",
                          (folder / "input.ply").is_file(), builder.artifacts(("input.ply", "Gaussians"))))
        return steps

    hemisphere = mode == "hemisphere"
    coverage = 180 if hemisphere else 360
    steps.append(step(
        "projection", "Projektion",
        "Das Foto wird mit angenommenen 65° Bildwinkel auf die vordere Halbkugel projiziert; schwarze Bereiche werden ergänzt."
        if hemisphere else "Das Foto wird als Perspektivansicht in die Kugelprojektion eingesetzt.",
        (folder / "panorama.control.png").is_file(), builder.artifacts(("panorama.control.png", "Projektion")),
    ))
    panorama_done = (folder / "panorama.png").is_file()
    if hemisphere:
        steps.append(step(
            "fill", "Ausfüllen mit FLUX Fill",
            "Fünf überlappende 120°-Ausschnitte werden nacheinander ergänzt und auf die Halbkugel zurückprojiziert.",
            panorama_done, groups=fill_groups(builder, "fill_steps"),
        ))
    else:
        steps.append(step(
            "fill", "Ausfüllen mit FLUX.2 Klein",
            "FLUX.2 Klein mit Panorama-LoRA erzeugt die Rundumsicht in einem Durchlauf.",
            panorama_done, builder.artifacts(("panorama.generated.png", "FLUX-Ergebnis")),
        ))
    panorama_meta = read_json(folder / "panorama.json") or {}
    steps.append(step(
        "panorama", f"{coverage}°-Panorama", "Das fertige equirectangulare Bild, aus dem die 3D-Szene entsteht.",
        panorama_done, builder.artifacts(("panorama.png", "Panorama")),
        meta={"Seed": str(panorama_meta["seed"])} if "seed" in panorama_meta else None,
    ))
    if estimator == "sharp" and manifest.get("depthAlignment") == "da2":
        steps.append(step(
            "da2", "DA²-Tiefe", "DA² schätzt die Entfernung für das ganze Panorama in einem Durchlauf (hell ist nah).",
            (folder / "da2_distance.npy").is_file(), builder.artifacts(("da2_distance.png", "Entfernung")),
        ))
    scene_done = (folder / "scene.ply").is_file()
    if estimator == "sharp":
        steps.append(step(
            "views", "SHARP-Ansichten",
            "Jede Perspektivansicht wird einzeln mit SHARP berechnet. Die Tiefenbilder sind pro Ansicht gleich skaliert (hell ist nah).",
            scene_done, groups=view_groups(builder, "scene_views"),
        ))
    scene_meta = read_json(folder / "scene.json") or {}
    meta = {}
    if "pointCount" in scene_meta:
        meta["Gaussians"] = f"{scene_meta['pointCount']:,}".replace(",", ".")
    if "durationSeconds" in scene_meta:
        meta["Rechenzeit"] = f"{scene_meta['durationSeconds']:.0f} s"
    steps.append(step(
        "scene", "3D-Szene", "Alle Ansichten werden um denselben Ursprung zur fertigen Szene zusammengeführt.",
        scene_done, builder.artifacts(("scene.ply", "Gaussians")), meta=meta,
    ))
    return steps


def pipeline_record(folder: Path, job_id: str, job: dict | None) -> dict:
    manifest = read_json(folder / "manifest.json")
    if manifest is None:
        raise FileNotFoundError(job_id)
    job = job or read_json(folder / "job.json") or {}
    steps = build_steps(folder, manifest, Builder(folder, job_id))
    output = folder / manifest.get("outputFile", "input.ply")
    status = job.get("status") or ("ready" if output.is_file() else "unknown")
    if status in {"queued", "processing"} and output.is_file():
        status = "ready"
    current = next((item for item in steps if not item["done"]), None)
    for item in steps:
        if item["done"]:
            item["status"] = "done"
        elif item is current and status in {"queued", "processing"}:
            item["status"] = "running"
            item["stage"] = job.get("stage")
        elif item is current and status in {"error", "cancelled"}:
            item["status"] = status
            item["error"] = job.get("error") or ("Abgebrochen" if status == "cancelled" else None)
        else:
            item["status"] = "pending" if status in {"queued", "processing"} else "skipped"
        finished = [artifact["finishedAt"] for artifact in item["artifacts"]]
        finished += [artifact["finishedAt"] for group in item["groups"] for artifact in group["artifacts"]]
        item["finishedAt"] = max(finished) if item["done"] and finished else None
    log_tail = None
    if status == "error":
        try:
            lines = (folder / "pipeline.log").read_text(encoding="utf-8", errors="replace").splitlines()
            log_tail = "\n".join(line for line in lines[-40:] if not line.startswith('{"'))[-4000:]
        except OSError:
            pass
    seams = read_json(folder / "seams.json")
    return {
        "id": job_id, "name": manifest.get("name"), "mode": manifest.get("mode", "image"),
        "depthEstimator": manifest.get("depthEstimator", "sharp"),
        "depthAlignment": manifest.get("depthAlignment"), "status": status,
        "stage": job.get("stage"), "progressPercent": job.get("progressPercent"),
        "error": job.get("error"), "created": manifest.get("created"), "startedAt": job.get("startedAt"),
        "steps": steps, "logTail": log_tail, "seams": seams,
        "seamsAvailable": (folder / "scene.ply").is_file() and bool((read_json(folder / "scene.json") or {}).get("views")),
    }


def measure_seams(folder: Path) -> dict:
    """Run seam_report on the finished scene and cache the result."""
    from seam_report import report

    result = report(folder / "scene.ply", bins=360, percentile=20, minimum=8)
    result.pop("scene", None)
    (folder / "seams.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result
