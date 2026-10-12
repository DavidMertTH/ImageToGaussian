"""Shared helpers for the panorama outpainting tools.

Every tool is a CLI that prints one JSON object per stdout line, the same
protocol the studio already uses: ``{"stage", "progressPercent", "message", ...}``.
Library chatter goes to stderr.
"""

from __future__ import annotations

import contextlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
TOOL_DIR = Path(__file__).resolve().parent
MODELS = ROOT / "models" / "outpaint"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hemisphere_geometry import erp_rays, sample_pinhole  # noqa: E402

PROTOCOL_STDOUT = sys.stdout
DEFAULT_FOV = 65.0
# Diagonal of a 35 mm film frame; EXIF "FocalLengthIn35mmFilm" refers to it.
FILM_DIAGONAL_MM = math.hypot(36, 24)


def emit(stage: str, progress: float, **fields) -> None:
    print(json.dumps({"stage": stage, "progressPercent": round(max(0.0, min(100.0, progress)), 1), **fields},
                     ensure_ascii=False), file=PROTOCOL_STDOUT, flush=True)


def quiet():
    """Send library output to stderr so stdout stays a clean JSON-lines protocol."""
    return contextlib.redirect_stdout(sys.stderr)


def run_cli(main_function) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        main_function()
        return 0
    except Exception as exc:
        import traceback
        traceback.print_exc(file=sys.stderr)
        emit("error", 0, message=str(exc)[:2000], error=str(exc)[:2000], exception=type(exc).__name__)
        return 1


def load_photo(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB"))


def photo_fov(path: Path, default: float = DEFAULT_FOV) -> tuple[float, str]:
    """Horizontal field of view from EXIF (35 mm equivalent focal length), else the default."""
    try:
        with Image.open(path) as image:
            exif = image.getexif()
            width, height = ImageOps.exif_transpose(image).size
        focal = exif.get_ifd(0x8769).get(41989)  # FocalLengthIn35mmFilm
        if focal and float(focal) > 1:
            focal_px = float(focal) * math.hypot(width, height) / FILM_DIAGONAL_MM
            fov = math.degrees(2 * math.atan(width / (2 * focal_px)))
            if 20 <= fov <= 140:
                return round(fov, 2), "exif"
    except (OSError, ValueError, TypeError, AttributeError, KeyError):
        pass
    return default, "default"


def project_photo(photo: np.ndarray, width: int, height: int, coverage: int, fov: float):
    """Pinhole-project the photo into an ERP canvas looking straight ahead.

    Returns the projected RGB (float32) and a boolean mask of covered pixels.
    """
    rays = erp_rays(width, height, coverage)
    projected, known = sample_pinhole(photo.astype(np.float32), rays, fov)
    projected[~known] = 0
    return projected, known


def green_canvas(projected: np.ndarray, known: np.ndarray) -> Image.Image:
    canvas = np.empty(projected.shape, dtype=np.uint8)
    canvas[:] = (0, 255, 0)
    canvas[known] = np.rint(projected[known]).clip(0, 255).astype(np.uint8)
    return Image.fromarray(canvas)


def smooth_extension(projected: np.ndarray, known: np.ndarray) -> np.ndarray:
    """Fill unknown pixels with a blurred continuation of the photo's colours.

    Inversion-based and inpainting pipelines start from the canvas; a colour
    prior instead of black or green keeps lighting consistent at the border.
    """
    import cv2

    height, width = known.shape
    scale = max(1, max(height, width) // 512)
    small = cv2.resize(projected.astype(np.float32), (width // scale, height // scale), interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize((~known).astype(np.uint8), (width // scale, height // scale), interpolation=cv2.INTER_NEAREST)
    filled = cv2.inpaint(np.rint(small).clip(0, 255).astype(np.uint8), small_mask, 24, cv2.INPAINT_TELEA)
    filled = cv2.GaussianBlur(filled.astype(np.float32), (0, 0), 6)
    filled = cv2.resize(filled, (width, height), interpolation=cv2.INTER_CUBIC)
    return np.where(known[..., None], projected, filled).astype(np.float32)


def reinsert_original(generated: np.ndarray, projected: np.ndarray, known: np.ndarray, band: int) -> np.ndarray:
    """Restore the projected photo over a generated panorama with a soft edge.

    The generated image is first colour-matched to the photo so the cross-fade
    does not show a brightness step.
    """
    from scipy.ndimage import distance_transform_edt

    from fill180_generate import match_colors

    generated = match_colors(generated.astype(np.float32), projected, known, np)
    inside = distance_transform_edt(known)
    weight = np.clip(inside / max(band, 1), 0, 1)
    weight = (weight * weight * (3 - 2 * weight))[..., None]
    return generated * (1 - weight) + projected * weight


def hemisphere_from_360(panorama: np.ndarray) -> np.ndarray:
    """Front 180 x 180 degrees of a 2:1 full-sphere ERP (the middle half of its columns)."""
    height, width = panorama.shape[:2]
    return panorama[:, width // 4: width // 4 + width // 2]


def soften_wrap(rgb: np.ndarray, band: int = 16) -> np.ndarray:
    """Blend the left and right edges of a 360 degree ERP so the wrap seam disappears."""
    output = rgb.astype(np.float32).copy()
    band = min(band, rgb.shape[1] // 32)
    for i in range(band):
        weight = 0.5 * (1.0 - i / band) ** 2
        left, right = rgb[:, i].astype(np.float32), rgb[:, -1 - i].astype(np.float32)
        output[:, i] = left * (1 - weight) + right * weight
        output[:, -1 - i] = right * (1 - weight) + left * weight
    return output


def save_png(rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(rgb).clip(0, 255).astype(np.uint8)).save(path)


def write_marker(directory: Path, **fields) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "ready.tmp"
    temporary.write_text(json.dumps({"ready": True, **fields}, indent=2), encoding="utf-8")
    temporary.replace(directory / "ready.json")


def marker_ready(directory: Path) -> bool:
    try:
        return json.loads((directory / "ready.json").read_text(encoding="utf-8")).get("ready") is True
    except (OSError, ValueError):
        return False


def snapshot(repo_id: str, revision: str, local_dir: Path, allow_patterns=None) -> Path:
    """Download a pinned model revision into a plain folder (no Windows symlinks)."""
    from huggingface_hub import snapshot_download

    with quiet():
        snapshot_download(repo_id=repo_id, revision=revision, local_dir=str(local_dir),
                          allow_patterns=allow_patterns, max_workers=4)
    return local_dir
