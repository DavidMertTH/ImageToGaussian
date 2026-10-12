"""Local runtime for native DA3 Nested Giant Gaussian prediction.

This module does not import SHARP, DepthPro, or an Apple checkpoint. DA3's
Any-view Giant network includes its own camera and Gaussian heads; the nested
Metric Large branch provides metric scale. The official refreshed 1.1 weights
use the same architecture as the pinned code's da3nested-giant-large preset.

Models are downloaded only by --download-models. Normal jobs validate and read
local files. Preparation checks imports, the installed preset, tensor headers
and file size without running inference or allocating a full model.

Model loading uses Accelerate parameter-only meta initialization, CPU-mapped
safetensors and load_state_dict(assign=True) before moving weights to CUDA.
Nonpersistent buffers (notably the Gaussian SH mask) initialize normally.
FP32 weights are retained because upstream disables autocast inside its heads.

Code: Apache-2.0; model weights: CC BY-NC-4.0 (noncommercial).
https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE-1.1
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Callable


ROOT = Path(__file__).resolve().parent
MODEL_ID = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
MODEL_REVISION = "b2359bdf726fb44ef62acca04d629dcf158053e7"
MODEL_NAME = "da3nested-giant-large"
MODEL_LICENSE = "CC-BY-NC-4.0"
CODE_REVISION = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
MODEL_CACHE = ROOT / "models" / "depth" / "nested-giant-large"
EXPECTED_WEIGHTS_BYTES = 6759558100
EXPECTED_PARAMETERS = 1689845519
PROTOCOL_STDOUT = sys.stdout
ProgressCallback = Callable[..., None]


def emit(stage: str, progressPercent: float, **fields: Any) -> None:
    """Write JSON Lines even while dependency logs are redirected to stderr."""
    print(json.dumps({
        "stage": stage,
        "progressPercent": round(max(0.0, min(100.0, progressPercent)), 1),
        **fields,
    }, ensure_ascii=False), file=PROTOCOL_STDOUT, flush=True)


def _progress(callback: ProgressCallback | None, stage: str, percent: float, **fields: Any) -> None:
    if callback is not None:
        callback(stage, percent, **fields)


def load_runtime() -> dict[str, Any]:
    """Import installed trusted libraries only; never execute model-repo code."""
    try:
        with contextlib.redirect_stdout(sys.stderr):
            import torch
            import e3nn
            from accelerate import init_empty_weights
            from depth_anything_3.api import DepthAnything3
            from depth_anything_3.cfg import load_config
            from depth_anything_3.registry import MODEL_REGISTRY
            from huggingface_hub import snapshot_download
            from omegaconf import OmegaConf
            from safetensors import safe_open
            from safetensors.torch import load_file
    except Exception as error:
        raise RuntimeError(
            "Der native DA3-Runtime fehlt oder ist inkompatibel. setup-depth.ps1 "
            "ausführen und .venv-depth verwenden. "
            f"({type(error).__name__}: {error})"
        ) from error
    # Only the installed, pinned source configuration is interpreted. The HF
    # JSON is compared as plain data and never passed to its dynamic loader.
    with contextlib.redirect_stdout(sys.stderr):
        expected_config = OmegaConf.to_container(load_config(MODEL_REGISTRY[MODEL_NAME]), resolve=True)
    code_commit = None
    try:
        source = importlib.metadata.distribution("depth-anything-3").read_text("direct_url.json")
        if source:
            code_commit = json.loads(source).get("vcs_info", {}).get("commit_id")
    except (importlib.metadata.PackageNotFoundError, json.JSONDecodeError):
        pass
    if code_commit is not None and code_commit != CODE_REVISION:
        raise RuntimeError("Die installierte DA3-Code-Revision passt nicht. setup-depth.ps1 erneut ausführen.")
    return {
        "torch": torch, "DepthAnything3": DepthAnything3,
        "init_empty_weights": init_empty_weights, "load_file": load_file,
        "safe_open": safe_open, "snapshot_download": snapshot_download,
        "expected_config": expected_config, "code_commit": code_commit,
        "versions": {
            "torch": torch.__version__, "e3nn": e3nn.__version__,
            "accelerate": importlib.metadata.version("accelerate"),
            "depth-anything-3": importlib.metadata.version("depth-anything-3"),
        },
    }


def resolve_model_files(cache_dir: Path | str | None = None) -> tuple[Path, Path]:
    """Resolve local weights/config without any Hugging Face network request."""
    directory = Path(cache_dir or MODEL_CACHE).resolve()
    config_path, weights_path = directory / "config.json", directory / "model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise RuntimeError(
            "Native DA3-Gaussian-Gewichte fehlen. setup-depth.ps1 ausführen oder "
            "'.venv-depth\\Scripts\\python.exe da3_runtime.py --download-models' starten."
        )
    return config_path, weights_path


def validate_model_files(config_path: Path, weights_path: Path, runtime: dict) -> dict[str, Any]:
    """Validate architecture and safetensors headers without loading tensors."""
    if weights_path.stat().st_size != EXPECTED_WEIGHTS_BYTES:
        raise RuntimeError(
            "Native DA3-Gewichte sind unvollständig oder gehören zu einem anderen Modell. "
            "setup-depth.ps1 erneut ausführen."
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_name") != MODEL_NAME or config.get("config") != runtime["expected_config"]:
        raise RuntimeError("Die native DA3-Modellkonfiguration passt nicht zur installierten Code-Revision.")
    anyview = config["config"].get("anyview", {})
    if not all(name in anyview for name in ("gs_head", "gs_adapter", "cam_enc", "cam_dec")):
        raise RuntimeError("Das DA3-Modell enthält keine nativen Gaussian- und Kameraköpfe.")
    if anyview["gs_head"].get("output_dim") != 38 or anyview["gs_adapter"].get("sh_degree") != 2:
        raise RuntimeError("Der native DA3-Gaussian-Head hat eine unerwartete Konfiguration.")
    required_prefixes = (
        "model.da3.gs_head.", "model.da3.cam_enc.", "model.da3.cam_dec.",
        "model.da3.backbone.", "model.da3_metric.",
    )
    with runtime["safe_open"](str(weights_path), framework="pt", device="cpu") as tensors:
        keys = list(tensors.keys())
        for prefix in required_prefixes:
            if not any(key.startswith(prefix) for key in keys):
                raise RuntimeError(f"Native DA3-Gewichte enthalten keinen erwarteten Head: {prefix}")
        total = 0
        for key in keys:
            tensor = tensors.get_slice(key)
            if tensor.get_dtype() != "F32":
                raise RuntimeError("Diese native DA3-Pipeline erwartet die offiziellen FP32-Gewichte.")
            count = 1
            for dimension in tensor.get_shape():
                count *= dimension
            total += count
    if total != EXPECTED_PARAMETERS:
        raise RuntimeError("Die Zahl der DA3-Gewichtselemente passt nicht zum gepinnten Modell.")
    return {
        "weightsBytes": EXPECTED_WEIGHTS_BYTES,
        "weightElements": total, "tensorCount": len(keys),
        "configSha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    }


def _write_marker(directory: Path, data: dict[str, Any]) -> Path:
    marker = directory.parent / "native-ready.json"
    temporary = marker.with_name(marker.name + ".partial")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    # A polling UI can briefly hold the old file open on Windows.
    for attempt in range(10):
        try:
            temporary.replace(marker)
            return marker
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.1)
    return marker


def download_models(cache_dir: Path | str | None = None, progress_callback: ProgressCallback | None = emit) -> dict:
    """Download official pinned data only; no inference/model allocation."""
    directory = Path(cache_dir or MODEL_CACHE).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    runtime = load_runtime()
    _progress(progress_callback, "download", 0, message="Native DA3-Gaussian-Gewichte werden geladen", modelId=MODEL_ID)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            runtime["snapshot_download"](
                repo_id=MODEL_ID, revision=MODEL_REVISION, local_dir=str(directory),
                allow_patterns=["config.json", "model.safetensors"], max_workers=2,
            )
    except Exception as error:
        raise RuntimeError(f"Der native DA3-Modell-Download ist fehlgeschlagen: {error}") from error
    config_path, weights_path = resolve_model_files(directory)
    _progress(progress_callback, "validate", 95, message="Native Gaussian-Heads und Modellkonfiguration werden geprüft")
    validation = validate_model_files(config_path, weights_path, runtime)
    marker = {
        "ready": True, "pipeline": "da3-native", "modelDirectory": directory.name,
        "weightsPath": directory.name + "/model.safetensors",
        "modelId": MODEL_ID, "modelRevision": MODEL_REVISION,
        "modelName": MODEL_NAME, "modelLicense": MODEL_LICENSE,
        "codeRevision": CODE_REVISION, "installedCodeRevision": runtime["code_commit"],
        "gaussianHead": True, "cameraHead": True, "metricDepth": True,
        "versions": runtime["versions"], **validation,
    }
    marker_path = _write_marker(directory, marker)
    _progress(progress_callback, "ready", 100, message="Native DA3-Gaussians sind vorbereitet", marker=str(marker_path), **marker)
    return marker


def load_native_model(
    device: Any = "cuda", cache_dir: Path | str | None = None,
    progress_callback: ProgressCallback | None = None,
):
    """Return a local DepthAnything3 model with its native learned GS head.

    Callback: callback(stage: str, progressPercent: float, **fields).
    The returned network is FP32/eval; its forward handles mixed precision.
    No pretrained SHARP component or checkpoint is created or consulted.
    """
    runtime = load_runtime()
    torch = runtime["torch"]
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA ist für die native DA3-Pipeline nicht verfügbar.")
    config_path, weights_path = resolve_model_files(cache_dir)
    _progress(progress_callback, "validate", 2, message="Lokale native DA3-Gewichte werden geprüft")
    validate_model_files(config_path, weights_path, runtime)
    _progress(progress_callback, "load", 5, message="Native DA3-Gaussian-Architektur wird vorbereitet")
    try:
        with contextlib.redirect_stdout(sys.stderr):
            # Parameter-only meta keeps constructor scalar computations on CPU
            # and retains initialized nonpersistent SH buffers. A plain global
            # torch.device('meta') breaks DINO's linspace(...).item() constructor.
            with runtime["init_empty_weights"](include_buffers=False):
                model = runtime["DepthAnything3"](model_name=MODEL_NAME)
            _progress(progress_callback, "load", 8, message="DA3-Gewichte werden ohne zweite vollständige RAM-Kopie eingebunden")
            state = runtime["load_file"](str(weights_path), device="cpu")
            model.load_state_dict(state, strict=True, assign=True)
            del state
            unresolved = [name for name, value in model.named_parameters() if value.device.type == "meta"]
            if unresolved:
                raise RuntimeError("Nicht geladene native DA3-Parameter: " + ", ".join(unresolved[:3]))
            _progress(progress_callback, "load", 12, message="Native DA3-Gewichte werden zur GPU übertragen")
            model = model.to(device=device, dtype=torch.float32).eval()
            model.requires_grad_(False)
    except torch.cuda.OutOfMemoryError as error:
        gc.collect()
        torch.cuda.empty_cache()
        raise RuntimeError(
            "Nicht genügend GPU-Speicher für DA3 Nested Giant. Die FP32-Gewichte "
            "benötigen rund 6,3 GiB zuzüglich Inferenzspeicher. Andere GPU-Anwendungen "
            "schließen oder --device cpu verwenden."
        ) from error
    _progress(progress_callback, "loaded", 15, message="Native DA3-Gaussian-Heads sind geladen", modelId=MODEL_ID)
    return model


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-models", action="store_true", help="Only download and validate native DA3 models")
    parser.add_argument("--cache-dir", type=Path, default=MODEL_CACHE)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if not args.download_models:
        parser.error("Use --download-models for preparation; image jobs run through da3_splats.py")
    try:
        download_models(args.cache_dir)
    except Exception as error:
        emit("error", 0, status="error", error=str(error))
        if args.debug:
            traceback.print_exc(file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
