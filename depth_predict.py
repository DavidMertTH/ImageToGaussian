"""Run DA3 Metric Large in the isolated .venv-depth environment.

    python depth_predict.py --download-models
    python depth_predict.py --input photo.jpg --output depth.npz
    python depth_predict.py --input panorama.png --output depth.npz \
        --projection equirectangular

The model and code revisions are pinned. Normal jobs use local files only;
downloads are exclusively performed by --download-models. Nothing is inferred
during model preparation. The 334M-parameter monocular metric model is selected
to keep six sequential perspective predictions practical on a 16-GB GPU.

Output depth is CAMERA-Z IN METERS, not disparity or radial distance. DA3Metric
raw canonical Z is multiplied ONCE by the focal length at the actual processed
image resolution and divided by 300, as documented by the official repository.

For ERP, six known 100-degree cameras share the same origin. Each is inferred
independently: this metric model has no multi-view attention/pose conditioning.
The optional common ERP map is a heuristic blend of radial metric distances,
not jointly inferred depth. It reduces discontinuities but cannot guarantee
correct geometry or metric scale.

Primary sources:
https://github.com/ByteDance-Seed/Depth-Anything-3#-faq
https://huggingface.co/depth-anything/DA3METRIC-LARGE
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from depth_estimation import (
    CODE_REVISION,
    MODEL_ID,
    MODEL_LICENSE,
    MODEL_REVISION,
    DepthBundle,
    build_radial_consensus,
    cube_views,
    estimate_focal_length,
    make_intrinsics,
    sample_perspective_rgb,
)


ROOT = Path(__file__).resolve().parent
PROTOCOL_STDOUT = sys.stdout
EXPECTED_WEIGHTS_BYTES = 1336734448
EXPECTED_PARAMETERS = 334171394


def emit(stage: str, progress: float, **details: Any) -> None:
    """Keep stdout as newline-delimited JSON, including dependency import logs."""
    print(
        json.dumps(
            {"stage": stage, "progressPercent": round(max(0, min(100, progress)), 1), **details},
            ensure_ascii=False,
        ),
        file=PROTOCOL_STDOUT,
        flush=True,
    )


def load_runtime() -> dict[str, Any]:
    """Import only from the dedicated depth environment, without model inference."""
    try:
        with contextlib.redirect_stdout(sys.stderr):
            import torch
            from depth_anything_3.api import DepthAnything3
            from huggingface_hub import snapshot_download
            from safetensors import safe_open
    except Exception as error:
        raise RuntimeError(
            "DA3-Pakete fehlen oder sind inkompatibel. Bitte setup-depth.ps1 ausführen; "
            "depth_predict.py gehört in die getrennte .venv-depth Umgebung. "
            f"({type(error).__name__}: {error})"
        ) from error
    try:
        package_version = importlib.metadata.version("depth-anything-3")
    except importlib.metadata.PackageNotFoundError:
        package_version = "unknown"
    return {
        "torch": torch,
        "DepthAnything3": DepthAnything3,
        "snapshot_download": snapshot_download,
        "safe_open": safe_open,
        "versions": {"torch": torch.__version__, "depthAnything3": package_version},
    }


def resolve_model(cache_dir: Path, runtime: dict[str, Any], *, download: bool = False) -> Path:
    """Resolve pinned local weights and validate small metadata/safetensors headers."""
    model_path = cache_dir / "da3metric-large"
    if download:
        model_path.mkdir(parents=True, exist_ok=True)
        with contextlib.redirect_stdout(sys.stderr):
            runtime["snapshot_download"](
                repo_id=MODEL_ID,
                revision=MODEL_REVISION,
                local_dir=str(model_path),
                allow_patterns=["config.json", "model.safetensors", "README.md", "LICENSE", "*.json"],
                max_workers=2,
            )
    config_path = model_path / "config.json"
    weights_path = model_path / "model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise RuntimeError(
            "DA3-Modellgewichte fehlen im lokalen Cache. Bitte setup-depth.ps1 oder "
            "'.venv-depth\\Scripts\\python.exe depth_predict.py --download-models' ausführen."
        )
    configuration = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(configuration, dict):
        raise RuntimeError("Ungültige DA3-Modellkonfiguration.")
    model_name = configuration.get("model_name")
    if model_name is not None and str(model_name).lower() != "da3metric-large":
        raise RuntimeError("Der lokale Cache enthält nicht das erwartete DA3METRIC-LARGE Modell.")
    if weights_path.stat().st_size != EXPECTED_WEIGHTS_BYTES:
        raise RuntimeError("DA3-Modellgewichte sind unvollständig oder gehören zu einer anderen Revision.")
    with runtime["safe_open"](str(weights_path), framework="pt", device="cpu") as handle:
        if not handle.keys():
            raise RuntimeError("Die DA3-Safetensors-Datei enthält keine Tensoren.")
    return model_path


def download_models(args: argparse.Namespace, runtime: dict[str, Any]) -> None:
    """Download weights and publish readiness after import/header checks only."""
    emit("download", 0, message="Depth Anything 3 Metric Large wird heruntergeladen")
    model_path = resolve_model(args.cache_dir, runtime, download=True)
    emit("validate", 95, message="DA3-Konfiguration und Modellheader werden geprüft")
    # load_runtime already verified that the official DepthAnything3 API imports.
    # resolve_model opened the safetensors header; weights were not instantiated.
    marker = {
        "ready": True,
        "modelId": MODEL_ID,
        "modelRevision": MODEL_REVISION,
        "modelPath": str(model_path),
        "codeRevision": CODE_REVISION,
        "modelLicense": MODEL_LICENSE,
        "parameterCount": EXPECTED_PARAMETERS,
        "weightsBytes": EXPECTED_WEIGHTS_BYTES,
        "versions": runtime["versions"],
        "inferencePerformed": False,
    }
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    ready_path = args.cache_dir / "ready.json"
    temporary = ready_path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(ready_path)
    emit("ready", 100, message="DA3 ist lokal verfügbar", cacheDir=str(args.cache_dir), status="ready")


def resolve_device(requested: str, torch: Any) -> Any:
    """Select CUDA when available, without hiding an explicitly requested failure."""
    if requested in ("auto", "default"):
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA ist nicht verfügbar. Nutze --device auto oder --device cpu.")
    return device


def run(args: argparse.Namespace, runtime: dict[str, Any]) -> dict[str, Any]:
    """Predict metric Z maps and optionally geometrically fuse a radial ERP map."""
    started = time.monotonic()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if output_path.suffix.lower() != ".npz":
        output_path = output_path / "depth.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    emit("depth_prepare", 0, message="Bild für DA3 wird vorbereitet", status="processing")
    with Image.open(input_path) as source:
        oriented = ImageOps.exif_transpose(source)
        image = oriented.convert("RGB")
        width, height = image.size
        original_focal = args.focal_length_px or estimate_focal_length(source, image.size)
        rgb = np.asarray(image)
    rotations: list[np.ndarray]
    if args.projection == "equirectangular":
        if abs(width / height - 2) > 0.02:
            raise ValueError("Ein vollständiges equirectangulares Panorama im 2:1-Format ist nötig.")
        views = cube_views()
        focal = args.view_resolution / (2 * math.tan(math.radians(args.view_fov) / 2))
        images = [sample_perspective_rgb(rgb, view.rotation, args.view_resolution, focal) for view in views]
        rotations = [view.rotation for view in views]
        names = tuple(view.name for view in views)
        originals = [make_intrinsics(args.view_resolution, args.view_resolution, focal) for _ in views]
    else:
        images = [rgb]
        rotations = [np.eye(3, dtype=np.float32)]
        names = ("front",)
        originals = [make_intrinsics(width, height, original_focal)]
    device = resolve_device(args.device, runtime["torch"])
    emit("depth_load", 5, message="DA3-Modell wird geladen", device=str(device), viewCount=len(images))
    model_path = resolve_model(args.cache_dir, runtime)
    with contextlib.redirect_stdout(sys.stderr):
        model = runtime["DepthAnything3"].from_pretrained(str(model_path), local_files_only=True)
        model.eval().to(device=device)
    depths, masks, intrinsics, confidences, records = [], [], [], [], []
    with runtime["torch"].inference_mode():
        for index, (perspective, original_k) in enumerate(zip(images, originals)):
            emit(
                "depth",
                10 + 65 * index / len(images),
                message=f"DA3: Tiefenschätzung {index + 1}/{len(images)}",
                view=names[index],
                viewIndex=index + 1,
                viewCount=len(images),
            )
            # Metric Large is monocular. Infer separately to avoid incorrectly
            # suggesting that an image batch makes its geometry multi-view.
            # No extrinsics are passed: Metric Large has no pose-conditioning head.
            with contextlib.redirect_stdout(sys.stderr):
                prediction = model.inference(
                    image=[perspective],
                    intrinsics=original_k[None],
                    process_res=args.process_resolution,
                    process_res_method="upper_bound_resize",
                    infer_gs=False,
                    export_dir=None,
                )
            canonical = np.asarray(prediction.depth, dtype=np.float32)
            if canonical.ndim != 3 or canonical.shape[0] != 1:
                raise RuntimeError("Unerwartete DA3-Tiefenform; [1,height,width] wurde erwartet.")
            processed_height, processed_width = canonical.shape[1:]
            source_height, source_width = perspective.shape[:2]
            processed_k = original_k.copy()
            processed_k[0] *= processed_width / source_width
            processed_k[1] *= processed_height / source_height
            # API output_processor does not convert the mono metric model's
            # canonical output into meters. K changes with BOTH resize steps,
            # including the final multiple-of-14 patch resolution adjustment.
            processed_focal = float((processed_k[0, 0] + processed_k[1, 1]) / 2)
            canonical_scale = 1.0 if getattr(prediction, "is_metric", 0) else processed_focal / 300.0
            metric_depth = canonical[0] * canonical_scale
            valid = np.isfinite(metric_depth) & (metric_depth > 0)
            sky = getattr(prediction, "sky", None)
            if sky is not None:
                sky_mask = np.asarray(sky)[0].astype(bool)
                if sky_mask.shape != metric_depth.shape:
                    raise RuntimeError("Unerwartete DA3-Himmelsmaskenform.")
                # DA3 gives sky an arbitrary far depth. Do not use that value as
                # an observed metric surface to deform SHARP's Gaussians.
                valid &= ~sky_mask
            metric_depth = np.where(valid, metric_depth, np.nan).astype(np.float32)
            confidence = getattr(prediction, "conf", None)
            if confidence is not None:
                confidence = np.asarray(confidence, dtype=np.float32)[0]
                if confidence.shape != metric_depth.shape:
                    raise RuntimeError("Unerwartete DA3-Konfidenzform.")
            depths.append(metric_depth)
            masks.append(valid)
            intrinsics.append(processed_k)
            confidences.append(confidence)
            records.append({
                "name": names[index],
                "sourceResolution": [source_width, source_height],
                "processedResolution": [processed_width, processed_height],
                "originalIntrinsics": original_k.tolist(),
                "processedIntrinsics": processed_k.tolist(),
                "canonicalToMetersScale": canonical_scale,
                "cameraToWorld": rotations[index].tolist(),
                "validPixelFraction": float(valid.mean()),
            })
            del prediction
    del model
    if device.type == "cuda":
        runtime["torch"].cuda.empty_cache()
    confidence_maps = (
        np.stack(confidences).astype(np.float32)
        if all(value is not None for value in confidences)
        else None
    )
    bundle = DepthBundle(
        depth=np.stack(depths),
        intrinsics=np.stack(intrinsics),
        camera_to_world=np.stack(rotations),
        view_names=names,
        valid=np.stack(masks),
        confidence=confidence_maps,
    )
    if args.projection == "equirectangular" and not args.no_consensus:
        emit("depth_fuse", 82, message="Radiale Tiefenkarten werden geometrisch abgeglichen")
        bundle.radial_depth = build_radial_consensus(
            bundle, width=args.consensus_width, height=args.consensus_width // 2
        )
    limitation = (
        "Sechs unabhängig geschätzte metrische Perspektivkarten; die gemeinsame radiale Karte "
        "ist eine geometrische Überblendung, keine gemeinsame DA3-Multiview-Schätzung. "
        "Unterschiedliche Objektgeometrie und Tiefensprünge können erhalten bleiben."
        if args.projection == "equirectangular"
        else "Monokular geschätzte metrische Kamera-Z-Tiefe; absolute Entfernungen sind nicht garantiert."
    )
    metadata = {
        "version": 1,
        "estimator": "da3metric-large",
        "modelId": MODEL_ID,
        "modelRevision": MODEL_REVISION,
        "codeRevision": CODE_REVISION,
        "modelLicense": MODEL_LICENSE,
        "parameterCount": EXPECTED_PARAMETERS,
        "depthDefinition": "camera_z_meters",
        "radialDefinition": "distance_from_common_origin_meters",
        "projection": args.projection,
        "input": str(input_path),
        "output": str(output_path),
        "originalResolution": [width, height],
        "originalFocalLengthPx": original_focal if args.projection == "perspective" else None,
        "originalIntrinsics": originals[0].tolist() if args.projection == "perspective" else None,
        "viewResolution": args.view_resolution if args.projection == "equirectangular" else None,
        "viewFovDegrees": args.view_fov if args.projection == "equirectangular" else None,
        "independentViews": True,
        "jointMultiviewInference": False,
        "radialConsensus": bundle.radial_depth is not None,
        "radialConsensusMethod": "frustum_fade_robust_log_mean" if bundle.radial_depth is not None else None,
        "worldCoordinates": "OpenCV +X right +Y down +Z forward",
        "origin": [0, 0, 0],
        "views": records,
        "invalidValues": "NaN; includes sky and invalid model predictions",
        "limitation": limitation,
        "durationSeconds": round(time.monotonic() - started, 2),
        "versions": runtime["versions"],
    }
    fields = {
        "depth": bundle.depth,
        "valid": bundle.valid,
        "intrinsics": bundle.intrinsics,
        "camera_to_world": bundle.camera_to_world,
        "view_names": np.asarray(bundle.view_names),
        "metadata_json": np.asarray(json.dumps(metadata, ensure_ascii=False)),
    }
    if bundle.confidence is not None:
        fields["confidence"] = bundle.confidence
    if bundle.radial_depth is not None:
        fields["radial_depth"] = bundle.radial_depth
    emit("depth_save", 95, message="Metrische Tiefenkarten werden gespeichert")
    temporary = output_path.with_name(output_path.name + ".partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **fields)
    temporary.replace(output_path)
    manifest = output_path.with_suffix(".json")
    temporary_manifest = manifest.with_name(manifest.name + ".partial")
    temporary_manifest.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_manifest.replace(manifest)
    emit(
        "ready", 100, message="DA3-Tiefenschätzung ist fertig", status="ready",
        output=str(output_path), durationSeconds=metadata["durationSeconds"],
        viewCount=len(images), limitation=limitation,
    )
    return metadata


def main() -> int:
    """Run model preparation or local inference through the JSON line protocol."""
    if hasattr(PROTOCOL_STDOUT, "reconfigure"):
        PROTOCOL_STDOUT.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--download-models", action="store_true", help="Download/import/header-check; no inference.")
    parser.add_argument("--input", type=Path, help="Perspective photo or complete 2:1 ERP panorama.")
    parser.add_argument("--output", type=Path, help="Output .npz file or directory.")
    parser.add_argument("--projection", choices=("perspective", "equirectangular"), default="perspective")
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0 or cpu.")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "models" / "depth")
    parser.add_argument("--focal-length-px", type=float, help="Known original focal; default mirrors SHARP EXIF heuristic.")
    parser.add_argument("--view-resolution", type=int, default=1024)
    parser.add_argument("--view-fov", type=float, default=100)
    parser.add_argument("--process-resolution", "--process-res", type=int, default=504)
    parser.add_argument("--consensus-width", type=int, default=1024)
    parser.add_argument("--no-consensus", action="store_true", help="Keep independent ERP view maps without radial fusion.")
    args = parser.parse_args()
    if not args.download_models and (args.input is None or args.output is None):
        parser.error("--input and --output are required unless --download-models is used")
    if args.focal_length_px is not None and (not math.isfinite(args.focal_length_px) or args.focal_length_px <= 0):
        parser.error("--focal-length-px must be finite and positive")
    if not 256 <= args.view_resolution <= 2048 or not 92 <= args.view_fov <= 120:
        parser.error("--view-resolution must be256..2048 and --view-fov92..120")
    if not 196 <= args.process_resolution <= 1008:
        parser.error("--process-resolution must be between196 and1008")
    if args.consensus_width < 256 or args.consensus_width > 4096 or args.consensus_width % 2:
        parser.error("--consensus-width must be an even number256..4096")
    args.cache_dir = args.cache_dir.resolve()
    try:
        runtime = load_runtime()
        if args.download_models:
            download_models(args, runtime)
        else:
            run(args, runtime)
        return 0
    except Exception as error:
        emit("error", 0, message=str(error), status="error", error=str(error))
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
