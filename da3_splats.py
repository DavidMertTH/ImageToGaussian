"""Native DA3 Gaussian inference for a perspective image or surrounding ERP.

Run with .venv-depth; this module never imports SHARP or Apple Depth Pro.
DA3 Nested Giant predicts the Gaussian attributes with its learned GS head and
uses its own Metric Large branch for scale. Views run independently, one at a
time, so 360/180 exports do not claim jointly inferred or seamless geometry.

Depth definitions are deliberately kept separate. DA3's ordinary depth export
is camera-Z, but its pinned GaussianAdapter places means on *normalized* rays
at depth + a learned offset. Native Gaussian radius is preserved by an angular
intrinsic calibration, with exact first-order covariance transport. Nested's
scale_factor is applied to the native GS attributes once; a known/predicted
focal ratio then corrects the metric branch's use of predicted intrinsics.
This calibration is experimental and does not make monocular estimates exact.

The output uses OpenCV axes (+X right, +Y down, +Z forward), quaternion wxyz,
log standard deviations, logit opacity, and native SH DC coefficients. It is
compatible with the existing Spark viewer's 180-degree X mesh rotation. Only
the invariant DC band is exported; the learned view-dependent SH bands are
discarded rather than incorrectly rotated after the nonlinear calibration.

The official API is called with export_dir=None: its GS file export otherwise
also requests GS video rendering, which would require the gsplat CUDA build.
No gsplat renderer is used here. Default process_res=352 (rounded by DA3 to a
multiple of 14) keeps one-view activations smaller for a 16-GB GPU; no hardware
memory guarantee is made. Weights are provided by da3_runtime.py.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
import gc
import json
import math
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from depth_estimation import cube_views, estimate_focal_length, make_intrinsics, sample_perspective_rgb
from hemisphere_geometry import FILL_VIEWS, PATCH_FOV, camera_rotation, pinhole_rays, sample_erp


PROTOCOL_STDOUT = sys.stdout
SKY_THRESHOLD = 0.3
SKY_ONLY_ASSERT = "Insufficient non-sky pixels for alignment"
PLY_FIELDS = (
    "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
    "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
)


def emit(stage: str, progressPercent: float, **fields: Any) -> None:
    """JSON Lines only on stdout, including while library logs use stderr."""
    print(json.dumps({
        "stage": stage,
        "progressPercent": round(min(100.0, max(0.0, progressPercent)), 1),
        **fields,
    }, ensure_ascii=False, allow_nan=False), file=PROTOCOL_STDOUT, flush=True)


@dataclass(frozen=True)
class View:
    name: str
    rotation: np.ndarray
    intrinsics: np.ndarray
    width: int
    height: int


@dataclass
class Splats:
    means: np.ndarray
    scales: np.ndarray
    rotations: np.ndarray
    dc: np.ndarray
    opacities: np.ndarray

    def __len__(self) -> int:
        return len(self.means)

    def select(self, indices: np.ndarray) -> "Splats":
        return Splats(*(getattr(self, field)[indices] for field in self.__dataclass_fields__))


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _matrix_from_wxyz(quaternions: np.ndarray) -> np.ndarray:
    """Use the quaternion convention consumed by official DA3 export/gsplat."""
    quaternions = quaternions / np.maximum(np.linalg.norm(quaternions, axis=1, keepdims=True), 1e-12)
    w, x, y, z = quaternions.T
    matrices = np.empty((len(quaternions), 3, 3), dtype=np.float32)
    matrices[:, 0, 0] = 1 - 2 * (y * y + z * z)
    matrices[:, 0, 1] = 2 * (x * y - z * w)
    matrices[:, 0, 2] = 2 * (x * z + y * w)
    matrices[:, 1, 0] = 2 * (x * y + z * w)
    matrices[:, 1, 1] = 1 - 2 * (x * x + z * z)
    matrices[:, 1, 2] = 2 * (y * z - x * w)
    matrices[:, 2, 0] = 2 * (x * z - y * w)
    matrices[:, 2, 1] = 2 * (y * z + x * w)
    matrices[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return matrices


def _wxyz_from_matrix(matrices: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    xyzw = Rotation.from_matrix(matrices).as_quat()
    result = xyzw[:, [3, 0, 1, 2]].astype(np.float32)
    result *= np.where(result[:, :1] < 0, -1, 1)
    return result


def _processed_intrinsics(source: View, width: int, height: int) -> np.ndarray:
    """Exact two-stage resize composition for one upper_bound_resize image.

    The pinned processor resizes, then resizes again to multiples of 14.
    It only center-crops to unify differently sized images in a batch; we pass
    exactly one image. Therefore the actual output/source ratios include every
    resize and no principal-point crop correction is needed.
    """
    result = source.intrinsics.astype(np.float64).copy()
    result[0] *= width / source.width
    result[1] *= height / source.height
    return result


def _view_weights(directions: np.ndarray, views: list[View], current_index: int) -> np.ndarray:
    """Blend optical density by a normalized directional overlap partition."""
    selected = np.zeros(len(directions), dtype=np.float32)
    total = np.zeros_like(selected)
    for index, view in enumerate(views):
        local = directions @ view.rotation
        z = local[:, 2]
        projected = local @ view.intrinsics.T
        x = projected[:, 0] / np.maximum(z, 1e-8)
        y = projected[:, 1] / np.maximum(z, 1e-8)
        edge = np.maximum(np.abs(x - view.width / 2) / (view.width / 2),
                          np.abs(y - view.height / 2) / (view.height / 2))
        fade = np.clip((1 - edge) / 0.22, 0, 1)
        fade = fade * fade * (3 - 2 * fade)
        fade *= z > 1e-8
        if len(views) == 1:
            # A lone photograph has no overlapping view to blend with. Keep
            # native opacity throughout its frustum instead of darkening edges.
            return ((z > 1e-8) & (edge <= 1)).astype(np.float32)
        weight = fade * np.maximum(z, 0) ** 4
        total += weight
        if index == current_index:
            selected = weight
    return selected / np.maximum(total, 1e-12)


def _sky_map(value: Any, shape: tuple[int, int] | None = None) -> np.ndarray | None:
    if value is None:
        return None
    result = _numpy(value).squeeze()
    if result.ndim != 2 or (shape is not None and result.shape != shape):
        raise RuntimeError("Die native DA3-Himmelsmaske hat eine unerwartete Form.")
    return result


def _copy_gaussians(prediction: Any) -> Splats:
    gaussian = prediction.gaussians
    if gaussian is None:
        raise RuntimeError("Das DA3-Modell hat keine nativen Gaussians ausgegeben.")
    means = _numpy(gaussian.means)
    scales = _numpy(gaussian.scales)
    rotations = _numpy(gaussian.rotations)
    # Copy only the invariant SH band to CPU, not all 27 coefficients.
    dc = _numpy(gaussian.harmonics[..., 0])
    opacities = _numpy(gaussian.opacities)
    if means.ndim != 3 or means.shape[0] != 1 or means.shape[-1] != 3:
        raise RuntimeError("Unerwartete native DA3-Gaussian-Form: genau eine Ansicht wird erwartet.")
    count = means.shape[1]
    if scales.shape != (1, count, 3) or rotations.shape != (1, count, 4) or dc.shape != (1, count, 3):
        raise RuntimeError("Die nativen DA3-Gaussian-Attribute passen nicht zusammen.")
    if opacities.size != count:
        raise RuntimeError("Die native DA3-Opazität ist keine skalare Opazität je Gaussian.")
    return Splats(means[0], scales[0], rotations[0], dc[0], opacities.reshape(count))


def _calibrate(
    prediction: Any,
    captured_sky: np.ndarray | None,
    views: list[View],
    view_index: int,
    coverage: int,
    panorama: bool,
    min_opacity: float,
    budget: int,
) -> tuple[Splats, dict[str, Any]]:
    """Transport native means and full covariance into one known-origin view."""
    splats = _copy_gaussians(prediction)
    depth = np.asarray(prediction.depth)
    if depth.ndim != 3 or depth.shape[0] != 1:
        raise RuntimeError("Unerwartete DA3-Tiefenform für eine einzelne Ansicht.")
    height, width = depth.shape[-2:]
    if len(splats) != height * width:
        raise RuntimeError("Die native Gaussian-Zahl entspricht nicht dem DA3-Pixelraster.")
    source = views[view_index]
    known_k = _processed_intrinsics(source, width, height)
    if prediction.intrinsics is None or prediction.extrinsics is None:
        raise RuntimeError("DA3 hat keine Kamerakalibrierung für seine Gaussians ausgegeben.")
    predicted_k = _numpy(prediction.intrinsics)[0].astype(np.float64)
    extrinsic = _numpy(prediction.extrinsics)[0].astype(np.float64)
    if predicted_k.shape != (3, 3) or extrinsic.shape not in ((3, 4), (4, 4)):
        raise RuntimeError("Die nativen DA3-Kameramatrizen haben eine unerwartete Form.")
    if not np.isfinite(predicted_k).all() or not np.isfinite(extrinsic).all():
        raise RuntimeError("DA3 hat ungültige Kameramatrizen ausgegeben.")
    if min(predicted_k[0, 0], predicted_k[1, 1]) <= 0:
        raise RuntimeError("DA3 hat eine ungültige Brennweite ausgegeben.")
    scale_factor = float(prediction.scale_factor) if prediction.scale_factor is not None else float("nan")
    if not math.isfinite(scale_factor) or scale_factor <= 0 or not prediction.is_metric:
        raise RuntimeError("Der native Nested-DA3-Ausgabe fehlt eine gültige metrische Skalierung.")
    focal_adjustment = float((known_k[0, 0] + known_k[1, 1]) /
                             (predicted_k[0, 0] + predicted_k[1, 1]))
    angular_matrix = np.linalg.solve(known_k, predicted_k)
    predicted_rotation = extrinsic[:3, :3]
    predicted_translation = extrinsic[:3, 3]
    known_rotation = source.rotation.astype(np.float64)

    # Nested rescales its extrinsic translation, but leaves native means/scales
    # unscaled. Scale the latter BEFORE applying the already-metric extrinsic.
    local = (splats.means.astype(np.float64) * scale_factor) @ predicted_rotation.T + predicted_translation
    radius = np.linalg.norm(local, axis=1)
    mask = (np.isfinite(local).all(axis=1) & np.isfinite(splats.scales).all(axis=1)
            & np.isfinite(splats.rotations).all(axis=1) & np.isfinite(splats.dc).all(axis=1)
            & np.isfinite(splats.opacities) & (splats.scales > 0).all(axis=1)
            & (np.linalg.norm(splats.rotations, axis=1) > 1e-8)
            & (local[:, 2] > 1e-7) & (radius > 1e-7)
            & (splats.opacities >= min_opacity))
    sky = _sky_map(captured_sky, (height, width)) if captured_sky is not None else None
    if sky is None and prediction.sky is not None:
        sky = _sky_map(prediction.sky, (height, width))
        non_sky = ~sky.astype(bool)
    elif sky is not None:
        non_sky = np.isfinite(sky) & (sky < SKY_THRESHOLD)
    else:
        # The chosen pinned Nested model must expose the metric branch sky.
        # Exporting its artificial sky depth would invent a finite surface.
        raise RuntimeError("Die Himmelsmaske des nativen Nested-DA3-Modells fehlt.")
    mask &= non_sky.reshape(-1)
    candidate_indices = np.flatnonzero(mask)
    original_count = len(splats)
    if candidate_indices.size:
        candidate_local = local[candidate_indices]
        remapped = candidate_local @ angular_matrix.T
        remapped_radius = np.linalg.norm(remapped, axis=1)
        good = np.isfinite(remapped_radius) & (remapped_radius > 1e-9)
        candidate_indices = candidate_indices[good]
        candidate_local = candidate_local[good]
        remapped = remapped[good]
        remapped_radius = remapped_radius[good]
        candidate_radius = radius[candidate_indices]
        directions_local = remapped / remapped_radius[:, None]
        directions_world = directions_local @ known_rotation.T
        weights = _view_weights(directions_world.astype(np.float32), views, view_index)
        if panorama and coverage == 180:
            weights *= directions_world[:, 2] > 1e-7
        alpha = np.clip(splats.opacities[candidate_indices], 1e-7, 1 - 1e-7)
        blended_opacity = -np.expm1(np.log1p(-alpha) * weights)
        retained = np.isfinite(blended_opacity) & (blended_opacity >= min_opacity)
        candidate_indices = candidate_indices[retained]
        candidate_local = candidate_local[retained]
        candidate_radius = candidate_radius[retained]
        remapped_radius = remapped_radius[retained]
        directions_local = directions_local[retained]
        directions_world = directions_world[retained]
        blended_opacity = blended_opacity[retained]
    else:
        candidate_local = np.empty((0, 3), dtype=np.float64)
        candidate_radius = remapped_radius = blended_opacity = np.empty(0, dtype=np.float64)
        directions_local = directions_world = np.empty((0, 3), dtype=np.float64)

    retained_before_budget = len(candidate_indices)
    if retained_before_budget > budget:
        # Reproducible uniform sampling avoids a spatial bias towards one edge.
        chosen = np.sort(np.random.default_rng(2718 + view_index).choice(retained_before_budget, budget, replace=False))
        candidate_indices = candidate_indices[chosen]
        candidate_local = candidate_local[chosen]
        candidate_radius = candidate_radius[chosen]
        remapped_radius = remapped_radius[chosen]
        directions_local = directions_local[chosen]
        directions_world = directions_world[chosen]
        blended_opacity = blended_opacity[chosen]

    count = len(candidate_indices)
    means_out = (directions_world * (candidate_radius * focal_adjustment)[:, None]).astype(np.float32)
    scales_out = np.empty((count, 3), dtype=np.float32)
    rotations_out = np.empty((count, 4), dtype=np.float32)
    identity = np.eye(3, dtype=np.float64)
    for start in range(0, count, 32768):
        stop = min(start + 32768, count)
        selected_indices = candidate_indices[start:stop]
        p = candidate_local[start:stop]
        r = candidate_radius[start:stop]
        q = directions_local[start:stop]
        u = p / r[:, None]
        s = r / remapped_radius[start:stop]
        qq = q[:, :, None] * q[:, None, :]
        qu = q[:, :, None] * u[:, None, :]
        # F(p)=|p| A p/|A p|. Exact differential of this radius-preserving map.
        jacobian = s[:, None, None] * ((identity - qq) @ angular_matrix) + qu
        transport = known_rotation @ (focal_adjustment * jacobian) @ predicted_rotation
        native_rotation = _matrix_from_wxyz(splats.rotations[selected_indices]).astype(np.float64)
        native_scales = splats.scales[selected_indices].astype(np.float64) * scale_factor
        covariance_factor = native_rotation * native_scales[:, None, :]
        transported_factor = transport @ covariance_factor
        # SVD of the covariance factor recovers the full transported covariance
        # without explicitly squaring its condition number in an eigensolver.
        axes, singular_values, _ = np.linalg.svd(transported_factor, full_matrices=False)
        reflected = np.linalg.det(axes) < 0
        axes[reflected, :, -1] *= -1
        scales_out[start:stop] = singular_values.astype(np.float32)
        rotations_out[start:stop] = _wxyz_from_matrix(axes)

    output = Splats(means_out, scales_out, rotations_out,
                    splats.dc[candidate_indices], blended_opacity.astype(np.float32))
    valid_output = (np.isfinite(output.means).all(axis=1) & np.isfinite(output.scales).all(axis=1)
                    & (output.scales > 1e-10).all(axis=1) & np.isfinite(output.rotations).all(axis=1))
    output = output.select(valid_output)
    metadata = {
        "name": source.name, "status": "ready" if len(output) else "skipped",
        "pointCount": len(output), "nativePointCount": original_count,
        "retainedBeforeBudget": retained_before_budget,
        "processedResolution": [width, height],
        "knownIntrinsics": known_k.tolist(), "predictedIntrinsics": predicted_k.tolist(),
        "knownCameraToWorld": source.rotation.tolist(),
        "predictedWorldToCamera": extrinsic.tolist(),
        "nestedScaleFactor": scale_factor, "metricFocalAdjustment": focal_adjustment,
        "skyPixelCount": int((~non_sky).sum()),
    }
    if not len(output):
        metadata["reason"] = "no_valid_surface"
    return output, metadata


def _make_views(args: argparse.Namespace, image: Image.Image, focal: float) -> list[View]:
    if args.projection == "perspective":
        return [View("perspective", np.eye(3, dtype=np.float32),
                     make_intrinsics(image.width, image.height, focal), image.width, image.height)]
    resolution = args.view_resolution
    field_of_view = PATCH_FOV if args.coverage == 180 else args.view_fov
    focal = resolution / (2 * math.tan(math.radians(field_of_view) / 2))
    known_k = make_intrinsics(resolution, resolution, focal)
    if args.coverage == 180:
        return [View(name, camera_rotation(yaw, pitch), known_k.copy(), resolution, resolution)
                for name, yaw, pitch in FILL_VIEWS]
    # The shared full-ERP sampler uses integer pixel positions rather than
    # +0.5-centered positions. DA3's native rays use centers, so this half-pixel
    # principal-point adjustment describes the sampler exactly.
    known_k[0, 2] += 0.5
    known_k[1, 2] += 0.5
    return [View(view.name, view.rotation, known_k.copy(), resolution, resolution) for view in cube_views()]


def _view_image(args: argparse.Namespace, source: np.ndarray, view: View) -> np.ndarray:
    if args.projection == "perspective":
        return source
    if args.coverage == 360:
        return sample_perspective_rgb(source, view.rotation, view.width, float(view.intrinsics[0, 0]))
    rays = pinhole_rays(view.width, view.height, PATCH_FOV, view.rotation)
    rgb, valid = sample_erp(source, rays, coverage=180)
    rgb[~valid] = 0
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def _save_ply_atomic(path: Path, parts: list[Splats]) -> int:
    count = sum(len(part) for part in parts)
    if count == 0:
        raise RuntimeError("DA3 hat keine sichtbaren Oberflächen erkannt; es gibt keine Gaussians zum Exportieren.")
    header = "\n".join([
        "ply", "format binary_little_endian 1.0",
        "comment native DA3 Gaussian head; OpenCV axes; quaternion wxyz; SH degree 0",
        f"element vertex {count}",
        *(f"property float {field}" for field in PLY_FIELDS), "end_header", "",
    ]).encode("ascii")
    temporary = path.with_name(path.name + ".partial")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("wb") as stream:
            stream.write(header)
            for part in parts:
                for start in range(0, len(part), 65536):
                    stop = min(start + 65536, len(part))
                    record = np.empty(stop - start, dtype=[(name, "<f4") for name in PLY_FIELDS])
                    for column, name in enumerate(("x", "y", "z")):
                        record[name] = part.means[start:stop, column]
                    for column in range(3):
                        record[f"f_dc_{column}"] = part.dc[start:stop, column]
                        record[f"scale_{column}"] = np.log(part.scales[start:stop, column])
                    alpha = np.clip(part.opacities[start:stop], 1e-6, 1 - 1e-6)
                    record["opacity"] = np.log(alpha) - np.log1p(-alpha)
                    for column in range(4):
                        record[f"rot_{column}"] = part.rotations[start:stop, column]
                    stream.write(record.tobytes())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return count


def _save_json_atomic(path: Path, metadata: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".partial")
    try:
        temporary.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    if input_path == output_path:
        raise ValueError("Eingabe und PLY-Ausgabe müssen unterschiedliche Dateien sein.")
    if not input_path.is_file():
        raise FileNotFoundError(f"Das Eingabebild fehlt: {input_path}")
    emit("prepare", 0, message="Bild wird für native DA3-Gaussians vorbereitet")
    with Image.open(input_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        focal = args.focal_length_px or estimate_focal_length(opened, image.size)
    source = np.asarray(image)
    if args.projection == "equirectangular":
        expected_ratio = 1 if args.coverage == 180 else 2
        if abs(image.width / image.height - expected_ratio) > 0.03:
            raise ValueError(f"Ein {args.coverage}°-ERP-Bild benötigt ein Seitenverhältnis von {expected_ratio}:1.")
    views = _make_views(args, image, focal)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        import da3_runtime

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # Reserve 5..20% for the one-time model load, leaving the view loop monotonic.
    def model_progress(stage: str, progressPercent: float, **fields: Any) -> None:
        emit(stage, 5 + 0.15 * progressPercent, **fields)

    with contextlib.redirect_stdout(sys.stderr):
        model = da3_runtime.load_native_model(device=device, cache_dir=args.model_cache,
                                             progress_callback=model_progress)
    metric_branch = getattr(getattr(model, "model", None), "da3_metric", None)
    if metric_branch is None:
        raise RuntimeError("Dieses Modell besitzt nicht den benötigten nativen Nested-DA3-Metrikzweig.")
    captured: dict[str, np.ndarray | None] = {"sky": None}

    def capture_metric_sky(module: Any, inputs: Any, output: Any) -> None:
        # Store only a small CPU mask. Never retain the forward output/features.
        captured["sky"] = _sky_map(output.get("sky"))

    hook = metric_branch.register_forward_hook(capture_metric_sky)
    parts: list[Splats] = []
    view_metadata: list[dict[str, Any]] = []
    view_budget = max(1, args.max_splats // len(views))
    try:
        for index, view in enumerate(views):
            captured["sky"] = None
            progress = 20 + 70 * index / len(views)
            emit("predict", progress, message=f"Native DA3-Gaussians: {view.name}",
                 view=index + 1, viewCount=len(views), viewName=view.name)
            rgb = _view_image(args, source, view)
            prediction = None
            try:
                with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
                    prediction = model.inference(
                        image=[rgb], extrinsics=None, intrinsics=None,
                        infer_gs=True, export_dir=None,
                        process_res=args.process_resolution,
                        process_res_method="upper_bound_resize",
                    )
            except AssertionError as error:
                sky = captured["sky"]
                sky_only = (str(error) == SKY_ONLY_ASSERT and sky is not None
                            and np.isfinite(sky).all() and int((sky < SKY_THRESHOLD).sum()) <= 10)
                if not sky_only:
                    raise
                if args.projection == "perspective":
                    raise RuntimeError("Das Bild enthält für DA3 fast nur Himmel und zu wenige erkennbare Oberflächen.") from error
                metadata = {"name": view.name, "status": "skipped", "reason": "sky_only",
                            "pointCount": 0, "skyPixelCount": int((sky >= SKY_THRESHOLD).sum()),
                            "knownCameraToWorld": view.rotation.tolist()}
                view_metadata.append(metadata)
                emit("view", 20 + 70 * (index + 1) / len(views),
                     message=f"Himmelsansicht {view.name} wird ausgelassen", **metadata)
                continue
            finally:
                del rgb
                if prediction is None:
                    captured["sky"] = None
                    gc.collect()
                    if str(device).startswith("cuda"):
                        torch.cuda.empty_cache()
            try:
                emit("calibrate", progress + 4 / len(views),
                     message=f"Gaussians werden ausgerichtet: {view.name}")
                part, metadata = _calibrate(
                    prediction, captured["sky"], views, index, args.coverage,
                    args.projection == "equirectangular", args.min_opacity, view_budget,
                )
                if len(part):
                    parts.append(part)
                view_metadata.append(metadata)
                emit("view", 20 + 70 * (index + 1) / len(views),
                     message=f"Ansicht {view.name} verarbeitet", view=index + 1,
                     viewCount=len(views), pointCount=len(part), viewName=view.name)
            finally:
                captured["sky"] = None
                del prediction
                gc.collect()
                if str(device).startswith("cuda"):
                    torch.cuda.empty_cache()
    finally:
        hook.remove()
        captured.clear()
        del model
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    emit("export", 93, message="Native Gaussians werden als PLY gespeichert")
    count = sum(len(part) for part in parts)
    if not count:
        raise RuntimeError("DA3 hat in diesem Bild keine sichtbaren Oberflächen erkannt.")
    metadata = {
        "pipeline": "da3-native", "estimator": "da3", "input": str(input_path),
        "output": str(output_path), "projection": args.projection,
        "coverageDegrees": args.coverage if args.projection == "equirectangular" else None,
        "modelId": da3_runtime.MODEL_ID, "modelRevision": da3_runtime.MODEL_REVISION,
        "modelLicense": da3_runtime.MODEL_LICENSE, "codeRevision": da3_runtime.CODE_REVISION,
        "processResolution": args.process_resolution, "processResolutionMethod": "upper_bound_resize",
        "sourceResolution": [image.width, image.height],
        "originalFocalLengthPx": focal if args.projection == "perspective" else None,
        "pointCount": count, "maxSplats": args.max_splats, "viewCount": len(views),
        "successfulViewCount": len(parts), "views": view_metadata,
        "coordinateSystem": "opencv_x_right_y_down_z_forward",
        "viewerRotationXDegrees": 180, "quaternionConvention": "wxyz",
        "sphericalHarmonicDegree": 0, "nativeGaussianHead": True,
        "depthDefinition": "native_unit_ray_radius_plus_learned_offset",
        "depthMapDefinition": "camera_z_meters",
        "calibration": {
            "experimental": True, "method": "radius_preserving_intrinsic_angular_remap",
            "covarianceTransport": "analytic_jacobian_then_svd",
            "nestedScaleAppliedToGaussians": True,
            "knownPredictedFocalMetricAdjustment": True,
            "commonCameraOrigin": [0, 0, 0], "jointMultiViewInference": False,
        },
        "skyRemoved": True, "skyThreshold": SKY_THRESHOLD,
        "opacityBlending": "optical_density_directional_partition" if len(views) > 1 else "none",
        "limitations": [
            "Monocular metric scale is an estimate; intrinsic calibration is experimental.",
            "Panorama views are inferred independently; overlapping surfaces can disagree in depth.",
            "Only DC color is exported; learned view-dependent SH color is not retained.",
            "Sky is omitted rather than exported as an arbitrary finite-distance surface.",
        ],
        "durationSeconds": round(time.perf_counter() - started, 3),
    }
    # The final PLY is the backend's visibility signal. Commit its metadata
    # first, then expose the completed binary file with one atomic replacement.
    sidecar = output_path.with_suffix(".json")
    _save_json_atomic(sidecar, metadata)
    _save_ply_atomic(output_path, parts)
    emit("ready", 100, message="Native DA3-Gaussians sind bereit", status="ready",
         output=str(output_path), metadata=str(sidecar), pointCount=count,
         durationSeconds=round(time.perf_counter() - started, 3))
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Native DA3 Gaussian inference ohne SHARP oder Apple Depth Pro.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--projection", choices=("perspective", "equirectangular"), default="perspective")
    parser.add_argument("--coverage", type=int, choices=(180, 360), default=360)
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, cpu oder auto")
    parser.add_argument("--process-res", "--process-resolution", dest="process_resolution", type=int, default=352)
    parser.add_argument("--view-resolution", type=int, default=1024)
    parser.add_argument("--view-fov", type=float, default=100.0, help="360°-Ansichten; 180° verwendet feste 120°")
    parser.add_argument("--focal-length-px", type=float, default=None, help="Perspektivbild: EXIF/Fallback bei fehlendem Wert")
    parser.add_argument("--max-splats", type=int, default=3_000_000)
    parser.add_argument("--min-opacity", type=float, default=0.005)
    parser.add_argument("--model-cache", default=None)
    args = parser.parse_args()
    if not 140 <= args.process_resolution <= 1024:
        parser.error("--process-res muss zwischen 140 und 1024 liegen.")
    if not 256 <= args.view_resolution <= 4096:
        parser.error("--view-resolution muss zwischen 256 und 4096 liegen.")
    if not math.isfinite(args.view_fov) or not 95 <= args.view_fov <= 130:
        parser.error("--view-fov muss zwischen 95 und 130 Grad liegen.")
    if args.focal_length_px is not None and (not math.isfinite(args.focal_length_px) or args.focal_length_px <= 0):
        parser.error("--focal-length-px muss endlich und positiv sein.")
    if args.max_splats < 6:
        parser.error("--max-splats muss mindestens 6 sein.")
    if not math.isfinite(args.min_opacity) or not 0 < args.min_opacity < 1:
        parser.error("--min-opacity muss zwischen 0 und 1 liegen.")
    return args


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        run(parse_args())
    except Exception as error:
        emit("error", 0, message=str(error), error=str(error), status="error",
             errorType=type(error).__name__)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
