"""Create an experimental 180/360-degree SHARP scene from an ERP image.

The input is a full 2:1 panorama or a square front hemisphere (--coverage 180).
Five hemisphere views or six full-sphere overlapping
pinhole views are sampled with known intrinsics and inferred independently with
one SHARP model. Their positions and wxyz orientation quaternions are rotated
around the SAME camera origin. No translation, shell radius, EXIF focal estimate,
or arbitrary placement of flat panels is used.

World coordinates remain Apple's OpenCV convention: +X right, +Y down, +Z
forward. Longitude zero is the middle of the panorama, north is its top edge.
The studio's existing 180-degree rotation about X therefore remains correct.

This is not a jointly optimized 360-degree reconstruction. Monocular depth and
object geometry may disagree between views; overlap fades reduce double edges
but cannot guarantee seamless geometry or reliable metric scale. A scene is
most convincing when viewed near its original camera position.

With --reference-depth, a seamless DA² distance map of the whole panorama
(da2_reference.py) is sampled along each view's rays and passed to SHARP's
learned depth alignment. One scene-wide scale, calibrated on the front view,
makes all views share the same geometry. Each view's own SHARP depth is first
scaled to the reference median, since the alignment only fixes moderate errors. This path is experimental: Apple
trained the alignment with ground-truth depth, not with another model's
estimate.

Example (use the project's Python environment)::

    python panorama_splats.py --input panorama.png --output scene.ply --device cuda

Progress is emitted as one UTF-8 JSON object per stdout line. Diagnostics use
stderr. A sidecar scene.json records projection, view transforms and limitations.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from plyfile import PlyData, PlyElement

import ai_upscale
from sharp.cli.predict import (
    DEFAULT_INTERNAL_RESOLUTION,
    DEFAULT_MODEL_URL,
    predict_image,
    predict_monodepth,
    validate_internal_resolution,
)
from sharp.models import PredictorParams, create_predictor
from sharp.utils import color_space
from sharp.utils.gaussians import Gaussians3D, convert_rgb_to_spherical_harmonics
from sharp.utils.linalg import quaternions_from_rotation_matrices


LIMITATION = (
    "Experimentelle 360°-Szene: SHARP schätzt jede Ansicht separat. "
    "An Übergängen sind unterschiedliche Tiefen und doppelte Konturen möglich; "
    "die Szene eignet sich vor allem zum Umschauen nahe dem Aufnahmezentrum."
)


@dataclass(frozen=True)
class PerspectiveView:
    """A camera-to-world rotation and its signed dominant world axis."""

    name: str
    rotation: np.ndarray
    axis: int
    sign: int


def emit(stage: str, **details: Any) -> None:
    """Emit machine-readable progress without buffering stdout."""
    if "progress" in details and "progressPercent" not in details:
        details["progressPercent"] = round(float(details["progress"]) * 100)
    print(json.dumps({"stage": stage, **details}, ensure_ascii=False), flush=True)


def cube_views() -> list[PerspectiveView]:
    """Return six right-handed camera rotations in Apple's OpenCV convention."""
    specifications = [
        ("front", (0, 0, 1), 2, 1),
        ("right", (1, 0, 0), 0, 1),
        ("back", (0, 0, -1), 2, -1),
        ("left", (-1, 0, 0), 0, -1),
        ("up", (0, -1, 0), 1, -1),
        ("down", (0, 1, 0), 1, 1),
    ]
    result = []
    for name, forward_tuple, axis, sign in specifications:
        forward = np.asarray(forward_tuple, dtype=np.float32)
        if axis == 1:
            right = np.asarray((1, 0, 0), dtype=np.float32)
        else:
            right = np.cross(forward, np.asarray((0, -1, 0), dtype=np.float32))
        down = np.cross(forward, right)
        # Columns describe where local +X, +Y and +Z point in world space.
        rotation = np.stack((right, down, forward), axis=1)
        result.append(PerspectiveView(name, rotation, axis, sign))
    return result


def view_rays(view: PerspectiveView, resolution: int, focal_length_px: float) -> tuple[np.ndarray, np.ndarray]:
    """Return local (x, y, 1) pinhole rays and their world directions."""
    coordinates = (np.arange(resolution, dtype=np.float32) - resolution / 2) / focal_length_px
    xx, yy = np.meshgrid(coordinates, coordinates)
    local_rays = np.stack((xx, yy, np.ones_like(xx)), axis=-1)
    return local_rays, local_rays @ view.rotation.T


def sample_perspective(
    panorama: np.ndarray,
    view: PerspectiveView,
    resolution: int,
    focal_length_px: float,
    coverage: int = 360,
) -> np.ndarray:
    """Sample a pinhole camera from ERP, with wrap-aware bilinear interpolation.

    Pixel coordinates and principal point (resolution / 2) match predict_image's
    actual intrinsics. ERP pixels are centered at half-integer spherical UVs.
    Horizontal interpolation wraps at the panorama seam; latitude is clamped.
    """
    _, world_rays = view_rays(view, resolution, focal_length_px)
    sampled, _ = sample_erp_rays(panorama, world_rays, coverage)
    return np.clip(np.rint(sampled), 0, 255).astype(np.uint8)


def sample_erp_rays(panorama: np.ndarray, world_rays: np.ndarray, coverage: int) -> tuple[np.ndarray, np.ndarray]:
    """Bilinearly sample an ERP image or map along world rays; also return validity."""
    if coverage == 180:
        from hemisphere_geometry import sample_erp

        return sample_erp(panorama, world_rays, coverage=180)
    radius = np.linalg.norm(world_rays, axis=-1)
    longitude = np.arctan2(world_rays[..., 0], world_rays[..., 2])
    latitude = np.arcsin(np.clip(-world_rays[..., 1] / radius, -1, 1))
    height, width = panorama.shape[:2]
    source_x = ((longitude / (2 * np.pi) + 0.5) * width - 0.5) % width
    source_y = np.clip((0.5 - latitude / np.pi) * height - 0.5, 0, height - 1)
    x0 = np.floor(source_x).astype(np.int32)
    y0 = np.floor(source_y).astype(np.int32)
    x1 = (x0 + 1) % width
    y1 = np.minimum(y0 + 1, height - 1)
    wx = (source_x - x0).astype(np.float32)[..., None]
    wy = (source_y - y0).astype(np.float32)[..., None]
    if panorama.ndim == 2:
        wx, wy = wx[..., 0], wy[..., 0]
    top = panorama[y0, x0] * (1 - wx) + panorama[y0, x1] * wx
    bottom = panorama[y1, x0] * (1 - wx) + panorama[y1, x1] * wx
    return top * (1 - wy) + bottom * wy, np.ones(world_rays.shape[:-1], dtype=bool)


def sample_reference_depth(
    distance: np.ndarray, view: PerspectiveView, resolution: int, focal_length_px: float, coverage: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert ERP radial distance into this view's camera-Z depth.

    A point at radial distance d along the local ray (x, y, 1) has camera depth
    d / |(x, y, 1)|. Rays outside a 180-degree hemisphere are marked invalid.
    """
    local_rays, world_rays = view_rays(view, resolution, focal_length_px)
    radial, valid = sample_erp_rays(distance, world_rays, coverage)
    return (radial / np.linalg.norm(local_rays, axis=-1)).astype(np.float32), valid


def reference_depth_for_view(
    reference: np.ndarray,
    valid: np.ndarray,
    monodepth: torch.Tensor,
    scale: float | None,
) -> tuple[torch.Tensor, float, float]:
    """Scale DA² depth to SHARP's units and fill rays the reference cannot see.

    The scene-wide scale is calibrated once, on the first (front) view, so every
    view receives the same geometry. Invalid rays fall back to SHARP's own depth,
    rescaled to the reference in that view; those Gaussians are cropped anyway.
    Returns the depth map, the scene scale and SHARP's depth relative to the
    reference in this view (1.0 means the independent prediction already agreed).
    """
    shape = monodepth.shape
    reference_pt = F.interpolate(
        torch.from_numpy(reference)[None, None].to(monodepth.device), size=shape,
        mode="bilinear", align_corners=True,
    )[0, 0]
    valid_pt = F.interpolate(
        torch.from_numpy(valid.astype(np.float32))[None, None].to(monodepth.device), size=shape,
        mode="bilinear", align_corners=True,
    )[0, 0] > 0.999
    if valid_pt.sum() < 0.05 * valid_pt.numel():
        raise RuntimeError("Die Referenztiefe deckt diese Ansicht nicht ab.")
    log_ratio = (monodepth.clamp_min(1e-6).log() - reference_pt.clamp_min(1e-6).log())[valid_pt]
    if scale is None:
        scale = float(log_ratio.median().exp())
    view_ratio = float((log_ratio - math.log(scale)).median().exp())
    scaled = reference_pt * scale
    return torch.where(valid_pt, scaled, monodepth / view_ratio), scale, view_ratio


def save_depth_previews(directory: Path, name: str, depths: dict[str, torch.Tensor]) -> None:
    """Save inverse-depth previews of one view on a shared scale (bright is near)."""
    inverse = {key: 1 / depth.float().clamp_min(1e-6) for key, depth in depths.items()}
    reference = inverse.get("reference-depth", next(iter(inverse.values())))
    low, high = torch.quantile(reference.flatten()[::97], torch.tensor([0.02, 0.98], device=reference.device)).tolist()
    for key, value in inverse.items():
        normalized = ((value - low) / max(high - low, 1e-8)).clamp(0, 1)
        Image.fromarray((normalized * 255).byte().cpu().numpy()).save(directory / f"{name}.{key}.png")


def multiply_quaternions(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Compute a Hamilton product of broadcastable wxyz quaternions."""
    left, right = torch.broadcast_tensors(left, right)
    scalar = left[..., :1] * right[..., :1] - (left[..., 1:] * right[..., 1:]).sum(
        dim=-1, keepdim=True
    )
    vector = (
        left[..., :1] * right[..., 1:]
        + right[..., :1] * left[..., 1:]
        + torch.cross(left[..., 1:], right[..., 1:], dim=-1)
    )
    return F.normalize(torch.cat((scalar, vector), dim=-1), dim=-1)


def direction_weights(
    means_world: torch.Tensor, view: PerspectiveView, tangent_half_fov: float
) -> torch.Tensor:
    """Partition optical density across the overlapping cube camera frustums.

    A direction belongs only to cameras that can actually see it. Within each
    frustum, a smoothstep tapers the outer pixels. Dividing by the sum of the
    three possible positive-facing cube weights gives a partition of unity,
    including cube edges/corners, with no intentional black gaps.
    """
    directions = F.normalize(means_world, dim=-1)
    absolute = directions.abs()
    inner = 0.8  # Start tapering inside the nominal 90-degree cube face.
    raw_weights = []
    for axis in range(3):
        transverse = absolute[:, [other for other in range(3) if other != axis]].amax(dim=-1)
        cosine = absolute[:, axis]
        ratio = transverse / cosine.clamp_min(1e-8)
        fade = ((tangent_half_fov - ratio) / (tangent_half_fov - inner)).clamp(0, 1)
        fade = fade.square() * (3 - 2 * fade)
        raw_weights.append(fade * cosine.pow(4))
    denominator = sum(raw_weights).clamp_min(1e-8)
    front_facing = directions[:, view.axis] * view.sign > 0
    return raw_weights[view.axis] / denominator * front_facing


def rotate_and_filter(
    gaussians: Gaussians3D,
    view: PerspectiveView,
    tangent_half_fov: float,
    point_budget: int,
    minimum_opacity: float,
    seed: int,
    coverage: int = 360,
) -> tuple[Gaussians3D, dict[str, int]]:
    """Rotate full Gaussian orientations and remove unsupported directions."""
    flat = {name: tensor[0] for name, tensor in gaussians._asdict().items()}
    means = flat["mean_vectors"]
    valid = means[:, 2] > 1e-4
    for tensor in flat.values():
        finite = torch.isfinite(tensor)
        valid &= finite.all(dim=-1) if tensor.ndim > 1 else finite
    valid &= (flat["singular_values"] > 0).all(dim=-1)
    valid &= torch.linalg.vector_norm(flat["quaternions"], dim=-1) > 1e-8
    flat = {name: tensor[valid] for name, tensor in flat.items()}
    rotation = torch.as_tensor(view.rotation, dtype=means.dtype, device=means.device)
    world_means = flat["mean_vectors"] @ rotation.T
    weights = direction_weights(world_means, view, tangent_half_fov)
    opacity = flat["opacities"].clamp(1e-6, 1 - 1e-6)
    # Alpha is nonlinear: partition optical density rather than alpha directly.
    opacity = -torch.expm1(torch.log1p(-opacity) * weights)
    keep = opacity >= minimum_opacity
    if coverage == 180:
        keep &= world_means[:, 2] > 0
    flat = {name: tensor[keep] for name, tensor in flat.items()}
    world_means = world_means[keep]
    opacity = opacity[keep]
    supported = len(world_means)
    if supported == 0:
        raise RuntimeError(f"SHARP hat für Ansicht '{view.name}' keine gültigen Splats erzeugt.")
    if point_budget and supported > point_budget:
        # A fixed seed makes the point budget reproducible. Uniform sampling
        # avoids preferentially deleting the transition regions.
        generator = torch.Generator(device="cpu").manual_seed(seed)
        indices = torch.randperm(supported, generator=generator)[:point_budget].to(means.device)
        flat = {name: tensor[indices] for name, tensor in flat.items()}
        world_means = world_means[indices]
        opacity = opacity[indices]
        # Compensate expected optical density after uniform thinning. This is
        # approximate; it cannot replace missing spatial samples.
        keep_fraction = point_budget / supported
        opacity = -torch.expm1(torch.log1p(-opacity.clamp_max(1 - 1e-6)) / keep_fraction)
    world_quaternion = quaternions_from_rotation_matrices(rotation).reshape(1, 4)
    orientations = multiply_quaternions(world_quaternion, flat["quaternions"])
    result = Gaussians3D(
        mean_vectors=world_means.unsqueeze(0).cpu(),
        singular_values=flat["singular_values"].unsqueeze(0).cpu(),
        quaternions=orientations.unsqueeze(0).cpu(),
        colors=flat["colors"].clamp(0, 1).unsqueeze(0).cpu(),
        opacities=opacity.clamp(1e-6, 1 - 1e-6).unsqueeze(0).cpu(),
    )
    return result, {"predicted": len(means), "supported": supported, "kept": len(world_means)}


def write_panorama_ply(
    chunks: list[Gaussians3D], path: Path, resolution: int, focal_length_px: float,
    coverage: int = 360,
) -> int:
    """Write standard 3DGS PLY without building millions of Python tuples.

    Colors follow SHARP's public-viewer sRGB export. Intrinsics describe the
    forward perspective view, not a fictional pinhole 360-degree camera. The
    supplemental disparity is inverse radial distance, since global Z may be
    negative or zero in a surrounding scene. Spark uses the vertex attributes.
    """
    attributes = (
        ["x", "y", "z"]
        + [f"f_dc_{index}" for index in range(3)]
        + ["opacity"]
        + [f"scale_{index}" for index in range(3)]
        + [f"rot_{index}" for index in range(4)]
    )
    count = sum(chunk.mean_vectors.shape[1] for chunk in chunks)
    vertices = np.empty(count, dtype=[(attribute, "f4") for attribute in attributes])
    inverse_radii = np.empty(count, dtype=np.float32)
    offset = 0
    for chunk in chunks:
        means = chunk.mean_vectors[0]
        size = len(means)
        colors = convert_rgb_to_spherical_harmonics(color_space.linearRGB2sRGB(chunk.colors[0]))
        alpha = chunk.opacities[0].clamp(1e-6, 1 - 1e-6)
        values = torch.cat(
            (
                means,
                colors,
                (torch.log(alpha) - torch.log1p(-alpha)).unsqueeze(-1),
                torch.log(chunk.singular_values[0].clamp_min(1e-8)),
                chunk.quaternions[0],
            ),
            dim=-1,
        ).numpy()
        section = slice(offset, offset + size)
        for index, attribute in enumerate(attributes):
            vertices[attribute][section] = values[:, index]
        inverse_radii[section] = 1 / torch.linalg.vector_norm(means, dim=-1).clamp_min(1e-8).numpy()
        offset += size

    def element(name: str, field: str, values: Any, dtype: str) -> PlyElement:
        values_array = np.asarray(values).reshape(-1)
        structured = np.empty(len(values_array), dtype=[(field, dtype)])
        structured[field] = values_array
        return PlyElement.describe(structured, name)

    intrinsics = [
        focal_length_px, 0, resolution / 2,
        0, focal_length_px, resolution / 2,
        0, 0, 1,
    ]
    ply = PlyData(
        [
            PlyElement.describe(vertices, "vertex"),
            element("extrinsic", "extrinsic", np.eye(4), "f4"),
            element("intrinsic", "intrinsic", intrinsics, "f4"),
            element("image_size", "image_size", [resolution, resolution], "u4"),
            element("frame", "frame", [1, count], "i4"),
            element("disparity", "disparity", np.quantile(inverse_radii, [0.1, 0.9]), "f4"),
            element("color_space", "color_space", [0], "u1"),
            element("version", "version", [1, 5, 0], "u1"),
        ],
        text=False,
        byte_order="<",
        comments=[
            f"panorama_{coverage} OpenCV axes x-right y-down z-forward; shared camera origin",
            "intrinsics are the forward view; disparity is inverse radial distance",
            "independent monocular predictions; geometry is not globally optimized",
        ],
    )
    temporary_path = path.with_name(path.name + ".partial")
    ply.write(temporary_path)
    temporary_path.replace(path)
    return count


def resolve_device(requested: str) -> torch.device:
    """Resolve automatic device selection without silently hiding CUDA failures."""
    if requested in ("auto", "default"):
        if torch.cuda.is_available():
            requested = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA ist nicht verfügbar. Nutze --device auto oder --device cpu.")
    return device


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    """Convert an ERP panorama or front hemisphere into a Gaussian scene."""
    started = time.monotonic()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if output_path.suffix.lower() != ".ply":
        output_path = output_path / "scene.ply"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    coverage = args.coverage
    emit(f"{coverage}°-Panorama wird vorbereitet", progress=0, status="processing")
    with Image.open(input_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        width, height = image.size
        expected_ratio = coverage / 180
        if abs(width / height - expected_ratio) > 0.02:
            raise ValueError(f"Für {coverage}° ist ein equirectangulares Bild im Format {expected_ratio:g}:1 nötig.")
        if width < 256 or height < 128:
            raise ValueError("Das Panorama muss mindestens 256 × 128 Pixel groß sein.")
        panorama = np.asarray(image)
    reference_distance = None
    if args.reference_depth:
        reference_distance = np.load(args.reference_depth.resolve()).astype(np.float32)
        if reference_distance.ndim != 2 or abs(reference_distance.shape[1] / reference_distance.shape[0] - width / height) > 0.02:
            raise ValueError("Die Referenztiefe muss dieselbe Projektion wie das Panorama haben.")
        if not np.isfinite(reference_distance).all() or (reference_distance <= 0).any():
            raise ValueError("Die Referenztiefe enthält ungültige Entfernungen.")
    reference_scale = None
    device = resolve_device(args.device)
    views = cube_views()
    if coverage == 180:
        views = [view for view in views if view.name != "back"]
    tangent_half_fov = math.tan(math.radians(args.view_fov) / 2)
    focal_length_px = args.view_resolution / (2 * tangent_half_fov)
    emit("SHARP-Modell wird geladen", progress=0.03, device=str(device), viewCount=len(views))
    if args.checkpoint:
        state_dict = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=True)
    else:
        state_dict = torch.hub.load_state_dict_from_url(
            DEFAULT_MODEL_URL, map_location="cpu", progress=False, weights_only=True
        )
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    del state_dict
    predictor.eval().to(device)
    upscaler = None
    if args.upscale == "auto" and args.resolution / args.view_resolution >= ai_upscale.MIN_UPSCALE_FACTOR:
        upscaler = ai_upscale.load_model(device)
    chunks = []
    view_records = []
    view_directory = output_path.parent / (output_path.stem + "_views")
    if args.save_views:
        view_directory.mkdir(parents=True, exist_ok=True)
    for index, view in enumerate(views):
        emit(
            f"SHARP: {coverage}°-Ansicht {index + 1}/{len(views)}",
            progress=0.08 + index / len(views) * 0.8,
            view=view.name,
            viewIndex=index + 1,
            viewCount=len(views),
        )
        perspective = sample_perspective(panorama, view, args.view_resolution, focal_length_px, coverage)
        if args.save_views:
            Image.fromarray(perspective).save(view_directory / (view.name + ".jpg"), quality=95)
        view_focal_px = focal_length_px
        if upscaler is not None:
            perspective, factor = ai_upscale.upscale_to(perspective, args.resolution, device, upscaler)
            view_focal_px *= factor
        view_reference = None
        monodepth_scale = 1.0
        reference_record = {}
        own_depth = None
        if args.save_views and reference_distance is None:
            own_depth = predict_monodepth(predictor, perspective, view_focal_px, device, args.resolution)
            save_depth_previews(view_directory, view.name, {"sharp-depth": own_depth})
        if reference_distance is not None:
            reference, valid = sample_reference_depth(
                reference_distance, view, args.view_resolution, focal_length_px, coverage
            )
            monodepth = predict_monodepth(predictor, perspective, view_focal_px, device, args.resolution)
            view_reference, reference_scale, view_ratio = reference_depth_for_view(
                reference, valid, monodepth, reference_scale
            )
            # Pre-scale SHARP's depth to the reference so alignment only fixes local residuals.
            monodepth_scale = 1 / view_ratio
            reference_record = {"sharpToReferenceDepthRatio": round(view_ratio, 4)}
            if args.save_views:
                save_depth_previews(view_directory, view.name, {
                    "sharp-depth": monodepth / view_ratio, "reference-depth": view_reference,
                })
            del monodepth
        try:
            predicted = predict_image(
                predictor, perspective, view_focal_px, device, args.resolution, view_reference,
                monodepth_scale,
            )
        except torch.OutOfMemoryError as error:
            raise RuntimeError(
                f"Nicht genügend GPU-Speicher für {args.resolution} px. "
                "Eine kleinere SHARP-Auflösung wählen oder andere GPU-Anwendungen schließen."
            ) from error
        point_budget = 0
        if args.max_splats:
            point_budget = args.max_splats // len(views) + (index < args.max_splats % len(views))
        chunk, counts = rotate_and_filter(
            predicted, view, tangent_half_fov, point_budget, args.minimum_opacity,
            seed=7100 + index, coverage=coverage,
        )
        chunks.append(chunk)
        view_records.append(
            {"name": view.name, "cameraToWorld": view.rotation.tolist(), **counts, **reference_record}
        )
        if args.save_views:
            # Per-view Gaussians (with overlap fades) and stats for the pipeline view.
            write_panorama_ply([chunk], view_directory / f"{view.name}.ply", args.view_resolution, focal_length_px, coverage)
            (view_directory / f"{view.name}.json").write_text(json.dumps(view_records[-1]), encoding="utf-8")
        del predicted
    del predictor, upscaler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    emit(f"{coverage}°-Splats werden zusammengeführt", progress=0.9)
    count = write_panorama_ply(chunks, output_path, args.view_resolution, focal_length_px, coverage)
    metadata = {
        "version": 1,
        "kind": f"panorama{coverage}",
        "coverageDegrees": coverage,
        "depthEstimator": "sharp",
        "input": str(input_path),
        "output": str(output_path),
        "projection": "equirectangular",
        "panoramaResolution": [width, height],
        "viewResolution": args.view_resolution,
        "viewFovDegrees": args.view_fov,
        "sharpResolution": args.resolution,
        "upscale": args.upscale,
        "focalLengthPx": focal_length_px,
        "worldCoordinates": "OpenCV +X right +Y down +Z forward",
        "origin": [0, 0, 0],
        "viewerRotationXYZDegrees": [180, 0, 0],
        "pointCount": count,
        "durationSeconds": round(time.monotonic() - started, 2),
        "referenceDepth": None if reference_distance is None else {
            "path": str(args.reference_depth.resolve()),
            "sceneScale": reference_scale,
            "calibration": "median SHARP/reference depth ratio on the front view",
        },
        "views": view_records,
        "limitation": LIMITATION.replace("360°", f"{coverage}°"),
    }
    manifest_path = output_path.with_suffix(".json")
    temporary_manifest = manifest_path.with_name(manifest_path.name + ".partial")
    temporary_manifest.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    emit(
        f"{coverage}°-Szene ist fertig",
        status="ready",
        progress=1,
        output=str(output_path),
        pointCount=count,
        durationSeconds=metadata["durationSeconds"],
        limitation=metadata["limitation"],
    )
    return metadata


def main() -> int:
    """Parse CLI arguments and keep progress/errors in the JSON line protocol."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="A 2:1 full ERP panorama or square 180-degree ERP image.")
    parser.add_argument("--output", type=Path, required=True, help="Output .ply file or directory.")
    parser.add_argument("--coverage", type=int, choices=(180, 360), default=360,
                        help="Horizontal ERP coverage: square front hemisphere or full 2:1 sphere.")
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, cpu or mps.")
    parser.add_argument("--checkpoint", type=Path, help="Optional existing SHARP .pt checkpoint.")
    parser.add_argument("--view-resolution", type=int, default=1024, help="Square perspective image size.")
    parser.add_argument("--view-fov", type=float, default=100, help="Square perspective field of view in degrees.")
    parser.add_argument("--max-splats", type=int,
                        help="Scene point budget; 0 keeps all supported splats. "
                        "Default: 3 M at 1536 px, growing with the square of --resolution.")
    parser.add_argument("--minimum-opacity", type=float, default=0.003, help="Discard almost invisible splats.")
    parser.add_argument("--save-views", action="store_true",
                        help="Also save each view's image, depth previews and Gaussians next to the output.")
    parser.add_argument("--reference-depth", type=Path,
                        help="Experimental: ERP .npy distance map from da2_reference.py, same projection as --input.")
    parser.add_argument("--resolution", type=int, default=DEFAULT_INTERNAL_RESOLUTION,
                        help="Square SHARP network resolution per view (multiple of 64, 768-4096; trained: 1536).")
    parser.add_argument("--upscale", choices=("off", "auto"), default="off",
                        help="auto: AI-upscale each view to --resolution before SHARP.")
    args = parser.parse_args()
    try:
        validate_internal_resolution(args.resolution)
    except ValueError as error:
        parser.error(str(error))
    if args.max_splats is None:
        # Finer views yield more splats; uniform thinning would discard that detail.
        args.max_splats = round(3000000 * (args.resolution / DEFAULT_INTERNAL_RESOLUTION) ** 2)
    if not 256 <= args.view_resolution <= 2048:
        parser.error("--view-resolution must be between 256 and 2048")
    if not 92 <= args.view_fov <= 120:
        parser.error("--view-fov must be between 92 and 120 for complete overlapping cube coverage")
    if args.max_splats < 0 or 0 < args.max_splats < 6:
        parser.error("--max-splats must be 0 or at least 6")
    if not 0 < args.minimum_opacity < 1:
        parser.error("--minimum-opacity must be between 0 and 1")
    try:
        run(args)
        return 0
    except Exception as error:
        emit(f"{args.coverage}°-Verarbeitung fehlgeschlagen", status="error", error=str(error))
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
