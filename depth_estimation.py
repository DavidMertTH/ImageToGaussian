"""Geometry helpers for external metric depth without importing DA3 or PyTorch.

Depth bundles store camera-Z in meters, together with each map's actual
intrinsics and camera-to-world rotation. A surrounding panorama has no single
camera-Z axis: it is fused/sampled as radial distance and converted back to Z
for each SHARP perspective camera.

DA3METRIC-LARGE estimates each perspective image independently. The optional
radial consensus below is a geometric blend of those estimates, not joint
multi-view neural inference and not a guarantee of consistent metric geometry.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import ExifTags, Image, ImageOps


MODEL_ID = "depth-anything/DA3METRIC-LARGE"
MODEL_REVISION = "4010e39f3634a45bc60553321fb49fb760bd594e"
MODEL_LICENSE = "Apache-2.0"
CODE_REVISION = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"


@dataclass
class DepthBundle:
    """Metric perspective maps and optional common ERP radial consensus."""

    depth: np.ndarray
    intrinsics: np.ndarray
    camera_to_world: np.ndarray
    view_names: tuple[str, ...]
    valid: np.ndarray
    confidence: np.ndarray | None = None
    radial_depth: np.ndarray | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class CubeView:
    """OpenCV camera rotation matching panorama_splats.py exactly."""

    name: str
    rotation: np.ndarray


def cube_views() -> list[CubeView]:
    """Create six right-handed perspective cameras around one origin."""
    specifications = [
        ("front", (0, 0, 1)), ("right", (1, 0, 0)),
        ("back", (0, 0, -1)), ("left", (-1, 0, 0)),
        ("up", (0, -1, 0)), ("down", (0, 1, 0)),
    ]
    views = []
    for name, forward_values in specifications:
        forward = np.asarray(forward_values, dtype=np.float32)
        right = (
            np.asarray((1, 0, 0), dtype=np.float32)
            if abs(forward[1]) == 1
            else np.cross(forward, np.asarray((0, -1, 0), dtype=np.float32))
        )
        down = np.cross(forward, right)
        views.append(CubeView(name, np.stack((right, down, forward), axis=1)))
    return views


def make_intrinsics(width: int, height: int, focal_length_px: float) -> np.ndarray:
    """Return the SHARP-compatible pinhole matrix with principal point w/2,h/2."""
    return np.asarray(
        [[focal_length_px, 0, width / 2], [0, focal_length_px, height / 2], [0, 0, 1]],
        dtype=np.float32,
    )


def estimate_focal_length(
    image: Image.Image | Path | str, image_size: tuple[int, int] | None = None
) -> float:
    """Mirror SHARP's EXIF/30-mm fallback focal estimate in pixel units.

    An explicit image_size is (width,height) after orientation handling. The
    image diagonal is unchanged by EXIF rotation. Tiny physical focal lengths
    are multiplied by SHARP's approximate 8.4 sensor crop factor.
    """
    if isinstance(image, (Path, str)):
        with Image.open(image) as opened:
            dimensions = image_size or ImageOps.exif_transpose(opened).size
            return estimate_focal_length(opened, dimensions)
    exif = image.getexif()
    entries = dict(exif)
    try:
        entries.update(exif.get_ifd(0x8769))
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    by_name = {ExifTags.TAGS.get(key, str(key)): value for key, value in entries.items()}

    def number(value: Any) -> float | None:
        try:
            numeric = float(value)
            return numeric if math.isfinite(numeric) else None
        except (TypeError, ValueError, ZeroDivisionError):
            return None

    equivalent = number(by_name.get("FocalLengthIn35mmFilm", by_name.get("FocalLenIn35mmFilm")))
    if equivalent is None or equivalent < 1:
        equivalent = number(by_name.get("FocalLength"))
        if equivalent is None or equivalent <= 0:
            equivalent = 30.0
        if equivalent < 10:
            equivalent *= 8.4
    width, height = image_size or image.size
    return equivalent * math.hypot(width, height) / math.hypot(36, 24)


def _bilinear(
    values: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray | None = None,
    *,
    wrap_x: bool = False,
) -> np.ndarray:
    """Interpolate finite valid samples, preserving missing/sky regions as NaN."""
    height, width = values.shape
    finite_coordinates = np.isfinite(x) & np.isfinite(y)
    inside = finite_coordinates & (y >= 0) & (y <= height - 1)
    if not wrap_x:
        inside &= (x >= 0) & (x <= width - 1)
    safe_x = np.where(finite_coordinates, x, 0)
    safe_y = np.where(finite_coordinates, y, 0)
    safe_x = safe_x % width if wrap_x else np.clip(safe_x, 0, width - 1)
    safe_y = np.clip(safe_y, 0, height - 1)
    x0 = np.floor(safe_x).astype(np.int32)
    y0 = np.floor(safe_y).astype(np.int32)
    x1 = (x0 + 1) % width if wrap_x else np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = (safe_x - x0).astype(np.float32)
    wy = (safe_y - y0).astype(np.float32)
    validity = np.isfinite(values)
    if valid is not None:
        validity &= valid
    total = np.zeros_like(wx, dtype=np.float32)
    weight_sum = np.zeros_like(wx, dtype=np.float32)
    for xs, ys, weight in (
        (x0, y0, (1 - wx) * (1 - wy)),
        (x1, y0, wx * (1 - wy)),
        (x0, y1, (1 - wx) * wy),
        (x1, y1, wx * wy),
    ):
        sample_valid = validity[ys, xs]
        effective = weight * sample_valid
        total += np.where(sample_valid, values[ys, xs], 0) * effective
        weight_sum += effective
    result = total / np.maximum(weight_sum, 1e-8)
    return np.where(inside & (weight_sum >= 0.5), result, np.nan).astype(np.float32)


def sample_perspective_rgb(
    panorama: np.ndarray, rotation: np.ndarray, resolution: int, focal_length_px: float
) -> np.ndarray:
    """Sample full ERP into the exact cube-camera geometry used by SHARP."""
    coordinates = (np.arange(resolution, dtype=np.float32) - resolution / 2) / focal_length_px
    xx, yy = np.meshgrid(coordinates, coordinates)
    world_rays = np.stack((xx, yy, np.ones_like(xx)), axis=-1) @ rotation.T
    radius = np.linalg.norm(world_rays, axis=-1)
    longitude = np.arctan2(world_rays[..., 0], world_rays[..., 2])
    latitude = np.arcsin(np.clip(-world_rays[..., 1] / radius, -1, 1))
    height, width = panorama.shape[:2]
    x = ((longitude / (2 * np.pi) + 0.5) * width - 0.5) % width
    y = np.clip((0.5 - latitude / np.pi) * height - 0.5, 0, height - 1)
    rgb = np.stack([_bilinear(panorama[..., channel], x, y, wrap_x=True) for channel in range(3)], axis=-1)
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def load_depth_bundle(path: Path | str) -> DepthBundle:
    """Load only numeric/string NPZ fields; object arrays are never unpickled."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as stored:
        depth = np.asarray(stored["depth"], dtype=np.float32)
        intrinsics = np.asarray(stored["intrinsics"], dtype=np.float32)
        rotations = np.asarray(stored["camera_to_world"], dtype=np.float32)
        names = tuple(str(name) for name in stored["view_names"])
        valid = np.asarray(stored["valid"], dtype=bool) if "valid" in stored else np.isfinite(depth) & (depth > 0)
        confidence = np.asarray(stored["confidence"], dtype=np.float32) if "confidence" in stored else None
        radial = np.asarray(stored["radial_depth"], dtype=np.float32) if "radial_depth" in stored else None
        metadata = json.loads(str(stored["metadata_json"].item())) if "metadata_json" in stored else {}
    if depth.ndim != 3 or depth.shape[0] != len(names):
        raise ValueError("Ungültige Tiefenkarten: [views,height,width] wird erwartet.")
    count = depth.shape[0]
    if intrinsics.shape != (count, 3, 3) or rotations.shape != (count, 3, 3):
        raise ValueError("Ungültige Kameramatrizen im Tiefen-Bundle.")
    if not np.isfinite(intrinsics).all() or not np.isfinite(rotations).all():
        raise ValueError("Kameramatrizen enthalten ungültige Werte.")
    if (intrinsics[:, (0, 1), (0, 1)] <= 0).any():
        raise ValueError("Brennweiten im Tiefen-Bundle müssen positiv sein.")
    if valid.shape != depth.shape or (confidence is not None and confidence.shape != depth.shape):
        raise ValueError("Ungültige Maskenform im Tiefen-Bundle.")
    if radial is not None and radial.ndim != 2:
        raise ValueError("Die radiale Konsensuskarte muss zweidimensional sein.")
    if metadata.get("depthDefinition", "camera_z_meters") != "camera_z_meters":
        raise ValueError("Das Tiefen-Bundle enthält keine Kamera-Z-Tiefe in Metern.")
    valid &= np.isfinite(depth) & (depth > 0)
    return DepthBundle(depth, intrinsics, rotations, names, valid, confidence, radial, metadata)


def _sample_views(
    bundle: DepthBundle, world_directions: np.ndarray, view_name: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Reproject world rays into each map and return radial estimates + weights."""
    estimates, weights = [], []
    indices = (
        [bundle.view_names.index(view_name)] if view_name is not None
        else range(len(bundle.view_names))
    )
    for index in indices:
        camera_dirs = world_directions @ bundle.camera_to_world[index]
        z = camera_dirs[:, 2]
        front = z > 1e-6
        projected = camera_dirs @ bundle.intrinsics[index].T
        x = projected[:, 0] / np.maximum(z, 1e-6)
        y = projected[:, 1] / np.maximum(z, 1e-6)
        depth = _bilinear(bundle.depth[index], x, y, bundle.valid[index])
        radial = depth / np.maximum(z, 1e-6)
        height, width = bundle.depth[index].shape
        radius_x = np.abs(x - bundle.intrinsics[index, 0, 2]) / (width / 2)
        radius_y = np.abs(y - bundle.intrinsics[index, 1, 2]) / (height / 2)
        edge = np.maximum(radius_x, radius_y)
        fade = np.clip((1 - edge) / 0.22, 0, 1)
        fade = fade * fade * (3 - 2 * fade)
        weight = fade * np.maximum(z, 0) ** 4
        if bundle.confidence is not None:
            conf = _bilinear(bundle.confidence[index], x, y, bundle.valid[index])
            # Confidence is a model score, not a calibrated probability. Only
            # positive finite scores are used; do not present it as certainty.
            weight *= np.where(np.isfinite(conf), np.maximum(conf, 0), 0)
        supported = front & np.isfinite(radial) & (radial > 0)
        weights.append(np.where(supported, weight, 0))
        estimates.append(np.where(supported, radial, np.nan))
    return np.asarray(estimates, dtype=np.float32), np.asarray(weights, dtype=np.float32)


def _blend_radial(estimates: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Blend metric distances in log space and softly downweight disagreements."""
    active = (weights > 0) & np.isfinite(estimates) & (estimates > 0)
    log_depth = np.log(np.where(active, estimates, 1))
    sums = np.where(active, weights, 0).sum(axis=0)
    center = (log_depth * np.where(active, weights, 0)).sum(axis=0) / np.maximum(sums, 1e-8)
    deviation = np.abs(log_depth - center[None])
    robust = np.minimum(1, 0.3 / np.maximum(deviation, 1e-8))
    effective = np.where(active, weights * robust, 0)
    total = effective.sum(axis=0)
    blended = np.exp((log_depth * effective).sum(axis=0) / np.maximum(total, 1e-8))
    return np.where(total > 1e-8, blended, np.nan).astype(np.float32)


def build_radial_consensus(bundle: DepthBundle, width: int = 1024, height: int = 512) -> np.ndarray:
    """Fuse six independently inferred maps into an explicitly heuristic ERP map."""
    if width < 2 or height < 2 or abs(width / height - 2) > 0.01:
        raise ValueError("Die radiale Panorama-Karte braucht das Seitenverhältnis 2:1.")
    longitude = ((np.arange(width, dtype=np.float32) + 0.5) / width - 0.5) * (2 * np.pi)
    latitude = (0.5 - (np.arange(height, dtype=np.float32) + 0.5) / height) * np.pi
    lon, lat = np.meshgrid(longitude, latitude)
    directions = np.stack((np.sin(lon) * np.cos(lat), -np.sin(lat), np.cos(lon) * np.cos(lat)), axis=-1)
    estimates, weights = _sample_views(bundle, directions.reshape(-1, 3))
    return _blend_radial(estimates, weights).reshape(height, width)


def sample_radial_depth(
    bundle: DepthBundle,
    world_directions: np.ndarray,
    *,
    use_consensus: bool = True,
    view_name: str | None = None,
) -> np.ndarray:
    """Return radial meters for arbitrary world directions; missing values are NaN.

    Input may be [N,3] or an image-shaped [...,3]. It need not be normalized.
    use_consensus=True prefers a stored common radial map. Otherwise perspective
    maps are reprojected on demand (one named view, or a weighted blend).
    """
    world_directions = np.asarray(world_directions, dtype=np.float32)
    if world_directions.shape[-1] != 3:
        raise ValueError("Richtungen müssen drei Koordinaten haben.")
    output_shape = world_directions.shape[:-1]
    flattened = world_directions.reshape(-1, 3)
    norms = np.linalg.norm(flattened, axis=-1)
    directions = flattened / np.maximum(norms[:, None], 1e-8)
    valid_dirs = np.isfinite(directions).all(axis=-1) & (norms > 1e-8)
    if use_consensus and bundle.radial_depth is not None:
        height, width = bundle.radial_depth.shape
        longitude = np.arctan2(directions[:, 0], directions[:, 2])
        latitude = np.arcsin(np.clip(-directions[:, 1], -1, 1))
        x = ((longitude / (2 * np.pi) + 0.5) * width - 0.5) % width
        y = np.clip((0.5 - latitude / np.pi) * height - 0.5, 0, height - 1)
        result = _bilinear(bundle.radial_depth, x, y, wrap_x=True)
    else:
        estimates, weights = _sample_views(bundle, directions, view_name)
        result = _blend_radial(estimates, weights)
    return np.where(valid_dirs, result, np.nan).reshape(output_shape)


def sample_depth_in_camera(
    bundle: DepthBundle,
    camera_to_world: np.ndarray,
    intrinsics: np.ndarray,
    image_shape: tuple[int, int],
    *,
    view_name: str | None = None,
    use_consensus: bool = True,
) -> np.ndarray:
    """Resample camera-Z meters for a requested SHARP camera and (height,width)."""
    height, width = image_shape
    yy, xx = np.mgrid[:height, :width]
    pixels = np.stack((xx, yy, np.ones_like(xx)), axis=-1).astype(np.float32)
    camera_rays = pixels @ np.linalg.inv(np.asarray(intrinsics, dtype=np.float32)).T
    world_rays = camera_rays @ np.asarray(camera_to_world, dtype=np.float32).T
    radial = sample_radial_depth(bundle, world_rays, use_consensus=use_consensus, view_name=view_name)
    norm = np.linalg.norm(camera_rays, axis=-1)
    # Pinhole rays have local Z=1: radial distance / ray length equals Z.
    return (radial * camera_rays[..., 2] / np.maximum(norm, 1e-8)).astype(np.float32)
