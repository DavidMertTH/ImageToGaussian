"""Front-hemisphere projections in OpenCV coordinates (+Z forward, +Y down)."""

import math

import numpy as np


def erp_rays(width: int, height: int, coverage: int = 180) -> np.ndarray:
    longitude = ((np.arange(width, dtype=np.float32) + 0.5) / width - 0.5) * math.radians(coverage)
    latitude = (0.5 - (np.arange(height, dtype=np.float32) + 0.5) / height) * math.pi
    lon, lat = np.meshgrid(longitude, latitude)
    return np.stack((np.sin(lon) * np.cos(lat), -np.sin(lat), np.cos(lon) * np.cos(lat)), -1)


def camera_rotation(yaw: float, pitch: float) -> np.ndarray:
    """Camera-to-world matrix; positive pitch looks down."""
    yaw, pitch = math.radians(yaw), math.radians(pitch)
    forward = np.array([math.sin(yaw) * math.cos(pitch), math.sin(pitch), math.cos(yaw) * math.cos(pitch)], np.float32)
    right = np.array([math.cos(yaw), 0, -math.sin(yaw)], np.float32)
    return np.stack((right, np.cross(forward, right), forward), axis=1)


def pinhole_rays(width: int, height: int, hfov: float, rotation=None) -> np.ndarray:
    focal = width / (2 * math.tan(math.radians(hfov) / 2))
    x = (np.arange(width, dtype=np.float32) + 0.5 - width / 2) / focal
    y = (np.arange(height, dtype=np.float32) + 0.5 - height / 2) / focal
    xx, yy = np.meshgrid(x, y)
    rays = np.stack((xx, yy, np.ones_like(xx)), -1)
    return rays if rotation is None else rays @ rotation.T


def bilinear(array: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Clamp at image edges; a front hemisphere never wraps left to right."""
    height, width = array.shape[:2]
    x, y = np.clip(x, 0, width - 1), np.clip(y, 0, height - 1)
    x0, y0 = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    wx, wy = x - x0, y - y0
    if array.ndim == 3:
        wx, wy = wx[..., None], wy[..., None]
    return (array[y0, x0] * (1 - wx) + array[y0, x1] * wx) * (1 - wy) + (array[y1, x0] * (1 - wx) + array[y1, x1] * wx) * wy


def sample_erp(array: np.ndarray, rays: np.ndarray, coverage: int = 180):
    height, width = array.shape[:2]
    longitude = np.arctan2(rays[..., 0], rays[..., 2])
    latitude = np.arctan2(-rays[..., 1], np.hypot(rays[..., 0], rays[..., 2]))
    x = (longitude / math.radians(coverage) + 0.5) * width - 0.5
    y = (0.5 - latitude / math.pi) * height - 0.5
    valid = np.abs(longitude) <= math.radians(coverage) / 2 + 1e-6
    return bilinear(array, x, y), valid


def sample_pinhole(array: np.ndarray, rays: np.ndarray, hfov: float, rotation=None):
    height, width = array.shape[:2]
    local = rays if rotation is None else rays @ rotation
    z = local[..., 2]
    focal = width / (2 * math.tan(math.radians(hfov) / 2))
    safe_z = np.maximum(z, 1e-8)
    x = focal * local[..., 0] / safe_z + width / 2 - 0.5
    y = focal * local[..., 1] / safe_z + height / 2 - 0.5
    valid = (z > 1e-6) & (x >= -0.5) & (x <= width - 0.5) & (y >= -0.5) & (y <= height - 0.5)
    return bilinear(array, x, y), valid


# 120-degree patches overlap at the hemisphere's corner directions as well.
FILL_VIEWS = (("Mitte", 0, 0), ("Rechts", 45, 0), ("Links", -45, 0), ("Oben", 0, -45), ("Unten", 0, 45))
PATCH_FOV = 120.0
