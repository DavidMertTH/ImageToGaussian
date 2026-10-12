"""Contains `sharp predict` CLI implementation.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data

from sharp.models import (
    PredictorParams,
    RGBGaussianPredictor,
    create_predictor,
)
from sharp.utils import io
from sharp.utils import logging as logging_utils
from sharp.utils.gaussians import (
    Gaussians3D,
    SceneMetaData,
    save_ply,
    unproject_gaussians,
)

from .render import render_gaussians

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
DEFAULT_INTERNAL_RESOLUTION = 1536
MIN_INTERNAL_RESOLUTION = 768
MAX_INTERNAL_RESOLUTION = 4096


def validate_internal_resolution(value: int) -> None:
    """Raise if the SHARP network cannot run at this square input size."""
    if not MIN_INTERNAL_RESOLUTION <= value <= MAX_INTERNAL_RESOLUTION or value % 64:
        raise ValueError(
            f"Internal resolution must be a multiple of 64 between {MIN_INTERNAL_RESOLUTION} "
            f"and {MAX_INTERNAL_RESOLUTION}, got {value}."
        )


@click.command()
@click.option(
    "-i",
    "--input-path",
    type=click.Path(path_type=Path, exists=True),
    help="Path to an image or containing a list of images.",
    required=True,
)
@click.option(
    "-o",
    "--output-path",
    type=click.Path(path_type=Path, file_okay=False),
    help="Path to save the predicted Gaussians and renderings.",
    required=True,
)
@click.option(
    "-c",
    "--checkpoint-path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Path to the .pt checkpoint. If not provided, downloads the default model automatically.",
    required=False,
)
@click.option(
    "--render/--no-render",
    "with_rendering",
    is_flag=True,
    default=False,
    help="Whether to render trajectory for checkpoint.",
)
@click.option(
    "--device",
    type=str,
    default="default",
    help="Device to run on. ['cpu', 'mps', 'cuda']",
)
@click.option(
    "--resolution",
    "internal_resolution",
    type=int,
    default=DEFAULT_INTERNAL_RESOLUTION,
    show_default=True,
    help="Square network input size in px (multiple of 64, 768-4096). "
    "Larger values produce more, finer Gaussians and need more GPU memory.",
)
@click.option("-v", "--verbose", is_flag=True, help="Activate debug logs.")
def predict_cli(
    input_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    with_rendering: bool,
    device: str,
    internal_resolution: int,
    verbose: bool,
):
    """Predict Gaussians from input images."""
    logging_utils.configure(logging.DEBUG if verbose else logging.INFO)
    try:
        validate_internal_resolution(internal_resolution)
    except ValueError as error:
        raise click.BadParameter(str(error), param_hint="--resolution") from error

    extensions = io.get_supported_image_extensions()

    image_paths = []
    if input_path.is_file():
        if input_path.suffix in extensions:
            image_paths = [input_path]
    else:
        for ext in extensions:
            image_paths.extend(list(input_path.glob(f"**/*{ext}")))

    if len(image_paths) == 0:
        LOGGER.info("No valid images found. Input was %s.", input_path)
        return

    LOGGER.info("Processing %d valid image files.", len(image_paths))

    if device == "default":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    LOGGER.info("Using device %s", device)

    if with_rendering and device != "cuda":
        LOGGER.warning("Can only run rendering with gsplat on CUDA. Rendering is disabled.")
        with_rendering = False

    # Load or download checkpoint
    if checkpoint_path is None:
        LOGGER.info("No checkpoint provided. Downloading default model from %s", DEFAULT_MODEL_URL)
        state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)
    else:
        LOGGER.info("Loading checkpoint from %s", checkpoint_path)
        state_dict = torch.load(checkpoint_path, weights_only=True)

    gaussian_predictor = create_predictor(PredictorParams())
    gaussian_predictor.load_state_dict(state_dict)
    gaussian_predictor.eval()
    gaussian_predictor.to(device)

    output_path.mkdir(exist_ok=True, parents=True)

    for image_path in image_paths:
        LOGGER.info("Processing %s", image_path)
        image, _, f_px = io.load_rgb(image_path)
        height, width = image.shape[:2]
        intrinsics = torch.tensor(
            [
                [f_px, 0, (width - 1) / 2.0, 0],
                [0, f_px, (height - 1) / 2.0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            device=device,
            dtype=torch.float32,
        )
        gaussians = predict_image(
            gaussian_predictor, image, f_px, torch.device(device), internal_resolution
        )

        LOGGER.info("Saving 3DGS to %s", output_path)
        save_ply(gaussians, f_px, (height, width), output_path / f"{image_path.stem}.ply")

        if with_rendering:
            output_video_path = (output_path / image_path.stem).with_suffix(".mp4")
            LOGGER.info("Rendering trajectory to %s", output_video_path)

            metadata = SceneMetaData(intrinsics[0, 0].item(), (width, height), "linearRGB")
            render_gaussians(gaussians, metadata, output_video_path)


def _prepare_image(
    image: np.ndarray, f_px: float, device: torch.device, internal_resolution: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize an RGB image to the network input and compute its disparity factor."""
    validate_internal_resolution(internal_resolution)
    image_pt = torch.from_numpy(image.copy()).float().to(device).permute(2, 0, 1) / 255.0
    _, height, width = image_pt.shape
    disparity_factor = torch.tensor([f_px / width]).float().to(device)
    image_resized_pt = F.interpolate(
        image_pt[None],
        size=(internal_resolution, internal_resolution),
        mode="bilinear",
        align_corners=True,
    )
    return image_resized_pt, disparity_factor


@torch.no_grad()
def predict_monodepth(
    predictor: RGBGaussianPredictor,
    image: np.ndarray,
    f_px: float,
    device: torch.device,
    internal_resolution: int = DEFAULT_INTERNAL_RESOLUTION,
) -> torch.Tensor:
    """Return SHARP's own camera-Z depth of the visible layer at internal resolution."""
    image_resized_pt, disparity_factor = _prepare_image(image, f_px, device, internal_resolution)
    disparity = predictor.monodepth_model(image_resized_pt).disparity[:, 0]
    return (disparity_factor[:, None, None] / disparity.clamp(min=1e-4, max=1e4))[0]


@torch.no_grad()
def predict_image(
    predictor: RGBGaussianPredictor,
    image: np.ndarray,
    f_px: float,
    device: torch.device,
    internal_resolution: int = DEFAULT_INTERNAL_RESOLUTION,
    reference_depth: torch.Tensor | None = None,
    monodepth_scale: float = 1.0,
) -> Gaussians3D:
    """Predict Gaussians from an image.

    The image is resized to a square of internal_resolution pixels. The model was
    trained at 1536; larger values yield more and finer Gaussians (4x per doubling)
    at the cost of memory and with an input scale the model has not seen in training.

    reference_depth is an optional camera-Z depth map in the units of SHARP's own
    monodepth (any resolution, same framing as image). It is passed to SHARP's
    learned depth alignment, which rescales its monodepth towards the reference
    with a local scale map. Apple trained this path with ground-truth depth; feeding
    another model's estimate at inference time is experimental.

    monodepth_scale multiplies SHARP's own depth before that alignment. The learned
    alignment only corrects moderate deviations, so a view whose depth is off by a
    large global factor should be pre-scaled towards the reference first.
    """
    internal_shape = (internal_resolution, internal_resolution)

    LOGGER.info("Running preprocessing.")
    height, width = image.shape[:2]
    image_resized_pt, disparity_factor = _prepare_image(image, f_px, device, internal_resolution)
    # Monodepth is disparity_factor / disparity; the factor is not used anywhere else.
    disparity_factor = disparity_factor * monodepth_scale
    depth_pt = None
    if reference_depth is not None:
        depth_pt = F.interpolate(
            reference_depth.float().to(device)[None, None],
            size=(internal_shape[1], internal_shape[0]),
            mode="bilinear",
            align_corners=True,
        )

    # Predict Gaussians in the NDC space.
    LOGGER.info("Running inference.")
    gaussians_ndc = predictor(image_resized_pt, disparity_factor, depth_pt)

    LOGGER.info("Running postprocessing.")
    intrinsics = (
        torch.tensor(
            [
                [f_px, 0, width / 2, 0],
                [0, f_px, height / 2, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        .float()
        .to(device)
    )
    intrinsics_resized = intrinsics.clone()
    intrinsics_resized[0] *= internal_shape[0] / width
    intrinsics_resized[1] *= internal_shape[1] / height

    # Convert Gaussians to metrics space.
    gaussians = unproject_gaussians(
        gaussians_ndc, torch.eye(4).to(device), intrinsics_resized, internal_shape
    )

    return gaussians
