"""Finish a generated 2:1 ERP panorama for the studio.

- Restores the original photo with a soft, colour-matched edge (360 degree
  models may redraw it).
- For 180 degrees: cuts out the front hemisphere (the middle half of the
  columns) and enlarges it to a square with Real-ESRGAN, the format the
  studio's FLUX Fill path produces (180 x 180 degrees, e.g. 2048 x 2048).
- For 360 degrees: blends the left/right wrap seam.
Runs in .venv-outpaint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from common import (emit, hemisphere_from_360, load_photo, photo_fov, project_photo, reinsert_original, run_cli,
                    save_png, soften_wrap)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="Generated 2:1 ERP PNG")
    parser.add_argument("--photo", type=Path, required=True, help="Original photo")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coverage", type=int, choices=(180, 360), required=True)
    parser.add_argument("--fov", type=float)
    parser.add_argument("--size", type=int, default=2048, help="Output height (180: square side)")
    parser.add_argument("--band", type=int, default=48, help="Soft edge around the original in px")
    args = parser.parse_args()
    fov = args.fov or photo_fov(args.photo)[0]
    with Image.open(args.input) as source:
        panorama = np.asarray(source.convert("RGB"))
    if abs(panorama.shape[1] / panorama.shape[0] - 2) > 0.01:
        raise ValueError("Das erzeugte Panorama ist nicht im Format 2:1.")
    if args.coverage == 180:
        emit("finish", 10, message="Vordere Halbkugel wird ausgeschnitten und per KI vergrößert")
        import torch

        import ai_upscale

        front = hemisphere_from_360(panorama)
        if ai_upscale.status()["ready"]:
            front, _ = ai_upscale.upscale_to(front, args.size, torch.device("cuda"))
        if front.shape[:2] != (args.size, args.size):
            front = np.asarray(Image.fromarray(front).resize((args.size, args.size), Image.Resampling.LANCZOS))
        height = width = args.size
        panorama = front
    else:
        height, width = args.size, 2 * args.size
        if panorama.shape[:2] != (height, width):
            panorama = np.asarray(Image.fromarray(panorama).resize((width, height), Image.Resampling.LANCZOS))
    emit("finish", 60, message="Originalfoto wird eingesetzt")
    projected, known = project_photo(load_photo(args.photo), width, height, args.coverage, fov)
    result = reinsert_original(panorama.astype(np.float32), projected, known, args.band)
    if args.coverage == 360:
        result = soften_wrap(result)
    save_png(result, args.output)
    emit("ready", 100, message="Panorama ist fertig", output=str(args.output), fov=fov)


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
