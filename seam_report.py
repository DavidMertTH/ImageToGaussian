"""Measure depth disagreement between overlapping views of a panorama scene.

Experimental diagnostic for panorama_splats.py output. The PLY stores the views'
Gaussians one after another; scene.json lists how many each view kept. For every
direction bin seen by two views, the front surface of each view is estimated as
a low percentile of its Gaussians' radial distances. The report gives the median
and 90th percentile of |log(r_a / r_b)| over all shared bins, per view pair:
0.10 means the two views place the same surface about 10 % apart.

Usage (project Python environment)::

    python seam_report.py output/ui/<id>/scene.ply [other.ply ...]
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData


def view_chunks(ply_path: Path) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Return (view name, unit directions, radii) for each view's Gaussians."""
    metadata = json.loads(ply_path.with_suffix(".json").read_text(encoding="utf-8"))
    vertices = PlyData.read(str(ply_path))["vertex"]
    means = np.stack([vertices[axis] for axis in ("x", "y", "z")], axis=-1).astype(np.float64)
    opacity = 1 / (1 + np.exp(-np.asarray(vertices["opacity"], dtype=np.float64)))
    counts = [view["kept"] for view in metadata["views"]]
    if sum(counts) != len(means):
        raise ValueError(f"{ply_path}: {sum(counts)} Splats laut scene.json, {len(means)} in der PLY.")
    chunks = []
    for view, (start, end) in zip(metadata["views"], itertools.pairwise(np.cumsum([0, *counts]))):
        # Faded overlap Gaussians are nearly transparent; they would bias the surface estimate.
        keep = opacity[start:end] > 0.05
        radius = np.linalg.norm(means[start:end][keep], axis=-1)
        chunks.append((view["name"], means[start:end][keep] / radius[:, None], radius))
    return chunks


def front_surface(directions: np.ndarray, radius: np.ndarray, bins: int, percentile: float, minimum: int):
    """Per direction bin: a low radius percentile, or NaN where too few Gaussians fall."""
    longitude = np.arctan2(directions[:, 0], directions[:, 2])
    latitude = np.arcsin(np.clip(-directions[:, 1], -1, 1))
    column = np.clip(((longitude / (2 * np.pi) + 0.5) * 2 * bins).astype(int), 0, 2 * bins - 1)
    row = np.clip(((0.5 - latitude / np.pi) * bins).astype(int), 0, bins - 1)
    index = row * 2 * bins + column
    order = np.argsort(index, kind="stable")
    index, radius = index[order], radius[order]
    result = np.full(2 * bins * bins, np.nan)
    starts = np.flatnonzero(np.r_[True, index[1:] != index[:-1]])
    for start, end in itertools.pairwise([*starts, len(index)]):
        if end - start >= minimum:
            result[index[start]] = np.percentile(radius[start:end], percentile)
    return result


def report(ply_path: Path, bins: int, percentile: float, minimum: int) -> dict:
    surfaces = {name: front_surface(d, r, bins, percentile, minimum) for name, d, r in view_chunks(ply_path)}
    pairs, everything = {}, []
    for (name_a, a), (name_b, b) in itertools.combinations(surfaces.items(), 2):
        shared = np.isfinite(a) & np.isfinite(b)
        if shared.sum() < 20:
            continue
        error = np.abs(np.log(a[shared] / b[shared]))
        everything.append(error)
        pairs[f"{name_a}/{name_b}"] = {
            "bins": int(shared.sum()),
            "median": round(float(np.median(error)), 4),
            "p90": round(float(np.percentile(error, 90)), 4),
        }
    error = np.concatenate(everything)
    return {"scene": str(ply_path), "median": round(float(np.median(error)), 4),
            "p90": round(float(np.percentile(error, 90)), 4), "pairs": pairs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenes", type=Path, nargs="+")
    parser.add_argument("--bins", type=int, default=360, help="Latitude bins; longitude uses twice as many.")
    parser.add_argument("--percentile", type=float, default=20, help="Radius percentile used as the front surface.")
    parser.add_argument("--minimum", type=int, default=8, help="Gaussians a view needs in a bin.")
    args = parser.parse_args()
    for scene in args.scenes:
        result = report(scene, args.bins, args.percentile, args.minimum)
        print(f"{result['scene']}\n  alle Nähte: Median {result['median']:.3f}, 90 % {result['p90']:.3f}")
        for pair, values in result["pairs"].items():
            print(f"  {pair:<12} Median {values['median']:.3f}, 90 % {values['p90']:.3f} ({values['bins']} Felder)")


if __name__ == "__main__":
    main()
