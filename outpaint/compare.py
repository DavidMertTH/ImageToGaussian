"""Run several panorama tools on the same photos and lay the results side by side.

    .venv-outpaint\\Scripts\\python.exe outpaint\\compare.py foto1.jpg foto2.jpg
    .venv-outpaint\\Scripts\\python.exe outpaint\\compare.py foto.jpg --tools krea2 dit360 --coverage 360

Writes outpaint/compare/<time>/: one folder per photo and tool, a sheet per photo
(<photo>.jpg) and results.json with durations and errors. The scene
description is computed once per photo and shared by all tools, so the
comparison is fair. Fixed seed by default.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from common import TOOL_DIR
from tools import TOOLS, run_job, run_step, python, ENV_OUTPAINT, tool_status

TILE_HEIGHT = 512


def sheet(photo: Path, results: list[tuple[str, Path | None, str]], coverage: int, target: Path) -> None:
    """Original photo plus one tile per tool, each with a label."""
    tile_width = TILE_HEIGHT if coverage == 180 else 2 * TILE_HEIGHT
    tiles = []
    with Image.open(photo) as source:
        original = ImageOps.exif_transpose(source).convert("RGB")
    original.thumbnail((tile_width, TILE_HEIGHT))
    tiles.append(("Original", original))
    for label, path, note in results:
        if path and path.is_file():
            with Image.open(path) as image:
                tiles.append((f"{label} · {note}", image.convert("RGB").resize((tile_width, TILE_HEIGHT))))
        else:
            tiles.append((f"{label} · {note}", Image.new("RGB", (tile_width, TILE_HEIGHT), (60, 20, 20))))
    columns = 3 if coverage == 180 else 2
    rows = (len(tiles) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * tile_width, rows * (TILE_HEIGHT + 28)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, tile) in enumerate(tiles):
        x, y = (index % columns) * tile_width, (index // columns) * (TILE_HEIGHT + 28)
        canvas.paste(tile, (x + (tile_width - tile.width) // 2, y + 28))
        draw.text((x + 8, y + 7), label, fill="black")
    canvas.save(target, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("photos", nargs="+", type=Path)
    parser.add_argument("--tools", nargs="+", choices=tuple(TOOLS), help="Default: all ready tools")
    parser.add_argument("--coverage", type=int, choices=(180, 360), default=180)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    status = tool_status()
    names = args.tools or [name for name, item in status.items() if item["ready"]]
    names = [name for name in names if args.coverage in TOOLS[name]["coverages"]]
    if not names:
        raise SystemExit("Keine eingerichteten Tools für diese Abdeckung. setup.ps1 ausführen.")
    output = TOOL_DIR / "compare" / datetime.now().strftime("%Y%m%d-%H%M%S")
    output.mkdir(parents=True)
    summary = {}
    for photo in args.photos:
        photo = photo.resolve()
        print(f"\n== {photo.name}", flush=True)
        folder = output / photo.stem
        folder.mkdir()
        with (folder / "caption.log").open("w", encoding="utf-8") as log:
            run_step([python(ENV_OUTPAINT), TOOL_DIR / "caption.py", "--input", photo,
                      "--output", folder / "caption.json"], log, lambda _event: None)
        caption = json.loads((folder / "caption.json").read_text(encoding="utf-8"))["caption"]
        print(f"   Beschreibung: {caption}", flush=True)
        results, entries = [], {}
        for name in names:
            label = TOOLS[name]["label"]
            print(f"   {label} …", end="", flush=True)
            started = time.monotonic()
            try:
                result = run_job(name, photo, args.coverage, folder / name, lambda _stage, _percent: None,
                                 prompt=caption if TOOLS[name]["caption"] else None, seed=args.seed)
                seconds = round(time.monotonic() - started)
                shutil.copy(result, folder / f"{name}.png")
                results.append((label, result, f"{seconds} s"))
                entries[name] = {"seconds": seconds, "output": str(folder / f"{name}.png")}
                print(f" {seconds} s", flush=True)
            except Exception as exc:
                results.append((label, None, "Fehler"))
                entries[name] = {"error": str(exc)}
                print(f" Fehler: {str(exc)[:300]}", flush=True)
        sheet(photo, results, args.coverage, output / f"{photo.stem}.jpg")
        summary[photo.name] = {"caption": caption, "tools": entries}
        (output / "results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nVergleich: {output}", flush=True)


if __name__ == "__main__":
    main()
