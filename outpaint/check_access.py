"""Check Hugging Face access to the gated base models before the large downloads.

Prints one JSON object: {"user": ..., "missing": {tool: url}}.
"""

from __future__ import annotations

import json
import sys
import tempfile

from huggingface_hub import HfApi, hf_hub_download

GATED = {
    "flux-fill": "black-forest-labs/FLUX.1-Fill-dev",
    "flux-fill-bf16": "black-forest-labs/FLUX.1-Fill-dev",
    "dit360": "black-forest-labs/FLUX.1-dev",
    "klein-9b": "black-forest-labs/FLUX.2-klein-base-9B",
    "krea2": "krea/Krea-2-Turbo",
}


def main() -> int:
    try:
        user = HfApi().whoami()["name"]
    except Exception:
        print(json.dumps({"user": None, "missing": {}}))
        return 0
    wanted = sys.argv[1:] or list(GATED)
    missing = {}
    with tempfile.TemporaryDirectory() as directory:
        for tool in wanted:
            repo = GATED.get(tool)
            if not repo:
                continue
            try:
                hf_hub_download(repo, "model_index.json", local_dir=directory)
            except Exception:
                missing[tool] = f"https://huggingface.co/{repo}"
    print(json.dumps({"user": user, "missing": missing}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
