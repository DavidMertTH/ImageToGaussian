$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

# DA² (Depth Anything in Any Direction) shares .venv-depth with DA3; run setup-depth.ps1 first.
$depthPython = Join-Path $PSScriptRoot '.venv-depth\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $depthPython)) {
    throw 'Die Umgebung .venv-depth fehlt. Zuerst setup-depth.ps1 ausführen.'
}

# Upstream pins torch 2.5, which does not support current GPUs. The model code only
# needs torch, torchvision, einops and huggingface_hub from .venv-depth, so install
# the pinned sources without their dependency lists. utils3d is imported by the
# package's __init__ and is pure Python.
& $depthPython -m pip install --no-deps `
    'da2 @ git+https://github.com/EnVision-Research/DA-2.git@d659838585f2bc9967c7e9367af573795271dbbb#subdirectory=src' `
    'utils3d @ git+https://github.com/EasternJournalist/utils3d.git@3913c65d81e05e47b9f367250cf8c0f7462a0900'
if ($LASTEXITCODE -ne 0) { throw 'DA² konnte nicht installiert werden.' }

& $depthPython -c @'
from huggingface_hub import snapshot_download
snapshot_download("haodongli/DA-2", revision="0d55ccb5e46b8ed4715fae3a4c04fc897f1689f3",
                  local_dir="models/da2", allow_patterns=["model.safetensors", "README.md"])
'@
if ($LASTEXITCODE -ne 0) { throw 'Die DA²-Gewichte konnten nicht geladen werden.' }

Write-Host 'DA² ist bereit (Code und Gewichte: Apache 2.0).'
