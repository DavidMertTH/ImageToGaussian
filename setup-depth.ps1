$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

if (-not (Test-Path -LiteralPath '.venv-depth\Scripts\python.exe')) {
    py -3.12 -m venv .venv-depth
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 wird für DA3 benötigt.' }
}

$depthPython = Join-Path $PSScriptRoot '.venv-depth\Scripts\python.exe'
& $depthPython -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'Die CUDA-Abhängigkeiten konnten nicht installiert werden.' }

& $depthPython -m pip install -r requirements-depth.txt
if ($LASTEXITCODE -ne 0) { throw 'Die DA3-Abhängigkeiten konnten nicht installiert werden.' }

# Native GS uses DA3's learned Gaussian heads. e3nn rotates their spherical
# harmonics; no gsplat renderer or compiled CUDA extension is needed for export.
# Install the source without unrelated upstream web/development dependencies.
& $depthPython -m pip install --no-deps 'depth-anything-3 @ git+https://github.com/ByteDance-Seed/Depth-Anything-3.git@3d835ec1a5802d64a8b8b15f817a1ab54809bfe4'
if ($LASTEXITCODE -ne 0) { throw 'Depth Anything 3 konnte nicht installiert werden.' }

& $depthPython da3_runtime.py --download-models
if ($LASTEXITCODE -ne 0) { throw 'Das native DA3-Gaussian-Modell konnte nicht vorbereitet werden.' }

Write-Host 'Native DA3-Gaussians sind bereit. Modellgewichte: CC BY-NC 4.0.'
