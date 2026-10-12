$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw 'Die SHARP-Umgebung .venv fehlt. Bitte zuerst die Python-Abhängigkeiten installieren.'
}

& $python -m pip install -r requirements-upscale.txt
if ($LASTEXITCODE -ne 0) { throw 'Die Upscaler-Abhängigkeiten konnten nicht installiert werden.' }

& $python ai_upscale.py --download-model
if ($LASTEXITCODE -ne 0) { throw 'Das Upscaler-Modell konnte nicht geladen werden.' }

Write-Host 'KI-Upscaler Real-ESRGAN x4plus ist bereit (BSD-3-Clause).'
