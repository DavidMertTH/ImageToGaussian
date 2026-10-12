$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    throw 'Die SHARP-Umgebung .venv fehlt. Bitte zuerst die Python-Abhängigkeiten installieren.'
}

if (-not (Test-Path -LiteralPath 'web\node_modules')) {
    npm --prefix web ci
    if ($LASTEXITCODE -ne 0) { throw 'npm ci ist fehlgeschlagen.' }
}

npm --prefix web run build
if ($LASTEXITCODE -ne 0) { throw 'Der Web-Build ist fehlgeschlagen.' }

& '.venv\Scripts\python.exe' 'app.py' --open
