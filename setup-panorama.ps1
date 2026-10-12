param(
    [ValidateSet('flux-fill', 'klein-lora')]
    [string]$FillTool = 'flux-fill'
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

if (-not (Test-Path -LiteralPath '.venv-panorama\Scripts\python.exe')) {
    py -3.12 -m venv .venv-panorama
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 wird für die Panorama-Umgebung benötigt.' }
}

$panoramaPython = Join-Path $PSScriptRoot '.venv-panorama\Scripts\python.exe'
& $panoramaPython -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'Die CUDA-Abhängigkeiten konnten nicht installiert werden.' }

& $panoramaPython -m pip install -r requirements-panorama.txt
if ($LASTEXITCODE -ne 0) { throw 'Die Panorama-Abhängigkeiten konnten nicht installiert werden.' }

$generatorScript = if ($FillTool -eq 'flux-fill') { 'fill180_generate.py' } else { 'panorama_generate.py' }
& $panoramaPython $generatorScript --download-models
if ($LASTEXITCODE -ne 0) { throw 'Die Panorama-Modelle konnten nicht heruntergeladen werden.' }

Write-Host 'Das Ausfülltool ist bereit. SHARP Studio starten und das Tool in den Einstellungen auswählen.'
