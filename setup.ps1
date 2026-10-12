<#
Installiert SHARP Studio auf diesem PC und fragt, welche zusätzlichen Modelle
dazukommen sollen. Alles läuft lokal auf diesem Rechner.

Mehrfaches Ausführen ist unbedenklich: Fertige Downloads werden übersprungen,
weitere Modelle lassen sich später ergänzen.

  powershell -ExecutionPolicy Bypass -File .\setup.ps1
  powershell -ExecutionPolicy Bypass -File .\setup.ps1 -Models upscale,flux-fill,dit360
#>
param(
    [ValidateSet('upscale', 'flux-fill', 'klein-lora', 'da3', 'da2', 'dit360', 'flux-fill-bf16', 'klein-9b', 'qwen360', 'krea2')]
    [string[]]$Models
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$DiT360Commit = '3779fe7965473f6824994c663a0ae7a76bc7aafa'

function Step($text) { Write-Host ''; Write-Host "== $text" -ForegroundColor Cyan }

function Invoke-Checked($description, [scriptblock]$command) {
    & $command
    if ($LASTEXITCODE -ne 0) { throw "$description ist fehlgeschlagen (Code $LASTEXITCODE)." }
}

function Test-Marker($path, $pattern) {
    $file = Join-Path $PSScriptRoot $path
    if (-not (Test-Path -LiteralPath $file)) { return $false }
    if (-not $pattern) { return $true }
    return (Get-Content -Raw -LiteralPath $file) -match $pattern
}

function New-Environment($name, $requirements) {
    $python = Join-Path $PSScriptRoot "$name\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $python)) {
        Invoke-Checked "Anlegen von $name" { py -3.12 -m venv $name }
    }
    Invoke-Checked "PyTorch-Installation in $name" {
        & $python -m pip install --disable-pip-version-check -q torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
    }
    foreach ($file in $requirements) {
        Invoke-Checked "Paketinstallation in $name ($file)" {
            & $python -m pip install --disable-pip-version-check -q -r $file
        }
    }
    return $python
}

# Name, Beschreibung, ungefähre Downloadgröße, Lizenz, Prüfung auf fertige Installation.
$Catalog = @(
    [pscustomobject]@{ Name = 'upscale';        Label = 'KI-Upscaler Real-ESRGAN';                     GB = 1;  License = 'BSD-3-Clause';                    Marker = 'models\upscale\RealESRGAN_x4plus.pth'; Pattern = $null }
    [pscustomobject]@{ Name = 'flux-fill';      Label = 'FLUX Fill 4-Bit · 180° (16-GB-Karten)';      GB = 20; License = 'FLUX.1 Fill: nicht-kommerziell';  Marker = 'models\fill180\nf4\transformer\config.json'; Pattern = $null }
    [pscustomobject]@{ Name = 'klein-lora';     Label = 'FLUX.2 Klein 4B + 360°-LoRA';                 GB = 16; License = 'Apache 2.0';                      Marker = 'models\panorama\ready.json'; Pattern = $null }
    [pscustomobject]@{ Name = 'da3';            Label = 'Depth Anything 3 (anderes 3D-Modell)';       GB = 7;  License = 'CC BY-NC 4.0';                    Marker = 'models\depth\native-ready.json'; Pattern = $null }
    [pscustomobject]@{ Name = 'da2';            Label = 'DA² Tiefenangleich (braucht DA3)';           GB = 2;  License = 'Apache 2.0';                      Marker = 'models\da2\model.safetensors'; Pattern = $null }
    [pscustomobject]@{ Name = 'dit360';         Label = 'DiT360 · 180°/360° (ab ~40 GB VRAM)';        GB = 34; License = 'FLUX.1-dev: nicht-kommerziell';   Marker = 'models\outpaint\dit360\ready.json'; Pattern = $null }
    [pscustomobject]@{ Name = 'flux-fill-bf16'; Label = 'FLUX Fill volle Genauigkeit · 180° (~40 GB)'; GB = 34; License = 'FLUX.1 Fill: nicht-kommerziell';  Marker = 'models\fill180\ready.json';  Pattern = '"bf16"' }
    [pscustomobject]@{ Name = 'klein-9b';       Label = 'FLUX.2 Klein 9B + 360°-LoRA (~40 GB)';       GB = 34; License = 'FLUX.2 Klein: nicht-kommerziell'; Marker = 'models\panorama-9b\ready.json'; Pattern = $null }
    [pscustomobject]@{ Name = 'qwen360';        Label = 'Qwen-Image-2512 + 360°-LoRA (~64 GB)';       GB = 58; License = 'Apache 2.0 / MIT';                Marker = 'models\outpaint\qwen360\ready.json'; Pattern = $null }
    [pscustomobject]@{ Name = 'krea2';          Label = 'Krea 2 + 360°-Outpaint-LoRA (~48 GB)';       GB = 40; License = 'Krea 2 Community License';        Marker = 'models\outpaint\krea2\ready.json'; Pattern = $null }
)
# Models that use the scene description (Qwen3-VL, ~17 GB) and the .venv-outpaint environment.
$NeedsOutpaint = @('dit360', 'flux-fill-bf16', 'klein-9b', 'qwen360', 'krea2')
$Recommended = @('upscale', 'flux-fill', 'dit360')

if (-not $Models) {
    Write-Host ''
    Write-Host 'Welche Modelle sollen installiert werden? (SHARP selbst wird immer eingerichtet.)' -ForegroundColor Cyan
    for ($i = 0; $i -lt $Catalog.Count; $i++) {
        $item = $Catalog[$i]
        $state = if (Test-Marker $item.Marker $item.Pattern) { '  [installiert]' } else { '' }
        Write-Host ("  {0,2}) {1,-46} ~{2,3} GB  {3}{4}" -f ($i + 1), $item.Label, $item.GB, $item.License, $state)
    }
    Write-Host '  Die Modelle 6-10 nutzen zusätzlich eine automatische Bildbeschreibung (~17 GB, einmalig).'
    $default = ($Recommended | ForEach-Object { $name = $_; [array]::IndexOf(@($Catalog.Name), $name) + 1 }) -join ','
    $answer = Read-Host "Nummern mit Komma trennen, ""a"" für alle, Enter für $default"
    if (-not $answer.Trim()) { $answer = $default }
    if ($answer.Trim() -eq 'a') {
        $Models = @($Catalog.Name)
    } else {
        $Models = @()
        foreach ($part in ($answer -split '[,; ]+' | Where-Object { $_ })) {
            $number = 0
            if (-not [int]::TryParse($part, [ref]$number) -or $number -lt 1 -or $number -gt $Catalog.Count) {
                throw "Ungültige Auswahl: $part"
            }
            $Models += $Catalog[$number - 1].Name
        }
    }
}
$Models = @($Models | Select-Object -Unique)
if ($Models -contains 'da2' -and $Models -notcontains 'da3' -and -not (Test-Marker 'models\depth\native-ready.json')) {
    $Models += 'da3'
    Write-Host 'DA² braucht Depth Anything 3; DA3 wird mit installiert.'
}
$outpaint = @($Models | Where-Object { $NeedsOutpaint -contains $_ })
$total = 3 + ($Catalog | Where-Object { $Models -contains $_.Name } | Measure-Object -Property GB -Sum).Sum
if ($outpaint.Count) { $total += 17 }
Write-Host "Ausgewählt: $(if ($Models.Count) { $Models -join ', ' } else { 'nur SHARP' }) (Download bis zu ~$total GB)"

Step 'Voraussetzungen prüfen'
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw 'Der Python-Launcher "py" fehlt. Python 3.12 von python.org installieren (mit "py launcher").'
}
Invoke-Checked 'Python 3.12' { py -3.12 --version }
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    throw 'Node.js (npm) fehlt. Die LTS-Version von nodejs.org installieren und das Skript erneut starten.'
}
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    throw 'nvidia-smi fehlt. Bitte zuerst den aktuellen NVIDIA-Treiber installieren.'
}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$freeGB = [math]::Round((Get-PSDrive -Name (Get-Location).Drive.Name).Free / 1GB)
Write-Host "Freier Speicher: $freeGB GB (benötigt: etwa $($total + 25) GB)"
if ($freeGB -lt $total + 25) { Write-Warning 'Möglicherweise reicht der Speicherplatz nicht.' }

Step 'SHARP und Weboberfläche einrichten'
$sharp = New-Environment '.venv' @('requirements.txt', 'requirements-upscale.txt')
Invoke-Checked 'SHARP-Checkpoint' {
    & $sharp -c "import torch; from sharp.cli.predict import DEFAULT_MODEL_URL; torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)"
}
Invoke-Checked 'npm ci' { npm --prefix web ci }
Invoke-Checked 'Web-Build' { npm --prefix web run build }

$gated = @($Models | Where-Object { @('flux-fill', 'flux-fill-bf16', 'dit360', 'klein-9b', 'krea2') -contains $_ })
$blocked = @()
if ($gated.Count) {
    Step 'Hugging-Face-Anmeldung'
    $access = & $sharp outpaint\check_access.py @gated | ConvertFrom-Json
    if (-not $access.user) {
        Write-Host 'Bitte mit einem Hugging-Face-Token (Berechtigung "Read") anmelden: https://huggingface.co/settings/tokens'
        Invoke-Checked 'Hugging-Face-Anmeldung' { & (Join-Path $PSScriptRoot '.venv\Scripts\hf.exe') auth login }
        $access = & $sharp outpaint\check_access.py @gated | ConvertFrom-Json
    }
    Write-Host "Angemeldet als $($access.user)."
    foreach ($property in $access.missing.PSObject.Properties) {
        $blocked += $property.Name
        Write-Warning "$($property.Name): Modellbedingungen noch nicht angenommen. Bitte öffnen und zustimmen, dann das Skript erneut starten: $($property.Value)"
    }
}

$failed = @()
$panoramaPython = $null
$outpaintPython = $null
$dit360Python = $null
if ($Models | Where-Object { @('flux-fill-bf16', 'klein-9b') -contains $_ }) {
    Step 'Umgebung .venv-panorama'
    $panoramaPython = New-Environment '.venv-panorama' @('requirements-panorama.txt')
}
if ($outpaint.Count) {
    Step 'Umgebung .venv-outpaint und Bildbeschreibung'
    $outpaintPython = New-Environment '.venv-outpaint' @('outpaint\requirements-outpaint.txt')
    Invoke-Checked 'Bildbeschreibung' { & $outpaintPython outpaint\caption.py --download-models }
    Invoke-Checked 'KI-Upscaler für 180°-Ausschnitte' { & $outpaintPython ai_upscale.py --download-model }
}
if ($Models -contains 'dit360') {
    Step 'Umgebung .venv-dit360'
    $dit360Python = New-Environment '.venv-dit360' @('outpaint\requirements-dit360.txt')
    $vendor = Join-Path $PSScriptRoot 'outpaint\vendor\DiT360'
    if (-not (Test-Path -LiteralPath (Join-Path $vendor 'pa_src\pipeline.py'))) {
        Write-Host "DiT360-Code wird geladen (Commit $($DiT360Commit.Substring(0, 7)))"
        $zip = Join-Path $env:TEMP "DiT360-$DiT360Commit.zip"
        Invoke-WebRequest -UseBasicParsing "https://github.com/Insta360-Research-Team/DiT360/archive/$DiT360Commit.zip" -OutFile $zip
        $extract = Join-Path $env:TEMP "DiT360-$DiT360Commit"
        if (Test-Path -LiteralPath $extract) { Remove-Item -Recurse -Force -LiteralPath $extract }
        Expand-Archive -LiteralPath $zip -DestinationPath $extract
        New-Item -ItemType Directory -Force -Path (Split-Path $vendor) | Out-Null
        if (Test-Path -LiteralPath $vendor) { Remove-Item -Recurse -Force -LiteralPath $vendor }
        Move-Item -LiteralPath (Join-Path $extract "DiT360-$DiT360Commit") -Destination $vendor
        Remove-Item -Force -LiteralPath $zip
    }
}

# Order matters: DA² builds on the DA3 environment.
foreach ($model in @($Catalog.Name | Where-Object { $Models -contains $_ })) {
    if ($blocked -contains $model) { $failed += $model; continue }
    Step "Modell: $model"
    $code = 0
    switch ($model) {
        'upscale'        { & powershell -NoProfile -ExecutionPolicy Bypass -File .\setup-upscale.ps1; $code = $LASTEXITCODE }
        'flux-fill'      { & powershell -NoProfile -ExecutionPolicy Bypass -File .\setup-panorama.ps1 -FillTool flux-fill; $code = $LASTEXITCODE }
        'klein-lora'     { & powershell -NoProfile -ExecutionPolicy Bypass -File .\setup-panorama.ps1 -FillTool klein-lora; $code = $LASTEXITCODE }
        'da3'            { & powershell -NoProfile -ExecutionPolicy Bypass -File .\setup-depth.ps1; $code = $LASTEXITCODE }
        'da2'            { & powershell -NoProfile -ExecutionPolicy Bypass -File .\setup-da2.ps1; $code = $LASTEXITCODE }
        'dit360'         { & $dit360Python outpaint\dit360_pano.py --download-models; $code = $LASTEXITCODE }
        'flux-fill-bf16' { & $panoramaPython fill180_generate.py --download-models --precision bf16; $code = $LASTEXITCODE }
        'klein-9b'       { & $panoramaPython panorama_generate.py --download-models --model 9b; $code = $LASTEXITCODE }
        'qwen360'        { & $outpaintPython outpaint\qwen360_pano.py --download-models; $code = $LASTEXITCODE }
        'krea2'          { & $outpaintPython outpaint\krea2_pano.py --download-models; $code = $LASTEXITCODE }
    }
    if ($code -ne 0) { $failed += $model }
}

Step 'Fertig'
if ($outpaint.Count) { & $sharp outpaint\tools.py }
if ($failed.Count) {
    Write-Warning ('Nicht eingerichtet: ' + ($failed -join ', ') + '. Hinweise oben beachten und das Skript erneut ausführen.')
}
Write-Host ''
Write-Host 'Starten: Doppelklick auf "SHARP Studio.cmd". Die Bilderweiterung wählt man im Zahnrad unter "Bild-Ausfülltool".'
