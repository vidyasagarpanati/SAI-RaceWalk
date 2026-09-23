<#
    SAI race walking biomechanics pipeline - Windows bootstrap.

    Run once, from the repository root, in PowerShell:

        Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
        .\scripts\bootstrap.ps1

    Creates .venv, installs pinned dependencies, installs ffmpeg, downloads the
    MediaPipe pose model, and runs the environment check.
#>

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
Write-Host "Repository: $repo" -ForegroundColor Cyan

# --- 1. Python --------------------------------------------------------------
$pythonExe = $null
foreach ($candidate in @("py -3.12", "py -3.11", "python")) {
    $parts = $candidate.Split(" ")
    $cmd = $parts[0]
    if (Get-Command $cmd -ErrorAction SilentlyContinue) {
        $argsList = if ($parts.Length -gt 1) { @($parts[1], "-c", "import sys;print('%d.%d'%sys.version_info[:2])") }
                    else { @("-c", "import sys;print('%d.%d'%sys.version_info[:2])") }
        try { $ver = & $cmd @argsList } catch { continue }
        if ($ver -match "^3\.(11|12)$") { $pythonExe = $candidate; Write-Host "Python $ver via '$candidate'" -ForegroundColor Green; break }
    }
}
if (-not $pythonExe) {
    throw "Need Python 3.11 or 3.12 on PATH. MediaPipe publishes no wheels for 3.13. Install from python.org, then rerun."
}

# --- 2. Virtual environment -------------------------------------------------
if (-not (Test-Path ".venv")) {
    Write-Host "Creating .venv ..." -ForegroundColor Cyan
    Invoke-Expression "$pythonExe -m venv .venv"
}
$venvPy = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) { throw "venv creation failed: $venvPy not found" }

Write-Host "Installing dependencies (pinned) ..." -ForegroundColor Cyan
& $venvPy -m pip install --upgrade pip --quiet
& $venvPy -m pip install -r requirements.txt
& $venvPy -m pip install -e .

# --- 3. ffmpeg --------------------------------------------------------------
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host "ffmpeg not on PATH." -ForegroundColor Yellow
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Host "Installing ffmpeg via winget ..." -ForegroundColor Cyan
        winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
        Write-Host "Open a NEW PowerShell window afterwards so PATH picks up ffmpeg." -ForegroundColor Yellow
    } else {
        Write-Host "winget unavailable. imageio-ffmpeg ships a bundled binary and will be used as a fallback." -ForegroundColor Yellow
        Write-Host "ffprobe will be missing, so video metadata falls back to OpenCV." -ForegroundColor Yellow
    }
}

# --- 4. MediaPipe pose model ------------------------------------------------
$modelDir = Join-Path $repo "models"
New-Item -ItemType Directory -Force -Path $modelDir | Out-Null
$modelPath = Join-Path $modelDir "pose_landmarker_full.task"
if (-not (Test-Path $modelPath)) {
    $url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
    Write-Host "Downloading MediaPipe pose model (model_complexity 1 == 'full') ..." -ForegroundColor Cyan
    Invoke-WebRequest -Uri $url -OutFile $modelPath -UseBasicParsing
}
Write-Host ("Pose model: {0} ({1:N1} MB)" -f $modelPath, ((Get-Item $modelPath).Length / 1MB)) -ForegroundColor Green

# --- 5. Ollama --------------------------------------------------------------
if (Get-Command ollama -ErrorAction SilentlyContinue) {
    Write-Host "`nOllama models available:" -ForegroundColor Cyan
    ollama list
    Write-Host "Copy the tag you want into config/pipeline.yaml under llm.model" -ForegroundColor Yellow
} else {
    Write-Host "Ollama not found on PATH. Install it from https://ollama.com/download, then run: ollama list" -ForegroundColor Yellow
}

# --- 6. Verify --------------------------------------------------------------
Write-Host "`nRunning environment check ..." -ForegroundColor Cyan
& $venvPy -m racewalk doctor

Write-Host @"

Next:
  .\.venv\Scripts\Activate.ps1
  racewalk init-session --video "D:\path\to\walk.mp4"
  notepad sessions\walk.json      # set athlete_name, camera_view, treadmill_speed_kmh
  racewalk run --video "D:\path\to\walk.mp4" --to S5
"@ -ForegroundColor Green
