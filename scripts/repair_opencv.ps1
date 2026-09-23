<#
    Repairs the OpenCV / MediaPipe setup on Windows.

    MediaPipe imports cv2 at package-import time (mediapipe/__init__.py pulls in
    solutions -> drawing_utils -> cv2), so cv2 is mandatory, and it must be the
    single distribution MediaPipe is built against: opencv-contrib-python.
    opencv-python, opencv-python-headless and the contrib-headless variant all
    install into the same site-packages\cv2 directory, and mixing them leaves
    mismatched native DLLs behind.

    Two failure modes this fixes:
      ImportError: DLL load failed while importing _framework_bindings   (mixed installs)
      ModuleNotFoundError: No module named 'cv2'                         (none installed)

    Run from the repository root:
        .\scripts\repair_opencv.ps1
#>

$ErrorActionPreference = "Stop"

function Invoke-Checked {
    # $ErrorActionPreference does NOT trip on a native command's exit code, so
    # every pip call is checked by hand. The previous version of this script
    # swallowed a failed install and reported success.
    param([string[]]$Cmd, [string]$What, [switch]$AllowFailure)
    Write-Host "  > $($Cmd -join ' ')" -ForegroundColor DarkGray
    & $Cmd[0] @($Cmd[1..($Cmd.Length - 1)])
    if ($LASTEXITCODE -ne 0) {
        if ($AllowFailure) {
            Write-Host "  (non-fatal) $What exited $LASTEXITCODE" -ForegroundColor Yellow
            return $false
        }
        throw "$What failed with exit code $LASTEXITCODE. Output is above."
    }
    return $true
}

$repo = Split-Path -Parent $PSScriptRoot
$py = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "venv not found at $py. Run scripts\bootstrap.ps1 first." }

$TARGET = "opencv-contrib-python==4.11.0.86"

Write-Host "OpenCV distributions currently installed:" -ForegroundColor Cyan
& $py -m pip list | Select-String -Pattern "opencv"

Write-Host "`nRemoving every OpenCV distribution ..." -ForegroundColor Cyan
Invoke-Checked @($py, "-m", "pip", "uninstall", "-y",
                 "opencv-python", "opencv-python-headless",
                 "opencv-contrib-python", "opencv-contrib-python-headless") `
               "uninstall" -AllowFailure | Out-Null

$cv2 = & $py -c "import sysconfig,os;print(os.path.join(sysconfig.get_paths()['purelib'],'cv2'))"
if (Test-Path $cv2) {
    Write-Host "Removing leftover $cv2 ..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $cv2
}

Write-Host "`nInstalling $TARGET ..." -ForegroundColor Cyan
$ok = Invoke-Checked @($py, "-m", "pip", "install", "--no-cache-dir", $TARGET) `
                     "pinned opencv install" -AllowFailure
if (-not $ok) {
    Write-Host "Pinned version unavailable for this interpreter. Falling back to the newest 4.x ..." -ForegroundColor Yellow
    Invoke-Checked @($py, "-m", "pip", "install", "--no-cache-dir",
                     "opencv-contrib-python<5") "fallback opencv install" | Out-Null
    Write-Host "NOTE: the installed OpenCV no longer matches requirements.txt. Tell Claude which version landed." -ForegroundColor Yellow
}

Write-Host "`nEnsuring MediaPipe is intact (without disturbing numpy) ..." -ForegroundColor Cyan
Invoke-Checked @($py, "-m", "pip", "install", "--no-cache-dir", "--no-deps",
                 "--force-reinstall", "mediapipe==0.10.21") "mediapipe reinstall" | Out-Null
Invoke-Checked @($py, "-m", "pip", "install", "numpy==1.26.4", "protobuf==4.25.9") `
               "pin numpy and protobuf back" | Out-Null

Write-Host "`nVerifying the exact imports S2 performs ..." -ForegroundColor Cyan
$verify = @'
import cv2, numpy, mediapipe as mp
from mediapipe.tasks.python import vision
print("cv2", cv2.__version__, "| numpy", numpy.__version__, "| mediapipe", mp.__version__)
print("mediapipe.tasks.python.vision imported OK")
'@
$verify | & $py -
if ($LASTEXITCODE -ne 0) {
    throw "Verification failed. Run: .\.venv\Scripts\python.exe scripts\diagnose_mediapipe.py"
}

Write-Host "`nRunning the environment check ..." -ForegroundColor Cyan
& $py -m racewalk doctor
