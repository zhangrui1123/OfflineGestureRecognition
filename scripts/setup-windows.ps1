# Install finger-ml on Windows with Python 3.10+.
# Usage: powershell -ExecutionPolicy Bypass -File scripts/setup-windows.ps1

$ErrorActionPreference = "Stop"

function Find-Python {
    foreach ($ver in @("3.11", "3.12", "3.10")) {
        $cmd = Get-Command "py" -ErrorAction SilentlyContinue
        if ($cmd) {
            $out = & py "-$ver" -c "import sys; print(sys.version_info[:2] >= (3, 10))" 2>$null
            if ($out -eq "True") {
                return @("py", "-$ver")
            }
        }
        $exe = Get-Command "python$ver" -ErrorAction SilentlyContinue
        if ($exe) {
            return @($exe.Source)
        }
    }
    throw "Python 3.10+ is required. Install from https://www.python.org/downloads/windows/"
}

$py = Find-Python
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "[setup] Using Python: $($py -join ' ')"
& @py -m pip install --upgrade pip
& @py -m pip install -e .

Write-Host "[setup] Installing CPU PyTorch (pinned for Windows DLL compatibility)..."
& @py -m pip install "torch==2.4.1+cpu" "torchvision==0.19.1+cpu" --index-url https://download.pytorch.org/whl/cpu
& @py -m pip install "tqdm>=4.66.0" "scikit-learn>=1.4.0"

Write-Host "[setup] Verifying imports..."
& @py -c "import finger_ml, torch; print('[ok] finger-ml', finger_ml.__version__, '| torch', torch.__version__)"

Write-Host ""
Write-Host "Setup complete. Run commands with:"
Write-Host "  py -3.11 train.py --data-dir data"
Write-Host "  py -3.11 inference.py --video data/video/<session>.mp4"
