# Setup script for Windows (with optional CUDA support)
# Uses uv package manager

$ErrorActionPreference = "Stop"

Write-Host "=== AI Audio Detector - Windows Setup ===" -ForegroundColor Cyan

# Check for uv
try {
    $uvVersion = uv --version
    Write-Host "Using uv: $uvVersion"
} catch {
    Write-Host "Installing uv package manager..." -ForegroundColor Yellow
    irm https://astral.sh/uv/install.ps1 | iex
    Write-Host "Please restart your shell and run this script again." -ForegroundColor Yellow
    exit 1
}

# Create virtual environment
Write-Host ""
Write-Host "Creating virtual environment..." -ForegroundColor Green
uv venv --python 3.12

# Activate venv
.\.venv\Scripts\Activate.ps1

# Install core dependencies
Write-Host ""
Write-Host "Installing core dependencies..." -ForegroundColor Green
uv pip install -e .

# Detect CUDA availability
$hasCuda = $false
try {
    $nvidiaSmi = nvidia-smi 2>$null
    if ($LASTEXITCODE -eq 0) {
        $hasCuda = $true
    }
} catch {}

if ($hasCuda) {
    Write-Host ""
    Write-Host "NVIDIA GPU detected. Installing PyTorch with CUDA 12.1..." -ForegroundColor Green
    uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
} else {
    Write-Host ""
    Write-Host "No NVIDIA GPU detected. Installing PyTorch CPU version..." -ForegroundColor Yellow
    uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
}

# Install optional vocal analysis
Write-Host ""
Write-Host "Installing vocal analysis (parselmouth)..." -ForegroundColor Green
uv pip install -e ".[vocal]"

Write-Host ""
Write-Host "=== Setup Complete ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "To activate the environment:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "To start the server:"
Write-Host "  uvicorn backend.app.main:app --reload"
Write-Host ""
Write-Host "To install watermark detection (optional):"
Write-Host "  uv pip install audioseal"
Write-Host ""

# Verify CUDA
if ($hasCuda) {
    python -c @"
import torch
if torch.cuda.is_available():
    print(f'CUDA acceleration: AVAILABLE ({torch.cuda.get_device_name(0)})')
else:
    print('CUDA: Installed but not available (check drivers)')
"@
}
