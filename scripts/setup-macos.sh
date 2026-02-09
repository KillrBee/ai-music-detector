#!/bin/bash
# Setup script for macOS (Apple Silicon / Intel)
# Uses uv package manager with MPS acceleration on ARM Macs

set -e

echo "=== AI Audio Detector - macOS Setup ==="

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "Please restart your shell and run this script again."
    exit 1
fi

echo "Using uv: $(uv --version)"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
uv venv --python 3.12

# Activate venv
source .venv/bin/activate

# Install core dependencies
echo ""
echo "Installing core dependencies..."
uv pip install -e .

# Install PyTorch (CPU wheels work for both Intel and Apple Silicon)
# MPS acceleration is automatic on Apple Silicon
echo ""
echo "Installing PyTorch (with MPS support on Apple Silicon)..."
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install optional vocal analysis
echo ""
echo "Installing vocal analysis (parselmouth)..."
uv pip install -e ".[vocal]"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To start the server:"
echo "  uvicorn backend.app.main:app --reload"
echo ""
echo "To install watermark detection (optional):"
echo "  uv pip install audioseal"
echo ""

# Check for Apple Silicon MPS support
python3 -c "
import platform
if platform.processor() == 'arm':
    try:
        import torch
        if torch.backends.mps.is_available():
            print('MPS (Metal Performance Shaders) acceleration: AVAILABLE')
        else:
            print('MPS acceleration: Not available (may need macOS 12.3+)')
    except Exception:
        print('MPS acceleration: Could not check (torch may need update)')
else:
    print('Running on Intel Mac - CPU mode')
" 2>/dev/null || true
