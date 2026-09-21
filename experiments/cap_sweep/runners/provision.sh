#!/usr/bin/env bash
set -euo pipefail

echo "========================================="
echo "   Provisioning Training Node (A100)     "
echo "========================================="

# 1. System Dependencies (Ubuntu/Debian)
# We assume standard GPU AMIs (Lambda/RunPod) have CUDA/Drivers pre-installed.
echo "[1/4] Installing System Tools..."
sudo apt-get update -qq
sudo apt-get install -y -qq git curl tmux htop build-essential

# 2. Install 'uv' (Python Package Manager)
if ! command -v uv &> /dev/null; then
    echo "[2/4] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Source env for immediate use in this script
    export PATH="$HOME/.cargo/bin:$PATH"
else
    echo "[2/4] uv already installed."
fi

# 3. Project Setup
echo "[3/4] Installing Python Dependencies..."
if [ ! -f "pyproject.toml" ]; then
    echo "CRITICAL ERROR: pyproject.toml not found!"
    echo "Please run this script from the repository root."
    exit 1
fi

# Sync environment with training extras (Axolotl, FlashAttn, etc.)
# This creates the .venv automatically
uv sync --extra train

# 4. Workspace Prep
echo "[4/4] Creating Directory Structure..."
mkdir -p data/tasks
mkdir -p data/traces
mkdir -p data/training
mkdir -p out
mkdir -p cache

# 5. Validation
echo "========================================="
echo "   Provisioning Complete!                "
echo "========================================="
echo "Python: $(uv run which python)"
echo "NVIDIA Driver:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo ""
echo "NEXT STEPS:"
echo "1. Run generation:  make traces-persistent"
echo "2. Run training:    make train-persistent"