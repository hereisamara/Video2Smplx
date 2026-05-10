#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${1:-video2smplx_shared310}"

echo "=============================================="
echo " Video2SMPLX Shared Py3.10 Environment Setup"
echo "=============================================="
echo "Project directory: $SCRIPT_DIR"
echo "Conda environment: $ENV_NAME"
echo

if conda env list | grep -q "^${ENV_NAME} \\|^${ENV_NAME}$"; then
    echo "Environment '$ENV_NAME' already exists, reusing it."
else
    conda create -n "$ENV_NAME" python=3.10 -y
fi

CONDA_BASE="$(conda info --base)"
PYTHON_BIN="${CONDA_BASE}/envs/${ENV_NAME}/bin/python"
PIP_BIN="${CONDA_BASE}/envs/${ENV_NAME}/bin/pip"

echo
echo "[1/5] Installing shared PyTorch stack..."
"$PIP_BIN" install \
    torch==2.0.1 \
    torchvision==0.15.2 \
    torchaudio==2.0.2

echo
echo "[2/5] Installing shared Python requirements..."
"$PIP_BIN" install -r "$SCRIPT_DIR/requirements310.txt"

echo
echo "[3/5] Installing PyTorch3D..."
if ! "$PIP_BIN" install pytorch3d; then
    echo "PyTorch3D wheel not available; falling back to source install."
    "$PIP_BIN" install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
fi

echo
echo "[4/5] Installing EMOCA package in editable mode..."
"$PIP_BIN" install -e "$SCRIPT_DIR"

echo
echo "[5/5] Running a quick import smoke test..."
"$PYTHON_BIN" "$SCRIPT_DIR/test_shared_env.py"

echo
echo "Environment ready."
echo "Activate with: conda activate $ENV_NAME"

