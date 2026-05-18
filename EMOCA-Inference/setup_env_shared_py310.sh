#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${1:-video2smplx_shared310}"
ENV_PREFIX="${VIDEO2SMPLX_ENV_PREFIX:-}"
PIP_INSTALL_ARGS=(install --no-cache-dir --progress-bar on)
CONDA_INSTALL_ARGS=(install -y -c conda-forge)

echo "=============================================="
echo " Video2SMPLX Shared Py3.10 Environment Setup"
echo "=============================================="

echo "Project directory: $SCRIPT_DIR"

if [[ -n "$ENV_PREFIX" ]]; then
    echo "Conda environment prefix: $ENV_PREFIX"
else
    echo "Conda environment: $ENV_NAME"
fi
echo

if [[ -n "$ENV_PREFIX" ]]; then
    if [[ -d "$ENV_PREFIX" ]]; then
        echo "Environment prefix '$ENV_PREFIX' already exists, reusing it."
    else
        conda create -p "$ENV_PREFIX" python=3.10 pip -y
    fi
elif conda env list | grep -q "^${ENV_NAME} \\|^${ENV_NAME}$"; then
    echo "Environment '$ENV_NAME' already exists, reusing it."
else
    conda create -n "$ENV_NAME" python=3.10 pip -y
fi

run_in_env() {
    if [[ -n "$ENV_PREFIX" ]]; then
        conda run --no-capture-output -p "$ENV_PREFIX" "$@"
    else
        conda run --no-capture-output -n "$ENV_NAME" "$@"
    fi
}

pip_in_env() {
    run_in_env python -m pip "$@"
}

echo
echo "[0/7] Checking Python and pip inside the environment..."
run_in_env python --version 
pip_in_env --version 

echo
echo "[1/7] Installing ffmpeg and ffprobe executables..."
if [[ -n "$ENV_PREFIX" ]]; then
    conda "${CONDA_INSTALL_ARGS[@]}" -p "$ENV_PREFIX" ffmpeg
else
    conda "${CONDA_INSTALL_ARGS[@]}" -n "$ENV_NAME" ffmpeg
fi
run_in_env ffmpeg -version
run_in_env ffprobe -version

echo
echo "[2/7] Installing critical pins before Torch..." 
pip_in_env "${PIP_INSTALL_ARGS[@]}" \
    "numpy==1.24.4" \
    "setuptools<70.0.0" \
    "wheel"

echo
echo "[3/7] Installing shared PyTorch stack..." 
pip_in_env "${PIP_INSTALL_ARGS[@]}" \
    torch==2.0.1 \
    torchvision==0.15.2 \
    torchaudio==2.0.2



echo
echo "[4/7] Installing shared Python requirements..."
pip_in_env "${PIP_INSTALL_ARGS[@]}" -r "$SCRIPT_DIR/requirements310.txt"
pip_in_env "${PIP_INSTALL_ARGS[@]}" chumpy==0.70 --no-build-isolation
pip_in_env "${PIP_INSTALL_ARGS[@]}" pandas imgaug scikit-learn scikit-video torchfile

echo
echo "[5/7] Installing PyTorch3D..."
if ! pip_in_env "${PIP_INSTALL_ARGS[@]}" pytorch3d; then
    echo "PyTorch3D wheel not available; falling back to source install."
    # pip_in_env "${PIP_INSTALL_ARGS[@]}" "git+https://github.com/facebookresearch/pytorch3d.git@stable"
    pip_in_env "${PIP_INSTALL_ARGS[@]}" pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt201/download.html
fi

echo
echo "[6/7] Installing EMOCA package in editable mode..."
pip_in_env install --no-cache-dir -e "$SCRIPT_DIR"

echo
echo "[7/7] Running a quick import smoke test..."
run_in_env python "$SCRIPT_DIR/test_shared_env.py"

echo
echo "Environment ready."
if [[ -n "$ENV_PREFIX" ]]; then
    echo "Activate with: conda activate $ENV_PREFIX"
else
    echo "Activate with: conda activate $ENV_NAME"
fi
