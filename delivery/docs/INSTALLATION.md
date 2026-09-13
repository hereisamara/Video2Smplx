# Installation Manual

The recommended runtime is the existing server environment:

```bash
module load Mamba/23.11.0-0
conda activate video2smplx_shared310
cd /home/khtun/video2simplx/Video2SmplxPy10/Video2Smplx
```

## Python Environment

The code expects Python 3.10 with CUDA-enabled PyTorch and the dependencies used
by SMPLest-X, WiLoR, EMOCA, SMPL-X rendering, and OpenCV.

Minimum runtime components:

```text
torch
torchvision
opencv-python
numpy
scipy
tqdm
trimesh
pyrender
smplx
ultralytics
```

Use the existing `video2smplx_shared310` environment on the cluster when possible.
It already matches the current pipeline scripts and model wrappers.

## Repository Layout

Expected project root:

```text
/home/khtun/video2simplx/Video2SmplxPy10/Video2Smplx
```

Expected subprojects:

```text
SMPLest-X-Inference/
WiLoR-Inference/
EMOCA-Inference/
video2smplx/
zero_filter_render.py
delivery/
```

## GPU Requirement

SMPLest-X requires CUDA in this wrapper because the upstream tester path calls
CUDA internally. Rendering also works best with GPU/EGL on the server.

For headless rendering:

```bash
export PYOPENGL_PLATFORM=egl
```

## Quick Smoke Test

Run help commands first:

```bash
python delivery/source/body_estimation_cli.py --help
python delivery/source/hand_estimation_cli.py --help
python delivery/source/face_estimation_cli.py --help
python delivery/source/combine_smooth_render_cli.py --help
```

If `python` is not available on the server shell, use:

```bash
conda run --no-capture-output -n video2smplx_shared310 python delivery/source/body_estimation_cli.py --help
```
