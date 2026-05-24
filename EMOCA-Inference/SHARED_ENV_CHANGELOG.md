# Shared Environment Changelog

This document records progressive fixes for running `EMOCA-Inference` and `SMPLest-X-Inference` inside one Python 3.10 server environment.

Format for future entries:

- Date
- Scope
- Problem observed
- Root cause
- Code changes
- Manual server action
- Verification
- Notes

## 2026-05-24 - EMOCA audio attachment ffmpeg resolver fix

### Scope

- Repository: `EMOCA-Inference`
- File: `gdl/datasets/FaceVideoDataModule.py`
- Function: `attach_audio_to_reconstruction_video(...)`
- Runtime environment: existing `video2smplx_shared310`

### Problem observed

EMOCA finished reconstruction-video frame writing, then failed while attaching the original audio:

```text
FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'
```

The failure happened after the decimal-FPS crash was fixed, at:

```text
attach_audio_to_reconstruction_video(...)
subprocess.run(...)
```

### Root cause

Most `ffmpeg` calls had already been routed through `_ffmpeg_exe()`, which resolves the executable from PATH and then falls back to `imageio-ffmpeg`.

`attach_audio_to_reconstruction_video(...)` still used a literal command name:

```python
"ffmpeg"
```

On the server batch job, `ffmpeg` was not visible on PATH inside the launched process, so `subprocess.Popen` failed before the command could run.

### Code changes

- Replaced the final hardcoded `"ffmpeg"` subprocess call with `_ffmpeg_exe()`.
- This makes audio attachment use the same executable-resolution path as frame extraction and audio extraction.

### Manual server action

Pull or copy the updated repository code on the server:

```bash
cd /lustrefs/disk/home/khtun/video2simplx/Video2SmplxPy10/Video2Smplx/EMOCA-Inference
git pull
```

Then rerun the same EMOCA command in the existing environment.

If it still reports that `ffmpeg` cannot be found, install or expose `ffmpeg` manually:

```bash
conda install -n video2smplx_shared310 -c conda-forge ffmpeg
```

If the cluster uses modules instead of conda packages, load the module that provides both `ffmpeg` and `ffprobe` before submitting the job.

### Verification

After pulling the code, verify the environment path:

```bash
conda run -n video2smplx_shared310 ffmpeg -version
conda run -n video2smplx_shared310 ffprobe -version
```

Then rerun EMOCA. The previous log already shows the FPS issue is fixed because it reached `attach_audio_to_reconstruction_video(...)`.

### Notes

- This fix is code-only if `imageio-ffmpeg` is installed and usable.
- A manual `ffmpeg` install is still the most reliable server setup because `ffprobe` is also required for metadata.
- The warnings about PyTorch Lightning checkpoint migration, torchvision `pretrained`, FLAME tensor construction, missing `template.mtl`, and `grid_sample` are warnings, not the active failure in this log.

## 2026-05-24 - EMOCA decimal FPS video writer fix

### Scope

- Repository: `EMOCA-Inference`
- File: `gdl/datasets/FaceVideoDataModule.py`
- Runtime environment: existing `video2smplx_shared310`

### Problem observed

EMOCA completed frame processing but failed while creating the reconstruction video:

```text
ValueError: invalid literal for int() with base 10: '24.0'
```

The failure happened in `create_reconstruction_video(...)` when opening `cv2.VideoWriter`.

### Root cause

The video metadata FPS value can appear in more than one format depending on how metadata was read:

- Fractional string from `ffprobe`, for example `24/1` or `30000/1001`
- Decimal string from OpenCV fallback or other metadata paths, for example `24.0`

The old reconstruction writer code assumed every FPS value contained `/` and tried to parse the numerator and denominator with `int(...)`. That fails for `24.0`.

### Code changes

- Added `_parse_video_fps(...)`.
- Routed all OpenCV `VideoWriter` FPS values through `_parse_video_fps(...)`.
- Supported both server-observed FPS formats:
  - `24.0` -> `24.0`
  - `30000/1001` -> `29.97002997002997`
- Removed remaining direct `int(self.video_metas[sequence_id]['fps'].split('/')[0])` writer parsing.

### Manual server action

No library update is required for this fix.

Pull or copy the updated repository code on the server:

```bash
cd /lustrefs/disk/home/khtun/video2simplx/Video2SmplxPy10/Video2Smplx/EMOCA-Inference
git pull
```

Then rerun the same EMOCA command in the existing environment.

### Verification

Local syntax verification passed:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/codex_pycache python3 -m py_compile gdl/datasets/FaceVideoDataModule.py
```

The first `py_compile` attempt without `PYTHONPYCACHEPREFIX` failed only because Apple Python tried to write bytecode under `~/Library/Caches`, which is outside the local sandbox.

### Notes

If the server fails earlier around metadata, audio extraction, or video probing, verify `ffmpeg` and `ffprobe` are available inside the conda environment:

```bash
conda run -n video2smplx_shared310 ffmpeg -version
conda run -n video2smplx_shared310 ffprobe -version
```

If either command is missing, install `ffmpeg` into the conda environment or load the cluster module that provides both executables.

## 2026-05-24 - Shared Python 3.10 environment baseline

### Scope

- Repositories:
  - `EMOCA-Inference`
  - `SMPLest-X-Inference`
- Goal: run both projects in one Python 3.10 environment.

### Problem observed

The original projects had dependency assumptions from different Python and Torch stacks. Running both in a single server environment exposed compatibility issues around NumPy, PyTorch, MediaPipe, `chumpy`, `ffmpeg`, and video metadata probing.

### Root cause

- EMOCA depends on older code paths and legacy packages.
- `SMPLest-X` requires a newer runtime stack.
- NumPy 2.x breaks older compiled or legacy assumptions.
- Python 3.10 removed APIs expected by `chumpy`.
- Some servers have a Python package named `ffmpeg` that is not the expected `ffmpeg-python` package.
- Batch jobs may not expose `ffmpeg` and `ffprobe` unless they are installed in conda or provided by a loaded cluster module.

### Code and file changes

#### `requirements310.txt`

- Converted the file from loose modernization ranges to an explicit shared-stack manifest.
- Added the `SMPLest-X` runtime packages so this file can act as the union requirement set.
- Pinned `numpy` below 2 to avoid the ABI error already observed in EMOCA.

#### `environment_shared_py310.yml`

- Added a conda environment file that creates the shared Python 3.10 environment and installs the union requirements.

#### `setup_env_shared_py310.sh`

- Added a repeatable setup script that installs the shared Torch stack first, then the union requirements, then `pytorch3d`, then runs a smoke test.
- Updated the script for GPU servers with strict quotas:
  - Uses `conda run --no-capture-output` so pip progress is visible.
  - Installs `numpy==1.24.4` before Torch so pip does not pull NumPy 2.x during Torch installation.
  - Uses `pip install --no-cache-dir` to reduce cache/quota pressure.
  - Supports `VIDEO2SMPLX_ENV_PREFIX=/path/to/env` for installing the conda env on scratch or project storage.
  - Installs the `ffmpeg` conda package so both `ffmpeg` and `ffprobe` executables are available inside batch jobs.
  - Patches installed `chumpy` so direct `import chumpy` works in Python 3.10.

#### `test_shared_env.py`

- Added a quick validation script that checks both dependency imports and the EMOCA Swin fallback path.
- Verifies key `SMPLest-X` imports.
- Checks for `ffmpeg` and `ffprobe` because EMOCA calls both through subprocess.

#### `gdl/datasets/FaceVideoDataModule.py`

- Replaced `ffmpeg.probe(...)` from the Python `ffmpeg` module with a direct `ffprobe` subprocess call that parses JSON output.
- This avoids failures when a server has the wrong Python package named `ffmpeg` installed while still using the real `ffprobe` executable.
- Added executable resolution helpers for `ffmpeg` and `ffprobe`; they first use PATH and then try `imageio-ffmpeg` as a fallback.
- Added an OpenCV metadata fallback so EMOCA can continue video processing when `ffprobe` is unavailable but OpenCV can read the video stream.
- Fixed invalid-video cleanup for `TestFaceVideoDM` by keeping `annotation_list` aligned with `video_list` and guarding deletes when optional lists are shorter.
- Added `ffprobe` stderr output when metadata probing fails, so corrupt, missing, or unreadable videos report the actual reason.
- Included all invalid-video probe reasons in the final `RuntimeError`, so batch job logs show the exact video path and `ffprobe` error instead of only a generic metadata-scan failure.

#### `gdl/models/DecaFLAME.py`

- Added a compatibility shim before FLAME pickle loading so legacy `chumpy` imports work with NumPy 1.24 and Python 3.10.
- Restores the removed NumPy aliases expected by `chumpy`.
- Provides `inspect.getargspec` via `inspect.getfullargspec`.

### Dependency baseline

Shared environment standardizes on:

- `python==3.10`
- `numpy==1.24.4`
- `torch==2.0.1`
- `torchvision==0.15.2`
- `torchaudio==2.0.2`
- `pytorch-lightning==1.9.5`
- `torchmetrics==0.11.4`
- `timm==1.0.14`
- `opencv-python==4.11.0.86`
- `mediapipe==0.10.14`
- `protobuf>=4.25.3,<5`
- `trimesh==4.6.2`
- `smplx==0.1.28`
- `pyrender==0.1.45`
- `json_tricks==3.17.3`
- `einops==0.8.1`
- `ultralytics==8.3.75`
- `tqdm==4.67.1`

Additional EMOCA runtime packages included in the shared environment:

- `omegaconf==2.3.0`
- `hydra-core==1.3.2`
- `face-alignment==1.3.5`
- `kornia==0.6.12`
- `facenet-pytorch==2.5.2`
- `wandb==0.15.12`

### Manual server action

For a fresh shared environment, use:

```bash
cd /path/to/EMOCA-Inference
bash setup_env_shared_py310.sh
```

For strict server quotas, put the environment on scratch or project storage:

```bash
VIDEO2SMPLX_ENV_PREFIX=/path/to/scratch/video2smplx_shared310 bash setup_env_shared_py310.sh
```

### Verification

Run:

```bash
conda run -n video2smplx_shared310 python test_shared_env.py
conda run -n video2smplx_shared310 ffmpeg -version
conda run -n video2smplx_shared310 ffprobe -version
```

### Notes

- `pytorch3d` is still the strictest dependency in the merged stack. It must match the chosen Torch and CUDA build, so the setup script installs it after Torch rather than pinning a wheel URL that only works for one platform.
- `protobuf` is intentionally aligned to MediaPipe `0.10.14`. The older EMOCA-era `protobuf==3.20.3` pin conflicts with this MediaPipe release.
- `ffprobe` is required for EMOCA video metadata. Prefer installing conda-forge `ffmpeg` or loading the cluster's ffmpeg module. `imageio-ffmpeg` is only a fallback and may not provide `ffprobe` on every platform.
- This changelog describes the dependency merge and follow-up server fixes. It does not claim that every downstream runtime path in both projects has been executed in this workspace.
