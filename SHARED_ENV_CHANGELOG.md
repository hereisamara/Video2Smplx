# Shared Environment Changelog

This document records progressive fixes for running the combined `Video2Smplx` pipeline in the shared `video2smplx_shared310` environment.

Format for future entries:

- Date
- Scope
- Problem observed
- Root cause
- Code changes
- Manual server action
- Verification
- Notes

## 2026-05-25 - Headless pyrender EGL backend

### Scope

- Repository root: `Video2Smplx`
- File: `zero_filter_render.py`
- Stage: final render
- Runtime environment: existing `video2smplx_shared310`

### Problem observed

The full combined pipeline completed frame extraction, SMPLest-X, WiLoR, EMOCA, fusion, zeroing translation, smoothing, and saving processed params. It failed only at final rendering:

```text
pyglet.display.xlib.NoSuchDisplayException: Cannot connect to "None"
```

The failing line was:

```text
renderer = pyrender.OffscreenRenderer(...)
```

### Root cause

`pyrender` defaulted to the pyglet/X11 backend. On the server batch node there is no attached X display, so pyglet could not create a hidden window.

For headless server rendering, `pyrender` should use an offscreen OpenGL backend such as EGL.

### Code changes

- Set `PYOPENGL_PLATFORM=egl` by default at the top of `zero_filter_render.py`.
- The environment variable is set before `import pyrender`, which is required because PyOpenGL chooses the backend at import time.
- Used `os.environ.setdefault(...)` so a server job can still override the backend externally if needed.

### Manual server action

Pull or copy the updated repository code on the server:

```bash
cd /home/khtun/video2simplx/Video2SmplxPy10/Video2Smplx
git pull
```

Then resume from render only:

```bash
conda run --no-capture-output -n video2smplx_shared310 python pipeline.py \
  --video /lustrefs/disk/home/khtun/video2simplx/Video2Smplx/SMPLest-X-Inference/demo/P.mp4 \
  --output /home/khtun/video2simplx/Video2SmplxPy10/Video2Smplx/demo/output_combined_P \
  --name P \
  --smplestx_env video2smplx_shared310 \
  --wilor_env video2smplx_shared310 \
  --emoca_env video2smplx_shared310 \
  --render_env video2smplx_shared310 \
  --smplestx_ckpt smplest_x_h \
  --emoca_model EMOCA_v2_lr_mse_20 \
  --fps 24 \
  --viewport 512 \
  --smooth_window 15 \
  --smooth_poly 3 \
  --skip_extract \
  --skip_smplestx \
  --skip_wilor \
  --skip_emoca \
  --skip_fuse
```

If EGL is unavailable on the cluster, try overriding with OSMesa if installed:

```bash
PYOPENGL_PLATFORM=osmesa conda run --no-capture-output -n video2smplx_shared310 python zero_filter_render.py ...
```

### Verification

Local syntax verification:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/codex_pycache python3 -m py_compile zero_filter_render.py
```

### Notes

- The combined parameter output already exists at `demo/output_combined_P/fused_params`.
- The active failure is only the headless OpenGL backend for rendering.
- The warnings earlier in the log are not fatal.

## 2026-05-25 - WiLoR detector dill dependency

### Scope

- Repository root: `Video2Smplx`
- Files:
  - `EMOCA-Inference/requirements310.txt`
  - `EMOCA-Inference/test_shared_env.py`
- Stage: `stage_wilor(...)`
- Runtime environment: existing `video2smplx_shared310`

### Problem observed

WiLoR loaded its main checkpoint and MANO assets, then failed while loading the YOLO hand detector:

```text
ModuleNotFoundError: No module named 'dill'
```

The failing path was:

```text
detector = YOLO('./pretrained_models/detector.pt')
```

### Root cause

`WiLoR-Inference/pretrained_models/detector.pt` was serialized with a pickle object that imports `dill`.

The shared environment had `ultralytics` and `torch`, but not `dill`, so `torch.load(...)` failed while unpickling the detector checkpoint.

### Code changes

- Added `dill==0.3.8` to `EMOCA-Inference/requirements310.txt`.
- Added `dill` to `EMOCA-Inference/test_shared_env.py` so future shared-env smoke tests catch this dependency.

### Manual server action

Install `dill` into the existing shared environment:

```bash
conda run -n video2smplx_shared310 python -m pip install dill==0.3.8
```

Verify:

```bash
conda run -n video2smplx_shared310 python -c "import dill; print(dill.__version__)"
```

Then rerun the combined pipeline. Resume from WiLoR if SMPLest-X and frame extraction are already complete:

```bash
--skip_extract --skip_smplestx
```

### Verification

Local syntax verification:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/codex_pycache python3 -m py_compile EMOCA-Inference/test_shared_env.py
```

### Notes

- The Lightning checkpoint-upgrade message is still a warning, not the active failure.
- The active failure is the missing `dill` Python package while unpickling `detector.pt`.
- Use positional syntax if you choose to permanently upgrade the Lightning checkpoint:

```bash
conda run -n video2smplx_shared310 python -m pytorch_lightning.utilities.upgrade_checkpoint pretrained_models/wilor_final.ckpt
```

## 2026-05-25 - Combined pipeline WiLoR MANO asset preflight

### Scope

- Repository root: `Video2Smplx`
- File: `pipeline.py`
- Stage: `stage_wilor(...)`
- Runtime environment: existing `video2smplx_shared310`

### Problem observed

WiLoR loaded its config and checkpoint, then failed while creating the model backbone:

```text
FileNotFoundError: [Errno 2] No such file or directory: './mano_data/mano_mean_params.npz'
```

The log also showed this Lightning message:

```text
Lightning automatically upgraded your loaded checkpoint from v1.8.1 to v1.9.5.
```

Running the suggested command failed:

```text
python -m pytorch_lightning.utilities.upgrade_checkpoint --file pretrained_models/wilor_final.ckpt
upgrade_checkpoint.py: error: unrecognized arguments: --file
```

### Root cause

WiLoR needs local MANO assets in:

```text
WiLoR-Inference/mano_data/
```

The server run was missing:

```text
WiLoR-Inference/mano_data/mano_mean_params.npz
```

The Lightning checkpoint-upgrade message is only a warning. It is not the active failure.

The installed PyTorch Lightning `1.9.5` upgrade utility expects the checkpoint path as a positional argument, not `--file`.

### Code changes

- Extended the WiLoR preflight check in `stage_wilor(...)`.
- The pipeline now checks for:
  - `WiLoR-Inference/pretrained_models/wilor_final.ckpt`
  - `WiLoR-Inference/pretrained_models/detector.pt`
  - `WiLoR-Inference/pretrained_models/model_config.yaml`
  - `WiLoR-Inference/mano_data/mano_mean_params.npz`
  - `WiLoR-Inference/mano_data/MANO_RIGHT.pkl`

### Manual server action

Check WiLoR MANO files:

```bash
cd /home/khtun/video2simplx/Video2SmplxPy10/Video2Smplx/WiLoR-Inference
ls -lh mano_data/
ls -lh mano_data/mano_mean_params.npz
ls -lh mano_data/MANO_RIGHT.pkl
```

If missing, copy/download them into:

```text
WiLoR-Inference/mano_data/
```

To permanently upgrade the WiLoR checkpoint for the installed Lightning CLI, use the positional path syntax:

```bash
cd /home/khtun/video2simplx/Video2SmplxPy10/Video2Smplx/WiLoR-Inference
conda run -n video2smplx_shared310 python -m pytorch_lightning.utilities.upgrade_checkpoint pretrained_models/wilor_final.ckpt
```

This checkpoint upgrade is optional for this error. The required fix is adding the missing MANO files.

### Verification

Local syntax verification:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/codex_pycache python3 -m py_compile pipeline.py
```

### Notes

- The active failure is the missing `mano_mean_params.npz`.
- The Lightning checkpoint-upgrade message is not fatal.
- The OpenCV and `timm` warnings are not fatal.

## 2026-05-25 - Combined pipeline WiLoR pretrained asset preflight

### Scope

- Repository root: `Video2Smplx`
- File: `pipeline.py`
- Stage: `stage_wilor(...)`
- Runtime environment: existing `video2smplx_shared310`

### Problem observed

The combined pipeline reached WiLoR, then failed while loading the WiLoR config:

```text
FileNotFoundError: [Errno 2] No such file or directory: './pretrained_models/model_config.yaml'
```

### Root cause

WiLoR requires pretrained assets in:

```text
WiLoR-Inference/pretrained_models/
```

At minimum, the combined pipeline expects:

- `wilor_final.ckpt`
- `detector.pt`
- `model_config.yaml`
- `mano_mean_params.npz`
- `MANO_RIGHT.pkl`

The server run was missing `model_config.yaml`. This is a required model config file, not a Python package issue.

### Code changes

- Added a WiLoR preflight check in `stage_wilor(...)`.
- The pipeline now checks for the required WiLoR files before launching `demo_params_unified.py`.
- If any are missing, it raises one clear `FileNotFoundError` listing every missing path.

### Manual server action

Check the files on the server:

```bash
cd /home/khtun/video2simplx/Video2SmplxPy10/Video2Smplx/WiLoR-Inference
ls -lh pretrained_models/
```

Make sure these files exist:

```bash
ls -lh pretrained_models/wilor_final.ckpt
ls -lh pretrained_models/detector.pt
ls -lh pretrained_models/model_config.yaml
```

If `model_config.yaml` is missing, copy it from the WiLoR release/repository into:

```text
WiLoR-Inference/pretrained_models/model_config.yaml
```

Then rerun the combined pipeline.

### Verification

Local syntax verification:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/codex_pycache python3 -m py_compile pipeline.py
```

### Notes

- The OpenCV `imwrite_ Unsupported depth image` warning is not the active failure.
- The `timm.models.layers` FutureWarning is not the active failure.
- The active failure is the missing WiLoR pretrained config file.

## 2026-05-25 - Combined pipeline WiLoR argument fix

### Scope

- Repository root: `Video2Smplx`
- Files:
  - `pipeline.py`
  - `WiLoR-Inference/README.md`
- Stage: `stage_wilor(...)`
- Runtime environment: existing `video2smplx_shared310`

### Problem observed

The combined pipeline finished SMPLest-X and then failed at WiLoR:

```text
demo_params_unified.py: error: unrecognized arguments: --save_params
```

The failed command was:

```text
python WiLoR-Inference/demo_params_unified.py --img_folder ... --out_folder ... --save_params
```

### Root cause

The local `WiLoR-Inference/demo_params_unified.py` script does not define a `--save_params` option.

Its accepted arguments are:

- `--img_folder`
- `--out_folder`
- `--rescale_factor`
- `--file_type`

It saves parameter `.pkl` files automatically, so `--save_params` is not needed.

The pipeline also expected WiLoR output in:

```text
demo/result_params_unified/params/
```

but previously passed:

```text
demo/result_params_unified/
```

as `--out_folder`.

### Code changes

- Removed `--save_params` from the WiLoR subprocess arguments in `stage_wilor(...)`.
- Changed the WiLoR output folder passed to `demo_params_unified.py` from:

```text
demo/result_params_unified/
```

to:

```text
demo/result_params_unified/params/
```

- Kept the fusion stage unchanged because it already reads from `demo/result_params_unified/params/`.
- Updated `WiLoR-Inference/README.md` so future manual runs do not include the unsupported `--save_params` flag.

### Manual server action

Pull or copy the updated repository code on the server:

```bash
cd /home/khtun/video2simplx/Video2SmplxPy10/Video2Smplx
git pull
```

Then rerun the combined pipeline command.

### Verification

Local syntax verification:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/codex_pycache python3 -m py_compile pipeline.py
```

### Notes

- The OpenCV warning about `imwrite_ Unsupported depth image` is not the active failure in this log.
- The `timm.models.layers` FutureWarning is not the active failure.
- The active failure was only the unsupported WiLoR CLI argument.
