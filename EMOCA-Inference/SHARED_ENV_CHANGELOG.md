# Shared Environment Changelog

This document records the files and dependency changes needed to run `EMOCA-Inference` and `SMPLest-X-Inference` inside one Python 3.10 environment.

## Files changed in `EMOCA-Inference`

- `requirements310.txt`
  - Converted the file from loose modernization ranges to an explicit shared-stack manifest.
  - Added the `SMPLest-X` runtime packages so this file can act as the union requirement set.
  - Pinned `numpy` below 2 to avoid the ABI error already observed in EMOCA.
- `environment_shared_py310.yml`
  - Added a conda environment file that creates the shared Python 3.10 environment and installs the union requirements.
- `setup_env_shared_py310.sh`
  - Added a repeatable setup script that installs the shared Torch stack first, then the union requirements, then `pytorch3d`, then runs a smoke test.
- `test_shared_env.py`
  - Added a quick validation script that checks both dependency imports and the EMOCA Swin fallback path, then verifies the key `SMPLest-X` imports.

## Library updates for `EMOCA-Inference`

Compared with the older EMOCA setup script and the original modernization draft, the shared environment standardizes on:

- `python` -> `3.10`
- `numpy` -> `1.24.4`
- `torch` -> `2.0.1`
- `torchvision` -> `0.15.2`
- `torchaudio` -> `2.0.2`
- `pytorch-lightning` -> `1.9.5`
- `torchmetrics` -> `0.11.4`
- `timm` -> `1.0.14`
- `opencv-python` -> `4.11.0.86`
- `mediapipe` -> `0.10.14`
- `trimesh` -> `4.6.2`

Additional shared-environment packages that are now present in the EMOCA stack because `SMPLest-X` requires them:

- `smplx==0.1.28`
- `pyrender==0.1.45`
- `pyopengl`
- `json_tricks==3.17.3`
- `einops==0.8.1`
- `ultralytics==8.3.75`
- `tqdm==4.67.1`

## Applied library updates for `SMPLest-X-Inference`

`SMPLest-X-Inference/requirements.txt` was updated to align that repo with the shared environment. The main changes were:

- `numpy: 1.23.1 -> 1.24.4`
- add `torch==2.0.1`
- add `torchvision==0.15.2`
- add `torchaudio==2.0.2`
- keep `smplx==0.1.28`
- keep `ultralytics==8.3.75`
- keep `pyrender==0.1.45`
- keep `json_tricks==3.17.3`
- keep `einops==0.8.1`
- keep `timm==1.0.14`
- align `opencv-python` to `4.11.0.86`
- align `trimesh` to `4.6.2`

Additional packages that become available in the shared environment because EMOCA needs them:

- `pytorch-lightning==1.9.5`
- `torchmetrics==0.11.4`
- `omegaconf==2.3.0`
- `hydra-core==1.3.2`
- `face-alignment==1.3.5`
- `kornia==0.6.12`
- `facenet-pytorch==2.5.2`
- `mediapipe==0.10.14`
- `wandb==0.15.12`

## Notes

- `pytorch3d` is still the strictest dependency in the merged stack. It must match the chosen Torch/CUDA build, so the setup script installs it after Torch rather than pinning a wheel URL that only works for one platform.
- This changelog describes the dependency merge. It does not claim that every downstream runtime path in both projects has been executed in this workspace.
