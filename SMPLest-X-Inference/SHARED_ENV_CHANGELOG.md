# Shared Environment Changelog

This project has been aligned to the shared Python 3.10 environment used with `EMOCA-Inference`.

## Files changed

- `requirements.txt`
  - Added the explicit shared Torch stack: `torch==2.0.1`, `torchvision==0.15.2`, `torchaudio==2.0.2`
  - Moved `numpy` from `1.23.1` to `1.24.4` to match the merged environment while still staying below NumPy 2.x
  - Kept the existing `SMPLest-X` inference-specific packages:
    - `smplx==0.1.28`
    - `ultralytics==8.3.75`
    - `pyrender==0.1.45`
    - `json_tricks==3.17.3`
    - `einops==0.8.1`
    - `timm==1.0.14`
  - Added the EMOCA-side runtime packages that are now part of the shared environment, including `pytorch-lightning`, `omegaconf`, `hydra-core`, `mediapipe`, `face-alignment`, `facenet-pytorch`, and related utility packages.

## Library updates

- `numpy: 1.23.1 -> 1.24.4`
- added `scipy==1.10.1`
- added `torch==2.0.1`
- added `torchvision==0.15.2`
- added `torchaudio==2.0.2`
- added `protobuf>=4.25.3,<5`
- added `imageio-ffmpeg==0.6.0`
- kept `opencv-python==4.11.0.86`
- kept `smplx==0.1.28`
- kept `trimesh==4.6.2`
- kept `pyrender==0.1.45`
- kept `timm==1.0.14`
- kept `ultralytics==8.3.75`

## Notes

- The shared environment also includes EMOCA dependencies that `SMPLest-X` does not import directly. They are present so the same environment can run both repositories without swapping envs.
- `pytorch3d` is managed from the EMOCA side because it is the package most sensitive to the Torch/CUDA build.
- `protobuf` follows MediaPipe `0.10.14`'s requirement. The old `protobuf==3.20.3` pin is not compatible with this shared environment.
