# Required Model Setup And Download Instructions

Model files are not redistributed in this delivery unless the license permits it.
Place all model files in the existing project subdirectories below.

## SMPLest-X

Expected directory:

```text
SMPLest-X-Inference/
  pretrained_models/
    smplest_x_h/
      config_base.py
      smplest_x_h.pth.tar
    yolov8x.pt
  human_models/
    human_model_files/
      smplx/
        SMPLX_NEUTRAL.npz
        SMPLX_MALE.npz
        SMPLX_FEMALE.npz
        SMPLX_to_J14.pkl
        MANO_SMPLX_vertex_ids.pkl
        SMPL-X__FLAME_vertex_ids.npy
```

Download SMPLest-X from the official project/repository and follow its license.
Download SMPL-X model files from the official SMPL-X website after accepting the
SMPL-X license.

## WiLoR

Expected directory:

```text
WiLoR-Inference/
  pretrained_models/
    wilor_final.ckpt
    detector.pt
    model_config.yaml
  mano_data/
    mano_mean_params.npz
    MANO_RIGHT.pkl
```

Download WiLoR weights from the official WiLoR repository or model page. Download
MANO files from the official MANO website after accepting the MANO license.

## EMOCA

Expected directory:

```text
EMOCA-Inference/
  assets/
    EMOCA/models/EMOCA_v2_lr_mse_20/
    DECA/data/deca_model.tar
    FLAME/geometry/generic_model.pkl
    FLAME/geometry/landmark_embedding.npy
    FLAME/geometry/mediapipe_landmark_embedding.npz
    FLAME/geometry/head_template.obj
    FLAME/geometry/fixed_uv_displacements/fixed_displacement_256.npy
    FLAME/mask/uv_face_mask.png
    FLAME/mask/uv_face_eye_mask.png
```

Download EMOCA assets from the official EMOCA instructions. Download FLAME assets
from the official FLAME website after accepting the FLAME license.

EMOCA's FAN face detector also uses cached `2DFAN` landmark and `s3fd` detector
weights from the PyTorch hub checkpoint directory. The separate model packager
copies that cache into:

```text
runtime_model_cache/torch/hub/checkpoints/
```

The EMOCA runner uses this bundled cache automatically when present.

## Post-Processing Corrector Models

The research correctors trained for SignLanguage are optional post-processors.
They are not required for the four base CLI programs, but they are used for the
best reported result.

Portable delivery locations:

```text
delivery/models/correctors/global/best_model.pt
delivery/models/correctors/hand/best_model.pt
delivery/models/correctors/upper2d/best_model.pt
```

Run `delivery/scripts/install_server_delivery_assets.sh` from the repository
root on the training server to copy the verified checkpoints and final sample
outputs into these portable delivery locations. The script also writes SHA-256
checksums. The corrector provenance and original server locations are documented
in `delivery/models/correctors/README.md`.

The licensed SMPL-X model is not copied into `delivery/`. Its runtime location
is:

```text
SMPLest-X-Inference/human_models/human_model_files/smplx/SMPLX_FEMALE.npz
```

Use the matching programs under `tools/postprocessing/` when reproducing the
corrected benchmark pipeline.

## Separate Physical Model Package

When the recipient is authorized to receive every third-party model, package
the complete runtime model set separately from the source/document archive:

```bash
ACKNOWLEDGE_RESTRICTED_MODEL_LICENSES=1 \
MODEL_SOURCE_ROOT=/path/to/existing/Video2Smplx \
YOLO_POSE_MODEL=/absolute/path/to/yolov8x-pose.pt \
bash delivery/scripts/package_runtime_models.sh /project/path/to/output
```

See `delivery/docs/MODEL_BUNDLE.md` for archive contents, extraction,
verification, splitting for FAT32 media, and licensing restrictions.
