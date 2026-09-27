# Separate Runtime Model Bundle

The source/document archive does not embed large model weights. Create a
separate runtime-model archive on the authorized server with:

```bash
ACKNOWLEDGE_RESTRICTED_MODEL_LICENSES=1 \
YOLO_POSE_MODEL=/project/lt200246-mmacma/khtun/signlanguage_2d_guided_upperbody_outputs/models/yolov8x-pose.pt \
COMPRESSION=none \
bash delivery/scripts/package_runtime_models.sh \
  /project/lt200246-mmacma/khtun/video2smplx_delivery_release
```

The archive contains:

- SMPLest-X-H checkpoint and its YOLO person detector;
- SMPL-X neutral, male, female, and mapping assets;
- WiLoR checkpoint, hand detector, configuration, and MANO assets;
- EMOCA, DECA, FLAME, and face-recognition assets;
- cached FAN landmark and SFD face-detector weights used by EMOCA;
- YOLOv8x-pose for the 2D-guided corrector;
- the global, hand, and 2D upper-body learned correctors;
- a SHA-256 checksum file and model setup documentation.

The bundle preserves repository-relative paths. Extract its top-level contents
over a Video2Smplx source checkout, then verify:

```bash
tar -xf Video2Smplx_runtime_models_<STAMP>.tar
cp -a Video2Smplx_runtime_models_<STAMP>/. /path/to/Video2Smplx/
cd /path/to/Video2Smplx
bash delivery/scripts/verify_runtime_models.sh
```

The EMOCA runner automatically sets `TORCH_HOME` to
`runtime_model_cache/torch` when that bundled directory is present, preventing
the FAN/SFD dependency from downloading weights during an offline run.

Model archives are already compressed internally, so `COMPRESSION=none` is
recommended. Use `COMPRESSION=gzip` only when archive size is more important
than packaging time.

For FAT32 media, split the archive into pieces smaller than 4 GB:

```bash
ACKNOWLEDGE_RESTRICTED_MODEL_LICENSES=1 \
SPLIT_SIZE=3900M \
bash delivery/scripts/package_runtime_models.sh /project/path/to/output
```

Reassemble before extraction:

```bash
cat Video2Smplx_runtime_models_<STAMP>.tar.part-* > Video2Smplx_runtime_models_<STAMP>.tar
```

## Licensing

SMPL-X, MANO, FLAME, DECA, EMOCA, SMPLest-X, WiLoR, Ultralytics, and associated
weights remain subject to their original licenses. Several assets require
registration and are restricted to non-commercial research use. Create or
transfer this bundle only when the recipient is authorized under every
applicable license. The acknowledgement flag records an intentional packaging
decision; it does not grant redistribution rights.
