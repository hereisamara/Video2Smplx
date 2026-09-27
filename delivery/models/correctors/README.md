# SignLanguage Corrector Checkpoints

The final accurate pipeline expects three project-trained checkpoints:

```text
delivery/models/correctors/global/best_model.pt
delivery/models/correctors/hand/best_model.pt
delivery/models/correctors/upper2d/best_model.pt
```

They predict, respectively:

- global orientation and translation correction;
- wrist and finger-pose correction;
- YOLO-keypoint-guided global orientation and arm-pose correction.

These checkpoints are project outputs, not SMPL-X, MANO, FLAME, SMPLest-X,
WiLoR, EMOCA, or YOLO base-model weights. Those third-party assets retain their
own licenses and must be installed according to `delivery/docs/MODEL_SETUP.md`.

## Original Server Sources

Run from the repository root on the training server:

```bash
bash delivery/scripts/install_server_delivery_assets.sh
```

The script copies the checkpoints from these experiment outputs by default:

```text
outputs_fusion_signlanguage_no_stablized/evaluation/global_correction_model_upper_r1/best_model.pt
/project/lt200246-mmacma/khtun/signlanguage_hand_correction_outputs/evaluation/hand_correction_model_wrist_fingers_r1/best_model.pt
/project/lt200246-mmacma/khtun/signlanguage_2d_guided_upperbody_outputs/evaluation/2d_guided_upperbody_model_r1/best_model.pt
```

Override `GLOBAL_CKPT_SOURCE`, `HAND_CKPT_SOURCE`, or
`UPPER2D_CKPT_SOURCE` when the experiment outputs are stored elsewhere.

After copying, verify the assets with:

```bash
bash delivery/scripts/verify_delivery_readiness.sh
```
