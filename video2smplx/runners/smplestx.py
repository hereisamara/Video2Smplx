"""In-process SMPLest-X runner."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import numpy as np

from video2smplx.contracts import FrameInput
from video2smplx.runners._runtime import add_project_to_path, project_cwd


class SmplestXRunner:
    """Load SMPLest-X once and run body prediction frame-by-frame."""

    def __init__(
        self,
        project_dir: Path,
        ckpt_name: str = "smplest_x_h",
        device: str = "cuda",
        multi_person: bool = False,
    ):
        self.project_dir = Path(project_dir).resolve()
        self.ckpt_name = ckpt_name
        self.device = device
        self.multi_person = multi_person

        add_project_to_path(self.project_dir)
        with project_cwd(self.project_dir):
            import torch
            import torchvision.transforms as transforms
            from main.base import Tester
            from main.config import Config
            from ultralytics import YOLO
            from utils.data_utils import generate_patch_image, load_img, process_bbox
            from utils.inference_utils import non_max_suppression

            self.torch = torch
            self.transforms = transforms
            self.load_img = load_img
            self.process_bbox = process_bbox
            self.generate_patch_image = generate_patch_image
            self.non_max_suppression = non_max_suppression

            if device != "cuda" or not torch.cuda.is_available():
                raise RuntimeError(
                    "SMPLest-X runner currently requires CUDA because the "
                    "legacy Tester path calls .cuda() internally."
                )

            config_path = self.project_dir / "pretrained_models" / ckpt_name / "config_base.py"
            checkpoint_path = (
                self.project_dir
                / "pretrained_models"
                / ckpt_name
                / f"{ckpt_name}.pth.tar"
            )
            detector_path = self.project_dir / "pretrained_models" / "yolov8x.pt"
            human_model_path = self.project_dir / "human_models" / "human_model_files"

            cfg = Config.load_config(str(config_path))
            exp_name = (
                f"integrated_{ckpt_name}_"
                f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            cfg.update_config(
                {
                    "model": {
                        "pretrained_model_path": str(checkpoint_path),
                        "human_model_path": str(human_model_path),
                    },
                    "inference": {
                        "detection": {
                            "model_path": str(detector_path),
                        }
                    },
                    "log": {
                        "exp_name": exp_name,
                        "output_dir": str(self.project_dir / "outputs" / exp_name),
                        "model_dir": str(self.project_dir / "outputs" / exp_name / "model_dump"),
                        "log_dir": str(self.project_dir / "outputs" / exp_name / "log"),
                        "result_dir": str(self.project_dir / "outputs" / exp_name / "result"),
                    },
                }
            )
            cfg.prepare_log()
            self.cfg = cfg
            self.demoer = Tester(cfg)
            self.demoer.logger.info("Integrated SMPLest-X runner loading model.")
            self.demoer._make_model()
            self.detector = YOLO(str(detector_path))
            self.transform = transforms.ToTensor()

    def predict(self, frame: FrameInput) -> list[dict[str, Any]]:
        """Return SMPL-X parameter dictionaries for the detected people."""
        torch = self.torch
        original_img = self.load_img(str(frame.path))
        original_img_height, original_img_width = original_img.shape[:2]

        detections = self.detector.predict(
            original_img,
            device=self.device,
            classes=0,
            conf=self.cfg.inference.detection.conf,
            save=self.cfg.inference.detection.save,
            verbose=self.cfg.inference.detection.verbose,
        )
        yolo_bbox = detections[0].boxes.xyxy.detach().cpu().numpy()
        if len(yolo_bbox) < 1:
            return []

        if not self.multi_person:
            areas = (yolo_bbox[:, 2] - yolo_bbox[:, 0]) * (yolo_bbox[:, 3] - yolo_bbox[:, 1])
            yolo_bbox = [yolo_bbox[int(np.argmax(areas))]]
        else:
            yolo_bbox = self.non_max_suppression(
                yolo_bbox,
                self.cfg.inference.detection.iou_thr,
            )

        people: list[dict[str, Any]] = []
        for bbox_id, current_yolo_bbox in enumerate(yolo_bbox):
            yolo_bbox_xywh = np.zeros((4), dtype=np.float32)
            yolo_bbox_xywh[0] = current_yolo_bbox[0]
            yolo_bbox_xywh[1] = current_yolo_bbox[1]
            yolo_bbox_xywh[2] = abs(current_yolo_bbox[2] - current_yolo_bbox[0])
            yolo_bbox_xywh[3] = abs(current_yolo_bbox[3] - current_yolo_bbox[1])

            bbox = self.process_bbox(
                bbox=yolo_bbox_xywh,
                img_width=original_img_width,
                img_height=original_img_height,
                input_img_shape=self.cfg.model.input_img_shape,
                ratio=getattr(self.cfg.data, "bbox_ratio", 1.25),
            )
            if bbox is None:
                continue

            img, _, _ = self.generate_patch_image(
                cvimg=original_img,
                bbox=bbox,
                scale=1.0,
                rot=0.0,
                do_flip=False,
                out_shape=self.cfg.model.input_img_shape,
            )
            img = self.transform(img.astype(np.float32)) / 255
            img = img.cuda()[None, :, :, :]

            with torch.no_grad():
                out = self.demoer.model({"img": img}, {}, {}, "test")

            global_orient = out["smplx_root_pose"].detach().cpu().numpy()[0]
            body_pose = out["smplx_body_pose"].detach().cpu().numpy()[0]
            left_hand_pose = out["smplx_lhand_pose"].detach().cpu().numpy()[0]
            right_hand_pose = out["smplx_rhand_pose"].detach().cpu().numpy()[0]
            jaw_pose = out["smplx_jaw_pose"].detach().cpu().numpy()[0]
            betas = out["smplx_shape"].detach().cpu().numpy()[0]
            expression = out["smplx_expr"].detach().cpu().numpy()[0]
            transl = out["cam_trans"].detach().cpu().numpy()[0]

            smplx_param_vector = np.concatenate(
                [
                    global_orient,
                    body_pose,
                    left_hand_pose,
                    right_hand_pose,
                    jaw_pose,
                    betas,
                    expression,
                    transl,
                ],
                axis=-1,
            )

            people.append(
                {
                    "smplx_param_vector": smplx_param_vector,
                    "global_orient": global_orient,
                    "body_pose": body_pose,
                    "left_hand_pose": left_hand_pose,
                    "right_hand_pose": right_hand_pose,
                    "jaw_pose": jaw_pose,
                    "betas": betas,
                    "expression": expression,
                    "transl": transl,
                    "bbox_xyxy": np.asarray(current_yolo_bbox, dtype=np.float32),
                    "bbox_xywh": yolo_bbox_xywh,
                    "person_index": bbox_id,
                }
            )

        return people
