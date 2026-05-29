"""In-process WiLoR runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from video2smplx.contracts import FrameInput
from video2smplx.runners._runtime import add_project_to_path, project_cwd


class WiLoRRunner:
    """Load WiLoR once and return hand parameters for each frame."""

    def __init__(
        self,
        project_dir: Path,
        device: str = "cuda",
        rescale_factor: float = 2.0,
        batch_size: int = 16,
    ):
        self.project_dir = Path(project_dir).resolve()
        self.device_name = device
        self.rescale_factor = rescale_factor
        self.batch_size = batch_size

        add_project_to_path(self.project_dir)
        with project_cwd(self.project_dir):
            import torch
            from scipy.spatial.transform import Rotation
            from ultralytics import YOLO
            from wilor.datasets.vitdet_dataset import ViTDetDataset
            from wilor.models import load_wilor
            from wilor.utils import recursive_to

            self.torch = torch
            self.Rotation = Rotation
            self.ViTDetDataset = ViTDetDataset
            self.recursive_to = recursive_to

            self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
            self.model, self.model_cfg = load_wilor(
                checkpoint_path="./pretrained_models/wilor_final.ckpt",
                cfg_path="./pretrained_models/model_config.yaml",
            )
            self.detector = YOLO("./pretrained_models/detector.pt")
            self.model = self.model.to(self.device)
            self.detector = self.detector.to(self.device)
            self.model.eval()

    @staticmethod
    def empty_result() -> dict[str, Any]:
        return {
            "right_hand_pose": None,
            "left_hand_pose": None,
            "right_hand_betas": None,
            "left_hand_betas": None,
            "right_hand_global_orient": None,
            "left_hand_global_orient": None,
        }

    def predict(self, frame: FrameInput) -> dict[str, Any]:
        img_cv2 = cv2.imread(str(frame.path))
        if img_cv2 is None:
            raise FileNotFoundError(f"Could not read frame: {frame.path}")

        detections = self.detector(img_cv2, conf=0.3, verbose=False)[0]
        boxes_obj = detections.boxes
        if boxes_obj is None or len(boxes_obj) == 0:
            return self.empty_result()

        boxes = boxes_obj.xyxy.detach().cpu().numpy()
        right = boxes_obj.cls.detach().cpu().numpy()
        if len(boxes) == 0:
            return self.empty_result()

        dataset = self.ViTDetDataset(
            self.model_cfg,
            img_cv2,
            boxes,
            right,
            rescale_factor=self.rescale_factor,
        )
        dataloader = self.torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        frame_params = self.empty_result()
        for batch in dataloader:
            batch = self.recursive_to(batch, self.device)
            with self.torch.no_grad():
                out = self.model(batch)

            batch_size = batch["img"].shape[0]
            for n in range(batch_size):
                is_right_hand = batch["right"][n].detach().cpu().numpy()
                wrist_rotmat = (
                    out["pred_mano_params"]["global_orient"][n]
                    .detach()
                    .cpu()
                    .numpy()
                )
                finger_rotmat = (
                    out["pred_mano_params"]["hand_pose"][n]
                    .detach()
                    .cpu()
                    .numpy()
                )
                betas = out["pred_mano_params"]["betas"][n].detach().cpu().numpy()

                wrist_aa = self.Rotation.from_matrix(wrist_rotmat.squeeze(0)).as_rotvec()
                fingers_aa = self.Rotation.from_matrix(finger_rotmat).as_rotvec()

                if is_right_hand == 1.0:
                    frame_params["right_hand_pose"] = fingers_aa.flatten()
                    frame_params["right_hand_betas"] = betas
                    frame_params["right_hand_global_orient"] = wrist_aa
                else:
                    reflection_vector = np.array([1, -1, -1])
                    frame_params["left_hand_pose"] = (fingers_aa * reflection_vector).flatten()
                    frame_params["left_hand_betas"] = betas
                    frame_params["left_hand_global_orient"] = wrist_aa * reflection_vector

        return frame_params

