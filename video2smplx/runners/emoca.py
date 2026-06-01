"""In-process EMOCA runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from video2smplx.contracts import FrameInput
from video2smplx.runners._runtime import add_project_to_path, project_cwd, require_files


class EMOCARunner:
    """Load EMOCA once and return face expression/jaw parameters per frame."""

    def __init__(
        self,
        project_dir: Path,
        model_name: str = "EMOCA_v2_lr_mse_20",
        device: str = "cuda",
        crop_size: int = 224,
        scale: float = 1.25,
        face_detector_threshold: float = 0.5,
    ):
        self.project_dir = Path(project_dir).resolve()
        self.model_name = model_name
        self.crop_size = crop_size
        self.scale = scale
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        path_to_models = self.project_dir / "assets" / "EMOCA" / "models"
        checkpoint_dir = path_to_models / model_name / "detail" / "checkpoints"
        require_files(
            [
                path_to_models / model_name / "cfg.yaml",
                self.project_dir / "assets" / "DECA" / "data" / "deca_model.tar",
                self.project_dir / "assets" / "FLAME" / "geometry" / "generic_model.pkl",
                self.project_dir / "assets" / "FLAME" / "geometry" / "landmark_embedding.npy",
                self.project_dir / "assets" / "FLAME" / "geometry" / "mediapipe_landmark_embedding.npz",
                self.project_dir / "assets" / "FLAME" / "geometry" / "head_template.obj",
                self.project_dir / "assets" / "FLAME" / "geometry" / "fixed_uv_displacements" / "fixed_displacement_256.npy",
                self.project_dir / "assets" / "FLAME" / "mask" / "uv_face_mask.png",
                self.project_dir / "assets" / "FLAME" / "mask" / "uv_face_eye_mask.png",
            ],
            "EMOCA",
        )
        checkpoints = []
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob("*.ckpt"))
            if not checkpoints:
                checkpoints = [
                    path
                    for path in checkpoint_dir.rglob("*")
                    if path.is_file() and path.suffix == ""
                ]
        if not checkpoints:
            raise FileNotFoundError(
                "EMOCA checkpoint is missing:\n"
                f"  - expected at least one checkpoint file in {checkpoint_dir}"
            )

        add_project_to_path(self.project_dir)
        with project_cwd(self.project_dir):
            from gdl.datasets.ImageDatasetHelpers import bbox2point
            from gdl.utils.FaceDetector import FAN
            from gdl_apps.EMOCA.utils.load import load_model
            from skimage.transform import estimate_transform, warp

            self.bbox2point = bbox2point
            self.estimate_transform = estimate_transform
            self.warp = warp

            self.emoca, self.conf = load_model(path_to_models, model_name, "detail")
            self.emoca = self.emoca.to(self.device)
            self.emoca.eval()
            self.face_detector = FAN(device=str(self.device), threshold=face_detector_threshold)

    @staticmethod
    def empty_result() -> dict[str, Any]:
        return {
            "exp": None,
            "pose": None,
            "global_orient": None,
            "jaw_pose": None,
            "shape": None,
            "cam": None,
            "light": None,
            "tex": None,
        }

    def _crop_face(self, image_rgb: np.ndarray) -> torch.Tensor:
        h, w, _ = image_rgb.shape
        bboxes, bbox_type = self.face_detector.run(image_rgb)
        if len(bboxes) < 1:
            left, right, top, bottom = 0, w - 1, 0, h - 1
        else:
            bbox = bboxes[0]
            left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]

        old_size, center = self.bbox2point(left, right, top, bottom, type=bbox_type)
        size = int(old_size * self.scale)
        src_pts = np.array(
            [
                [center[0] - size / 2, center[1] - size / 2],
                [center[0] - size / 2, center[1] + size / 2],
                [center[0] + size / 2, center[1] - size / 2],
            ]
        )

        image = image_rgb.astype(np.float32) / 255.0
        dst_pts = np.array(
            [
                [0, 0],
                [0, self.crop_size - 1],
                [self.crop_size - 1, 0],
            ]
        )
        tform = self.estimate_transform("similarity", src_pts, dst_pts)
        dst_image = self.warp(
            image,
            tform.inverse,
            output_shape=(self.crop_size, self.crop_size),
        )
        return torch.tensor(dst_image.transpose(2, 0, 1)).float()

    @staticmethod
    def _numpy(vals: dict[str, Any], key: str):
        if key not in vals:
            return None
        value = vals[key]
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        return value

    def predict(self, frame: FrameInput) -> dict[str, Any]:
        if frame.image_bgr is not None:
            image_bgr = frame.image_bgr
        elif frame.path is not None:
            image_bgr = cv2.imread(str(frame.path))
        else:
            raise ValueError("FrameInput must contain either image_bgr or path.")

        if image_bgr is None:
            raise FileNotFoundError(f"Could not read frame: {frame.path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        cropped = self._crop_face(image_rgb)
        batch = {"image": cropped.unsqueeze(0).to(self.device)}

        with torch.no_grad():
            vals = self.emoca.encode(batch, training=False)

        result = self.empty_result()
        exp = self._numpy(vals, "expcode")
        if exp is None:
            exp = self._numpy(vals, "exp")
        pose = self._numpy(vals, "posecode")
        if pose is None:
            pose = self._numpy(vals, "pose")

        result["exp"] = exp
        result["pose"] = pose
        if pose is not None:
            result["global_orient"] = pose[:, :3]
            result["jaw_pose"] = pose[:, 3:]

        shape = self._numpy(vals, "shapecode")
        if shape is None:
            shape = self._numpy(vals, "sha")
        result["shape"] = shape
        result["cam"] = self._numpy(vals, "cam")
        result["light"] = self._numpy(vals, "lightcode")
        result["tex"] = self._numpy(vals, "texcode")
        return result
