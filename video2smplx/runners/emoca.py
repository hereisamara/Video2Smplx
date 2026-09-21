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
        detector_stride: int = 1,
        batch_size: int = 16,
    ):
        self.project_dir = Path(project_dir).resolve()
        self.model_name = model_name
        self.crop_size = crop_size
        self.scale = scale
        self.batch_size = max(1, int(batch_size))
        self.detector_stride = max(1, int(detector_stride))
        self._cached_face_bbox: tuple[float, float, float, float, Any] | None = None
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
            checkpoints = list(checkpoint_dir.rglob("*.ckpt"))
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

    def _should_run_detector(self, frame_id: int) -> bool:
        if self.detector_stride <= 1:
            return True
        if self._cached_face_bbox is None:
            return True
        return (frame_id - 1) % self.detector_stride == 0

    def _face_bbox(
        self,
        image_rgb: np.ndarray,
        frame_id: int,
    ) -> tuple[float, float, float, float, Any]:
        h, w, _ = image_rgb.shape
        if not self._should_run_detector(frame_id):
            assert self._cached_face_bbox is not None
            return self._cached_face_bbox

        bboxes, bbox_type = self.face_detector.run(image_rgb)
        if len(bboxes) < 1:
            result = (0.0, 0.0, float(w - 1), float(h - 1), bbox_type)
        else:
            bbox = bboxes[0]
            result = (
                float(bbox[0]),
                float(bbox[1]),
                float(bbox[2]),
                float(bbox[3]),
                bbox_type,
            )
        self._cached_face_bbox = result
        return result

    def _crop_face(self, image_rgb: np.ndarray, frame_id: int) -> torch.Tensor:
        left, top, right, bottom, bbox_type = self._face_bbox(image_rgb, frame_id)

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
        cropped = self._crop_face(image_rgb, frame.frame_id)
        batch = {"image": cropped.unsqueeze(0).to(self.device)}

        with torch.inference_mode():
            vals = self.emoca.encode(batch, training=False)

        return self._result_from_values(vals)

    def _result_from_values(self, vals: dict[str, Any]) -> dict[str, Any]:
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

    def predict_batch(self, frames: list[FrameInput]) -> list[dict[str, Any]]:
        """Crop all faces first, then encode them in GPU micro-batches."""
        crops = []
        for frame in frames:
            if frame.image_bgr is not None:
                image_bgr = frame.image_bgr
            elif frame.path is not None:
                image_bgr = cv2.imread(str(frame.path))
            else:
                raise ValueError("FrameInput must contain either image_bgr or path.")
            if image_bgr is None:
                raise FileNotFoundError(f"Could not read frame: {frame.path}")
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            crops.append(self._crop_face(image_rgb, frame.frame_id))

        results: list[dict[str, Any]] = []
        for start in range(0, len(crops), self.batch_size):
            crop_batch = torch.stack(crops[start : start + self.batch_size]).to(self.device)
            with torch.inference_mode():
                values = self.emoca.encode({"image": crop_batch}, training=False)
            batch_count = int(crop_batch.shape[0])
            for index in range(batch_count):
                one_values = {}
                for key, value in values.items():
                    shape = getattr(value, "shape", None)
                    if shape is not None and len(shape) > 0 and shape[0] == batch_count:
                        one_values[key] = value[index : index + 1]
                    else:
                        one_values[key] = value
                results.append(self._result_from_values(one_values))
        return results
