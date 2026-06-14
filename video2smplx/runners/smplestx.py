"""In-process SMPLest-X runner."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import numpy as np

from video2smplx.contracts import FrameInput
from video2smplx.runners._runtime import add_project_to_path, project_cwd, require_files


class SmplestXRunner:
    """Load SMPLest-X once and run body prediction frame-by-frame."""

    def __init__(
        self,
        project_dir: Path,
        ckpt_name: str = "smplest_x_h",
        device: str = "cuda",
        multi_person: bool = False,
        detector_stride: int = 1,
        use_inference_mode: bool = False,
        single_gpu_model: bool = False,
        model_batch_size: int = 1,
    ):
        self.project_dir = Path(project_dir).resolve()
        self.ckpt_name = ckpt_name
        self.device = device
        self.multi_person = multi_person
        self.detector_stride = max(1, int(detector_stride))
        self.use_inference_mode = use_inference_mode
        self.single_gpu_model = single_gpu_model
        self.model_batch_size = max(1, int(model_batch_size))
        self._cached_yolo_bbox: np.ndarray | None = None

        add_project_to_path(self.project_dir)
        with project_cwd(self.project_dir):
            import torch
            import torchvision.transforms as transforms
            from human_models.human_models import SMPLX
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
            require_files(
                [
                    config_path,
                    checkpoint_path,
                    detector_path,
                    human_model_path / "smplx" / "SMPLX_NEUTRAL.npz",
                    human_model_path / "smplx" / "SMPLX_MALE.npz",
                    human_model_path / "smplx" / "SMPLX_FEMALE.npz",
                    human_model_path / "smplx" / "SMPLX_to_J14.pkl",
                    human_model_path / "smplx" / "MANO_SMPLX_vertex_ids.pkl",
                    human_model_path / "smplx" / "SMPL-X__FLAME_vertex_ids.npy",
                ],
                "SMPLest-X",
            )

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
            # SMPLest-X model construction calls SMPLX() without arguments in
            # TransformerDecoderHead, so initialize the singleton first.
            self.smpl_x = SMPLX(cfg.model.human_model_path)
            self.demoer = Tester(cfg)
            self.demoer.logger.info("Integrated SMPLest-X runner loading model.")
            self.demoer._make_model()
            self.model = self.demoer.model
            if self.single_gpu_model and hasattr(self.model, "module"):
                self.model = self.model.module
                self.model.cuda()
                self.model.eval()
            self.detector = YOLO(str(detector_path))
            self.transform = transforms.ToTensor()

    def _load_original_img(self, frame: FrameInput) -> np.ndarray:
        if frame.image_bgr is not None:
            return frame.image_bgr[:, :, ::-1].copy().astype(np.float32)
        if frame.path is None:
            raise ValueError("FrameInput must contain either image_bgr or path.")
        return self.load_img(str(frame.path))

    def _model_context(self):
        if self.use_inference_mode:
            return self.torch.inference_mode()
        return self.torch.no_grad()

    def _should_run_detector(self, frame: FrameInput) -> bool:
        if self.detector_stride <= 1:
            return True
        if self._cached_yolo_bbox is None:
            return True
        return (frame.frame_id - 1) % self.detector_stride == 0

    def _detect_yolo_bboxes(self, original_img: np.ndarray) -> np.ndarray:
        detections = self.detector.predict(
            original_img,
            device=self.device,
            classes=0,
            conf=self.cfg.inference.detection.conf,
            save=self.cfg.inference.detection.save,
            verbose=self.cfg.inference.detection.verbose,
        )
        boxes = detections[0].boxes
        if boxes is None or len(boxes) < 1:
            return np.empty((0, 4), dtype=np.float32)
        return boxes.xyxy.detach().cpu().numpy().astype(np.float32)

    def _select_yolo_bboxes(self, yolo_bbox: np.ndarray) -> np.ndarray:
        if len(yolo_bbox) < 1:
            return np.empty((0, 4), dtype=np.float32)

        if not self.multi_person:
            areas = (yolo_bbox[:, 2] - yolo_bbox[:, 0]) * (
                yolo_bbox[:, 3] - yolo_bbox[:, 1]
            )
            return np.asarray([yolo_bbox[int(np.argmax(areas))]], dtype=np.float32)

        selected = self.non_max_suppression(
            yolo_bbox,
            self.cfg.inference.detection.iou_thr,
        )
        return np.asarray(selected, dtype=np.float32).reshape(-1, 4)

    def _get_frame_bboxes(
        self,
        frame: FrameInput,
        original_img: np.ndarray,
    ) -> np.ndarray:
        if self._should_run_detector(frame):
            yolo_bbox = self._select_yolo_bboxes(self._detect_yolo_bboxes(original_img))
            self._cached_yolo_bbox = yolo_bbox.copy() if len(yolo_bbox) else None
            return yolo_bbox

        return self._cached_yolo_bbox.copy()

    @staticmethod
    def _bbox_xyxy_to_xywh(current_yolo_bbox: np.ndarray) -> np.ndarray:
        yolo_bbox_xywh = np.zeros((4), dtype=np.float32)
        yolo_bbox_xywh[0] = current_yolo_bbox[0]
        yolo_bbox_xywh[1] = current_yolo_bbox[1]
        yolo_bbox_xywh[2] = abs(current_yolo_bbox[2] - current_yolo_bbox[0])
        yolo_bbox_xywh[3] = abs(current_yolo_bbox[3] - current_yolo_bbox[1])
        return yolo_bbox_xywh

    @staticmethod
    def _numpy_outputs(out: dict[str, Any]) -> dict[str, np.ndarray]:
        return {
            "global_orient": out["smplx_root_pose"].detach().cpu().numpy(),
            "body_pose": out["smplx_body_pose"].detach().cpu().numpy(),
            "left_hand_pose": out["smplx_lhand_pose"].detach().cpu().numpy(),
            "right_hand_pose": out["smplx_rhand_pose"].detach().cpu().numpy(),
            "jaw_pose": out["smplx_jaw_pose"].detach().cpu().numpy(),
            "betas": out["smplx_shape"].detach().cpu().numpy(),
            "expression": out["smplx_expr"].detach().cpu().numpy(),
            "transl": out["cam_trans"].detach().cpu().numpy(),
        }

    @staticmethod
    def _build_person(
        arrays: dict[str, np.ndarray],
        row_idx: int,
        current_yolo_bbox: np.ndarray,
        yolo_bbox_xywh: np.ndarray,
        bbox_id: int,
    ) -> dict[str, Any]:
        global_orient = arrays["global_orient"][row_idx]
        body_pose = arrays["body_pose"][row_idx]
        left_hand_pose = arrays["left_hand_pose"][row_idx]
        right_hand_pose = arrays["right_hand_pose"][row_idx]
        jaw_pose = arrays["jaw_pose"][row_idx]
        betas = arrays["betas"][row_idx]
        expression = arrays["expression"][row_idx]
        transl = arrays["transl"][row_idx]

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

        return {
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

    def predict(self, frame: FrameInput) -> list[dict[str, Any]]:
        """Return SMPL-X parameter dictionaries for the detected people."""
        return self.predict_batch([frame])[0]

    def predict_batch(self, frames: list[FrameInput]) -> list[list[dict[str, Any]]]:
        """Return SMPL-X parameter dictionaries for a batch of frames."""
        if not frames:
            return []

        torch = self.torch
        people_by_frame: list[list[dict[str, Any]]] = [[] for _ in frames]
        crop_tensors = []
        crop_records: list[dict[str, Any]] = []

        for frame_idx, frame in enumerate(frames):
            original_img = self._load_original_img(frame)
            original_img_height, original_img_width = original_img.shape[:2]
            yolo_bbox = self._get_frame_bboxes(frame, original_img)
            if len(yolo_bbox) < 1:
                continue

            for bbox_id, current_yolo_bbox in enumerate(yolo_bbox):
                yolo_bbox_xywh = self._bbox_xyxy_to_xywh(current_yolo_bbox)
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
                crop_tensors.append(self.transform(img.astype(np.float32)) / 255)
                crop_records.append(
                    {
                        "frame_idx": frame_idx,
                        "bbox_id": bbox_id,
                        "current_yolo_bbox": current_yolo_bbox,
                        "yolo_bbox_xywh": yolo_bbox_xywh,
                    }
                )

        if not crop_tensors:
            return people_by_frame

        for start in range(0, len(crop_tensors), self.model_batch_size):
            end = start + self.model_batch_size
            batch = torch.stack(crop_tensors[start:end], dim=0).cuda()
            with self._model_context():
                out = self.model({"img": batch}, {}, {}, "test")

            arrays = self._numpy_outputs(out)
            for row_idx, record in enumerate(crop_records[start:end]):
                person = self._build_person(
                    arrays=arrays,
                    row_idx=row_idx,
                    current_yolo_bbox=record["current_yolo_bbox"],
                    yolo_bbox_xywh=record["yolo_bbox_xywh"],
                    bbox_id=record["bbox_id"],
                )
                people_by_frame[record["frame_idx"]].append(person)

        return people_by_frame
