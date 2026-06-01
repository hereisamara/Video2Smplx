"""Optional pose stabilization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from video2smplx.fusion import flattened_array


# SMPL-X body_pose stores 21 body joints after the pelvis/root joint:
# L_Hip, R_Hip, Spine_1, L_Knee, R_Knee, Spine_2, L_Ankle, R_Ankle,
# Spine_3, L_Foot, R_Foot, Neck, L_Collar, R_Collar, Head,
# L_Shoulder, R_Shoulder, L_Elbow, R_Elbow, L_Wrist, R_Wrist.
LOWER_BODY_JOINT_INDICES = (0, 1, 3, 4, 6, 7, 9, 10)


@dataclass
class LowerBodyStabilizer:
    """Keep lower-body SMPL-X body_pose joints fixed from the first valid frame."""

    joint_indices: tuple[int, ...] = LOWER_BODY_JOINT_INDICES
    reference_pose: np.ndarray | None = None
    reference_frame_id: int | None = None
    applied_frames: int = 0

    def apply(self, people: list[dict[str, Any]], frame_id: int | None = None) -> list[dict[str, Any]]:
        if not people or not isinstance(people[0], dict) or "body_pose" not in people[0]:
            return people

        body_pose = flattened_array(people[0]["body_pose"])
        if body_pose.shape[0] != 63:
            return people

        joint_indices = list(self.joint_indices)
        pose_21x3 = body_pose.reshape(21, 3).copy()
        if self.reference_pose is None:
            self.reference_pose = pose_21x3[joint_indices].copy()
            self.reference_frame_id = frame_id
            return people

        pose_21x3[joint_indices] = self.reference_pose
        people[0]["body_pose"] = pose_21x3.reshape(-1)
        self.applied_frames += 1
        return people


@dataclass
class GlobalOrientStabilizer:
    """Keep SMPL-X root orientation fixed from the first valid frame."""

    reference_orient: np.ndarray | None = None
    reference_frame_id: int | None = None
    applied_frames: int = 0

    def apply(self, people: list[dict[str, Any]], frame_id: int | None = None) -> list[dict[str, Any]]:
        if not people or not isinstance(people[0], dict) or "global_orient" not in people[0]:
            return people

        global_orient = flattened_array(people[0]["global_orient"])
        if global_orient.shape[0] != 3:
            return people

        if self.reference_orient is None:
            self.reference_orient = global_orient.copy()
            self.reference_frame_id = frame_id
            return people

        people[0]["global_orient"] = self.reference_orient.copy()
        self.applied_frames += 1
        return people
