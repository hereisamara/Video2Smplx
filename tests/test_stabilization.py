import unittest

import numpy as np

from video2smplx.stabilization import (
    LOWER_BODY_JOINT_INDICES,
    TORSO_JOINT_INDICES,
    GlobalOrientStabilizer,
    LowerBodyStabilizer,
    TorsoStabilizer,
)


class LowerBodyStabilizerTest(unittest.TestCase):
    def test_captures_first_frame_reference_pose(self):
        stabilizer = LowerBodyStabilizer()
        first_pose = np.arange(63, dtype=np.float32)

        people = [{"body_pose": first_pose.copy()}]

        stabilizer.apply(people, frame_id=1)

        expected = first_pose.reshape(21, 3)[list(LOWER_BODY_JOINT_INDICES)]
        self.assertTrue(np.allclose(stabilizer.reference_pose, expected))
        self.assertEqual(stabilizer.reference_frame_id, 1)

    def test_freezes_only_lower_body_joints_after_first_valid_frame(self):
        stabilizer = LowerBodyStabilizer()
        first_pose = np.arange(63, dtype=np.float32)
        second_pose = np.full(63, 100.0, dtype=np.float32)

        first = [{"body_pose": first_pose.copy()}]
        second = [{"body_pose": second_pose.copy()}]

        stabilizer.apply(first, frame_id=1)
        stabilizer.apply(second, frame_id=2)

        second_21x3 = second[0]["body_pose"].reshape(21, 3)
        first_21x3 = first_pose.reshape(21, 3)
        for joint_idx in LOWER_BODY_JOINT_INDICES:
            self.assertTrue(np.allclose(second_21x3[joint_idx], first_21x3[joint_idx]))

        upper_joint_idx = 15
        self.assertTrue(np.allclose(second_21x3[upper_joint_idx], 100.0))
        self.assertEqual(stabilizer.reference_frame_id, 1)
        self.assertEqual(stabilizer.applied_frames, 1)


class GlobalOrientStabilizerTest(unittest.TestCase):
    def test_freezes_global_orient_after_first_valid_frame(self):
        stabilizer = GlobalOrientStabilizer()
        first = [{"global_orient": np.array([0.1, 0.2, 0.3], dtype=np.float32)}]
        second = [{"global_orient": np.array([0.4, 0.5, 0.6], dtype=np.float32)}]

        stabilizer.apply(first, frame_id=1)
        stabilizer.apply(second, frame_id=2)

        self.assertTrue(np.allclose(second[0]["global_orient"], first[0]["global_orient"]))
        self.assertEqual(stabilizer.reference_frame_id, 1)
        self.assertEqual(stabilizer.applied_frames, 1)


class TorsoStabilizerTest(unittest.TestCase):
    def test_freezes_only_spine_joints_after_first_valid_frame(self):
        stabilizer = TorsoStabilizer()
        first_pose = np.arange(63, dtype=np.float32)
        second_pose = np.full(63, 100.0, dtype=np.float32)

        first = [{"body_pose": first_pose.copy()}]
        second = [{"body_pose": second_pose.copy()}]

        stabilizer.apply(first, frame_id=1)
        stabilizer.apply(second, frame_id=2)

        second_21x3 = second[0]["body_pose"].reshape(21, 3)
        first_21x3 = first_pose.reshape(21, 3)
        for joint_idx in TORSO_JOINT_INDICES:
            self.assertTrue(np.allclose(second_21x3[joint_idx], first_21x3[joint_idx]))

        shoulder_joint_idx = 15
        self.assertTrue(np.allclose(second_21x3[shoulder_joint_idx], 100.0))
        self.assertEqual(stabilizer.reference_frame_id, 1)
        self.assertEqual(stabilizer.applied_frames, 1)


if __name__ == "__main__":
    unittest.main()
