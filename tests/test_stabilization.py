import unittest

import numpy as np

from video2smplx.stabilization import LOWER_BODY_JOINT_INDICES, LowerBodyStabilizer


class LowerBodyStabilizerTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

