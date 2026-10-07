import unittest

import numpy as np

from video2smplx.fusion import fuse_person, validate_person


def make_base_person():
    return {
        "global_orient": np.zeros(3),
        "body_pose": np.zeros(63),
        "left_hand_pose": np.zeros(45),
        "right_hand_pose": np.zeros(45),
        "jaw_pose": np.zeros(3),
        "betas": np.zeros(10),
        "expression": np.zeros(10),
        "transl": np.zeros(3),
        "smplx_param_vector": np.zeros(182),
    }


class FusionTest(unittest.TestCase):
    def test_fuse_person_replaces_specialist_fields_and_rebuilds_vector(self):
        base = make_base_person()
        wilor = {
            "right_hand_pose": np.ones(45),
            "left_hand_pose": None,
        }
        emoca = {
            "exp": np.full((1, 50), 2.0),
            "jaw_pose": np.full((1, 3), 3.0),
        }

        fused = fuse_person(base, wilor, emoca)

        self.assertTrue(np.allclose(fused["right_hand_pose"], 1.0))
        self.assertTrue(np.allclose(fused["left_hand_pose"], 0.0))
        self.assertTrue(np.allclose(fused["expression"], 2.0))
        self.assertTrue(np.allclose(fused["jaw_pose"], 3.0))
        self.assertEqual(fused["smplx_param_vector"].shape, (222,))
        self.assertEqual(base["smplx_param_vector"].shape, (182,))
        self.assertEqual(validate_person(fused), [])

    def test_validate_person_reports_bad_shapes(self):
        person = make_base_person()
        person["left_hand_pose"] = np.zeros(44)

        errors = validate_person(person)

        self.assertEqual(errors, ["key 'left_hand_pose' has dim 44, expected 45"])


if __name__ == "__main__":
    unittest.main()

