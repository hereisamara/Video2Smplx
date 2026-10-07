import unittest

import numpy as np

from video2smplx.reprojection import compare_landmarks, shoulder_similarity


class ReprojectionMetricsTest(unittest.TestCase):
    def test_shoulder_similarity_recovers_known_transform(self):
        source = np.array([[0.0, 0.0], [2.0, 0.0]])
        target = np.array([[5.0, 7.0], [5.0, 11.0]])

        transform = shoulder_similarity(source, target)

        np.testing.assert_allclose(transform.apply(source), target, atol=1e-8)
        np.testing.assert_allclose(transform.apply([[1.0, 1.0]]), [[3.0, 9.0]], atol=1e-8)

    def test_compare_landmarks_is_zero_after_similarity_alignment(self):
        reference = np.zeros((33, 2), dtype=float)
        rendered = np.zeros((33, 2), dtype=float)
        reference[11], reference[12], reference[13] = [10, 10], [30, 10], [8, 20]
        rendered[11], rendered[12], rendered[13] = [1, 1], [5, 1], [0.6, 3]
        visibility = np.ones(33, dtype=float)

        result = compare_landmarks(
            reference, visibility, rendered, visibility, scored_landmarks=(13,)
        )

        self.assertIsNotNone(result)
        np.testing.assert_allclose(result["errors_px"], [0.0], atol=1e-8)

    def test_compare_landmarks_filters_invisible_joints(self):
        points = np.zeros((33, 2), dtype=float)
        points[11], points[12] = [0, 0], [10, 0]
        visibility = np.ones(33, dtype=float)
        visibility[13] = 0.1

        result = compare_landmarks(
            points, visibility, points, np.ones(33), scored_landmarks=(13, 14)
        )

        self.assertEqual(result["joint_indices"].tolist(), [14])


if __name__ == "__main__":
    unittest.main()
