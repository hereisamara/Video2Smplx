import unittest

from video2smplx.integrated_pipeline import (
    _effective_smooth_window,
    _is_valid_fused_frame,
)


class IntegratedPipelineHelpersTest(unittest.TestCase):
    def test_valid_fused_frame_requires_primary_person_dict(self):
        self.assertTrue(_is_valid_fused_frame([{"body_pose": []}]))
        self.assertFalse(_is_valid_fused_frame([]))
        self.assertFalse(_is_valid_fused_frame([None]))

    def test_effective_smooth_window_clamps_to_valid_odd_frame_count(self):
        self.assertEqual(_effective_smooth_window(15, valid_frames=10, polyorder=3), 9)
        self.assertEqual(_effective_smooth_window(15, valid_frames=15, polyorder=3), 15)
        self.assertIsNone(_effective_smooth_window(15, valid_frames=3, polyorder=3))


if __name__ == "__main__":
    unittest.main()
