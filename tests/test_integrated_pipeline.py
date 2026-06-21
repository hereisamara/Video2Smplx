import unittest
from concurrent.futures import ThreadPoolExecutor

from video2smplx.integrated_pipeline import (
    _effective_smooth_window,
    _format_timing_ms,
    _iter_batches,
    _is_valid_fused_frame,
    _predict_frame_parallel,
    _timed_predict,
    _timing_average,
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

    def test_timing_helpers_format_missing_and_average_values(self):
        self.assertEqual(_format_timing_ms(None), "skipped")
        self.assertEqual(_format_timing_ms(0.12345), "123.5 ms")
        self.assertEqual(_timing_average(3.0, 2), 1.5)
        self.assertIsNone(_timing_average(3.0, 0))

    def test_iter_batches_groups_items_and_keeps_tail(self):
        self.assertEqual(
            list(_iter_batches([1, 2, 3, 4, 5], 2)),
            [[1, 2], [3, 4], [5]],
        )

    def test_timed_predict_returns_result_and_duration(self):
        result = _timed_predict(lambda frame: f"ok-{frame}", "frame")

        self.assertEqual(result["result"], "ok-frame")
        self.assertIsNone(result["exception"])
        self.assertGreaterEqual(result["seconds"], 0.0)

    def test_predict_frame_parallel_collects_all_model_outputs(self):
        class Runner:
            def __init__(self, value):
                self.value = value

            def predict(self, frame):
                return f"{self.value}-{frame}"

        with ThreadPoolExecutor(max_workers=3) as executor:
            body, hands, face, timings = _predict_frame_parallel(
                executor,
                Runner("body"),
                Runner("hands"),
                Runner("face"),
                "frame",
            )

        self.assertEqual(body, "body-frame")
        self.assertEqual(hands, "hands-frame")
        self.assertEqual(face, "face-frame")
        self.assertEqual(set(timings), {"smplestx", "wilor", "emoca"})


if __name__ == "__main__":
    unittest.main()
