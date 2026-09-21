import unittest
from concurrent.futures import ThreadPoolExecutor

from video2smplx.integrated_pipeline import (
    _effective_smooth_window,
    _format_timing_ms,
    _iter_batches,
    _is_valid_fused_frame,
    _predict_frame_parallel,
    _resolve_model_execution_mode,
    _timed_predict,
    _timing_average,
    build_parser,
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

    def test_execution_mode_resolution_preserves_legacy_flags(self):
        self.assertEqual(
            _resolve_model_execution_mode(None, parallel_models=True, smplestx_only=False),
            "frame_parallel",
        )
        self.assertEqual(
            _resolve_model_execution_mode(None, parallel_models=False, smplestx_only=False),
            "sequential_batched",
        )
        self.assertEqual(
            _resolve_model_execution_mode("streaming", parallel_models=False, smplestx_only=False),
            "streaming",
        )
        self.assertEqual(
            _resolve_model_execution_mode("streaming", parallel_models=True, smplestx_only=True),
            "sequential_batched",
        )

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

    def test_fast_io_related_cli_flags(self):
        args = build_parser().parse_args(
            [
                "--video",
                "input.mp4",
                "--output",
                "output",
                "--no_write_fused_params",
                "--per_frame_timing_log_interval",
                "0",
                "--wilor_detector_stride",
                "3",
                "--emoca_detector_stride",
                "5",
                "--model_execution_mode",
                "streaming",
                "--max_inflight_frames",
                "3",
                "--disable_wilor",
                "--disable_emoca",
            ]
        )

        self.assertFalse(args.write_fused_params)
        self.assertEqual(args.per_frame_timing_log_interval, 0)
        self.assertEqual(args.wilor_detector_stride, 3)
        self.assertEqual(args.emoca_detector_stride, 5)
        self.assertEqual(args.model_execution_mode, "streaming")
        self.assertEqual(args.max_inflight_frames, 3)
        self.assertTrue(args.disable_wilor)
        self.assertTrue(args.disable_emoca)


if __name__ == "__main__":
    unittest.main()
