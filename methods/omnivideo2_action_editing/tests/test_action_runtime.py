from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_omnivideo2_action import (  # noqa: E402
    _begin_optimizer_window,
    _finish_optimizer_window,
    _update_runtime_maxima,
    _validate_runtime_all_ranks,
)


class _FakeCuda:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def synchronize(self, device: str) -> None:
        self.calls.append(("synchronize", device))

    def reset_peak_memory_stats(self, device: str) -> None:
        self.calls.append(("reset_peak_memory_stats", device))

    def max_memory_allocated(self, device: str) -> int:
        self.calls.append(("max_memory_allocated", device))
        return 123

    def max_memory_reserved(self, device: str) -> int:
        self.calls.append(("max_memory_reserved", device))
        return 456


class _Clock:
    def __init__(self, *values: float) -> None:
        self.values = iter(values)

    def __call__(self) -> float:
        return next(self.values)


class ActionRuntimeEvidenceTest(unittest.TestCase):
    def test_window_boundaries_synchronize_reset_and_measure_peaks(self) -> None:
        cuda = _FakeCuda()
        started = _begin_optimizer_window(
            cuda, "cuda:3", clock=_Clock(10.0)
        )
        record = _finish_optimizer_window(
            cuda,
            "cuda:3",
            started_at=started,
            rank=3,
            optimizer_step=1,
            microbatches=8,
            clock=_Clock(12.5),
        )
        self.assertEqual(
            cuda.calls,
            [
                ("synchronize", "cuda:3"),
                ("reset_peak_memory_stats", "cuda:3"),
                ("synchronize", "cuda:3"),
                ("max_memory_allocated", "cuda:3"),
                ("max_memory_reserved", "cuda:3"),
            ],
        )
        self.assertEqual(record["rank"], 3)
        self.assertEqual(record["optimizer_step"], 1)
        self.assertEqual(record["microbatches"], 8)
        self.assertEqual(record["isolated_optimizer_window_seconds"], 2.5)
        self.assertEqual(record["peak_memory_allocated_bytes"], 123)
        self.assertEqual(record["peak_memory_reserved_bytes"], 456)

    def test_gather_validation_orders_ranks_and_maxima_span_windows(self) -> None:
        step1 = [
            {
                "rank": 1,
                "optimizer_step": 1,
                "microbatches": 1,
                "isolated_optimizer_window_seconds": 2.0,
                "peak_memory_allocated_bytes": 200,
                "peak_memory_reserved_bytes": 300,
            },
            {
                "rank": 0,
                "optimizer_step": 1,
                "microbatches": 1,
                "isolated_optimizer_window_seconds": 1.5,
                "peak_memory_allocated_bytes": 100,
                "peak_memory_reserved_bytes": 400,
            },
        ]
        ordered = _validate_runtime_all_ranks(
            step1, world_size=2, optimizer_step=1
        )
        self.assertEqual([item["rank"] for item in ordered], [0, 1])
        maxima: dict[int, dict[str, int | float]] = {}
        _update_runtime_maxima(maxima, ordered)

        step2 = [
            {**ordered[0], "optimizer_step": 2, "peak_memory_allocated_bytes": 500},
            {
                **ordered[1],
                "optimizer_step": 2,
                "isolated_optimizer_window_seconds": 3.0,
            },
        ]
        _update_runtime_maxima(
            maxima,
            _validate_runtime_all_ranks(step2, world_size=2, optimizer_step=2),
        )
        self.assertEqual(maxima[0]["optimizer_windows"], 2)
        self.assertEqual(maxima[0]["max_peak_memory_allocated_bytes"], 500)
        self.assertEqual(maxima[0]["max_peak_memory_reserved_bytes"], 400)
        self.assertEqual(maxima[1]["max_isolated_optimizer_window_seconds"], 3.0)

        duplicate = [dict(ordered[0]), dict(ordered[0])]
        with self.assertRaisesRegex(RuntimeError, "duplicate or missing"):
            _validate_runtime_all_ranks(
                duplicate, world_size=2, optimizer_step=1
            )


if __name__ == "__main__":
    unittest.main()
