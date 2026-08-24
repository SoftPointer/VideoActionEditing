from __future__ import annotations

import unittest
from unittest import mock

import extract_anchor_raft_flow_v1 as flow
from tools import materialize_vae


class _Frame:
    shape = (240, 320, 3)


class SourceBucketTests(unittest.TestCase):
    def test_small_source_uses_same_upscaled_bucket_as_native_inference(self) -> None:
        with mock.patch.object(flow, "_read_video", return_value=([_Frame()], 25.0)):
            actual = flow._source_bucket(mock.sentinel.source)
        expected = materialize_vae.source_aspect_bucket(
            240, 320, max_pixels=flow.MAX_PIXELS, stride=16
        )
        self.assertEqual(actual, expected)
        self.assertGreater(actual[0], 240)
        self.assertGreater(actual[1], 320)


if __name__ == "__main__":
    unittest.main()
