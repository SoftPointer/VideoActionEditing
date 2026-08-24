#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import packed_preservation_checkpoint_review_v2 as review
import packed_preservation_lora_v2 as core
import infer_packed_preservation_checkpoint_review_v2 as runner


class IncrementalCheckpointAuthorityTests(unittest.TestCase):
    def _checkpoint(
        self, root: Path, *, step: int = 2, scope: str = "optimizer-canary-2"
    ) -> tuple[Path, dict[str, object]]:
        directory = root / f"checkpoint-{step:08d}"
        directory.mkdir()
        adapter = directory / "adapter.pt"
        optimizer = directory / "optimizer.pt"
        adapter.write_bytes(b"adapter")
        optimizer.write_bytes(b"optimizer")
        inventory: list[object] = []
        parameter_sha = hashlib.sha256(f"parameter-{step}".encode()).hexdigest()
        metadata: dict[str, object] = {
            "schema_version": review.TRAINING_RECEIPT_SCHEMA,
            "method": review.TRAINING_METHOD,
            "execution_scope": scope,
            "step": step,
            "lora_scope": "all-attention",
            "rank": core.LORA_RANK,
            "source_only_manifest_sha256": review.SOURCE_ONLY_MANIFEST_SHA256,
            "adapter_file": "adapter.pt",
            "adapter_sha256": review.file_sha256(adapter),
            "optimizer_file": "optimizer.pt",
            "optimizer_sha256": review.file_sha256(optimizer),
            "parameter_sha256": parameter_sha,
            "roundtrip_parameter_sha256": parameter_sha,
            "strict_loader": "packed_preservation_lora_v2.load_trainable_state_strict",
            "adapter_reload_verified": True,
            "optimizer_reload_verified": True,
            "same_architecture_strict_reload_verified": True,
            "fresh_official_rv2v_inference_process_verified": False,
            "architecture": {
                "scope": "all-attention",
                "rank": core.LORA_RANK,
                "target_row_gating": False,
                "all_local_packed_tokens_receive_lora": True,
            },
            "trainable_inventory": inventory,
            "trainable_inventory_sha256": core.object_sha256(inventory),
        }
        (directory / "metadata.json").write_text(
            json.dumps(metadata, sort_keys=True), encoding="ascii"
        )
        return directory, metadata

    def test_canary_p2_is_admitted_before_terminal_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory, metadata = self._checkpoint(Path(raw).resolve())
            authority = review.load_checkpoint_authority(
                directory,
                expected_step=2,
                expected_lora_scope="all-attention",
                expected_execution_scope="optimizer-canary-2",
            )
            self.assertEqual(authority.step, 2)
            self.assertEqual(authority.parameter_sha256, metadata["parameter_sha256"])
            self.assertEqual(authority.adapter_sha256, metadata["adapter_sha256"])

    def test_canary_checkpoint_cannot_masquerade_as_exact80(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory, _ = self._checkpoint(Path(raw).resolve())
            with self.assertRaises(review.PackedPreservationReviewError):
                review.load_checkpoint_authority(
                    directory,
                    expected_step=2,
                    expected_lora_scope="all-attention",
                    expected_execution_scope="exact80",
                )

    def test_changed_adapter_bytes_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory, _ = self._checkpoint(Path(raw).resolve())
            (directory / "adapter.pt").write_bytes(b"changed")
            with self.assertRaises(review.PackedPreservationReviewError):
                review.load_checkpoint_authority(
                    directory,
                    expected_step=2,
                    expected_lora_scope="all-attention",
                    expected_execution_scope="optimizer-canary-2",
                )


class RuntimeAndLauncherStaticTests(unittest.TestCase):
    def test_pyav_probe_fully_decodes_exact81_without_external_ffprobe(self) -> None:
        codec = types.SimpleNamespace(width=736, height=704, name="h264")
        stream = types.SimpleNamespace(
            average_rate=Fraction(25, 1), codec_context=codec
        )
        frames = [types.SimpleNamespace(width=736, height=704) for _ in range(81)]

        class Container:
            streams = types.SimpleNamespace(video=[stream])

            def __enter__(self):
                return self

            def __exit__(self, *unused):
                return False

            def decode(self, *, video):
                self.assert_video = video
                return iter(frames)

        fake_av = types.SimpleNamespace(
            __version__=runner.PYAV_VERSION,
            open=lambda *args, **kwargs: Container(),
        )
        with mock.patch.dict(sys.modules, {"av": fake_av}):
            observed = runner._pyav_exact81(Path("/not-opened-by-fake/source.mp4"))
        self.assertEqual(observed["frame_count"], 81)
        self.assertEqual(observed["fps"], 25)
        self.assertEqual(observed["probe_backend"], "pyav-13.1.0-full-decode")

    def test_runtime_routes_native_source_ids_and_has_no_evaluator(self) -> None:
        runtime = (METHOD_ROOT / "packed_preservation_checkpoint_review_v2.py").read_text(
            encoding="utf-8"
        )
        runner = (
            METHOD_ROOT / "infer_packed_preservation_checkpoint_review_v2.py"
        ).read_text(encoding="utf-8")
        self.assertIn("source_id_value > 0.0", runtime)
        self.assertIn("route.typed.source_delta", runtime)
        self.assertIn("route.typed.target_delta", runtime)
        self.assertIn('trace["source_calls"] != review.NUM_INFERENCE_STEPS * 9', runner)
        self.assertIn('trace["target_calls"] != review.NUM_INFERENCE_STEPS', runner)
        self.assertIn("step-0 endpoint is not byte-exact native", runner)
        self.assertIn("_pyav_exact81(path)", runner)
        self.assertNotIn("authoring._ffprobe_exact81", runner)
        self.assertNotIn("optimizer.step", runner)
        self.assertNotIn("backward()", runner)

    def test_fp32_authority_precedes_inference_only_bfloat16_cast(self) -> None:
        runner = (
            METHOD_ROOT / "infer_packed_preservation_checkpoint_review_v2.py"
        ).read_text(encoding="utf-8")
        strict_load = runner.index("loaded = review.strict_load_adapter(model, checkpoint)")
        fp32_digest = runner.index(
            "checkpoint_parameter_fp32 = review.trainable_parameter_digest(model)"
        )
        inference_cast = runner.index(
            "parameter.data = parameter.data.to(dtype=torch.bfloat16)"
        )
        dtype_gate = runner.index("transformer.dtype != torch.bfloat16")
        adapted_sample = runner.index("for sentinel_id in sentinel_order:", dtype_gate)
        self.assertLess(strict_load, fp32_digest)
        self.assertLess(fp32_digest, inference_cast)
        self.assertLess(inference_cast, dtype_gate)
        self.assertLess(dtype_gate, adapted_sample)
        self.assertIn("step-0 endpoint is not byte-exact native", runner)
        self.assertIn("native_scheduler_outside_autocast", runner)
        self.assertNotIn("torch.autocast(", runner)

    def test_job136309_launcher_is_hard_bound_and_parent_safe(self) -> None:
        launcher = (
            METHOD_ROOT
            / "scripts/auh_decode_packed_preservation_checkpoint_v2_job136309.sh"
        ).read_text(encoding="utf-8")
        for binding in (
            "136140@auh7-1b-gpu-215",
            "136309@auh7-1b-gpu-280",
            "136141@auh7-1b-gpu-299",
            "135096@auh7-1b-gpu-246",
        ):
            self.assertIn(binding, launcher)
        self.assertIn("holder is outside the immutable decode allowlist", launcher)
        self.assertIn("target node already has a numbered child", launcher)
        self.assertIn("holder/lane binding differs", launcher)
        self.assertIn("assert_gpu_kfd_idle\nsleep 2\nassert_parent\nassert_gpu_kfd_idle", launcher)
        self.assertIn("fuser /dev/kfd", launcher)
        self.assertIn("--nproc_per_node=4", launcher)
        self.assertIn("--gres=gpu:mi210:4", launcher)
        self.assertIn("SMOKE_ONLY_P2_LOADER_DECODE_COMPLETE", launcher)
        self.assertNotIn("scancel", launcher)
        self.assertNotIn("scontrol release", launcher)
        self.assertNotIn("scontrol requeue", launcher)


if __name__ == "__main__":
    unittest.main()
