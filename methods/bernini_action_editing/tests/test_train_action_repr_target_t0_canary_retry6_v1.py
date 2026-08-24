#!/usr/bin/env python3

from __future__ import annotations

import copy
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest import mock


try:
    import torch
    from torch import nn
except ModuleNotFoundError as error:  # pragma: no cover - host dependent
    raise unittest.SkipTest("PyTorch unavailable") from error


METHOD_ROOT = Path(__file__).resolve().parents[1]
SUBJECT_PATH = METHOD_ROOT / "train_action_repr_target_t0_canary_retry6_v1.py"
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))


def _load_subject():
    spec = importlib.util.spec_from_file_location("target_t0_canary_test", SUBJECT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


subject = _load_subject()
RESOLVED_TEMP_ROOT = Path(tempfile.gettempdir()).resolve()


class _Block(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.native = nn.Linear(width, width, bias=False)
        with torch.no_grad():
            self.native.weight.copy_(torch.eye(width) * 0.015625)

    def forward(self, hidden_states, *args, **kwargs):
        del args, kwargs
        return hidden_states + self.native(hidden_states)


class _Transformer(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_Block(width) for _ in range(30)])

    def forward(self, hidden_states):
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return hidden_states


def _sha(character: str) -> str:
    return character * 64


class TargetT0CanaryTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260824)
        self.width = 8
        self.teacher_width = 4
        self.layout = subject.g2a.TokenLayout(
            total_tokens=6, source_tokens=0, phase_count=3
        )
        self.hidden = torch.randn((1, 6, self.width), dtype=torch.float32)
        self.activity = torch.zeros((1, 6, 1), dtype=torch.bool)
        # Phase 0 is hard inactive; phase 1 onset and phase 2 terminal each
        # contain active and outside tokens.
        self.activity[:, 2, 0] = True
        self.activity[:, 4, 0] = True
        self.projection = torch.tensor(
            [
                [0.5, -0.5, 0.5, -0.5],
                [0.5, 0.5, -0.5, -0.5],
                [-0.5, 0.5, 0.5, -0.5],
                [-0.5, -0.5, 0.5, 0.5],
                [0.5, -0.5, -0.5, 0.5],
                [-0.5, 0.5, -0.5, 0.5],
                [0.5, 0.5, 0.5, 0.5],
                [-0.5, -0.5, -0.5, -0.5],
            ],
            dtype=torch.float32,
        )

    @staticmethod
    def _visible_device_environment(
        raw: str,
        *,
        hip: str = "",
        cuda: str = "",
    ) -> dict[str, str]:
        digest = hashlib.sha256(raw.encode("ascii")).hexdigest()
        return {
            "ROCR_VISIBLE_DEVICES": raw,
            "HIP_VISIBLE_DEVICES": hip,
            "CUDA_VISIBLE_DEVICES": cuda,
            "ACTION_REPR_T0_SLURM_ROCR_VISIBLE_DEVICES": raw,
            "ACTION_REPR_T0_SLURM_ROCR_VISIBLE_DEVICES_SHA256": digest,
            "ACTION_REPR_T0_SLURM_ROCR_VISIBLE_DEVICES_COUNT": "4",
            "ACTION_REPR_T0_SLURM_ROCR_VISIBLE_DEVICES_PRESERVED": "true",
        }

    @staticmethod
    def _claim_mapping(raw: str = "4,5,6,7") -> dict[str, object]:
        return {
            "raw": raw,
            "devices": raw.split(","),
            "device_count": 4,
            "sha256": hashlib.sha256(raw.encode("ascii")).hexdigest(),
            "mapping_preserved": True,
            "hip_visible_devices_nonempty": False,
            "cuda_visible_devices_nonempty": False,
        }

    def _create_claim(self, output: Path, *, job_id: str = "151620"):
        return subject.create_preoptimizer_attempt_claim(
            output=output,
            marker_path=output.parent / subject.ATTEMPT_CLAIM_MARKER_NAME,
            authority_sha256=_sha("6"),
            source_hash_pins_digest=_sha("d"),
            slurm_visible_devices=self._claim_mapping(),
            environ={"SLURM_JOB_ID": job_id},
        )

    def test_retry6_atomic_attempt_claim_has_exactly_one_concurrent_winner(self):
        with tempfile.TemporaryDirectory(dir=str(RESOLVED_TEMP_ROOT)) as temporary:
            output = Path(temporary) / "single_update"
            start = threading.Barrier(8)

            def compete():
                start.wait()
                try:
                    return ("won", self._create_claim(output))
                except subject.TargetT0CanaryError as error:
                    return ("lost", str(error))

            with ThreadPoolExecutor(max_workers=8) as executor:
                results = [
                    future.result()
                    for future in as_completed([executor.submit(compete) for _ in range(8)])
                ]
            winners = [row for status, row in results if status == "won"]
            losers = [row for status, row in results if status == "lost"]
            self.assertEqual(len(winners), 1)
            self.assertEqual(len(losers), 7)
            self.assertTrue(all("use retry7" in message for message in losers))
            evidence = winners[0]
            marker = Path(evidence["path"])
            self.assertEqual(marker.name, subject.ATTEMPT_CLAIM_MARKER_NAME)
            self.assertEqual(marker.stat().st_mode & 0o777, 0o600)
            self.assertEqual(evidence["claim"]["source_hash_pins_digest"], _sha("d"))
            self.assertEqual(evidence["claim"]["slurm_job_id"], "151620")
            self.assertEqual(
                subject.replay_preoptimizer_attempt_claim(
                    marker, expected_claim=evidence["claim"]
                ),
                evidence,
            )

    def test_retry6_attempt_claim_missing_partial_symlink_and_tamper_fail_closed(self):
        with tempfile.TemporaryDirectory(dir=str(RESOLVED_TEMP_ROOT)) as temporary:
            root = Path(temporary)
            missing = root / subject.ATTEMPT_CLAIM_MARKER_NAME
            with self.assertRaisesRegex(subject.TargetT0CanaryError, "missing"):
                subject.replay_preoptimizer_attempt_claim(missing)

        with tempfile.TemporaryDirectory(dir=str(RESOLVED_TEMP_ROOT)) as temporary:
            root = Path(temporary)
            marker = root / subject.ATTEMPT_CLAIM_MARKER_NAME
            marker.write_bytes(b'{"partial":')
            os.chmod(marker, 0o600)
            with self.assertRaisesRegex(subject.TargetT0CanaryError, "partial"):
                subject.replay_preoptimizer_attempt_claim(marker)
            with self.assertRaisesRegex(subject.TargetT0CanaryError, "use retry7"):
                self._create_claim(root / "single_update")
            self.assertTrue(marker.exists())

        with tempfile.TemporaryDirectory(dir=str(RESOLVED_TEMP_ROOT)) as temporary:
            root = Path(temporary)
            target = root / "target.json"
            target.write_text("{}", encoding="ascii")
            marker = root / subject.ATTEMPT_CLAIM_MARKER_NAME
            marker.symlink_to(target)
            with self.assertRaises(subject.TargetT0CanaryError):
                subject.replay_preoptimizer_attempt_claim(marker)
            with self.assertRaisesRegex(subject.TargetT0CanaryError, "use retry7"):
                self._create_claim(root / "single_update")
            self.assertTrue(marker.is_symlink())

        with tempfile.TemporaryDirectory(dir=str(RESOLVED_TEMP_ROOT)) as temporary:
            root = Path(temporary)
            evidence = self._create_claim(root / "single_update")
            marker = Path(evidence["path"])
            marker.write_bytes(marker.read_bytes() + b" ")
            with self.assertRaisesRegex(subject.TargetT0CanaryError, "canonical"):
                subject.replay_preoptimizer_attempt_claim(marker)
            with self.assertRaisesRegex(subject.TargetT0CanaryError, "use retry7"):
                self._create_claim(root / "single_update")
            self.assertTrue(marker.exists())

    def test_retry6_rank0_claim_failure_is_broadcast_before_all_ranks_fail(self):
        with tempfile.TemporaryDirectory(dir=str(RESOLVED_TEMP_ROOT)) as temporary:
            output = Path(temporary) / "single_update"
            self._create_claim(output)
            broadcasts = []

            def capture(values, *, src):
                broadcasts.append((copy.deepcopy(values), src))

            with self.assertRaisesRegex(subject.TargetT0CanaryError, "failed permanently"):
                subject.acquire_preoptimizer_attempt_claim_world4(
                    output=output,
                    marker_path=output.parent / subject.ATTEMPT_CLAIM_MARKER_NAME,
                    authority_sha256=_sha("6"),
                    source_hash_pins_digest=_sha("d"),
                    slurm_visible_devices=self._claim_mapping(),
                    world_size=4,
                    sequence_parallel_size=4,
                    rank=0,
                    broadcast_object_list=capture,
                    environ={"SLURM_JOB_ID": "151621"},
                )
            self.assertEqual(len(broadcasts), 1)
            self.assertEqual(broadcasts[0][1], 0)
            self.assertFalse(broadcasts[0][0][0]["ok"])
            self.assertIn("use retry7", broadcasts[0][0][0]["error"])

    def test_same_authority_cannot_claim_a_second_cli_output_parent(self):
        with tempfile.TemporaryDirectory(dir=str(RESOLVED_TEMP_ROOT)) as temporary:
            root = Path(temporary)
            authorized_output = root / "authorized" / "single_update"
            authorized_output.parent.mkdir()
            authority_marker = (
                authorized_output.parent / subject.ATTEMPT_CLAIM_MARKER_NAME
            )
            winner = subject.create_preoptimizer_attempt_claim(
                output=authorized_output,
                marker_path=authority_marker,
                authority_sha256=_sha("6"),
                source_hash_pins_digest=_sha("d"),
                slurm_visible_devices=self._claim_mapping(),
                environ={"SLURM_JOB_ID": "151620"},
            )
            alternate_output = root / "alternate" / "single_update"
            alternate_output.parent.mkdir()
            with mock.patch.object(subject.torch.optim, "AdamW") as optimizer:
                with self.assertRaisesRegex(
                    subject.TargetT0CanaryError, "marker path/output binding"
                ):
                    subject.create_preoptimizer_attempt_claim(
                        output=alternate_output,
                        marker_path=authority_marker,
                        authority_sha256=_sha("6"),
                        source_hash_pins_digest=_sha("d"),
                        slurm_visible_devices=self._claim_mapping(),
                        environ={"SLURM_JOB_ID": "151621"},
                    )
                optimizer.assert_not_called()
            self.assertTrue(Path(winner["path"]).is_file())
            self.assertFalse(
                (alternate_output.parent / subject.ATTEMPT_CLAIM_MARKER_NAME).exists()
            )

    def test_retry6_claim_call_precedes_cache_model_adapter_forward_and_optimizer(self):
        source = SUBJECT_PATH.read_text(encoding="utf-8")
        main = source.split("def main(", 1)[1].split(
            "# Stable launcher-facing name.", 1
        )[0]
        claim = main.index("acquire_preoptimizer_attempt_claim_world4(")
        cache = main.index("load_authenticated_route_cache_maps(")
        posterior_model_load = main.index("_source_posterior_world4(")
        renderer_model_load = main.index("renderer = BerniniRendererModel(config)")
        adapter_forward_optimizer = main.index("run_one_step_optimizer_canary(")
        self.assertLess(claim, cache)
        self.assertLess(claim, posterior_model_load)
        self.assertLess(claim, renderer_model_load)
        self.assertLess(claim, adapter_forward_optimizer)
        between_consensus_and_cache = main[
            main.index('label="preoptimizer authority"'):cache
        ]
        self.assertIn("acquire_preoptimizer_attempt_claim_world4(", between_consensus_and_cache)
        self.assertIn("broadcast_object_list=dist.broadcast_object_list", main)
        self.assertIn("os.O_CREAT | os.O_EXCL | os.O_WRONLY | nofollow", source)
        self.assertIn("os.fsync(descriptor)", source)

    def test_slurm_visible_device_environment_preserves_legal_four_device_mappings(self):
        for raw in ("0,1,2,3", "4,5,6,7", "0,2,5,7", "1,3,4,6"):
            with self.subTest(raw=raw):
                receipt = subject.validate_slurm_visible_device_environment(
                    self._visible_device_environment(raw)
                )
                self.assertEqual(receipt["raw"], raw)
                self.assertEqual(receipt["devices"], raw.split(","))
                self.assertEqual(receipt["device_count"], 4)
                self.assertTrue(receipt["mapping_preserved"])
                self.assertEqual(
                    subject.validate_slurm_visible_device_receipt(receipt), receipt
                )

    def test_slurm_visible_device_environment_rejects_overrides_and_dangerous_values(self):
        rejected = (
            "",
            "0,1,2",
            "0,1,2,3,4",
            "0,1,2,2",
            "0,1,2,8",
            "0,1,2,-1",
            "0,1,2,a",
            "0,1,2, 3",
            "0,1,2,/3",
            "0,1,2,\\3",
            "0,1,,3",
            "00,1,2,3",
        )
        for raw in rejected:
            with self.subTest(raw=raw):
                with self.assertRaises(subject.TargetT0CanaryError):
                    subject.validate_slurm_visible_device_environment(
                        self._visible_device_environment(raw)
                    )
        for override in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
            environment = self._visible_device_environment("0,2,5,7")
            environment[override] = "0,1,2,3"
            with self.subTest(override=override), self.assertRaisesRegex(
                subject.TargetT0CanaryError, override
            ):
                subject.validate_slurm_visible_device_environment(environment)
        environment = self._visible_device_environment("0,2,5,7")
        environment["ACTION_REPR_T0_SLURM_ROCR_VISIBLE_DEVICES_SHA256"] = _sha("f")
        with self.assertRaisesRegex(subject.TargetT0CanaryError, "launcher audit"):
            subject.validate_slurm_visible_device_environment(environment)

    def test_slurm_visible_device_receipt_and_publication_replay_fail_closed(self):
        receipt = subject.validate_slurm_visible_device_environment(
            self._visible_device_environment("0,3,5,7")
        )
        for key, value in (
            ("raw", "0,3,5,6"),
            ("devices", ["0", "3", "5", "6"]),
            ("device_count", 8),
            ("sha256", _sha("f")),
            ("mapping_preserved", False),
            ("hip_visible_devices_nonempty", True),
            ("cuda_visible_devices_nonempty", True),
        ):
            tampered = dict(receipt)
            tampered[key] = value
            with self.subTest(key=key), self.assertRaises(
                subject.TargetT0CanaryError
            ):
                subject.validate_slurm_visible_device_receipt(tampered)
        source = SUBJECT_PATH.read_text(encoding="utf-8")
        publication = source.split(
            "def validate_published_t0_output", 1
        )[1].split("def _parse_args", 1)[0]
        self.assertIn("validate_slurm_visible_device_receipt", publication)

    def _active_route(self, kind: str, offset: float):
        middle = {}
        for block in subject.BLOCK_INDICES:
            value = torch.randn((1, 6, self.teacher_width), dtype=torch.float32)
            value[:, :2] = 0
            middle[block] = value.detach()
        return subject.g2a.ActionRepresentationRoute(
            kind=kind,
            optimizer_step=0,
            layout=self.layout,
            flow=(torch.randn((1, 6, 12)) + offset).detach(),
            activity=self.activity,
            middle_by_block=middle,
            representation_origin=(
                "real_target_frozen_extractor"
                if kind == "correct"
                else "counterfactual_control"
            ),
            representation_cache_sha256=str((int(offset) + 1) % 10) * 64,
            middle_value_kind="post_attention_residual",
            matched_noise_timestep_rotary=True,
        )

    def _routes(self):
        return {
            "correct": self._active_route("correct", 0.0),
            "zero": subject.g2a.ActionRepresentationRoute(
                kind="zero", optimizer_step=0, layout=self.layout
            ),
            "temporal_shuffle": self._active_route("temporal_shuffle", 1.0),
            "reverse": self._active_route("reverse", 2.0),
            "incomplete": self._active_route("incomplete", 3.0),
            "wrong_action": self._active_route("wrong_action", 4.0),
        }

    def test_one_real_step_changes_only_allowlist_and_keeps_bypasses(self):
        model = _Transformer(self.width).eval().requires_grad_(False)
        base_before = {
            name: value.detach().clone() for name, value in model.named_parameters()
        }
        expected_base_digest = subject.g2a_world4.renderer_base_snapshot(model).digest
        expected_native_digest = subject.g2a.tensor_sha256(model(self.hidden).detach())
        result = subject.run_one_step_optimizer_canary(
            model=model,
            forward_native=lambda: model(self.hidden),
            input_digest=lambda: _sha("a"),
            routes=self._routes(),
            feature_projection=self.projection,
            hidden_width=self.width,
            middle_width=self.teacher_width,
            bottleneck_width=4,
            learning_rate=0.05,
            expected_input_digest=_sha("a"),
            expected_base_digest=expected_base_digest,
            expected_native_output_digest=expected_native_digest,
        )
        self.assertTrue(result.facts["optimizer_created"])
        self.assertEqual(result.facts["optimization_steps"], 1)
        self.assertGreater(result.facts["parameter_updates"], 0)
        self.assertEqual(result.facts["control_gradient_pass_count"], 5)
        self.assertTrue(result.facts["all_five_control_gradient_passes_executed"])
        self.assertTrue(result.facts["all_gradients_finite_before_and_after_all_reduce"])
        self.assertTrue(result.facts["raw_gradients_all_reduced_before_global_pcgrad"])
        self.assertEqual(result.facts["global_pcgrad_and_norm_clip_count"], 1)
        self.assertTrue(result.facts["matched_production_g2a_source_batch"])
        self.assertTrue(result.facts["matched_production_g2a_renderer_base"])
        self.assertEqual(result.facts["middle_projector_parameter_updates"], 0)
        self.assertFalse(result.facts["middle_projector_trained_claimed"])
        self.assertEqual(
            set(result.facts["updated_parameter_names"]),
            {
                f"blocks.{index}.{subject.g2a.MODULE_NAME}.motion_adapter.output.weight"
                for index in subject.BLOCK_INDICES
            },
        )
        self.assertTrue(all(
            row["gradient_pass_executed"]
            and row["hinge_active_all_four_blocks"]
            and row["raw_gradient_all_reduced_before_pcgrad"]
            for row in result.facts["control_gradient_passes"].values()
        ))
        self.assertTrue(result.facts["renderer_base_identity_versions_bytes_unchanged"])
        self.assertTrue(result.facts["route_off_step1_exact_native"])
        self.assertTrue(result.facts["zero_step1_exact_native"])
        self.assertTrue(result.facts["correct_step1_internal_residual_nonzero"])
        self.assertTrue(result.facts["correct_step1_post_head_changed"])
        self.assertNotEqual(
            result.facts["step0_state"]["state_digest"],
            result.facts["step1_state"]["state_digest"],
        )
        self.assertTrue(all(
            torch.equal(base_before[name], value)
            for name, value in model.named_parameters()
        ))
        self.assertTrue(all(
            not hasattr(block, subject.g2a.MODULE_NAME) for block in model.blocks
        ))

    def test_production_g2a_batch_and_base_are_bound_before_adapter_install(self):
        model = _Transformer(self.width).eval().requires_grad_(False)
        expected_base_digest = subject.g2a_world4.renderer_base_snapshot(model).digest
        expected_native_digest = subject.g2a.tensor_sha256(model(self.hidden).detach())
        with self.assertRaisesRegex(subject.TargetT0CanaryError, "FM batch differs"):
            subject.run_one_step_optimizer_canary(
                model=model,
                forward_native=lambda: model(self.hidden),
                input_digest=lambda: _sha("a"),
                routes=self._routes(),
                feature_projection=self.projection,
                hidden_width=self.width,
                middle_width=self.teacher_width,
                bottleneck_width=4,
                expected_input_digest=_sha("b"),
                expected_base_digest=expected_base_digest,
                expected_native_output_digest=expected_native_digest,
            )
        self.assertTrue(all(
            not hasattr(block, subject.g2a.MODULE_NAME) for block in model.blocks
        ))

        with self.assertRaisesRegex(subject.TargetT0CanaryError, "renderer base differs"):
            subject.run_one_step_optimizer_canary(
                model=model,
                forward_native=lambda: model(self.hidden),
                input_digest=lambda: _sha("a"),
                routes=self._routes(),
                feature_projection=self.projection,
                hidden_width=self.width,
                middle_width=self.teacher_width,
                bottleneck_width=4,
                expected_input_digest=_sha("a"),
                expected_base_digest=_sha("b"),
                expected_native_output_digest=expected_native_digest,
            )
        self.assertTrue(all(
            not hasattr(block, subject.g2a.MODULE_NAME) for block in model.blocks
        ))

        with self.assertRaisesRegex(subject.TargetT0CanaryError, "native post-head"):
            subject.run_one_step_optimizer_canary(
                model=model,
                forward_native=lambda: model(self.hidden),
                input_digest=lambda: _sha("a"),
                routes=self._routes(),
                feature_projection=self.projection,
                hidden_width=self.width,
                middle_width=self.teacher_width,
                bottleneck_width=4,
                expected_input_digest=_sha("a"),
                expected_base_digest=expected_base_digest,
                expected_native_output_digest=_sha("b"),
            )
        self.assertTrue(all(
            not hasattr(block, subject.g2a.MODULE_NAME) for block in model.blocks
        ))

    @unittest.skipUnless(
        importlib.util.find_spec("safetensors") is not None,
        "safetensors unavailable",
    )
    def test_create_only_publication_is_fully_replayed_and_tamper_closed(self):
        model = _Transformer(self.width).eval().requires_grad_(False)
        base_digest = subject.g2a_world4.renderer_base_snapshot(model).digest
        native_digest = subject.g2a.tensor_sha256(model(self.hidden).detach())
        result = subject.run_one_step_optimizer_canary(
            model=model,
            forward_native=lambda: model(self.hidden),
            input_digest=lambda: _sha("a"),
            routes=self._routes(),
            feature_projection=self.projection,
            hidden_width=self.width,
            middle_width=self.teacher_width,
            bottleneck_width=4,
            expected_input_digest=_sha("a"),
            expected_base_digest=base_digest,
            expected_native_output_digest=native_digest,
        )
        result = subject.OneStepResult(
            step0_state=result.step0_state,
            step1_state=result.step1_state,
            facts={**dict(result.facts), "world_size": 4},
        )
        authority = subject.PreoptimizerAuthority(
            case=subject.FixedFitCase(
                manifest_path=Path("/fixed/manifest.json"),
                manifest_sha256=_sha("1"),
                case_id=subject.FIXED_CASE_ID,
                instruction="fixed instruction",
                seed=1,
                source_path=Path("/fixed/source.mp4"),
                source_sha256=_sha("2"),
            ),
            g1=None,
            g1_receipt_sha256=_sha("3"),
            g2a_path=Path("/fixed/g2a.json"),
            g2a_file_sha256=_sha("4"),
            g2a_receipt={
                "receipt_digest": _sha("5"),
                "source_owned_native_input": {
                    "matched_native_batch_sha256": _sha("a")
                },
                "parameter_firewall": {
                    "renderer_base_snapshot_digest_before": base_digest,
                    "native_post_head_tensor_sha256": native_digest,
                },
            },
            sigma_index=1,
            authorization_path=Path("/fixed/authority.json"),
            authorization_sha256=_sha("6"),
            authorization={"source_hash_pins_digest": _sha("d")},
        )
        projection = {
            "kind": "case_independent_fixed_rademacher_jl",
            "seed": 2026082401,
            "width": 256,
            "student_native_width": subject.HIDDEN_WIDTH,
            "student_projection_applied_differentiably": True,
            "sha256": _sha("7"),
            "upstream_receipt_sha256": _sha("8"),
        }
        runtime = {
            "world_size": 4,
            "ulysses_size": 4,
            "backend": "nccl/rccl",
            "slurm_rocr_visible_devices": {
                "raw": "4,5,6,7",
                "devices": ["4", "5", "6", "7"],
                "device_count": 4,
                "sha256": hashlib.sha256(b"4,5,6,7").hexdigest(),
                "mapping_preserved": True,
                "hip_visible_devices_nonempty": False,
                "cuda_visible_devices_nonempty": False,
            },
            "exact_transformer_block_count": 30,
            "bernini_revision": "fixed",
            "veomni_revision": "fixed",
            "checkpoint_tree_sha256": _sha("9"),
            "source_owned_native_batch": True,
            "selected_sigma_index": 1,
            "selected_sigma": 0.5,
            "patch_grid": [5, 2, 3],
            "route_facts_digest": _sha("b"),
            "source_posterior_tensor_sha256": _sha("c"),
        }
        source_lock = {
            name: _sha(str((index + 1) % 10))
            for index, name in enumerate(
                (
                    "train_action_repr_target_t0_canary_retry6_v1.py",
                    "action_repr_g2a_adapter_v1.py",
                    "action_representation_joint_objective_v1.py",
                    "audit_action_repr_g2a_world4_v1.py",
                    "materialize_decoded_middle_action_repr_v1.py",
                    "dense_flow_token_adapter_v1.py",
                    "exact_local_video_materializer_v1.py",
                    "train_lora.py",
                    "train_self_generated_action_quotient_v1.py",
                )
            )
        }
        with tempfile.TemporaryDirectory(dir=str(RESOLVED_TEMP_ROOT)) as temporary:
            output = Path(temporary) / "single_update"
            runtime["preoptimizer_attempt_claim"] = self._create_claim(output)
            subject.publish_create_only_result(
                output=output,
                result=result,
                authority=authority,
                projection=projection,
                runtime=runtime,
                source_lock=source_lock,
            )
            published = subject.validate_published_t0_output(output)
            self.assertEqual(published["optimization_steps"], 1)
            with self.assertRaisesRegex(subject.TargetT0CanaryError, "create-only"):
                subject.publish_create_only_result(
                    output=output,
                    result=result,
                    authority=authority,
                    projection=projection,
                    runtime=runtime,
                    source_lock=source_lock,
                )
            state_path = output / "step0001" / "adapter_model.safetensors"
            payload = bytearray(state_path.read_bytes())
            payload[-1] ^= 1
            state_path.write_bytes(payload)
            with self.assertRaisesRegex(subject.TargetT0CanaryError, "file SHA-256"):
                subject.validate_published_t0_output(output)

    def test_heldout_manifest_is_rejected(self):
        manifest = {
            "schema_version": "mev-target-selfgen-flow-calibration-manifest-v1",
            "splits": {"fit": [], "heldout": [subject.FIXED_CASE_ID]},
            "cases": [
                {
                    "case_id": subject.FIXED_CASE_ID,
                    "split": "heldout",
                    "instruction": "x",
                    "seed": 1,
                    "source": {"path": "/not-opened", "sha256": _sha("a")},
                }
            ],
        }
        with tempfile.TemporaryDirectory(dir=str(RESOLVED_TEMP_ROOT)) as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="ascii")
            with self.assertRaisesRegex(subject.TargetT0CanaryError, "heldout"):
                subject.load_fixed_fit_case(path)

    def test_target_media_fields_are_rejected_before_optimizer(self):
        for key in ("target_video_path", "target_latent", "anchor_video_path"):
            with self.subTest(key=key):
                with self.assertRaisesRegex(
                    subject.TargetT0CanaryError, "forbidden target/anchor media"
                ):
                    subject.reject_forbidden_media_fields(
                        {"nested": {key: "/not-opened"}}, label="tampered receipt"
                    )

    def test_g1_and_g2a_tamper_paths_fail_closed(self):
        manifest = {
            "schema_version": "mev-target-selfgen-flow-calibration-manifest-v1",
            "splits": {"fit": [subject.FIXED_CASE_ID], "heldout": []},
            "cases": [{
                "case_id": subject.FIXED_CASE_ID,
                "split": "fit",
                "instruction": "one fit instruction",
                "seed": 1,
                "source": {"path": "/not-reached", "sha256": _sha("a")},
            }],
        }
        with tempfile.TemporaryDirectory(dir=str(RESOLVED_TEMP_ROOT)) as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            g1_path = root / "g1.json"
            g2a_path = root / "g2a.json"
            addendum_path = root / "authorization.json"
            manifest_path.write_text(json.dumps(manifest), encoding="ascii")
            g1_path.write_text("{}", encoding="ascii")
            g2a_path.write_text("{}", encoding="ascii")
            addendum_path.write_text("{}", encoding="ascii")
            with mock.patch.object(
                subject.g2a_world4,
                "resolve_target_g1_authority",
                side_effect=RuntimeError("tampered"),
            ), mock.patch.object(subject.torch.optim, "AdamW") as optimizer:
                with self.assertRaisesRegex(subject.TargetT0CanaryError, "G1"):
                    subject.authorize_preoptimizer_inputs(
                        manifest=manifest_path,
                        g1_admission_receipt=g1_path,
                        g2a_receipt=g2a_path,
                        authorization_addendum=addendum_path,
                        bernini_root="/b",
                        veomni_root="/v",
                        checkpoint="/c",
                        output=subject.EXPECTED_CANONICAL_OUTPUT_PATH,
                    )
                optimizer.assert_not_called()
            fake_authority = types.SimpleNamespace(
                admission_sha256=subject.file_sha256(g1_path)
            )
            with mock.patch.object(
                subject.g2a_world4,
                "resolve_target_g1_authority",
                return_value=fake_authority,
            ), mock.patch.object(
                subject.g2a_world4,
                "validate_world4_receipt",
                side_effect=RuntimeError("tampered G2a"),
            ), mock.patch.object(subject.torch.optim, "AdamW") as optimizer:
                with self.assertRaisesRegex(subject.TargetT0CanaryError, "G2a"):
                    subject.authorize_preoptimizer_inputs(
                        manifest=manifest_path,
                        g1_admission_receipt=g1_path,
                        g2a_receipt=g2a_path,
                        authorization_addendum=addendum_path,
                        bernini_root="/b",
                        veomni_root="/v",
                        checkpoint="/c",
                        output=subject.EXPECTED_CANONICAL_OUTPUT_PATH,
                    )
                optimizer.assert_not_called()

    def test_cli_surface_and_main_matched_pair_call_are_static_closed(self):
        required = [
            "--authorization-addendum", "/a",
            "--manifest", "/m",
            "--g1-admission-receipt", "/g1",
            "--g2a-receipt", "/g2a",
            "--bernini-root", "/b",
            "--veomni-root", "/v",
            "--checkpoint", "/c",
            "--output", "/o",
        ]
        parsed = subject._parse_args(required)
        self.assertEqual(parsed.output, "/o")
        with self.assertRaises(SystemExit):
            subject._parse_args(required + ["--target-video", "/forbidden"])
        tree = ast.parse(SUBJECT_PATH.read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "recover_matched_patch_pair"
        ]
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0].args), 2)

    def test_authority_validator_accepts_only_fresh_retry6_source_revision(self):
        source = SUBJECT_PATH.read_text(encoding="utf-8")
        self.assertIn(
            '"methods/bernini_action_editing/scripts/'
            'auh_stage_b_t0_single_update_20260824_retry6.sh"',
            source,
        )
        self.assertIn(
            'runtime_paths.get("fresh_source_root_name") '
            '!= "source_stage_b_t0_retry6"',
            source,
        )
        self.assertIn(
            'runtime_paths.get("fresh_stage_root_name") != "stage_b_t0_retry6"',
            source,
        )
        self.assertIn(
            'runtime_paths.get("fresh_log_root_name") != "logs/stage_b_t0_retry6"',
            source,
        )
        self.assertNotIn("source_stage_b_t0_retry3", source)
        self.assertNotIn("stage_b_t0_retry3", source)

    def test_active_retry6_addendum_replays_through_runner_validator(self):
        repository = METHOD_ROOT.parents[1]
        addendum_path = (
            repository
            / "md"
            / "action_editing"
            / "20260824_reward"
            / "stage_b_t0_single_update_retry6_authority_addendum.json"
        )
        addendum = json.loads(addendum_path.read_text(encoding="utf-8"))
        upstream = addendum["upstream_gate_evidence"]
        runtime = addendum["runtime_paths"]
        projection_authority = addendum["representation_contract"]["fixed_jl"]
        case = subject.FixedFitCase(
            manifest_path=Path(upstream["manifest"]["path"]),
            manifest_sha256=upstream["manifest"]["sha256"],
            case_id=subject.FIXED_CASE_ID,
            instruction="not consumed by addendum validation",
            seed=1,
            source_path=Path("/not/consumed/by/addendum/validation.mp4"),
            source_sha256=_sha("a"),
        )
        validated = subject.validate_authorization_addendum(
            addendum,
            addendum_path=addendum_path,
            case=case,
            g1_path=Path(upstream["g1_target"]["path"]),
            g1_sha256=upstream["g1_target"]["receipt_sha256"],
            g2a_path=Path(upstream["production_g2a"]["path"]),
            g2a_sha256=upstream["production_g2a"]["receipt_sha256"],
            g2a_receipt_digest=upstream["production_g2a"]["receipt_digest"],
            projection={
                "kind": projection_authority["kind"],
                "seed": projection_authority["seed"],
                "width": projection_authority["output_width"],
                "sha256": projection_authority["tensor_sha256"],
            },
            bernini_root=runtime["bernini_root"],
            veomni_root=runtime["veomni_root"],
            checkpoint=runtime["checkpoint"],
            output=addendum["output_contract"]["canonical_output_path"],
        )
        self.assertIs(validated, addendum)


if __name__ == "__main__":
    unittest.main()
