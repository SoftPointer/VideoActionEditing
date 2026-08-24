from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / (
    "full644_target_free_bernini_runtime_v1.py"
)
SPEC = importlib.util.spec_from_file_location("full644_target_free_bernini_runtime_v1", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
runtime = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runtime
SPEC.loader.exec_module(runtime)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


class FakeScheduler:
    def __init__(self) -> None:
        self.step_index = None


class FakeSerialDist:
    def __init__(self, *, failed_owner=None) -> None:
        self.failed_owner = failed_owner
        self.gathered_owners = []
        self.barrier_owners = []
        self._last_owner = None

    @staticmethod
    def _failure_digest() -> str:
        return hashlib.sha256(b"RuntimeError\0boom").hexdigest()

    def all_gather_object(self, output, local_row, *, group) -> None:
        del group
        owner = local_row["owner_rank"]
        self._last_owner = owner
        self.gathered_owners.append(owner)
        for rank in range(runtime.WORLD_SIZE):
            row = dict(local_row)
            row.update(
                {
                    "reporter_rank": rank,
                    "role": "owner" if rank == owner else "waiter",
                    "status": "WAITING",
                    "evidence_digest": None,
                    "error_type": None,
                    "error_digest": None,
                }
            )
            if rank == owner:
                if local_row["reporter_rank"] == owner:
                    row = dict(local_row)
                elif owner == self.failed_owner:
                    row.update(
                        {
                            "status": "FAILED",
                            "evidence_digest": None,
                            "error_type": "RuntimeError",
                            "error_digest": self._failure_digest(),
                        }
                    )
                else:
                    row.update(
                        {
                            "status": "COMPLETE",
                            "evidence_digest": sha(
                                f"{local_row['phase']} owner {owner} evidence"
                            ),
                        }
                    )
            output[rank] = row

    def barrier(self, *, group) -> None:
        del group
        self.barrier_owners.append(self._last_owner)


class FakeDevice:
    def __init__(self, index: int) -> None:
        self.type = "cuda"
        self.index = index

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, FakeDevice)
            and self.type == other.type
            and self.index == other.index
        )


class FakeTensor:
    def __init__(self, shape, *, device, data=None) -> None:
        self.shape = tuple(shape)
        self.device = device
        self.dtype = "torch.float32"
        self.layout = "torch.strided"
        self.requires_grad = False
        count = 1
        for item in self.shape:
            count *= item
        self.data = list(data) if data is not None else [0.0] * count

    def is_contiguous(self) -> bool:
        return True

    def numel(self) -> int:
        return len(self.data)

    def detach(self):
        return self

    def clone(self, *, memory_format=None):
        del memory_format
        return FakeTensor(self.shape, device=self.device, data=self.data)


class FakeFinite:
    def all(self):
        return self

    def item(self) -> bool:
        return True


class FakeTorch:
    Tensor = FakeTensor
    float32 = "torch.float32"
    strided = "torch.strided"
    contiguous_format = "torch.contiguous_format"

    @staticmethod
    def empty(shape, *, dtype, device):
        if dtype != FakeTorch.float32:
            raise AssertionError("fake dtype differs")
        return FakeTensor(shape, device=device)

    @staticmethod
    def isfinite(value):
        if not isinstance(value, FakeTensor):
            raise AssertionError("fake finite input differs")
        return FakeFinite()


class FakeRank0TensorBroadcastDist:
    def __init__(self, *, rank, descriptor, root_tensor, corrupt=False) -> None:
        self.rank = rank
        self.descriptor = descriptor
        self.root_tensor = root_tensor
        self.corrupt = corrupt
        self.tensor_broadcast_count = 0

    def broadcast_object_list(self, output, *, src, group) -> None:
        del group
        if src != 0:
            raise AssertionError("fake descriptor source differs")
        if self.rank != 0:
            output[0] = {"ok": True, "descriptor": copy.deepcopy(self.descriptor)}

    def all_gather_object(self, output, local_row, *, group) -> None:
        del group
        replay = local_row["schema_version"].endswith("-replay-v1")
        for rank in range(runtime.WORLD_SIZE):
            if rank == self.rank:
                output[rank] = copy.deepcopy(local_row)
                continue
            row = {
                "schema_version": (
                    "bernini-full644-rank0-vae-source-replay-v1"
                    if replay
                    else "bernini-full644-rank0-vae-source-ready-v1"
                ),
                "world_rank": rank,
                "status": "REPLAYED" if replay else "READY",
                "device_index": rank,
                "descriptor_digest": self.descriptor["descriptor_digest"],
                "error_type": None,
                "error_digest": None,
            }
            if replay:
                row.update(
                    {
                        "source_state_shape": self.descriptor["source_state_shape"],
                        "source_state_dtype": "torch.float32",
                        "source_state_contiguous": True,
                        "source_state_requires_grad": False,
                        "source_state_sha256": self.descriptor["source_state_sha256"],
                    }
                )
            output[rank] = row

    def broadcast(self, tensor, *, src, group) -> None:
        del group
        if src != 0:
            raise AssertionError("fake tensor source differs")
        self.tensor_broadcast_count += 1
        tensor.data[:] = self.root_tensor.data
        if self.corrupt:
            tensor.data[-1] += 1.0


class ContractTests(unittest.TestCase):
    @staticmethod
    def _rollout_stage_rows_fixture(*, peaks=None, total=1000):
        if peaks is None:
            peaks = [101, 202, 303, 404, 111, 222, 333, 444]
        rows = []
        for rank in range(runtime.WORLD_SIZE):
            arm = rank // runtime.SP_SIZE
            prefix = f"/closed/arm{arm}"
            row = {
                "world_rank": rank,
                "dp_arm": arm,
                "sp_rank": rank % runtime.SP_SIZE,
                "rollout_id": f"rollout-arm{arm}",
                "rollout_seed": 700 + arm,
                "behavior_policy_sha256": sha("behavior policy"),
                "trajectory_receipt_path": f"{prefix}.trajectory.json",
                "trajectory_receipt_sha256": sha(f"trajectory receipt {arm}"),
                "trajectory_receipt_digest": sha(f"trajectory digest {arm}"),
                "trajectory_artifact_sha256": sha(f"trajectory artifact {arm}"),
                "terminal_state_sha256": sha(f"terminal state {arm}"),
                "decoded_rollout_receipt_path": f"{prefix}.decoded.json",
                "decoded_rollout_receipt_sha256": sha(f"decoded receipt {arm}"),
                "decoded_rollout_receipt_digest": sha(f"decoded digest {arm}"),
                "candidate_media_path": f"{prefix}.mp4",
                "candidate_media_sha256": sha(f"candidate media {arm}"),
                "candidate_full_decode_tree_digest": sha(f"decode tree {arm}"),
                "candidate_exact81_25fps": True,
                "peak_memory_allocated_bytes": peaks[rank],
                "total_device_memory_bytes": total,
            }
            rows.append({**row, "row_digest": runtime.object_sha256(row)})
        return rows

    @staticmethod
    def _resign_rollout_stage_row(row):
        unsigned = {key: value for key, value in row.items() if key != "row_digest"}
        row["row_digest"] = runtime.object_sha256(unsigned)

    def test_rollout_stage_aggregates_varying_rank_local_peaks(self) -> None:
        peaks = [101, 202, 303, 404, 111, 222, 333, 444]
        rows = self._rollout_stage_rows_fixture(peaks=peaks)
        arm_rows = runtime._aggregate_rollout_stage_world8_v1(rows)
        self.assertEqual(len(arm_rows), runtime.DP_SIZE)
        self.assertEqual(
            [row["peak_memory_allocated_bytes"] for row in arm_rows],
            [404, 444],
        )
        expected_fields = runtime._ROLLOUT_STAGE_LOCAL_FIELDS - {
            "world_rank",
            "sp_rank",
            "row_digest",
        }
        self.assertEqual(set(arm_rows[0]), expected_fields)
        self.assertEqual(arm_rows[0]["total_device_memory_bytes"], 1000)

        same_peak = self._rollout_stage_rows_fixture(peaks=[512] * runtime.WORLD_SIZE)
        same_result = runtime._aggregate_rollout_stage_world8_v1(same_peak)
        self.assertEqual(
            [row["peak_memory_allocated_bytes"] for row in same_result],
            [512, 512],
        )
        reloaded = json.loads(runtime.canonical_json_bytes(rows))
        self.assertEqual(
            runtime._aggregate_rollout_stage_world8_v1(reloaded), arm_rows
        )

    def test_rollout_stage_rejects_memory_and_semantic_hostiles(self) -> None:
        for label, field, value, expression in (
            ("peak_bool", "peak_memory_allocated_bytes", True, "memory evidence"),
            ("total_bool", "total_device_memory_bytes", True, "memory evidence"),
            ("peak_zero", "peak_memory_allocated_bytes", 0, "memory evidence"),
            ("peak_negative", "peak_memory_allocated_bytes", -1, "memory evidence"),
            ("peak_gt_total", "peak_memory_allocated_bytes", 1001, "memory evidence"),
            ("total_zero", "total_device_memory_bytes", 0, "memory evidence"),
            ("semantic", "rollout_id", "wrong-rollout", "arm0 differs"),
            ("total_mismatch", "total_device_memory_bytes", 999, "arm0 differs"),
        ):
            rows = self._rollout_stage_rows_fixture()
            rows[1][field] = value
            self._resign_rollout_stage_row(rows[1])
            with self.subTest(hostile=label), self.assertRaisesRegex(
                runtime.TargetFreeBerniniRuntimeError, expression
            ):
                runtime._aggregate_rollout_stage_world8_v1(rows)

        rows = self._rollout_stage_rows_fixture()
        rows[1]["world_rank"] = True
        self._resign_rollout_stage_row(rows[1])
        with self.assertRaisesRegex(
            runtime.TargetFreeBerniniRuntimeError, "placement differs"
        ):
            runtime._aggregate_rollout_stage_world8_v1(rows)
        rows = self._rollout_stage_rows_fixture()
        rows[1]["rollout_id"] = "unsigned-change"
        with self.assertRaisesRegex(
            runtime.TargetFreeBerniniRuntimeError, "digest differs"
        ):
            runtime._aggregate_rollout_stage_world8_v1(rows)

    @staticmethod
    def _rank0_vae_descriptor_fixture():
        source_row_id = "row-000"
        source_video_sha = sha("source-video")
        instruction_sha = sha("instruction")
        source_row_digest = sha("source-row")
        catalog = {
            "catalog_sha256": sha("catalog-bytes"),
            "catalog_digest": sha("catalog-object"),
            "binding_digest": sha("catalog-binding"),
        }
        authority = {
            "schema_version": "bernini-full644-owned-vae-authority-v1",
            "base_checkpoint_tree_sha256": sha("checkpoint-tree"),
            "checkpoint_content_manifest_sha256": sha("checkpoint-manifest"),
            "checkpoint_snapshot_digest": sha("checkpoint-snapshot"),
            "vae_file_inventory_digest": sha("vae-files"),
            "vae_config_sha256": sha("vae-config"),
        }
        decode_unsigned = {
            "schema_version": "bernini-full644-owned-source-decode-v1",
            "source_row_id": source_row_id,
            "source_video_sha256": source_video_sha,
            "frame_count": runtime.FRAME_COUNT,
            "fps_float64_hex": runtime.FPS.hex(),
            "source_derived_bucket_hw": [16, 24],
            "target_media_read_count": 0,
        }
        decode = {
            **decode_unsigned,
            "decode_digest": runtime.object_sha256(decode_unsigned),
        }
        shape = [1, 16, 21, 2, 3]
        descriptor_unsigned = {
            "schema_version": "bernini-full644-rank0-vae-source-broadcast-v1",
            "producer_rank": 0,
            "source_row_id": source_row_id,
            "source_video_sha256": source_video_sha,
            "instruction_sha256": instruction_sha,
            "source_row_digest": source_row_digest,
            "catalog_sha256": catalog["catalog_sha256"],
            "catalog_digest": catalog["catalog_digest"],
            "catalog_binding_digest": catalog["binding_digest"],
            "source_decode": decode,
            "source_decode_digest": decode["decode_digest"],
            "source_state_shape": shape,
            "source_state_numel": 1 * 16 * 21 * 2 * 3,
            "source_state_dtype": "torch.float32",
            "source_state_layout": "torch.strided",
            "source_state_contiguous": True,
            "source_state_requires_grad": False,
            "source_state_sha256": sha("rank0-source-state"),
            "vae_authority": authority,
            "vae_authority_digest": runtime.object_sha256(authority),
            "rank0_only_source_decode_and_vae_encode": True,
            "vae_released_without_cpu_rematerialization": True,
            "host_allocator_trim": {
                "allocator": "glibc_malloc_trim",
                "called": True,
                "return_code": 1,
            },
        }
        descriptor = {
            **descriptor_unsigned,
            "descriptor_digest": runtime.object_sha256(descriptor_unsigned),
        }
        expected = {
            "expected_source_row_id": source_row_id,
            "expected_source_video_sha256": source_video_sha,
            "expected_instruction_sha256": instruction_sha,
            "expected_source_row_digest": source_row_digest,
            "expected_catalog_binding": catalog,
            "expected_vae_authority": authority,
        }
        return descriptor, expected

    @staticmethod
    def _real_catalog_release_schema_fixtures():
        receipt = {
            "schema_version": "bernini-full644-target-free-source-catalog-receipt-v1",
            "status": "SOURCE_ONLY_EXACT644_CATALOG_COMPLETE",
            "receipt_digest": runtime.FULL644_CATALOG_RECEIPT_DIGEST,
            "catalog_sha256": runtime.FULL644_CATALOG_SHA256,
            "catalog_size": runtime.FULL644_CATALOG_SIZE,
            "catalog_digest": runtime.FULL644_CATALOG_DIGEST,
            "extractor_self_sha256": runtime.FULL644_CATALOG_EXTRACTOR_SHA256,
            "extractor_self_size": runtime.FULL644_CATALOG_EXTRACTOR_SIZE,
            "source_count": 644,
            "target_media_used": False,
            "paired_edited_target_present": False,
            "ffprobe_binding": {
                "sha256": runtime.FULL644_CATALOG_FFPROBE_SHA256,
                "size": runtime.FULL644_CATALOG_FFPROBE_SIZE,
                "mode": 0o555,
                "nlink": 1,
                "held_fd_execution": True,
            },
            "ffprobe_held_fd_prepost_replay_verified": True,
        }
        postflight = {
            "schema_version": "bernini-full644-source-catalog-external-postflight-v1",
            "status": "SOURCE_ONLY_EXACT644_CATALOG_POSTFLIGHT_COMPLETE",
            "complete": True,
            "release_digest": runtime.FULL644_CATALOG_POSTFLIGHT_DIGEST,
            "downstream_source_rehash_required": True,
            "external_release_sha_must_be_pinned_before_consumption": True,
            "controller": {
                "sha256": runtime.FULL644_CATALOG_CONTROLLER_SHA256,
                "size": runtime.FULL644_CATALOG_CONTROLLER_SIZE,
            },
            "controller_pre_admission": {
                "sha256": runtime.FULL644_CATALOG_PRE_ADMISSION_SHA256,
                "size": runtime.FULL644_CATALOG_PRE_ADMISSION_SIZE,
            },
            "producer": {
                "source_count": 644,
                "extractor_sha256": runtime.FULL644_CATALOG_EXTRACTOR_SHA256,
                "exact_member_closure": [
                    "source_catalog.json", "source_catalog_receipt.json"
                ],
                "catalog": {
                    "sha256": runtime.FULL644_CATALOG_SHA256,
                    "size": runtime.FULL644_CATALOG_SIZE,
                    "manifest_digest": runtime.FULL644_CATALOG_DIGEST,
                },
                "receipt": {
                    "sha256": runtime.FULL644_CATALOG_RECEIPT_SHA256,
                    "size": runtime.FULL644_CATALOG_RECEIPT_SIZE,
                    "receipt_digest": runtime.FULL644_CATALOG_RECEIPT_DIGEST,
                },
            },
            "authority": {
                "catalog_integrity_release": True,
                "paired_edited_target_present": False,
                "trainer_launched": False,
                "training_runtime_authorized": False,
                "upstream_training_use_forbidden": True,
            },
            "trace": {
                "held_ffprobe_exec_success_count": 644,
                "held_ffprobe_exit0_count": 644,
                "source_inventory_count": 644,
                "source_path_seen_count": 644,
                "target_exact_path_seen_count": 0,
            },
        }
        return receipt, postflight

    def test_real_catalog_postflight_locates_ffprobe_only_in_receipt(self) -> None:
        receipt, postflight = self._real_catalog_release_schema_fixtures()
        postflight_payload = runtime.canonical_json_bytes(postflight)
        self.assertNotIn(
            runtime.FULL644_CATALOG_FFPROBE_SHA256,
            postflight_payload.decode("ascii"),
        )
        projection = runtime._validate_frozen_catalog_release_envelopes_v1(
            runtime.canonical_json_bytes(receipt), postflight_payload
        )
        self.assertEqual(
            projection["ffprobe_sha256_from_receipt_binding"],
            runtime.FULL644_CATALOG_FFPROBE_SHA256,
        )
        hostile_receipt = json.loads(json.dumps(receipt))
        hostile_receipt["ffprobe_binding"]["sha256"] = sha("wrong ffprobe")
        with self.assertRaisesRegex(
            runtime.TargetFreeBerniniRuntimeError,
            "catalog receipt authority schema",
        ):
            runtime._validate_frozen_catalog_release_envelopes_v1(
                runtime.canonical_json_bytes(hostile_receipt), postflight_payload
            )
        moved_receipt = json.loads(json.dumps(receipt))
        moved = moved_receipt.pop("ffprobe_binding")
        moved_postflight = json.loads(json.dumps(postflight))
        moved_postflight["ffprobe_binding"] = moved
        with self.assertRaisesRegex(
            runtime.TargetFreeBerniniRuntimeError,
            "catalog release nested schema",
        ):
            runtime._validate_frozen_catalog_release_envelopes_v1(
                runtime.canonical_json_bytes(moved_receipt),
                runtime.canonical_json_bytes(moved_postflight),
            )
        hostile_postflight = json.loads(json.dumps(postflight))
        hostile_postflight["controller"]["sha256"] = sha("wrong controller")
        with self.assertRaisesRegex(
            runtime.TargetFreeBerniniRuntimeError,
            "catalog postflight authority schema",
        ):
            runtime._validate_frozen_catalog_release_envelopes_v1(
                runtime.canonical_json_bytes(receipt),
                runtime.canonical_json_bytes(hostile_postflight),
            )

    def test_path_only_cli_has_no_opaque_provider_or_optimizer_seam(self) -> None:
        parser = runtime.build_argument_parser_v1()
        parsed = parser.parse_args(
            [
                "update",
                "--bernini-root", "/b",
                "--veomni-root", "/v",
                "--checkpoint", "/c",
                "--checkpoint-content-manifest", "/m",
                "--output", "/o",
                "--preference-set", "/p",
                "--expected-preference-set-sha256", sha("preference"),
            ]
        )
        self.assertEqual(
            set(vars(parsed)),
            {
                "stage", "bernini_root", "veomni_root", "checkpoint",
                "checkpoint_content_manifest", "output", "preference_set",
                "expected_preference_set_sha256",
            },
        )
        signature = inspect.signature(runtime.BerniniExact40PolicyV1)
        self.assertNotIn("logprob_provider", signature.parameters)
        self.assertNotIn("optimizer_factory", signature.parameters)
        self.assertFalse(hasattr(runtime, "engineering_one_update_from_paths_v1"))

    def test_output_candidate_has_no_preinit_torchrun_freshness_race(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            candidate = root / "rank0-may-have-created-this"
            candidate.mkdir()
            # Every rank may reach CLI validation after rank 0's mkdir.  The
            # owned rank-0 transaction, not this pre-init parser, owns the
            # create-only rejection for a pre-existing directory.
            self.assertEqual(runtime._cli_fresh_output_v1(str(candidate)), candidate)
        source = inspect.getsource(runtime._cli_fresh_output_v1)
        self.assertNotIn("path.exists()", source)

    def test_frozen_qwen_qualification_is_derived_from_release_literals(self) -> None:
        authority = runtime._frozen_qwen_authority_v1()
        qualification = runtime._expected_qwen_qualification_v1(authority)
        self.assertEqual(
            qualification["verifier_release_sha256"],
            runtime.QWEN_VERIFIER_SOURCE_SHA256,
        )
        self.assertEqual(
            qualification["verifier_model_sha256"],
            runtime.QWEN_MODEL_CLOSURE_SHA256,
        )
        self.assertEqual(qualification["hard_axis_conjunction"], list(runtime.HARD_AXES))
        hostile = runtime.EngineeringVerifierAuthorityV1(
            source_path=authority.source_path,
            source_sha256=sha("alternate verifier"),
            source_size_bytes=authority.source_size_bytes,
            model_closure_path=authority.model_closure_path,
            model_closure_sha256=authority.model_closure_sha256,
            model_closure_size_bytes=authority.model_closure_size_bytes,
            model_revision=authority.model_revision,
        )
        with self.assertRaisesRegex(
            runtime.TargetFreeBerniniRuntimeError, "qualification authority"
        ):
            runtime._expected_qwen_qualification_v1(hostile)

    def test_world8_device_projection_accepts_zero_to_seven_only(self) -> None:
        for rank in range(runtime.WORLD_SIZE):
            row = runtime.validate_world8_device_placement_v1(
                world_rank=rank,
                local_rank=rank,
                dp_arm=rank // runtime.SP_SIZE,
                sp_rank=rank % runtime.SP_SIZE,
                device_index=rank,
            )
            self.assertEqual(row["device_index"], rank)
        with self.assertRaisesRegex(
            runtime.TargetFreeBerniniRuntimeError, "device placement"
        ):
            runtime.validate_world8_device_placement_v1(
                world_rank=7, local_rank=7, dp_arm=1, sp_rank=3, device_index=0
            )

    def test_pilot_source_contains_no_forbidden_supervision_seam(self) -> None:
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("paired_target", source)
        self.assertNotIn("frozen_velocity", source)
        self.assertNotIn("teacher_forcing", source)
        self.assertIn("exact40", source)
        self.assertIn("source_only_input", source)

    def test_held_git_and_exact_environment_literals(self) -> None:
        self.assertEqual(runtime.GIT_EXECUTABLE, Path("/usr/bin/git"))
        self.assertEqual(
            runtime.GIT_EXECUTABLE_SHA256,
            "fd7c9389e200d626b46551835e5233bbde49a6a2326f9ebb85c70ed235861001",
        )
        self.assertNotEqual(
            runtime.GIT_EXECUTABLE_SHA256,
            "5a39a7909c023f92a84b77b49e6b008f3f152b833135b96d73ac7c403314a88a",
        )
        self.assertEqual(runtime.GIT_EXECUTABLE_SIZE, 3_710_360)
        self.assertEqual(runtime.GIT_EXECUTABLE_MODE, 0o755)
        self.assertEqual(runtime.TORCH_VERSION, "2.7.1+rocm6.3")
        self.assertEqual(runtime.TRANSFORMERS_VERSION, "5.5.4")
        self.assertEqual(runtime.PEFT_VERSION, "0.19.1")
        self.assertEqual(runtime.DIFFUSERS_VERSION, "0.38.0")
        snapshot_source = inspect.getsource(runtime._snapshot_python_source_tree_v1)
        self.assertNotIn(
            '"exact_tracked_package_bytes_from_committed_archives": True',
            snapshot_source,
        )
        self.assertIn("bernini_projected_from_frozen_no_git_snapshot", snapshot_source)
        self.assertIn("veomni_projected_from_exact_git_archive_head", snapshot_source)

    def test_shared_snapshot_replay_is_metadata_only_and_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            snapshot_root = root / "shared-snapshot"
            snapshot_root.mkdir()
            member = snapshot_root / "member.bin"
            payload = b"rank0 already authenticated these bytes"
            member.write_bytes(payload)
            member.chmod(0o444)
            snapshot_root.chmod(0o555)
            unsigned = {
                "schema_version": "test-private-snapshot-v1",
                "destination": str(snapshot_root),
                "file_count": 1,
                "files": [
                    {
                        "path": "member.bin",
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "size_bytes": len(payload),
                        "mode_octal": "0444",
                    }
                ],
            }
            value = {
                **unsigned,
                "snapshot_digest": runtime.object_sha256(unsigned),
            }
            try:
                with mock.patch.object(
                    runtime,
                    "read_stable_file",
                    side_effect=AssertionError("snapshot contents were reread"),
                ):
                    replay = runtime._verify_projected_snapshot_metadata_v1(value)
                self.assertFalse(replay["content_bytes_reread"])
                self.assertEqual(replay["file_count"], 1)

                snapshot_root.chmod(0o700)
                extra = snapshot_root / "extra-directory"
                extra.mkdir()
                extra.chmod(0o555)
                snapshot_root.chmod(0o555)
                with self.assertRaisesRegex(
                    runtime.TargetFreeBerniniRuntimeError,
                    "physical file/directory set",
                ):
                    runtime._verify_projected_snapshot_metadata_v1(value)
            finally:
                snapshot_root.chmod(0o700)
                extra = snapshot_root / "extra-directory"
                if extra.exists():
                    extra.chmod(0o700)

    def test_fake_world8_serial_factory_order_peak_and_liveness(self) -> None:
        for rank in range(runtime.WORLD_SIZE):
            fake = FakeSerialDist()
            calls = []
            active = 0
            peak = 0
            owners = []
            for owner in range(runtime.WORLD_SIZE):
                def construct(owner=owner):
                    nonlocal active, peak
                    active += 1
                    peak = max(peak, active)
                    calls.append(owner)
                    active -= 1
                    return {"owner": owner}, {"owner": owner, "closed": True}

                payload, status = runtime._run_world8_serial_owner_round_v1(
                    dist=fake,
                    group="world8",
                    rank=rank,
                    owner_rank=owner,
                    phase="renderer_lora_construct",
                    constructor=construct,
                    payload_validator=lambda value, owner=owner: self.assertEqual(
                        value, {"owner": owner}
                    ),
                )
                owners.append(status["owner_rank"])
                self.assertEqual(payload is not None, rank == owner)
            self.assertEqual(calls, [rank])
            self.assertEqual(peak, 1)
            self.assertEqual(owners, list(range(runtime.WORLD_SIZE)))
            self.assertEqual(fake.gathered_owners, list(range(runtime.WORLD_SIZE)))
            self.assertEqual(fake.barrier_owners, list(range(runtime.WORLD_SIZE)))

    def test_fake_world8_serial_owner_failure_is_uniform_before_barrier(self) -> None:
        messages = []
        for rank in (0, 3, 7):
            fake = FakeSerialDist(failed_owner=3)

            def construct():
                raise RuntimeError("boom")

            with self.assertRaises(runtime.TargetFreeBerniniRuntimeError) as caught:
                runtime._run_world8_serial_owner_round_v1(
                    dist=fake,
                    group="world8",
                    rank=rank,
                    owner_rank=3,
                    phase="renderer_lora_construct",
                    constructor=construct,
                    payload_validator=lambda value: None,
                )
            messages.append(str(caught.exception))
            self.assertEqual(fake.gathered_owners, [3])
            self.assertEqual(fake.barrier_owners, [])
        self.assertEqual(messages, [messages[0]] * len(messages))
        self.assertIn(FakeSerialDist._failure_digest(), messages[0])

    def test_rank0_vae_constructor_executes_exactly_once_world8(self) -> None:
        calls = []
        for rank in range(runtime.WORLD_SIZE):
            fake = FakeSerialDist()

            def construct(rank=rank):
                calls.append(rank)
                return {"owner": 0}, {"rank0_only": True}

            payload, status = runtime._run_rank0_vae_source_constructor_v1(
                dist=fake,
                group="world8",
                rank=rank,
                constructor=construct,
                payload_validator=lambda value: self.assertEqual(
                    value, {"owner": 0}
                ),
            )
            self.assertEqual(payload is not None, rank == 0)
            self.assertEqual(status["owner_rank"], 0)
            self.assertEqual(status["phase"], "rank0_vae_source_encode")
            self.assertEqual(fake.gathered_owners, [0])
            self.assertEqual(fake.barrier_owners, [0])
        self.assertEqual(calls, [0])

    def test_rank0_vae_descriptor_and_world8_replay_are_fail_closed(self) -> None:
        descriptor, expected = self._rank0_vae_descriptor_fixture()
        validated = runtime._validate_rank0_vae_source_descriptor_v1(
            descriptor, **expected
        )
        self.assertEqual(validated["producer_rank"], 0)
        rows = []
        for rank in range(runtime.WORLD_SIZE):
            rows.append(
                {
                    "schema_version": "bernini-full644-rank0-vae-source-replay-v1",
                    "world_rank": rank,
                    "status": "REPLAYED",
                    "device_index": rank,
                    "descriptor_digest": descriptor["descriptor_digest"],
                    "source_state_shape": descriptor["source_state_shape"],
                    "source_state_dtype": "torch.float32",
                    "source_state_contiguous": True,
                    "source_state_requires_grad": False,
                    "source_state_sha256": descriptor["source_state_sha256"],
                    "error_type": None,
                    "error_digest": None,
                }
            )
        self.assertEqual(
            len(runtime._validate_world8_rank0_vae_rows_v1(
                rows, descriptor=descriptor, replay=True
            )),
            runtime.WORLD_SIZE,
        )
        for field, hostile_value in (
            ("producer_rank", 1),
            ("source_video_sha256", sha("altered-source-video")),
        ):
            hostile = copy.deepcopy(descriptor)
            hostile[field] = hostile_value
            unsigned = dict(hostile)
            unsigned.pop("descriptor_digest")
            hostile["descriptor_digest"] = runtime.object_sha256(unsigned)
            with self.assertRaises(runtime.TargetFreeBerniniRuntimeError):
                runtime._validate_rank0_vae_source_descriptor_v1(
                    hostile, **expected
                )
        hostile_rows = copy.deepcopy(rows)
        hostile_rows[5]["source_state_sha256"] = sha("rank5-altered")
        with self.assertRaises(runtime.TargetFreeBerniniRuntimeError):
            runtime._validate_world8_rank0_vae_rows_v1(
                hostile_rows, descriptor=descriptor, replay=True
            )

    def test_rank0_tensor_broadcast_clones_exact_and_rejects_nonroot_or_corruption(self) -> None:
        descriptor, expected = self._rank0_vae_descriptor_fixture()
        root = FakeTensor(
            descriptor["source_state_shape"],
            device=FakeDevice(0),
            data=[float(index) for index in range(descriptor["source_state_numel"])],
        )

        def fake_tensor_sha(value):
            return runtime.object_sha256(
                {"shape": list(value.shape), "data": value.data}
            )

        descriptor_unsigned = dict(descriptor)
        descriptor_unsigned.pop("descriptor_digest")
        descriptor_unsigned["source_state_sha256"] = fake_tensor_sha(root)
        descriptor = {
            **descriptor_unsigned,
            "descriptor_digest": runtime.object_sha256(descriptor_unsigned),
        }
        with mock.patch.object(runtime, "tensor_sha256", side_effect=fake_tensor_sha):
            for rank in (0, 5):
                fake = FakeRank0TensorBroadcastDist(
                    rank=rank, descriptor=descriptor, root_tensor=root
                )
                payload = (
                    {"source_state": root, "descriptor": descriptor}
                    if rank == 0 else None
                )
                clone, _decode, consensus = (
                    runtime._broadcast_rank0_vae_source_state_v1(
                        dist=fake,
                        torch_module=FakeTorch,
                        group="world8",
                        rank=rank,
                        device=FakeDevice(rank),
                        rank0_payload=payload,
                        **expected,
                    )
                )
                self.assertIsNot(clone, root)
                self.assertEqual(clone.data, root.data)
                self.assertEqual(fake.tensor_broadcast_count, 1)
                self.assertFalse(consensus["reduction_or_averaging_used"])
            with self.assertRaises(runtime.TargetFreeBerniniRuntimeError):
                runtime._broadcast_rank0_vae_source_state_v1(
                    dist=FakeRank0TensorBroadcastDist(
                        rank=5, descriptor=descriptor, root_tensor=root
                    ),
                    torch_module=FakeTorch,
                    group="world8",
                    rank=5,
                    device=FakeDevice(5),
                    rank0_payload={"source_state": root, "descriptor": descriptor},
                    **expected,
                )
            with self.assertRaises(runtime.TargetFreeBerniniRuntimeError):
                runtime._broadcast_rank0_vae_source_state_v1(
                    dist=FakeRank0TensorBroadcastDist(
                        rank=5, descriptor=descriptor, root_tensor=root, corrupt=True
                    ),
                    torch_module=FakeTorch,
                    group="world8",
                    rank=5,
                    device=FakeDevice(5),
                    rank0_payload=None,
                    **expected,
                )

    def test_factory_receipt_truthfully_uses_one_rank0_vae_broadcast(self) -> None:
        source = inspect.getsource(runtime._build_owned_runtime_v1)
        self.assertIn('"rank0_vae_source_broadcast"', source)
        self.assertNotIn("serialized_vae_construction", source)
        self.assertNotIn('phase="vae_source_encode"', source)
        broadcast = inspect.getsource(
            runtime._broadcast_rank0_vae_source_state_v1
        )
        self.assertEqual(broadcast.count("dist.broadcast(source_state"), 1)
        self.assertIn(".detach().clone(", broadcast)
        tree = ast.parse(broadcast)
        self.assertFalse(
            any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "all_reduce"
                for node in ast.walk(tree)
            )
        )

    def test_no_vae_is_rematerialized_on_cpu_during_release(self) -> None:
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        forbidden = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "to"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "vae"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "cpu"
            ):
                forbidden.append(node.lineno)
        self.assertEqual(forbidden, [])
        terminal_decode = inspect.getsource(
            runtime.decode_and_seal_recorded_trajectory_v1
        )
        self.assertIn("_trim_host_allocator_v1()", terminal_decode)

    def test_rank_local_miopen_cache_is_fresh_writable_and_pre_torch(self) -> None:
        self.assertNotIn("torch", sys.modules)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            output = base / "shared-output-candidate"
            inherited = "/definitely/read-only/inherited-miopen"
            environment = {
                "RANK": "0",
                "LOCAL_RANK": "0",
                "WORLD_SIZE": "8",
                "LOCAL_WORLD_SIZE": "8",
                "SLURM_JOB_ID": "141620",
                "SLURM_STEP_ID": "cache-test",
                "MIOPEN_USER_DB_PATH": inherited,
                "MIOPEN_CUSTOM_CACHE_DIR": inherited,
            }
            with mock.patch.dict(os.environ, environment, clear=False):
                binding = runtime._prepare_rank_local_miopen_cache_v1(
                    output_path=output, base_root=base
                )
                validated = runtime._validate_rank_local_miopen_cache_binding_v1(
                    binding,
                    expected_rank=0,
                    require_environment=True,
                    require_physical=True,
                )
                self.assertNotEqual(
                    os.environ["MIOPEN_USER_DB_PATH"], inherited
                )
                self.assertNotEqual(
                    os.environ["MIOPEN_CUSTOM_CACHE_DIR"], inherited
                )
                self.assertTrue(validated["torch_absent_at_configuration"])
                self.assertEqual(validated["sqlite_probe"]["quick_check"], "ok")
                self.assertEqual(
                    validated["output_path_sha256"],
                    hashlib.sha256(str(output).encode("utf-8")).hexdigest(),
                )
                for row in validated["directories"]:
                    self.assertEqual(list(Path(row["path"]).iterdir()), [])
                with self.assertRaisesRegex(
                    runtime.TargetFreeBerniniRuntimeError, "fresh rank-local"
                ):
                    runtime._prepare_rank_local_miopen_cache_v1(
                        output_path=output, base_root=base
                    )

    def test_world8_miopen_cache_bindings_are_distinct_and_fail_closed(self) -> None:
        self.assertNotIn("torch", sys.modules)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            output = base / "shared-output-candidate"
            environment = {
                "RANK": "0",
                "LOCAL_RANK": "0",
                "WORLD_SIZE": "8",
                "LOCAL_WORLD_SIZE": "8",
                "SLURM_JOB_ID": "141620",
                "SLURM_STEP_ID": "cache-world8-test",
            }
            with mock.patch.dict(os.environ, environment, clear=False):
                rows = []
                for rank in range(runtime.WORLD_SIZE):
                    os.environ["RANK"] = str(rank)
                    os.environ["LOCAL_RANK"] = str(rank)
                    rows.append(
                        runtime._prepare_rank_local_miopen_cache_v1(
                            output_path=output, base_root=base
                        )
                    )
                validated = runtime._validate_world8_miopen_cache_bindings_v1(
                    rows, local_rank=7
                )
                self.assertEqual(len({row["rank_root"] for row in validated}), 8)
                self.assertEqual(
                    len({row["binding_digest"] for row in validated}), 8
                )

                os.environ["MIOPEN_USER_DB_PATH"] = "/escaped/read-only"
                with self.assertRaisesRegex(
                    runtime.TargetFreeBerniniRuntimeError, "live environment"
                ):
                    runtime._validate_rank_local_miopen_cache_binding_v1(
                        rows[7],
                        expected_rank=7,
                        require_environment=True,
                        require_physical=True,
                    )

                hostile = copy.deepcopy(rows)
                hostile[1]["rank_root"] = hostile[0]["rank_root"]
                unsigned = dict(hostile[1])
                unsigned.pop("binding_digest")
                hostile[1]["binding_digest"] = runtime.object_sha256(unsigned)
                with self.assertRaises(runtime.TargetFreeBerniniRuntimeError):
                    runtime._validate_world8_miopen_cache_bindings_v1(
                        hostile, local_rank=7
                    )

    def test_main_prepares_miopen_cache_before_owned_factory(self) -> None:
        source = inspect.getsource(runtime.main)
        self.assertLess(
            source.index("_prepare_rank_local_miopen_cache_v1"),
            source.index("_build_owned_runtime_v1"),
        )
        prepare_source = inspect.getsource(
            runtime._prepare_rank_local_miopen_cache_v1
        )
        self.assertIn('if "torch" in sys.modules', prepare_source)
        self.assertIn('"MIOPEN_USER_DB_PATH"', MODULE_PATH.read_text())
        self.assertIn('"MIOPEN_CUSTOM_CACHE_DIR"', MODULE_PATH.read_text())

    def test_frozen_no_git_bernini_snapshot_is_projectable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory).resolve()
            source_root = parent / "Bernini-2d2b4591"
            package_root = source_root / "bernini"
            config_root = source_root / "configs" / "pilot"
            package_root.mkdir(parents=True)
            config_root.mkdir(parents=True)
            package_bytes = b"VALUE = 'held-source'\n"
            config_bytes = b'{"held":true}\n'
            package_path = package_root / "__init__.py"
            config_path = config_root / "config.json"
            package_path.write_bytes(package_bytes)
            config_path.write_bytes(config_bytes)
            package_path.chmod(0o444)
            config_path.chmod(0o444)
            for path in (package_root, config_root, config_root.parent, source_root):
                path.chmod(0o555)
            try:
                rows, evidence = runtime._frozen_no_git_source_projection_v1(
                    source_root=source_root,
                    expected_commit=runtime.BERNINI_COMMIT,
                    package_name="bernini",
                    extra_relative_files=("configs/pilot/config.json",),
                    prefix="bernini_root",
                    critical_sha256={
                        "bernini/__init__.py": hashlib.sha256(package_bytes).hexdigest(),
                        "configs/pilot/config.json": hashlib.sha256(config_bytes).hexdigest(),
                    },
                )
                self.assertEqual(
                    [name for name, _ in rows],
                    [
                        "bernini_root/bernini/__init__.py",
                        "bernini_root/configs/pilot/config.json",
                    ],
                )
                self.assertEqual(evidence["commit_claim"], runtime.BERNINI_COMMIT)
                self.assertTrue(evidence["all_projected_files_mode_0444_nlink1"])
                self.assertTrue(
                    evidence["controlled_environment_no_concurrent_mutator_assumed"]
                )
            finally:
                for path in (package_root, config_root, config_root.parent, source_root):
                    path.chmod(0o700)

    def test_two_fresh_snapshot_roots_have_same_policy_model_closure(self) -> None:
        checkpoint_files = sha("checkpoint exact23 files")
        source_files = sha("committed vendor files")
        versions = {
            "torch": runtime.TORCH_VERSION,
            "transformers": runtime.TRANSFORMERS_VERSION,
            "peft": runtime.PEFT_VERSION,
            "diffusers": runtime.DIFFUSERS_VERSION,
            "decord": runtime.DECORD_VERSION,
            "safetensors": runtime.SAFETENSORS_VERSION,
        }

        def closure(root: str):
            return runtime._path_independent_model_closure_v1(
                checkpoint_snapshot={
                    "destination": f"{root}/checkpoint",
                    "snapshot_digest": sha(f"physical checkpoint {root}"),
                    "files_digest": checkpoint_files,
                },
                source_snapshot={
                    "destination": f"{root}/source",
                    "snapshot_digest": sha(f"physical source {root}"),
                    "files_digest": source_files,
                },
                package_versions=versions,
                lora_installation_digest=sha("lora installation"),
                peft_config_transition_digest=sha("peft exact240 to exact4"),
                source_state_sha256=sha("source state"),
                negative_condition_sha256=sha("negative condition"),
                positive_condition_sha256=sha("positive condition"),
            )

        first = closure("/fresh-a")
        second = closure("/fresh-b")
        self.assertEqual(first, second)
        self.assertEqual(runtime.object_sha256(first), runtime.object_sha256(second))
        self.assertNotIn("destination", json.dumps(first, sort_keys=True))

    def test_success_checkpoint_is_after_world8_policy_consensus(self) -> None:
        source = inspect.getsource(runtime._engineering_one_update_loaded_v1)
        candidate_consensus = source.index(
            'label="full644 target-free prepublication updated policy"'
        )
        commit = source.index("runtime.commit_updated_policy_digest_v1")
        final_consensus = source.index(
            'label="full644 target-free updated policy"'
        )
        checkpoint = source.index("checkpoint_binding = _save_reload_exact_one_checkpoint_v1")
        self.assertLess(candidate_consensus, commit)
        self.assertLess(commit, final_consensus)
        self.assertLess(final_consensus, checkpoint)
        zero_source = inspect.getsource(runtime._zero_update_receipt_v1)
        self.assertIn("zero-update postbranch parameters", zero_source)
        publication_source = inspect.getsource(
            runtime._publish_one_source_update_stage_v1
        )
        reload_index = publication_source.index(
            '_strict_json(raw, label="update-stage receipt")'
        )
        ack_index = publication_source.index("gathered_ack")
        return_index = publication_source.rindex("return {")
        self.assertLess(reload_index, ack_index)
        self.assertLess(ack_index, return_index)
        rollout_source = inspect.getsource(runtime._run_one_source_rollout_stage_v1)
        rollout_reload = rollout_source.index(
            '_strict_json(raw, label="rollout-stage receipt")'
        )
        rollout_ack = rollout_source.index("gathered_ack")
        rollout_return = rollout_source.rindex("return {")
        self.assertLess(rollout_reload, rollout_ack)
        self.assertLess(rollout_ack, rollout_return)

    @unittest.skipUnless(
        os.environ.get("FULL644_TARGET_FREE_NATIVE_SMOKE") == "1",
        "requires torchrun WORLD8 and frozen AUH assets",
    )
    def test_real_owned_factory_native_smoke(self) -> None:
        required = {
            name: os.environ.get(name)
            for name in (
                "FULL644_TF_BERNINI_ROOT", "FULL644_TF_VEOMNI_ROOT",
                "FULL644_TF_CHECKPOINT", "FULL644_TF_CHECKPOINT_MANIFEST",
                "FULL644_TF_SMOKE_OUTPUT",
            )
        }
        self.assertTrue(all(required.values()), required)
        output_path = Path(required["FULL644_TF_SMOKE_OUTPUT"])
        miopen_cache_binding = runtime._prepare_rank_local_miopen_cache_v1(
            output_path=output_path
        )
        bundle = runtime._build_owned_runtime_v1(
            bernini_root=Path(required["FULL644_TF_BERNINI_ROOT"]),
            veomni_root=Path(required["FULL644_TF_VEOMNI_ROOT"]),
            checkpoint_root=Path(required["FULL644_TF_CHECKPOINT"]),
            checkpoint_content_manifest=Path(required["FULL644_TF_CHECKPOINT_MANIFEST"]),
            output_root=output_path,
            miopen_cache_binding=miopen_cache_binding,
        )
        self.assertEqual(
            len(bundle.runtime.named_trainable_parameters), runtime.LORA_TENSOR_COUNT
        )
        self.assertEqual(
            bundle.runtime.activation_checkpoint_blocks,
            runtime.ACTIVATION_CHECKPOINT_BLOCKS,
        )


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import torch
        except ImportError as error:  # pragma: no cover - environment gate
            raise unittest.SkipTest(f"torch unavailable: {error}")
        cls.torch = torch

    def test_canonical_fp32_projection_matches_legacy_bytes_and_sha(self) -> None:
        torch = self.torch
        for shape in ((), (7,), (2, 3), (1, 2, 3, 2, 2)):
            with self.subTest(shape=shape):
                count = math.prod(shape) if shape else 1
                value = torch.linspace(-1.25, 2.5, count, dtype=torch.float32).reshape(
                    shape
                )
                legacy = value.detach().to("cpu", torch.float32).contiguous()
                legacy_raw = bytes(legacy.untyped_storage())
                projected_shape, raw = runtime._canonical_fp32_tensor_bytes_v1(
                    value, label="test tensor"
                )
                self.assertEqual(projected_shape, list(shape))
                self.assertEqual(raw, legacy_raw)
                expected = runtime._tensor_sha256_from_canonical_fp32_bytes_v1(
                    projected_shape, legacy_raw
                )
                self.assertEqual(runtime.tensor_sha256(value), expected)

    def test_canonical_fp32_projection_closes_views_endian_and_finite(self) -> None:
        torch = self.torch
        base = torch.arange(30, dtype=torch.float32).reshape(5, 6)
        value = base[:, 1::2]
        self.assertFalse(value.is_contiguous())
        shape, raw = runtime._canonical_fp32_tensor_bytes_v1(
            value, label="test tensor"
        )
        expected = (
            value.detach()
            .to("cpu", torch.float32)
            .contiguous()
            .numpy()
            .tobytes(order="C")
        )
        self.assertEqual(raw, expected)
        self.assertEqual(len(raw), value.numel() * 4)

        for label, hostile in (
            ("nonzero_offset", base.reshape(-1)[3:15]),
            ("oversized_backing", base.reshape(-1)[:12]),
        ):
            with self.subTest(storage=label), self.assertRaisesRegex(
                runtime.TargetFreeBerniniRuntimeError,
                "source FP32 storage differs",
            ):
                runtime._canonical_fp32_tensor_bytes_v1(
                    hostile, label="test tensor"
                )

        with mock.patch.object(runtime.sys, "byteorder", "big"):
            with self.assertRaisesRegex(
                runtime.TargetFreeBerniniRuntimeError, "little-endian"
            ):
                runtime.tensor_sha256(torch.zeros((2,), dtype=torch.float32))
        for label, value in (
            ("nan", torch.tensor([float("nan")], dtype=torch.float32)),
            ("inf", torch.tensor([float("inf")], dtype=torch.float32)),
        ):
            with self.subTest(nonfinite=label), self.assertRaisesRegex(
                runtime.TargetFreeBerniniRuntimeError, "non-finite"
            ):
                runtime._tensor_fp32_bytes(value)

    def test_exact40_artifact_projects_each_tensor_once_without_storage_iteration(
        self,
    ) -> None:
        torch = self.torch
        initial = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
        actions = [initial.add(index) for index in range(40)]
        original = runtime._canonical_fp32_tensor_bytes_v1
        with mock.patch.object(
            runtime,
            "_canonical_fp32_tensor_bytes_v1",
            wraps=original,
        ) as projection, mock.patch.object(
            runtime,
            "tensor_sha256",
            side_effect=AssertionError("artifact must reuse its projected raw bytes"),
        ):
            payload, header = runtime.build_trajectory_artifact_v1(
                initial_state=initial, actions=actions
            )
        self.assertEqual(projection.call_count, runtime.TRAJECTORY_STEPS + 1)
        self.assertEqual(header["tensor_count"], runtime.TRAJECTORY_STEPS + 1)
        self.assertGreater(len(payload), header["payload_size_bytes"])
        projection_source = inspect.getsource(
            runtime._canonical_fp32_tensor_bytes_v1
        )
        self.assertIn("untyped_storage().nbytes()", projection_source)
        self.assertNotIn("bytes(logical.untyped_storage", projection_source)
        self.assertNotIn("bytes(cpu.untyped_storage", projection_source)

    def _write_trajectory(self, root: Path, *, arm: int = 0):
        torch = self.torch
        policy_sha = sha("policy")
        initial = torch.tensor([[0.25, -0.5]], dtype=torch.float32)
        state = initial.clone()
        parameter = torch.nn.Parameter(torch.tensor(0.125, dtype=torch.float32))
        actions = []
        epsilons = []
        steps = []
        generator = torch.Generator().manual_seed(73 + arm)
        for index in range(runtime.TRAJECTORY_STEPS):
            mean = parameter.detach().expand_as(state).contiguous()
            epsilon = torch.randn(state.shape, generator=generator)
            action = (mean + runtime.ACTION_STD * epsilon).contiguous()
            next_state = (state - 0.01 * action).contiguous()
            step = {
                "schema_version": runtime.TRAJECTORY_STEP_SCHEMA,
                "schedule_index": index,
                "timestep": 1000 - index,
                "sigma_float32_be_hex": "3f800000",
                "state_before_sha256": runtime.tensor_sha256(state),
                "policy_mean_sha256": runtime.tensor_sha256(mean),
                "action_noise_key_sha256": sha(f"noise-key-{arm}-{index}"),
                "action_noise_sha256": runtime.tensor_sha256(epsilon),
                "executed_action_sha256": runtime.tensor_sha256(action),
                "state_after_sha256": runtime.tensor_sha256(next_state),
                "scheduler_step_index_after": index + 1,
            }
            step["step_digest"] = runtime.object_sha256(step)
            actions.append(action)
            epsilons.append(epsilon)
            steps.append(step)
            state = next_state
        artifact, _ = runtime.build_trajectory_artifact_v1(
            initial_state=initial, actions=actions
        )
        artifact_path = root / f"arm{arm}.fp32"
        artifact_binding = runtime.write_create_only(artifact_path, artifact)
        receipt = {
            "schema_version": runtime.TRAJECTORY_SCHEMA,
            "runtime_schema_version": runtime.SCHEMA_VERSION,
            "rollout_id": f"rollout-arm{arm}",
            "source_row_id": "row-000",
            "source_video_sha256": sha("source"),
            "instruction_sha256": sha("instruction"),
            "behavior_policy_sha256": policy_sha,
            "round_index": 0,
            "rollout_seed": 73 + arm,
            "dp_arm": arm,
            "sp_size": runtime.SP_SIZE,
            "step_count": runtime.TRAJECTORY_STEPS,
            "latent_shape": list(initial.shape),
            "latent_numel": initial.numel(),
            "latent_dtype": "torch.float32",
            "schedule_sha256": runtime.SCHEDULE_SHA256,
            "gaussian_kernel": runtime.GAUSSIAN_KERNEL,
            "gaussian_kernel_sha256": runtime.GAUSSIAN_KERNEL_SHA256,
            "apg_guidance_sha256": runtime.APG_GUIDANCE_SHA256,
            "initial_noise_key_sha256": sha(f"initial-{arm}"),
            "initial_state_sha256": runtime.tensor_sha256(initial),
            "steps": steps,
            "terminal_state_sha256": runtime.tensor_sha256(state),
            "artifact_path": str(artifact_path),
            "artifact_sha256": artifact_binding["sha256"],
            "artifact_size_bytes": artifact_binding["size_bytes"],
            "artifact_mode_octal": "0444",
            "artifact_nlink": 1,
            "sp4_noise_broadcast": True,
            "sp4_step_consensus": True,
            "source_only_input": True,
            "paired_reference_read_count": 0,
            "external_velocity_read_count": 0,
        }
        receipt["receipt_digest"] = runtime.object_sha256(receipt)
        receipt_path = root / f"arm{arm}.json"
        receipt_binding = runtime.write_create_only(
            receipt_path, runtime.canonical_json_bytes(receipt)
        )
        loaded = runtime.load_trajectory_receipt_v1(
            receipt_path, expected_sha256=receipt_binding["sha256"]
        )
        fake = object.__new__(runtime.BerniniExact40PolicyV1)
        fake.source_row_id = "row-000"
        fake.source_video_sha256 = sha("source")
        fake.instruction_sha256 = sha("instruction")
        fake._behavior_policy_sha256 = policy_sha
        fake.device = torch.device("cpu")
        fake.parallel = types.SimpleNamespace(
            contract=types.SimpleNamespace(arm_index=arm, sp_rank=0)
        )
        fake.parameter = parameter
        fake.initial = initial
        fake.epsilons = epsilons

        def fresh_scheduler(self):
            return FakeScheduler()

        def coordinate(self, scheduler, index):
            return runtime.Exact40CoordinateV1(
                index=index,
                timestep=torch.tensor([1000 - index]),
                sigma=torch.tensor(1.0),
                timestep_value=1000 - index,
                sigma_float32_be_hex="3f800000",
            )

        def mean_no_grad(self, *, state, coordinate):
            return self.parameter.detach().expand_as(state).contiguous()

        def keyed_noise(self, *, shape, rollout_seed, purpose, index):
            if index == -1:
                value = self.initial.clone()
                key = sha(f"initial-{arm}")
            else:
                value = self.epsilons[index].clone()
                key = sha(f"noise-key-{arm}-{index}")
            return value, {
                "key_sha256": key,
                "tensor_sha256": runtime.tensor_sha256(value),
            }

        def scheduler_step(self, *, scheduler, coordinate, action, state):
            expected = coordinate.index if scheduler.step_index is None else scheduler.step_index
            if expected != coordinate.index:
                raise AssertionError("fake scheduler history skipped")
            scheduler.step_index = coordinate.index + 1
            return (state - 0.01 * action).detach().contiguous()

        def backward_mean(
            self,
            *,
            state,
            coordinate,
            mean_cotangent,
            expected_mean_sha256,
        ):
            value = self.parameter.expand_as(state)
            if runtime.tensor_sha256(value) != expected_mean_sha256:
                raise AssertionError("fake replay mean differs")
            value.backward(mean_cotangent)
            return {
                "branch_order": ["negative", "positive"],
                "serial_graph_release_verified": True,
                "index": coordinate.index,
            }

        fake.fresh_scheduler = types.MethodType(fresh_scheduler, fake)
        fake.coordinate = types.MethodType(coordinate, fake)
        fake.policy_mean_no_grad = types.MethodType(mean_no_grad, fake)
        fake.keyed_noise = types.MethodType(keyed_noise, fake)
        fake.scheduler_step = types.MethodType(scheduler_step, fake)
        fake.backward_policy_mean_v1 = types.MethodType(backward_mean, fake)
        return fake, loaded, actions

    def test_exact40_artifact_streaming_and_stateful_two_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            fake, receipt, actions = self._write_trajectory(root)
            self.assertEqual(len(actions), 40)
            pass1 = runtime.replay_trajectory_pass1_v1(fake, receipt)
            self.assertEqual(pass1["step_count"], 40)
            self.assertTrue(pass1["fresh_stateful_unipc_replay"])
            self.assertEqual(len(pass1["step_logprob_float64_hex"]), 40)
            result = runtime.replay_trajectory_pass2_backward_v1(
                fake, receipt, local_trajectory_coefficient=2.0
            )
            self.assertEqual(result["step_count"], 40)
            self.assertTrue(result["per_step_graph_released"])
            self.assertIsNotNone(fake.parameter.grad)
            self.assertTrue(bool(self.torch.isfinite(fake.parameter.grad).all().item()))
            self.assertNotEqual(float(fake.parameter.grad.item()), 0.0)

    def test_artifact_action_byte_flip_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            _, receipt, _ = self._write_trajectory(root)
            path = receipt.artifact_path
            path.chmod(0o644)
            raw = bytearray(path.read_bytes())
            raw[-1] ^= 1
            path.write_bytes(raw)
            path.chmod(0o444)
            with self.assertRaisesRegex(
                runtime.TargetFreeBerniniRuntimeError, "artifact bytes differ"
            ):
                with runtime.TrajectoryArtifactReaderV1(
                    path, expected_sha256=receipt.artifact_sha256
                ):
                    pass

    def test_receipt_requires_exact40_and_closed_chain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            _, receipt, _ = self._write_trajectory(root)
            hostile = json.loads(json.dumps(receipt.value))
            hostile["steps"] = hostile["steps"][:-1]
            hostile["step_count"] = 39
            unsigned = dict(hostile)
            unsigned.pop("receipt_digest")
            hostile["receipt_digest"] = runtime.object_sha256(unsigned)
            with self.assertRaisesRegex(
                runtime.TargetFreeBerniniRuntimeError, "fixed closure differs"
            ):
                runtime.validate_trajectory_receipt_value_v1(hostile)

    def test_dp2_endpoint_split_gradient_is_exact_single_pair_gradient(self) -> None:
        torch = self.torch
        theta = torch.tensor(0.2, requires_grad=True)
        chosen = 3.0 * theta
        rejected = -2.0 * theta
        true_loss = torch.nn.functional.softplus(-(chosen - rejected))
        (true_gradient,) = torch.autograd.grad(true_loss, theta)
        coefficients = runtime.preference_coefficients_v1(
            float(chosen.detach().item()), float(rejected.detach().item())
        )
        arm0 = coefficients[
            "chosen_local_coefficient_after_dp2_compensation"
        ] * 3.0
        arm1 = coefficients[
            "rejected_local_coefficient_after_dp2_compensation"
        ] * -2.0
        after_dp2_mean = (arm0 + arm1) / 2.0
        self.assertAlmostEqual(after_dp2_mean, float(true_gradient.item()), places=6)

    def test_logprob_reduction_is_dimension_normalized(self) -> None:
        torch = self.torch
        mean = torch.zeros((2, 3), dtype=torch.float32)
        action = torch.full_like(mean, runtime.ACTION_STD)
        first = runtime.normalized_gaussian_step_logprob_v1(action, mean)
        tiled = runtime.normalized_gaussian_step_logprob_v1(
            action.repeat(100, 1), mean.repeat(100, 1)
        )
        self.assertEqual(float(first.item()), float(tiled.item()))
        self.assertEqual(
            runtime.GAUSSIAN_SCORE_REDUCTION,
            "mean_over_latent_elements_then_sum_exact40",
        )

    def test_gaussian_mean_cotangent_matches_autograd_with_sign(self) -> None:
        torch = self.torch
        mean = torch.tensor([[-0.2, 0.4]], dtype=torch.float32, requires_grad=True)
        action = torch.tensor([[0.1, -0.3]], dtype=torch.float32)
        coefficient = -1.75
        score = coefficient * (
            -0.5 * ((action - mean) / runtime.ACTION_STD).square().mean()
        )
        (expected,) = torch.autograd.grad(score, mean)
        observed = runtime.gaussian_mean_cotangent_v1(
            action=action,
            mean=mean.detach(),
            trajectory_coefficient=coefficient,
        )
        self.assertTrue(torch.equal(expected, observed))
        self.assertLess(float(observed[0, 0]), 0.0)
        self.assertGreater(float(observed[0, 1]), 0.0)

    def test_chosen_endpoint_may_be_dp_arm_one(self) -> None:
        self.assertEqual(
            runtime.endpoint_roles_by_dp_arm_v1(1, 0),
            {1: "chosen", 0: "rejected"},
        )
        with self.assertRaisesRegex(
            runtime.TargetFreeBerniniRuntimeError, "distinct DP arms"
        ):
            runtime.endpoint_roles_by_dp_arm_v1(0, 0)

    def test_exact480_name_key_closure(self) -> None:
        names = []
        keys = set()
        for block in range(30):
            for attention in (1, 2):
                for projection in ("to_q", "to_k", "to_v", "to_out.0"):
                    for side in ("A", "B"):
                        name = (
                            "base_model.model.diff_dec.transformer.blocks."
                            f"{block}.attn{attention}.{projection}.lora_{side}.default.weight"
                        )
                        match = runtime._LORA_NAME.fullmatch(name)
                        self.assertIsNotNone(match)
                        assert match is not None
                        keys.add(
                            (
                                int(match.group("block")),
                                int(match.group("attention")),
                                match.group("projection"),
                                match.group("side"),
                            )
                        )
                        names.append(name)
        self.assertEqual(len(names), runtime.LORA_TENSOR_COUNT)
        self.assertEqual(len(keys), runtime.LORA_TENSOR_COUNT)
        self.assertEqual(len(sorted(names)), runtime.LORA_TENSOR_COUNT)

    def test_peft_exact240_request_exact4_canonical_and_hostiles(self) -> None:
        requested_targets = runtime._target_free_requested_lora_targets_v1()
        canonical_targets = set(runtime._PEFT_CANONICAL_TARGET_MODULES)

        def config_value(targets, *, base_model_name_or_path=None):
            return {
                "alora_invocation_tokens": None,
                "alpha_pattern": {},
                "arrow_config": None,
                "auto_mapping": None,
                "base_model_name_or_path": base_model_name_or_path,
                "bias": "none",
                "corda_config": None,
                "ensure_weight_tying": False,
                "eva_config": None,
                "exclude_modules": None,
                "fan_in_fan_out": False,
                "inference_mode": False,
                "init_lora_weights": True,
                "layer_replication": None,
                "layers_pattern": None,
                "layers_to_transform": None,
                "loftq_config": {},
                "lora_alpha": runtime.LORA_ALPHA,
                "lora_bias": False,
                "lora_dropout": 0.0,
                "lora_ga_config": None,
                "megatron_config": None,
                "megatron_core": "megatron.core",
                "modules_to_save": None,
                "peft_type": "LORA",
                "peft_version": runtime.PEFT_VERSION,
                "qalora_group_size": 16,
                "r": runtime.LORA_RANK,
                "rank_pattern": {},
                "revision": None,
                "target_modules": set(targets),
                "target_parameters": None,
                "task_type": None,
                "trainable_token_indices": None,
                "use_bdlora": None,
                "use_dora": False,
                "use_qalora": False,
                "use_rslora": False,
            }

        class Config:
            def __init__(self, value):
                self.value = value

            def to_dict(self):
                return self.value

        fake_peft = types.SimpleNamespace(__version__=runtime.PEFT_VERSION)
        with mock.patch.dict(sys.modules, {"peft": fake_peft}):
            requested = runtime._validate_target_free_peft_config_v1(
                Config(config_value(requested_targets)),
                expected_targets=requested_targets,
                target_modules_contract=runtime._PEFT_REQUESTED_TARGET_CONTRACT,
            )
            canonical = runtime._validate_target_free_peft_config_v1(
                Config(
                    config_value(
                        canonical_targets,
                        base_model_name_or_path="",
                    )
                ),
                expected_targets=canonical_targets,
                target_modules_contract=runtime._PEFT_CANONICAL_TARGET_CONTRACT,
            )
            self.assertEqual(requested["target_module_count"], 240)
            self.assertEqual(canonical["target_module_count"], 4)
            transition = runtime._bind_target_free_peft_transition_v1(
                requested_receipt=requested,
                canonical_receipt=canonical,
                lora_installation_digest=sha("exact480 installation"),
            )
            self.assertEqual(
                transition["requested_target_modules_sha256"],
                requested["target_modules_sha256"],
            )
            self.assertEqual(
                transition["canonical_target_modules_sha256"],
                canonical["target_modules_sha256"],
            )
            self.assertEqual(
                transition["installed_exact480_lora_tensor_count"], 480
            )

            hostile = config_value(requested_targets)
            hostile["use_dora"] = True
            with self.assertRaisesRegex(
                runtime.TargetFreeBerniniRuntimeError, "use_dora"
            ):
                runtime._validate_target_free_peft_config_v1(
                    Config(hostile),
                    expected_targets=requested_targets,
                    target_modules_contract=runtime._PEFT_REQUESTED_TARGET_CONTRACT,
                )

            hostile_post_path = config_value(canonical_targets)
            hostile_post_path["base_model_name_or_path"] = "/fresh/output/path"
            with self.assertRaisesRegex(
                runtime.TargetFreeBerniniRuntimeError,
                "base_model_name_or_path",
            ):
                runtime._validate_target_free_peft_config_v1(
                    Config(hostile_post_path),
                    expected_targets=canonical_targets,
                    target_modules_contract=runtime._PEFT_CANONICAL_TARGET_CONTRACT,
                )

            for label, contract, targets, hostile_name in (
                (
                    "empty_pre",
                    runtime._PEFT_REQUESTED_TARGET_CONTRACT,
                    requested_targets,
                    "",
                ),
                (
                    "none_post",
                    runtime._PEFT_CANONICAL_TARGET_CONTRACT,
                    canonical_targets,
                    None,
                ),
            ):
                with self.subTest(name_sentinel=label), self.assertRaisesRegex(
                    runtime.TargetFreeBerniniRuntimeError,
                    "base_model_name_or_path",
                ):
                    runtime._validate_target_free_peft_config_v1(
                        Config(
                            config_value(
                                targets,
                                base_model_name_or_path=hostile_name,
                            )
                        ),
                        expected_targets=targets,
                        target_modules_contract=contract,
                    )

            for label, hostile_targets in (
                ("missing", requested_targets - {min(requested_targets)}),
                ("extra", requested_targets | {"diff_dec.transformer.evil"}),
                (
                    "renamed",
                    (requested_targets - {min(requested_targets)})
                    | {"diff_dec.transformer.blocks.0.attn1.to_evil"},
                ),
            ):
                with self.subTest(requested=label), self.assertRaisesRegex(
                    runtime.TargetFreeBerniniRuntimeError,
                    "requested_exact240_full_module_paths target modules",
                ):
                    runtime._validate_target_free_peft_config_v1(
                        Config(config_value(hostile_targets)),
                        expected_targets=requested_targets,
                        target_modules_contract=runtime._PEFT_REQUESTED_TARGET_CONTRACT,
                    )

            for label, hostile_targets in (
                ("missing", canonical_targets - {"to_q"}),
                ("extra", canonical_targets | {"to_evil"}),
                ("renamed", (canonical_targets - {"to_q"}) | {"q_proj"}),
                ("full240_post", requested_targets),
            ):
                with self.subTest(canonical=label), self.assertRaisesRegex(
                    runtime.TargetFreeBerniniRuntimeError,
                    "postinstall_exact4_unique_suffixes target modules",
                ):
                    runtime._validate_target_free_peft_config_v1(
                        Config(config_value(hostile_targets)),
                        expected_targets=canonical_targets,
                        target_modules_contract=runtime._PEFT_CANONICAL_TARGET_CONTRACT,
                    )

            resigned = copy.deepcopy(requested)
            resigned["target_modules"] = resigned["target_modules"][:-1]
            resigned["target_module_count"] = 239
            resigned["target_modules_sha256"] = runtime.object_sha256(
                resigned["target_modules"]
            )
            resigned["config"]["target_modules"] = list(
                resigned["target_modules"]
            )
            resigned["config_digest"] = runtime.object_sha256(resigned["config"])
            body = {key: resigned[key] for key in resigned if key != "receipt_digest"}
            resigned["receipt_digest"] = runtime.object_sha256(body)
            with self.assertRaisesRegex(
                runtime.TargetFreeBerniniRuntimeError, "phase receipt closure"
            ):
                runtime._bind_target_free_peft_transition_v1(
                    requested_receipt=resigned,
                    canonical_receipt=canonical,
                    lora_installation_digest=sha("exact480 installation"),
                )

            for label, mutate in (
                ("missing", lambda value: value.pop("target_modules_sha256")),
                ("extra", lambda value: value.update({"evil": False})),
            ):
                hostile_receipt = copy.deepcopy(canonical)
                mutate(hostile_receipt)
                with self.subTest(receipt=label), self.assertRaisesRegex(
                    runtime.TargetFreeBerniniRuntimeError,
                    "phase receipt fields",
                ):
                    runtime._bind_target_free_peft_transition_v1(
                        requested_receipt=requested,
                        canonical_receipt=hostile_receipt,
                        lora_installation_digest=sha("exact480 installation"),
                    )

    def test_owned_factory_validates_exact240_before_exact4_after_install(self) -> None:
        source = inspect.getsource(runtime._build_owned_runtime_v1)
        requested = source.index(
            "target_modules_contract=_PEFT_REQUESTED_TARGET_CONTRACT"
        )
        install = source.index("local_model = get_peft_model")
        canonical = source.index(
            "target_modules_contract=_PEFT_CANONICAL_TARGET_CONTRACT"
        )
        exact480 = source.index("packed.validate_lora_installation")
        self.assertLess(requested, install)
        self.assertLess(install, canonical)
        self.assertLess(canonical, exact480)
        self.assertIn(
            '"peft_config_three_layer_closure": runtime.peft_config_receipt',
            source,
        )
    def test_public_runtime_has_no_opaque_logprob_provider(self) -> None:
        signature = inspect.signature(runtime.BerniniExact40PolicyV1)
        self.assertNotIn("logprob_provider", signature.parameters)
        self.assertNotIn("optimizer_factory", signature.parameters)
        self.assertFalse(
            hasattr(runtime, "engineering_one_update_from_paths_v1")
        )
        self.assertNotIn("paired_target", MODULE_PATH.read_text())
        self.assertNotIn("frozen_velocity", MODULE_PATH.read_text())


if __name__ == "__main__":
    unittest.main()
