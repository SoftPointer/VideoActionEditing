from __future__ import annotations

from contextlib import contextmanager, nullcontext
from datetime import timedelta
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_case01_object_trajectory_oracle_v1 as wrapper  # noqa: E402
import object_trajectory_projection_v1 as projection  # noqa: E402


try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _world4_operational_failpoint_worker(
    rank, scenario, init_path, output_path, receipt_path, result_queue
):
    import torch
    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=20),
    )
    rejected = False
    rejection_text = None
    scheduler = None
    try:
        source = torch.zeros(
            1, 16, 21, 62, 60, dtype=torch.float32
        )
        aux = torch.ones_like(source)
        assets = _FakeAssets("trajectory_dog_bone")
        assets.scaffold = _scaffold()
        legacy = types.SimpleNamespace(_pack_wan_source_latent=_pack)
        state = wrapper.OracleExecutionState(legacy=legacy, assets=assets)
        if rank == 0:
            state.source_vae_encode_calls = 1
            state.aux_vae_encode_attempts = 1
            state.aux_vae_encode_calls = 1
            state.aux_latent = aux

        if scenario == "rank0_aux_encode":
            if rank == 0:
                state.aux_vae_encode_calls = 0
                state.aux_latent = None
                state.aux_encode_error = RuntimeError(
                    "synthetic rank0 aux encode failure"
                )
            state.distributed_aux(source)
        elif scenario in ("nonzero_aux_alloc", "nonzero_aux_abi"):
            if rank == 2 and scenario == "nonzero_aux_alloc":
                allocation = mock.patch.object(
                    torch,
                    "empty_like",
                    side_effect=RuntimeError(
                        "synthetic nonzero aux allocation failure"
                    ),
                )
            elif rank == 2:
                allocation = mock.patch.object(
                    torch,
                    "empty_like",
                    return_value=torch.empty(1, dtype=source.dtype),
                )
            else:
                allocation = nullcontext()
            with allocation:
                state.distributed_aux(source)
        elif scenario == "post_broadcast_validation":
            validation = (
                mock.patch.object(
                    torch,
                    "isfinite",
                    side_effect=RuntimeError(
                        "synthetic post-broadcast validation failure"
                    ),
                )
                if rank == 2
                else nullcontext()
            )
            with validation:
                state.distributed_aux(source)
        elif scenario == "projection_runtime_readiness":
            scheduler = UniPCMultistepScheduler()
            diffusion = types.SimpleNamespace(
                use_unipc=rank != 2, scheduler=scheduler
            )
            with state.clamp(diffusion, source, expected_steps=1):
                raise AssertionError("runtime-readiness failpoint unexpectedly yielded")
        elif scenario in ("row_build_tensor_hash", "projection_contract_build"):
            tensor_hash = (
                mock.patch.object(
                    wrapper,
                    "_tensor_byte_authority",
                    side_effect=RuntimeError("synthetic row tensor hash failure"),
                )
                if scenario == "row_build_tensor_hash" and rank == 2
                else nullcontext()
            )
            contract_build = (
                mock.patch.object(
                    wrapper,
                    "_projection_contract",
                    side_effect=RuntimeError(
                        "synthetic projection contract build failure"
                    ),
                )
                if scenario == "projection_contract_build" and rank == 2
                else nullcontext()
            )
            scheduler = UniPCMultistepScheduler()
            diffusion = types.SimpleNamespace(use_unipc=True, scheduler=scheduler)
            with tensor_hash, contract_build, state.clamp(
                diffusion, source, expected_steps=1
            ):
                raise AssertionError(
                    "row/contract build failpoint unexpectedly yielded"
                )
        elif scenario in (
            "lazy_bootstrap_install",
            "projector_install",
            "final_scheduler_validation",
        ):
            if scenario == "lazy_bootstrap_install" and rank == 2:
                class RejectBootstrapScheduler(UniPCMultistepScheduler):
                    def __setattr__(self, name, value):
                        if (
                            name == "step"
                            and getattr(
                                value,
                                "_bernini_case01_lazy_object_projection_v1",
                                False,
                            )
                        ):
                            raise RuntimeError(
                                "synthetic lazy bootstrap setattr failure"
                            )
                        super().__setattr__(name, value)

                scheduler = RejectBootstrapScheduler()
            else:
                scheduler = UniPCMultistepScheduler()
            diffusion = types.SimpleNamespace(use_unipc=True, scheduler=scheduler)

            class FailingProjectorContext:
                def __enter__(self):
                    raise RuntimeError("synthetic projector __enter__ failure")

                def __exit__(self, *error_info):
                    return False

            projector_patch = (
                mock.patch.object(
                    projection,
                    "project_object_trajectory_unipc_steps",
                    return_value=FailingProjectorContext(),
                )
                if scenario == "projector_install" and rank == 2
                else nullcontext()
            )
            with projector_patch, state.clamp(
                diffusion, source, expected_steps=1
            ) as facade:
                if scenario == "lazy_bootstrap_install":
                    raise AssertionError(
                        "lazy-bootstrap failpoint unexpectedly yielded"
                    )
                scheduler.sigmas = torch.tensor([1.0, 0.0])
                scheduler.timesteps = torch.tensor([1000.0])
                sample = torch.zeros(
                    1,
                    wrapper.EXPECTED_SEGMENT_TOKENS,
                    wrapper.PACKED_CHANNELS,
                ).contiguous()
                model_output = torch.zeros_like(sample).contiguous()
                scheduler.step(
                    model_output,
                    scheduler.timesteps[0],
                    sample,
                    return_dict=False,
                )
                if scenario == "final_scheduler_validation" and rank == 2:
                    facade.core_trace.records.clear()
        else:
            raise RuntimeError(f"unknown world4 failpoint scenario: {scenario}")

        Path(output_path).write_text("invalid output", encoding="utf-8")
        Path(receipt_path).write_text("invalid receipt", encoding="utf-8")
    except wrapper.ObjectOracleWrapperError as error:
        rejected = True
        rejection_text = str(error)
    finally:
        dist.destroy_process_group()
    result_queue.put(
        {
            "rank": rank,
            "scenario": scenario,
            "rejected": rejected,
            "rejection_text": rejection_text,
            "output_created": Path(output_path).exists(),
            "receipt_created": Path(receipt_path).exists(),
            "scheduler_instance_step_present": (
                None if scheduler is None else "step" in vars(scheduler)
            ),
        }
    )


class _FakeAuthority:
    def __init__(self, *, path="/canonical/fake", sha256="a" * 64, descriptor=8):
        self.path = Path(path)
        self.sha256 = sha256
        self.descriptor = descriptor
        self.identity = {
            "device": 1,
            "inode": 2,
            "mode": 0o100400,
            "nlink": 1,
            "uid": 3,
            "gid": 4,
            "size": 5,
            "mtime_ns": 6,
            "ctime_ns": 7,
        }
        self.replay_calls = 0
        self.close_calls = 0

    def replay(self):
        self.replay_calls += 1

    def close(self):
        self.close_calls += 1

    def receipt(self):
        value = {
            "path": str(self.path),
            "sha256": self.sha256,
            "identity": dict(self.identity),
        }
        value["authority_digest"] = wrapper._object_sha256(value)
        return value


class _FakeAssets:
    def __init__(self, arm: str):
        self.cli = wrapper.OracleCLI(
            arm=arm,
            scaffold="/canonical/scaffold.json",
            scaffold_sha256="b" * 64,
            scaffold_digest="c" * 64,
            bone_removed_video="/canonical/bone_removed.mp4",
            bone_removed_video_sha256="d" * 64,
        )
        self.scaffold = {
            "authority": {
                "source_video": {"sha256": "1" * 64, "size": 1},
                "bone_removed_auxiliary_video": {
                    "sha256": self.cli.bone_removed_video_sha256,
                    "size": 1,
                },
                "stage0_receipt": {"sha256": "2" * 64, "size": 1},
                "g0_sparse_annotations": {"sha256": "3" * 64, "size": 1},
            }
        }
        self.scaffold_file = _FakeAuthority(
            path=self.cli.scaffold, sha256=self.cli.scaffold_sha256
        )
        self.aux_file = _FakeAuthority(
            path=self.cli.bone_removed_video,
            sha256=self.cli.bone_removed_video_sha256,
        )
        self.projection_module = projection
        self.projection_source = _FakeAuthority(sha256=wrapper.PROJECTION_SHA256)
        self.scaffold_module = types.SimpleNamespace()
        self.scaffold_source = _FakeAuthority(sha256="e" * 64)
        self.legacy_source = _FakeAuthority(
            sha256=wrapper.LEGACY_INFER_LORA_SHA256
        )
        self.closed = False

    def producer_hashes(self):
        return {
            "wrapper_source_sha256": "f" * 64,
            "legacy_infer_lora_source_sha256": wrapper.LEGACY_INFER_LORA_SHA256,
            "projection_source_sha256": wrapper.PROJECTION_SHA256,
            "scaffold_source_sha256": "e" * 64,
        }

    def close(self):
        self.closed = True

    def replay_all(self):
        self.scaffold_file.replay()
        self.aux_file.replay()


def _base_receipt(trace=None):
    value = {
        "schema_version": "legacy-v5",
        "input": {
            "accepted_model_conditions": ["source_video", "edit_instruction"],
            "external_mask_or_swept_tube": False,
            "external_tracking_pose_or_trajectory": False,
            "reference_image_or_video": False,
        },
        "sampling": {},
    }
    if trace is not None:
        value["sampling"]["source_onset_solver_trace"] = trace
    value["receipt_digest"] = wrapper._object_sha256(value)
    return value


def _minimal_legacy(main_function):
    trainer = types.SimpleNamespace()

    def activate(*args, **kwargs):
        return (args, kwargs)

    trainer.activate_source_trees = activate

    @contextmanager
    def clamp(*args, **kwargs):
        yield types.SimpleNamespace(as_dict=lambda: {"legacy": True})

    return types.SimpleNamespace(
        main=main_function,
        trainer=trainer,
        hard_phase0_source_trajectory_clamp=clamp,
        build_inference_receipt=lambda *args, **kwargs: _base_receipt(),
    )


class NullAndRouteTests(unittest.TestCase):
    def test_projection_consensus_rejects_rank_split_arm_and_rows(self) -> None:
        common = {
            "plan_digest": "1" * 64,
            "tensor_authority": {"content_contract_digest": "2" * 64},
        }
        bone_specs = [
            {"name": "legacy_phase0_hard1_every_step"},
            {"name": "bone_conservation_all_sigma"},
        ]
        dog_specs = [*bone_specs, {"name": "dog_core_low_mid"}]
        bone = wrapper._projection_contract(
            arm="trajectory_bone_only",
            expected_steps=40,
            row_evidence={
                **common,
                "row_names": [item["name"] for item in bone_specs],
                "row_specs": bone_specs,
            },
        )
        dog = wrapper._projection_contract(
            arm="trajectory_dog_bone",
            expected_steps=40,
            row_evidence={
                **common,
                "row_names": [item["name"] for item in dog_specs],
                "row_specs": dog_specs,
            },
        )
        self.assertNotEqual(
            bone["projection_contract_digest"],
            dog["projection_contract_digest"],
        )
        with self.assertRaisesRegex(
            wrapper.ObjectOracleWrapperError, "arm/row/gate/plan/tensor"
        ):
            wrapper._validate_four_rank_projection_digests(
                bone["projection_contract_digest"],
                [
                    bone["projection_contract_digest"],
                    bone["projection_contract_digest"],
                    dog["projection_contract_digest"],
                    bone["projection_contract_digest"],
                ],
            )

    def test_nonempty_stable_authority_open_read_and_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "authority.bin"
            payload = (b"case01-retained-authority\x00" * 4097) + b"terminal"
            path.write_bytes(payload)
            expected = hashlib.sha256(payload).hexdigest()
            authority = wrapper.StableFileAuthority.open(
                path,
                label="nonempty synthetic authority",
                expected_sha256=expected,
                expected_size=len(payload),
            )
            try:
                self.assertEqual(authority.read_all(), payload)
                authority.replay()
                authority.replay()
                self.assertEqual(authority.receipt()["sha256"], expected)
            finally:
                authority.close()

    def test_stable_authority_rejects_a_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            target = root / "target.bin"
            target.write_bytes(b"authority")
            link = root / "link.bin"
            link.symlink_to(target)
            with self.assertRaisesRegex(
                wrapper.ObjectOracleWrapperError, "canonical|non-symlink"
            ):
                wrapper.StableFileAuthority.open(link, label="synthetic link")

    def test_real_scaffold_route_authority_preflight(self) -> None:
        repo_root = METHOD_ROOT.parents[1]
        scaffold_path = (
            repo_root / "artifacts/case01_oracle_object_trajectory_v1/scaffold.json"
        ).resolve()
        aux_path = (
            repo_root
            / "artifacts/object_grounded_case01_0821_bone_interventions_r4"
            / "videos/bone_removed.mp4"
        ).resolve()
        if not scaffold_path.is_file() or not aux_path.is_file():
            self.skipTest("real case01 oracle authority is not present")
        cli = wrapper.OracleCLI(
            arm="route_off",
            scaffold=str(scaffold_path),
            scaffold_sha256=(
                "7b1bec6e9764a1297bb0029f8fea01ebe4b2deab0acc2c7f07fdee96bc0a098a"
            ),
            scaffold_digest=(
                "5e6156909d8261a23c3add3134059bec20505b682ca0eb13dc88fa8512eeace1"
            ),
            bone_removed_video=str(aux_path),
            bone_removed_video_sha256=(
                "8c525385832586fa7b7fd7ae6e5701c599694d26ee27b502dbf0bb582e55e1c9"
            ),
        )
        legacy_source = _FakeAuthority(
            sha256=wrapper.LEGACY_INFER_LORA_SHA256
        )
        assets = wrapper._prepare_oracle_assets(
            cli,
            [
                "--source-video-sha256",
                "5ed911f66fea3ed2000f507412da75adecb8099b26b71089d0fd2c0ac2982b18",
            ],
            legacy_source=legacy_source,
        )
        try:
            self.assertEqual(
                assets.scaffold["schema_version"],
                "case01-oracle-object-trajectory-scaffold-v1",
            )
            self.assertEqual(
                assets.producer_hashes()["scaffold_source_sha256"],
                wrapper.SCAFFOLD_SOURCE_SHA256,
            )
            plan = wrapper.compile_scaffold_token_plan(assets.scaffold)
            self.assertEqual(plan["bone_correspondence_count"], 377)
            self.assertEqual(plan["bone_origin_clear_token_count"], 187)
            self.assertEqual(plan["dog_core_token_count"], 1548)
            self.assertEqual(
                plan["plan_digest"],
                "7eaef1dbd09e91afb9df109b358f0166757df5ddc2ac59fa09831bfeec955103",
            )
            phase10 = plan["phases"][10]
            self.assertEqual(phase10["typed_stage"], "lift")
            self.assertTrue(
                set(phase10["scaffold_origin"]) & set(phase10["target"])
            )
            self.assertFalse(
                set(phase10["effective_origin"]) & set(phase10["target"])
            )
        finally:
            assets.close()

    def test_off_is_exact_delegate_without_asset_open_or_patch(self) -> None:
        seen = {}

        def fake_main(argv):
            seen["argv"] = list(argv)
            seen["builder"] = legacy.build_inference_receipt
            seen["activate"] = legacy.trainer.activate_source_trees
            seen["clamp"] = legacy.hard_phase0_source_trajectory_clamp
            seen["receipt_bytes"] = json.dumps(
                legacy.build_inference_receipt(), sort_keys=True
            ).encode()
            return 17

        legacy = _minimal_legacy(fake_main)
        originals = (
            legacy.build_inference_receipt,
            legacy.trainer.activate_source_trees,
            legacy.hard_phase0_source_trajectory_clamp,
        )
        legacy_source = _FakeAuthority(
            sha256=wrapper.LEGACY_INFER_LORA_SHA256
        )
        expected_bytes = json.dumps(_base_receipt(), sort_keys=True).encode()
        argv = [
            "--object-oracle-arm",
            "off",
            "--object-oracle-scaffold",
            "/does/not/exist.json",
            "--object-oracle-bone-removed-video",
            "/also/missing.mp4",
            "--source-video",
            "/source.mp4",
        ]
        with mock.patch.object(
            wrapper, "_load_frozen_legacy", return_value=(legacy, legacy_source)
        ), mock.patch.object(
            wrapper,
            "_prepare_oracle_assets",
            side_effect=AssertionError("off opened oracle assets"),
        ):
            self.assertEqual(wrapper.main(argv), 17)
        self.assertEqual(
            seen["argv"], ["--source-video", "/source.mp4"]
        )
        self.assertEqual(seen["receipt_bytes"], expected_bytes)
        self.assertIs(seen["builder"], originals[0])
        self.assertIs(seen["activate"], originals[1])
        self.assertIs(seen["clamp"], originals[2])
        self.assertEqual(legacy_source.close_calls, 1)

    def test_route_off_validates_but_does_not_patch_renderer_or_encode(self) -> None:
        seen = {}

        def fake_main(argv):
            seen["activate"] = legacy.trainer.activate_source_trees
            seen["clamp"] = legacy.hard_phase0_source_trajectory_clamp
            seen["receipt"] = legacy.build_inference_receipt()
            return 0

        legacy = _minimal_legacy(fake_main)
        original_activate = legacy.trainer.activate_source_trees
        original_clamp = legacy.hard_phase0_source_trajectory_clamp
        assets = _FakeAssets("route_off")
        legacy_source = assets.legacy_source
        with mock.patch.object(
            wrapper, "_load_frozen_legacy", return_value=(legacy, legacy_source)
        ), mock.patch.object(
            wrapper, "_prepare_oracle_assets", return_value=assets
        ) as prepared:
            result = wrapper.main(
                [
                    "--object-oracle-arm",
                    "route_off",
                    "--source-video-sha256",
                    "1" * 64,
                ]
            )
        self.assertEqual(result, 0)
        prepared.assert_called_once()
        self.assertIs(seen["activate"], original_activate)
        self.assertIs(seen["clamp"], original_clamp)
        receipt = seen["receipt"]
        self.assertEqual(receipt["schema_version"], wrapper.WRAPPER_RECEIPT_SCHEMA)
        oracle = receipt["object_oracle"]
        self.assertEqual(oracle["status"], "validated_not_consumed")
        self.assertFalse(
            oracle["runtime"]["object_oracle_renderer_or_scheduler_patched"]
        )
        self.assertFalse(oracle["runtime"]["aux_bytes_consumed_by_renderer"])
        self.assertEqual(oracle["runtime"]["vae_encode"]["rank0_aux_original_calls"], 0)
        self.assertIsNone(oracle["runtime"]["projection_trace"])
        self.assertEqual(oracle["runtime"]["projection_collective_gates"], [])
        self.assertEqual(
            oracle["runtime"]["direct_runtime_conditions_consumed"],
            ["source_video", "edit_instruction"],
        )
        self.assertEqual(oracle["runtime"]["oracle_runtime_conditions_consumed"], [])
        self.assertEqual(
            receipt["input"]["direct_runtime_conditions"],
            [
                "source_video",
                "edit_instruction",
                "object_trajectory_scaffold",
                "aux_bone_removed_source",
            ],
        )
        self.assertEqual(
            receipt["input"]["derived_scaffold_authorities"],
            ["stage0_object_masks", "g0_mouth_track"],
        )
        self.assertFalse(receipt["input"]["raw_stage0_masks_accessed_at_runtime"])
        self.assertFalse(receipt["input"]["raw_g0_annotations_accessed_at_runtime"])
        self.assertEqual(
            oracle["runtime"][
                "derived_scaffold_authorities_consumed_directly"
            ],
            [],
        )
        payload = dict(receipt)
        digest = payload.pop("receipt_digest")
        self.assertEqual(digest, wrapper._object_sha256(payload))
        self.assertTrue(assets.closed)

    def test_support_loader_fails_closed_without_a_frozen_producer_hash(self):
        with self.assertRaisesRegex(
            wrapper.ObjectOracleWrapperError, "source SHA-256 is not frozen"
        ):
            wrapper._load_pinned_support(
                wrapper.SCAFFOLD_BASENAME,
                None,
                expected_size=wrapper.SCAFFOLD_SOURCE_SIZE,
                label="scaffold",
                module_name="unused",
            )


@unittest.skipIf(torch is None, "torch is unavailable")
class UniPCMultistepScheduler:
    def __init__(self):
        self.config = {
            "_class_name": "UniPCMultistepScheduler",
            "prediction_type": "flow_prediction",
            "use_flow_sigmas": True,
            "predict_x0": True,
            "final_sigmas_type": "zero",
            "flow_shift": 5.0,
        }
        self.sigmas = None
        self.timesteps = None
        self.step_index = None
        self.native_calls = 0
        self.phase0_model_outputs = []
        self.audit_mask = None
        self.audit_model_outputs = []

    def initialize_runtime_schedule(self):
        self.sigmas = torch.tensor([1.0, 0.5, 0.25, 0.0])
        self.timesteps = torch.tensor([1000.0, 500.0, 250.0])

    def step(self, model_output, timestep, sample, return_dict=True):
        self.native_calls += 1
        self.phase0_model_outputs.append(
            model_output[:, : wrapper.EXPECTED_SPATIAL_TOKENS, :]
            .detach()
            .clone()
            .contiguous()
        )
        if self.audit_mask is not None:
            self.audit_model_outputs.append(
                model_output[:, self.audit_mask, :]
                .detach()
                .clone()
                .contiguous()
            )
        index = 0 if self.step_index is None else int(self.step_index)
        self.step_index = index + 1
        return ((sample.float() - 0.1 * model_output.float()).contiguous(),)


def _scaffold() -> dict:
    latent = []
    for phase in range(wrapper.EXPECTED_LATENT_PHASES):
        phase_zero = phase == 0
        latent.append(
            {
                "phase_index": phase,
                "typed_stage": "hold" if phase >= 16 else "lift",
                "source_bone_tokens": [1],
                "target_bone_tokens": [1] if phase_zero else [2],
                "origin_clear_tokens": [] if phase_zero else [1],
                "dog_identity_core_tokens": [3],
                "target_responsibility_tokens": [1, 4] if phase_zero else [2, 4],
                "bone_token_correspondence": [[1, 1]] if phase_zero else [[1, 2]],
            }
        )
    return {
        "authority": {
            "source_video": {"sha256": "1" * 64, "size": 1},
            "bone_removed_auxiliary_video": {"sha256": "2" * 64, "size": 1},
            "stage0_receipt": {"sha256": "3" * 64, "size": 1},
            "g0_sparse_annotations": {"sha256": "4" * 64, "size": 1},
        },
        "latent_layout": {
            "latent_phases": wrapper.EXPECTED_TOKEN_SHAPE[0],
            "patch_rows": wrapper.EXPECTED_TOKEN_SHAPE[1],
            "patch_cols": wrapper.EXPECTED_TOKEN_SHAPE[2],
            "tokens_per_phase": wrapper.EXPECTED_SPATIAL_TOKENS,
            "packed_token_count": wrapper.EXPECTED_SEGMENT_TOKENS,
            "attention_target_half_offset": wrapper.EXPECTED_SEGMENT_TOKENS,
        },
        "latent_phases": latent,
    }


def _pack(source):
    batch, channels, phases, height, width = map(int, source.shape)
    return (
        source.reshape(batch, channels, phases, height // 2, 2, width // 2, 2)
        .permute(0, 2, 3, 5, 4, 6, 1)
        .reshape(batch, phases * (height // 2) * (width // 2), 64)
        .detach()
        .contiguous()
    )


@unittest.skipIf(torch is None, "torch is unavailable")
class ActiveIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260821)
        self.pipeline = types.ModuleType("bernini.pipeline")
        self.bernini = types.ModuleType("bernini")
        self.bernini.__path__ = []
        self.previous_bernini = sys.modules.get("bernini")
        self.previous_pipeline = sys.modules.get("bernini.pipeline")
        sys.modules["bernini"] = self.bernini
        sys.modules["bernini.pipeline"] = self.pipeline

    def tearDown(self):
        if self.previous_bernini is None:
            sys.modules.pop("bernini", None)
        else:
            sys.modules["bernini"] = self.previous_bernini
        if self.previous_pipeline is None:
            sys.modules.pop("bernini.pipeline", None)
        else:
            sys.modules["bernini.pipeline"] = self.previous_pipeline

    def _legacy_and_assets(
        self, *, arm="trajectory_dog_bone", fail=False, aux_fail=False
    ):
        calls = []
        source_latent = torch.arange(
            16 * 21 * 62 * 60, dtype=torch.float32
        ).reshape(1, 16, 21, 62, 60)
        aux_latent = torch.zeros_like(source_latent)

        def original_encode(vae, value):
            calls.append(value)
            if aux_fail and len(calls) == 2:
                raise RuntimeError("synthetic rank0 aux encode failure")
            return source_latent if len(calls) == 1 else aux_latent

        self.pipeline._vae_encode = original_encode

        def prepare_exact_source(path):
            return torch.ones(1, 3, 1, 2, 2), {
                "source_derived_bucket_hw": [2, 2]
            }

        trainer = types.SimpleNamespace()

        def activate(*args, **kwargs):
            return None

        trainer.activate_source_trees = activate
        scheduler = UniPCMultistepScheduler()
        diffusion = types.SimpleNamespace(use_unipc=True, scheduler=scheduler)
        legacy = types.SimpleNamespace(
            trainer=trainer,
            prepare_exact_source=prepare_exact_source,
            _pack_wan_source_latent=_pack,
            build_inference_receipt=None,
            hard_phase0_source_trajectory_clamp=None,
        )

        def build_receipt(*args, **kwargs):
            return _base_receipt(legacy.last_trace)

        legacy.build_inference_receipt = build_receipt

        @contextmanager
        def original_clamp(*args, **kwargs):
            yield types.SimpleNamespace(as_dict=lambda: {"legacy": True})

        legacy.hard_phase0_source_trajectory_clamp = original_clamp

        def main(argv):
            legacy.trainer.activate_source_trees("bernini", "veomni")
            if fail:
                raise LookupError("synthetic active failure")
            source_tensor = torch.zeros(1, 3, 1, 2, 2)
            encoded = sys.modules["bernini.pipeline"]._vae_encode(
                object(), source_tensor
            )
            legacy.source_encoded_returned = encoded is source_latent
            with legacy.hard_phase0_source_trajectory_clamp(
                diffusion, encoded, expected_steps=3
            ) as trace:
                # Mirrors Bernini: model.sample initializes UniPC only after
                # the outer clamp context has already been entered.
                scheduler.initialize_runtime_schedule()
                sample = torch.randn(
                    1,
                    wrapper.EXPECTED_SEGMENT_TOKENS,
                    wrapper.PACKED_CHANNELS,
                ).contiguous()
                for index in range(3):
                    output = torch.randn_like(sample).contiguous()
                    sample = scheduler.step(
                        output,
                        scheduler.timesteps[index],
                        sample,
                        return_dict=False,
                    )[0]
            legacy.last_trace = trace.as_dict()
            legacy.final_sample = sample
            legacy.receipt = legacy.build_inference_receipt()
            return 0

        legacy.main = main
        assets = _FakeAssets(arm)
        assets.scaffold = _scaffold()
        # A real open descriptor makes /proc/self/fd/N available to the VAE wrapper.
        temporary = tempfile.TemporaryFile()
        assets.aux_file.descriptor = temporary.fileno()
        assets._temporary = temporary
        return legacy, assets, calls, source_latent, aux_latent, scheduler

    def test_active_main_encodes_source_once_aux_once_and_projects_lazy_rows(self):
        legacy, assets, calls, source, aux, scheduler = self._legacy_and_assets()
        legacy_source = assets.legacy_source
        original_encode = self.pipeline._vae_encode
        original_activate = legacy.trainer.activate_source_trees
        original_clamp = legacy.hard_phase0_source_trajectory_clamp
        original_receipt = legacy.build_inference_receipt
        def gather_equal(output, value):
            output[:] = [value] * 4

        with mock.patch.object(
            wrapper, "_load_frozen_legacy", return_value=(legacy, legacy_source)
        ), mock.patch.object(
            wrapper, "_prepare_oracle_assets", return_value=assets
        ), mock.patch.object(
            torch.distributed, "get_rank", return_value=0
        ), mock.patch.object(
            torch.distributed, "broadcast"
        ) as broadcast, mock.patch.object(
            torch.distributed, "get_world_size", return_value=4
        ), mock.patch.object(
            torch.distributed, "all_gather_object", side_effect=gather_equal
        ) as gather, mock.patch.object(
            wrapper,
            "_linux_retained_fd_consumer_path",
            return_value=Path("/synthetic-retained-aux"),
        ):
            result = wrapper.main(
                [
                    "--object-oracle-arm",
                    "trajectory_dog_bone",
                    "--source-onset-policy",
                    "hard1_every_step",
                ]
            )
        self.assertEqual(result, 0)
        self.assertEqual(len(calls), 2)
        broadcast.assert_called_once()
        self.assertEqual(gather.call_count, 10)
        self.assertEqual(scheduler.native_calls, 3)
        packed_source = _pack(source)
        packed_aux = _pack(aux)
        for phase in range(wrapper.EXPECTED_LATENT_PHASES):
            offset = phase * wrapper.EXPECTED_SPATIAL_TOKENS
            if phase == 0:
                self.assertTrue(
                    torch.equal(
                        legacy.final_sample[:, offset + 1], packed_source[:, offset + 1]
                    )
                )
                self.assertTrue(
                    torch.equal(
                        legacy.final_sample[:, offset + 2], packed_source[:, offset + 2]
                    )
                )
            else:
                self.assertTrue(
                    torch.equal(
                        legacy.final_sample[:, offset + 1], packed_aux[:, offset + 1]
                    )
                )
                self.assertTrue(
                    torch.equal(
                        legacy.final_sample[:, offset + 2], packed_source[:, offset + 1]
                    )
                )
            self.assertTrue(
                torch.equal(
                    legacy.final_sample[:, offset + 3], packed_source[:, offset + 3]
                )
            )
        trace = legacy.last_trace
        self.assertEqual(trace["row_construction"]["row_names"], [
            "legacy_phase0_hard1_every_step",
            "bone_conservation_all_sigma",
            "dog_core_low_mid",
        ])
        self.assertTrue(
            trace["tensor_core"]["initial_noise_captured_from_first_native_sample"]
        )
        self.assertEqual(trace["tensor_core"]["step_count"], 3)
        tensor_authority = trace["row_construction"]["tensor_authority"]
        self.assertGreater(
            tensor_authority[
                "source_aux_effective_origin_differing_element_count"
            ],
            0,
        )
        self.assertTrue(
            trace["row_construction"]["projection_contract"][
                "four_rank_consensus"
            ][
                "all_ranks_exact_projection_contract_equal"
            ]
        )
        self.assertEqual(
            [
                gate["stage"]
                for gate in trace["projection_collective_gates"]
            ],
            [
                "projection_runtime_readiness",
                "projection_row_build",
                "projection_contract_build",
                "projection_projector_lookup",
                "projection_lazy_bootstrap_install",
                "projection_projector_install",
                "projection_final_validation",
            ],
        )
        self.assertEqual(
            trace["row_construction"]["projector_install_gate"]["stage"],
            "projection_projector_install",
        )
        self.assertEqual(
            trace["row_construction"]["final_validation_gate"]["stage"],
            "projection_final_validation",
        )
        receipt = legacy.receipt
        self.assertEqual(receipt["object_oracle"]["status"], "consumed_projection")
        self.assertEqual(
            receipt["object_oracle"]["runtime"]["vae_encode"],
            {
                "rank0_source_original_calls": 1,
                "rank0_aux_attempts": 1,
                "rank0_aux_original_calls": 1,
            },
        )
        self.assertEqual(
            receipt["sampling"]["source_onset_policy"],
            "case01_object_trajectory_oracle_v3",
        )
        self.assertEqual(
            receipt["input"]["accepted_model_conditions"],
            [
                "source_video",
                "edit_instruction",
                "stage0_object_masks",
                "g0_mouth_track",
                "object_trajectory_scaffold",
                "aux_bone_removed_source",
            ],
        )
        self.assertEqual(
            receipt["input"]["direct_runtime_conditions"],
            [
                "source_video",
                "edit_instruction",
                "object_trajectory_scaffold",
                "aux_bone_removed_source",
            ],
        )
        self.assertEqual(
            receipt["input"]["derived_scaffold_authorities"],
            ["stage0_object_masks", "g0_mouth_track"],
        )
        self.assertFalse(receipt["input"]["raw_stage0_masks_accessed_at_runtime"])
        self.assertFalse(receipt["input"]["raw_g0_annotations_accessed_at_runtime"])
        self.assertEqual(
            receipt["object_oracle"]["runtime"][
                "direct_runtime_conditions_consumed"
            ],
            [
                "source_video",
                "edit_instruction",
                "object_trajectory_scaffold",
                "aux_bone_removed_source",
            ],
        )
        self.assertEqual(
            receipt["object_oracle"]["runtime"][
                "oracle_runtime_conditions_consumed"
            ],
            ["object_trajectory_scaffold", "aux_bone_removed_source"],
        )
        self.assertEqual(
            receipt["object_oracle"]["runtime"][
                "derived_scaffold_authorities_consumed_directly"
            ],
            [],
        )
        self.assertIs(self.pipeline._vae_encode, original_encode)
        self.assertIs(legacy.trainer.activate_source_trees, original_activate)
        self.assertIs(legacy.hard_phase0_source_trajectory_clamp, original_clamp)
        self.assertIs(legacy.build_inference_receipt, original_receipt)
        assets._temporary.close()

    def test_active_phase0_matches_frozen_acc46_clamp_at_every_step(self):
        frozen_path = METHOD_ROOT / wrapper.LEGACY_BASENAME
        if not frozen_path.is_file():
            frozen_path = Path(
                "/private/tmp/r5f-source-snapshot.DqHeQL/methods/"
                "bernini_action_editing/infer_lora.py"
            )
        if not frozen_path.is_file():
            self.skipTest("frozen acc46 legacy source fixture is unavailable")
        authority = wrapper.StableFileAuthority.open(
            frozen_path.resolve(),
            label="frozen acc46 parity fixture",
            expected_sha256=wrapper.LEGACY_INFER_LORA_SHA256,
            expected_size=wrapper.LEGACY_INFER_LORA_SIZE,
        )
        try:
            frozen = wrapper._load_module_from_authority(
                authority, module_name="_test_frozen_acc46_phase0_parity"
            )
        finally:
            authority.close()
        scaffold_path = (
            METHOD_ROOT.parents[1]
            / "artifacts/case01_oracle_object_trajectory_v1/scaffold.json"
        )
        if not scaffold_path.is_file():
            self.skipTest("real case01 scaffold fixture is unavailable")
        scaffold = json.loads(scaffold_path.read_text(encoding="utf-8"))
        source = torch.arange(
            16 * 21 * 62 * 60, dtype=torch.float32
        ).reshape(1, 16, 21, 62, 60)
        aux = torch.zeros_like(source)
        source_packed = frozen._pack_wan_source_latent(source)
        aux_packed = frozen._pack_wan_source_latent(aux)
        self.assertTrue(torch.equal(source_packed, _pack(source)))
        generator = torch.Generator().manual_seed(20260821)
        noise = torch.randn(
            1,
            wrapper.EXPECTED_SEGMENT_TOKENS,
            wrapper.PACKED_CHANNELS,
            generator=generator,
        ).contiguous()
        outputs = [
            torch.randn(
                noise.shape, generator=generator, dtype=noise.dtype
            ).contiguous()
            for _ in range(3)
        ]

        for arm in ("trajectory_bone_only", "trajectory_dog_bone"):
            rows, evidence = wrapper.build_projection_rows(
                arm=arm,
                scaffold=scaffold,
                source_packed=source_packed,
                aux_packed=aux_packed,
                projection_module=projection,
            )
            baseline, bone, *dog = rows
            phase0_mask = baseline.projection_weights[:, :930].to(dtype=torch.bool)
            bone_overlap = phase0_mask & bone.projection_weights[:, :930].to(
                dtype=torch.bool
            )
            self.assertEqual(int(bone_overlap.count_nonzero().item()), 17)
            expanded_bone = bone_overlap.expand(-1, -1, 64)
            self.assertTrue(
                torch.equal(
                    baseline.clean_packed[:, :930][expanded_bone],
                    bone.clean_packed[:, :930][expanded_bone],
                )
            )
            if dog:
                dog_overlap = phase0_mask & dog[0].projection_weights[:, :930].to(
                    dtype=torch.bool
                )
                self.assertEqual(int(dog_overlap.count_nonzero().item()), 112)
                expanded_dog = dog_overlap.expand(-1, -1, 64)
                self.assertTrue(
                    torch.equal(
                        baseline.clean_packed[:, :930][expanded_dog],
                        dog[0].clean_packed[:, :930][expanded_dog],
                    )
                )

            selected = torch.zeros(
                wrapper.EXPECTED_SEGMENT_TOKENS, dtype=torch.bool
            )
            for row in rows:
                selected |= row.projection_weights[0, :, 0].cpu().to(dtype=torch.bool)
            unselected_nonphase = ~selected
            unselected_nonphase[: wrapper.EXPECTED_SPATIAL_TOKENS] = False
            self.assertGreater(
                int(unselected_nonphase.count_nonzero().item()), 0
            )

            legacy_scheduler = UniPCMultistepScheduler()
            active_scheduler = UniPCMultistepScheduler()
            native_scheduler = UniPCMultistepScheduler()
            active_scheduler.audit_mask = unselected_nonphase
            native_scheduler.audit_mask = unselected_nonphase
            for scheduler in (legacy_scheduler, active_scheduler, native_scheduler):
                scheduler.initialize_runtime_schedule()
            legacy_diffusion = types.SimpleNamespace(
                use_unipc=True, scheduler=legacy_scheduler
            )
            legacy_samples = []
            sample = noise.clone().contiguous()
            with frozen.hard_phase0_source_trajectory_clamp(
                legacy_diffusion, source, expected_steps=3
            ):
                for index, model_output in enumerate(outputs):
                    sample = legacy_scheduler.step(
                        model_output,
                        legacy_scheduler.timesteps[index],
                        sample,
                        return_dict=False,
                    )[0]
                    legacy_samples.append(sample.detach().clone())

            active_samples = []
            sample = noise.clone().contiguous()
            with projection.project_object_trajectory_unipc_steps(
                active_scheduler,
                rows=rows,
                initial_noise=None,
                source_token_count=wrapper.EXPECTED_SEGMENT_TOKENS,
                target_token_count=wrapper.EXPECTED_SEGMENT_TOKENS,
                expected_steps=3,
            ) as active_trace:
                for index, model_output in enumerate(outputs):
                    sample = active_scheduler.step(
                        model_output,
                        active_scheduler.timesteps[index],
                        sample,
                        return_dict=False,
                    )[0]
                    active_samples.append(sample.detach().clone())

            native_samples = []
            sample = noise.clone().contiguous()
            for index, model_output in enumerate(outputs):
                sample = native_scheduler.step(
                    model_output,
                    native_scheduler.timesteps[index],
                    sample,
                    return_dict=False,
                )[0]
                native_samples.append(sample.detach().clone())

            for index in range(3):
                self.assertTrue(
                    torch.equal(
                        legacy_scheduler.phase0_model_outputs[index],
                        active_scheduler.phase0_model_outputs[index],
                    )
                )
                self.assertTrue(
                    torch.equal(
                        legacy_samples[index][:, :930],
                        active_samples[index][:, :930],
                    )
                )
                self.assertTrue(
                    torch.equal(
                        active_scheduler.audit_model_outputs[index],
                        native_scheduler.audit_model_outputs[index],
                    )
                )
                self.assertTrue(
                    torch.equal(
                        active_samples[index][:, unselected_nonphase],
                        native_samples[index][:, unselected_nonphase],
                    )
                )
            self.assertTrue(
                torch.equal(active_samples[-1][:, :930], source_packed[:, :930])
            )
            self.assertTrue(
                all(record.unselected_velocity_exact for record in active_trace.records)
            )
            self.assertTrue(
                all(record.unselected_post_step_exact for record in active_trace.records)
            )
            self.assertTrue(evidence["matched_legacy_phase0_baseline"])

    def test_real_world4_operational_failpoints_reject_without_publication(self):
        if not torch.distributed.is_available():
            self.skipTest("torch.distributed is unavailable")
        if not torch.distributed.is_gloo_available():
            self.skipTest("Torch Gloo backend is unavailable")

        context = multiprocessing.get_context("spawn")
        scenarios = (
            "rank0_aux_encode",
            "nonzero_aux_alloc",
            "nonzero_aux_abi",
            "post_broadcast_validation",
            "projection_runtime_readiness",
            "row_build_tensor_hash",
            "projection_contract_build",
            "lazy_bootstrap_install",
            "projector_install",
            "final_scheduler_validation",
        )
        expected_failure_stage = {
            "rank0_aux_encode": "failed aux_readiness",
            "nonzero_aux_alloc": "failed aux_readiness",
            "nonzero_aux_abi": "failed aux_readiness",
            "post_broadcast_validation": "failed aux_post_broadcast",
            "projection_runtime_readiness": (
                "failed projection_runtime_readiness"
            ),
            "row_build_tensor_hash": "failed projection_row_build",
            "projection_contract_build": "failed projection_contract_build",
            "lazy_bootstrap_install": (
                "failed projection_lazy_bootstrap_install"
            ),
            "projector_install": "failed projection_projector_install",
            "final_scheduler_validation": "failed projection_final_validation",
        }
        for scenario in scenarios:
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as root:
                root_path = Path(root)
                init_path = root_path / f"gloo-init-{scenario}"
                output_path = root_path / f"must-not-publish-{scenario}.mp4"
                receipt_path = root_path / f"must-not-publish-{scenario}.json"
                result_queue = context.Queue()
                processes = [
                    context.Process(
                        target=_world4_operational_failpoint_worker,
                        args=(
                            rank,
                            scenario,
                            str(init_path),
                            str(output_path),
                            str(receipt_path),
                            result_queue,
                        ),
                    )
                    for rank in range(4)
                ]
                started = time.monotonic()
                for process in processes:
                    process.start()
                deadline = started + 20.0
                for process in processes:
                    process.join(max(0.0, deadline - time.monotonic()))
                still_alive = [process for process in processes if process.is_alive()]
                for process in still_alive:
                    process.terminate()
                reap_deadline = started + 25.0
                for process in still_alive:
                    process.join(max(0.0, reap_deadline - time.monotonic()))
                self.assertEqual(still_alive, [], "world4 gate timed out")
                self.assertEqual([process.exitcode for process in processes], [0] * 4)
                queue_deadline = started + 30.0
                results = [
                    result_queue.get(
                        timeout=max(0.01, queue_deadline - time.monotonic())
                    )
                    for _ in range(4)
                ]
                self.assertLess(time.monotonic() - started, 30.0)
                self.assertEqual(
                    sorted(result["rank"] for result in results), list(range(4))
                )
                self.assertTrue(all(result["rejected"] for result in results))
                self.assertTrue(
                    all(
                        expected_failure_stage[scenario]
                        in result["rejection_text"]
                        for result in results
                    )
                )
                self.assertTrue(
                    all(not result["output_created"] for result in results)
                )
                self.assertTrue(
                    all(not result["receipt_created"] for result in results)
                )
                self.assertTrue(
                    all(
                        result["scheduler_instance_step_present"] in (None, False)
                        for result in results
                    )
                )
                self.assertFalse(output_path.exists())
                self.assertFalse(receipt_path.exists())
                result_queue.close()
                result_queue.join_thread()

    def test_rank0_aux_failure_returns_source_then_all_rank_gate_rejects(self):
        legacy, assets, calls, source, _, scheduler = self._legacy_and_assets(
            aux_fail=True
        )
        legacy_source = assets.legacy_source
        original_encode = self.pipeline._vae_encode

        def gather_same(output, value):
            output[:] = [value] * 4

        with mock.patch.object(
            wrapper, "_load_frozen_legacy", return_value=(legacy, legacy_source)
        ), mock.patch.object(
            wrapper, "_prepare_oracle_assets", return_value=assets
        ), mock.patch.object(
            torch.distributed, "get_rank", return_value=0
        ), mock.patch.object(
            torch.distributed, "get_world_size", return_value=4
        ), mock.patch.object(
            torch.distributed, "all_gather_object", side_effect=gather_same
        ) as gather, mock.patch.object(
            torch.distributed, "broadcast"
        ) as broadcast, mock.patch.object(
            wrapper,
            "_linux_retained_fd_consumer_path",
            return_value=Path("/synthetic-retained-aux"),
        ):
            with self.assertRaisesRegex(
                wrapper.ObjectOracleWrapperError, "failed aux_readiness"
            ):
                wrapper.main(
                    [
                        "--object-oracle-arm",
                        "trajectory_dog_bone",
                        "--source-onset-policy",
                        "hard1_every_step",
                    ]
                )
        self.assertEqual(len(calls), 2)
        self.assertTrue(legacy.source_encoded_returned)
        self.assertEqual(scheduler.native_calls, 0)
        self.assertEqual(gather.call_count, 2)
        broadcast.assert_not_called()
        self.assertIs(self.pipeline._vae_encode, original_encode)
        self.assertTrue(assets.closed)
        assets._temporary.close()

    def test_bone_only_row_omits_dog_authority(self):
        source = torch.randn(
            1, wrapper.EXPECTED_SEGMENT_TOKENS, 64
        ).contiguous()
        aux = torch.randn_like(source).contiguous()
        rows, evidence = wrapper.build_projection_rows(
            arm="trajectory_bone_only",
            scaffold=_scaffold(),
            source_packed=source,
            aux_packed=aux,
            projection_module=projection,
        )
        self.assertEqual(
            [row.name for row in rows],
            ["legacy_phase0_hard1_every_step", "bone_conservation_all_sigma"],
        )
        self.assertFalse(evidence["dog_row_consumed"])
        self.assertGreater(evidence["dog_core_token_count"], 0)

    def test_all_temporary_patches_restore_when_legacy_raises(self):
        legacy, assets, _, _, _, _ = self._legacy_and_assets(fail=True)
        legacy_source = assets.legacy_source
        original_encode = self.pipeline._vae_encode
        original_activate = legacy.trainer.activate_source_trees
        original_clamp = legacy.hard_phase0_source_trajectory_clamp
        original_receipt = legacy.build_inference_receipt
        with mock.patch.object(
            wrapper, "_load_frozen_legacy", return_value=(legacy, legacy_source)
        ), mock.patch.object(
            wrapper, "_prepare_oracle_assets", return_value=assets
        ):
            with self.assertRaisesRegex(LookupError, "synthetic active failure"):
                wrapper.main(
                    [
                        "--object-oracle-arm",
                        "trajectory_dog_bone",
                        "--source-onset-policy",
                        "hard1_every_step",
                    ]
                )
        self.assertIs(self.pipeline._vae_encode, original_encode)
        self.assertIs(legacy.trainer.activate_source_trees, original_activate)
        self.assertIs(legacy.hard_phase0_source_trajectory_clamp, original_clamp)
        self.assertIs(legacy.build_inference_receipt, original_receipt)
        self.assertTrue(assets.closed)
        assets._temporary.close()


if __name__ == "__main__":
    unittest.main()
