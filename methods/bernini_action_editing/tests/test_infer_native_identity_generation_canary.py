from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_native_identity_generation_canary as canary


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
SAFETENSORS_AVAILABLE = importlib.util.find_spec("safetensors") is not None


class NativeIdentityGenerationCanaryContractTests(unittest.TestCase):
    @staticmethod
    def _valid_world4_load_gate() -> dict[str, object]:
        return {
            "schema_version": canary.WORLD4_LOAD_COMPLETION_GATE_SCHEMA,
            "world_size": 4,
            "hostname": "test-node",
            "ranks": [0, 1, 2, 3],
            "renderer_gpu_resident_trimmed_monotonic_ns_by_rank": [
                101,
                102,
                103,
                104,
            ],
            "load_completion_barrier_returned_monotonic_ns_by_rank": [
                201,
                202,
                203,
                204,
            ],
            "source_tokenizer_setup_entered_monotonic_ns_by_rank": [
                301,
                302,
                303,
                304,
            ],
            "native_sampling_entered_monotonic_ns_by_rank": [401, 402, 403, 404],
            "world4_barrier_completed_before_source_tokenizer_setup": True,
            "all_four_renderer_loads_complete_before_any_source_tokenizer_setup": True,
            "all_four_renderer_loads_complete_before_first_native_sampling": True,
        }

    @staticmethod
    def _valid_t5_residency_gate() -> dict[str, object]:
        rows = []
        for rank in range(4):
            rows.append(
                {
                    "rank": rank,
                    "local_rank": rank,
                    "hostname": "test-node",
                    "guard_required": True,
                    "guard_active": True,
                    "module_path": "model.t5_text_encoder",
                    "exact_positional_cpu_offload_request_only": True,
                    "cpu_offload_requests_observed": 1,
                    "cpu_offload_requests_suppressed": 1,
                    "successful_cpu_materializations": 0,
                    "delegated_to_requests": 1,
                    "parameter_device_before": f"cuda:{rank}",
                    "parameter_device_after": f"cuda:{rank}",
                    "storage_fingerprint_before": f"{rank + 1:064x}",
                    "storage_fingerprint_after": f"{rank + 1:064x}",
                    "guard_method_restored": True,
                    "vmrss_kib": 1000 + rank,
                    "vmhwm_kib": 2000 + rank,
                    "gpu_memory_limit_gib": canary.T2V_GPU_MEMORY_LIMIT_GIB,
                    "gpu_memory_limit_bytes": canary.T2V_GPU_MEMORY_LIMIT_BYTES,
                    "gpu_total_memory_bytes": 64 * 1024**3,
                    "gpu_peak_allocated_bytes": 31 * 1024**3 + rank,
                    "gpu_peak_reserved_bytes": 32 * 1024**3 + rank,
                    "gpu_peak_reserved_within_limit": True,
                }
            )
        return {
            "schema_version": canary.T2V_TEXT_ENCODER_GPU_RESIDENCY_GATE_SCHEMA,
            "world_size": 4,
            "hostname": "test-node",
            "ranks": [0, 1, 2, 3],
            "module_path": "model.t5_text_encoder",
            "rank_evidence": rows,
            "all_rank_exactly_one_cpu_offload_request_suppressed": True,
            "all_rank_zero_successful_cpu_materializations": True,
            "all_rank_gpu_resident_before_and_after_sampling": True,
            "all_rank_storage_fingerprint_unchanged": True,
            "all_rank_guard_method_restored": True,
            "all_rank_peak_reserved_within_52_gib": True,
        }

    @staticmethod
    def _valid_cuda_memory(_: object) -> dict[str, int]:
        return {
            "gpu_total_memory_bytes": 64 * 1024**3,
            "gpu_peak_allocated_bytes": 31 * 1024**3,
            "gpu_peak_reserved_bytes": 32 * 1024**3,
        }

    def _valid_args(self, **overrides: object) -> argparse.Namespace:
        prompt = "A small brown dog picks up a bone and holds it in its mouth."
        values: dict[str, object] = {
            "bernini_root": "/tmp/bernini",
            "veomni_root": "/tmp/veomni",
            "checkpoint": "/tmp/checkpoint",
            "checkpoint_content_manifest": "/tmp/checkpoint.sha256",
            "source_video": "/tmp/source.mp4",
            "expected_source_sha256": "1" * 64,
            "action_prompt": prompt,
            "expected_action_prompt_sha256": hashlib.sha256(
                prompt.encode("utf-8")
            ).hexdigest(),
            "output_dir": "/tmp/native-canary-output",
            "arms": ["t2v", "r2v", "rv2v"],
            "num_inference_steps": 40,
            "seed": 2027,
            "expected_bernini_commit": canary.legacy.trainer.BERNINI_OFFICIAL_COMMIT,
            "expected_veomni_commit": canary.legacy.trainer.VEOMNI_TESTED_COMMIT,
            "expected_checkpoint_tree_sha256": canary.legacy.trainer.CHECKPOINT_TREE_SHA256,
            "method_source_revision": "2" * 40,
            "method_source_archive_sha256": "3" * 64,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_exact81_reference_indices_cover_the_whole_source(self) -> None:
        self.assertEqual(canary.FRAME_COUNT, 81)
        self.assertEqual(canary.LATENT_FRAME_COUNT, 21)
        self.assertEqual(canary.R2V_REFERENCE_INDICES, (0, 20, 40, 60, 80))
        self.assertEqual(canary.RV2V_REFERENCE_INDICES, (0, 27, 53, 80))
        self.assertEqual(canary.canonical_reference_indices(81, 1), (40,))
        with self.assertRaises(canary.NativeIdentityCanaryError):
            canary.canonical_reference_indices(0, 5)

    def test_resource_lifecycle_is_native_receipt_v4_and_environment_truthful(
        self,
    ) -> None:
        self.assertEqual(
            canary.SCHEMA_VERSION,
            "bernini-native-identity-generation-canary-v2",
        )
        with mock.patch.dict(
            os.environ,
            {
                "NATIVE_SERIALIZED_HOST_LOAD_REQUIRED": "1",
                "NATIVE_V_AXIS_LOAD_LOCK": "/tmp/renderer-load.lock",
                "NATIVE_T2V_KEEP_T5_ON_RANK_GPU_REQUIRED": "1",
            },
            clear=True,
        ):
            receipt = canary._resource_lifecycle_receipt(
                t2v_vae_deferred_until_post_sampling=True,
                world4_load_completion_gate=self._valid_world4_load_gate(),
                world4_t2v_text_encoder_gpu_residency_gate=(
                    self._valid_t5_residency_gate()
                ),
            )
            self.assertEqual(
                {
                    field: receipt[field]
                    for field in canary.T2V_RESOURCE_LIFECYCLE_CONTRACT
                },
                canary.T2V_RESOURCE_LIFECYCLE_CONTRACT,
            )
            canary.validate_t2v_resource_lifecycle(
                receipt, require_serialized_load=True
            )
        with mock.patch.dict(os.environ, {}, clear=True):
            optional = canary._resource_lifecycle_receipt(
                t2v_vae_deferred_until_post_sampling=True,
                world4_load_completion_gate=self._valid_world4_load_gate(),
                world4_t2v_text_encoder_gpu_residency_gate=None,
            )
        self.assertFalse(optional["serialized_host_checkpoint_load_required"])
        self.assertFalse(
            optional["renderer_deserialized_and_moved_to_rank_gpu_under_lock"]
        )
        self.assertFalse(
            optional["host_allocator_trim_called_before_load_lock_release"]
        )
        build_receipt = inspect.getsource(canary._build_receipt)
        self.assertIn('"resource_lifecycle": dict(resource_lifecycle)', build_receipt)

    def test_t2v_guard_suppresses_only_exact_cpu_request_and_restores(self) -> None:
        class Tensor:
            def __init__(self) -> None:
                self.shape = (2, 3)
                self.dtype = "bfloat16"
                self.device = "cuda:0"

            def numel(self) -> int:
                return 6

            def data_ptr(self) -> int:
                return 12345

        class Encoder:
            def __init__(self) -> None:
                self.tensor = Tensor()
                self.calls: list[str] = []

            def named_parameters(self):
                return [("weight", self.tensor)]

            def named_buffers(self):
                return []

            def parameters(self):
                return [self.tensor]

            def to(self, device):
                self.calls.append(str(device))
                self.tensor.device = str(device)
                return self

        class Model:
            def __init__(self) -> None:
                self.t5_text_encoder = Encoder()

        model = Model()
        original_function = model.t5_text_encoder.to.__func__
        with mock.patch.dict(
            os.environ,
            {"NATIVE_T2V_KEEP_T5_ON_RANK_GPU_REQUIRED": "1"},
            clear=True,
        ):
            with canary._t2v_text_encoder_rank_gpu_residency(
                model,
                arm="t2v",
                device="cuda:0",
                memory_reader=lambda: {"vmrss_kib": 100, "vmhwm_kib": 120},
                cuda_peak_reset=lambda _: None,
                cuda_memory_reader=self._valid_cuda_memory,
            ) as evidence:
                model.t5_text_encoder.to("cuda:0")
                returned = model.t5_text_encoder.to("cpu")
                self.assertIs(returned, model.t5_text_encoder)
                self.assertEqual(model.t5_text_encoder.tensor.device, "cuda:0")
        self.assertEqual(model.t5_text_encoder.calls, ["cuda:0"])
        self.assertIs(model.t5_text_encoder.to.__func__, original_function)
        self.assertEqual(evidence["cpu_offload_requests_suppressed"], 1)
        self.assertEqual(evidence["successful_cpu_materializations"], 0)
        self.assertTrue(evidence["guard_method_restored"])
        self.assertLess(
            evidence["gpu_peak_reserved_bytes"],
            canary.T2V_GPU_MEMORY_LIMIT_BYTES,
        )

    def test_t2v_guard_fails_closed_on_wrong_scope_signature_or_call_count(self) -> None:
        class Tensor:
            shape = (1,)
            dtype = "bfloat16"
            device = "cuda:0"
            def numel(self): return 1
            def data_ptr(self): return 7
        class Encoder:
            def named_parameters(self): return [("weight", Tensor())]
            def named_buffers(self): return []
            def parameters(self): return [Tensor()]
            def to(self, device=None, **kwargs): return self
        model = type("Model", (), {"t5_text_encoder": Encoder()})()
        environment = {"NATIVE_T2V_KEEP_T5_ON_RANK_GPU_REQUIRED": "1"}
        with mock.patch.dict(os.environ, environment, clear=True):
            with self.assertRaisesRegex(canary.NativeIdentityCanaryError, "non-T2V"):
                with canary._t2v_text_encoder_rank_gpu_residency(
                    model,
                    arm="r2v",
                    device="cuda:0",
                    cuda_peak_reset=lambda _: None,
                    cuda_memory_reader=self._valid_cuda_memory,
                ):
                    pass
            with self.assertRaisesRegex(
                canary.NativeIdentityCanaryError, "offload signature"
            ):
                with canary._t2v_text_encoder_rank_gpu_residency(
                    model,
                    arm="t2v",
                    device="cuda:0",
                    memory_reader=lambda: {"vmrss_kib": 1, "vmhwm_kib": 1},
                    cuda_peak_reset=lambda _: None,
                    cuda_memory_reader=self._valid_cuda_memory,
                ):
                    model.t5_text_encoder.to(device="cpu")
            with self.assertRaisesRegex(
                canary.NativeIdentityCanaryError, "evidence differs"
            ):
                with canary._t2v_text_encoder_rank_gpu_residency(
                    model,
                    arm="t2v",
                    device="cuda:0",
                    memory_reader=lambda: {"vmrss_kib": 1, "vmhwm_kib": 1},
                    cuda_peak_reset=lambda _: None,
                    cuda_memory_reader=self._valid_cuda_memory,
                ):
                    model.t5_text_encoder.to("cuda:0")

    def test_t2v_guard_restores_method_when_sampler_raises(self) -> None:
        class Tensor:
            shape = (1,)
            dtype = "bfloat16"
            device = "cuda:0"
            def numel(self): return 1
            def data_ptr(self): return 8
        class Encoder:
            def named_parameters(self): return [("weight", Tensor())]
            def named_buffers(self): return []
            def parameters(self): return [Tensor()]
            def to(self, device): return self
        model = type("Model", (), {"t5_text_encoder": Encoder()})()
        original_function = model.t5_text_encoder.to.__func__
        with mock.patch.dict(
            os.environ,
            {"NATIVE_T2V_KEEP_T5_ON_RANK_GPU_REQUIRED": "1"},
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "sampler failed"):
                with canary._t2v_text_encoder_rank_gpu_residency(
                    model,
                    arm="t2v",
                    device="cuda:0",
                    cuda_peak_reset=lambda _: None,
                    cuda_memory_reader=self._valid_cuda_memory,
                ):
                    raise RuntimeError("sampler failed")
        self.assertIs(model.t5_text_encoder.to.__func__, original_function)
        self.assertNotIn("_native_t2v_gpu_residency_guard_active", vars(model.t5_text_encoder))

    def test_t2v_residency_gate_rejects_old_or_hostile_evidence(self) -> None:
        valid = self._valid_t5_residency_gate()
        canary._validate_world4_t2v_text_encoder_gpu_residency_gate(valid)
        old = dict(valid)
        old["schema_version"] = "bernini-native-t2v-text-encoder-gpu-residency-gate-v1"
        with self.assertRaises(canary.NativeIdentityCanaryError):
            canary._validate_world4_t2v_text_encoder_gpu_residency_gate(old)
        hostile = self._valid_t5_residency_gate()
        hostile["rank_evidence"][2]["parameter_device_after"] = "cpu"
        with self.assertRaises(canary.NativeIdentityCanaryError):
            canary._validate_world4_t2v_text_encoder_gpu_residency_gate(hostile)
        over_limit = self._valid_t5_residency_gate()
        over_limit["rank_evidence"][1]["gpu_peak_reserved_bytes"] = (
            canary.T2V_GPU_MEMORY_LIMIT_BYTES + 1
        )
        with self.assertRaises(canary.NativeIdentityCanaryError):
            canary._validate_world4_t2v_text_encoder_gpu_residency_gate(over_limit)

    def test_world4_hostile_last_rank_load_blocks_early_setup_and_sampling(
        self,
    ) -> None:
        class World:
            def __init__(self) -> None:
                self.sync = threading.Barrier(4)
                self.lock = threading.Lock()
                self.rows: dict[int, dict[int, object]] = {}

            def dist(self, rank: int) -> object:
                world = self

                class Dist:
                    call_index = 0

                    def barrier(self) -> None:
                        world.sync.wait(timeout=5)

                    def all_gather_object(
                        self, output: list[object], local: object
                    ) -> None:
                        generation = self.call_index
                        with world.lock:
                            world.rows.setdefault(generation, {})[rank] = local
                        world.sync.wait(timeout=5)
                        with world.lock:
                            for item_rank in range(4):
                                output[item_rank] = world.rows[generation][item_rank]
                        world.sync.wait(timeout=5)
                        self.call_index += 1

                return Dist()

        world = World()
        events: dict[int, dict[str, int]] = {}
        evidences: dict[int, dict[str, object]] = {}

        def worker(rank: int) -> None:
            # r10's hostile case: rank 3 is still deserializing while ranks
            # 0-2 would otherwise start tokenizer/source allocation.
            time.sleep(0.18 if rank == 3 else 0.01 * rank)
            loaded = time.monotonic_ns()
            distributed = world.dist(rank)
            gate = canary._world4_renderer_load_completion_barrier(
                distributed,
                rank=rank,
                world_size=4,
                renderer_gpu_resident_trimmed_monotonic_ns=loaded,
                hostname="hostile-world4",
            )
            setup = time.monotonic_ns()
            time.sleep(0.01 * (3 - rank))
            gate = canary._complete_world4_load_completion_gate_before_sampling(
                distributed,
                gate,
                rank=rank,
                world_size=4,
            )
            sampled = time.monotonic_ns()
            events[rank] = {"loaded": loaded, "setup": setup, "sampled": sampled}
            evidences[rank] = gate

        threads = [threading.Thread(target=worker, args=(rank,)) for rank in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(len(events), 4)
        last_load = max(row["loaded"] for row in events.values())
        self.assertGreater(min(row["setup"] for row in events.values()), last_load)
        self.assertGreater(min(row["sampled"] for row in events.values()), last_load)
        self.assertTrue(
            all(evidence == evidences[0] for evidence in evidences.values())
        )
        canary._validate_world4_load_completion_gate(
            evidences[0], sampling_required=True
        )

    def test_world4_load_gate_rejects_resigned_old_schema_and_hostile_order(
        self,
    ) -> None:
        old = self._valid_world4_load_gate()
        old["schema_version"] = "bernini-native-world4-renderer-load-completion-gate-v0"
        with self.assertRaisesRegex(canary.NativeIdentityCanaryError, "ordering"):
            canary._validate_world4_load_completion_gate(old, sampling_required=True)
        hostile = self._valid_world4_load_gate()
        hostile["native_sampling_entered_monotonic_ns_by_rank"] = [
            90,
            402,
            403,
            404,
        ]
        with self.assertRaisesRegex(canary.NativeIdentityCanaryError, "sampling"):
            canary._validate_world4_load_completion_gate(
                hostile, sampling_required=True
            )

    def test_three_arms_use_native_guidance_and_one_shared_gaussian_contract(self) -> None:
        expected_modes = {
            "t2v": "t2v_apg",
            "r2v": "r2v_apg",
            "rv2v": "rv2v",
        }
        for arm, mode in expected_modes.items():
            with self.subTest(arm=arm):
                contract = canary.native_sampling_contract(
                    arm, steps=40, seed=2027
                )
                self.assertEqual(contract["num_frames"], 81)
                self.assertEqual(contract["num_inference_steps"], 40)
                self.assertEqual(contract["seed"], 2027)
                self.assertEqual(contract["guidance_mode"], mode)
                self.assertEqual(contract["flow_shift"], 5.0)
                self.assertEqual(contract["omega_vid"], 1.25)
                self.assertEqual(contract["omega_img"], 4.5)
                self.assertEqual(contract["omega_txt"], 4.0)
        self.assertEqual(
            canary.TARGET_INITIALIZATION,
            "official_gen_wanx22_fresh_gaussian",
        )

    def test_public_rv2v_arm_keeps_training_task_and_guidance_namespaces_separate(self) -> None:
        self.assertEqual(canary.ARM_TRAINING_TASK_NAMES["rv2v"], "vr2v")
        self.assertEqual(canary.ARM_GUIDANCE_MODES["rv2v"], "rv2v")
        self.assertIn("vr2v", canary.TASK_SYSTEM_PROMPTS)
        self.assertNotIn("rv2v", canary.TASK_SYSTEM_PROMPTS)

    def test_conditions_are_exactly_t2v_r2v_and_rv2v(self) -> None:
        full = object()
        refs = {index: object() for index in range(81)}

        t2v = canary.select_native_conditions(
            "t2v", full_source_latent=full, reference_latents=refs
        )
        self.assertIsNone(t2v["image_vae_latents"])
        self.assertIsNone(t2v["multi_video_vae_latents"])
        self.assertIsNone(t2v["multi_image_vae_latents"])

        r2v = canary.select_native_conditions(
            "r2v", full_source_latent=None, reference_latents=refs
        )
        self.assertIsNone(r2v["multi_video_vae_latents"])
        self.assertEqual(
            r2v["multi_image_vae_latents"],
            [refs[index] for index in (0, 20, 40, 60, 80)],
        )

        rv2v = canary.select_native_conditions(
            "rv2v", full_source_latent=full, reference_latents=refs
        )
        self.assertEqual(rv2v["multi_video_vae_latents"], [full])
        self.assertEqual(
            rv2v["multi_image_vae_latents"],
            [refs[index] for index in (0, 27, 53, 80)],
        )
        with self.assertRaisesRegex(
            canary.NativeIdentityCanaryError, "full source video latent"
        ):
            canary.select_native_conditions(
                "rv2v", full_source_latent=None, reference_latents=refs
            )

    def test_source_ids_stay_inside_the_pretrained_one_through_five_range(self) -> None:
        self.assertEqual(
            canary.source_id_contract("t2v")["reference_source_ids"], []
        )
        self.assertEqual(
            canary.source_id_contract("r2v")["reference_source_ids"],
            [1, 2, 3, 4, 5],
        )
        rv2v = canary.source_id_contract("rv2v")
        self.assertEqual(rv2v["video_source_ids"], [1])
        self.assertEqual(rv2v["reference_source_ids"], [2, 3, 4, 5])
        for arm in canary.ARM_ORDER:
            self.assertTrue(
                canary.source_id_contract(arm)[
                    "within_pretrained_source_ids_1_through_5"
                ]
            )
            self.assertFalse(
                canary.source_id_contract(arm)["source_id_interpolation_required"]
            )

    def test_task_specific_native_prefixes_and_reference_binding_are_fixed(self) -> None:
        prompt = "A dog turns its head to the right."
        outputs = {
            arm: canary.build_task_prompt(
                arm, prompt, prompt_cleaner=lambda value: value
            )
            for arm in canary.ARM_ORDER
        }
        self.assertTrue(outputs["t2v"].startswith(canary.TASK_SYSTEM_PROMPTS["t2v"]))
        self.assertTrue(outputs["rv2v"].startswith(canary.TASK_SYSTEM_PROMPTS["vr2v"]))
        self.assertNotIn("image0", outputs["t2v"])
        for name in ("image0", "image1", "image2", "image3", "image4"):
            self.assertIn(name, outputs["r2v"])
        for name in ("image0", "image1", "image2", "image3"):
            self.assertIn(name, outputs["rv2v"])
        self.assertIn("source video", outputs["rv2v"])
        self.assertTrue(all(output.endswith(prompt) for output in outputs.values()))

    def test_reference_frames_are_independently_vae_encoded_from_rgb(self) -> None:
        source = inspect.getsource(canary.main)
        self.assertIn(
            "source_pixels[:, :, index : index + 1, :, :].contiguous()",
            source,
        )
        self.assertRegex(source, r"_vae_encode\(\s+vae,\s+source_pixels")
        self.assertNotIn("full_source_latent[:,", source)
        module_doc = canary.__doc__ or ""
        self.assertIn("independently encoded", module_doc)
        self.assertIn("no temporal-video", module_doc)

    def test_required_model_load_lock_fails_closed_and_rejects_symlink(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"NATIVE_SERIALIZED_HOST_LOAD_REQUIRED": "1"},
            clear=True,
        ):
            with self.assertRaisesRegex(
                canary.NativeIdentityCanaryError, "NATIVE_V_AXIS_LOAD_LOCK"
            ):
                with canary._serialized_host_checkpoint_load():
                    pass
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            target = root / "target.lock"
            target.write_bytes(b"")
            linked = root / "linked.lock"
            linked.symlink_to(target)
            with mock.patch.dict(
                os.environ,
                {
                    "NATIVE_SERIALIZED_HOST_LOAD_REQUIRED": "1",
                    "NATIVE_V_AXIS_LOAD_LOCK": str(linked),
                },
                clear=True,
            ):
                with self.assertRaisesRegex(
                    canary.NativeIdentityCanaryError, "lock differs"
                ):
                    with canary._serialized_host_checkpoint_load():
                        pass

    def test_model_load_lock_serializes_four_real_processes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            lock = root / "renderer-load.lock"
            lock.write_bytes(b"")
            lock.chmod(0o400)
            events = root / "events.txt"
            code = (
                "import os,sys,time\n"
                "sys.path.insert(0, os.environ['METHOD_ROOT'])\n"
                "import infer_native_identity_generation_canary as c\n"
                "with c._serialized_host_checkpoint_load():\n"
                "    with open(os.environ['EVENTS'], 'a', encoding='ascii') as h:\n"
                "        h.write('enter ' + os.environ['RANK'] + '\\n'); h.flush()\n"
                "    time.sleep(0.08)\n"
                "    with open(os.environ['EVENTS'], 'a', encoding='ascii') as h:\n"
                "        h.write('exit ' + os.environ['RANK'] + '\\n'); h.flush()\n"
            )
            processes = []
            for rank in range(4):
                environment = dict(os.environ)
                environment.update(
                    {
                        "METHOD_ROOT": str(METHOD_ROOT),
                        "EVENTS": str(events),
                        "RANK": str(rank),
                        "NATIVE_SERIALIZED_HOST_LOAD_REQUIRED": "1",
                        "NATIVE_V_AXIS_LOAD_LOCK": str(lock),
                    }
                )
                processes.append(
                    subprocess.Popen(
                        [sys.executable, "-B", "-c", code],
                        env=environment,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                )
            outputs = [process.communicate(timeout=30) for process in processes]
            self.assertEqual(
                [process.returncode for process in processes],
                [0, 0, 0, 0],
                outputs,
            )
            rows = events.read_text(encoding="ascii").splitlines()
            self.assertEqual(len(rows), 8)
            for index in range(0, 8, 2):
                entered = rows[index].split()
                exited = rows[index + 1].split()
                self.assertEqual(entered[0], "enter")
                self.assertEqual(exited, ["exit", entered[1]])
            self.assertEqual({row.split()[1] for row in rows[::2]}, set("0123"))

    def test_world4_synthetic_lifecycle_keeps_deserialize_gpu_move_and_trim_atomic(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            lock = root / "renderer-load.lock"
            lock.write_bytes(b"")
            lock.chmod(0o400)
            events = root / "lifecycle.txt"
            code = (
                "import os,sys\n"
                "sys.path.insert(0, os.environ['METHOD_ROOT'])\n"
                "import infer_native_identity_generation_canary as c\n"
                "def record(phase):\n"
                "    with open(os.environ['EVENTS'], 'a', encoding='ascii') as h:\n"
                "        h.write(phase + ' ' + os.environ['RANK'] + '\\n'); h.flush()\n"
                "class Model:\n"
                "    def __init__(self, config): record('deserialize')\n"
                "    def requires_grad_(self, value): return self\n"
                "    def eval(self): return self\n"
                "    def to(self, device): record('gpu'); return self\n"
                "def trim(): record('trim'); return True\n"
                "c._trim_host_allocator = trim\n"
                "c._load_frozen_renderer_gpu_resident_serialized(Model, None, 'cuda')\n"
            )
            processes = []
            for rank in range(4):
                environment = dict(os.environ)
                environment.update(
                    {
                        "METHOD_ROOT": str(METHOD_ROOT),
                        "EVENTS": str(events),
                        "RANK": str(rank),
                        "NATIVE_SERIALIZED_HOST_LOAD_REQUIRED": "1",
                        "NATIVE_V_AXIS_LOAD_LOCK": str(lock),
                    }
                )
                processes.append(
                    subprocess.Popen(
                        [sys.executable, "-B", "-c", code],
                        env=environment,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                )
            outputs = [process.communicate(timeout=30) for process in processes]
            self.assertEqual(
                [process.returncode for process in processes],
                [0, 0, 0, 0],
                outputs,
            )
            rows = [row.split() for row in events.read_text(encoding="ascii").splitlines()]
            self.assertEqual(len(rows), 12)
            for index in range(0, 12, 3):
                rank = rows[index][1]
                self.assertEqual(
                    rows[index : index + 3],
                    [["deserialize", rank], ["gpu", rank], ["trim", rank]],
                )
            self.assertEqual({rows[index][1] for index in range(0, 12, 3)}, set("0123"))

    def test_serialized_lifecycle_rejects_failed_allocator_trim(self) -> None:
        class Model:
            def __init__(self, config: object) -> None:
                pass

            def requires_grad_(self, value: bool) -> "Model":
                return self

            def eval(self) -> "Model":
                return self

            def to(self, device: object) -> "Model":
                return self

        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory).resolve() / "renderer-load.lock"
            lock.write_bytes(b"")
            with mock.patch.dict(
                os.environ,
                {
                    "NATIVE_SERIALIZED_HOST_LOAD_REQUIRED": "1",
                    "NATIVE_V_AXIS_LOAD_LOCK": str(lock),
                },
                clear=True,
            ), mock.patch.object(canary, "_trim_host_allocator", return_value=False):
                with self.assertRaisesRegex(
                    canary.NativeIdentityCanaryError, "allocator trim"
                ):
                    canary._load_frozen_renderer_gpu_resident_serialized(
                        Model, None, "cuda"
                    )

    def test_model_load_lock_covers_gpu_move_and_t2v_defers_vae_weights(self) -> None:
        source = inspect.getsource(canary.main)
        helper = inspect.getsource(
            canary._load_frozen_renderer_gpu_resident_serialized
        )
        lock_at = helper.index("with _serialized_host_checkpoint_load():")
        model_at = helper.index("model = model_factory(config)", lock_at)
        gpu_at = helper.index("model.to(device)", model_at)
        trim_at = helper.index("_trim_host_allocator()", gpu_at)
        self.assertLess(lock_at, model_at)
        self.assertLess(model_at, gpu_at)
        self.assertLess(gpu_at, trim_at)
        load_at = source.index("_load_frozen_renderer_gpu_resident_serialized(")
        load_complete_at = source.index(
            "renderer_gpu_resident_trimmed_ns = time.monotonic_ns()", load_at
        )
        load_gate_at = source.index(
            "_world4_renderer_load_completion_barrier(", load_complete_at
        )
        source_setup_at = source.index(
            "source_audit.prepare_hashed_source_snapshot(source_path)", load_gate_at
        )
        tokenizer_at = source.index("AutoTokenizer.from_pretrained(", load_gate_at)
        sampling_gate_at = source.index(
            "_complete_world4_load_completion_gate_before_sampling(", tokenizer_at
        )
        sample_at = source.index("model.sample(", sampling_gate_at)
        self.assertLess(load_at, load_complete_at)
        self.assertLess(load_complete_at, load_gate_at)
        self.assertLess(load_gate_at, source_setup_at)
        self.assertLess(load_gate_at, tokenizer_at)
        self.assertLess(sampling_gate_at, sample_at)
        self.assertIn("vae = None", source)
        self.assertIn("AutoencoderKLWan.load_config(", source)
        self.assertIn("if distributed.rank == 0:\n        if vae is None:", source)
        self.assertNotIn('model.to("cpu")', source)
        retire_at = source.index("del model")
        trim_at = source.index("_trim_host_allocator()", retire_at)
        empty_at = source.index("torch.cuda.empty_cache()", trim_at)
        barrier_at = source.index("dist.barrier()", empty_at)
        rank_zero_vae_at = source.index(
            "if distributed.rank == 0:\n        if vae is None:", barrier_at
        )
        self.assertLess(retire_at, trim_at)
        self.assertLess(trim_at, empty_at)
        self.assertLess(empty_at, barrier_at)
        self.assertLess(barrier_at, rank_zero_vae_at)

    def test_local_vae_conditions_are_rank_zero_broadcast_before_identity_audit(
        self,
    ) -> None:
        source = inspect.getsource(canary.main)
        encode_at = source.index("reference_latents = {")
        broadcast_at = source.index("condition_broadcasts = {")
        identity_at = source.index("condition_identities: dict[str, Any] = {")
        sample_at = source.index("sample_fn=lambda: model.sample(")
        self.assertLess(encode_at, broadcast_at)
        self.assertLess(broadcast_at, identity_at)
        self.assertLess(identity_at, sample_at)
        helper = inspect.getsource(canary._broadcast_condition_from_rank_zero)
        self.assertIn("dist.broadcast(value, src=0)", helper)
        self.assertIn("all_gather_object", helper)

    @unittest.skipUnless(
        TORCH_AVAILABLE and SAFETENSORS_AVAILABLE,
        "PyTorch and safetensors are required",
    )
    def test_native_clean_latent_is_saved_before_decode_without_mp4_roundtrip(self) -> None:
        import torch
        from safetensors.torch import load_file

        latent = torch.arange(1 * 16 * 21 * 2 * 2, dtype=torch.bfloat16).reshape(
            1, 16, 21, 2, 2
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "t2v.normalized-clean-latent.safetensors"
            artifact = canary._save_normalized_clean_latent_atomically(path, latent)
            restored = load_file(str(path), device="cpu")["normalized_clean_latent"]
            self.assertTrue(torch.equal(restored, latent.float()))
            self.assertEqual(artifact["shape"][:3], [1, 16, 21])
            self.assertEqual(artifact["stored_dtype"], "torch.float32")
            self.assertTrue(artifact["native_sampler_before_vae_decode"])
            self.assertFalse(artifact["mp4_decode_reencode_used"])
            with self.assertRaisesRegex(canary.NativeIdentityCanaryError, "fresh"):
                canary._save_normalized_clean_latent_atomically(path, latent)
        source = inspect.getsource(canary._save_outputs)
        self.assertLess(
            source.index("_save_normalized_clean_latent_atomically"),
            source.index("decoded = _vae_decode"),
        )

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required")
    def test_raw_initial_noise_hash_is_clone_stable_and_bit_sensitive(self) -> None:
        import torch

        value = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)
        clone = value.clone()
        changed = value.clone()
        changed.view(torch.uint8)[0] ^= 1
        digest = canary._tensor_raw_value_sha256(value, label="value")
        self.assertEqual(
            digest,
            canary._tensor_raw_value_sha256(clone, label="clone"),
        )
        self.assertNotEqual(
            digest,
            canary._tensor_raw_value_sha256(changed, label="changed"),
        )

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required")
    def test_native_noise_observer_forwards_original_object_and_restores_symbol(
        self,
    ) -> None:
        import torch

        returned_by_original: list[torch.Tensor] = []
        sampler_saw_original_object: list[bool] = []

        def canonical_randn_tensor(
            shape: object,
            *,
            generator: torch.Generator,
            device: object,
            dtype: torch.dtype,
        ) -> torch.Tensor:
            value = torch.randn(
                shape,
                generator=generator,
                device=device,
                dtype=dtype,
            )
            returned_by_original.append(value)
            return value

        module = types.SimpleNamespace(randn_tensor=canonical_randn_tensor)

        def sample() -> torch.Tensor:
            generator = torch.Generator(device="cpu").manual_seed(2027)
            value = module.randn_tensor(
                (1, 2, 3),
                device=torch.device("cpu"),
                dtype=torch.float32,
                generator=generator,
            )
            sampler_saw_original_object.append(value is returned_by_original[-1])
            return value.square()

        result, capture = canary._sample_with_native_initial_noise_observer(
            sample_fn=sample,
            wan_diffusion_module=module,
            expected_shape=(1, 2, 3),
            expected_device=torch.device("cpu"),
            expected_seed=2027,
            canonical_randn_tensor=canonical_randn_tensor,
        )
        self.assertIs(module.randn_tensor, canonical_randn_tensor)
        self.assertEqual(sampler_saw_original_object, [True])
        self.assertTrue(torch.equal(result, returned_by_original[0].square()))
        self.assertTrue(torch.equal(capture.tensor, returned_by_original[0].cpu()))
        self.assertIsNot(capture.tensor, returned_by_original[0])
        self.assertEqual(capture.call_count, 1)
        self.assertEqual(capture.generator_device, "cpu")
        self.assertEqual(capture.generator_initial_seed, 2027)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required")
    def test_native_noise_observer_fails_closed_on_call_count_and_restores(
        self,
    ) -> None:
        import torch

        def canonical_randn_tensor(
            shape: object,
            *,
            generator: torch.Generator,
            device: object,
            dtype: torch.dtype,
        ) -> torch.Tensor:
            return torch.randn(
                shape,
                generator=generator,
                device=device,
                dtype=dtype,
            )

        module = types.SimpleNamespace(randn_tensor=canonical_randn_tensor)

        def sample_twice() -> torch.Tensor:
            outputs = []
            for _ in range(2):
                outputs.append(
                    module.randn_tensor(
                        (1, 2),
                        generator=torch.Generator(device="cpu").manual_seed(3),
                        device=torch.device("cpu"),
                        dtype=torch.float32,
                    )
                )
            return outputs[0]

        with self.assertRaisesRegex(canary.NativeIdentityCanaryError, "exactly one"):
            canary._sample_with_native_initial_noise_observer(
                sample_fn=sample_twice,
                wan_diffusion_module=module,
                expected_shape=(1, 2),
                expected_device=torch.device("cpu"),
                expected_seed=3,
                canonical_randn_tensor=canonical_randn_tensor,
            )
        self.assertIs(module.randn_tensor, canonical_randn_tensor)

        def sample_failure() -> torch.Tensor:
            module.randn_tensor(
                (1, 2),
                generator=torch.Generator(device="cpu").manual_seed(3),
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            raise RuntimeError("sampler failed")

        with self.assertRaisesRegex(RuntimeError, "sampler failed"):
            canary._sample_with_native_initial_noise_observer(
                sample_fn=sample_failure,
                wan_diffusion_module=module,
                expected_shape=(1, 2),
                expected_device=torch.device("cpu"),
                expected_seed=3,
                canonical_randn_tensor=canonical_randn_tensor,
            )
        self.assertIs(module.randn_tensor, canonical_randn_tensor)

    @unittest.skipUnless(
        TORCH_AVAILABLE and SAFETENSORS_AVAILABLE,
        "PyTorch and safetensors are required",
    )
    def test_observed_initial_noise_artifact_has_stable_factorial_contract(self) -> None:
        import torch
        from safetensors.torch import load_file

        def canonical_randn_tensor(
            shape: object,
            *,
            generator: torch.Generator,
            device: object,
            dtype: torch.dtype,
        ) -> torch.Tensor:
            return torch.randn(
                shape,
                generator=generator,
                device=device,
                dtype=dtype,
            )

        module = types.SimpleNamespace(randn_tensor=canonical_randn_tensor)

        def sample() -> torch.Tensor:
            return module.randn_tensor(
                (1, 16, 21, 2, 2),
                generator=torch.Generator(device="cpu").manual_seed(11),
                device=torch.device("cpu"),
                dtype=torch.float32,
            )

        _, capture = canary._sample_with_native_initial_noise_observer(
            sample_fn=sample,
            wan_diffusion_module=module,
            expected_shape=(1, 16, 21, 2, 2),
            expected_device=torch.device("cpu"),
            expected_seed=11,
            canonical_randn_tensor=canonical_randn_tensor,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "t2v.official-initial-gaussian.safetensors"
            artifact = canary._save_initial_noise_atomically(
                path,
                capture,
                all_rank_identity={
                    "all_rank_exact": True,
                    "identity": {
                        "raw_storage_sha256": capture.raw_value_sha256,
                        "shape": list(capture.requested_shape),
                        "dtype": capture.returned_dtype,
                    },
                },
            )
            restored = load_file(str(path), device="cpu")[
                "official_initial_gaussian"
            ]
            self.assertTrue(torch.equal(restored, capture.tensor))
            self.assertEqual(artifact["path"], str(path))
            self.assertRegex(artifact["sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(
                artifact["tensor_value_sha256"], capture.raw_value_sha256
            )
            self.assertEqual(artifact["shape"], [1, 16, 21, 2, 2])
            self.assertEqual(artifact["stored_dtype"], "torch.float32")
            self.assertEqual(artifact["official_randn_tensor_call_count"], 1)
            self.assertTrue(artifact["captured_from_native_sampler"])
            self.assertFalse(artifact["observer_changed_return_value"])
            self.assertFalse(artifact["source_or_target_derived"])
            self.assertFalse(artifact["external_initial_noise_injection"])
        receipt_source = inspect.getsource(canary._build_receipt)
        self.assertIn('"initial_noise_artifacts"', receipt_source)

    @unittest.skipUnless(
        TORCH_AVAILABLE and SAFETENSORS_AVAILABLE,
        "PyTorch and safetensors are required",
    )
    def test_source_condition_latent_has_a_distinct_predecode_role(self) -> None:
        import torch

        latent = torch.zeros(1, 16, 21, 2, 2, dtype=torch.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.normalized-clean-latent.safetensors"
            artifact = canary._save_normalized_clean_latent_atomically(
                path,
                latent,
                artifact_role="source_video_condition",
            )
            self.assertEqual(artifact["artifact_role"], "source_video_condition")
            self.assertTrue(artifact["source_video_vae_encode_before_any_decode"])
            self.assertFalse(artifact["native_sampler_before_vae_decode"])
            self.assertFalse(artifact["mp4_decode_reencode_used"])

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                canary.NativeIdentityCanaryError, "unsupported"
            ):
                canary._save_normalized_clean_latent_atomically(
                    Path(directory) / "bad.safetensors",
                    latent,
                    artifact_role="paired_target",
                )

    def test_token_budget_matches_real_exact81_patching(self) -> None:
        receipt = canary._latent_geometry_receipt(bucket_hw=(496, 480), z_dim=16)
        self.assertEqual(receipt["video_latent_shape"], [1, 16, 21, 62, 60])
        self.assertEqual(receipt["reference_latent_shape"], [1, 16, 1, 62, 60])
        self.assertEqual(receipt["target_patch_tokens"], 19_530)
        self.assertEqual(receipt["one_reference_patch_tokens"], 930)
        self.assertEqual(
            receipt["per_arm_total_visual_tokens"],
            {"t2v": 19_530, "r2v": 24_180, "rv2v": 42_780},
        )

    def test_cli_accepts_arm_subsets_but_rejects_ambiguous_contracts(self) -> None:
        args = self._valid_args(arms=["rv2v", "t2v"])
        self.assertEqual(canary.validate_cli(args), ("t2v", "rv2v"))
        with self.assertRaisesRegex(canary.NativeIdentityCanaryError, "unique"):
            canary.validate_cli(self._valid_args(arms=["t2v", "t2v"]))
        with self.assertRaisesRegex(canary.NativeIdentityCanaryError, "40 UniPC"):
            canary.validate_cli(self._valid_args(num_inference_steps=41))
        with self.assertRaisesRegex(canary.NativeIdentityCanaryError, "prompt SHA"):
            canary.validate_cli(
                self._valid_args(expected_action_prompt_sha256="0" * 64)
            )

    def test_cli_exposes_no_privileged_or_external_reference_condition(self) -> None:
        destinations = {action.dest for action in canary.build_parser()._actions}
        for forbidden in (
            "target_video",
            "mask",
            "flow",
            "pose",
            "track",
            "trajectory",
            "first_frame",
            "reference_image",
            "reference_video",
            "initial_latent",
            "initial_noise",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, destinations)
        self.assertIn("source_video", destinations)
        self.assertIn("action_prompt", destinations)


if __name__ == "__main__":
    unittest.main()
