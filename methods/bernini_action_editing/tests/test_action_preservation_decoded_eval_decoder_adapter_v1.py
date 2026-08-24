from __future__ import annotations

import copy
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import action_preservation_decoded_eval_decoder_adapter_v1 as adapter
import action_preservation_decoded_eval_executor_v2 as executor
import action_preservation_decoded_eval_model_authority_v2 as authority
import action_preservation_decoded_eval_plan_v1 as plan


PINNED_MANIFEST = ROOT / "audits/bernini_r13_ff4c5d4_checkpoint.sha256"


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def canonical_write(path: pathlib.Path, value: dict) -> None:
    path.write_bytes(plan.canonical_json_bytes(value) + b"\n")


def action_contract(iid: str) -> dict:
    description = f"Complete the fitted action for source {iid}, then hold the terminal pose."
    row = {
        "schema_version": plan.ACTION_REVIEW_CONTRACT_SCHEMA,
        "action_order_description": description,
        "action_order_description_sha256": plan.text_sha256(description),
        "expected_onset_frame_min": 4,
        "expected_onset_frame_max": 20,
        "terminal_hold_start_frame_min": 65,
        "terminal_hold_end_frame": 80,
        "full_video_frame_count": 81,
        "fps_num": 25,
        "fps_den": 1,
    }
    row["contract_digest"] = plan.object_sha256(row)
    return row


class RealDecoderAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = pathlib.Path(temporary.name).resolve()
        self.evaluation_root = self.root / "evaluation"
        self.private_parent = self.root / "authority-private"
        self.private_parent.mkdir()
        self.private_parent_fd = os.open(
            self.private_parent,
            os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0),
        )
        os.set_inheritable(self.private_parent_fd, False)
        self.task_root = self.root / "task-publication"
        self.task_root.mkdir()
        self.task_root_fd = os.open(
            self.task_root,
            os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0),
        )
        os.set_inheritable(self.task_root_fd, False)
        sources = []
        for index, iid in enumerate(plan.FITTED_IIDS):
            instruction = f"Perform the fitted action for source {iid}."
            sources.append(
                {
                    "iid": iid,
                    "source_video_sha256": digest(f"source:{iid}"),
                    "source_receipt_sha256": digest(f"receipt:{iid}"),
                    "instruction": instruction,
                    "instruction_sha256": plan.text_sha256(instruction),
                    "action_review_contract": action_contract(iid),
                    "seed": 2026081801 + index,
                }
            )
        self.checkpoint_payloads = {}
        checkpoints = []
        for arm in plan.ARMS:
            for step in plan.CHECKPOINT_STEPS:
                receipt_raw = (
                    json.dumps(
                        {"arm": arm, "step": step},
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    + b"\n"
                )
                config_raw = b'{"peft_type":"LORA"}\n'
                model_raw = f"adapter:{arm}:{step}".encode("utf-8")
                self.checkpoint_payloads[(arm, step)] = (
                    receipt_raw,
                    config_raw,
                    model_raw,
                )
                checkpoints.append(
                    {
                        "arm": arm,
                        "checkpoint_step": step,
                        "checkpoint_receipt_sha256": hashlib.sha256(
                            receipt_raw
                        ).hexdigest(),
                        "adapter_sha256": hashlib.sha256(
                            model_raw
                        ).hexdigest(),
                    }
                )
        input_spec = plan.build_input_spec(
            evaluation_id="decoder-adapter-test",
            evaluation_root=self.evaluation_root,
            pins={key: digest(key) for key in plan.PIN_FIELDS},
            sources=sources,
            checkpoints=checkpoints,
        )
        self.bundle = plan.build_bundle(input_spec)
        self.bundle["publication_receipt"] = plan.build_publication_receipt(
            self.bundle
        )
        self.shard = self.bundle["shards"][plan.HOLDER_ROWS[0]["job_id"]]
        self.task = self.shard["tasks"][0]
        record = self.task["record"]
        task_id = executor._task_id(self.task)
        source = next(
            item for item in sources if item["iid"] == record["iid"]
        )
        self.physical_identity = {
            "path": str(self.root / "physical_bindings.json"),
            "sha256": digest("physical bindings"),
        }
        self.source_binding = {
            "iid": source["iid"],
            "source_video": {
                "path": str(self.root / "source.mp4"),
                "sha256": source["source_video_sha256"],
            },
            "source_receipt": {
                "path": str(self.root / "source-receipt.json"),
                "sha256": source["source_receipt_sha256"],
            },
            "instruction_sha256": source["instruction_sha256"],
            "action_review_contract_digest": source[
                "action_review_contract"
            ]["contract_digest"],
            "seed": source["seed"],
        }
        checkpoint_root = self.root / "checkpoint"
        receipt_raw, config_raw, model_raw = self.checkpoint_payloads[
            (record["arm"], record["checkpoint_step"])
        ]
        for relative, raw in (
            ("receipt.json", receipt_raw),
            ("adapter/adapter_config.json", config_raw),
            ("adapter/adapter_model.safetensors", model_raw),
            ("optimizer.pt", b"optimizer-not-consumed"),
        ):
            path = checkpoint_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            path.chmod(0o444)
        (checkpoint_root / "adapter").chmod(0o555)
        checkpoint_root.chmod(0o555)

        def captured(path: pathlib.Path) -> dict:
            return {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }

        self.checkpoint_binding = {
            "arm": record["arm"],
            "checkpoint_step": record["checkpoint_step"],
            "checkpoint_root": str(checkpoint_root),
            "checkpoint_receipt": captured(checkpoint_root / "receipt.json"),
            "checkpoint_receipt_digest": digest("checkpoint receipt digest"),
            "adapter_config": captured(
                checkpoint_root / "adapter" / "adapter_config.json"
            ),
            "adapter_model": captured(
                checkpoint_root / "adapter" / "adapter_model.safetensors"
            ),
        }

        model_root = self.root / "model"
        model_root.mkdir()
        manifest_rows = []
        for index, line in enumerate(
            PINNED_MANIFEST.read_text(encoding="utf-8").splitlines()
        ):
            relative = line.split("  ./", 1)[1]
            raw = f"model:{index}:{relative}\n".encode("utf-8")
            path = model_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            path.chmod(0o644)
            manifest_rows.append(
                f"{hashlib.sha256(raw).hexdigest()}  ./{relative}"
            )
        model_manifest = self.root / "model.sha256"
        model_manifest.write_text(
            "\n".join(manifest_rows) + "\n", encoding="utf-8"
        )
        model_manifest.chmod(0o644)
        self.model_authority = authority.ModelAuthority.capture(
            model_root=model_root,
            manifest_path=model_manifest,
            private_parent=self.private_parent,
            private_parent_fd=self.private_parent_fd,
            view_name="decoder-model-fd-view",
            expected_uid=os.getuid(),
            expected_gid=os.getgid(),
            expected_device=None,
            expected_manifest_sha256=hashlib.sha256(
                model_manifest.read_bytes()
            ).hexdigest(),
            proc_fd_prefix="/dev/fd",
        )
        model_capture_path = self.task_root / "model-capture.json"
        canonical_write(
            model_capture_path, self.model_authority.capture_receipt
        )
        model_pre = self.model_authority.begin_task(task_id)
        adapter_info = (checkpoint_root / "receipt.json").stat()
        self.adapter_authority = authority.AdapterAuthority.capture(
            task_id=task_id,
            checkpoint_root=checkpoint_root,
            expected_sha256={
                relative: self.checkpoint_binding[key]["sha256"]
                for relative, key in (
                    ("receipt.json", "checkpoint_receipt"),
                    ("adapter/adapter_config.json", "adapter_config"),
                    ("adapter/adapter_model.safetensors", "adapter_model"),
                )
            },
            private_parent=self.private_parent,
            private_parent_fd=self.private_parent_fd,
            view_name="decoder-adapter-fd-view",
            expected_uid=adapter_info.st_uid,
            expected_gid=adapter_info.st_gid,
            proc_fd_prefix="/dev/fd",
        )
        adapter_capture_path = self.task_root / "adapter-capture.json"
        canonical_write(
            adapter_capture_path, self.adapter_authority.capture_receipt
        )
        adapter_pre = self.adapter_authority.begin_use()
        task_publication_root = authority.task_publication_root_binding(
            descriptor=self.task_root_fd,
            path=self.task_root,
        )
        inherited_fds = authority.build_inherited_fd_binding(
            task_id=task_id,
            model_capture=self.model_authority.capture_receipt,
            adapter_capture=self.adapter_authority.capture_receipt,
            task_publication_root=task_publication_root,
        )
        self.bindings = {
            "evaluation_id": self.bundle["manifest"]["evaluation_id"],
            "evaluation_root": str(self.evaluation_root),
            "input_digest": self.bundle["input_spec"]["input_digest"],
            "manifest_digest": self.bundle["manifest"]["manifest_digest"],
            "physical_bindings_digest": digest("physical-bindings-object"),
            "sources": [self.source_binding],
            "checkpoints": [self.checkpoint_binding],
            "runtime": {
                "python": {"path": "/pinned/python"},
                "infer_lora": {"path": "/pinned/infer_lora.py"},
                "decoder_adapter": {
                    "path": "/adapter",
                    "sha256": digest("adapter tool"),
                },
                "ffprobe": {
                    "path": "/ffprobe",
                    "sha256": digest("ffprobe"),
                },
                "bernini_root": "/pinned/bernini",
                "veomni_root": "/pinned/veomni",
                "model_checkpoint_root": str(model_root),
                "expected_bernini_commit": hashlib.sha1(b"bernini").hexdigest(),
                "expected_veomni_commit": hashlib.sha1(b"veomni").hexdigest(),
                "expected_checkpoint_tree_sha256": digest("model tree"),
                "method_source_revision": hashlib.sha1(b"method").hexdigest(),
                "method_source_archive_sha256": digest("method archive"),
                "num_inference_steps": 40,
            },
        }
        consumption_input = authority.build_consumption_input(
            task_id=task_id,
            physical_bindings_digest=self.bindings[
                "physical_bindings_digest"
            ],
            model_capture=self.model_authority.capture_receipt,
            model_pre_use=model_pre,
            model_capture_receipt_path=model_capture_path,
            model_capture_receipt_sha256=hashlib.sha256(
                model_capture_path.read_bytes()
            ).hexdigest(),
            adapter_capture=self.adapter_authority.capture_receipt,
            adapter_pre_use=adapter_pre,
            adapter_capture_receipt_path=adapter_capture_path,
            adapter_capture_receipt_sha256=hashlib.sha256(
                adapter_capture_path.read_bytes()
            ).hexdigest(),
            inherited_fd_binding=inherited_fds,
            task_publication_root=task_publication_root,
            production_mode=False,
            task_member_path_prefix=self.task_root,
        )
        self.inherited_fds = inherited_fds
        self.consumption_input = consumption_input
        consumption_path = self.task_root / "consumption-input.json"
        canonical_write(consumption_path, consumption_input)
        self.request = executor.build_task_input_receipt(
            bundle=self.bundle,
            shard=self.shard,
            task=self.task,
            decoder_identity={
                "path": "/adapter",
                "sha256": digest("adapter tool"),
            },
            ffprobe_identity={
                "path": "/ffprobe",
                "sha256": digest("ffprobe"),
            },
            physical_bindings_identity=self.physical_identity,
            consumption_input_identity={
                "path": str(consumption_path),
                "sha256": hashlib.sha256(
                    consumption_path.read_bytes()
                ).hexdigest(),
                "consumption_input_digest": consumption_input[
                    "consumption_input_digest"
                ],
            },
            verify_tools=False,
        )
        old_fd_environment = os.environ.get(
            authority.INHERITED_FD_BINDING_ENV
        )
        os.environ[authority.INHERITED_FD_BINDING_ENV] = (
            authority.inherited_fd_environment_value(inherited_fds)
        )

        def cleanup_authorities() -> None:
            if old_fd_environment is None:
                os.environ.pop(authority.INHERITED_FD_BINDING_ENV, None)
            else:
                os.environ[
                    authority.INHERITED_FD_BINDING_ENV
                ] = old_fd_environment
            self.adapter_authority.abort(reason="decoder unit test cleanup")
            self.model_authority.abort(reason="decoder unit test cleanup")
            os.close(self.task_root_fd)
            os.close(self.private_parent_fd)

        self.addCleanup(cleanup_authorities)

    def native_receipt(self, output_path: pathlib.Path) -> dict:
        record = self.request["task_record"]
        runtime = self.bindings["runtime"]
        checkpoint = self.checkpoint_binding
        model_capture = self.model_authority.capture_receipt
        adapter_capture = self.adapter_authority.capture_receipt
        rank_evidence = {
            "consumption_input_digest": self.consumption_input[
                "consumption_input_digest"
            ],
            "task_input_digest": self.request["input_digest"],
            "model_capture_digest": model_capture["capture_digest"],
            "model_view_root": self.consumption_input["model"]["view_root"],
            "adapter_capture_digest": adapter_capture["capture_digest"],
            "adapter_view_root": self.consumption_input["adapter"]["view_root"],
            "fd_view_files_authorized": 26,
            "inherited_fd_binding_digest": self.inherited_fds[
                "fd_binding_digest"
            ],
            "inherited_fd_count": 29,
            "ptrace_authorization_used": False,
            "source_video_sha256": self.source_binding[
                "source_video"
            ]["sha256"],
            "source_video_physical_authority_digest": adapter.object_sha256(
                self.source_binding["source_video"]
            ),
            "all_ranks_use_retained_source_fd": True,
        }
        rank_evidence_digest = adapter.object_sha256(rank_evidence)
        value = {
            "schema_version": adapter.INFERENCE_RECEIPT_SCHEMA,
            "method_source_revision": runtime["method_source_revision"],
            "method_source_archive_sha256": runtime[
                "method_source_archive_sha256"
            ],
            "bernini_commit": runtime["expected_bernini_commit"],
            "veomni_commit": runtime["expected_veomni_commit"],
            "checkpoint_tree_sha256": runtime["expected_checkpoint_tree_sha256"],
            "consumption_input_digest": self.consumption_input[
                "consumption_input_digest"
            ],
            "task_input_digest": self.request["input_digest"],
            "model_consumption": {
                **rank_evidence,
                "four_rank_attestation": {
                    "world_size": 4,
                    "all_ranks_replayed_exact_fd_views": True,
                    "rank_evidence_digest": rank_evidence_digest,
                    "ordered_rank_evidence_digests": [
                        rank_evidence_digest
                    ]
                    * 4,
                },
            },
            "adapter": {
                "enabled": True,
                "mode": "lora_safe_merge",
                "checkpoint_root": self.consumption_input["adapter"]["view_root"],
                "adapter_model_path": str(
                    pathlib.Path(self.consumption_input["adapter"]["view_root"])
                    / "adapter/adapter_model.safetensors"
                ),
                "adapter_model_sha256": checkpoint["adapter_model"]["sha256"],
                "training_receipt_path": str(
                    pathlib.Path(self.consumption_input["adapter"]["view_root"])
                    / "receipt.json"
                ),
                "training_receipt_digest": checkpoint[
                    "checkpoint_receipt_digest"
                ],
                "training_global_step": checkpoint["checkpoint_step"],
                "strictly_reloaded": True,
                "safe_merged_for_inference": True,
                "tensor_count": 1,
            },
            "input": {
                "source_video_path": self.source_binding["source_video"]["path"],
                "source_video_sha256": record["source_video_sha256"],
                "instruction_utf8_sha256": record["instruction_sha256"],
                "accepted_model_conditions": ["source_video", "edit_instruction"],
                "target_video_argument": False,
                "target_accessed_by_inference": False,
                "external_mask_or_swept_tube": False,
                "external_tracking_pose_or_trajectory": False,
                "reference_image_or_video": False,
                "external_shared_i0": False,
            },
            "sampling": {
                "seed": record["seed"],
                "num_inference_steps": 40,
                "num_frames": 81,
                "source_onset_policy": record["onset_policy"]["name"],
                "ulysses_size": 4,
                "rank0_decode_and_save_only": True,
            },
            "output": {
                "path": str(output_path),
                "sha256": adapter.file_sha256(output_path),
                "frame_count": 81,
                "fps": 25.0,
            },
            "experimental_inference": True,
            "production_claim_forbidden": True,
            "scientific_claim_authorized": False,
        }
        value["receipt_digest"] = adapter.object_sha256(value)
        return value

    def test_exact_request_maps_to_four_rank_infer_lora_and_native_receipt(self) -> None:
        request_path = self.task_root / "request.json"
        output_path = self.task_root / "decoded.mp4"
        canonical_write(request_path, self.request)
        calls = []
        forwarded = []

        def runner(argv, environment):
            calls.append((list(argv), dict(environment)))
            output_path.write_bytes(b"real-decoded-output")
            canonical_write(
                output_path.with_name(output_path.name + ".receipt.json"),
                self.native_receipt(output_path),
            )
            return subprocess.CompletedProcess(
                argv, 0, b"native-inference-line\n", b"infer-diagnostic\n"
            )

        with mock.patch.object(
            adapter.bridge, "load_physical_bindings", return_value=self.bindings
        ), mock.patch.object(
            adapter,
            "_write_all",
            side_effect=lambda descriptor, payload: forwarded.append(
                (descriptor, bytes(payload))
            ),
        ):
            result = adapter.execute(
                request_path=request_path,
                output_path=output_path,
                runner=runner,
                verify_files=False,
            )
        self.assertEqual(len(calls), 1)
        argv, environment = calls[0]
        self.assertEqual(argv[:7], [
            "/pinned/python", "-B", "-m", "torch.distributed.run",
            "--standalone", "--nproc_per_node=4", "/pinned/infer_lora.py",
        ])
        self.assertIn("--adapter-checkpoint", argv)
        self.assertEqual(argv[argv.index("--num-inference-steps") + 1], "40")
        self.assertNotIn("PYTHONPATH", environment)
        self.assertEqual(environment["HF_HUB_OFFLINE"], "1")
        self.assertEqual(
            forwarded,
            [(1, b"native-inference-line\n"), (2, b"infer-diagnostic\n")],
        )
        self.assertEqual(
            result["receipt_digest"],
            self.native_receipt(output_path)["receipt_digest"],
        )
        native = self.native_receipt(output_path)
        native_raw = plan.canonical_json_bytes(native) + b"\n"
        decoder_raw = plan.canonical_json_bytes(result) + b"\n"
        self.assertEqual(
            executor._validate_decoder_stdout_authority(
                stdout=native_raw + decoder_raw,
                native_receipt_raw=native_raw,
                native_receipt=native,
                decoder_result=result,
            ),
            native,
        )

    def test_main_retries_short_writes_until_second_line_is_complete(self) -> None:
        result = {
            "receipt_path": "/task/decoded.mp4.receipt.json",
            "receipt_sha256": digest("native receipt bytes"),
            "receipt_digest": digest("native receipt object"),
            "decoder_verified_release_capture": None,
            "inference_verified_release_capture": None,
        }
        written = bytearray()
        calls = []

        def short_write(descriptor, payload):
            self.assertEqual(descriptor, 1)
            chunk = bytes(payload[:1])
            calls.append(bytes(payload))
            written.extend(chunk)
            return len(chunk)

        with mock.patch.object(adapter, "execute", return_value=result), mock.patch.object(
            adapter.os, "write", side_effect=short_write
        ):
            self.assertEqual(
                adapter.main(["--request", "/task/request.json", "--output", "/task/decoded.mp4"]),
                0,
            )
        self.assertGreater(len(calls), 1)
        self.assertEqual(bytes(written), plan.canonical_json_bytes(result) + b"\n")

    def test_protocol_write_rejects_zero_progress(self) -> None:
        with mock.patch.object(adapter.os, "write", return_value=0), self.assertRaisesRegex(
            adapter.DecodedEvaluationDecoderError, "made no progress"
        ):
            adapter._write_all(1, b"protocol-line\n")

    def test_legacy_decoder_stdout_field_names_do_not_match_executor_wire(self) -> None:
        output_path = self.task_root / "decoded.mp4"
        output_path.write_bytes(b"real-decoded-output")
        native = self.native_receipt(output_path)
        native_raw = plan.canonical_json_bytes(native) + b"\n"
        legacy = {
            "native_inference_receipt_path": str(
                output_path.with_name(output_path.name + ".receipt.json")
            ),
            "native_inference_receipt_sha256": digest("native receipt"),
            "native_inference_receipt_digest": native["receipt_digest"],
            "decoder_verified_release_capture": None,
            "inference_verified_release_capture": None,
        }
        expected = {
            "receipt_path": legacy["native_inference_receipt_path"],
            "receipt_sha256": legacy["native_inference_receipt_sha256"],
            "receipt_digest": legacy["native_inference_receipt_digest"],
            "decoder_verified_release_capture": None,
            "inference_verified_release_capture": None,
        }
        with self.assertRaisesRegex(
            executor.DecodedEvaluationExecutorError,
            "stdout/file authority differs",
        ):
            executor._validate_decoder_stdout_authority(
                stdout=(native_raw + plan.canonical_json_bytes(legacy) + b"\n"),
                native_receipt_raw=native_raw,
                native_receipt=native,
                decoder_result=expected,
            )

    def test_self_resigned_instruction_swap_is_rejected(self) -> None:
        hostile = copy.deepcopy(self.request)
        hostile["task_record"]["instruction"] = "Swap the source identity."
        hostile["task_record"]["record_digest"] = plan.object_sha256(
            {
                key: value
                for key, value in hostile["task_record"].items()
                if key != "record_digest"
            }
        )
        hostile["task_record_digest"] = hostile["task_record"]["record_digest"]
        hostile["input_digest"] = plan.object_sha256(
            {key: value for key, value in hostile.items() if key != "input_digest"}
        )
        with mock.patch.object(
            adapter.bridge, "load_physical_bindings", return_value=self.bindings
        ), self.assertRaisesRegex(
            adapter.DecodedEvaluationDecoderError, "semantic binding"
        ):
            adapter.resolve_request(hostile, verify_files=False)

    def test_nonzero_infer_is_terminal_and_never_retried(self) -> None:
        request_path = self.task_root / "request.json"
        canonical_write(request_path, self.request)
        calls = []

        def runner(argv, _environment):
            calls.append(list(argv))
            return subprocess.CompletedProcess(argv, 9, b"", b"")

        with mock.patch.object(
            adapter.bridge, "load_physical_bindings", return_value=self.bindings
        ), self.assertRaisesRegex(
            adapter.DecodedEvaluationDecoderError, "returned 9"
        ):
            adapter.execute(
                request_path=request_path,
                output_path=self.task_root / "failed.mp4",
                runner=runner,
                verify_files=False,
            )
        self.assertEqual(len(calls), 1)

    def test_resigned_environment_fd_binding_cannot_replace_d0(self) -> None:
        hostile = copy.deepcopy(self.inherited_fds)
        hostile["task_id"] = "hostile-resigned-task"
        hostile["fd_binding_digest"] = authority.object_sha256(
            {
                key: value
                for key, value in hostile.items()
                if key != "fd_binding_digest"
            }
        )
        original = os.environ[authority.INHERITED_FD_BINDING_ENV]
        os.environ[authority.INHERITED_FD_BINDING_ENV] = (
            authority.inherited_fd_environment_value(hostile)
        )
        try:
            with mock.patch.object(
                adapter.bridge,
                "load_physical_bindings",
                return_value=self.bindings,
            ), self.assertRaisesRegex(
                adapter.DecodedEvaluationDecoderError,
                "inherited-FD task|environment/consumption FD binding|"
                "consumption input inherited FD binding",
            ):
                adapter.resolve_request(self.request, verify_files=False)
        finally:
            os.environ[authority.INHERITED_FD_BINDING_ENV] = original

    def test_hard1_every_step_requires_exact_native_solver_trace(self) -> None:
        output_path = self.task_root / "hard1.mp4"
        output_path.write_bytes(b"hard1-decoded-output")
        with mock.patch.object(
            adapter.bridge,
            "load_physical_bindings",
            return_value=self.bindings,
        ):
            _, resolved_bindings, resolved_source, resolved_checkpoint = (
                adapter.resolve_request(self.request, verify_files=False)
            )
        request = copy.deepcopy(self.request)
        request["task_record"]["onset_policy"] = plan._policy_contract(
            "hard1_every_step"
        )
        receipt = self.native_receipt(output_path)
        receipt["sampling"]["source_onset_policy"] = "hard1_every_step"
        receipt["sampling"]["source_onset_solver_trace"] = {
            "schema_version": adapter.SOURCE_TRAJECTORY_CLAMP_SCHEMA,
            "policy": "hard1_every_step",
            "integrator": "original_unipc_scheduler_step",
            "prediction_type": "flow_prediction",
            "phase": 0,
            "latent_phases": 21,
            "initial_packed_noise_captured": True,
            "step_count": 40,
            "expected_steps": 40,
            "steps": [
                {
                    "step_index": index,
                    "timestep": float(40 - index),
                    "sigma": float(40 - index) / 40.0,
                    "next_sigma": float(39 - index) / 40.0,
                    "phase0_velocity": "captured_epsilon_minus_clean_source",
                    "phase0_post_step": "source_noise_flow_trajectory_projection",
                    "other_phases_projected": False,
                    "original_scheduler_step_calls": 1,
                }
                for index in range(40)
            ],
            "target_video_accessed": False,
            "identity_or_background_claim": False,
        }
        receipt["receipt_digest"] = adapter.object_sha256(
            {key: value for key, value in receipt.items() if key != "receipt_digest"}
        )
        adapter.validate_inference_receipt(
            receipt,
            request=request,
            bindings=resolved_bindings,
            source=resolved_source,
            checkpoint=resolved_checkpoint,
            output_path=output_path,
        )

        missing = copy.deepcopy(receipt)
        missing["sampling"].pop("source_onset_solver_trace")
        missing["receipt_digest"] = adapter.object_sha256(
            {key: value for key, value in missing.items() if key != "receipt_digest"}
        )
        with self.assertRaisesRegex(
            adapter.DecodedEvaluationDecoderError, "solver trace"
        ):
            adapter.validate_inference_receipt(
                missing,
                request=request,
                bindings=resolved_bindings,
                source=resolved_source,
                checkpoint=resolved_checkpoint,
                output_path=output_path,
            )

        hostile = copy.deepcopy(receipt)
        hostile["sampling"]["source_onset_solver_trace"]["steps"][17][
            "other_phases_projected"
        ] = True
        hostile["receipt_digest"] = adapter.object_sha256(
            {key: value for key, value in hostile.items() if key != "receipt_digest"}
        )
        with self.assertRaisesRegex(
            adapter.DecodedEvaluationDecoderError, "solver step"
        ):
            adapter.validate_inference_receipt(
                hostile,
                request=request,
                bindings=resolved_bindings,
                source=resolved_source,
                checkpoint=resolved_checkpoint,
                output_path=output_path,
            )


if __name__ == "__main__":
    unittest.main()
