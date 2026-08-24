from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import stat
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import action_preservation_decoded_eval_executor_v2 as executor
import action_preservation_decoded_eval_model_authority_v2 as authority
import action_preservation_decoded_eval_plan_v1 as plan


PINNED_MANIFEST = ROOT / "audits/bernini_r13_ff4c5d4_checkpoint.sha256"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def fd_view_sha(path: Path) -> str:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        digest = hashlib.sha256()
        offset = 0
        while True:
            chunk = os.pread(descriptor, 1024 * 1024, offset)
            if not chunk:
                break
            digest.update(chunk)
            offset += len(chunk)
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def text_sha(label: str) -> str:
    return sha(label.encode("utf-8"))


def write(path: Path, raw: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.chmod(mode)


def action_contract(iid: str) -> dict:
    description = f"Complete the fitted action for {iid}, then hold."
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


def probe() -> dict:
    return {
        "video_stream_count": 1,
        "frame_count": 81,
        "fps_num": 25,
        "fps_den": 1,
        "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
        "frame_timestamp_times": [f"{index / 25:.6f}" for index in range(81)],
    }


class ExecutorV2Fixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.parent = Path(self.temporary.name).resolve()
        self.evaluation_root = self.parent / "evaluation"
        self.model = self.parent / "model"
        self.model.mkdir()
        manifest_rows = []
        for index, line in enumerate(
            PINNED_MANIFEST.read_text(encoding="utf-8").splitlines()
        ):
            relative = line.split("  ./", 1)[1]
            raw = f"model:{index}:{relative}\n".encode()
            write(self.model / relative, raw, 0o644)
            manifest_rows.append(f"{sha(raw)}  ./{relative}")
        for relative in authority.MODEL_RELATIVE_DIRECTORIES:
            (self.model if relative == "." else self.model / relative).chmod(0o755)
        self.model_manifest = self.parent / "model.sha256"
        self.model_manifest.write_text(
            "\n".join(manifest_rows) + "\n", encoding="utf-8"
        )
        self.model_manifest.chmod(0o644)
        self.model_manifest_sha = sha(self.model_manifest.read_bytes())

        checkpoints = []
        physical_checkpoints = []
        for arm in plan.ARMS:
            for step in plan.CHECKPOINT_STEPS:
                root = self.parent / "training" / "runs" / arm / f"checkpoint-{step:08d}"
                receipt_raw = json.dumps(
                    {"arm": arm, "step": step}, sort_keys=True, separators=(",", ":")
                ).encode() + b"\n"
                config_raw = b'{"peft_type":"LORA"}\n'
                model_raw = f"adapter:{arm}:{step}".encode()
                write(root / "receipt.json", receipt_raw, 0o444)
                write(root / "adapter/adapter_config.json", config_raw, 0o444)
                write(root / "adapter/adapter_model.safetensors", model_raw, 0o444)
                write(root / "optimizer.pt", b"optimizer-not-consumed", 0o444)
                (root / "adapter").chmod(0o555)
                root.chmod(0o555)
                checkpoints.append(
                    {
                        "arm": arm,
                        "checkpoint_step": step,
                        "checkpoint_receipt_sha256": sha(receipt_raw),
                        "adapter_sha256": sha(model_raw),
                    }
                )
                def binding(path: Path) -> dict:
                    info = path.stat()
                    return {
                        "path": str(path),
                        "sha256": sha(path.read_bytes()),
                        "uid": info.st_uid,
                        "gid": info.st_gid,
                    }
                physical_checkpoints.append(
                    {
                        "arm": arm,
                        "checkpoint_step": step,
                        "checkpoint_root": str(root),
                        "checkpoint_receipt": binding(root / "receipt.json"),
                        "adapter_config": binding(root / "adapter/adapter_config.json"),
                        "adapter_model": binding(root / "adapter/adapter_model.safetensors"),
                    }
                )
        sources = []
        for index, iid in enumerate(plan.FITTED_IIDS):
            instruction = f"Perform the fitted action for source {iid}."
            sources.append(
                {
                    "iid": iid,
                    "source_video_sha256": text_sha(f"source:{iid}"),
                    "source_receipt_sha256": text_sha(f"receipt:{iid}"),
                    "instruction": instruction,
                    "instruction_sha256": plan.text_sha256(instruction),
                    "action_review_contract": action_contract(iid),
                    "seed": 2026081801 + index,
                }
            )
        pins = {key: text_sha(key) for key in plan.PIN_FIELDS}
        input_spec = plan.build_input_spec(
            evaluation_id="preservation-v2-executor-v2-fixture",
            evaluation_root=self.evaluation_root,
            pins=pins,
            sources=sources,
            checkpoints=checkpoints,
        )
        bundle = plan.build_bundle(input_spec)
        plan.publish_bundle(bundle)
        self.bundle = executor.load_published_bundle(self.evaluation_root)
        self.physical_bindings = {
            "physical_bindings_digest": text_sha("physical-bindings"),
            "checkpoints": physical_checkpoints,
        }
        private_parent_fd = os.open(
            self.parent, os.O_RDONLY | os.O_DIRECTORY
        )
        try:
            self.model_authority = authority.ModelAuthority.capture(
                model_root=self.model,
                manifest_path=self.model_manifest,
                private_parent=self.parent,
                private_parent_fd=private_parent_fd,
                view_name="injected-model-fd-view",
                expected_uid=os.getuid(),
                expected_gid=os.getgid(),
                expected_device=None,
                expected_manifest_sha256=self.model_manifest_sha,
                proc_fd_prefix="/dev/fd",
            )
        finally:
            os.close(private_parent_fd)
        self.decoder_identity = {
            "path": "/stub/decoder",
            "sha256": text_sha("decoder"),
        }
        self.ffprobe_identity = {
            "path": "/stub/ffprobe",
            "sha256": text_sha("ffprobe"),
        }
        self.physical_identity = {
            "path": "/stub/physical-bindings.json",
            "sha256": text_sha("physical-bindings-file"),
        }
        self.rank_consumption_replays: list[dict[str, object]] = []

    def replay_consumption_as_four_ranks(
        self, request: dict, inherited_fd_binding: dict
    ) -> None:
        identity = request["model_consumption_input"]
        validated_fd_binding = authority.validate_inherited_fd_binding(
            inherited_fd_binding,
            verify_open_fds=True,
            expected_inheritable=False,
        )
        for rank in range(4):
            with mock.patch.dict(
                os.environ,
                {
                    authority.INHERITED_FD_BINDING_ENV:
                    authority.inherited_fd_environment_value(
                        validated_fd_binding
                    )
                },
                clear=False,
            ):
                consumption, model_capture, adapter_capture = (
                    authority.load_consumption_input(
                        identity["path"],
                        expected_sha256=identity["sha256"],
                        expected_digest=identity["consumption_input_digest"],
                    )
                )
            model_view = Path(model_capture["model_view_root"])
            self.assertNotEqual(model_view, self.model)
            model_rows = {
                row["relative_path"]: row for row in model_capture["files"]
            }
            for relative in (
                "transformer/config.json",
                "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
            ):
                self.assertEqual(
                    fd_view_sha(model_view / relative),
                    model_rows[relative]["sha256"],
                )
            adapter_hashes: dict[str, str] = {}
            if adapter_capture is not None:
                adapter_view = Path(adapter_capture["adapter_view_root"])
                self.assertNotEqual(
                    adapter_view, Path(adapter_capture["checkpoint_root"])
                )
                adapter_rows = {
                    row["relative_path"]: row
                    for row in adapter_capture["files"]
                }
                for relative in authority.ADAPTER_RELATIVE_FILES:
                    observed = fd_view_sha(adapter_view / relative)
                    self.assertEqual(observed, adapter_rows[relative]["sha256"])
                    adapter_hashes[relative] = observed
            self.rank_consumption_replays.append(
                {
                    "task_id": consumption["task_id"],
                    "rank": rank,
                    "model_view_root": str(model_view),
                    "adapter_hashes": adapter_hashes,
                    "fd_binding_digest": validated_fd_binding[
                        "fd_binding_digest"
                    ],
                }
            )

    def decoder(
        self,
        request_path: Path,
        output_path: Path,
        inherited_fd_binding: dict,
    ) -> dict:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        self.replay_consumption_as_four_ranks(
            request, inherited_fd_binding
        )
        consumption = json.loads(
            Path(request["model_consumption_input"]["path"]).read_text(
                encoding="utf-8"
            )
        )
        write(output_path, b"stub-exact81:" + request["task_id"].encode(), 0o600)
        native = {
            "schema_version": "injected-native-v2",
            "consumption_input_digest": consumption["consumption_input_digest"],
        }
        native["receipt_digest"] = plan.object_sha256(native)
        receipt_path = output_path.with_name(output_path.name + ".receipt.json")
        receipt_path.write_bytes(plan.canonical_json_bytes(native) + b"\n")
        receipt_path.chmod(0o400)
        return {"return_code": 0, "stdout": b"ok", "stderr": b""}

    def execute_holder(
        self, completion_anchor_sink: object | None = None
    ) -> dict:
        return executor.execute_shard(
            bundle=self.bundle,
            holder_job_id="136719",
            decoder_identity=self.decoder_identity,
            ffprobe_identity=self.ffprobe_identity,
            physical_bindings_identity=self.physical_identity,
            run_decoder=self.decoder,
            probe_video=lambda _path: probe(),
            verify_tools=False,
            injected_physical_bindings=self.physical_bindings,
            injected_model_consumption=self.model_authority,
            injected_proc_fd_prefix="/dev/fd",
            completion_anchor_sink=completion_anchor_sink,
        )


class ExecutorV2Tests(ExecutorV2Fixture):
    def test_claim_failure_closes_all_retained_shard_resources(self) -> None:
        captured_handles: dict[str, executor._HeldDirectory] = {}
        reservations: list[executor._HeldCompletionReservation] = []
        original_handles = executor._holder_directory_handles
        original_capture = executor._HeldCompletionReservation.capture
        original_write_json = executor._HeldDirectory.write_json

        def capture_handles(*args: object, **kwargs: object) -> dict:
            value = original_handles(*args, **kwargs)
            captured_handles.update(value)
            return value

        def capture_reservation(*args: object, **kwargs: object) -> object:
            value = original_capture(*args, **kwargs)
            reservations.append(value)
            return value

        def fail_claim(
            held: executor._HeldDirectory,
            name: str,
            value: object,
        ) -> Path:
            if name == executor.EXECUTION_CLAIM_FILENAME:
                raise RuntimeError("injected claim write failure")
            return original_write_json(held, name, value)

        with mock.patch.object(
            executor, "_holder_directory_handles", side_effect=capture_handles
        ), mock.patch.object(
            executor._HeldCompletionReservation,
            "capture",
            side_effect=capture_reservation,
        ), mock.patch.object(
            executor._HeldDirectory, "write_json", new=fail_claim
        ), self.assertRaisesRegex(RuntimeError, "injected claim"):
            self.execute_holder()
        self.assertTrue(captured_handles)
        self.assertTrue(all(handle.closed for handle in captured_handles.values()))
        self.assertEqual(len(reservations), 1)
        self.assertTrue(reservations[0].closed)

    def test_model_capture_publication_failure_aborts_model_authority(self) -> None:
        original_write_json = executor._HeldDirectory.write_json

        def fail_model_capture(
            held: executor._HeldDirectory,
            name: str,
            value: object,
        ) -> Path:
            if name == executor.MODEL_CAPTURE_FILENAME:
                raise RuntimeError("injected model capture publication failure")
            return original_write_json(held, name, value)

        with mock.patch.object(
            executor._HeldDirectory, "write_json", new=fail_model_capture
        ), self.assertRaisesRegex(RuntimeError, "model capture publication"):
            self.execute_holder()
        self.assertTrue(self.model_authority._closed)

    def test_adapter_begin_failure_aborts_and_closes_retained_view(self) -> None:
        captured: list[authority.AdapterAuthority] = []
        original_capture = authority.AdapterAuthority.capture

        def capture_adapter(*args: object, **kwargs: object) -> object:
            value = original_capture(*args, **kwargs)
            captured.append(value)
            return value

        with mock.patch.object(
            authority.AdapterAuthority,
            "capture",
            side_effect=capture_adapter,
        ), mock.patch.object(
            authority.AdapterAuthority,
            "begin_use",
            side_effect=RuntimeError("injected adapter begin failure"),
        ), self.assertRaisesRegex(RuntimeError, "adapter begin failure"):
            self.execute_holder()
        self.assertEqual(len(captured), 1)
        self.assertTrue(captured[0]._closed)
        self.assertFalse(captured[0].view_root.exists())

    def test_publication_gate_failure_closes_retained_staging_fd(self) -> None:
        retained: list[executor._RetainedTaskMedia] = []
        original_capture = executor._RetainedTaskMedia.capture
        original_write_json = executor._HeldDirectory.write_json

        def capture_media(*args: object, **kwargs: object) -> object:
            value = original_capture(*args, **kwargs)
            retained.append(value)
            return value

        def fail_gate(
            held: executor._HeldDirectory,
            name: str,
            value: object,
        ) -> Path:
            if name == executor.PUBLICATION_GATE_FILENAME:
                raise RuntimeError("injected publication gate failure")
            return original_write_json(held, name, value)

        with mock.patch.object(
            executor._RetainedTaskMedia,
            "capture",
            side_effect=capture_media,
        ), mock.patch.object(
            executor._HeldDirectory, "write_json", new=fail_gate
        ), mock.patch.object(
            executor,
            "_publish_failure",
            side_effect=RuntimeError("stop after injected failure"),
        ), self.assertRaisesRegex(RuntimeError, "stop after"):
            self.execute_holder()
        self.assertEqual(len(retained), 1)
        self.assertTrue(retained[0].closed)
        with self.assertRaises(OSError):
            os.fstat(retained[0].descriptor)

    def test_final_publication_replay_failure_closes_staging_fd(self) -> None:
        retained: list[executor._RetainedTaskMedia] = []
        replay_counts: dict[int, int] = {}
        original_capture = executor._RetainedTaskMedia.capture
        original_replay = executor._RetainedTaskMedia.replay_published

        def capture_media(*args: object, **kwargs: object) -> object:
            value = original_capture(*args, **kwargs)
            retained.append(value)
            return value

        def fail_second_replay(
            held: executor._RetainedTaskMedia,
            **kwargs: object,
        ) -> None:
            key = id(held)
            replay_counts[key] = replay_counts.get(key, 0) + 1
            if replay_counts[key] == 2:
                raise RuntimeError("injected final publication replay failure")
            original_replay(held, **kwargs)

        with mock.patch.object(
            executor._RetainedTaskMedia,
            "capture",
            side_effect=capture_media,
        ), mock.patch.object(
            executor._RetainedTaskMedia,
            "replay_published",
            new=fail_second_replay,
        ), self.assertRaisesRegex(RuntimeError, "final publication replay"):
            self.execute_holder()
        self.assertEqual(len(retained), 1)
        self.assertTrue(retained[0].closed)
        with self.assertRaises(OSError):
            os.fstat(retained[0].descriptor)

    def test_exact66_consumption_chain_precedes_publication(self) -> None:
        anchors: list[dict] = []
        summary = self.execute_holder(anchors.append)
        self.assertEqual(summary["planned_task_count"], 66)
        self.assertEqual(summary["success_count"], 66)
        self.assertEqual(summary["failure_count"], 0)
        self.assertEqual(len(summary["results"]), 66)
        self.assertEqual(len(self.rank_consumption_replays), 66 * 4)
        self.assertEqual(
            {row["rank"] for row in self.rank_consumption_replays},
            {0, 1, 2, 3},
        )
        self.assertEqual(
            len({row["consumption_digest"] for row in summary["results"]}), 66
        )
        self.assertTrue(
            all(row["publication_gate_digest"] is not None for row in summary["results"])
        )
        shard = self.evaluation_root / executor.EXECUTION_DIRECTORY / "136719"
        for result in summary["results"]:
            task = shard / "tasks" / result["task_id"]
            chain = json.loads(
                (task / executor.CONSUMPTION_CHAIN_FILENAME).read_text(encoding="utf-8")
            )
            gate = json.loads(
                (task / executor.PUBLICATION_GATE_FILENAME).read_text(encoding="utf-8")
            )
            output = self.evaluation_root / result["output_relpath"]
            self.assertEqual(chain["consumption_digest"], result["consumption_digest"])
            self.assertEqual(gate["consumption_digest"], result["consumption_digest"])
            self.assertTrue(gate["post_use_closed_before_publication"] if "post_use_closed_before_publication" in gate else gate["publication_authorized"])
            self.assertEqual(output.stat().st_ino, (task / executor.STAGING_VIDEO_FILENAME).stat().st_ino)
        authority_root = shard / executor.CONSUMPTION_AUTHORITY_DIRECTORY
        self.assertEqual(
            {path.name for path in authority_root.iterdir()},
            {executor.MODEL_CAPTURE_FILENAME, executor.MODEL_FINAL_FILENAME},
        )
        completion_path = (
            self.evaluation_root
            / plan.holder_completion_reservation_relative("136719")
        )
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        topology = plan.build_directory_topology(
            self.bundle["manifest"], input_spec=self.bundle["input_spec"]
        )
        validated_completion = plan.validate_holder_directory_completion(
            completion,
            topology=topology,
            base_directory_authority=self.bundle["directory_authority"],
        )
        reservation = next(
            row
            for row in self.bundle["publication_receipt"][
                "holder_completion_reservations"
            ]
            if row["holder_job_id"] == "136719"
        )
        self.assertEqual(
            validated_completion["holder_summary_digest"],
            summary["summary_digest"],
        )
        self.assertEqual(validated_completion["row_count"], 101)
        self.assertEqual(
            completion_path.stat().st_ino, reservation["identity"]["inode"]
        )
        self.assertEqual(completion_path.stat().st_mode & 0o777, 0o444)
        self.assertEqual(len(anchors), 1)
        anchor = executor.validate_holder_completion_anchor(anchors[0])
        self.assertEqual(anchor["holder_job_id"], "136719")
        self.assertEqual(
            anchor["completion_sha256"],
            hashlib.sha256(completion_path.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            anchor["completion_digest"], completion["completion_digest"]
        )
        self.assertEqual(
            anchor["holder_summary_digest"], summary["summary_digest"]
        )

        # Replacing both names with a same-byte hard-linked inode must not be
        # mistaken for the exact retained staging inode published at gate time.
        first = summary["results"][0]
        first_task = shard / "tasks" / first["task_id"]
        first_staging = first_task / executor.STAGING_VIDEO_FILENAME
        first_final = self.evaluation_root / first["output_relpath"]
        same_bytes = first_staging.read_bytes()
        first_staging.unlink()
        first_final.unlink()
        write(first_staging, same_bytes, 0o444)
        os.link(first_staging, first_final)
        input_receipt = json.loads(
            (first_task / executor.INPUT_RECEIPT_FILENAME).read_text("utf-8")
        )
        process_receipt = json.loads(
            (first_task / executor.PROCESS_RECEIPT_FILENAME).read_text("utf-8")
        )
        output_receipt = json.loads(
            (first_task / executor.OUTPUT_RECEIPT_FILENAME).read_text("utf-8")
        )
        with self.assertRaisesRegex(
            executor.DecodedEvaluationExecutorError,
            "bytes/inode differ",
        ):
            executor.validate_output_receipt(
                output_receipt,
                input_receipt=input_receipt,
                process_receipt=process_receipt,
                output_path=first_final,
            )

    def test_model_mutation_aborts_before_any_final_publication(self) -> None:
        calls = 0

        def hostile(
            request_path: Path,
            output_path: Path,
            inherited_fd_binding: dict,
        ) -> dict:
            nonlocal calls
            calls += 1
            if calls == 1:
                target = self.model / "transformer/config.json"
                target.write_bytes(target.read_bytes() + b"hostile")
            return self.decoder(
                request_path, output_path, inherited_fd_binding
            )

        with self.assertRaisesRegex(
            executor.DecodedEvaluationExecutorError,
            "consumption authority failed",
        ):
            executor.execute_shard(
                bundle=self.bundle,
                holder_job_id="136719",
                decoder_identity=self.decoder_identity,
                ffprobe_identity=self.ffprobe_identity,
                physical_bindings_identity=self.physical_identity,
                run_decoder=hostile,
                probe_video=lambda _path: probe(),
                verify_tools=False,
                injected_physical_bindings=self.physical_bindings,
                injected_model_consumption=self.model_authority,
                injected_proc_fd_prefix="/dev/fd",
            )
        self.assertEqual(calls, 1)
        self.assertEqual(list((self.evaluation_root / "candidates").rglob("*.mp4")), [])

    def test_adapter_fd_view_retarget_aborts_before_publication(self) -> None:
        retargeted = False

        def hostile(
            request_path: Path,
            output_path: Path,
            inherited_fd_binding: dict,
        ) -> dict:
            nonlocal retargeted
            observation = self.decoder(
                request_path, output_path, inherited_fd_binding
            )
            request = json.loads(request_path.read_text(encoding="utf-8"))
            identity = request["model_consumption_input"]
            with mock.patch.dict(
                os.environ,
                {
                    authority.INHERITED_FD_BINDING_ENV:
                    authority.inherited_fd_environment_value(
                        inherited_fd_binding
                    )
                },
                clear=False,
            ):
                _consumption, _model_capture, adapter_capture = (
                    authority.load_consumption_input(
                        identity["path"],
                        expected_sha256=identity["sha256"],
                        expected_digest=identity["consumption_input_digest"],
                    )
                )
            if adapter_capture is not None and not retargeted:
                leaf = (
                    Path(adapter_capture["adapter_view_root"])
                    / "adapter/adapter_model.safetensors"
                )
                leaf.unlink()
                leaf.symlink_to("/dev/null")
                retargeted = True
            return observation

        with self.assertRaisesRegex(
            executor.DecodedEvaluationExecutorError,
            "post-use consumption authority",
        ):
            executor.execute_shard(
                bundle=self.bundle,
                holder_job_id="136719",
                decoder_identity=self.decoder_identity,
                ffprobe_identity=self.ffprobe_identity,
                physical_bindings_identity=self.physical_identity,
                run_decoder=hostile,
                probe_video=lambda _path: probe(),
                verify_tools=False,
                injected_physical_bindings=self.physical_bindings,
                injected_model_consumption=self.model_authority,
                injected_proc_fd_prefix="/dev/fd",
            )
        self.assertTrue(retargeted)
        self.assertEqual(
            list((self.evaluation_root / "candidates").rglob("*.mp4")), []
        )

    def test_post_gate_staging_mutation_is_not_published(self) -> None:
        original = authority.build_publication_gate
        calls = 0

        def hostile_gate(**kwargs: object) -> dict:
            nonlocal calls
            gate = original(**kwargs)
            calls += 1
            if calls == 1:
                Path(kwargs["staging_path"]).write_bytes(
                    b"hostile-bytes-after-publication-gate"
                )
            return gate

        with mock.patch.object(
            authority, "build_publication_gate", side_effect=hostile_gate
        ):
            summary = self.execute_holder()
        self.assertEqual(summary["failure_count"], 1)
        failed = next(
            row for row in summary["results"] if row["status"] == "failure"
        )
        self.assertFalse(
            (self.evaluation_root / failed["output_relpath"]).exists()
        )

    def test_task_root_rename_replacement_cannot_redirect_decoder_artifacts(
        self,
    ) -> None:
        swapped = False

        def hostile(
            request_path: Path,
            output_path: Path,
            inherited_fd_binding: dict,
        ) -> dict:
            nonlocal swapped
            observation = self.decoder(
                request_path, output_path, inherited_fd_binding
            )
            if not swapped:
                task_root = request_path.parent
                moved = task_root.with_name(task_root.name + ".moved")
                task_root.rename(moved)
                task_root.mkdir(mode=0o700)
                swapped = True
            return observation

        with self.assertRaisesRegex(
            executor.DecodedEvaluationExecutorError,
            "retained directory|inherited authority FD identity|decoder consumption authority",
        ):
            executor.execute_shard(
                bundle=self.bundle,
                holder_job_id="136719",
                decoder_identity=self.decoder_identity,
                ffprobe_identity=self.ffprobe_identity,
                physical_bindings_identity=self.physical_identity,
                run_decoder=hostile,
                probe_video=lambda _path: probe(),
                verify_tools=False,
                injected_physical_bindings=self.physical_bindings,
                injected_model_consumption=self.model_authority,
                injected_proc_fd_prefix="/dev/fd",
            )
        self.assertTrue(swapped)
        self.assertEqual(
            list((self.evaluation_root / "candidates").rglob("*.mp4")), []
        )

    def test_output_parent_rename_replacement_is_rejected_before_link(
        self,
    ) -> None:
        original_gate = authority.build_publication_gate
        swapped_output: list[Path] = []

        def hostile_gate(**kwargs: object) -> dict:
            gate = original_gate(**kwargs)
            if not swapped_output:
                task_id = kwargs["consumption_chain"]["task_id"]
                record = next(
                    task["record"]
                    for shard in self.bundle["shards"].values()
                    for task in shard["tasks"]
                    if task["record"].get(
                        "candidate_id", task["record"].get("control_id")
                    ) == task_id
                )
                output = self.evaluation_root / record["output_relpath"]
                parent = output.parent
                moved = parent.with_name(parent.name + ".moved")
                parent.rename(moved)
                parent.mkdir(mode=0o700)
                swapped_output.append(output)
            return gate

        with mock.patch.object(
            authority, "build_publication_gate", side_effect=hostile_gate
        ):
            with self.assertRaisesRegex(
                executor.DecodedEvaluationExecutorError,
                "retained directory identity",
            ):
                self.execute_holder()
        self.assertEqual(len(swapped_output), 1)
        self.assertFalse(swapped_output[0].exists())


class PublishedFileReplayTests(unittest.TestCase):
    def test_native_stdout_binds_the_exact_staging_inode(self) -> None:
        identity = {
            "device": 1, "inode": 2, "uid": 3, "gid": 4,
            "mode": stat.S_IFREG | 0o444, "nlink": 1, "rdev": 0,
            "size": 9, "blocks": 1, "mtime_ns": 5, "ctime_ns": 6,
        }
        native = {
            "output": {
                "sha256": "a" * 64,
                "size": 9,
                "publication_identity": dict(identity),
            }
        }
        executor._validate_retained_staging_against_native(
            native_receipt=native,
            staging_sha256="a" * 64,
            staging_size=9,
            staging_identity=identity,
        )
        replaced = dict(identity)
        replaced["inode"] = 7
        with self.assertRaisesRegex(
            executor.DecodedEvaluationExecutorError, "bytes/inode differ"
        ):
            executor._validate_retained_staging_against_native(
                native_receipt=native,
                staging_sha256="a" * 64,
                staging_size=9,
                staging_identity=replaced,
            )

    def test_decoder_stdout_exact_two_lines_anchor_native_files(self) -> None:
        native = {"output": {"sha256": "a" * 64}, "receipt_digest": "b" * 64}
        decoder_result = {
            "receipt_sha256": "c" * 64,
            "receipt_digest": "b" * 64,
        }
        native_raw = plan.canonical_json_bytes(native) + b"\n"
        result_raw = plan.canonical_json_bytes(decoder_result) + b"\n"
        observed = executor._validate_decoder_stdout_authority(
            stdout=native_raw + result_raw,
            native_receipt_raw=native_raw,
            native_receipt=native,
            decoder_result=decoder_result,
        )
        self.assertEqual(observed, native)
        with self.assertRaisesRegex(
            executor.DecodedEvaluationExecutorError, "exact-two"
        ):
            executor._validate_decoder_stdout_authority(
                stdout=native_raw + result_raw + result_raw,
                native_receipt_raw=native_raw,
                native_receipt=native,
                decoder_result=decoder_result,
            )

    def test_completion_anchor_channel_is_stripped_from_decoder_environment(
        self,
    ) -> None:
        hostile = {
            "APV2_EVAL_COMPLETION_ANCHOR_CHANNEL": "signed-channel",
            "APV2_EVAL_COMPLETION_ANCHOR_SENT_DIGEST": "f" * 64,
        }
        with mock.patch.dict(os.environ, hostile, clear=False):
            sanitized = executor.sanitized_subprocess_environment()
        self.assertNotIn(
            "APV2_EVAL_COMPLETION_ANCHOR_CHANNEL", sanitized
        )
        self.assertNotIn(
            "APV2_EVAL_COMPLETION_ANCHOR_SENT_DIGEST", sanitized
        )

    def test_ffprobe_proc_fd_is_the_only_inherited_descriptor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            media = Path(temporary).resolve() / "retained.mp4"
            media.write_bytes(b"retained-media")
            descriptor = os.open(media, os.O_RDONLY)
            self.addCleanup(os.close, descriptor)
            os.set_inheritable(descriptor, False)
            completed = subprocess.CompletedProcess(
                args=[], returncode=0, stdout=b"{}", stderr=b""
            )
            with mock.patch.object(
                executor.subprocess, "run", return_value=completed
            ) as run, mock.patch.object(
                executor, "parse_ffprobe_json", return_value=probe()
            ):
                observed = executor.ffprobe_video_prober("/pinned/ffprobe")(
                    Path(f"/proc/self/fd/{descriptor}")
                )
            self.assertEqual(observed, probe())
            kwargs = run.call_args.kwargs
            self.assertEqual(kwargs["pass_fds"], (descriptor,))
            self.assertIs(kwargs["close_fds"], True)
            self.assertEqual(
                run.call_args.args[0][-1], f"/proc/self/fd/{descriptor}"
            )

    def test_production_publication_forces_destination_dir_fd_linkat(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            staging = root / "candidate.staging.mp4"
            final = root / "candidate.mp4"
            raw = b"retained-fd-publication"
            staging.write_bytes(raw)
            staging.chmod(0o600)
            gate = {
                "staging_sha256": sha(raw),
                "staging_size": len(raw),
            }
            original_link = os.link
            original_is_dir = Path.is_dir
            calls: list[tuple[str, str, dict[str, object]]] = []

            def proc_fd_is_dir(path: Path) -> bool:
                if path == Path("/proc/self/fd"):
                    return True
                return original_is_dir(path)

            def emulate_linux_linkat(
                source: str,
                destination: str,
                **kwargs: object,
            ) -> None:
                calls.append((source, destination, dict(kwargs)))
                original_link(
                    staging,
                    destination,
                    dst_dir_fd=kwargs["dst_dir_fd"],
                    follow_symlinks=False,
                )

            with mock.patch.object(
                executor.sys, "platform", "linux"
            ), mock.patch.object(
                Path, "is_dir", proc_fd_is_dir
            ), mock.patch.object(
                executor.os, "link", side_effect=emulate_linux_linkat
            ):
                held = executor._HeldDirectory.open_root(
                    root, label="publication test root"
                )
                try:
                    published = executor._publish_gated_staging_inode(
                        staging_path=staging,
                        final_path=final,
                        publication_gate=gate,
                        production_mode=True,
                        staging_directory=held,
                        final_directory=held,
                    )
                finally:
                    held.close()
            self.assertEqual(len(calls), 1)
            source, destination, kwargs = calls[0]
            self.assertRegex(source, r"^/proc/self/fd/[0-9]+$")
            self.assertEqual(destination, final.name)
            self.assertIs(type(kwargs["dst_dir_fd"]), int)
            self.assertIs(kwargs["follow_symlinks"], True)
            self.assertEqual(published, executor._stat_identity_row(final.lstat()))

    def test_named_swap_during_hash_cannot_splice_identity_and_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve() / "published.mp4"
            raw = b"same-published-video-bytes"
            path.write_bytes(raw)
            path.chmod(0o444)
            sibling = path.with_name("published.staging.mp4")
            os.link(path, sibling)
            expected_identity = executor._stat_identity_row(path.lstat())
            expected_sha = sha(raw)
            executor._stable_published_file(
                path,
                expected_identity=expected_identity,
                expected_sha256=expected_sha,
                expected_size=len(raw),
                label="stable published fixture",
            )

            original_hash_fd = executor._hash_fd
            swapped = False

            def swap_name_before_hash(descriptor: int) -> tuple[str, int]:
                nonlocal swapped
                if not swapped:
                    swapped = True
                    path.unlink()
                    sibling.unlink()
                    path.write_bytes(raw)
                    path.chmod(0o444)
                    os.link(path, sibling)
                return original_hash_fd(descriptor)

            with mock.patch.object(
                executor, "_hash_fd", side_effect=swap_name_before_hash
            ), self.assertRaisesRegex(
                executor.DecodedEvaluationExecutorError, "bytes/inode differ"
            ):
                executor._stable_published_file(
                    path,
                    expected_identity=expected_identity,
                    expected_sha256=expected_sha,
                    expected_size=len(raw),
                    label="hostile published fixture",
                )
            self.assertTrue(swapped)


class RetainedExecutorDirectoryTests(unittest.TestCase):
    def test_staging_probe_uses_retained_regular_file_fd(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            media = root / executor.STAGING_VIDEO_FILENAME
            media.write_bytes(b"retained-staging-video")
            held = executor._HeldDirectory.open_root(
                root, label="retained task root"
            )
            held.adopt_entries({executor.STAGING_VIDEO_FILENAME})
            retained = executor._RetainedTaskMedia.capture(
                held,
                name=executor.STAGING_VIDEO_FILENAME,
                label="staging test",
            )
            original_is_dir = Path.is_dir

            def proc_fd_is_dir(path: Path) -> bool:
                if path == Path("/proc/self/fd"):
                    return True
                return original_is_dir(path)

            try:
                with mock.patch.object(
                    executor.sys, "platform", "linux"
                ), mock.patch.object(Path, "is_dir", proc_fd_is_dir):
                    consumer = retained.consumer_path(production_mode=True)
                self.assertEqual(
                    consumer, Path(f"/proc/self/fd/{retained.descriptor}")
                )
                self.assertTrue(
                    stat.S_ISREG(os.fstat(retained.descriptor).st_mode)
                )
                self.assertFalse(os.get_inheritable(retained.descriptor))
            finally:
                retained.close()
                held.close()

    def test_relative_create_only_namespace_stays_on_retained_inode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            root = parent / "evaluation"
            root.mkdir(mode=0o700)
            held = executor._HeldDirectory.open_root(
                root,
                expected_entries=set(),
                label="test evaluation root",
            )
            child = held.create_child("tasks", label="task parent")
            artifact = child.write_json("receipt.json", {"ok": True})
            self.assertEqual(
                artifact.read_bytes(),
                executor.canonical_json_bytes({"ok": True}) + b"\n",
            )
            child.replay(expected_entries={"receipt.json"})
            held.replay(expected_entries={"tasks"})
            child.close()
            held.close()

    def test_same_name_root_replacement_is_rejected_before_write(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            root = parent / "evaluation"
            root.mkdir(mode=0o700)
            held = executor._HeldDirectory.open_root(
                root,
                expected_entries=set(),
                label="test evaluation root",
            )
            moved = parent / "evaluation.moved"
            root.rename(moved)
            root.mkdir(mode=0o700)
            with self.assertRaisesRegex(
                executor.DecodedEvaluationExecutorError,
                "identity or entry closure differs",
            ):
                held.create_child("tasks", label="task parent")
            self.assertEqual(list(root.iterdir()), [])
            held.close()

    def test_child_replacement_is_rejected_before_create_only_write(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            root = parent / "evaluation"
            root.mkdir(mode=0o700)
            held = executor._HeldDirectory.open_root(
                root,
                expected_entries=set(),
                label="test evaluation root",
            )
            child = held.create_child("tasks", label="task parent")
            moved = root / "tasks.moved"
            (root / "tasks").rename(moved)
            (root / "tasks").mkdir(mode=0o700)
            with self.assertRaisesRegex(
                executor.DecodedEvaluationExecutorError,
                "identity or entry closure differs",
            ):
                child.write_json("receipt.json", {"hostile": False})
            self.assertEqual(list((root / "tasks").iterdir()), [])
            child.close()
            held.close()


if __name__ == "__main__":
    unittest.main()
