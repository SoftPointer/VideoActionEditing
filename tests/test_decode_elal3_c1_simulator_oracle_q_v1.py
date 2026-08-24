from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "methods/bernini_action_editing/decode_elal3_c1_simulator_oracle_q_v1.py"
)
SPEC = importlib.util.spec_from_file_location("elal3_c1_decoder_under_test", SOURCE)
assert SPEC is not None and SPEC.loader is not None
decoder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = decoder
SPEC.loader.exec_module(decoder)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def write_plain(path: Path, raw: bytes, mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.chmod(mode)


def write_json(path: Path, value: dict) -> str:
    raw = decoder.canonical_json_bytes(value) + b"\n"
    write_plain(path, raw)
    return sha(raw)


class DecoderContractTests(unittest.TestCase):
    def build_decode_manifest(
        self,
        root: Path,
        *,
        decoder_sha: str = "a" * 64,
    ):
        unsigned = {
            "schema_version": decoder.DECODE_RELEASE_SCHEMA,
            "scope": decoder.DECODE_RELEASE_SCOPE,
            "row_id": decoder.ROW_ID,
            "archive_format": decoder.DECODE_ARCHIVE_FORMAT,
            "archive_sha256": "e" * 64,
            "archive_size": 10240,
            "decoder_member": decoder.DECODER_RELATIVE,
            "decoder_source_sha256": decoder_sha,
            "files": [
                {
                    "path": decoder.CHECKPOINT_CONTENT_MANIFEST_RELATIVE,
                    "sha256": decoder.CHECKPOINT_CONTENT_MANIFEST_SHA256,
                    "size": decoder.CHECKPOINT_CONTENT_MANIFEST_SIZE,
                    "archive_mode": 0o444,
                },
                {
                    "path": decoder.DECODER_RELATIVE,
                    "sha256": decoder_sha,
                    "size": 12345,
                    "archive_mode": 0o444,
                }
            ],
            "checkpoint_content_manifest": {
                "member": decoder.CHECKPOINT_CONTENT_MANIFEST_RELATIVE,
                "sha256": decoder.CHECKPOINT_CONTENT_MANIFEST_SHA256,
                "size": decoder.CHECKPOINT_CONTENT_MANIFEST_SIZE,
                "row_count": decoder.CHECKPOINT_CONTENT_MANIFEST_ROW_COUNT,
            },
            "training_release": {
                "manifest_sha256": decoder.TRAINING_RELEASE_MANIFEST_SHA256,
                "manifest_digest": decoder.TRAINING_RELEASE_MANIFEST_DIGEST,
                "archive_sha256": decoder.TRAINING_RELEASE_ARCHIVE_SHA256,
                "trainer_source_sha256": decoder.TRAINER_SOURCE_SHA256,
            },
            "training_artifacts_by_seed": copy.deepcopy(
                decoder.TRAINING_ARTIFACTS_BY_SEED
            ),
            "runtime": {
                "world_size": 4,
                "ulysses_size": 4,
                "num_inference_steps": 40,
                "authorized_training_seeds": list(
                    decoder.AUTHORIZED_TRAINING_SEEDS
                ),
                "sampling_seed_equals_training_seed": True,
                "branch_order": list(decoder.REVIEW_BRANCH_ORDER),
            },
            "authority_bindings": {
                "model_authority_sha256": decoder.MODEL_AUTHORITY_SHA256,
                "derivative_authority_sha256": decoder.DERIVATIVE_AUTHORITY_SHA256,
                "packet_manifest_sha256": decoder.PACKET_MANIFEST_SHA256,
                "latent_bundle_sha256": decoder.LATENT_BUNDLE_SHA256,
                "latent_bundle_receipt_sha256": decoder.LATENT_BUNDLE_RECEIPT_SHA256,
                "checkpoint_content_manifest_sha256": (
                    decoder.CHECKPOINT_CONTENT_MANIFEST_SHA256
                ),
            },
            "formal_c1_authorized": False,
            "exact160_authorized": False,
            "source_instruction_inference_authorized": False,
            "real_video_generalization_authorized": False,
            "production_model_authorized": False,
            "scientific_claim_authorized": False,
        }
        value = {**unsigned, "manifest_digest": decoder.object_sha256(unsigned)}
        path = root / "decode.manifest.json"
        manifest_sha = write_json(path, value)
        return path, manifest_sha, value

    def build_release(self, root: Path):
        release = root / "source"
        release.mkdir()
        runtime = {}
        rows = []
        all_names = sorted(
            set(decoder.REQUIRED_RUNTIME)
            | {
                decoder.DERIVATIVE_RELATIVE,
                decoder.MODEL_RELATIVE,
                decoder.LATENT_RECEIPT_RELATIVE,
            }
        )
        for index, name in enumerate(all_names):
            raw = f"frozen-{index}-{name}\n".encode("ascii")
            path = release.joinpath(*Path(name).parts)
            write_plain(path, raw)
            row = {
                "path": name,
                "sha256": sha(raw),
                "size": len(raw),
                "mode": "0444",
            }
            rows.append(row)
            if name in decoder.REQUIRED_RUNTIME:
                runtime[name] = row["sha256"]
        unsigned = {
            "schema_version": decoder.RELEASE_SCHEMA,
            "row_id": decoder.ROW_ID,
            "execution_scope": "simulator_oracle_q_exact_one_row_optimizer_diagnostic_only",
            "simulator_optimizer_diagnostic_authorized": True,
            "teacher_forced_oracle_q_required": True,
            "representation_variant": "full",
            "attention_width": 64,
            "lora_rank": 256,
            "optimizer_update_sequence": [0, 1, 10],
            "maximum_authorized_optimizer_updates": 20,
            "archive_format": "test",
            "archive_sha256": "0" * 64,
            "archive_size": 1,
            "distributed_topology": {},
            "run_assignments": [],
            "authority_bindings": {
                "derivative_authority_sha256": decoder.DERIVATIVE_AUTHORITY_SHA256,
                "derivative_authority_digest": decoder.DERIVATIVE_AUTHORITY_DIGEST,
                "model_authority_sha256": decoder.MODEL_AUTHORITY_SHA256,
                "model_authority_digest": decoder.MODEL_AUTHORITY_DIGEST,
                "latent_receipt_sha256": decoder.LATENT_BUNDLE_RECEIPT_SHA256,
                "latent_receipt_digest": "1" * 64,
                "packet_manifest_sha256": decoder.PACKET_MANIFEST_SHA256,
            },
            "external_latent_bundle": {
                "sha256": decoder.LATENT_BUNDLE_SHA256,
                "size": 39_138_208,
                "mode": "0444",
                "nlink": 1,
            },
            "files": rows,
            "runtime_pins": runtime,
            "formal_c1_authorized": False,
            "exact160_authorized": False,
            "source_instruction_inference_authorized": False,
            "real_video_generalization_authorized": False,
            "production_model_authorized": False,
            "scientific_claim_authorized": False,
        }
        manifest = {**unsigned, "manifest_digest": decoder.object_sha256(unsigned)}
        manifest_path = root / "source.manifest.json"
        manifest_sha = write_json(manifest_path, manifest)
        closure = decoder.validate_release_closure(
            release.resolve(),
            manifest_path.resolve(),
            expected_manifest_sha256=manifest_sha,
            expected_trainer_sha256=runtime[decoder.TRAINER_RELATIVE],
        )
        return closure, manifest_path, manifest_sha

    def build_training(self, root: Path, release):
        run = root / "elal3_c1_training"
        run.mkdir()
        initial = "2" * 64
        final = "3" * 64
        records = []
        adapter_shas = []
        for step, parameter_sha in ((0, initial), (10, final)):
            directory = run / "checkpoints" / f"checkpoint-{step:08d}"
            directory.mkdir(parents=True)
            adapter_raw = f"adapter-step-{step}".encode("ascii")
            adapter = directory / "adapter-and-elal3.pt"
            write_plain(adapter, adapter_raw)
            adapter_sha = sha(adapter_raw)
            adapter_shas.append(adapter_sha)
            metadata = {
                "schema_version": decoder.CHECKPOINT_SCHEMA,
                "step": step,
                "row_id": decoder.ROW_ID,
                "adapter_file": adapter.name,
                "adapter_sha256": adapter_sha,
                "trainable_parameter_sha256": parameter_sha,
                "strict_weights_only_reload_verified": True,
                "oracle_q_teacher_forced": True,
                "source_instruction_inference": False,
                "formal_c1_authorized": False,
                "exact160_authorized": False,
                "scientific_claim_authorized": False,
                "lora_affines": decoder.LORA_AFFINES,
                "lora_rank": decoder.LORA_RANK,
                "elal3_variant": "full-w64",
                "trainable_parameter_count": decoder.TRAINABLE_PARAMETERS,
            }
            checkpoint_receipt = {
                **metadata,
                "receipt_digest": decoder.object_sha256(metadata),
            }
            checkpoint_receipt_path = directory / "CHECKPOINT_RECEIPT.json"
            checkpoint_receipt_sha = write_json(
                checkpoint_receipt_path, checkpoint_receipt
            )
            records.append(
                {
                    "step": step,
                    "path": str(directory.resolve()),
                    "adapter_sha256": adapter_sha,
                    "optimizer_sha256": None if step == 0 else "4" * 64,
                    "checkpoint_receipt_sha256": checkpoint_receipt_sha,
                }
            )
        source_names = {
            "train_lora": "methods/bernini_action_editing/train_lora.py",
            "elal3_core": "methods/bernini_action_editing/elal3_c0_v1.py",
            "elal3_label": "methods/bernini_action_editing/elal3_simulator_label_v1.py",
            "packed_lora": "methods/bernini_action_editing/packed_preservation_lora_v2.py",
            "runtime": "methods/bernini_action_editing/source_self_runtime.py",
            "sigma": "methods/bernini_action_editing/inference_sigma_strata.py",
        }
        unsigned = {
            "schema_version": decoder.TRAINING_RECEIPT_SCHEMA,
            "status": "TRAINING_COMPLETE_SIMULATOR_ORACLE_Q_OVERFIT_DIAGNOSTIC_ONLY",
            "row_id": decoder.ROW_ID,
            "completed_optimizer_steps": 10,
            "requested_optimizer_steps": 10,
            "preflight_only": False,
            "fresh_initialization_verified": True,
            "parameters_changed": True,
            "decoded_review_pending": True,
            "oracle_q_teacher_forced": True,
            "source_instruction_inference": False,
            "formal_c1_authorized": False,
            "exact160_authorized": False,
            "scientific_claim_authorized": False,
            "real_video_data": False,
            "lora_affines": decoder.LORA_AFFINES,
            "lora_rank": decoder.LORA_RANK,
            "elal3_variant": "full-w64",
            "trainable_parameter_count": decoder.TRAINABLE_PARAMETERS,
            "latent_bundle_sha256": decoder.LATENT_BUNDLE_SHA256,
            "latent_bundle_receipt_sha256": decoder.LATENT_BUNDLE_RECEIPT_SHA256,
            "external_optimizer_authority_sha256": decoder.DERIVATIVE_AUTHORITY_SHA256,
            "model_authority_sha256": decoder.MODEL_AUTHORITY_SHA256,
            "model_authority_digest": decoder.MODEL_AUTHORITY_DIGEST,
            "runtime_placement": {
                "holder_job_id": "141620",
                "node": "auh7-1b-gpu-226",
            },
            "seed": 20260817,
            "initial_parameter_sha256": initial,
            "final_parameter_sha256": final,
            "local_source_closure": {
                logical: {"sha256": release.runtime_pins[relative]}
                for logical, relative in source_names.items()
            },
            "checkpoint_records": records,
        }
        receipt = {**unsigned, "receipt_digest": decoder.object_sha256(unsigned)}
        receipt_sha = write_json(run / "TRAINING_RECEIPT.json", receipt)
        return run, receipt_sha, adapter_shas

    def test_sampler_and_distributed_contract(self):
        value = decoder.sampler_contract(steps=40, seed=20260817)
        self.assertEqual(value["guidance_mode"], "v2v_apg")
        self.assertEqual(value["flow_shift"], 5.0)
        distributed = decoder.distributed_contract(
            {"WORLD_SIZE": "4", "RANK": "2", "LOCAL_RANK": "2"}
        )
        self.assertEqual(distributed.rank, 2)
        with self.assertRaises(decoder.ELAL3C1DecodeError):
            decoder.sampler_contract(steps=10, seed=1)
        with self.assertRaises(decoder.ELAL3C1DecodeError):
            decoder.distributed_contract(
                {"WORLD_SIZE": "8", "RANK": "0", "LOCAL_RANK": "0"}
            )

    def test_html_has_exact9_and_hard_claim_boundaries(self):
        rows = [
            {
                "key": f"k{index}",
                "label": f"label {index}",
                "relative_path": f"{index}.mp4",
                "q_condition": "oracle" if index else "no q",
                "sha256": f"{index:x}" * 64,
            }
            for index in range(9)
        ]
        raw = decoder.build_review_html(
            instruction="red pushes blue",
            training_seed=1,
            sampling_seed=2,
            sampling_steps=40,
            rows=rows,
        )
        text = raw.decode("utf-8")
        self.assertEqual(text.count("<video "), 9)
        self.assertIn("SIMULATOR ORACLE-Q", text)
        self.assertIn("NOT SOURCE+INSTRUCTION", text)
        self.assertIn("NOT FORMAL C1", text)
        self.assertIn("do not demonstrate an action encoder", text)

    def test_release_closure_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            closure, manifest, manifest_sha = self.build_release(root)
            self.assertEqual(
                closure.runtime_pins[decoder.TRAINER_RELATIVE],
                sha(closure.trainer_path.read_bytes()),
            )
            target = closure.root / "methods/bernini_action_editing/elal3_c0_v1.py"
            target.chmod(0o644)
            target.write_bytes(b"tampered\n")
            target.chmod(0o444)
            with self.assertRaises(decoder.ELAL3C1DecodeError):
                decoder.validate_release_closure(
                    closure.root,
                    manifest,
                    expected_manifest_sha256=manifest_sha,
                    expected_trainer_sha256=closure.runtime_pins[
                        decoder.TRAINER_RELATIVE
                    ],
                )

    def test_training_run_checkpoint_cross_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            release, _, _ = self.build_release(root)
            run, receipt_sha, adapter_shas = self.build_training(root, release)
            binding = decoder.validate_training_run(
                run.resolve(),
                expected_seed=20260817,
                expected_runtime_placement={
                    "holder_job_id": "141620",
                    "node": "auh7-1b-gpu-226",
                },
                expected_receipt_sha256=receipt_sha,
                expected_step0_adapter_sha256=adapter_shas[0],
                expected_trained_adapter_sha256=adapter_shas[1],
                release=release,
            )
            self.assertEqual(binding.completed_steps, 10)
            self.assertEqual(binding.step0.step, 0)
            self.assertEqual(binding.trained.step, 10)
            with self.assertRaises(decoder.ELAL3C1DecodeError):
                decoder.validate_training_run(
                    run.resolve(),
                    expected_seed=20260817,
                    expected_runtime_placement={
                        "holder_job_id": "141620",
                        "node": "auh7-1b-gpu-226",
                    },
                    expected_receipt_sha256=receipt_sha,
                    expected_step0_adapter_sha256="f" * 64,
                    expected_trained_adapter_sha256=adapter_shas[1],
                    release=release,
                )
            with self.assertRaises(decoder.ELAL3C1DecodeError):
                decoder.validate_training_run(
                    run.resolve(),
                    expected_seed=20260817,
                    expected_runtime_placement={
                        "holder_job_id": "141618",
                        "node": "auh7-1b-gpu-249",
                    },
                    expected_receipt_sha256=receipt_sha,
                    expected_step0_adapter_sha256=adapter_shas[0],
                    expected_trained_adapter_sha256=adapter_shas[1],
                    release=release,
                )

    def test_validate_args_requires_scope_ack_and_fresh_elal_path(self):
        cli = decoder.parser()
        self.assertEqual(
            sum(
                "--expected-decode-release-manifest-sha256" in action.option_strings
                for action in cli._actions
            ),
            1,
        )
        self.assertEqual(
            sum(
                "--expected-checkpoint-content-manifest-sha256"
                in action.option_strings
                for action in cli._actions
            ),
            1,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = dict(
                ack_simulator_oracle_q_only=True,
                ack_not_source_instruction_inference=True,
                ack_not_formal_c1=True,
                ack_no_scientific_claim=True,
                expected_decode_release_manifest_sha256="0" * 64,
                expected_checkpoint_content_manifest_sha256=(
                    decoder.CHECKPOINT_CONTENT_MANIFEST_SHA256
                ),
                expected_decode_launcher_sha256="1" * 64,
                expected_release_manifest_sha256="a" * 64,
                expected_decoder_source_sha256="b" * 64,
                expected_trainer_source_sha256="c" * 64,
                expected_training_receipt_sha256="d" * 64,
                expected_step0_adapter_sha256="e" * 64,
                expected_trained_adapter_sha256="f" * 64,
                num_inference_steps=40,
                sampling_seed=20260817,
                output=str(root / "elal3_c1_review"),
            )
            output = decoder.validate_args(argparse.Namespace(**base))
            self.assertEqual(output, root / "elal3_c1_review")
            base["ack_not_formal_c1"] = False
            with self.assertRaises(decoder.ELAL3C1DecodeError):
                decoder.validate_args(argparse.Namespace(**base))

    def test_decode_release_semantic_binding_and_negative(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, manifest_sha, value = self.build_decode_manifest(root)
            selected = decoder.TRAINING_ARTIFACTS_BY_SEED["20260817"]
            placement = {"holder_job_id": "141620", "node": "auh7-1b-gpu-226"}
            loaded = decoder.validate_decode_release_manifest_v3(
                path.resolve(),
                expected_sha256=manifest_sha,
                expected_decoder_source_sha256="a" * 64,
                expected_checkpoint_content_manifest_sha256=(
                    decoder.CHECKPOINT_CONTENT_MANIFEST_SHA256
                ),
                expected_trainer_source_sha256=decoder.TRAINER_SOURCE_SHA256,
                sampling_seed=20260817,
                runtime_placement=placement,
                expected_training_receipt_sha256=selected["training_receipt_sha256"],
                expected_step0_adapter_sha256=selected["step0_adapter_sha256"],
                expected_trained_adapter_sha256=selected["trained_adapter_sha256"],
            )
            self.assertEqual(loaded["runtime"]["world_size"], 4)
            hostile = copy.deepcopy(value)
            hostile["runtime"]["world_size"] = 8
            hostile_unsigned = dict(hostile)
            hostile_unsigned.pop("manifest_digest")
            hostile["manifest_digest"] = decoder.object_sha256(hostile_unsigned)
            hostile_path = root / "hostile.manifest.json"
            hostile_sha = write_json(hostile_path, hostile)
            with self.assertRaises(decoder.ELAL3C1DecodeError):
                decoder.validate_decode_release_manifest_v3(
                    hostile_path.resolve(),
                    expected_sha256=hostile_sha,
                    expected_decoder_source_sha256="a" * 64,
                    expected_checkpoint_content_manifest_sha256=(
                        decoder.CHECKPOINT_CONTENT_MANIFEST_SHA256
                    ),
                    expected_trainer_source_sha256=decoder.TRAINER_SOURCE_SHA256,
                    sampling_seed=20260817,
                    runtime_placement=placement,
                    expected_training_receipt_sha256=selected["training_receipt_sha256"],
                    expected_step0_adapter_sha256=selected["step0_adapter_sha256"],
                    expected_trained_adapter_sha256=selected["trained_adapter_sha256"],
                )

            old = copy.deepcopy(value)
            old["schema_version"] = (
                "bernini-elal3-c1-simulator-oracle-q-decode-release-v2"
            )
            old_unsigned = dict(old)
            old_unsigned.pop("manifest_digest")
            old["manifest_digest"] = decoder.object_sha256(old_unsigned)
            old_path = root / "old-v1.manifest.json"
            old_sha = write_json(old_path, old)
            with self.assertRaises(decoder.ELAL3C1DecodeError):
                decoder.validate_decode_release_manifest_v3(
                    old_path.resolve(),
                    expected_sha256=old_sha,
                    expected_decoder_source_sha256="a" * 64,
                    expected_checkpoint_content_manifest_sha256=(
                        decoder.CHECKPOINT_CONTENT_MANIFEST_SHA256
                    ),
                    expected_trainer_source_sha256=decoder.TRAINER_SOURCE_SHA256,
                    sampling_seed=20260817,
                    runtime_placement=placement,
                    expected_training_receipt_sha256=selected[
                        "training_receipt_sha256"
                    ],
                    expected_step0_adapter_sha256=selected[
                        "step0_adapter_sha256"
                    ],
                    expected_trained_adapter_sha256=selected[
                        "trained_adapter_sha256"
                    ],
                )

            with self.assertRaises(decoder.ELAL3C1DecodeError):
                decoder.validate_decode_release_manifest_v3(
                    path.resolve(),
                    expected_sha256=manifest_sha,
                    expected_decoder_source_sha256="a" * 64,
                    expected_checkpoint_content_manifest_sha256=(
                        decoder.CHECKPOINT_CONTENT_MANIFEST_SHA256
                    ),
                    expected_trainer_source_sha256=decoder.TRAINER_SOURCE_SHA256,
                    sampling_seed=20260818,
                    runtime_placement=placement,
                    expected_training_receipt_sha256=selected[
                        "training_receipt_sha256"
                    ],
                    expected_step0_adapter_sha256=selected[
                        "step0_adapter_sha256"
                    ],
                    expected_trained_adapter_sha256=selected[
                        "trained_adapter_sha256"
                    ],
                )

    def test_world4_model_authority_real_path_and_hostile_replay(self):
        authority_path = (
            ROOT
            / "md/action_editing/20260817_box/evidence/"
            "elal3_c1_real_model_authority_v1.json"
        ).resolve()
        self.assertEqual(
            decoder.file_sha256(authority_path), decoder.MODEL_AUTHORITY_SHA256
        )
        import json

        reference = json.loads(authority_path.read_text(encoding="utf-8"))
        receipt = decoder.require_decoder_model_authority_replay_identity_v1(
            reference, copy.deepcopy(reference), stage="decoder_post_deserialize"
        )
        self.assertEqual(receipt["world_size"], 4)
        self.assertNotIn("world8", decoder.canonical_json_bytes(receipt).decode("ascii"))
        with self.assertRaises(decoder.ELAL3C1DecodeError):
            decoder.require_decoder_model_authority_replay_identity_v1(
                reference, reference, stage="post_deserialize"
            )
        hostile = copy.deepcopy(reference)
        hostile["file_count"] = 8
        with self.assertRaises(decoder.ELAL3C1DecodeError):
            decoder.require_decoder_model_authority_replay_identity_v1(
                reference, hostile, stage="decoder_final_pre_publish"
            )

        class FakeDist:
            def __init__(self):
                self.barriers = 0

            def barrier(self, *, group):
                self.barriers += 1

            def broadcast_object_list(self, box, *, src, group):
                self.assertions = (src, group)

            def all_gather_object(self, output, value, *, group):
                output[:] = [value] * 4

        fake = FakeDist()

        def validator(path, **kwargs):
            self.assertEqual(path, authority_path)
            self.assertEqual(kwargs["expected_sha256"], decoder.MODEL_AUTHORITY_SHA256)
            return copy.deepcopy(reference)

        replay = decoder.replay_decoder_model_authority_world4_v1(
            dist=fake,
            group="world4",
            rank=0,
            reference=reference,
            authority_path=authority_path,
            bernini_root=ROOT,
            checkpoint_root=ROOT,
            stage="decoder_final_pre_publish",
            validator=validator,
        )
        self.assertEqual(fake.barriers, 2)
        self.assertTrue(replay["world4_rank_receipt_digest_consensus"])
        self.assertEqual(len(replay["ordered_world4_rank_receipt_digests"]), 4)

    def test_checkpoint_exact23_authority_and_hostile_runtime_inputs(self):
        manifest_path = (
            ROOT
            / "methods/bernini_action_editing/audits/"
            "bernini_r13_ff4c5d4_checkpoint.sha256"
        ).resolve()
        authority = decoder.load_checkpoint_content_manifest_v1(
            manifest_path,
            expected_sha256=decoder.CHECKPOINT_CONTENT_MANIFEST_SHA256,
        )
        self.assertEqual(authority["row_count"], 23)
        names = [row["relative_path"] for row in authority["ordered_rows"]]
        self.assertEqual(
            len([name for name in names if name.startswith("tokenizer/")]), 4
        )
        self.assertEqual(
            len([name for name in names if name.startswith("text_encoder/")]), 7
        )
        self.assertIn("scheduler/scheduler_config.json", names)

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary).resolve()
            payloads = {
                "tokenizer/tokenizer_config.json": b"tokenizer-authority\n",
                "text_encoder/config.json": b"text-encoder-authority\n",
                "scheduler/scheduler_config.json": b"scheduler-authority\n",
            }
            rows = []
            for relative, raw in payloads.items():
                path = checkpoint.joinpath(*Path(relative).parts)
                write_plain(path, raw)
                rows.append({"relative_path": relative, "sha256": sha(raw)})
            baseline = decoder.rehash_checkpoint_content_rows_v1(
                checkpoint_root=checkpoint, rows=rows
            )
            self.assertEqual(baseline["row_count"], 3)
            for relative, raw in payloads.items():
                path = checkpoint.joinpath(*Path(relative).parts)
                path.chmod(0o644)
                path.write_bytes(raw + b"tampered")
                path.chmod(0o444)
                with self.assertRaises(decoder.ELAL3C1DecodeError):
                    decoder.rehash_checkpoint_content_rows_v1(
                        checkpoint_root=checkpoint, rows=rows
                    )
                path.chmod(0o644)
                path.write_bytes(raw)
                path.chmod(0o444)

            outside = checkpoint / "outside"
            outside.mkdir()
            write_plain(outside / "authority.json", b"outside-authority\n")
            (checkpoint / "linked-parent").symlink_to(
                outside, target_is_directory=True
            )
            with self.assertRaises(decoder.ELAL3C1DecodeError):
                decoder.stable_file_digest_v1(
                    checkpoint / "linked-parent/authority.json",
                    label="parent symlink hostile",
                    maximum_bytes=1 << 20,
                )

            target = checkpoint / "tokenizer/tokenizer_config.json"
            replacement = target.with_name("replacement.json")
            displaced = target.with_name("displaced.json")
            original_read = decoder.os.read
            triggered = False

            def replacing_read(descriptor, size):
                nonlocal triggered
                block = original_read(descriptor, size)
                if block and not triggered:
                    triggered = True
                    target.rename(displaced)
                    write_plain(replacement, b"replacement-bytes\n")
                    replacement.rename(target)
                return block

            decoder.os.read = replacing_read
            try:
                with self.assertRaises(decoder.ELAL3C1DecodeError):
                    decoder.stable_file_digest_v1(
                        target,
                        label="hostile replacement",
                        maximum_bytes=1 << 20,
                    )
            finally:
                decoder.os.read = original_read

        source_text = SOURCE.read_text(encoding="utf-8")
        pre_replay = source_text.index('stage="decoder_checkpoint_pre_load"')
        checkpoint_load = source_text.index(
            "checkpoint_root, transformer_config = legacy.validate_checkpoint("
        )
        vae_decode = source_text.index("frames = _vae_decode(vae, latent)")
        final_replay = source_text.index(
            'stage="decoder_checkpoint_final_pre_publish"'
        )
        html_publish = source_text.index('html_path = output_root / "index.html"')
        self.assertLess(pre_replay, checkpoint_load)
        self.assertLess(vae_decode, final_replay)
        self.assertLess(final_replay, html_publish)
        self.assertIn("trust_remote_code=False", source_text)
        self.assertNotIn("trust_remote_code=True", source_text)

    def test_exact40_live_unipc_schedule_and_hostile_config(self):
        sigma_path = (
            ROOT / "methods/bernini_action_editing/inference_sigma_strata.py"
        )
        spec = importlib.util.spec_from_file_location(
            "elal3_decoder_sigma_under_test", sigma_path
        )
        assert spec is not None and spec.loader is not None
        sigma = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = sigma
        spec.loader.exec_module(sigma)

        class Device:
            type = "cpu"

        class Vector:
            def __init__(self, values, dtype):
                self.values = list(values)
                self.dtype = dtype
                self.device = Device()
                self.ndim = 1

            def detach(self):
                return self

            def cpu(self):
                return self

            def tolist(self):
                return list(self.values)

            def __iter__(self):
                return iter(self.values)

            def __getitem__(self, index):
                return self.values[index]

        class Scheduler:
            def __init__(self):
                self.config = {
                    "_class_name": "UniPCMultistepScheduler",
                    "num_train_timesteps": 1000,
                    "flow_shift": 5.0,
                    "prediction_type": "flow_prediction",
                    "predict_x0": True,
                    "use_flow_sigmas": True,
                    "thresholding": False,
                    "solver_order": 2,
                    "solver_type": "bh2",
                    "final_sigmas_type": "zero",
                }

            def set_timesteps(self, steps):
                self.steps = steps
                self.timesteps = Vector(sigma.PINNED_TIMESTEPS, "torch.int64")
                self.sigmas = Vector(
                    (*sigma.PINNED_POSITIVE_SIGMAS, 0.0), "torch.float32"
                )

        scheduler = Scheduler()
        reference = decoder.audit_exact40_unipc_schedule_v1(
            sigma_module=sigma,
            scheduler=scheduler,
            initialize=True,
        )
        self.assertEqual(scheduler.steps, 40)
        self.assertEqual(reference["schedule_sha256"], sigma.SCHEDULE_SHA256)
        self.assertEqual(len(reference["timesteps"]), 40)
        decoder.audit_exact40_unipc_schedule_v1(
            sigma_module=sigma,
            scheduler=scheduler,
            initialize=False,
            reference=reference,
        )
        scheduler.config["flow_shift"] = 4.0
        with self.assertRaises(decoder.ELAL3C1DecodeError):
            decoder.audit_exact40_unipc_schedule_v1(
                sigma_module=sigma,
                scheduler=scheduler,
                initialize=False,
                reference=reference,
            )

    def test_reference_media_provenance_and_final_replay_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            packet = Path(temporary).resolve()
            bindings = {}
            for role in ("source", "target", "anchor"):
                path = packet / "media" / decoder.ROW_ID / f"{role}.mp4"
                raw = (role + "-registered-media").encode("ascii")
                write_plain(path, raw)
                info = path.stat()
                bindings[role] = {
                    "path": str(path),
                    "sha256": sha(raw),
                    "size": len(raw),
                    "mode": 0o444,
                    "device": info.st_dev,
                    "inode": info.st_ino,
                    "nlink": 1,
                }
            retained = decoder.retain_registered_reference_bindings_v1(
                bundle_receipt={"media_bindings": bindings},
                packet_root=packet,
            )
            source = Path(bindings["source"]["path"])
            copied = {
                "source_path": str(source),
                "source_sha256": bindings["source"]["sha256"],
                "sha256": bindings["source"]["sha256"],
                "size": bindings["source"]["size"],
            }
            proof = decoder.verify_registered_reference_copy_v1(
                role="source", source=source, copied=copied, retained=retained
            )
            self.assertTrue(proof["late_copy_matches_early_latent_receipt"])
            hostile = dict(copied)
            hostile["sha256"] = "0" * 64
            with self.assertRaises(decoder.ELAL3C1DecodeError):
                decoder.verify_registered_reference_copy_v1(
                    role="source",
                    source=source,
                    copied=hostile,
                    retained=retained,
                )
            with self.assertRaises(decoder.ELAL3C1DecodeError):
                decoder.verify_registered_reference_copy_v1(
                    role="source",
                    source=Path(bindings["target"]["path"]),
                    copied=copied,
                    retained=retained,
                )

        source_text = SOURCE.read_text(encoding="utf-8")
        vae_load = source_text.index("vae = AutoencoderKLWan.from_pretrained(")
        vae_decode = source_text.index("frames = _vae_decode(vae, latent)")
        final_replay = source_text.index('stage="decoder_final_pre_publish"')
        html_publish = source_text.index('html_path = output_root / "index.html"')
        self.assertLess(vae_load, vae_decode)
        self.assertLess(vae_decode, final_replay)
        self.assertLess(final_replay, html_publish)

    def test_bf16_renderer_fp32_scheduler_fake_run_and_static_scope(self):
        class FakeTensor:
            def __init__(self, dtype):
                self.dtype = dtype

        class FakeTorch:
            Tensor = FakeTensor
            bfloat16 = "bf16"
            float32 = "fp32"
            enabled = False

            @classmethod
            @contextmanager
            def autocast(cls, *, device_type, dtype):
                self.assertEqual(device_type, "cuda")
                self.assertEqual(dtype, cls.bfloat16)
                old = cls.enabled
                cls.enabled = True
                try:
                    yield
                finally:
                    cls.enabled = old

            @classmethod
            def is_autocast_enabled(cls, device_type):
                self.assertEqual(device_type, "cuda")
                return cls.enabled

        class Handle:
            def __init__(self, callbacks, callback):
                self.callbacks = callbacks
                self.callback = callback

            def remove(self):
                self.callbacks.remove(self.callback)

        class Block:
            def __init__(self):
                self.pre = []
                self.post = []

            def register_forward_pre_hook(self, callback):
                self.pre.append(callback)
                return Handle(self.pre, callback)

            def register_forward_hook(self, callback):
                self.post.append(callback)
                return Handle(self.post, callback)

            def run(self):
                value = FakeTensor(FakeTorch.bfloat16)
                for callback in tuple(self.pre):
                    callback(self, (value,))
                for callback in tuple(self.post):
                    callback(self, (value,), value)
                return value

        class Transformer:
            def __init__(self):
                self.blocks = [Block() for _ in range(30)]

        class Scheduler:
            def step(self, model_output, timestep, sample):
                self.assert_not_autocast = not FakeTorch.enabled
                return (FakeTensor(FakeTorch.float32),)

        class Diffusion:
            def __init__(self):
                self.transformer = Transformer()
                self.scheduler = Scheduler()

            def shared_step(self):
                self.assert_autocast = FakeTorch.enabled
                for block in self.transformer.blocks:
                    block.run()
                return FakeTensor(FakeTorch.bfloat16)

        class Renderer:
            def __init__(self):
                self.diff_dec = Diffusion()

        renderer = Renderer()
        with decoder.bf16_renderer_fp32_scheduler_path_v1(
            renderer=renderer,
            branch="fake",
            expected_steps=2,
            torch_module=FakeTorch,
        ) as audit:
            for _ in range(4):
                renderer.diff_dec.shared_step()
            for _ in range(2):
                renderer.diff_dec.scheduler.step(
                    FakeTensor(FakeTorch.bfloat16),
                    1,
                    FakeTensor(FakeTorch.float32),
                )
        numeric = audit.as_dict()
        self.assertEqual(numeric["shared_step_calls"], 4)
        self.assertEqual(numeric["scheduler_step_calls"], 2)
        self.assertEqual(numeric["transformer_block_input_calls"], 120)
        self.assertEqual(numeric["transformer_block_output_calls"], 120)
        self.assertEqual(numeric["expected_transformer_block_calls"], 120)
        self.assertTrue(numeric["scheduler_outside_autocast"])
        source = SOURCE.read_text(encoding="utf-8")
        self.assertNotIn("replay_model_authority_world8_v1(", source)
        self.assertIn("diff_dec.shared_step_only", source)

        hostile = Renderer()
        with self.assertRaises(decoder.ELAL3C1DecodeError):
            with decoder.bf16_renderer_fp32_scheduler_path_v1(
                renderer=hostile,
                branch="hostile",
                expected_steps=1,
                torch_module=FakeTorch,
            ):
                hostile.diff_dec.shared_step()
                hostile.diff_dec.shared_step()
                hostile.diff_dec.scheduler.step(
                    FakeTensor(FakeTorch.bfloat16),
                    1,
                    FakeTensor(FakeTorch.bfloat16),
                )


if __name__ == "__main__":
    unittest.main()
