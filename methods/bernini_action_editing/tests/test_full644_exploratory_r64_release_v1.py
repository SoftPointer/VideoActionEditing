from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import stat
import tarfile
import tempfile
import unittest
from unittest import mock

from methods.bernini_action_editing import full644_exploratory_r64_release_v1 as release


class Full644ExploratoryR64ReleaseTests(unittest.TestCase):
    @staticmethod
    def _repo_root() -> Path:
        return Path(__file__).resolve().parents[3]

    def _build_archive(self, root: Path, name: str = "source.tar") -> Path:
        output = root / name
        built = release.build_source_archive(self._repo_root(), output)
        self.assertEqual(built["sha256"], release.SOURCE_ARCHIVE_SHA256)
        self.assertEqual(built["size"], 1045504)
        return output

    def test_manual_canonical_ustar_is_deterministic_and_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            first = self._build_archive(root, "first.tar")
            second = self._build_archive(root, "second.tar")
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(
                hashlib.sha256(first.read_bytes()).hexdigest(),
                "12a28ddec99704963af42f1a82b09dff31828e3af8e53e5d0bbd0d43db272828",
            )
            payloads = release.audit_source_archive(first)
            self.assertEqual(len(payloads), 14)
            self.assertEqual(
                hashlib.sha256(payloads[release.REVIEW_SNAPSHOT_PATH]).hexdigest(),
                release.REVIEW_SNAPSHOT_SHA256,
            )
            self.assertEqual(
                hashlib.sha256(payloads[release.SOURCE_AUTHORITY_PATH]).hexdigest(),
                release.SOURCE_AUTHORITY_SHA256,
            )

    def test_canonical_ustar_fields_do_not_use_tarfile_writer(self) -> None:
        payloads = {"a.txt": b"a", "z/y.bin": b"xyz"}
        raw = release.canonical_ustar_bytes(payloads)
        self.assertEqual(len(raw) % 512, 0)
        self.assertTrue(raw.endswith(b"\0" * 1024))
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
            self.assertEqual([item.name for item in archive], ["a.txt", "z/y.bin"])
            for item in archive.getmembers():
                self.assertEqual(item.mode, 0o444)
                self.assertEqual(item.uid, 0)
                self.assertEqual(item.gid, 0)
                self.assertEqual(item.mtime, 0)

    def test_archive_extract_reopens_exact_plain_0444_closure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            archive = self._build_archive(root)
            extracted = root / "extracted"
            result = release.extract_source_archive(archive, extracted)
            self.assertEqual(result["file_count"], 14)
            actual = set()
            for path in extracted.rglob("*"):
                if path.is_dir():
                    continue
                actual.add(path.relative_to(extracted).as_posix())
                info = path.lstat()
                self.assertTrue(stat.S_ISREG(info.st_mode))
                self.assertEqual(info.st_nlink, 1)
                self.assertEqual(stat.S_IMODE(info.st_mode), 0o444)
            self.assertEqual(actual, set(release.audit_source_archive(archive)))

    def test_archive_rejects_traversal_extra_and_tampered_core(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            archive = self._build_archive(root)
            payloads = dict(release.audit_source_archive(archive))
            hostile = root / "hostile.tar"
            good_raw = release.canonical_ustar_bytes(payloads)
            hostile_raw = (
                good_raw[:-1024]
                + release._canonical_ustar_header("../escape", 1)
                + b"x"
                + b"\0" * 511
                + b"\0" * 1024
            )
            hostile.write_bytes(hostile_raw)
            hostile.chmod(0o444)
            with self.assertRaisesRegex(release.ReleaseError, "closure/order"):
                release.audit_source_archive(hostile, require_frozen_archive_sha=False)

            trainer = "methods/bernini_action_editing/train_lora.py"
            payloads[trainer] = b"X" + payloads[trainer][1:]
            tampered = root / "tampered.tar"
            tampered.write_bytes(release.canonical_ustar_bytes(payloads))
            tampered.chmod(0o444)
            with self.assertRaisesRegex(release.ReleaseError, "member SHA"):
                release.audit_source_archive(tampered, require_frozen_archive_sha=False)

    def test_archive_eexist_racer_is_never_unlinked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            output = root / "race.tar"

            def lose_race(_source: object, destination: object) -> None:
                Path(destination).write_bytes(b"owned-by-racer")
                raise FileExistsError("lost publication race")

            with mock.patch.object(release.os, "link", side_effect=lose_race):
                with self.assertRaises(FileExistsError):
                    release.build_source_archive(self._repo_root(), output)
            self.assertEqual(output.read_bytes(), b"owned-by-racer")

    def test_runtime_summary_uses_real_raw_keys_not_derived_identity_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            index = root / "dataset_index.jsonl"
            index.write_bytes(b'{"iid":"x"}\n')
            authority = root / "authority.json"
            authority.write_bytes(b'{"authority":true}\n')
            summary_value = {
                "schema_version": "bernini-r-action-vae-dataset-summary-v2",
                "complete": True,
                "preview_only": True,
                "training_authorized": False,
                "training_use_forbidden": True,
                "experimental_training_acknowledged": True,
                "production_claim_forbidden": True,
                "scientific_claim_authorized": False,
                "expected_sample_count": 644,
                "materialized_sample_count": 644,
                "missing_sample_count": 0,
                "index_path": str(index),
                "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
            }
            digest = release.object_sha256(summary_value)
            summary_value["summary_digest"] = digest
            summary = root / "dataset_summary.json"
            summary.write_text(json.dumps(summary_value), encoding="utf-8")
            patches = (
                mock.patch.object(
                    release,
                    "DATASET_SUMMARY_SHA256",
                    hashlib.sha256(summary.read_bytes()).hexdigest(),
                ),
                mock.patch.object(release, "DATASET_SUMMARY_DIGEST", digest),
                mock.patch.object(
                    release,
                    "DATASET_INDEX_SHA256",
                    hashlib.sha256(index.read_bytes()).hexdigest(),
                ),
                mock.patch.object(
                    release,
                    "SOURCE_AUTHORITY_SHA256",
                    hashlib.sha256(authority.read_bytes()).hexdigest(),
                ),
                mock.patch.object(release, "SOURCE_AUTHORITY_SIZE", len(authority.read_bytes())),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                verified = release.verify_runtime_inputs(summary, index, authority)
            self.assertEqual(verified["rows"], 644)
            self.assertNotIn("sha256", summary_value)
            self.assertNotIn("materialized_rows", summary_value)
            self.assertNotIn("reward_selected_synthetic_targets", summary_value)

    @staticmethod
    def _receipt(mode: str, step: int) -> dict:
        objective = "sft" if mode == "capacity-smoke" else "reference_dpo_preservation"
        value = {
            "schema_version": release.TRAIN_RECEIPT_SCHEMA,
            "global_step": step,
            "max_steps": 1 if mode == "capacity-smoke" else 644,
            "last_loss": 1.25,
            "last_preclip_gradient_norm": 2.5,
            "bernini_commit": release.BERNINI_COMMIT,
            "bernini_training_files_index_sha256": (
                release.BERNINI_TRAINING_FILES_INDEX_SHA256
            ),
            "veomni_commit": release.VEOMNI_COMMIT,
            "method_source_revision": release.METHOD_SOURCE_REVISION,
            "method_source_archive_sha256": release.SOURCE_ARCHIVE_SHA256,
            "checkpoint": {"path": "/checkpoint", "configs": {}},
            "checkpoint_tree_sha256": release.CHECKPOINT_TREE_SHA256,
            "dataset": {
                "path": "/dataset/shards",
                "rows": 644,
                "signature": "dataset-signature",
                "content_signature": "dataset-content-signature",
                "summary": {
                    "path": "/dataset/dataset_summary.json",
                    "sha256": release.DATASET_SUMMARY_SHA256,
                    "summary_digest": release.DATASET_SUMMARY_DIGEST,
                    "complete": True,
                    "allow_incomplete": False,
                    "expected_rows": 644,
                    "materialized_rows": 644,
                    "index_path": "/dataset/dataset_index.jsonl",
                    "index_sha256": release.DATASET_INDEX_SHA256,
                    "indexed_shards_sha256": "1" * 64,
                    "dataset_content_signature": "dataset-content-signature",
                    "reward_selected_synthetic_targets": False,
                    "arm": None,
                },
            },
            "training_contract": {
                "model": "Bernini-R-1.3B-Diffusers renderer-only",
                "single_expert": "transformer_1",
                "noise_tmin": 0.0,
                "noise_tmax": 1.0,
                "mv2v_flow_shift": 5.0,
                "num_frames": 81,
                "latent_frames": 21,
                "task_source_name": "mv2v$action_editing_81f",
                "external_spatial_mask": False,
                "external_tracking_or_swept_tube": False,
                "conditioning": ["clean_source_video_vae", "edit_instruction"],
                "supervision": ["noisy_target_video_vae", "target_velocity"],
                "target_embedding_or_caption_conditioning": False,
                "lora_rank": 64,
                "lora_alpha": 64,
                "lora_scope": "all Wan attn1/attn2 q,k,v,out projections",
                "tokenizer_fix_mistral_regex": True,
                "peft_version": release.PEFT_VERSION,
                "transformers_version": "4.test",
                "gradient_checkpointing": True,
                "objective": objective,
                "preference_weight": 1.0,
                "preference_margin": 0.05,
                "preference_temperature": 20.0,
                "dpo_beta": 10.0,
                "preservation_weight": 0.25,
                "contrastive_negative_kinds": ["noop", "reverse", "incomplete"],
                "contrastive_negative_schedule": "rotate",
                "preservation_branch": (
                    None
                    if mode == "capacity-smoke"
                    else "source_as_target_conditional_identity"
                ),
            },
            "optimizer": {
                "type": "AdamW",
                "learning_rate": 1.0e-4,
                "weight_decay": 0.0,
                "max_gradient_norm": 1.0,
            },
            "distributed": {
                "world_size": 4,
                "ulysses_size": 4,
                "backend": "nccl/rccl",
                "same_sample_all_ranks": True,
                "same_seed_all_ranks": True,
                "lora_initialization_seeded_all_ranks": True,
                "lora_parameters_broadcast_from_rank": 0,
                "lora_initialization_digest": "2" * 64,
                "explicit_lora_gradient_all_reduce": True,
            },
            "seed": 20260817,
            "target_module_count": 240,
            "target_modules_sha256": "3" * 64,
            "trainable_parameter_count": 47_185_920,
            "resumed_from": None,
            "experimental_training": True,
            "production_claim_forbidden": True,
            "scientific_claim_authorized": False,
        }
        if mode == "full644":
            terminal = step == 644
            value["exploratory_full644"] = {
                "profile": release.FULL644_PROFILE,
                "optimizer_rows_consumed": step,
                "next_row_index": None if terminal else step,
                "row_sequence_prefix": "0..%d" % (step - 1),
                "row_sequence_sha256": release.object_sha256(list(range(step))),
                "no_replacement_within_pass": True,
                "complete_one_pass": terminal,
                "intermediate_checkpoints_archival_only": True,
                "interrupted_run_requires_fresh_step0_restart": True,
                "resume_policy": "forbidden_for_this_profile",
                "dataset_quality_accepted_under_0817": False,
                "formal_training_dataset_authorized": False,
                "formal_heldout_contribution": 0,
                "dataset_summary_sha256": release.DATASET_SUMMARY_SHA256,
                "dataset_summary_digest": release.DATASET_SUMMARY_DIGEST,
                "dataset_index_sha256": release.DATASET_INDEX_SHA256,
                "indexed_source_and_target_vae_shards_reverified_after_training": terminal,
                "source_authority": {"sha256": release.SOURCE_AUTHORITY_SHA256},
            }
        value["receipt_digest"] = release.object_sha256(value)
        return value

    @staticmethod
    def _write_json(path: Path, value: dict) -> bytes:
        raw = release.canonical_json_bytes(value) + b"\n"
        path.write_bytes(raw)
        return raw

    def _checkpoint(self, output: Path, mode: str, step: int) -> dict:
        checkpoint = output / ("checkpoint-%08d" % step)
        adapter = checkpoint / "adapter"
        adapter.mkdir(parents=True)
        (adapter / "adapter_config.json").write_bytes(b"{}\n")
        (adapter / "adapter_model.safetensors").write_bytes(b"frozen-adapter-state")
        (checkpoint / "optimizer.pt").write_bytes(b"real-optimizer-state")
        receipt = self._receipt(mode, step)
        receipt_raw = self._write_json(checkpoint / "receipt.json", receipt)
        entries = []
        for path in sorted(checkpoint.rglob("*")):
            if path.is_file():
                raw = path.read_bytes()
                entries.append(
                    {
                        "path": path.relative_to(checkpoint).as_posix(),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "size": len(raw),
                    }
                )
        manifest = {
            "schema_version": release.CHECKPOINT_MANIFEST_SCHEMA,
            "global_step": step,
            "receipt_digest": receipt["receipt_digest"],
            "file_count": len(entries),
            "entries": entries,
        }
        manifest["manifest_digest"] = release.object_sha256(manifest)
        manifest_raw = self._write_json(checkpoint / "checkpoint_manifest.json", manifest)
        return {
            "checkpoint": checkpoint,
            "receipt_raw": receipt_raw,
            "manifest_raw": manifest_raw,
        }

    def _output(self, root: Path, mode: str) -> Path:
        output = root / ("smoke" if mode == "capacity-smoke" else "full")
        output.mkdir()
        steps = [1] if mode == "capacity-smoke" else list(range(64, 641, 64)) + [644]
        terminal = None
        for step in steps:
            terminal = self._checkpoint(output, mode, step)
        assert terminal is not None
        latest = {
            "checkpoint": str(terminal["checkpoint"]),
            "global_step": steps[-1],
            "checkpoint_manifest_path": str(
                terminal["checkpoint"] / "checkpoint_manifest.json"
            ),
            "checkpoint_manifest_sha256": hashlib.sha256(
                terminal["manifest_raw"]
            ).hexdigest(),
            "checkpoint_receipt_sha256": hashlib.sha256(
                terminal["receipt_raw"]
            ).hexdigest(),
        }
        self._write_json(output / "latest.json", latest)
        return output

    def _reseal_capacity_terminal_receipt(self, output: Path, receipt: dict) -> None:
        checkpoint = output / "checkpoint-00000001"
        receipt.pop("receipt_digest", None)
        receipt["receipt_digest"] = release.object_sha256(receipt)
        receipt_raw = self._write_json(checkpoint / "receipt.json", receipt)
        entries = []
        for path in sorted(checkpoint.rglob("*")):
            if path.is_file() and path.name != "checkpoint_manifest.json":
                raw = path.read_bytes()
                entries.append(
                    {
                        "path": path.relative_to(checkpoint).as_posix(),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "size": len(raw),
                    }
                )
        manifest = {
            "schema_version": release.CHECKPOINT_MANIFEST_SCHEMA,
            "global_step": 1,
            "receipt_digest": receipt["receipt_digest"],
            "file_count": len(entries),
            "entries": entries,
        }
        manifest["manifest_digest"] = release.object_sha256(manifest)
        manifest_raw = self._write_json(
            checkpoint / "checkpoint_manifest.json", manifest
        )
        latest = {
            "checkpoint": str(checkpoint),
            "global_step": 1,
            "checkpoint_manifest_path": str(checkpoint / "checkpoint_manifest.json"),
            "checkpoint_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
            "checkpoint_receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
        }
        self._write_json(output / "latest.json", latest)

    def _cache_receipt(self, root: Path, mode: str, step: str) -> Path:
        uid = 501
        cache_root = "/tmp/cache/full644-r64-u%d-j141620-s%s-v1" % (uid, step)
        value = {
            "schema_version": "full644-r64-rank-cache-receipt-v1",
            "mode": (
                "PRELAUNCH_CAPACITY_ONLY"
                if mode == "capacity-smoke"
                else "FULL644_EXPLORATORY"
            ),
            "job_id": "141620",
            "step_id": step,
            "node": "auh7-1b-gpu-226",
            "filesystem_type": "tmpfs",
            "cache_root": cache_root,
            "cache_root_device": 7,
            "cache_root_inode": 99,
            "cache_root_uid": uid,
            "cache_root_mode": "0700",
            "rank_caches": [
                {
                    "rank": rank,
                    "path": "%s/rank-%d" % (cache_root, rank),
                    "device": 7,
                    "inode": 100 + rank,
                    "uid": uid,
                    "mode": "0700",
                }
                for rank in range(4)
            ],
            "world_size": 4,
            "rank_local": True,
            "scheduler_tmpdir_observed": "/tmp",
            "scheduler_tmpdir_normalized_to_unset": True,
        }
        value["receipt_digest"] = release.object_sha256(value)
        path = root / ("%s-%s-cache.json" % (mode, step))
        self._write_json(path, value)
        path.chmod(0o400)
        return path

    def test_capacity_and_full_outputs_require_exact_checkpoint_closure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            smoke = self._output(root, "capacity-smoke")
            full = self._output(root, "full644")
            self.assertTrue(
                release.verify_training_output("capacity-smoke", smoke)[
                    "optimizer_update_proven"
                ]
            )
            self.assertEqual(
                release.verify_training_output("full644", full)["global_step"], 644
            )

            (smoke / "unmanifested-root.txt").write_bytes(b"x")
            with self.assertRaisesRegex(release.ReleaseError, "extra root member"):
                release.verify_training_output("capacity-smoke", smoke)

    def test_checkpoint_rejects_unmanifested_recursive_member_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            smoke = self._output(root, "capacity-smoke")
            checkpoint = smoke / "checkpoint-00000001"
            (checkpoint / "extra.bin").write_bytes(b"x")
            with self.assertRaisesRegex(release.ReleaseError, "does not close"):
                release.verify_training_output("capacity-smoke", smoke)
            (checkpoint / "extra.bin").unlink()
            os.symlink("optimizer.pt", checkpoint / "extra-link")
            with self.assertRaisesRegex(release.ReleaseError, "symlink"):
                release.verify_training_output("capacity-smoke", smoke)

    def test_checkpoint_rejects_incomplete_and_nfs_residue_files_or_directories(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            for index, (name, kind) in enumerate(
                (
                    (".INCOMPLETE", "file"),
                    (".INCOMPLETE", "directory"),
                    (".nfs-test", "file"),
                    (".nfs-test", "directory"),
                )
            ):
                case_root = root / ("case-%d" % index)
                case_root.mkdir()
                smoke = self._output(case_root, "capacity-smoke")
                hostile = smoke / "checkpoint-00000001" / name
                if kind == "file":
                    hostile.write_bytes(b"residue")
                else:
                    hostile.mkdir()
                with self.subTest(name=name, kind=kind):
                    with self.assertRaisesRegex(
                        release.ReleaseError, "does not close actual membership"
                    ):
                        release.verify_training_output("capacity-smoke", smoke)

    def test_smoke_cannot_claim_full644_or_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            smoke = self._output(root, "capacity-smoke")
            receipt_path = smoke / "checkpoint-00000001/receipt.json"
            receipt = json.loads(receipt_path.read_text())
            receipt["resumed_from"] = "/old/checkpoint"
            receipt.pop("receipt_digest")
            receipt["receipt_digest"] = release.object_sha256(receipt)
            self._write_json(receipt_path, receipt)
            with self.assertRaises(release.ReleaseError):
                release.verify_training_output("capacity-smoke", smoke)

    def test_terminal_numeric_evidence_rejects_json_booleans(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            for field in ("last_loss", "last_preclip_gradient_norm"):
                case_root = root / field
                case_root.mkdir()
                smoke = self._output(case_root, "capacity-smoke")
                receipt_path = smoke / "checkpoint-00000001/receipt.json"
                receipt = json.loads(receipt_path.read_text())
                receipt[field] = True
                self._reseal_capacity_terminal_receipt(smoke, receipt)
                with self.assertRaises(release.ReleaseError):
                    release.verify_training_output("capacity-smoke", smoke)

    def test_capacity_completion_seal_and_immediate_reaudit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            smoke = self._output(root, "capacity-smoke")
            cache = self._cache_receipt(root, "capacity-smoke", "42")
            completion_path = root / "capacity-runner-completion.json"
            published = release.publish_runner_completion(
                "capacity-smoke",
                smoke,
                completion_path,
                cache,
                slurm_job_id="141620",
                slurm_step_id="42",
                node="auh7-1b-gpu-226",
            )
            self.assertEqual(stat.S_IMODE(completion_path.stat().st_mode), 0o400)
            self.assertEqual(completion_path.stat().st_nlink, 1)
            self.assertEqual(
                published["receipt"]["status"], "PRELAUNCH_CAPACITY_ONLY_COMPLETE"
            )
            self.assertFalse(
                published["receipt"]["this_receipt_authorizes_full644_training_result"]
            )
            self.assertFalse(published["receipt"]["scientific_claim_authorized"])
            self.assertFalse(published["receipt"]["formal_claim_authorized"])
            completion_audit = release.audit_runner_completion(
                "capacity-smoke",
                smoke,
                completion_path,
                cache,
                slurm_job_id="141620",
                slurm_step_id="42",
                node="auh7-1b-gpu-226",
            )
            self.assertEqual(
                completion_audit["status"], "PRELAUNCH_CAPACITY_ONLY_COMPLETE"
            )
            sacct = root / "sacct.psv"
            sacct.write_text("141620.42|COMPLETED|0:0|55G|\n", encoding="ascii")
            gate = root / "capacity-gate.json"
            sealed = release.seal_capacity_gate(
                smoke, completion_path, sacct, "42", gate
            )
            self.assertEqual(
                sealed["gate"]["scope"], "PRELAUNCH_CAPACITY_ONLY"
            )
            self.assertFalse(sealed["gate"]["formal_or_scientific_authority"])
            audited = release.audit_capacity_gate(
                smoke, completion_path, sacct, "42", gate
            )
            self.assertEqual(audited["status"], "PASS_FULL644_MAY_START_FRESH")
            self.assertEqual(audited["max_rss_bytes"], 55 * 1024**3)

            gate_value = json.loads(gate.read_text())
            gate_value["max_rss_bytes"] -= 1
            gate_value.pop("receipt_digest")
            gate_value["receipt_digest"] = release.object_sha256(gate_value)
            gate.chmod(0o600)
            self._write_json(gate, gate_value)
            gate.chmod(0o400)
            with self.assertRaisesRegex(release.ReleaseError, "bytes/bindings"):
                release.audit_capacity_gate(
                    smoke, completion_path, sacct, "42", gate
                )

    def test_rank_cache_receipt_rejects_mode_step_world_and_path_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            for index, (field, value) in enumerate(
                (
                    ("mode", "FULL644_EXPLORATORY"),
                    ("step_id", "999"),
                    ("world_size", 8),
                    ("cache_root", "/tmp/cache/wrong"),
                )
            ):
                receipt = self._cache_receipt(root, "capacity-smoke", str(20 + index))
                payload = json.loads(receipt.read_text())
                payload[field] = value
                payload.pop("receipt_digest")
                payload["receipt_digest"] = release.object_sha256(payload)
                receipt.chmod(0o600)
                self._write_json(receipt, payload)
                receipt.chmod(0o400)
                with self.assertRaises(release.ReleaseError):
                    release.verify_rank_cache_receipt(
                        receipt,
                        mode="capacity-smoke",
                        slurm_job_id="141620",
                        slurm_step_id=str(20 + index),
                        node="auh7-1b-gpu-226",
                    )

    def test_rank_cache_receipt_rejects_scheduler_tmpdir_hostiles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            cases = (
                ("wrong", "scheduler_tmpdir_observed", "relative/tmp"),
                ("arbitrary", "scheduler_tmpdir_observed", "/private/arbitrary"),
                ("false", "scheduler_tmpdir_normalized_to_unset", False),
                ("missing-observed", "scheduler_tmpdir_observed", None),
                ("missing-normalized", "scheduler_tmpdir_normalized_to_unset", None),
            )
            for index, (label, field, value) in enumerate(cases):
                step = str(70 + index)
                receipt = self._cache_receipt(root, "capacity-smoke", step)
                payload = json.loads(receipt.read_text())
                if label.startswith("missing"):
                    del payload[field]
                else:
                    payload[field] = value
                payload.pop("receipt_digest")
                payload["receipt_digest"] = release.object_sha256(payload)
                receipt.chmod(0o600)
                self._write_json(receipt, payload)
                receipt.chmod(0o400)
                with self.assertRaisesRegex(
                    release.ReleaseError, "rank-cache receipt contract"
                ):
                    release.verify_rank_cache_receipt(
                        receipt,
                        mode="capacity-smoke",
                        slurm_job_id="141620",
                        slurm_step_id=step,
                        node="auh7-1b-gpu-226",
                    )

    def test_full_runner_completion_can_be_replayed_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            full = self._output(root, "full644")
            cache = self._cache_receipt(root, "full644", "88")
            receipt = root / "full-completion.json"
            release.publish_runner_completion(
                "full644",
                full,
                receipt,
                cache,
                slurm_job_id="141620",
                slurm_step_id="88",
                node="auh7-1b-gpu-226",
            )
            audited = release.audit_runner_completion(
                "full644",
                full,
                receipt,
                cache,
                slurm_job_id="141620",
                slurm_step_id="88",
                node="auh7-1b-gpu-226",
            )
            self.assertEqual(
                audited["status"], "EXPOSED_FULL644_EXPLORATORY_ABLATION_COMPLETE"
            )

    def test_capacity_gate_rejects_oversize_failed_or_tampered_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            smoke = self._output(root, "capacity-smoke")
            cache = self._cache_receipt(root, "capacity-smoke", "7")
            completion = root / "completion.json"
            release.publish_runner_completion(
                "capacity-smoke",
                smoke,
                completion,
                cache,
                slurm_job_id="141620",
                slurm_step_id="7",
                node="auh7-1b-gpu-226",
            )
            for name, row, pattern in (
                ("large", "141620.7|COMPLETED|0:0|56.1G\n", "headroom gate"),
                ("failed", "141620.7|FAILED|1:0|40G\n", "terminate successfully"),
                ("wrong", "141620.8|COMPLETED|0:0|40G\n", "exact-row closure"),
            ):
                sacct = root / (name + ".psv")
                sacct.write_text(row, encoding="ascii")
                with self.assertRaisesRegex(release.ReleaseError, pattern):
                    release.seal_capacity_gate(
                        smoke, completion, sacct, "7", root / (name + ".json")
                    )

    def test_maxrss_parser_uses_slurm_binary_units(self) -> None:
        self.assertEqual(release.parse_max_rss("1024K"), 1024**2)
        self.assertEqual(release.parse_max_rss("55G"), 55 * 1024**3)
        self.assertEqual(release.parse_max_rss("55.5G"), int(55.5 * 1024**3))
        self.assertEqual(release.CAPACITY_MAX_RSS_BYTES, 56 * 1024**3)
        with self.assertRaises(release.ReleaseError):
            release.parse_max_rss("")

    def test_publication_refuses_overwrite_and_preserves_existing_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            path = root / "receipt.json"
            path.write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                release._atomic_create_json(path, {"x": 1})
            self.assertEqual(path.read_bytes(), b"existing")


if __name__ == "__main__":
    unittest.main()
