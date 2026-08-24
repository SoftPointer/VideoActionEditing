from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import hashlib
import inspect
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))
TOOLS_ROOT = METHOD_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import graft_a_lite_source_release_consumer_v1 as consumer  # noqa: E402
import materialize_graft_a_lite_terminal_admission_v1 as terminal_materializer  # noqa: E402


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _object_sha(value: object) -> str:
    return _sha(consumer.canonical_json_bytes(value))


def _seal(value: dict[str, object]) -> dict[str, object]:
    result = deepcopy(value)
    result["receipt_digest"] = _object_sha(result)
    return result


def _canonical_line(value: object) -> bytes:
    return consumer.canonical_json_bytes(value) + b"\n"


def _false_authority() -> dict[str, bool]:
    return {
        "action_authority": False,
        "identity_authority": False,
        "cross_clip_identity_authority": False,
        "quality_authority": False,
        "training_authority": False,
        "production_authority": False,
        "data_governance_authority": False,
        "data_license_authority": False,
        "scientific_success_claimed": False,
    }


class _Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.sources = self.root / "sources"
        self.release = self.root / "release"
        self.sources.mkdir()
        self.release.mkdir()
        self.stem = self.release / "job132549-canary4"
        self.manifest_path = self.stem.with_name(
            f"{self.stem.name}{consumer.MANIFEST_SUFFIX}"
        )
        self.producer_path = self.stem.with_name(
            f"{self.stem.name}{consumer.PRODUCER_SUFFIX}"
        )
        self.execution_path = self.stem.with_name(
            f"{self.stem.name}{consumer.EXECUTION_SUFFIX}"
        )
        self.submission_path = self.stem.with_name(
            f"{self.stem.name}{consumer.SUBMISSION_SUFFIX}"
        )
        self.rows: list[dict[str, object]] = []
        self.source_dimensions: dict[str, tuple[int, int]] = {}
        for index, (iid, split, update, confirmation) in enumerate(
            consumer.CANARY4
        ):
            source = self.sources / f"{iid}.mp4"
            raw = (f"owned-source-{iid}\n" * (index + 1)).encode("ascii")
            source.write_bytes(raw)
            source.chmod(0o444)
            observed = source.stat()
            width, height = 704, 896
            self.source_dimensions[iid] = (width, height)
            core: dict[str, object] = {
                "schema_version": consumer.ROW_SCHEMA,
                "release_mode": "canary4",
                "row_index": index,
                "iid": iid,
                "split": split,
                "split_assignment": "preregistered_core4:test",
                "optimizer_update_authorized": update,
                "optimizer_confirmation_only": confirmation,
                "prior_research_exposure": True,
                "global_holdout": False,
                "stable_identity_disjoint_split_claimed": False,
                "source_cohort": "goku_fullmotion_v17_next1000",
                "source_video_path": str(source),
                "source_video_sha256": _sha(raw),
                "source_file_size_bytes": len(raw),
                "source_mtime_ns": observed.st_mtime_ns,
                "source_ctime_ns_observed": observed.st_ctime_ns,
                "source_hash_and_probe_same_open_fd": True,
                "source_sha256_recomputed_before_and_after_probe": True,
                "source_pre_post_probe_sha256_matched": True,
                "source_identity_includes_ctime_ns": True,
                "source_path_inode_binding_revalidated": True,
                "source_media": {
                    "frame_count": 81,
                    "fps": 25.0,
                    "fps_fraction": "25/1",
                    "reported_fps_fraction": "25/1",
                    "width": width,
                    "height": height,
                    "resolution_hw": [height, width],
                    "short_side": 704,
                    "probe_contract_matched": True,
                    "source_fd_transport": "linux_proc_self_fd",
                    "fresh_ffprobe_verified": True,
                    "test_only_probe_contract_matched": False,
                },
                "noop_instruction": consumer.NOOP_INSTRUCTION,
                "same_clip_noop_only": True,
                "source_video_is_clean_noop_endpoint": True,
                "cross_clip_identity_authority": False,
                "action_authority": False,
                "quality_authority": False,
                "production_authority": False,
                "publication_eligible": True,
                "upstream_candidate": {
                    "cohort": "goku_fullmotion_v17_next1000",
                    "manifest_path": str(self.root / "v17.jsonl"),
                    "manifest_sha256": consumer.V17_MANIFEST_SHA256,
                    "line_number": index + 1,
                    "row_bytes_sha256": f"{index + 2:x}" * 64,
                    "row_canonical_sha256": f"{index + 6:x}" * 64,
                    "row_schema_version": "motive-goku-action-anchor-prefilter-v1",
                },
            }
            # Keep fixture SHA strings exactly 64 characters.
            upstream = core["upstream_candidate"]
            assert isinstance(upstream, dict)
            upstream["row_bytes_sha256"] = str(upstream["row_bytes_sha256"])[:64]
            upstream["row_canonical_sha256"] = str(
                upstream["row_canonical_sha256"]
            )[:64]
            self.rows.append({**core, "row_digest": _object_sha(core)})
        self.terminal_state = "COMPLETED"
        self.terminal_exit_code = "0:0"
        self.terminal_job_success = True
        self.materializer_independent = True
        self.job_id = consumer.EXPECTED_JOB_ID
        self.execution_artifact_device_override: object | None = None
        self.execution_source_device_override: object | None = None
        self.execution_artifact_inode_override: object | None = None
        self.execution_source_inode_override: object | None = None
        self.write_release()

    def _write_readonly(self, path: Path, raw: bytes) -> None:
        if path.exists() or path.is_symlink():
            path.chmod(0o644)
            path.unlink()
        path.write_bytes(raw)
        path.chmod(0o444)

    def write_release(self) -> None:
        # Recompute row seals after callers mutate semantic fields.
        sealed_rows: list[dict[str, object]] = []
        for row in self.rows:
            unsigned = deepcopy(row)
            unsigned.pop("row_digest", None)
            sealed_rows.append({**unsigned, "row_digest": _object_sha(unsigned)})
        self.rows = sealed_rows
        manifest_raw = b"".join(_canonical_line(row) for row in self.rows)
        self._write_readonly(self.manifest_path, manifest_raw)

        train_iids = [row["iid"] for row in self.rows if row["split"] == "optimizer_train"]
        confirmation_iids = [
            row["iid"]
            for row in self.rows
            if row["split"] == "optimizer_confirmation"
        ]
        probe = {
            "pin_label": consumer.PORTABLE_FFPROBE_PIN_LABEL,
            "configured_path": consumer.PORTABLE_FFPROBE_PATH,
            "resolved_path": consumer.PORTABLE_FFPROBE_PATH,
            "exact_realpath_matched": True,
            "path_lookup_used": False,
            "file_sha256_expected": consumer.PORTABLE_FFPROBE_SHA256,
            "file_sha256_observed": consumer.PORTABLE_FFPROBE_SHA256,
            "file_sha256_matched": True,
            "version_stdout_sha256_expected": (
                consumer.PORTABLE_FFPROBE_VERSION_STDOUT_SHA256
            ),
            "version_stdout_sha256_observed": (
                consumer.PORTABLE_FFPROBE_VERSION_STDOUT_SHA256
            ),
            "version_stdout_sha256_matched": True,
            "version_first_line_expected": (
                consumer.PORTABLE_FFPROBE_VERSION_FIRST_LINE
            ),
            "version_first_line_observed": (
                consumer.PORTABLE_FFPROBE_VERSION_FIRST_LINE
            ),
            "version_first_line_matched": True,
            "pre_and_post_version_identity_and_file_sha_revalidated": True,
            "caller_process_observation_only": True,
            "trusted_or_official_authority_claimed": False,
            "executable_transport": "linux_proc_self_fd",
            "executable_fixed_inode_execution": True,
            "absolute_path_fallback_pre_post_inode_sha": False,
        }
        producer_inputs = [
            {
                "cohort": consumer.V16_COHORT,
                "path": str(self.root / "v16.jsonl"),
                "rows": consumer.V16_ROWS,
                "file_sha256": consumer.V16_MANIFEST_SHA256,
                "row_binding_digest": "8" * 64,
            },
            {
                "cohort": consumer.V17_COHORT,
                "path": str(self.root / "v17.jsonl"),
                "rows": consumer.V17_ROWS,
                "file_sha256": consumer.V17_MANIFEST_SHA256,
                "row_binding_digest": "9" * 64,
            },
        ]
        producer_core: dict[str, object] = {
            "schema_version": consumer.PRODUCER_RECEIPT_SCHEMA,
            "status": "complete",
            "release_id": "graft-a-lite-canary4-fixture",
            "release_mode": "canary4",
            "semantics": {
                "source_only": True,
                "same_clip_noop_only": True,
                "source_video_is_clean_noop_endpoint": True,
                "cross_clip_identity_authority": False,
                "action_authority": False,
                "quality_authority": False,
                "production_authority": False,
                "scientific_success_claimed": False,
                "canonical_noop_instruction": consumer.NOOP_INSTRUCTION,
            },
            "input_policy": {
                "external_target_artifacts_opened": False,
                "wan_preview_opened": False,
                "generated_target_opened": False,
                "legacy_latent_or_receipt_opened": False,
                "anchor_image_opened": False,
                "code_frozen_v16_v17_manifest_pins_required": True,
                "code_frozen_v16_v17_manifest_pins_matched": True,
                "custom_manifest_test_path": False,
                "custom_manifest_path_publication_eligible": False,
            },
            "inputs": producer_inputs,
            "input_binding_digest": _object_sha(producer_inputs),
            "selection": {
                "combined_candidate_rows": 1128,
                "selected_rows": 4,
                "selected_iid_digest": _object_sha(
                    [expected[0] for expected in consumer.CANARY4]
                ),
                "v16_v17_iid_unique": True,
                "v16_v17_source_path_unique": True,
                "v16_v17_source_sha256_unique": True,
                "order": "preregistered_core4_fit_then_confirmation",
            },
            "split": {
                "optimizer_train_rows": 2,
                "optimizer_confirmation_rows": 2,
                "optimizer_train_iid_digest": _object_sha(train_iids),
                "optimizer_confirmation_iid_digest": _object_sha(
                    confirmation_iids
                ),
                "iid_sets_disjoint": True,
                "optimizer_confirmation_update_intended": False,
                "optimizer_confirmation_update_authorized": False,
                "optimizer_confirmation_actual_use_claimed": False,
                "global_holdout": False,
            },
            "media_contract": {
                "source_sha256_verified_rows": 4,
                "source_open_once_rows": 4,
                "hash_and_probe_same_open_fd": True,
                "source_sha256_recomputed_before_and_after_probe": True,
                "source_pre_post_probe_sha256_matched_rows": 4,
                "probe_kind": consumer.PORTABLE_FFPROBE_PROBE_KIND,
                "fresh_ffprobe": True,
                "fresh_ffprobe_verified_rows": 4,
                "frame_count": 81,
                "fps_fraction": "25/1",
                "short_side": 704,
                "temporal_padding_allowed": False,
                "temporal_truncation_allowed": False,
                "retiming_allowed": False,
            },
            "training_consumer_requirements": {
                "must_revalidate_source_video_sha256": True,
                "must_fresh_probe_frame_count_fps_and_resolution": True,
                "must_hash_and_probe_same_open_source_fd": True,
                "must_recompute_source_sha256_before_and_after_probe": True,
                "must_bind_source_identity_including_ctime_ns": True,
                "must_revalidate_source_path_inode_and_parent_binding": True,
                "must_verify_preregistered_ffprobe_pin_in_sealed_runtime": True,
                "must_supply_independent_training_execution_receipt": True,
                "must_record_revalidation_in_training_execution_receipt": True,
                "must_reject_optimizer_confirmation_rows_for_updates": True,
                "actual_split_use_must_be_recorded_in_training_execution_receipt": True,
            },
            "implementation": {
                "path": str(self.root / "builder.py"),
                "sha256": "a" * 64,
                "media_probe_kind": consumer.PORTABLE_FFPROBE_PROBE_KIND,
                "ffprobe_executable_observation": probe,
                "formal_runtime_authority_claimed": False,
                "independent_execution_receipt_verified_by_this_receipt": False,
            },
            "artifact": {
                "manifest_suffix": consumer.MANIFEST_SUFFIX,
                "receipt_suffix": consumer.PRODUCER_SUFFIX,
                "manifest_rows": 4,
                "manifest_bytes": len(manifest_raw),
                "manifest_sha256": _sha(manifest_raw),
                "row_digest_sequence_sha256": _object_sha(
                    [row["row_digest"] for row in self.rows]
                ),
            },
        }
        producer = _seal(producer_core)
        producer_raw = _canonical_line(producer)
        self._write_readonly(self.producer_path, producer_raw)

        selected_sources = []
        for row in self.rows:
            source = Path(str(row["source_video_path"]))
            metadata = source.stat()
            selected_sources.append(
                {
                    "iid": row["iid"],
                    "path": str(source),
                    "resolved_path": str(source),
                    "sha256": row["source_video_sha256"],
                    "size_bytes": row["source_file_size_bytes"],
                    "mode": format(stat.S_IMODE(metadata.st_mode), "04o"),
                    "device": (
                        metadata.st_dev
                        if self.execution_source_device_override is None
                        else self.execution_source_device_override
                    ),
                    "inode": (
                        metadata.st_ino
                        if self.execution_source_inode_override is None
                        else self.execution_source_inode_override
                    ),
                }
            )
        output_record = lambda path, raw: {
            "leaf_name": path.name,
            "access": "retained_output_parent_fd_openat",
            "sha256": _sha(raw),
            "size_bytes": len(raw),
            "mode": "0444",
            "device": (
                path.stat().st_dev
                if self.execution_artifact_device_override is None
                else self.execution_artifact_device_override
            ),
            "inode": (
                path.stat().st_ino
                if self.execution_artifact_inode_override is None
                else self.execution_artifact_inode_override
            ),
        }
        execution_core: dict[str, object] = {
            "schema_version": consumer.EXECUTION_RECEIPT_SCHEMA,
            "status": "complete",
            "successful_return": True,
            "builder_successful_return": True,
            "builder_publication_reopened_and_verified": True,
            "python_runtime_closure_verified": False,
            "formal_runtime_authority": False,
            "observations_are_not_an_adversarial_runtime_boundary": True,
            "slurm": {
                "job_id": self.job_id,
                "job_name": "graft-a-lite-c4",
                "node_list": "fixture-node",
                "cluster_name": "fixture-cluster",
                "cpu_only_workload": True,
                "gpu_resource_requested_by_launcher": True,
                "gpu_resource_request": "gpu:mi210:1",
                "effective_submission_request_verified": False,
                "gpu_computation_used": False,
            },
            "runtime_observations": {
                "builder": {
                    "git_commit_observed_pin": "1" * 40,
                    "git_blob_sha1_observed_and_matched": "2" * 40,
                    "sha256_observed_and_matched": "a" * 64,
                    "archive_member": "build_graft_a_lite_source_release_v1.py",
                    "archive_exactly_one_plain_member": True,
                    "compiled_from_exact_in_memory_archive_member_bytes": True,
                    "executed_or_imported_from_builder_path": False,
                    "live_repository_imported": False,
                },
                "builder_ffprobe_expected_contract": {
                    "media_probe_kind": consumer.PORTABLE_FFPROBE_PROBE_KIND,
                    "pin_label": consumer.PORTABLE_FFPROBE_PIN_LABEL,
                    "configured_and_resolved_path": consumer.PORTABLE_FFPROBE_PATH,
                    "file_sha256": consumer.PORTABLE_FFPROBE_SHA256,
                    "version_stdout_sha256": (
                        consumer.PORTABLE_FFPROBE_VERSION_STDOUT_SHA256
                    ),
                    "version_first_line": (
                        consumer.PORTABLE_FFPROBE_VERSION_FIRST_LINE
                    ),
                    "shared_portable_compute_verified_label_is_provenance_not_runtime_authority": True,
                }
            },
            "inputs": {
                "v16_candidates": {
                    "path": str(self.root / "v16.jsonl"),
                    "resolved_path": str(self.root / "v16.jsonl"),
                    "sha256": consumer.V16_MANIFEST_SHA256,
                    "size_bytes": 128,
                    "mode": "0444",
                    "device": 1,
                    "inode": 2,
                },
                "v17_candidates": {
                    "path": str(self.root / "v17.jsonl"),
                    "resolved_path": str(self.root / "v17.jsonl"),
                    "sha256": consumer.V17_MANIFEST_SHA256,
                    "size_bytes": 1000,
                    "mode": "0444",
                    "device": 1,
                    "inode": 3,
                },
                "selected_source_videos": selected_sources,
                "all_selected_source_sha256_recomputed_after_publication": True,
                "target_video_opened": False,
                "wan_preview_opened": False,
                "anchor_image_opened": False,
                "legacy_latent_or_receipt_opened": False,
            },
            "outputs": {
                "logical_output_stem": str(self.stem),
                "manifest": output_record(self.manifest_path, manifest_raw),
                "producer_receipt": output_record(
                    self.producer_path, producer_raw
                ),
                "manifest_rows": 4,
                "release_mode": "canary4",
                "producer_receipt_digest": producer["receipt_digest"],
                "canonical_json_and_digests_verified": True,
            },
            "split_execution": {
                "optimizer_train_rows": 2,
                "optimizer_confirmation_rows": 2,
                "optimizer_update_performed": False,
                "optimizer_confirmation_update_performed": False,
                "optimizer_confirmation_update_authorized": False,
                "confirmation_rows_prior_research_exposure": True,
                "global_holdout_claimed": False,
            },
            "authority": _false_authority(),
            "failure_semantics": {
                "create_only": True,
                "automatic_cleanup_of_publication": False,
                "automatic_cleanup_of_builder_backing": False,
                "external_execution_receipt_is_last_commit_marker": True,
                "receipt_alone_proves_successful_process_return": False,
                "consumer_must_also_require_slurm_completed_exit_zero": True,
                "partial_builder_publication_is_preserved_but_not_success": True,
                "consumer_must_require_this_valid_execution_receipt": True,
            },
        }
        execution = _seal(execution_core)
        execution_raw = _canonical_line(execution)
        self._write_readonly(self.execution_path, execution_raw)

        stem_bytes = str(self.stem).encode("utf-8")
        submission_core: dict[str, object] = {
            "schema_version": consumer.SUBMISSION_RECEIPT_SCHEMA,
            "status": "submitted",
            "submission_success": True,
            "job_success": None,
            "job_terminal_state_observed": False,
            "effective_submission_request_verified": False,
            "submitted_job": {
                "job_id": self.job_id,
                "scheduler_cluster": "fixture-cluster",
            },
            "export_contract": {
                "exported_value_observations": [
                    {
                        "name": "GRAFT_A_LITE_OUTPUT_STEM",
                        "value_sha256": _sha(stem_bytes),
                        "value_size_bytes": len(stem_bytes),
                    }
                ]
            },
            "outputs": {
                "logical_output_stem": str(self.stem),
                "submission_receipt_path": str(self.submission_path),
                "submission_receipt_create_only": True,
                "submission_receipt_mode": "0444",
            },
            "authority": {
                key: value
                for key, value in _false_authority().items()
                if key != "cross_clip_identity_authority"
            },
            "failure_semantics": {
                "submission_success_is_not_job_success": True,
                "job_success_requires_terminal_scheduler_and_execution_receipts": True,
            },
        }
        submission = _seal(submission_core)
        submission_raw = _canonical_line(submission)
        self._write_readonly(self.submission_path, submission_raw)

        terminal_core: dict[str, object] = {
            "schema_version": consumer.TERMINAL_ADMISSION_SCHEMA,
            "status": "admitted",
            "materializer": {
                "schema_version": consumer.TERMINAL_MATERIALIZER_SCHEMA,
                "implementation_sha256": "b" * 64,
                "runtime_sha256": "c" * 64,
                "independent_of_submitted_job_process": self.materializer_independent,
                "job_process_wrote_this_receipt": False,
                "observed_after_job_became_terminal": True,
            },
            "sacct_admission": {
                "source": "sacct",
                "queried_fields": ["JobIDRaw", "State", "ExitCode"],
                "job_id": self.job_id,
                "state": self.terminal_state,
                "exit_code": self.terminal_exit_code,
                "terminal_state_observed": True,
                "job_success": self.terminal_job_success,
                "raw_stdout_sha256": "d" * 64,
                "raw_stdout_size_bytes": 32,
                "selected_record_sha256": "e" * 64,
            },
            "artifact_bindings": {
                "manifest_file_sha256": _sha(manifest_raw),
                "producer_receipt_file_sha256": _sha(producer_raw),
                "producer_receipt_digest": producer["receipt_digest"],
                "execution_receipt_file_sha256": _sha(execution_raw),
                "execution_receipt_digest": execution["receipt_digest"],
                "submission_receipt_file_sha256": _sha(submission_raw),
                "submission_receipt_digest": submission["receipt_digest"],
            },
            "authority": _false_authority(),
        }
        terminal = _seal(terminal_core)
        self.terminal_bytes = _canonical_line(terminal)
        self.pins = consumer.ReleaseArtifactPins(
            manifest_sha256=_sha(manifest_raw),
            producer_receipt_sha256=_sha(producer_raw),
            execution_receipt_sha256=_sha(execution_raw),
            submission_receipt_sha256=_sha(submission_raw),
            terminal_admission_sha256=_sha(self.terminal_bytes),
            terminal_materializer_implementation_sha256="b" * 64,
            terminal_materializer_runtime_sha256="c" * 64,
        )

    def probe(self, source: consumer._OpenedSource) -> dict[str, int]:
        width, height = self.source_dimensions[source.path.stem]
        return {
            "frame_count": 81,
            "fps_numerator": 25,
            "fps_denominator": 1,
            "reported_fps_numerator": 25,
            "reported_fps_denominator": 1,
            "width": width,
            "height": height,
        }

    def consume(
        self, *, probe: object | None = None, terminal: bytes | None = None
    ) -> consumer.TestOnlySourceReleaseObservation:
        return consumer._consume_with_test_probe(
            manifest_path=self.manifest_path,
            producer_receipt_path=self.producer_path,
            execution_receipt_path=self.execution_path,
            submission_receipt_path=self.submission_path,
            terminal_admission_bytes=(
                self.terminal_bytes if terminal is None else terminal
            ),
            pins=self.pins,
            media_probe=self.probe if probe is None else probe,  # type: ignore[arg-type]
        )

    def consume_production(
        self, *, terminal: bytes | None = None
    ) -> consumer.SealedALiteSourceRelease:
        descriptor = os.open(
            self.manifest_path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        metadata = os.fstat(descriptor)
        fake_executable = consumer._OpenedFFprobe(
            path=self.manifest_path,
            fd=descriptor,
            identity=consumer._identity(metadata),
            execution_path="test-framework-patched-not-executed",
            transport="test_framework_patch",
        )
        with mock.patch.object(
            consumer, "_open_frozen_ffprobe", return_value=fake_executable
        ), mock.patch.object(
            consumer,
            "_probe_with_frozen_ffprobe",
            side_effect=lambda source, executable: self.probe(source),
        ), mock.patch.object(
            consumer, "_revalidate_frozen_ffprobe", return_value=None
        ):
            return consumer.consume_graft_a_lite_source_release(
                manifest_path=self.manifest_path,
                producer_receipt_path=self.producer_path,
                execution_receipt_path=self.execution_path,
                submission_receipt_path=self.submission_path,
                terminal_admission_bytes=(
                    self.terminal_bytes if terminal is None else terminal
                ),
                pins=self.pins,
            )


class GraftALiteSourceReleaseConsumerTests(unittest.TestCase):
    def fixture(self) -> tuple[tempfile.TemporaryDirectory[str], _Fixture]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        return temporary, _Fixture(Path(temporary.name))

    def test_fake_probe_returns_distinct_test_only_type_never_trainer_routing(self) -> None:
        _, fixture = self.fixture()
        result = fixture.consume()
        self.assertIs(type(result), consumer.TestOnlySourceReleaseObservation)
        self.assertTrue(result.test_only)
        self.assertFalse(result.production_release_minted)
        self.assertFalse(result.eligible_for_training_validation)
        self.assertEqual([row.iid for row in result.rows], [r[0] for r in consumer.CANARY4])
        self.assertEqual(result.provenance.job_id, "132549")
        self.assertEqual(result.provenance.scheduler_state, "COMPLETED")
        self.assertEqual(result.provenance.scheduler_exit_code, "0:0")
        self.assertFalse(result.provenance.consumer_fresh_portable_ffprobe_verified)
        self.assertTrue(all(value is False for value in vars(result.authority).values()))
        for row in result.rows:
            self.assertIsInstance(row.source_bytes, bytes)
            self.assertEqual(_sha(row.source_bytes), row.source_sha256)
            self.assertEqual(row.media.frame_count, 81)
            self.assertEqual((row.media.fps_numerator, row.media.fps_denominator), (25, 1))
            with self.assertRaises(FrozenInstanceError):
                row.split = "optimizer_train"  # type: ignore[misc]
        with self.assertRaisesRegex(consumer.GraftALiteConsumerError, "opaque production"):
            consumer.validate_for_training(result)  # type: ignore[arg-type]
        core = consumer._consume_probe_neutral(
            manifest_path=fixture.manifest_path,
            producer_receipt_path=fixture.producer_path,
            execution_receipt_path=fixture.execution_path,
            submission_receipt_path=fixture.submission_path,
            terminal_admission_bytes=fixture.terminal_bytes,
            pins=fixture.pins,
            media_probe=fixture.probe,
        )
        self.assertIs(type(core), consumer._CoreConsumedEvidence)
        with self.assertRaisesRegex(consumer.GraftALiteConsumerError, "opaque production"):
            consumer.validate_for_training(core)  # type: ignore[arg-type]
        self.assertNotIn(
            "media_probe",
            inspect.signature(consumer._consume_core).parameters,
        )
        self.assertFalse(hasattr(consumer, "_mint_after_frozen_probe_revalidation"))
        self.assertFalse(hasattr(consumer, "_mint_production_release"))

    def test_production_mint_requires_validate_for_training_and_returns_path_free_rows(self) -> None:
        _, fixture = self.fixture()
        release = fixture.consume_production()
        self.assertIs(type(release), consumer.SealedALiteSourceRelease)
        self.assertRegex(release.result_digest, r"^[0-9a-f]{64}$")
        self.assertRegex(release.pinset_digest, r"^[0-9a-f]{64}$")
        self.assertFalse(hasattr(release, "optimizer_train_rows"))
        routing = consumer.validate_for_training(release)
        self.assertIs(type(routing), consumer.TrainerRouting)
        self.assertEqual(len(routing.update_rows), 2)
        self.assertEqual(len(routing.confirmation_rows), 2)
        self.assertTrue(all(row.optimizer_update_allowed for row in routing.update_rows))
        self.assertTrue(
            all(
                not row.optimizer_update_allowed
                and row.optimizer_confirmation_only
                for row in routing.confirmation_rows
            )
        )
        self.assertEqual(routing.source_release_result_digest, release.result_digest)
        self.assertTrue(all(value is False for value in vars(routing.authority).values()))
        for row in (*routing.update_rows, *routing.confirmation_rows):
            self.assertIs(type(row.source_bytes), bytes)
            self.assertEqual(_sha(row.source_bytes), row.source_sha256)
            self.assertFalse(hasattr(row, "source_path"))
            self.assertFalse(hasattr(row, "source_path_observed"))
            self.assertFalse(hasattr(row, "path"))

    def test_terminal_materializer_and_consumer_schema_integrate_byte_for_byte(self) -> None:
        _, fixture = self.fixture()
        terminal_pins = terminal_materializer.TerminalArtifactPins(
            manifest_sha256=fixture.pins.manifest_sha256,
            producer_receipt_sha256=fixture.pins.producer_receipt_sha256,
            execution_receipt_sha256=fixture.pins.execution_receipt_sha256,
            submission_receipt_sha256=fixture.pins.submission_receipt_sha256,
            materializer_implementation_sha256="b" * 64,
            materializer_runtime_sha256="c" * 64,
        )
        bundle = terminal_materializer._open_bundle(
            manifest_path=fixture.manifest_path,
            producer_receipt_path=fixture.producer_path,
            execution_receipt_path=fixture.execution_path,
            submission_receipt_path=fixture.submission_path,
            pins=terminal_pins,
        )
        try:
            raw_sacct = (
                b"132549|COMPLETED|0:0|2026-08-10T12:00:00|"
                b"2026-08-10T12:00:05|00:00:05|auh7-1b-gpu-186|\n"
            )
            observation = terminal_materializer._parse_sacct_stdout(
                raw_sacct,
                runtime_sha="c" * 64,
            )
            receipt, receipt_bytes = terminal_materializer._receipt_core(
                bundle,
                observation,
                implementation_sha256="b" * 64,
                schema_version=terminal_materializer.TERMINAL_SCHEMA,
                status="admitted",
                materializer_schema=terminal_materializer.MATERIALIZER_SCHEMA,
                source="sacct",
                independently_observed=True,
            )
        finally:
            terminal_materializer._close_bundle(bundle)
        self.assertEqual(receipt["status"], "admitted")
        fixture.terminal_bytes = receipt_bytes
        fixture.pins = consumer.ReleaseArtifactPins(
            manifest_sha256=fixture.pins.manifest_sha256,
            producer_receipt_sha256=fixture.pins.producer_receipt_sha256,
            execution_receipt_sha256=fixture.pins.execution_receipt_sha256,
            submission_receipt_sha256=fixture.pins.submission_receipt_sha256,
            terminal_admission_sha256=_sha(receipt_bytes),
            terminal_materializer_implementation_sha256="b" * 64,
            terminal_materializer_runtime_sha256="c" * 64,
        )
        routing = consumer.validate_for_training(fixture.consume_production())
        self.assertEqual(len(routing.update_rows), 2)
        self.assertEqual(len(routing.confirmation_rows), 2)

    def test_manifest_byte_tamper_fails_external_pin(self) -> None:
        _, fixture = self.fixture()
        fixture.manifest_path.chmod(0o644)
        fixture.manifest_path.write_bytes(fixture.manifest_path.read_bytes() + b" ")
        fixture.manifest_path.chmod(0o444)
        with self.assertRaisesRegex(consumer.GraftALiteConsumerError, "SHA-256 differs"):
            fixture.consume()

    def test_row_split_and_confirmation_promotion_fail_even_when_resealed(self) -> None:
        _, fixture = self.fixture()
        fixture.rows[0]["split"] = "optimizer_confirmation"
        fixture.write_release()
        with self.assertRaisesRegex(consumer.GraftALiteConsumerError, "routing differs"):
            fixture.consume()

        _, fixture = self.fixture()
        fixture.rows[2]["optimizer_update_authorized"] = True
        fixture.write_release()
        with self.assertRaisesRegex(consumer.GraftALiteConsumerError, "routing differs"):
            fixture.consume()

    def test_source_swap_during_same_fd_probe_is_detected(self) -> None:
        _, fixture = self.fixture()
        swapped = False

        def swapping_probe(source: consumer._OpenedSource) -> dict[str, int]:
            nonlocal swapped
            value = fixture.probe(source)
            if not swapped:
                swapped = True
                source.path.unlink()
                source.path.write_bytes(b"attacker replacement")
                source.path.chmod(0o444)
            return value

        with self.assertRaisesRegex(
            consumer.GraftALiteConsumerError, "source path or identity changed"
        ):
            fixture.consume(probe=swapping_probe)

    def test_source_content_tamper_before_consumption_fails(self) -> None:
        _, fixture = self.fixture()
        source = Path(str(fixture.rows[0]["source_video_path"]))
        source.chmod(0o644)
        source.write_bytes(b"source tamper")
        source.chmod(0o444)
        with self.assertRaisesRegex(
            consumer.GraftALiteConsumerError, "content/provenance differs"
        ):
            fixture.consume()

    def test_producer_mount_device_is_provenance_not_cross_node_identity(self) -> None:
        _, fixture = self.fixture()
        fixture.execution_artifact_device_override = 9_000_001
        fixture.execution_source_device_override = 9_000_002
        fixture.write_release()
        result = fixture.consume()
        self.assertEqual(
            [row.iid for row in result.rows], [r[0] for r in consumer.CANARY4]
        )

        _, fixture = self.fixture()
        fixture.execution_artifact_device_override = True
        fixture.write_release()
        with self.assertRaisesRegex(
            consumer.GraftALiteConsumerError, "producer-namespace device"
        ):
            fixture.consume()

        _, fixture = self.fixture()
        fixture.execution_source_device_override = 0
        fixture.write_release()
        with self.assertRaisesRegex(
            consumer.GraftALiteConsumerError,
            "source row 0 producer-namespace device",
        ):
            fixture.consume()

    def test_cross_node_device_portability_does_not_weaken_inode_binding(self) -> None:
        _, fixture = self.fixture()
        fixture.execution_artifact_device_override = 9_000_001
        fixture.execution_artifact_inode_override = 9_000_003
        fixture.write_release()
        with self.assertRaisesRegex(
            consumer.GraftALiteConsumerError, "execution manifest artifact binding"
        ):
            fixture.consume()

        _, fixture = self.fixture()
        fixture.execution_source_device_override = 9_000_002
        fixture.execution_source_inode_override = 9_000_004
        fixture.write_release()
        with self.assertRaisesRegex(
            consumer.GraftALiteConsumerError, "source row 0 content/provenance differs"
        ):
            fixture.consume()

    def test_wrong_fresh_media_status_fails(self) -> None:
        _, fixture = self.fixture()

        def short_probe(source: consumer._OpenedSource) -> dict[str, int]:
            value = fixture.probe(source)
            value["frame_count"] = 80
            return value

        with self.assertRaisesRegex(consumer.GraftALiteConsumerError, "not fresh exact81"):
            fixture.consume(probe=short_probe)

    def test_fake_terminal_state_and_exit_code_fail_when_fully_resealed(self) -> None:
        _, fixture = self.fixture()
        fixture.terminal_state = "FAILED"
        fixture.terminal_job_success = False
        fixture.write_release()
        with self.assertRaisesRegex(consumer.GraftALiteConsumerError, "not COMPLETED 0:0"):
            fixture.consume()

        _, fixture = self.fixture()
        fixture.terminal_exit_code = "1:0"
        fixture.terminal_job_success = False
        fixture.write_release()
        with self.assertRaisesRegex(consumer.GraftALiteConsumerError, "not COMPLETED 0:0"):
            fixture.consume()

    def test_submission_receipt_cannot_be_used_as_terminal_evidence(self) -> None:
        _, fixture = self.fixture()
        submission_raw = fixture.submission_path.read_bytes()
        with self.assertRaisesRegex(consumer.GraftALiteConsumerError, "pairwise distinct"):
            consumer.ReleaseArtifactPins(
                manifest_sha256=fixture.pins.manifest_sha256,
                producer_receipt_sha256=fixture.pins.producer_receipt_sha256,
                execution_receipt_sha256=fixture.pins.execution_receipt_sha256,
                submission_receipt_sha256=fixture.pins.submission_receipt_sha256,
                terminal_admission_sha256=_sha(submission_raw),
                terminal_materializer_implementation_sha256=(
                    fixture.pins.terminal_materializer_implementation_sha256
                ),
                terminal_materializer_runtime_sha256=(
                    fixture.pins.terminal_materializer_runtime_sha256
                ),
            )

    def test_terminal_must_be_independent_and_bind_all_four_artifacts(self) -> None:
        _, fixture = self.fixture()
        fixture.materializer_independent = False
        fixture.write_release()
        with self.assertRaisesRegex(consumer.GraftALiteConsumerError, "not independently"):
            fixture.consume()

        _, fixture = self.fixture()
        terminal = json.loads(fixture.terminal_bytes)
        terminal["artifact_bindings"]["execution_receipt_file_sha256"] = "f" * 64
        terminal.pop("receipt_digest")
        terminal = _seal(terminal)
        raw = _canonical_line(terminal)
        fixture.pins = consumer.ReleaseArtifactPins(
            manifest_sha256=fixture.pins.manifest_sha256,
            producer_receipt_sha256=fixture.pins.producer_receipt_sha256,
            execution_receipt_sha256=fixture.pins.execution_receipt_sha256,
            submission_receipt_sha256=fixture.pins.submission_receipt_sha256,
            terminal_admission_sha256=_sha(raw),
            terminal_materializer_implementation_sha256=(
                fixture.pins.terminal_materializer_implementation_sha256
            ),
            terminal_materializer_runtime_sha256=(
                fixture.pins.terminal_materializer_runtime_sha256
            ),
        )
        with self.assertRaisesRegex(consumer.GraftALiteConsumerError, "bind all artifacts"):
            fixture.consume(terminal=raw)

    def test_execution_status_and_submission_job_success_cannot_self_authorize(self) -> None:
        _, fixture = self.fixture()
        execution = json.loads(fixture.execution_path.read_bytes())
        execution["status"] = "failed"
        execution.pop("receipt_digest")
        execution = _seal(execution)
        fixture._write_readonly(fixture.execution_path, _canonical_line(execution))
        # External pin can be maliciously refreshed, but semantic gate still closes.
        fixture.pins = consumer.ReleaseArtifactPins(
            manifest_sha256=fixture.pins.manifest_sha256,
            producer_receipt_sha256=fixture.pins.producer_receipt_sha256,
            execution_receipt_sha256=_sha(fixture.execution_path.read_bytes()),
            submission_receipt_sha256=fixture.pins.submission_receipt_sha256,
            terminal_admission_sha256=fixture.pins.terminal_admission_sha256,
            terminal_materializer_implementation_sha256=(
                fixture.pins.terminal_materializer_implementation_sha256
            ),
            terminal_materializer_runtime_sha256=(
                fixture.pins.terminal_materializer_runtime_sha256
            ),
        )
        with self.assertRaisesRegex(consumer.GraftALiteConsumerError, "success boundary differs"):
            fixture.consume()

        _, fixture = self.fixture()
        submission = json.loads(fixture.submission_path.read_bytes())
        submission["job_success"] = True
        submission["job_terminal_state_observed"] = True
        submission.pop("receipt_digest")
        submission = _seal(submission)
        fixture._write_readonly(fixture.submission_path, _canonical_line(submission))
        fixture.pins = consumer.ReleaseArtifactPins(
            manifest_sha256=fixture.pins.manifest_sha256,
            producer_receipt_sha256=fixture.pins.producer_receipt_sha256,
            execution_receipt_sha256=fixture.pins.execution_receipt_sha256,
            submission_receipt_sha256=_sha(fixture.submission_path.read_bytes()),
            terminal_admission_sha256=fixture.pins.terminal_admission_sha256,
            terminal_materializer_implementation_sha256=(
                fixture.pins.terminal_materializer_implementation_sha256
            ),
            terminal_materializer_runtime_sha256=(
                fixture.pins.terminal_materializer_runtime_sha256
            ),
        )
        with self.assertRaisesRegex(consumer.GraftALiteConsumerError, "must remain non-terminal"):
            fixture.consume()

    def test_artifacts_require_exact_0444_link_count_one_and_nofollow(self) -> None:
        temporary, fixture = self.fixture()
        fixture.producer_path.chmod(0o644)
        with self.assertRaisesRegex(consumer.GraftALiteConsumerError, "mode-0444"):
            fixture.consume()

        _, fixture = self.fixture()
        hardlink = Path(temporary.name).resolve() / "producer-hardlink.json"
        os.link(fixture.producer_path, hardlink)
        with self.assertRaisesRegex(consumer.GraftALiteConsumerError, "link-count-one"):
            fixture.consume()

    def test_artifact_path_replacement_race_is_detected_at_terminal_revalidation(self) -> None:
        _, fixture = self.fixture()
        original = consumer._revalidate_opened_file
        replaced = False

        def replace_once(opened: consumer._OpenedFile, *, label: str) -> None:
            nonlocal replaced
            if label == "manifest" and not replaced:
                replaced = True
                opened.path.unlink()
                opened.path.write_bytes(opened.raw)
                opened.path.chmod(0o444)
            original(opened, label=label)

        with mock.patch.object(consumer, "_revalidate_opened_file", replace_once):
            with self.assertRaisesRegex(
                consumer.GraftALiteConsumerError, "path or identity changed"
            ):
                fixture.consume()

    def test_dataclass_replace_and_confirmation_promotion_cannot_mint_routing(self) -> None:
        _, fixture = self.fixture()
        release = fixture.consume_production()
        with self.assertRaises(TypeError):
            replace(release, rows=release.rows)
        promoted = replace(
            release.rows[2],
            optimizer_update_allowed=True,
            optimizer_confirmation_only=False,
            split="optimizer_train",
        )
        object.__setattr__(
            release,
            "_rows",
            (release.rows[0], release.rows[1], promoted, release.rows[3]),
        )
        with self.assertRaisesRegex(
            consumer.GraftALiteConsumerError, "owned row 2 differs"
        ):
            consumer.validate_for_training(release)

    def test_result_digest_and_owned_source_bytes_are_revalidated(self) -> None:
        _, fixture = self.fixture()
        release = fixture.consume_production()
        object.__setattr__(release, "_result_digest", "0" * 64)
        with self.assertRaisesRegex(consumer.GraftALiteConsumerError, "result digest"):
            consumer.validate_for_training(release)

        _, fixture = self.fixture()
        release = fixture.consume_production()
        changed = replace(release.rows[0], source_bytes=b"changed-owned-bytes")
        object.__setattr__(release, "_rows", (changed, *release.rows[1:]))
        with self.assertRaisesRegex(consumer.GraftALiteConsumerError, "owned row 0 differs"):
            consumer.validate_for_training(release)

    def test_validate_for_training_never_reopens_observed_paths(self) -> None:
        _, fixture = self.fixture()
        release = fixture.consume_production()
        with mock.patch.object(
            consumer.os, "open", side_effect=AssertionError("path reopen forbidden")
        ):
            routing = consumer.validate_for_training(release)
        self.assertEqual(len(routing.update_rows), 2)
        self.assertEqual(len(routing.confirmation_rows), 2)
        for row in (*routing.update_rows, *routing.confirmation_rows):
            self.assertFalse(any("path" in name for name in row.__slots__))

    def test_wrong_terminal_materializer_external_pins_fail(self) -> None:
        _, fixture = self.fixture()
        fixture.pins = consumer.ReleaseArtifactPins(
            manifest_sha256=fixture.pins.manifest_sha256,
            producer_receipt_sha256=fixture.pins.producer_receipt_sha256,
            execution_receipt_sha256=fixture.pins.execution_receipt_sha256,
            submission_receipt_sha256=fixture.pins.submission_receipt_sha256,
            terminal_admission_sha256=fixture.pins.terminal_admission_sha256,
            terminal_materializer_implementation_sha256="d" * 64,
            terminal_materializer_runtime_sha256=(
                fixture.pins.terminal_materializer_runtime_sha256
            ),
        )
        with self.assertRaisesRegex(
            consumer.GraftALiteConsumerError, "not independently materialized"
        ):
            fixture.consume()

        _, fixture = self.fixture()
        fixture.pins = consumer.ReleaseArtifactPins(
            manifest_sha256=fixture.pins.manifest_sha256,
            producer_receipt_sha256=fixture.pins.producer_receipt_sha256,
            execution_receipt_sha256=fixture.pins.execution_receipt_sha256,
            submission_receipt_sha256=fixture.pins.submission_receipt_sha256,
            terminal_admission_sha256=fixture.pins.terminal_admission_sha256,
            terminal_materializer_implementation_sha256=(
                fixture.pins.terminal_materializer_implementation_sha256
            ),
            terminal_materializer_runtime_sha256="d" * 64,
        )
        with self.assertRaisesRegex(
            consumer.GraftALiteConsumerError, "not independently materialized"
        ):
            fixture.consume()

    def test_opaque_release_and_trainer_rows_reject_direct_construction(self) -> None:
        _, fixture = self.fixture()
        release = fixture.consume_production()
        routing = consumer.validate_for_training(release)
        self.assertFalse(hasattr(consumer, "_TRAINER_ROUTING_MINT"))
        with self.assertRaisesRegex(consumer.GraftALiteConsumerError, "only be minted"):
            consumer.TrainerOwnedSourceRow(release.rows[0], _mint=object())
        with self.assertRaisesRegex(consumer.GraftALiteConsumerError, "only be minted"):
            consumer.TrainerRouting(
                update_rows=routing.confirmation_rows,
                confirmation_rows=routing.update_rows,
                source_release_result_digest=release.result_digest,
                pinset_digest=release.pinset_digest,
                routing_digest="f" * 64,
                authority=consumer.AuthorityBoundary(),
                _mint=object(),
            )
        with self.assertRaisesRegex(consumer.GraftALiteConsumerError, "only be minted"):
            consumer.SealedALiteSourceRelease(
                rows=release.rows,
                provenance=release.provenance,
                authority=release.authority,
                pins=fixture.pins,
                pinset_digest=release.pinset_digest,
                result_digest=release.result_digest,
                _mint=object(),
            )

    def test_public_api_has_no_probe_override_or_terminal_status_arguments(self) -> None:
        import inspect

        parameters = inspect.signature(
            consumer.consume_graft_a_lite_source_release
        ).parameters
        self.assertNotIn("media_probe", parameters)
        self.assertNotIn("job_state", parameters)
        self.assertNotIn("exit_code", parameters)
        self.assertNotIn("job_success", parameters)
        source = inspect.getsource(consumer._consume_core)
        self.assertIn("_open_frozen_ffprobe", source)
        self.assertIn("_probe_with_frozen_ffprobe", source)


if __name__ == "__main__":
    unittest.main()
