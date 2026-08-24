from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from motive import wan22_signed_release as release
from motive import wan22_i2v_batch as batch
from motive import wan22_parallel_shards as parallel
from motive import wan22_select_exact8 as selector


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


class Wan22SignedReleaseTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, list[dict[str, object]]]:
        rows: list[dict[str, object]] = []
        for index in range(release.RELEASE_ROW_COUNT):
            source = root / f"source-{index}.mp4"
            anchor = root / f"anchor-{index}.png"
            source.write_bytes(f"source-video-{index}".encode())
            anchor.write_bytes(f"anchor-image-{index}".encode())
            source_sha = _sha(source.read_bytes())
            anchor_sha = _sha(anchor.read_bytes())
            media = {
                "frame_count": 81,
                "fps": 25.0,
                "duration_seconds": 3.24,
            }
            verification = {
                "schema_version": release.MEDIA_FILE_VERIFICATION_SCHEMA,
                "source_video": {
                    "resolved_path": str(source.resolve()),
                    "sha256": source_sha,
                    "bytes": source.stat().st_size,
                },
                "anchor_image": {
                    "resolved_path": str(anchor.resolve()),
                    "sha256": anchor_sha,
                    "bytes": anchor.stat().st_size,
                },
            }
            verification["verification_digest"] = release._object_digest(
                verification
            )
            instruction = f"Make subject {index} perform action {index}."
            rows.append(
                {
                    "schema_version": release.GENERATION_MANIFEST_SCHEMA,
                    "iid": f"sample-{index:02d}",
                    "group_id": f"group-{index:02d}",
                    "action_category": "interaction",
                    "target_action_verb": "perform",
                    "target_action_normalized": f"perform action {index}",
                    "target_semantics_source": "judge_a_instruction_bound",
                    "action_change_substantive": "yes",
                    "source_video": f"source-{index}.mp4",
                    "resolved_source_video": str(source.resolve()),
                    "anchor_image": f"anchor-{index}.png",
                    "resolved_anchor_image": str(anchor.resolve()),
                    "anchor_sha256": anchor_sha,
                    "source_video_sha256": source_sha,
                    "selected_media_evidence": media,
                    "selected_media_evidence_sha256": release._object_digest(media),
                    "strict_temporal_geometry": {
                        "schema_version": release.TEMPORAL_GEOMETRY_SCHEMA,
                        "source_frame_count": 81,
                        "required_output_frame_count": 81,
                        "source_fps": 25.0,
                        "required_output_fps": 25.0,
                        "source_duration_seconds": 3.24,
                        "required_output_duration_seconds": 3.24,
                        "maximum_duration_delta_frames": 1,
                        "maximum_duration_delta_seconds": 0.04,
                        "frame_count_form": "4n+1",
                        "frame_count_modulus": 4,
                        "frame_count_remainder": 1,
                        "source_timeline_duration_seconds": 3.2,
                        "source_timeline_error_seconds": 0.04,
                        "requirements": {
                            "same_frame_count": True,
                            "same_fps": True,
                            "duration_absolute_delta_at_most_one_frame": True,
                        },
                    },
                    "finalizer_media_file_verification": verification,
                    "edit_instruction": instruction,
                    "edit_instruction_sha256": _sha(instruction.encode()),
                    "instruction_contract": {
                        "sole_candidate_instruction_field": "edit_instruction",
                        "candidate_instruction_source": "frozen_selected_prompt",
                        "writer_proposal_payload_included": False,
                        "writer_proposals_executable": False,
                        "requires_future_signed_release_verifier": True,
                    },
                    "source_caption": f"Source caption {index}",
                    "source_edited_caption_provenance": (
                        f"Non-executable target prose {index}"
                    ),
                    "source_edited_caption_provenance_role": (
                        "non_executable_provenance"
                    ),
                    "source_instruction_provenance": instruction,
                    "qwen_input_digest": "1" * 64,
                    "qwen_config_digest": "2" * 64,
                    "manifest_role": "review_proposal",
                    "production_eligible": False,
                    "human_review_status": "pending",
                    "generation_authorized": False,
                    "approval": None,
                    "authorization_interface_available": False,
                }
            )
        manifest = root / "generation_manifest.jsonl"
        manifest.write_bytes(
            b"".join(release._canonical_bytes(row) + b"\n" for row in rows)
        )
        acceptance = {
            "schema_version": release.ACCEPTANCE_RESULT_SCHEMA,
            "contract": {"sha256": "1" * 64},
            "submission_contract": {"sha256": "2" * 64},
            "completion_receipt": {"sha256": "3" * 64},
            "selected": {
                "rows": 16,
                "sha256": "4" * 64,
                "ordered_iids_sha256": "5" * 64,
            },
            "model": {
                "path": release.QWEN3_MODEL_PATH,
                "config_sha256": "6" * 64,
            },
            "model_closure": {
                "model_path": release.QWEN3_MODEL_PATH,
                "manifest_sha256": "7" * 64,
                "files_digest": "8" * 64,
            },
            "source_snapshot": {
                "tree_sha256": "9" * 64,
                "implementations": {
                    "qwen": {"sha256": "a" * 64},
                    "verifier": {
                        "path": str(
                            Path(release.__file__).resolve().with_name(
                                "goku_action_v13_acceptance.py"
                            )
                        ),
                        "sha256": _sha(
                            Path(release.__file__)
                            .resolve()
                            .with_name("goku_action_v13_acceptance.py")
                            .read_bytes()
                        ),
                    },
                },
            },
            "failures": [],
            "passed": True,
            "full_123_authorized": True,
            "generation_authorized": False,
            "production_eligible": False,
            "wan_generation_authorized": False,
            "authorization_interface_available": False,
        }
        acceptance_path = root / "acceptance_result.json"
        _write_json(acceptance_path, acceptance)
        return manifest, acceptance_path, rows

    def _finalizer_fixture(
        self,
        root: Path,
        rows: list[dict[str, object]],
    ) -> tuple[Path, Path, Path]:
        finalizer = root / "finalizer"
        finalizer.mkdir()
        review_rows: list[dict[str, object]] = []
        for index, generation_row in enumerate(rows, start=1):
            review_rows.append(
                {
                    "iid": generation_row["iid"],
                    "group_id": generation_row["group_id"],
                    "prompt": generation_row["edit_instruction"],
                    "action_anchor_finalization": {
                        "schema_version": selector.FINALIZER_REVIEW_SCHEMA,
                        "policy_version": selector.FINALIZER_POLICY_VERSION,
                        "hard_gate_passed": True,
                        "hard_gate_failures": [],
                        "review_rank": index,
                        "selection_bucket": "proposed",
                        "human_review_status": "pending",
                        "human_label": False,
                        "generation_authorized": False,
                        "manifest_role": "review_proposal",
                        "production_eligible": False,
                        "approval": None,
                        "authorization_interface_available": False,
                    },
                }
            )
        review_raw = b"".join(
            release._canonical_bytes(row) + b"\n" for row in review_rows
        )
        generation_raw = b"".join(
            release._canonical_bytes(row) + b"\n" for row in rows
        )
        (finalizer / selector.REVIEW_NAME).write_bytes(review_raw)
        (finalizer / selector.PROPOSED_NAME).write_bytes(review_raw)
        (finalizer / selector.RESERVE_NAME).write_bytes(b"")
        (finalizer / selector.PARENT_GENERATION_NAME).write_bytes(
            generation_raw
        )
        implementation_sha = _sha(
            Path(selector.__file__)
            .resolve()
            .with_name("goku_action_anchor_finalize.py")
            .read_bytes()
        )
        summary = {
            "schema_version": selector.FINALIZER_SUMMARY_SCHEMA,
            "policy_version": selector.FINALIZER_POLICY_VERSION,
            "seed": 260730,
            "input": {},
            "hard_gate": {},
            "diversity": {},
            "selection": {
                "review_rows": len(rows),
                "generation_rows": len(rows),
                "proposed_rows": len(rows),
            },
            "semantics": {
                "manifest_role": "review_proposal",
                "human_review_status": "pending",
                "human_labels_asserted": False,
                "generation_authorized": False,
                "production_eligible": False,
                "approval": None,
                "authorization_interface_available": False,
            },
            "implementation_sha256": implementation_sha,
            "output_sha256": {
                name: _sha((finalizer / name).read_bytes())
                for name in selector._SUMMARY_HASHED_OUTPUTS
            },
        }
        _write_json(finalizer / selector.SUMMARY_NAME, summary)
        done = {
            "schema_version": selector.FINALIZER_DONE_SCHEMA,
            "status": "complete",
            "summary_sha256": _sha(
                (finalizer / selector.SUMMARY_NAME).read_bytes()
            ),
            "implementation_sha256": implementation_sha,
            "output_sha256": {
                name: _sha((finalizer / name).read_bytes())
                for name in selector._FINALIZER_HASHED_OUTPUTS
            },
        }
        _write_json(finalizer / selector.DONE_NAME, done)
        selected = root / "selected-exact8"
        selector.select_exact8(
            finalizer_dir=finalizer,
            output_dir=selected,
        )
        return (
            finalizer,
            selected / selector.OUTPUT_MANIFEST_NAME,
            selected / selector.OUTPUT_RECEIPT_NAME,
        )

    def _key(self, root: Path) -> tuple[Path, str, str]:
        key = root / "test_signer"
        subprocess.run(
            ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)],
            check=True,
        )
        public = " ".join(
            (root / "test_signer.pub").read_text().split()[:2]
        )
        fingerprint = subprocess.run(
            ["ssh-keygen", "-lf", str(root / "test_signer.pub")],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout.split()[1]
        return key, public, fingerprint

    def _signed(
        self,
        root: Path,
    ) -> tuple[Path, Path, list[dict[str, object]]]:
        manifest, acceptance, rows = self._fixture(root)
        key, public, fingerprint = self._key(root)
        output = root / "signed_release.json"
        with (
            mock.patch.object(release, "SIGNER_PUBLIC_KEY", public),
            mock.patch.object(
                release,
                "SIGNER_KEY_FINGERPRINT",
                fingerprint,
            ),
            mock.patch.object(
                release,
                "_rerun_acceptance",
                return_value={},
            ),
        ):
            release.build_and_sign_release(
                manifest_path=manifest,
                smoke_acceptance_path=acceptance,
                output_path=output,
                signing_key=key,
                release_id="unit-test-release-001",
                issued_at_utc="2026-07-31T12:00:00+00:00",
            )
        return output, manifest, rows

    def _verify(self, release_path: Path, manifest: Path) -> dict[str, object]:
        public = " ".join(
            (release_path.parent / "test_signer.pub").read_text().split()[:2]
        )
        fingerprint = subprocess.run(
            ["ssh-keygen", "-lf", str(release_path.parent / "test_signer.pub")],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout.split()[1]
        with (
            mock.patch.object(release, "SIGNER_PUBLIC_KEY", public),
            mock.patch.object(
                release,
                "SIGNER_KEY_FINGERPRINT",
                fingerprint,
            ),
        ):
            return release.verify_signed_release(
                release_path=release_path,
                manifest_path=manifest,
                require_exact_manifest=True,
            )

    def _signed_finalizer(
        self,
        root: Path,
    ) -> tuple[Path, Path, Path, Path, list[dict[str, object]]]:
        _unused_manifest, _unused_acceptance, rows = self._fixture(root)
        finalizer, manifest, receipt = self._finalizer_fixture(root, rows)
        key, public, fingerprint = self._key(root)
        output = root / "finalizer-signed-release.json"
        with (
            mock.patch.object(release, "SIGNER_PUBLIC_KEY", public),
            mock.patch.object(
                release,
                "SIGNER_KEY_FINGERPRINT",
                fingerprint,
            ),
        ):
            release.build_and_sign_release(
                manifest_path=manifest,
                finalizer_dir=finalizer,
                selection_receipt_path=receipt,
                output_path=output,
                signing_key=key,
                release_id="unit-test-finalizer-release-001",
                issued_at_utc="2026-07-31T12:00:00+00:00",
            )
        return output, manifest, finalizer, receipt, rows

    def test_valid_exact_eight_release_and_contiguous_shard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            signed, manifest, rows = self._signed(root)
            result = self._verify(signed, manifest)
            self.assertEqual(result["selected_row_count"], 8)
            self.assertTrue(
                all(
                    row["_authorization_mode"] == release.AUTHORIZATION_MODE
                    for row in result["selected_rows"]
                )
            )
            shard = root / "shard.jsonl"
            shard.write_bytes(
                b"".join(
                    release._canonical_bytes(row) + b"\n"
                    for row in rows[2:5]
                )
            )
            public = " ".join((root / "test_signer.pub").read_text().split()[:2])
            fingerprint = subprocess.run(
                ["ssh-keygen", "-lf", str(root / "test_signer.pub")],
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            ).stdout.split()[1]
            with (
                mock.patch.object(release, "SIGNER_PUBLIC_KEY", public),
                mock.patch.object(
                    release,
                    "SIGNER_KEY_FINGERPRINT",
                    fingerprint,
                ),
            ):
                subset = release.verify_signed_release(
                    release_path=signed,
                    manifest_path=shard,
                    require_exact_manifest=False,
                )
            self.assertEqual(
                [row["_iid"] for row in subset["selected_rows"]],
                ["sample-02", "sample-03", "sample-04"],
            )

    def test_batch_loader_and_parallel_prepare_require_same_signature(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            signed, manifest, _ = self._signed(root)
            public = " ".join((root / "test_signer.pub").read_text().split()[:2])
            fingerprint = subprocess.run(
                ["ssh-keygen", "-lf", str(root / "test_signer.pub")],
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            ).stdout.split()[1]
            with (
                mock.patch.object(release, "SIGNER_PUBLIC_KEY", public),
                mock.patch.object(
                    release,
                    "SIGNER_KEY_FINGERPRINT",
                    fingerprint,
                ),
            ):
                loaded = batch.load_generation_manifest(
                    manifest,
                    allow_pending_review=False,
                    max_samples=None,
                    signed_release_path=signed,
                )
                plan = parallel.prepare_parallel_run(
                    manifest_path=manifest,
                    signed_release_path=signed,
                    parallel_root=root / "parallel",
                    geometry_job_id=12345,
                    shard_count=4,
                    allow_pending_review=False,
                    expected_row_count=8,
                )
            self.assertEqual(loaded["selected_row_count"], 8)
            self.assertTrue(
                all(
                    "absolute_target_prompt" not in row
                    and row["edit_instruction"]
                    for row in loaded["selected_rows"]
                )
            )
            self.assertEqual(
                plan["authorization"]["mode"],
                release.AUTHORIZATION_MODE,
            )
            self.assertEqual(
                [item["row_count"] for item in plan["shards"]],
                [2, 2, 2, 2],
            )

    def test_boolean_or_instruction_flip_cannot_forge_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            signed, manifest, rows = self._signed(root)
            rows[0]["generation_authorized"] = True
            rows[0]["edit_instruction"] = "Do something else."
            manifest.write_bytes(
                b"".join(
                    release._canonical_bytes(row) + b"\n" for row in rows
                )
            )
            with self.assertRaises(release.Wan22ReleaseError):
                self._verify(signed, manifest)

    def test_wrong_model_or_failed_smoke_cannot_be_signed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, acceptance, _ = self._fixture(root)
            key, public, fingerprint = self._key(root)
            value = json.loads(acceptance.read_text())
            value["model"]["path"] = "/cache/Qwen2.5-VL-7B"
            value["model_closure"]["model_path"] = value["model"]["path"]
            value["passed"] = False
            _write_json(acceptance, value)
            with (
                mock.patch.object(release, "SIGNER_PUBLIC_KEY", public),
                mock.patch.object(
                    release,
                    "SIGNER_KEY_FINGERPRINT",
                    fingerprint,
                ),
                mock.patch.object(
                    release,
                    "_rerun_acceptance",
                    return_value={},
                ),
                self.assertRaises(release.Wan22ReleaseError),
            ):
                release.build_and_sign_release(
                    manifest_path=manifest,
                    smoke_acceptance_path=acceptance,
                    output_path=root / "release.json",
                    signing_key=key,
                    release_id="bad",
                    issued_at_utc="2026-07-31T12:00:00+00:00",
                )

    def test_self_reported_acceptance_cannot_reach_signing_key(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, acceptance, _ = self._fixture(root)
            key, public, fingerprint = self._key(root)
            output = root / "must-not-exist.json"
            with (
                mock.patch.object(release, "SIGNER_PUBLIC_KEY", public),
                mock.patch.object(
                    release,
                    "SIGNER_KEY_FINGERPRINT",
                    fingerprint,
                ),
                self.assertRaisesRegex(
                    release.Wan22ReleaseError,
                    "acceptance",
                ),
            ):
                release.build_and_sign_release(
                    manifest_path=manifest,
                    smoke_acceptance_path=acceptance,
                    output_path=output,
                    signing_key=key,
                    release_id="forged-self-report",
                    issued_at_utc="2026-07-31T12:00:00+00:00",
                )
            self.assertFalse(output.exists())

    def test_prepared_request_signs_only_matching_challenge_and_sources(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, acceptance, _ = self._fixture(root)
            key, public, fingerprint = self._key(root)
            challenge = "c" * 64
            signed_payload = release._payload(
                manifest_path=manifest,
                smoke_acceptance_path=acceptance,
                release_id="prepared-request-test",
                issued_at_utc="2026-07-31T12:00:00+00:00",
            )
            request = {
                "schema_version": release.RELEASE_REQUEST_SCHEMA,
                "challenge_sha256": challenge,
                "builder": {
                    "release_module_sha256": _sha(
                        Path(release.__file__).resolve().read_bytes()
                    ),
                    "acceptance_verifier_sha256": _sha(
                        Path(release.__file__)
                        .resolve()
                        .with_name("goku_action_v13_acceptance.py")
                        .read_bytes()
                    ),
                },
                "signed": signed_payload,
            }
            request["request_digest"] = release._object_digest(request)
            request_path = root / "request.json"
            _write_json(request_path, request)
            output = root / "release.json"
            with (
                mock.patch.object(release, "SIGNER_PUBLIC_KEY", public),
                mock.patch.object(
                    release,
                    "SIGNER_KEY_FINGERPRINT",
                    fingerprint,
                ),
            ):
                release.sign_prepared_request(
                    request_path=request_path,
                    output_path=output,
                    signing_key=key,
                    expected_challenge=challenge,
                )
                verified = release.verify_signed_release(
                    release_path=output,
                    manifest_path=manifest,
                    require_exact_manifest=True,
                )
            self.assertEqual(verified["selected_row_count"], 8)
            wrong_output = root / "wrong-challenge.json"
            with (
                mock.patch.object(release, "SIGNER_PUBLIC_KEY", public),
                mock.patch.object(
                    release,
                    "SIGNER_KEY_FINGERPRINT",
                    fingerprint,
                ),
                self.assertRaisesRegex(
                    release.Wan22ReleaseError,
                    "challenge differs",
                ),
            ):
                release.sign_prepared_request(
                    request_path=request_path,
                    output_path=wrong_output,
                    signing_key=key,
                    expected_challenge="d" * 64,
                )
            self.assertFalse(wrong_output.exists())

            policy_tamper = json.loads(request_path.read_text())
            policy_tamper["signed"]["prompt_policy"][
                "writer_proposals_executable"
            ] = True
            del policy_tamper["request_digest"]
            policy_tamper["request_digest"] = release._object_digest(
                policy_tamper
            )
            policy_path = root / "policy-tamper-request.json"
            _write_json(policy_path, policy_tamper)
            with (
                mock.patch.object(release, "SIGNER_PUBLIC_KEY", public),
                mock.patch.object(
                    release,
                    "SIGNER_KEY_FINGERPRINT",
                    fingerprint,
                ),
                self.assertRaisesRegex(
                    release.Wan22ReleaseError,
                    "payload policy differs",
                ),
            ):
                release.sign_prepared_request(
                    request_path=policy_path,
                    output_path=root / "policy-tamper-release.json",
                    signing_key=key,
                    expected_challenge=challenge,
                )

            source_tamper = json.loads(request_path.read_text())
            source_tamper["builder"]["acceptance_verifier_sha256"] = "0" * 64
            del source_tamper["request_digest"]
            source_tamper["request_digest"] = release._object_digest(
                source_tamper
            )
            source_path = root / "source-tamper-request.json"
            _write_json(source_path, source_tamper)
            with (
                mock.patch.object(release, "SIGNER_PUBLIC_KEY", public),
                mock.patch.object(
                    release,
                    "SIGNER_KEY_FINGERPRINT",
                    fingerprint,
                ),
                self.assertRaisesRegex(
                    release.Wan22ReleaseError,
                    "remote/local acceptance verifier SHA differs",
                ),
            ):
                release.sign_prepared_request(
                    request_path=source_path,
                    output_path=root / "source-tamper-release.json",
                    signing_key=key,
                    expected_challenge=challenge,
                )

    def test_finalizer_selection_release_reruns_and_verifies(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            signed, manifest, _finalizer, _receipt, _rows = (
                self._signed_finalizer(root)
            )
            result = self._verify(signed, manifest)
            self.assertEqual(result["selected_row_count"], 8)
            self.assertIn("finalizer_selection", result["release"])
            self.assertNotIn("smoke_acceptance", result["release"])
            evidence = result["release"]["finalizer_selection"]
            self.assertEqual(
                evidence["mode"],
                release.FINALIZER_SELECTION_EVIDENCE_MODE,
            )
            self.assertEqual(
                evidence["selected_manifest"]["sha256"],
                _sha(manifest.read_bytes()),
            )
            public = " ".join(
                (root / "test_signer.pub").read_text().split()[:2]
            )
            fingerprint = subprocess.run(
                ["ssh-keygen", "-lf", str(root / "test_signer.pub")],
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            ).stdout.split()[1]
            with (
                mock.patch.object(release, "SIGNER_PUBLIC_KEY", public),
                mock.patch.object(
                    release,
                    "SIGNER_KEY_FINGERPRINT",
                    fingerprint,
                ),
            ):
                loaded = batch.load_generation_manifest(
                    manifest,
                    allow_pending_review=False,
                    max_samples=None,
                    signed_release_path=signed,
                )
                plan = parallel.prepare_parallel_run(
                    manifest_path=manifest,
                    signed_release_path=signed,
                    parallel_root=root / "finalizer-parallel",
                    geometry_job_id=12345,
                    shard_count=4,
                    allow_pending_review=False,
                    expected_row_count=8,
                )
            self.assertEqual(loaded["selected_row_count"], 8)
            self.assertIn(
                "finalizer_selection",
                plan["authorization"]["release"],
            )

    def test_finalizer_selection_rejects_nonidentical_or_ambiguous_input(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original, acceptance, rows = self._fixture(root)
            finalizer, _selected, receipt = self._finalizer_fixture(root, rows)
            reordered = root / "reordered.jsonl"
            reordered.write_bytes(
                b"".join(
                    release._canonical_bytes(row) + b"\n"
                    for row in reversed(rows)
                )
            )
            key, public, fingerprint = self._key(root)
            with (
                mock.patch.object(release, "SIGNER_PUBLIC_KEY", public),
                mock.patch.object(
                    release,
                    "SIGNER_KEY_FINGERPRINT",
                    fingerprint,
                ),
                self.assertRaisesRegex(
                    release.Wan22ReleaseError,
                    "not byte-identical",
                ),
            ):
                release.build_and_sign_release(
                    manifest_path=reordered,
                    finalizer_dir=finalizer,
                    selection_receipt_path=receipt,
                    output_path=root / "reordered-release.json",
                    signing_key=key,
                    release_id="reordered",
                    issued_at_utc="2026-07-31T12:00:00+00:00",
                )
            with self.assertRaisesRegex(
                release.Wan22ReleaseError,
                "exactly one evidence mode",
            ):
                release.build_and_sign_release(
                    manifest_path=original,
                    smoke_acceptance_path=acceptance,
                    finalizer_dir=finalizer,
                    selection_receipt_path=receipt,
                    output_path=root / "ambiguous-release.json",
                    signing_key=key,
                    release_id="ambiguous",
                    issued_at_utc="2026-07-31T12:00:00+00:00",
                )
            with self.assertRaisesRegex(
                release.Wan22ReleaseError,
                "requires both finalizer directory and selection receipt",
            ):
                release.build_and_sign_release(
                    manifest_path=original,
                    finalizer_dir=finalizer,
                    output_path=root / "missing-receipt-release.json",
                    signing_key=key,
                    release_id="missing-receipt",
                    issued_at_utc="2026-07-31T12:00:00+00:00",
                )

    def test_finalizer_selection_external_closure_tamper_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            signed, manifest, finalizer, _receipt, _rows = (
                self._signed_finalizer(root)
            )
            done_path = finalizer / selector.DONE_NAME
            done = json.loads(done_path.read_text(encoding="utf-8"))
            done["status"] = "incomplete"
            done_path.chmod(0o600)
            _write_json(done_path, done)
            with self.assertRaisesRegex(
                release.Wan22ReleaseError,
                "selector rerun failed",
            ):
                self._verify(signed, manifest)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            signed, manifest, _finalizer, receipt, _rows = (
                self._signed_finalizer(root)
            )
            receipt.write_bytes(receipt.read_bytes() + b" ")
            with self.assertRaisesRegex(
                release.Wan22ReleaseError,
                "receipt is not byte-identical",
            ):
                self._verify(signed, manifest)

    def test_finalizer_prepared_request_binds_selector_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _unused_manifest, _unused_acceptance, rows = self._fixture(root)
            finalizer, manifest, receipt = self._finalizer_fixture(root, rows)
            key, public, fingerprint = self._key(root)
            challenge = "e" * 64
            request_path = root / "finalizer-request.json"
            with (
                mock.patch.object(release, "SIGNER_PUBLIC_KEY", public),
                mock.patch.object(
                    release,
                    "SIGNER_KEY_FINGERPRINT",
                    fingerprint,
                ),
            ):
                request = release.prepare_release_request(
                    manifest_path=manifest,
                    finalizer_dir=finalizer,
                    selection_receipt_path=receipt,
                    request_path=request_path,
                    release_id="prepared-finalizer-request",
                    issued_at_utc="2026-07-31T12:00:00+00:00",
                    challenge=challenge,
                )
                output = root / "prepared-finalizer-release.json"
                release.sign_prepared_request(
                    request_path=request_path,
                    output_path=output,
                    signing_key=key,
                    expected_challenge=challenge,
                )
                verified = release.verify_signed_release(
                    release_path=output,
                    manifest_path=manifest,
                    require_exact_manifest=True,
                )
            self.assertEqual(verified["selected_row_count"], 8)
            self.assertEqual(
                set(request["builder"]),
                {
                    "release_module_sha256",
                    "selector_implementation_sha256",
                    "finalizer_implementation_sha256",
                },
            )

            tampered = json.loads(request_path.read_text(encoding="utf-8"))
            tampered["builder"]["selector_implementation_sha256"] = "0" * 64
            del tampered["request_digest"]
            tampered["request_digest"] = release._object_digest(tampered)
            tampered_path = root / "selector-source-tamper-request.json"
            _write_json(tampered_path, tampered)
            with (
                mock.patch.object(release, "SIGNER_PUBLIC_KEY", public),
                mock.patch.object(
                    release,
                    "SIGNER_KEY_FINGERPRINT",
                    fingerprint,
                ),
                self.assertRaisesRegex(
                    release.Wan22ReleaseError,
                    "remote/local exact-eight selector SHA differs",
                ),
            ):
                release.sign_prepared_request(
                    request_path=tampered_path,
                    output_path=root / "selector-source-tamper-release.json",
                    signing_key=key,
                    expected_challenge=challenge,
                )

    def test_media_tamper_and_non_eight_manifest_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            signed, manifest, rows = self._signed(root)
            (root / "source-0.mp4").write_bytes(b"tampered")
            with self.assertRaises(release.Wan22ReleaseError):
                self._verify(signed, manifest)
            manifest.write_bytes(
                b"".join(
                    release._canonical_bytes(row) + b"\n" for row in rows[:7]
                )
            )
            with self.assertRaises(release.Wan22ReleaseError):
                self._verify(signed, manifest)

    def test_signature_payload_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            signed, manifest, _ = self._signed(root)
            envelope = json.loads(signed.read_text())
            envelope["signed"]["prompt_policy"][
                "writer_proposals_executable"
            ] = True
            signed.chmod(0o600)
            _write_json(signed, envelope)
            with self.assertRaisesRegex(
                release.Wan22ReleaseError,
                "signature verification failed",
            ):
                self._verify(signed, manifest)


if __name__ == "__main__":
    unittest.main()
