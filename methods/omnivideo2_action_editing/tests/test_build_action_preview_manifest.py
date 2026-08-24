from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import build_action_preview_manifest as builder  # noqa: E402


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(builder.canonical_json_bytes(value) + b"\n")


def _self_digest(value: dict, field: str) -> dict:
    result = copy.deepcopy(value)
    result[field] = builder._object_sha256(result)
    return result


class PreviewFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.qwen_dir = root / "qwen" / "passed"
        self.wan_root = root / "wan"
        self.natural_root = root / "natural"
        self.qwen_dir.mkdir(parents=True)
        self.wan_root.mkdir()
        self.natural_root.mkdir()

    def qwen_row(
        self,
        iid: str,
        *,
        subjects: int = 1,
        source_camera: str = "locked_off",
        target_camera: str = "locked_off",
        camera_relation: str = "preserve_static",
        source_confidence: str = "high",
        target_confidence: str = "high",
    ) -> tuple[Path, dict]:
        media = self.root / "input_media" / iid
        media.mkdir(parents=True)
        source = media / "source.mp4"
        anchor = media / "anchor.png"
        source.write_bytes((f"source-{iid}").encode())
        anchor.write_bytes((f"anchor-{iid}").encode())
        dynamic_subjects = []
        targets = []
        for index in range(subjects):
            subject_id = f"subject_{index + 1:02d}"
            dynamic_subjects.append(
                {
                    "subject_id": subject_id,
                    "dynamic": True,
                    "stable_reference": f"actor {index + 1}",
                    "source_motion": "walks left",
                }
            )
            targets.append(
                {
                    "subject_id": subject_id,
                    "substantive_change": True,
                    "target_motion": "turns and crouches",
                }
            )
        census = {
            "iid": iid,
            "dynamic_subjects": dynamic_subjects,
            "camera": {"motion_class": source_camera},
            "confidence": source_confidence,
        }
        plan = {
            "iid": iid,
            "dynamic_subject_targets": targets,
            "camera_target": {
                "motion_class": target_camera,
                "relation": camera_relation,
            },
            "confidence": target_confidence,
        }
        instruction = f"Starting at frame zero, make actor {iid} crouch."
        instruction_sha = _sha(instruction.encode())
        compiled = {
            "iid": iid,
            "instruction": instruction,
            "instruction_sha256": instruction_sha,
            "source_census_sha256": builder._object_sha256(census),
            "target_plan_sha256": builder._object_sha256(plan),
        }
        row = {
            "schema_version": builder.QWEN_PASSED_SCHEMA,
            "iid": iid,
            "group_id": f"group-{iid}",
            "family": "turn",
            "resolved_source_video": str(source),
            "resolved_anchor_image": str(anchor),
            "source_video_sha256": _sha(source.read_bytes()),
            "anchor_sha256": _sha(anchor.read_bytes()),
            "edit_instruction": instruction,
            "edit_instruction_sha256": instruction_sha,
            "source_census": census,
            "target_plan": plan,
            "compiled_instruction": compiled,
            "qwen_record_digest": "a" * 64,
            "action_change_substantive": True,
            "all_dynamic_subjects_covered": True,
            "camera_covered": True,
            "human_review_status": "pending",
            "generation_authorized": False,
            "production_eligible": False,
        }
        path = self.qwen_dir / f"{iid}.jsonl"
        _write_jsonl(path, row)
        return path, row

    def wan_commit(self, row: dict, *, production_eligible: bool = False) -> dict:
        iid = row["iid"]
        wrapper = self.wan_root / "samples" / iid
        sample = wrapper / "samples" / iid
        sample.mkdir(parents=True)
        source = sample / "source_video.mp4"
        target = sample / "preview.mp4"
        instruction_file = sample / "edit_instruction.txt"
        anchor = sample / "conditioning_anchor_original.png"
        frame0_npy = sample / "conditioning_frame0_float32.npy"
        frame0_png = sample / "conditioning_frame0.png"
        original_source = Path(row["resolved_source_video"])
        original_anchor = Path(row["resolved_anchor_image"])
        source.write_bytes(original_source.read_bytes())
        target.write_bytes((f"target-{iid}").encode())
        instruction_file.write_bytes(row["edit_instruction"].encode())
        anchor.write_bytes(original_anchor.read_bytes())
        frame0_npy.write_bytes((f"npy-{iid}").encode())
        frame0_png.write_bytes((f"png-{iid}").encode())

        contract = _self_digest(
            {
                "schema_version": "fixture-wan-contract-v1",
                "production_use_forbidden": True,
                "authorization": {
                    "generation_authorized": False,
                    "human_review_status": "pending",
                },
            },
            "contract_digest",
        )
        _write_json(wrapper / "run_contract.json", contract)
        outputs = {
            "source_video": source.name,
            "source_video_sha256": _sha(source.read_bytes()),
            "source_video_bytes": source.stat().st_size,
            "preview_mp4": target.name,
            "preview_mp4_sha256": _sha(target.read_bytes()),
            "preview_mp4_bytes": target.stat().st_size,
            "edit_instruction_file": instruction_file.name,
            "edit_instruction_file_sha256": _sha(instruction_file.read_bytes()),
            "edit_instruction_file_bytes": instruction_file.stat().st_size,
            "conditioning_anchor_original": anchor.name,
            "conditioning_anchor_original_sha256": _sha(anchor.read_bytes()),
            "conditioning_frame0_float32": frame0_npy.name,
            "conditioning_frame0_float32_sha256": _sha(frame0_npy.read_bytes()),
            "conditioning_frame0_png": frame0_png.name,
            "conditioning_frame0_png_sha256": _sha(frame0_png.read_bytes()),
        }
        result = _self_digest(
            {
                "schema_version": builder.WAN_RESULT_SCHEMA,
                "iid": iid,
                "contract_digest": contract["contract_digest"],
                "outputs": outputs,
                "prompt": {
                    "field": "edit_instruction",
                    "text": row["edit_instruction"],
                    "sha256": row["edit_instruction_sha256"],
                },
                "production_eligible": False,
                "generation_authorized_in_manifest": False,
                "human_review_status_at_generation": "pending",
                "production_use_forbidden": True,
            },
            "result_digest",
        )
        result_path = sample / "result.json"
        _write_json(result_path, result)
        generated = {
            "schema_version": builder.WAN_GENERATED_SCHEMA,
            "iid": iid,
            "group_id": row["group_id"],
            "edit_instruction": row["edit_instruction"],
            "edit_instruction_sha256": row["edit_instruction_sha256"],
            "edit_instruction_file": str(instruction_file),
            "edit_instruction_file_sha256": outputs["edit_instruction_file_sha256"],
            "edit_instruction_file_bytes": outputs["edit_instruction_file_bytes"],
            "source_video": str(source),
            "source_video_sha256": outputs["source_video_sha256"],
            "source_video_bytes": outputs["source_video_bytes"],
            "target_preview_mp4": str(target),
            "target_preview_mp4_sha256": outputs["preview_mp4_sha256"],
            "target_preview_mp4_bytes": outputs["preview_mp4_bytes"],
            "conditioning_anchor_original": str(anchor),
            "conditioning_anchor_original_sha256": outputs[
                "conditioning_anchor_original_sha256"
            ],
            "conditioning_frame0_float32": str(frame0_npy),
            "conditioning_frame0_float32_sha256": outputs[
                "conditioning_frame0_float32_sha256"
            ],
            "conditioning_frame0_png": str(frame0_png),
            "conditioning_frame0_png_sha256": outputs[
                "conditioning_frame0_png_sha256"
            ],
            "result_json": str(result_path),
            "result_digest": result["result_digest"],
            "preview_bindings": {
                "iid": iid,
                "source_census_sha256": builder._object_sha256(row["source_census"]),
                "target_plan_sha256": builder._object_sha256(row["target_plan"]),
                "qwen_record_digest": row["qwen_record_digest"],
            },
            "production_eligible": production_eligible,
            "generation_authorized": False,
            "human_review_status": "pending",
            "production_use_forbidden": True,
        }
        generated_path = wrapper / "generated_manifest.jsonl"
        _write_jsonl(generated_path, generated)
        complete = _self_digest(
            {
                "schema_version": builder.WAN_COMPLETE_SCHEMA,
                "contract_digest": contract["contract_digest"],
                "selected_sample_count": 1,
                "completed_sample_count": 1,
                "generated_manifest": generated_path.name,
                "generated_manifest_sha256": _sha(generated_path.read_bytes()),
                "sample_result_digests": [result["result_digest"]],
            },
            "complete_digest",
        )
        _write_json(wrapper / "run_complete.json", complete)
        return {
            "target": target,
            "generated": generated,
            "generated_path": generated_path,
            "complete": complete,
        }

    def natural_commit(self, qwen_path: Path, row: dict) -> str:
        iid = row["iid"]
        instruction = (
            f"Have actor {iid} turn around and crouch gradually. Keep the camera "
            "locked off. Keep all identities and appearances intact, and leave the "
            "rest of the scene unchanged except for the physical consequences of "
            "these actions."
        )
        instruction_path = (
            self.natural_root / "instructions" / iid / "natural_edit_instruction.txt"
        )
        instruction_path.parent.mkdir(parents=True)
        instruction_path.write_text(instruction + "\n", encoding="utf-8")
        result = _self_digest(
            {
                "schema_version": builder.NATURAL_RESULT_SCHEMA,
                "iid": iid,
                "status": "ok",
                "source_passed_path": str(qwen_path),
                "source_passed_sha256": _sha(qwen_path.read_bytes()),
                "generation_prompt": row["edit_instruction"],
                "generation_prompt_sha256": row["edit_instruction_sha256"],
                "source_census_sha256": builder._object_sha256(row["source_census"]),
                "target_plan_sha256": builder._object_sha256(row["target_plan"]),
                "natural_edit_instruction": instruction,
                "natural_edit_instruction_sha256": _sha(instruction.encode()),
                "audit": {
                    "effective_verdict": "pass",
                    "model_reported_diagnostics": {"confidence": "high"},
                },
            },
            "record_digest",
        )
        result_path = self.natural_root / "rows" / iid / "result.json"
        _write_json(result_path, result)
        receipt = _self_digest(
            {
                "schema_version": builder.NATURAL_RECEIPT_SCHEMA,
                "iid": iid,
                "status": "ok",
                "result_path": str(result_path),
                "result_sha256": _sha(result_path.read_bytes()),
                "instruction_path": str(instruction_path),
                "instruction_sha256": _sha(instruction_path.read_bytes()),
            },
            "receipt_digest",
        )
        _write_json(self.natural_root / "terminal" / f"{iid}.receipt.json", receipt)
        return instruction

    def publish_natural_release(self, qwen_path: Path, row: dict) -> None:
        iid = row["iid"]
        result_path = self.natural_root / "rows" / iid / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        dataset_row = {
            "schema_version": builder.NATURAL_DATASET_ROW_SCHEMA,
            "iid": iid,
            "label_status": (
                "structured_plan_semantic_audit_passed_video_audit_pending"
            ),
            "natural_edit_instruction": result["natural_edit_instruction"],
            "natural_edit_instruction_sha256": result[
                "natural_edit_instruction_sha256"
            ],
            "source_passed_sha256": _sha(qwen_path.read_bytes()),
            "result_path": str(result_path),
            "result_sha256": _sha(result_path.read_bytes()),
            "semantic_audit": {
                "effective_verdict": "pass",
                "model_reported_diagnostics": {"confidence": "high"},
            },
        }
        manifest = self.natural_root / "natural_edit_instruction_manifest.jsonl"
        _write_jsonl(manifest, dataset_row)
        summary = {
            "schema_version": builder.NATURAL_VERIFY_SUMMARY_SCHEMA,
            "dataset_manifest_path": str(manifest),
            "dataset_manifest_sha256": _sha(manifest.read_bytes()),
            "expected_rows": 1,
            "terminal_rows": 1,
            "ok_rows": 1,
            "error_rows": 0,
        }
        summary["summary_digest"] = builder._object_sha256(summary)
        _write_json(self.natural_root / "verification_summary.json", summary)

    def natural_error_commit(self, qwen_path: Path, row: dict) -> None:
        iid = row["iid"]
        result = _self_digest(
            {
                "schema_version": builder.NATURAL_RESULT_SCHEMA,
                "iid": iid,
                "status": "error",
                "source_passed_path": str(qwen_path),
                "source_passed_sha256": _sha(qwen_path.read_bytes()),
                "generation_prompt": row["edit_instruction"],
                "generation_prompt_sha256": row["edit_instruction_sha256"],
                "source_census_sha256": builder._object_sha256(row["source_census"]),
                "target_plan_sha256": builder._object_sha256(row["target_plan"]),
                "natural_edit_instruction": None,
                "natural_edit_instruction_sha256": None,
                "audit": None,
                "error": {"type": "fixture", "message": "rewrite failed"},
            },
            "record_digest",
        )
        result_path = self.natural_root / "rows" / iid / "result.json"
        _write_json(result_path, result)
        receipt = _self_digest(
            {
                "schema_version": builder.NATURAL_RECEIPT_SCHEMA,
                "iid": iid,
                "status": "error",
                "result_path": str(result_path),
                "result_sha256": _sha(result_path.read_bytes()),
                "instruction_path": None,
                "instruction_sha256": None,
            },
            "receipt_digest",
        )
        _write_json(self.natural_root / "terminal" / f"{iid}.receipt.json", receipt)


class BuildActionPreviewManifestTests(unittest.TestCase):
    def _outputs(self, root: Path) -> tuple[Path, Path]:
        return root / "out" / "preview.jsonl", root / "out" / "summary.json"

    def test_structured_join_is_hash_verified_and_permanently_preview_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = PreviewFixture(root)
            _qwen_path, qwen = fixture.qwen_row("clip001")
            fixture.wan_commit(qwen)
            manifest, summary_path = self._outputs(root)
            summary = builder.build_preview_manifest(
                qwen_passed_dir=fixture.qwen_dir,
                wan_root=fixture.wan_root,
                instruction_source="structured",
                output_manifest=manifest,
                summary_output=summary_path,
            )
            rows = [json.loads(line) for line in manifest.read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            output = rows[0]
            self.assertEqual(output["edit_instruction"], qwen["edit_instruction"])
            self.assertEqual(output["instruction_source"], "structured")
            self.assertTrue(output["preview_only"])
            self.assertFalse(output["training_authorized"])
            self.assertTrue(output["training_use_forbidden"])
            self.assertFalse(output["production_eligible"])
            self.assertEqual(output["post_video_acceptance"], "pending")
            digest_candidate = dict(output)
            digest_candidate.pop("row_digest")
            self.assertEqual(output["row_digest"], builder._object_sha256(digest_candidate))
            self.assertEqual(summary["preview_rows"], 1)
            self.assertFalse(summary["training_authorized"])
            self.assertEqual(
                summary["output_manifest_sha256"], _sha(manifest.read_bytes())
            )
            with self.assertRaisesRegex(builder.PreviewManifestError, "create-only"):
                builder.build_preview_manifest(
                    qwen_passed_dir=fixture.qwen_dir,
                    wan_root=fixture.wan_root,
                    instruction_source="structured",
                    output_manifest=manifest,
                    summary_output=summary_path,
                )

    def test_selection_requires_single_actor_locked_preserved_camera_and_high_confidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = PreviewFixture(root)
            _path, accepted = fixture.qwen_row("accepted")
            fixture.wan_commit(accepted)
            fixture.qwen_row("twoactors", subjects=2)
            fixture.qwen_row("movingcam", source_camera="pan_left")
            fixture.qwen_row("replacecam", camera_relation="replace_motion")
            fixture.qwen_row("medium", target_confidence="medium")
            manifest, summary_path = self._outputs(root)
            summary = builder.build_preview_manifest(
                qwen_passed_dir=fixture.qwen_dir,
                wan_root=fixture.wan_root,
                instruction_source="structured",
                output_manifest=manifest,
                summary_output=summary_path,
            )
            self.assertEqual(summary["qwen_passed_rows"], 5)
            self.assertEqual(summary["selection_eligible_rows"], 1)
            self.assertEqual(summary["preview_rows"], 1)
            self.assertEqual(summary["gate_rejections"]["single_dynamic_actor"], 1)
            self.assertEqual(summary["gate_rejections"]["source_camera_locked_off"], 1)
            self.assertEqual(summary["gate_rejections"]["target_camera_preserve_static"], 1)
            self.assertEqual(summary["gate_rejections"]["target_plan_high_confidence"], 1)

    def test_natural_instruction_is_joined_through_receipt_and_qwen_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = PreviewFixture(root)
            qwen_path, qwen = fixture.qwen_row("natural01")
            fixture.wan_commit(qwen)
            natural = fixture.natural_commit(qwen_path, qwen)
            manifest, summary_path = self._outputs(root)
            summary = builder.build_preview_manifest(
                qwen_passed_dir=fixture.qwen_dir,
                wan_root=fixture.wan_root,
                natural_root=fixture.natural_root,
                instruction_source="natural",
                output_manifest=manifest,
                summary_output=summary_path,
            )
            output = json.loads(manifest.read_text().strip())
            self.assertEqual(output["instruction_source"], "natural")
            self.assertEqual(output["edit_instruction"], natural)
            self.assertEqual(output["generation_instruction"], qwen["edit_instruction"])
            self.assertIn("natural_receipt_sha256", output["provenance"])
            self.assertEqual(summary["preview_rows"], 1)

    def test_natural_release_policy_keeps_multi_actor_rows_as_broad_cohort(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = PreviewFixture(root)
            qwen_path, qwen = fixture.qwen_row("natural_multi", subjects=2)
            fixture.wan_commit(qwen)
            fixture.natural_commit(qwen_path, qwen)
            fixture.publish_natural_release(qwen_path, qwen)
            manifest, summary_path = self._outputs(root)
            summary = builder.build_preview_manifest(
                qwen_passed_dir=fixture.qwen_dir,
                wan_root=fixture.wan_root,
                natural_root=fixture.natural_root,
                instruction_source="natural",
                selection_policy=builder.SELECTION_POLICY_NATURAL_RELEASE,
                output_manifest=manifest,
                summary_output=summary_path,
            )
            output = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(summary["selection_eligible_rows"], 1)
            self.assertEqual(summary["strict_gate_eligible_rows"], 0)
            self.assertEqual(summary["preview_rows"], 1)
            self.assertFalse(output["selection_gates"]["single_dynamic_actor"])
            self.assertIn("natural_release_manifest_sha256", output["provenance"])

    def test_missing_background_natural_commit_is_skipped_not_promoted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = PreviewFixture(root)
            _qwen_path, qwen = fixture.qwen_row("waiting01")
            fixture.wan_commit(qwen)
            manifest, summary_path = self._outputs(root)
            summary = builder.build_preview_manifest(
                qwen_passed_dir=fixture.qwen_dir,
                wan_root=fixture.wan_root,
                natural_root=fixture.natural_root,
                instruction_source="natural",
                output_manifest=manifest,
                summary_output=summary_path,
            )
            self.assertEqual(manifest.read_bytes(), b"")
            self.assertEqual(summary["preview_rows"], 0)
            self.assertEqual(
                summary["skipped_rows"]["natural_commit_missing_or_in_progress"], 1
            )
            self.assertFalse(summary["training_authorized"])

    def test_verified_natural_error_receipt_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = PreviewFixture(root)
            qwen_path, qwen = fixture.qwen_row("naturalerror")
            fixture.wan_commit(qwen)
            fixture.natural_error_commit(qwen_path, qwen)
            manifest, summary_path = self._outputs(root)
            summary = builder.build_preview_manifest(
                qwen_passed_dir=fixture.qwen_dir,
                wan_root=fixture.wan_root,
                natural_root=fixture.natural_root,
                instruction_source="natural",
                output_manifest=manifest,
                summary_output=summary_path,
            )
            self.assertEqual(manifest.read_bytes(), b"")
            self.assertEqual(summary["skipped_rows"]["natural_terminal_error"], 1)

    def test_tampered_target_hash_fails_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = PreviewFixture(root)
            _qwen_path, qwen = fixture.qwen_row("tamper01")
            wan = fixture.wan_commit(qwen)
            wan["target"].write_bytes(b"tampered")
            manifest, summary_path = self._outputs(root)
            with self.assertRaisesRegex(builder.PreviewManifestError, "hash mismatch"):
                builder.build_preview_manifest(
                    qwen_passed_dir=fixture.qwen_dir,
                    wan_root=fixture.wan_root,
                    instruction_source="structured",
                    output_manifest=manifest,
                    summary_output=summary_path,
                )
            self.assertFalse(manifest.exists())
            self.assertFalse(summary_path.exists())

    def test_any_wan_production_claim_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = PreviewFixture(root)
            _qwen_path, qwen = fixture.qwen_row("claim01")
            fixture.wan_commit(qwen, production_eligible=True)
            manifest, summary_path = self._outputs(root)
            with self.assertRaisesRegex(builder.PreviewManifestError, "non-preview claim"):
                builder.build_preview_manifest(
                    qwen_passed_dir=fixture.qwen_dir,
                    wan_root=fixture.wan_root,
                    instruction_source="structured",
                    output_manifest=manifest,
                    summary_output=summary_path,
                )

    def test_outputs_cannot_be_written_inside_read_only_input_roots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = PreviewFixture(root)
            _qwen_path, qwen = fixture.qwen_row("readonly01")
            fixture.wan_commit(qwen)
            with self.assertRaisesRegex(builder.PreviewManifestError, "read-only input root"):
                builder.build_preview_manifest(
                    qwen_passed_dir=fixture.qwen_dir,
                    wan_root=fixture.wan_root,
                    instruction_source="structured",
                    output_manifest=fixture.wan_root / "forbidden.jsonl",
                    summary_output=root / "summary.json",
                )


if __name__ == "__main__":
    unittest.main()
