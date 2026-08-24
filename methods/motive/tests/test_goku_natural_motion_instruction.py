from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from motive import goku_natural_motion_instruction as natural


def _digest_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _passed(iid: str = "sample01", *, subjects: int = 1) -> dict:
    old = (
        "Starting from the exact first frame: from frame 0 to frame 40 the dog "
        "picks up a bone, then from frame 40 to frame 80 it stands and walks right."
    )
    dynamic = []
    targets = []
    ids = []
    for index in range(subjects):
        subject_id = f"subject_{index + 1:02d}"
        ids.append(subject_id)
        dynamic.append(
            {
                "subject_id": subject_id,
                "dynamic": True,
                "stable_reference": "the seated dog beside the bone",
                "i0_state": "a seated brown dog beside a bone",
                "source_motion": "sits and turns its head",
            }
        )
        targets.append(
            {
                "subject_id": subject_id,
                "substantive_change": True,
                "target_motion": (
                    "from frame 0 to frame 40, pick up the bone; from frame 40 "
                    "to frame 80, stand and walk right while carrying it"
                ),
            }
        )
    return {
        "schema_version": natural.EXPECTED_PASSED_SCHEMA,
        "iid": iid,
        "action_change_substantive": True,
        "all_dynamic_subjects_covered": True,
        "camera_covered": True,
        "edit_instruction": old,
        "edit_instruction_sha256": _digest_text(old),
        "compiled_instruction": {"instruction": old},
        "source_census": {
            "iid": iid,
            "dynamic_subjects": dynamic,
            "camera": {"motion_class": "locked_off", "source_motion": "fixed"},
        },
        "target_plan": {
            "iid": iid,
            "dynamic_subject_targets": targets,
            "coverage": {"dynamic_subject_ids": ids, "camera_covered": True},
            "camera_target": {
                "motion_class": "locked_off",
                "relation": "preserve_static",
                "target_motion": "camera remains locked off",
            },
        },
        "resolved_source_video": f"/data/{iid}/source.mp4",
        "source_video_sha256": "a" * 64,
    }


def _input_row(iid: str = "sample01", *, subjects: int = 1) -> dict:
    value = {
        "schema_version": natural.INPUT_SCHEMA,
        "iid": iid,
        "original_candidate_index": 7,
        "candidates_path": "/data/candidates.jsonl",
        "candidates_sha256": "b" * 64,
        "source_passed_path": f"/data/passed/{iid}.jsonl",
        "source_passed_sha256": "c" * 64,
        "passed_row": _passed(iid, subjects=subjects),
        "row_digest": None,
    }
    value["row_digest"] = natural._object_digest(value, omit="row_digest")
    return value


def _rewrite(iid: str = "sample01", *, subjects: int = 1) -> dict:
    mappings = []
    for index in range(subjects):
        mappings.append(
            {
                "schema_version": natural.SUBJECT_MAPPING_SCHEMA,
                "subject_id": f"subject_{index + 1:02d}",
                "natural_reference": "the seated brown dog beside the bone",
                "target_motion_summary": (
                    "pick up the bone, stand, and walk right while carrying it"
                ),
            }
        )
    return {
        "schema_version": natural.REWRITE_SCHEMA,
        "iid": iid,
        "action_instruction": (
            "Have the seated brown dog pick up the bone beside it, then stand and "
            "walk to the right while carrying it."
        ),
        "subject_mappings": mappings,
        "camera_instruction": "Keep the camera fixed.",
        "preservation_instruction": (
            "Keep the dog's appearance and the surrounding scene unchanged."
        ),
    }


def _audit(iid: str = "sample01", *, subjects: int = 1) -> dict:
    subject_audits = []
    for index in range(subjects):
        subject_audits.append(
            {
                "schema_version": natural.SUBJECT_AUDIT_SCHEMA,
                "subject_id": f"subject_{index + 1:02d}",
                "explicitly_grounded": True,
                "core_events_entailed": True,
                "no_extra_event": True,
                "direction_path_match": True,
                "object_role_match": True,
                "order_match": True,
                "concurrency_match": True,
                "substantive_vs_source": True,
            }
        )
    return {
        "schema_version": natural.AUDIT_SCHEMA,
        "iid": iid,
        "subject_audits": subject_audits,
        "camera_audit": {
            "schema_version": natural.CAMERA_AUDIT_SCHEMA,
            "explicit": True,
            "class_match": True,
            "direction_match": True,
            "no_contradiction": True,
        },
        "absolute_timing_absent": True,
        "source_future_dependency_absent": True,
        "appearance_content_preserved": True,
        "natural_imperative": True,
        "overall_verdict": "pass",
        "reason_codes": [],
        "confidence": "high",
    }


def _compiled_instruction(action: str | None = None) -> str:
    return (
        action
        or (
            "Have the seated brown dog pick up the bone beside it, then stand and "
            "walk to the right while carrying it."
        )
    ) + " Keep the camera fixed. " + natural.CANONICAL_PRESERVATION_INSTRUCTION


class FakeBackend:
    model_path = "/models/Qwen3-VL-32B-Instruct"
    model_revision = "test-revision"
    transformers_version = "test"
    mode = "text"

    def __init__(self, responses: list[dict]):
        self.responses = [json.dumps(item) for item in responses]
        self.calls = 0
        self.requests: list[dict[str, str]] = []

    def generate_text(self, *, system: str, user: str) -> str:
        self.requests.append({"system": system, "user": user})
        result = self.responses[self.calls]
        self.calls += 1
        return result


def _write_manifest(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_temporal_grid_rejected_but_spatial_frame_is_allowed() -> None:
    okay = _rewrite()
    okay["action_instruction"] = "Have the dog pick up the bone, run right, and exit the frame."
    validated = natural._validate_rewrite(
        okay, iid="sample01", subject_ids=["subject_01"], camera_class="locked_off"
    )
    assert "exit the frame" in validated["edit_instruction"]

    bad = _rewrite()
    bad["action_instruction"] = "Have the dog crouch from frame 0 to frame 20, then run right."
    with pytest.raises(natural.NaturalMotionInstructionError, match="absolute_timing"):
        natural._validate_rewrite(
            bad, iid="sample01", subject_ids=["subject_01"], camera_class="locked_off"
        )


@pytest.mark.parametrize(
    "bad_text",
    [
        "Have the dog repeat whatever happens later in the source video.",
        "Starting from I0, have subject_01 perform this complete target motion.",
        "Have the dog run during the first 0.8 seconds.",
        "Have the dog run at 00:01.250, then stop.",
        "Have the dog run for 24 frames at 24 FPS, then stop.",
        "Have the dog run in the opening third of the clip.",
    ],
)
def test_future_grid_and_compiler_forms_are_rejected(bad_text: str) -> None:
    with pytest.raises(natural.NaturalMotionInstructionError):
        natural._validate_natural_text(bad_text, "candidate", imperative=True)


def test_audit_requires_exact_subject_coverage() -> None:
    value = _audit(subjects=2)
    value["subject_audits"] = value["subject_audits"][:1]
    with pytest.raises(natural.NaturalMotionInstructionError, match="subject count"):
        natural._validate_audit(
            value,
            iid="sample01",
            subject_ids=["subject_01", "subject_02"],
            instruction=_compiled_instruction(),
        )


def test_model_timing_false_does_not_override_clean_deterministic_scan() -> None:
    instruction = _compiled_instruction(
        "Have the motorcyclist turn sharply to the right to complete a U-turn, "
        "then reverse straight back along the road and exit the frame on the right."
    )
    value = _audit()
    value["absolute_timing_absent"] = False
    value["overall_verdict"] = "fail"
    value["reason_codes"] = ["absolute_timing_present", "subject_motion_mismatch"]
    value["confidence"] = "low"

    accepted = natural._validate_audit(
        value,
        iid="sample01",
        subject_ids=["subject_01"],
        instruction=instruction,
    )

    assert accepted["effective_verdict"] == "pass"
    assert accepted["deterministic_gates"]["absolute_timing_absent"] is True
    assert (
        accepted["model_reported_diagnostics"]["absolute_timing_absent"] is False
    )
    assert accepted["model_reported_diagnostics"]["reason_codes"] == [
        "absolute_timing_present",
        "subject_motion_mismatch",
    ]
    assert accepted["model_reported_diagnostics"]["confidence"] == "low"
    disagreements = {
        item["field"]: item for item in accepted["model_effective_disagreements"]
    }
    assert disagreements["absolute_timing_absent"]["effective"] is True
    assert disagreements["overall_verdict"]["effective"] == "pass"
    assert disagreements["reason_codes"]["effective"] == []
    assert disagreements["confidence"]["model_reported"] == "low"
    assert accepted["aggregate_override_applied"] is True


def test_semantic_subject_failure_still_rejects_despite_clean_format() -> None:
    value = _audit()
    value["subject_audits"][0]["core_events_entailed"] = False
    value["absolute_timing_absent"] = False
    value["overall_verdict"] = "fail"
    value["reason_codes"] = ["subject_motion_mismatch"]
    with pytest.raises(natural.NaturalMotionInstructionError, match="rejected subject"):
        natural._validate_audit(
            value,
            iid="sample01",
            subject_ids=["subject_01"],
            instruction=_compiled_instruction(),
        )


def test_unexplained_aggregate_semantic_failure_is_not_normalized_away() -> None:
    value = _audit()
    value["overall_verdict"] = "fail"
    value["reason_codes"] = ["subject_motion_mismatch"]
    with pytest.raises(
        natural.NaturalMotionInstructionError,
        match="no deterministic-format disagreement",
    ):
        natural._validate_audit(
            value,
            iid="sample01",
            subject_ids=["subject_01"],
            instruction=_compiled_instruction(),
        )


@pytest.mark.parametrize(
    "field", ["source_future_dependency_absent", "natural_imperative"]
)
def test_non_timing_model_guards_are_not_overridden(field: str) -> None:
    value = _audit()
    value[field] = False
    value["overall_verdict"] = "fail"
    value["reason_codes"] = [field]
    with pytest.raises(natural.NaturalMotionInstructionError, match=field):
        natural._validate_audit(
            value,
            iid="sample01",
            subject_ids=["subject_01"],
            instruction=_compiled_instruction(),
        )


def test_low_confidence_is_normalized_only_in_deterministic_conflict_branch() -> None:
    value = _audit()
    value["absolute_timing_absent"] = False
    value["overall_verdict"] = "fail"
    value["reason_codes"] = ["absolute_timing_present"]
    value["confidence"] = "low"
    accepted = natural._validate_audit(
        value,
        iid="sample01",
        subject_ids=["subject_01"],
        instruction=_compiled_instruction(),
    )
    disagreements = {
        item["field"]: item for item in accepted["model_effective_disagreements"]
    }
    assert disagreements["confidence"]["model_reported"] == "low"

    ordinary = _audit()
    ordinary["confidence"] = "low"
    with pytest.raises(
        natural.NaturalMotionInstructionError,
        match="no deterministic-format disagreement",
    ):
        natural._validate_audit(
            ordinary,
            iid="sample01",
            subject_ids=["subject_01"],
            instruction=_compiled_instruction(),
        )


def test_audit_prompt_scopes_timing_to_candidate_and_source_contrast_is_not_required() -> None:
    prompt = natural._audit_prompt(_input_row(), _compiled_instruction())
    assert "applies ONLY to candidate_natural_edit_instruction" in prompt
    assert "must not restate source motion" in prompt
    assert "already had its absolute execution grid removed" in prompt
    assert '"core_events_entailed": null' in prompt
    assert "MUST be replaced by the JSON boolean true or false" in prompt
    assert "natural_imperative MUST be exactly true or false" in prompt
    assert "Never copy the candidate instruction into natural_imperative" in prompt

    payload = natural._audit_payload(_input_row(), _compiled_instruction())
    serialized = json.dumps(payload, sort_keys=True)
    assert "motion_evidence" not in serialized
    assert "from frame 0" not in serialized
    assert "frame 40" not in serialized
    motion = payload["target_semantics"]["dynamic_subject_targets"][0][
        "ordered_target_motion_without_timing_grid"
    ]
    assert "pick up the bone" in motion and "stand and walk right" in motion


def test_audit_reference_strips_only_timing_not_spatial_kinematics() -> None:
    text = natural._strip_reference_timing(
        "from frame 0 to frame 20, turn right by 15 degrees; for 24 frames, "
        "reverse and exit the frame",
        context="regression",
    )
    assert "frame 0" not in text and "24 frames" not in text
    assert "15 degrees" in text
    assert "exit the frame" in text


def test_worker_publishes_natural_sidecar_and_resume_skips_backend(tmp_path: Path) -> None:
    row = _input_row()
    manifest = tmp_path / "input.jsonl"
    _write_manifest(manifest, [row])
    output = tmp_path / "output"
    backend = FakeBackend([_rewrite(), _audit()])

    args = argparse.Namespace(
        input=manifest,
        output_root=output,
        model="/models/Qwen3-VL-32B-Instruct",
        worker_index=0,
        num_workers=1,
        num_rows=1,
        max_new_tokens=2048,
        max_attempts=2,
        attn_implementation="sdpa",
        allow_download=False,
        allow_errors=False,
        skip_source_revalidation=True,
    )
    assert natural.run_worker(args, backend_factory=lambda **_: backend) == 0
    assert backend.calls == 2
    sidecar = output / "instructions" / "sample01" / "natural_edit_instruction.txt"
    assert "frame 0" not in sidecar.read_text()
    assert "pick up the bone" in sidecar.read_text()

    def forbidden_backend(**_: object) -> object:
        raise AssertionError("resume must not load Qwen")

    assert natural.run_worker(args, backend_factory=forbidden_backend) == 0

    verify = argparse.Namespace(
        input=manifest,
        output_root=output,
        expected_rows=1,
        min_ok=1,
        manifest_output=tmp_path / "natural_manifest.jsonl",
        summary_output=tmp_path / "summary.json",
        skip_source_revalidation=True,
    )
    assert natural.verify_outputs(verify) == 0
    dataset = json.loads((tmp_path / "natural_manifest.jsonl").read_text())
    assert dataset["generation_prompt"].startswith("Starting from the exact first frame")
    assert dataset["natural_edit_instruction"].startswith("Have the seated brown dog")


def test_malformed_audit_boolean_retries_auditor_with_feedback_and_preserves_raw(
    tmp_path: Path,
) -> None:
    row = _input_row()
    manifest = tmp_path / "input.jsonl"
    _write_manifest(manifest, [row])
    malformed = _audit()
    malformed["natural_imperative"] = _compiled_instruction()
    backend = FakeBackend([_rewrite(), malformed, _rewrite(), _audit()])
    args = argparse.Namespace(
        input=manifest,
        output_root=tmp_path / "output",
        model="model",
        worker_index=0,
        num_workers=1,
        num_rows=1,
        max_new_tokens=2048,
        max_attempts=2,
        attn_implementation="sdpa",
        allow_download=False,
        allow_errors=False,
        skip_source_revalidation=True,
    )

    assert natural.run_worker(args, backend_factory=lambda **_: backend) == 0
    assert backend.calls == 4
    result = json.loads(
        (args.output_root / "rows" / "sample01" / "result.json").read_text()
    )
    assert result["status"] == "ok"
    assert len(result["attempts"]) == 2
    assert result["attempts"][0]["status"] == "error"
    assert result["attempts"][0]["error"] == (
        "audit natural_imperative is not boolean"
    )
    first_audit_raw = json.loads(result["attempts"][0]["audit_raw"])
    assert first_audit_raw["natural_imperative"] == _compiled_instruction()
    assert result["attempts"][1]["status"] == "ok"

    audit_prompts = [
        request["user"]
        for request in backend.requests
        if request["system"] == natural.AUDIT_SYSTEM
    ]
    assert len(audit_prompts) == 2
    assert "PREVIOUS AUDIT OUTPUT FAILED VALIDATION" not in audit_prompts[0]
    assert "PREVIOUS AUDIT OUTPUT FAILED VALIDATION" in audit_prompts[1]
    assert "audit natural_imperative is not boolean" in audit_prompts[1]
    assert "natural_imperative MUST be exactly true or false" in audit_prompts[1]


def test_semantic_failure_retries_then_records_error(tmp_path: Path) -> None:
    row = _input_row()
    manifest = tmp_path / "input.jsonl"
    _write_manifest(manifest, [row])
    failed = _audit()
    failed["subject_audits"][0]["order_match"] = False
    failed["overall_verdict"] = "fail"
    failed["reason_codes"] = ["event_order_differs"]
    backend = FakeBackend([_rewrite(), failed, _rewrite(), failed])
    args = argparse.Namespace(
        input=manifest,
        output_root=tmp_path / "output",
        model="model",
        worker_index=0,
        num_workers=1,
        num_rows=1,
        max_new_tokens=2048,
        max_attempts=2,
        attn_implementation="sdpa",
        allow_download=False,
        allow_errors=True,
        skip_source_revalidation=True,
    )
    assert natural.run_worker(args, backend_factory=lambda **_: backend) == 0
    result = json.loads((args.output_root / "rows" / "sample01" / "result.json").read_text())
    assert result["status"] == "error"
    assert len(result["attempts"]) == 2
    assert not (args.output_root / "instructions" / "sample01").exists()
