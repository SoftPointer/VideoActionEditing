#!/usr/bin/env python3
"""Dependency-light contract self-test for natural motion instructions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from motive import goku_natural_motion_instruction as natural


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _passed(iid: str = "selftest") -> dict:
    old = "Starting from the exact first frame: from frame 0 to frame 80, have the dog run."
    return {
        "schema_version": natural.EXPECTED_PASSED_SCHEMA,
        "iid": iid,
        "action_change_substantive": True,
        "all_dynamic_subjects_covered": True,
        "camera_covered": True,
        "edit_instruction": old,
        "edit_instruction_sha256": _sha(old),
        "compiled_instruction": {"instruction": old},
        "source_census": {
            "iid": iid,
            "dynamic_subjects": [{
                "subject_id": "subject_01",
                "dynamic": True,
                "stable_reference": "the seated brown dog beside the bone",
                "source_motion": "turns its head while seated",
            }],
            "camera": {"motion_class": "locked_off"},
        },
        "target_plan": {
            "iid": iid,
            "dynamic_subject_targets": [{
                "subject_id": "subject_01",
                "substantive_change": True,
                "target_motion": (
                    "from frame 0 to frame 40, pick up the bone; from frame 40 "
                    "to frame 80, stand and walk right while carrying it"
                ),
            }],
            "coverage": {"dynamic_subject_ids": ["subject_01"], "camera_covered": True},
            "camera_target": {
                "motion_class": "locked_off",
                "target_motion": "camera remains locked off",
            },
        },
        "resolved_source_video": "/data/selftest/source.mp4",
        "source_video_sha256": "a" * 64,
    }


def _input() -> dict:
    row = {
        "schema_version": natural.INPUT_SCHEMA,
        "iid": "selftest",
        "original_candidate_index": 3,
        "candidates_path": "/data/candidates.jsonl",
        "candidates_sha256": "b" * 64,
        "source_passed_path": "/data/passed/selftest.jsonl",
        "source_passed_sha256": "c" * 64,
        "passed_row": _passed(),
        "row_digest": None,
    }
    row["row_digest"] = natural._object_digest(row, omit="row_digest")
    return row


def _rewrite(action: str | None = None) -> dict:
    return {
        "schema_version": natural.REWRITE_SCHEMA,
        "iid": "selftest",
        "action_instruction": action or (
            "Have the seated brown dog pick up the bone beside it, then stand and "
            "walk to the right while carrying it."
        ),
        "subject_mappings": [{
            "schema_version": natural.SUBJECT_MAPPING_SCHEMA,
            "subject_id": "subject_01",
            "natural_reference": "the seated brown dog beside the bone",
            "target_motion_summary": "pick up the bone, stand, and walk right while carrying it",
        }],
        "camera_instruction": "Keep the camera fixed.",
        "preservation_instruction": "Preserve the dog's appearance and surrounding scene.",
    }


def _audit(*, pass_value: bool = True) -> dict:
    return {
        "schema_version": natural.AUDIT_SCHEMA,
        "iid": "selftest",
        "subject_audits": [{
            "schema_version": natural.SUBJECT_AUDIT_SCHEMA,
            "subject_id": "subject_01",
            "explicitly_grounded": True,
            "core_events_entailed": True,
            "no_extra_event": True,
            "direction_path_match": True,
            "object_role_match": True,
            "order_match": pass_value,
            "concurrency_match": True,
            "substantive_vs_source": True,
        }],
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
        "overall_verdict": "pass" if pass_value else "fail",
        "reason_codes": [] if pass_value else ["event_order_differs"],
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


class _Backend:
    model_path = "/models/Qwen3-VL-32B-Instruct"
    model_revision = "selftest"
    transformers_version = "selftest"
    mode = "text"

    def __init__(self, values: list[dict]):
        self.values = [json.dumps(value) for value in values]
        self.calls = 0
        self.requests: list[dict[str, str]] = []

    def generate_text(self, *, system: str, user: str) -> str:
        self.requests.append({"system": system, "user": user})
        value = self.values[self.calls]
        self.calls += 1
        return value


def _expect_rejected(text: str) -> None:
    try:
        natural._validate_natural_text(text, "redteam", imperative=True)
    except natural.NaturalMotionInstructionError:
        return
    raise AssertionError(f"red-team string was accepted: {text}")


def main() -> int:
    spatial = natural._validate_rewrite(
        _rewrite("Have the dog pick up the bone, run right, and exit the frame."),
        iid="selftest",
        subject_ids=["subject_01"],
        camera_class="locked_off",
    )
    assert "exit the frame" in spatial["edit_instruction"]
    assert "relative positions" not in spatial["edit_instruction"]
    assert natural.CANONICAL_PRESERVATION_INSTRUCTION in spatial["edit_instruction"]
    for bad in (
        "The motorcyclist turns right and reverses away from the camera.",
        "Have the dog crouch from frame 0 to frame 20, then run.",
        "Have the dog run during the first 0.8 seconds.",
        "Have the dog run at 00:01.250, then stop.",
        "Have the dog run for 24 frames at 24 FPS, then stop.",
        "Have the dog run in the opening third of the clip.",
        "Have the dog repeat whatever happens later in the source video.",
        "Starting from I0, have subject_01 perform this complete target motion.",
    ):
        _expect_rejected(bad)
    prompt = natural._rewrite_prompt(_input())
    assert "MUST start" not in prompt or "literal word Have" in prompt
    assert "Never output the caption form" in prompt
    audit_prompt = natural._audit_prompt(
        _input(),
        "Have the woman turn her head left by 15 degrees. Keep the camera fixed. "
        + natural.CANONICAL_PRESERVATION_INSTRUCTION,
    )
    assert "15 degrees" in audit_prompt and "NOT timing" in audit_prompt
    assert '"core_events_entailed": null' in audit_prompt
    assert "MUST be replaced by the JSON boolean true or false" in audit_prompt
    assert "natural_imperative MUST be exactly true or false" in audit_prompt
    assert "Never copy the candidate instruction into natural_imperative" in audit_prompt
    payload = natural._audit_payload(_input(), _compiled_instruction())
    serialized_payload = json.dumps(payload, sort_keys=True)
    assert "motion_evidence" not in serialized_payload
    assert "from frame 0" not in serialized_payload
    assert "frame 40" not in serialized_payload
    sanitized = natural._strip_reference_timing(
        "from frame 0 to frame 20, turn right by 15 degrees; for 24 frames, "
        "reverse and exit the frame",
        context="selftest",
    )
    assert "15 degrees" in sanitized and "exit the frame" in sanitized
    assert "frame 0" not in sanitized and "24 frames" not in sanitized
    try:
        natural._validate_audit(
            _audit(pass_value=False),
            iid="selftest",
            subject_ids=["subject_01"],
            instruction=_compiled_instruction(),
        )
    except natural.NaturalMotionInstructionError:
        pass
    else:
        raise AssertionError("failed independent audit was accepted")

    inconsistent = _audit()
    inconsistent["absolute_timing_absent"] = False
    inconsistent["overall_verdict"] = "fail"
    inconsistent["reason_codes"] = [
        "absolute_timing_present",
        "subject_motion_mismatch",
    ]
    inconsistent["confidence"] = "low"
    normalized = natural._validate_audit(
        inconsistent,
        iid="selftest",
        subject_ids=["subject_01"],
        instruction=_compiled_instruction(
            "Have the motorcyclist turn sharply right to complete a U-turn, then "
            "reverse along the road and exit the frame on the right."
        ),
    )
    assert normalized["effective_verdict"] == "pass"
    assert normalized["deterministic_gates"]["absolute_timing_absent"] is True
    assert normalized["model_reported_diagnostics"]["absolute_timing_absent"] is False
    assert normalized["model_reported_diagnostics"]["confidence"] == "low"
    assert any(
        item["field"] == "absolute_timing_absent"
        for item in normalized["model_effective_disagreements"]
    )

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        manifest = root / "input.jsonl"
        manifest.write_bytes(natural._canonical_bytes(_input()) + b"\n")
        output = root / "output"
        backend = _Backend([_rewrite(), _audit()])
        args = argparse.Namespace(
            input=manifest,
            output_root=output,
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
        assert backend.calls == 2
        instruction = (
            output / "instructions" / "selftest" / "natural_edit_instruction.txt"
        ).read_text()
        assert "pick up the bone" in instruction and "frame 0" not in instruction
        assert natural.run_worker(
            args,
            backend_factory=lambda **_: (_ for _ in ()).throw(
                AssertionError("resume loaded backend")
            ),
        ) == 0
        verify = argparse.Namespace(
            input=manifest,
            output_root=output,
            expected_rows=1,
            min_ok=1,
            manifest_output=root / "dataset.jsonl",
            summary_output=root / "summary.json",
            skip_source_revalidation=True,
        )
        assert natural.verify_outputs(verify) == 0
        assert natural.verify_outputs(verify) == 0
        dataset = json.loads((root / "dataset.jsonl").read_text())
        assert dataset["label_status"].endswith("video_audit_pending")

        # Exact v4 regression: the first auditor puts the complete candidate
        # instruction into a boolean field. The next audit must receive the
        # validator diagnostic, correct the type, and leave the raw bad output
        # intact in the first attempt trace.
        malformed = _audit()
        malformed["natural_imperative"] = _compiled_instruction()
        retry_backend = _Backend([_rewrite(), malformed, _rewrite(), _audit()])
        retry_output = root / "retry-output"
        retry_args = argparse.Namespace(
            input=manifest,
            output_root=retry_output,
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
        assert natural.run_worker(
            retry_args, backend_factory=lambda **_: retry_backend
        ) == 0
        assert retry_backend.calls == 4
        retry_result = json.loads(
            (retry_output / "rows" / "selftest" / "result.json").read_text()
        )
        assert retry_result["status"] == "ok"
        assert len(retry_result["attempts"]) == 2
        assert retry_result["attempts"][0]["error"] == (
            "audit natural_imperative is not boolean"
        )
        first_audit_raw = json.loads(retry_result["attempts"][0]["audit_raw"])
        assert first_audit_raw["natural_imperative"] == _compiled_instruction()
        assert retry_result["attempts"][1]["status"] == "ok"
        retry_audit_prompts = [
            request["user"]
            for request in retry_backend.requests
            if request["system"] == natural.AUDIT_SYSTEM
        ]
        assert len(retry_audit_prompts) == 2
        assert "PREVIOUS AUDIT OUTPUT FAILED VALIDATION" not in retry_audit_prompts[0]
        assert "PREVIOUS AUDIT OUTPUT FAILED VALIDATION" in retry_audit_prompts[1]
        assert "audit natural_imperative is not boolean" in retry_audit_prompts[1]
    print("[natural-motion-selftest] PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
