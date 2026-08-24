from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from motive.r10b_bernini_pilot_manifest import (
    AUDIT_ROW_SCHEMA,
    QUEUE_ROW_SCHEMA,
    QUEUE_SUMMARY_SCHEMA,
    _validate_audit_record,
    write_qwen_audit_queue,
)
from motive.r10b_family_qwen_audit import (
    ADAPTERS_NAME,
    BLIND_SCHEMA,
    DONE_NAME,
    PROMPT_CONTRACT,
    PROMPT_CONTRACT_SHA256,
    RECORDS_NAME,
    SUMMARY_NAME,
    R10BFamilyQwenAuditError,
    R10BFamilyQwenGenerationError,
    _adapter,
    _build_parser,
    _production_backend_execution,
    _test_backend_execution,
    _validate_backend_execution,
    deterministic_alignment,
    file_record,
    hard_classification,
    run_audit,
    validate_blind,
    validate_published_audit,
)
from motive.r10b_tangent_core import canonical_json, file_digest, object_digest


MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct@test-revision"
_AUTHORIZATION = {
    "human_label": False,
    "formal_evidence": False,
    "representation_promoted": False,
    "renderer_probe_authorized": False,
    "generation_authorized": False,
    "training_authorized": False,
}


def _empty_wave() -> dict:
    return {
        "limb_part": "none",
        "event_frames": [],
        "direction_sequence": [],
        "directed_toward_viewer": "unclear",
    }


def _empty_lie() -> dict:
    return {
        "start_posture": "unclear",
        "start_frame": -1,
        "lowering_frame": -1,
        "final_frame": -1,
        "final_posture": "unclear",
    }


def _blind(*, family: str = "wave") -> dict:
    target_wave = _empty_wave()
    target_lie = _empty_lie()
    morphology = "adult_human"
    if family == "wave":
        target_wave = {
            "limb_part": "arm",
            "event_frames": [2, 5, 8],
            "direction_sequence": ["left", "right", "left"],
            "directed_toward_viewer": "yes",
        }
    elif family == "quadruped_lie_down":
        morphology = "dog"
        target_lie = {
            "start_posture": "on_all_fours",
            "start_frame": 1,
            "lowering_frame": 5,
            "final_frame": 10,
            "final_posture": "prone_or_reclined",
        }
    else:  # pragma: no cover - fixture misuse
        raise ValueError(f"unsupported family: {family}")
    return {
        "schema_version": BLIND_SCHEMA,
        "subject_morphology": morphology,
        "source_wave": _empty_wave(),
        "target_wave": target_wave,
        "source_lie_down": _empty_lie(),
        "target_lie_down": target_lie,
        "source_actor_motion": "none",
        "target_actor_motion": "clear",
        "camera_motion": "none",
        "background_motion": "none",
        "artifact_level": "none",
        "preservation_quality": "acceptable",
        "identity_appearance_change": "none",
        "nonphysical_effect": "none",
        "deformation": "none",
        "flicker": "none",
        "reflection_or_sunglasses_artifact": "none",
        "secondary_action": "none",
        "uncertainty_codes": [],
    }


def _classify(blind: dict, family: str) -> dict:
    correct = deterministic_alignment(blind, family=family)
    cross = (
        "quadruped_lie_down" if family == "wave" else "wave"
    )
    counterfactual = deterministic_alignment(blind, family=cross)
    return hard_classification(
        blind,
        correct,
        intended_family=family,
        counterfactual_alignment=counterfactual,
    )


def _queue_bytes(rows: list[dict]) -> bytes:
    return "".join(
        canonical_json(row) + "\n" for row in rows
    ).encode("utf-8")


def _pretty_json_bytes(value: dict) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _rewrite_record_commit(output: Path, records: list[dict]) -> None:
    """Rewrite all metadata bindings so record-level tampering is exercised."""

    record_bytes = _queue_bytes(records)
    (output / RECORDS_NAME).write_bytes(record_bytes)
    summary = json.loads(
        (output / SUMMARY_NAME).read_text(encoding="utf-8")
    )
    summary["outputs"][RECORDS_NAME] = {
        "rows": len(records),
        **file_record(record_bytes),
    }
    summary_bytes = _pretty_json_bytes(summary)
    (output / SUMMARY_NAME).write_bytes(summary_bytes)
    done = json.loads(
        (output / DONE_NAME).read_text(encoding="utf-8")
    )
    done["files"][RECORDS_NAME] = file_record(record_bytes)
    done["files"][SUMMARY_NAME] = file_record(summary_bytes)
    (output / DONE_NAME).write_bytes(_pretty_json_bytes(done))


def _write_queue(
    root: Path,
    *,
    row_count: int,
) -> tuple[Path, Path, list[dict]]:
    data_root = root / "data"
    rows = []
    for index in range(row_count):
        iid = "case-wave" if row_count == 1 else f"case-wave-{index}"
        sample = data_root / iid
        sample.mkdir(parents=True)
        source = sample / "source.mp4"
        target = sample / "target.mp4"
        source.write_bytes(f"source-video-binding-{index}".encode())
        target.write_bytes(f"target-video-binding-{index}".encode())
        rows.append(
            {
                "schema_version": QUEUE_ROW_SCHEMA,
                "iid": iid,
                "component_id": f"component-wave-{index}",
                "screen_cell": "positive:wave",
                "screen_role_hint": "positive",
                "intended_family": "wave",
                "canonical_prompt": (
                    "Make the subject wave one forelimb toward the viewer."
                ),
                "prompt_variants": {
                    "canonical": (
                        "Make the subject wave one forelimb toward the viewer."
                    ),
                    "noop": "Keep the video unchanged.",
                    "cross_family_shuffle": "Make the quadruped lie down.",
                    "cross_family_shuffle_family": "quadruped_lie_down",
                },
                "media_binding": {
                    "data_root": str(data_root.resolve()),
                    "src_video": {
                        "relative_path": f"{iid}/source.mp4",
                        "sha256": file_digest(source),
                    },
                    "tgt_video": {
                        "relative_path": f"{iid}/target.mp4",
                        "sha256": file_digest(target),
                    },
                },
                "authorization": dict(_AUTHORIZATION),
            }
        )
    raw = _queue_bytes(rows)
    payload = {
        "rows": rows,
        "summary": {
            "schema_version": QUEUE_SUMMARY_SCHEMA,
            "rows": len(rows),
            "queue_sha256": hashlib.sha256(raw).hexdigest(),
            "qwen_audit": {
                "schema_version": AUDIT_ROW_SCHEMA,
                "qwen_model_id": MODEL_ID,
                "qwen_prompt_sha256": PROMPT_CONTRACT_SHA256,
            },
        },
    }
    queue_dir = root / "queue"
    write_qwen_audit_queue(payload, queue_dir)
    return queue_dir, data_root, rows


def _write_one_row_queue(root: Path) -> tuple[Path, Path, dict]:
    queue_dir, data_root, rows = _write_queue(root, row_count=1)
    return queue_dir, data_root, rows[0]


class _FakeBackend:
    model_revision = "fake-model-revision"
    transformers_version = "fake-transformers"

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.text_calls = 0
        self.visual_calls = 0

    def generate_visual_observation(self, **kwargs) -> tuple[str, str]:
        self.visual_calls += 1
        self.visual_kwargs = kwargs
        return canonical_json(_blind()), "d" * 64

    def generate_text(self, **kwargs) -> str:
        self.text_calls += 1
        raise AssertionError("deterministic stage 2 must not call Qwen text")


class _ScriptedBackend(_FakeBackend):
    def __init__(self, *, script: list[object], **kwargs) -> None:
        super().__init__(**kwargs)
        self.script = list(script)

    def generate_visual_observation(self, **kwargs) -> tuple[str, str]:
        self.visual_calls += 1
        self.visual_kwargs = kwargs
        outcome = self.script.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        if not isinstance(outcome, str):  # pragma: no cover - fixture guard
            raise AssertionError("script outcome must be text or exception")
        return outcome, "e" * 64


class R10BFamilyQwenAuditTests(unittest.TestCase):
    def test_blind_prompt_uses_single_token_neutral_json_template(self) -> None:
        prompt = PROMPT_CONTRACT["blind_prompt"]
        self.assertNotIn("|", prompt)
        self.assertIn('"limb_part": "none"', prompt)
        self.assertIn('"event_frames": []', prompt)
        self.assertIn('"direction_sequence": []', prompt)
        self.assertIn('"start_posture": "unclear"', prompt)
        self.assertIn('"uncertainty_codes": []', prompt)
        self.assertEqual(
            PROMPT_CONTRACT_SHA256,
            "25397b5cfde7e10975fdd0f650a638740b526a24851e937a0f616a94e37b26eb",
        )

    def test_production_backend_execution_proves_cuda0_only_placement(
        self,
    ) -> None:
        class Tensor:
            def __init__(self, device: str, elements: int) -> None:
                self.device = device
                self.elements = elements

            def numel(self) -> int:
                return self.elements

        class Model:
            device = "cuda:0"
            hf_device_map = {"": 0, "visual": "cuda:0"}

            def named_parameters(self):
                return iter(
                    [
                        ("model.weight", Tensor("cuda:0", 16)),
                        ("visual.weight", Tensor("cuda:0", 8)),
                    ]
                )

            def named_buffers(self):
                return iter([("rotary.inv_freq", Tensor("cuda:0", 4))])

        class Backend:
            model = Model()

        class Cuda:
            @staticmethod
            def is_available() -> bool:
                return True

            @staticmethod
            def device_count() -> int:
                return 1

            @staticmethod
            def current_device() -> int:
                return 0

            @staticmethod
            def get_device_name(index: int) -> str:
                if index != 0:  # pragma: no cover - fixture guard
                    raise AssertionError(index)
                return "AMD Instinct MI210"

        class Torch:
            cuda = Cuda()

        evidence = _production_backend_execution(
            Backend(), torch_module=Torch()
        )
        self.assertEqual(evidence["mode"], "production_local_qwen")
        self.assertTrue(evidence["production_backend"])
        self.assertFalse(evidence["test_backend"])
        self.assertTrue(evidence["cuda_only"])
        self.assertEqual(evidence["device_count"], 1)
        self.assertEqual(evidence["model_device"], "cuda:0")
        self.assertEqual(evidence["parameter_tensors"], 2)
        self.assertEqual(evidence["parameter_elements"], 24)
        self.assertEqual(evidence["parameter_devices"], ["cuda:0"])
        self.assertEqual(evidence["buffer_devices"], ["cuda:0"])
        self.assertTrue(evidence["hf_device_map_present"])
        self.assertEqual(evidence["hf_device_map_devices"], ["cuda:0"])
        self.assertEqual(_validate_backend_execution(evidence), evidence)

        Model.named_parameters = lambda self: iter(
            [("model.weight", Tensor("cpu", 16))]
        )
        with self.assertRaisesRegex(
            R10BFamilyQwenAuditError,
            "cpu/disk/meta offload is forbidden",
        ):
            _production_backend_execution(
                Backend(), torch_module=Torch()
            )

    def test_test_backend_evidence_is_explicit_and_strict(self) -> None:
        evidence = _test_backend_execution()
        self.assertEqual(evidence["mode"], "injected_test_backend")
        self.assertTrue(evidence["test_backend"])
        self.assertFalse(evidence["production_backend"])
        self.assertFalse(evidence["inspection_performed"])
        self.assertFalse(evidence["cuda_only"])
        self.assertEqual(_validate_backend_execution(evidence), evidence)
        tampered = copy.deepcopy(evidence)
        tampered["cuda_only"] = True
        with self.assertRaisesRegex(
            R10BFamilyQwenAuditError,
            "test backend execution evidence differs",
        ):
            _validate_backend_execution(tampered)

    def test_blind_v2_is_compact_strict_and_frame_bounded(self) -> None:
        accepted = validate_blind(_blind(), nframes=12)
        self.assertEqual(
            accepted["target_wave"]["event_frames"], [2, 5, 8]
        )
        self.assertNotIn("source_state_segments", accepted)
        self.assertNotIn("visible_change", canonical_json(accepted))

        out_of_range = _blind()
        out_of_range["target_wave"]["event_frames"][0] = 12
        with self.assertRaisesRegex(
            R10BFamilyQwenAuditError, "valid frame index"
        ):
            validate_blind(out_of_range, nframes=12)

        extra = _blind()
        extra["instruction"] = "make the person wave"
        with self.assertRaisesRegex(
            R10BFamilyQwenAuditError, "schema differs"
        ):
            validate_blind(extra, nframes=12)

    def test_wave_hard_rule_checks_every_compact_field(self) -> None:
        blind = validate_blind(_blind(), nframes=12)
        passed = _classify(blind, "wave")
        self.assertEqual(passed["role"], "positive")
        self.assertTrue(passed["wave_hard_pass"])
        self.assertTrue(passed["counterfactual_rejected"])
        self.assertTrue(
            passed["target_wave_frames_strictly_increasing"]
        )
        self.assertEqual(passed["observed_direction_reversals"], 2)

        cases = []
        wrong_part = copy.deepcopy(blind)
        wrong_part["target_wave"]["limb_part"] = "other"
        cases.append(wrong_part)
        too_short = copy.deepcopy(blind)
        too_short["target_wave"]["event_frames"] = [2]
        too_short["target_wave"]["direction_sequence"] = ["left"]
        cases.append(too_short)
        unequal = copy.deepcopy(blind)
        unequal["target_wave"]["direction_sequence"] = ["left", "right"]
        cases.append(unequal)
        unordered = copy.deepcopy(blind)
        unordered["target_wave"]["event_frames"] = [2, 8, 5]
        cases.append(unordered)
        no_opposite = copy.deepcopy(blind)
        no_opposite["target_wave"]["direction_sequence"] = [
            "left",
            "left",
            "left",
        ]
        cases.append(no_opposite)
        not_toward = copy.deepcopy(blind)
        not_toward["target_wave"]["directed_toward_viewer"] = "no"
        cases.append(not_toward)
        source_already_waves = copy.deepcopy(blind)
        source_already_waves["source_wave"] = copy.deepcopy(
            blind["target_wave"]
        )
        cases.append(source_already_waves)

        for case in cases:
            with self.subTest(case=case["target_wave"]):
                failed = _classify(case, "wave")
                self.assertEqual(failed["role"], "reject")
                self.assertFalse(failed["wave_hard_pass"])

    def test_lie_down_needs_ordered_frames_postures_and_no_source(self) -> None:
        blind = validate_blind(
            _blind(family="quadruped_lie_down"), nframes=12
        )
        passed = _classify(blind, "quadruped_lie_down")
        self.assertEqual(passed["role"], "positive")
        self.assertTrue(passed["lie_down_hard_pass"])

        bad_order = copy.deepcopy(blind)
        bad_order["target_lie_down"]["lowering_frame"] = 10
        bad_order["target_lie_down"]["final_frame"] = 5
        self.assertFalse(
            _classify(bad_order, "quadruped_lie_down")[
                "lie_down_hard_pass"
            ]
        )
        bad_start = copy.deepcopy(blind)
        bad_start["target_lie_down"]["start_posture"] = "other"
        self.assertFalse(
            _classify(bad_start, "quadruped_lie_down")[
                "lie_down_hard_pass"
            ]
        )
        bad_final = copy.deepcopy(blind)
        bad_final["target_lie_down"]["final_posture"] = "other"
        self.assertFalse(
            _classify(bad_final, "quadruped_lie_down")[
                "lie_down_hard_pass"
            ]
        )
        source_already_lies = copy.deepcopy(blind)
        source_already_lies["source_lie_down"] = copy.deepcopy(
            blind["target_lie_down"]
        )
        failed = _classify(
            source_already_lies, "quadruped_lie_down"
        )
        self.assertFalse(failed["lie_down_hard_pass"])
        self.assertTrue(failed["source_lie_transition"])

    def test_deterministic_correct_and_cross_share_one_blind(self) -> None:
        blind = validate_blind(_blind(), nframes=12)
        correct = deterministic_alignment(blind, family="wave")
        cross = deterministic_alignment(
            blind, family="quadruped_lie_down"
        )
        self.assertTrue(correct["matches_instruction"])
        self.assertEqual(correct["verdict"], "valid_action")
        self.assertFalse(cross["matches_instruction"])
        self.assertEqual(cross["verdict"], "wrong")
        self.assertEqual(cross["observed_family"], "wave")
        self.assertEqual(
            PROMPT_CONTRACT["stage2"]["mode"],
            "deterministic_python_no_model_call",
        )
        self.assertEqual(PROMPT_CONTRACT["qwen_text_calls_per_row"], 0)
        cli = _build_parser().parse_args(["--output-dir", "audit"])
        self.assertEqual(cli.max_new_tokens, 512)

    def test_nuisance_precedence_and_adapter_preserve_effect_role(self) -> None:
        blind = validate_blind(_blind(), nframes=12)
        blind["artifact_level"] = "high"
        blind["camera_motion"] = "high"
        correct = deterministic_alignment(blind, family="other")
        counterfactual = deterministic_alignment(blind, family="none")
        hard = hard_classification(
            blind,
            correct,
            intended_family="other",
            counterfactual_alignment=counterfactual,
        )
        self.assertEqual(hard["role"], "effect")
        self.assertTrue(hard["effect_precedence"])
        self.assertTrue(hard["camera_precedence"])

        row = {"iid": "case-effect", "intended_family": "other"}
        adapter = _adapter(
            row=row,
            model_id=MODEL_ID,
            blind=blind,
            correct=correct,
            hard=hard,
        )
        self.assertEqual(adapter["schema_version"], AUDIT_ROW_SCHEMA)
        self.assertEqual(adapter["success"], "no")
        self.assertEqual(adapter["nonphysical_effect"], "high")
        _validate_audit_record(
            adapter,
            queue_row=row,
            model_id=MODEL_ID,
            prompt_sha256=PROMPT_CONTRACT_SHA256,
        )

    def test_fake_backend_uses_one_visual_and_zero_text_calls(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue_dir, data_root, queue_row = _write_one_row_queue(root)
            model = root / "model"
            model.mkdir()
            output = root / "audit"
            instances: list[_FakeBackend] = []

            def factory(**kwargs):
                instance = _FakeBackend(**kwargs)
                instances.append(instance)
                return instance

            result = run_audit(
                queue_dir=queue_dir,
                data_root=data_root,
                model_path=model,
                output_dir=output,
                nframes=12,
                backend_factory=factory,
            )
            self.assertEqual(result["status"], "VALID")
            self.assertEqual(result["hard_role_counts"], {"positive": 1})
            self.assertFalse(result["representation_gate_passed"])
            self.assertEqual(instances[0].visual_calls, 1)
            self.assertEqual(instances[0].text_calls, 0)
            self.assertEqual(instances[0].kwargs["max_new_tokens"], 512)
            self.assertEqual(
                instances[0].visual_kwargs["visual_input"], "mosaic"
            )

            for name in (
                RECORDS_NAME,
                ADAPTERS_NAME,
                SUMMARY_NAME,
                DONE_NAME,
            ):
                self.assertTrue((output / name).is_file())
            record = json.loads(
                (output / RECORDS_NAME)
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertEqual(
                record["queue_row_sha256"], object_digest(queue_row)
            )
            self.assertEqual(
                set(record["raw_response_sha256"]), {"blind"}
            )
            self.assertEqual(
                record["raw_response_diagnostic"],
                {
                    "chars": 0,
                    "text": "",
                    "sha256": object_digest(""),
                },
            )
            self.assertEqual(
                record["counterfactual_reused_blind_observation_sha256"],
                record["blind_observation_sha256"],
            )
            self.assertFalse(
                record["counterfactual_alignment"][
                    "matches_instruction"
                ]
            )
            self.assertNotIn(
                queue_row["canonical_prompt"],
                canonical_json(record["blind_observation"]),
            )
            self.assertEqual(record["authorization"], _AUTHORIZATION)
            published_adapters = [
                json.loads(line)
                for line in (output / ADAPTERS_NAME)
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(published_adapters, result["adapters"])
            _validate_audit_record(
                published_adapters[0],
                queue_row=queue_row,
                model_id=MODEL_ID,
                prompt_sha256=PROMPT_CONTRACT_SHA256,
            )

            validated = validate_published_audit(output)
            self.assertEqual(validated["status"], "VALID")
            summary = json.loads(
                (output / SUMMARY_NAME).read_text(encoding="utf-8")
            )
            self.assertEqual(
                summary["prompt_contract"]["qwen_text_calls_per_row"], 0
            )
            self.assertEqual(
                summary["prompt_contract"]["stage2"],
                "deterministic_python_no_model_call",
            )
            self.assertEqual(summary["videos_copied"], 0)
            self.assertEqual(summary["videos_rendered"], 0)
            self.assertFalse(summary["formal_evidence"])
            self.assertEqual(
                summary["runtime"]["backend_execution"],
                _test_backend_execution(),
            )
            self.assertEqual(
                validated["backend_execution"],
                _test_backend_execution(),
            )

            summary["runtime"]["backend_execution"]["cuda_only"] = True
            summary_bytes = _pretty_json_bytes(summary)
            (output / SUMMARY_NAME).write_bytes(summary_bytes)
            done = json.loads(
                (output / DONE_NAME).read_text(encoding="utf-8")
            )
            done["files"][SUMMARY_NAME] = file_record(summary_bytes)
            (output / DONE_NAME).write_bytes(_pretty_json_bytes(done))
            with self.assertRaisesRegex(
                R10BFamilyQwenAuditError,
                "test backend execution evidence differs",
            ):
                validate_published_audit(output)

    def test_schema_error_is_publishable_ambiguous_reject(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue_dir, data_root, _row = _write_one_row_queue(root)
            model = root / "model"
            model.mkdir()
            output = root / "audit"

            def factory(**kwargs):
                return _ScriptedBackend(
                    script=["not strict JSON"],
                    **kwargs,
                )

            result = run_audit(
                queue_dir=queue_dir,
                data_root=data_root,
                model_path=model,
                output_dir=output,
                backend_factory=factory,
            )
            self.assertEqual(result["status"], "VALID")
            self.assertEqual(result["successful_rows"], 0)
            self.assertEqual(result["schema_error_rows"], 1)
            self.assertEqual(result["generation_error_rows"], 0)

            record = json.loads(
                (output / RECORDS_NAME).read_text(encoding="utf-8")
            )
            self.assertEqual(record["audit_outcome"], "schema_error")
            self.assertEqual(
                record["hard_classification"]["role"], "reject"
            )
            self.assertEqual(
                record["blind_observation"]["uncertainty_codes"],
                ["blind_schema_error"],
            )
            self.assertTrue(
                record["errors"][0].startswith("schema_error:")
            )
            self.assertEqual(
                record["raw_response_diagnostic"],
                {
                    "chars": len("not strict JSON"),
                    "text": "not strict JSON",
                    "sha256": object_digest("not strict JSON"),
                },
            )
            self.assertEqual(
                record["raw_response_diagnostic"]["sha256"],
                record["raw_response_sha256"]["blind"],
            )
            for name in (SUMMARY_NAME, DONE_NAME):
                metadata = json.loads(
                    (output / name).read_text(encoding="utf-8")
                )
                self.assertEqual(metadata["status"], "complete")
                self.assertEqual(metadata["successful_rows"], 0)
                self.assertEqual(metadata["schema_error_rows"], 1)
                self.assertEqual(metadata["generation_error_rows"], 0)

    def test_long_schema_error_response_is_fully_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue_dir, data_root, _row = _write_one_row_queue(root)
            model = root / "model"
            model.mkdir()
            output = root / "audit"
            raw_response = "not-json-" + ("界" * 8192)

            def factory(**kwargs):
                return _ScriptedBackend(
                    script=[raw_response],
                    **kwargs,
                )

            run_audit(
                queue_dir=queue_dir,
                data_root=data_root,
                model_path=model,
                output_dir=output,
                backend_factory=factory,
            )
            record = json.loads(
                (output / RECORDS_NAME).read_text(encoding="utf-8")
            )
            diagnostic = record["raw_response_diagnostic"]
            self.assertEqual(diagnostic["chars"], len(raw_response))
            self.assertEqual(diagnostic["text"], raw_response)
            self.assertEqual(diagnostic["sha256"], object_digest(raw_response))
            self.assertEqual(
                diagnostic["sha256"],
                record["raw_response_sha256"]["blind"],
            )
            self.assertEqual(
                validate_published_audit(output)["schema_error_rows"],
                1,
            )

    def test_raw_response_diagnostic_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue_dir, data_root, _row = _write_one_row_queue(root)
            model = root / "model"
            model.mkdir()
            output = root / "audit"

            def factory(**kwargs):
                return _ScriptedBackend(
                    script=["invalid-json"],
                    **kwargs,
                )

            run_audit(
                queue_dir=queue_dir,
                data_root=data_root,
                model_path=model,
                output_dir=output,
                backend_factory=factory,
            )
            base_record = json.loads(
                (output / RECORDS_NAME).read_text(encoding="utf-8")
            )
            mutations = []

            wrong_chars = copy.deepcopy(base_record)
            wrong_chars["raw_response_diagnostic"]["chars"] += 1
            mutations.append(wrong_chars)

            internally_consistent_but_unbound = copy.deepcopy(base_record)
            replacement = "different-invalid-json"
            internally_consistent_but_unbound[
                "raw_response_diagnostic"
            ] = {
                "chars": len(replacement),
                "text": replacement,
                "sha256": object_digest(replacement),
            }
            mutations.append(internally_consistent_but_unbound)

            wrong_text_hash = copy.deepcopy(base_record)
            wrong_text_hash["raw_response_diagnostic"]["sha256"] = "0" * 64
            mutations.append(wrong_text_hash)

            for record in mutations:
                with self.subTest(
                    diagnostic=record["raw_response_diagnostic"]
                ):
                    _rewrite_record_commit(output, [record])
                    with self.assertRaisesRegex(
                        R10BFamilyQwenAuditError,
                        "raw response diagnostic",
                    ):
                        validate_published_audit(output)

    def test_cuda_oom_is_generation_error_and_never_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue_dir, data_root, _rows = _write_queue(
                root, row_count=2
            )
            model = root / "model"
            model.mkdir()
            output = root / "audit"

            def factory(**kwargs):
                return _ScriptedBackend(
                    script=[
                        canonical_json(_blind()),
                        RuntimeError("CUDA out of memory"),
                    ],
                    **kwargs,
                )

            result = run_audit(
                queue_dir=queue_dir,
                data_root=data_root,
                model_path=model,
                output_dir=output,
                backend_factory=factory,
            )
            self.assertEqual(
                result["status"], "PARTIAL_GENERATION_FAILURE"
            )
            self.assertEqual(result["successful_rows"], 1)
            self.assertEqual(result["schema_error_rows"], 0)
            self.assertEqual(result["generation_error_rows"], 1)
            records = [
                json.loads(line)
                for line in (output / RECORDS_NAME)
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(
                [record["audit_outcome"] for record in records],
                ["success", "generation_error"],
            )
            self.assertTrue(
                records[1]["errors"][0].startswith(
                    "generation_error:RuntimeError:CUDA out of memory"
                )
            )
            self.assertNotIn("schema_error:", records[1]["errors"][0])
            self.assertEqual(
                records[1]["raw_response_diagnostic"],
                {
                    "chars": 0,
                    "text": "",
                    "sha256": object_digest(""),
                },
            )
            for name in (SUMMARY_NAME, DONE_NAME):
                metadata = json.loads(
                    (output / name).read_text(encoding="utf-8")
                )
                self.assertEqual(
                    metadata["status"], "partial_generation_failure"
                )
                self.assertEqual(metadata["successful_rows"], 1)
                self.assertEqual(metadata["schema_error_rows"], 0)
                self.assertEqual(metadata["generation_error_rows"], 1)

    def test_all_generation_failures_exit_without_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue_dir, data_root, _rows = _write_queue(
                root, row_count=2
            )
            model = root / "model"
            model.mkdir()
            output = root / "audit"

            def factory(**kwargs):
                return _ScriptedBackend(
                    script=[
                        RuntimeError("CUDA out of memory"),
                        OSError("video decoder failed"),
                    ],
                    **kwargs,
                )

            with self.assertRaisesRegex(
                R10BFamilyQwenGenerationError,
                "all Qwen rows failed.*no audit output was published",
            ):
                run_audit(
                    queue_dir=queue_dir,
                    data_root=data_root,
                    model_path=model,
                    output_dir=output,
                    backend_factory=factory,
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
