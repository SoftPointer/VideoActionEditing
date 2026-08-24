from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import checkpoint_visual_quality_gate_v3 as gate_v3  # noqa: E402
from tools import postvideo_quality_router_v4 as router  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _family(
    name: str,
    *,
    triggered: bool = False,
    unresolved: bool = False,
    unsupported_structure: bool = False,
) -> dict[str, object]:
    scales: dict[str, dict[str, object]] = {}
    for index, (width, height) in enumerate(gate_v3.ANALYSIS_SCALES):
        active = index == 0
        row: dict[str, object] = {
            "triggered": triggered and active,
            "unresolved": unresolved and active,
            "raw_conditions": {},
        }
        if name == "ROUTEOFF_STRUCTURE":
            row.update(
                {
                    "raw_candidate_triggered": (triggered or unresolved) and active,
                    "independent_spatial_artifact_support": (
                        triggered and active and not unsupported_structure
                    ),
                    "base_structure_reference_eligible": True,
                }
            )
        scales[f"{width}x{height}"] = row
    return {
        "triggered": triggered,
        "triggered_scales": ["192x144"] if triggered else [],
        "unresolved": unresolved,
        "unresolved_scales": ["192x144"] if unresolved else [],
        "per_scale": scales,
    }


def _gate_report(
    *,
    iid: str,
    source: Path,
    candidate: Path,
    frozen_base: Path,
    status: str,
    unsupported_hard_structure: bool = False,
) -> dict[str, object]:
    if status == "fail":
        families = {
            "NOISE": _family("NOISE"),
            "BLUR": (
                _family("BLUR")
                if unsupported_hard_structure
                else _family("BLUR", triggered=True)
            ),
            "ROUTEOFF_STRUCTURE": _family(
                "ROUTEOFF_STRUCTURE",
                triggered=unsupported_hard_structure,
                unsupported_structure=unsupported_hard_structure,
            ),
            "FREEZE": _family("FREEZE"),
        }
        failure_codes = [
            "quality_routeoff_structure" if unsupported_hard_structure else "quality_blur"
        ]
        unresolved_codes: list[str] = []
    elif status == "unresolved":
        families = {
            "NOISE": _family("NOISE"),
            "BLUR": _family("BLUR"),
            "ROUTEOFF_STRUCTURE": _family("ROUTEOFF_STRUCTURE", unresolved=True),
            "FREEZE": _family("FREEZE"),
        }
        failure_codes = []
        unresolved_codes = ["quality_routeoff_structure_requires_external_verifier"]
    else:
        families = {
            name: _family(name)
            for name in ("NOISE", "BLUR", "ROUTEOFF_STRUCTURE", "FREEZE")
        }
        failure_codes = []
        unresolved_codes = []

    def media(path: Path) -> dict[str, object]:
        return {
            "path": str(path.resolve()),
            "sha256": _sha(path),
            "size_bytes": path.stat().st_size,
        }

    hard = status == "fail"
    unresolved = status == "unresolved"
    return {
        "schema_version": gate_v3.SCHEMA_VERSION,
        "fail_closed": True,
        "status": status,
        "passed": status == "pass",
        "publishable": status == "pass",
        "input_contract_passed": True,
        "hard_artifact_failure": hard,
        "unresolved": unresolved,
        "failure_codes": failure_codes,
        "unresolved_codes": unresolved_codes,
        "metadata": {
            "sample_id": iid,
            "inputs": {
                "source": media(source),
                "candidate": media(candidate),
                "frozen_base": media(frozen_base),
            },
        },
        "decision": {
            "outcome": status,
            "passed": status == "pass",
            "hard_artifact_failure": hard,
            "unresolved": unresolved,
            "failure_codes": failure_codes,
            "unresolved_codes": unresolved_codes,
            "evidence_families": families,
        },
    }


def _quality(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "bernini-postvideo-quality-observation-v1",
        "action_implemented": "yes",
        "identity_preserved": "yes",
        "species_preserved": "yes",
        "clothing_preserved": "yes",
        "non_edited_content_preserved": "yes",
        "camera_preserved": "yes",
        "blur_level": "none",
        "flicker_level": "none",
        "artifact_level": "none",
        "confidence": "high",
        "evidence": {
            "action": [{"frames": ["T0", "T3"], "observation": "ordered target motion"}],
            "identity": [{"frames": ["S0", "T0"], "observation": "same subject"}],
            "preservation": [{"frames": ["S0", "T0"], "observation": "same scene and camera"}],
            "technical": [{"frames": ["T0", "T3"], "observation": "clear and stable"}],
        },
        "uncertainty_codes": [],
    }
    value.update(overrides)
    return value


def _qwen_record(
    *,
    iid: str,
    source: Path,
    candidate: Path,
    quality: dict[str, object] | None = None,
    outcome: str = "success",
) -> dict[str, object]:
    quality = quality or _quality()
    record: dict[str, object] = {
        "iid": iid,
        "audit_outcome": outcome,
        "input": {
            "source_video": {"path": str(source.resolve()), "sha256": _sha(source)},
            "target_video": {"path": str(candidate.resolve()), "sha256": _sha(candidate)},
        },
        "quality": quality,
        "quality_sha256": router.object_sha256(quality),
        "model_identity_sha256": "a" * 64,
        "prompt_contract_sha256": "b" * 64,
    }
    record["record_digest"] = router.object_sha256(record)
    return record


class PostVideoQualityRouterV4Test(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source.mp4"
        self.candidate = self.root / "candidate.mp4"
        self.base = self.root / "base.mp4"
        self.source.write_bytes(b"source-video")
        self.candidate.write_bytes(b"candidate-video")
        self.base.write_bytes(b"base-video")
        self.iid = "case00"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _validated_gate(self, status: str) -> dict[str, object]:
        return router.validate_gate_report(
            _gate_report(
                iid=self.iid,
                source=self.source,
                candidate=self.candidate,
                frozen_base=self.base,
                status=status,
            ),
            iid=self.iid,
        )

    def _route(
        self,
        *,
        gate_status: str,
        quality: dict[str, object] | None = None,
        qwen_outcome: str = "success",
    ) -> dict[str, object]:
        return router.route_one(
            iid=self.iid,
            gate=self._validated_gate(gate_status),
            gate_report_binding={"path": "/evidence/gate.json", "sha256": "c" * 64},
            qwen_record=_qwen_record(
                iid=self.iid,
                source=self.source,
                candidate=self.candidate,
                quality=quality,
                outcome=qwen_outcome,
            ),
            qwen_audit_done_sha256="d" * 64,
        )

    def test_all_axes_pass_is_the_only_automatic_promotion(self) -> None:
        row = self._route(gate_status="pass")
        self.assertEqual(row["decision"]["route"], "PROMOTE")
        self.assertTrue(row["decision"]["training_eligible"])
        self.assertEqual(
            {axis["status"] for axis in row["axes"].values()}, {"PASS"}
        )

    def test_gate_hard_failure_cannot_be_overridden_by_positive_qwen(self) -> None:
        row = self._route(gate_status="fail")
        self.assertEqual(row["decision"]["route"], "REJECT")
        self.assertEqual(row["axes"]["artifact_quality"]["status"], "FAIL")
        self.assertFalse(
            row["axes"]["artifact_quality"]["qwen_may_override_gate_v3_hard_failure"]
        )
        self.assertFalse(row["decision"]["training_eligible"])

    def test_low_ssim_style_unresolved_evidence_never_auto_promotes(self) -> None:
        row = self._route(gate_status="unresolved")
        self.assertEqual(row["decision"]["route"], "REVIEW")
        self.assertEqual(row["decision"]["overall_status"], "UNRESOLVED")
        self.assertEqual(row["axes"]["artifact_quality"]["status"], "UNRESOLVED")
        self.assertFalse(
            row["axes"]["artifact_quality"]["low_ssim_used_as_standalone_reject_signal"]
        )
        self.assertFalse(row["decision"]["training_eligible"])

    def test_unsupported_low_ssim_cannot_be_encoded_as_hard_gate_failure(self) -> None:
        report = _gate_report(
            iid=self.iid,
            source=self.source,
            candidate=self.candidate,
            frozen_base=self.base,
            status="fail",
            unsupported_hard_structure=True,
        )
        with self.assertRaisesRegex(router.QualityRouterV4Error, "low SSIM alone"):
            router.validate_gate_report(report, iid=self.iid)

    def test_qwen_content_or_action_failure_rejects_a_gate_pass(self) -> None:
        row = self._route(
            gate_status="pass",
            quality=_quality(
                action_implemented="no",
                identity_preserved="no",
            ),
        )
        self.assertEqual(row["decision"]["route"], "REJECT")
        self.assertEqual(row["axes"]["action_alignment"]["status"], "FAIL")
        self.assertEqual(
            row["axes"]["identity_content_preservation"]["status"], "FAIL"
        )

    def test_low_confidence_negative_qwen_label_is_review_not_reject(self) -> None:
        row = self._route(
            gate_status="pass",
            quality=_quality(
                action_implemented="no",
                artifact_level="high",
                confidence="low",
            ),
        )
        self.assertEqual(row["decision"]["route"], "REVIEW")
        self.assertEqual(row["axes"]["action_alignment"]["status"], "UNRESOLVED")
        self.assertEqual(row["axes"]["artifact_quality"]["status"], "UNRESOLVED")
        self.assertFalse(row["decision"]["training_eligible"])

    def test_missing_or_unclear_qwen_evidence_routes_to_review(self) -> None:
        unclear = _quality(
            action_implemented="unclear",
            identity_preserved="unclear",
            species_preserved="unclear",
            clothing_preserved="unclear",
            non_edited_content_preserved="unclear",
            camera_preserved="unclear",
            blur_level="unclear",
            flicker_level="unclear",
            artifact_level="unclear",
            confidence="unclear",
            uncertainty_codes=["generation_error"],
        )
        row = self._route(
            gate_status="pass", quality=unclear, qwen_outcome="generation_error"
        )
        self.assertEqual(row["decision"]["route"], "REVIEW")
        self.assertFalse(row["decision"]["training_eligible"])

    def test_gate_qwen_candidate_hash_mismatch_fails_without_output(self) -> None:
        other = self.root / "other.mp4"
        other.write_bytes(b"other-video")
        with self.assertRaisesRegex(router.QualityRouterV4Error, "candidate SHA-256"):
            router.route_one(
                iid=self.iid,
                gate=self._validated_gate("pass"),
                gate_report_binding={"path": "/gate", "sha256": "c" * 64},
                qwen_record=_qwen_record(
                    iid=self.iid,
                    source=self.source,
                    candidate=other,
                ),
                qwen_audit_done_sha256="d" * 64,
            )

    def _write_manifest(self, report: dict[str, object]) -> Path:
        report_path = self.root / "gate.json"
        report_path.write_text(
            json.dumps(report, sort_keys=True) + "\n", encoding="utf-8"
        )
        rows = [
            {
                "iid": self.iid,
                "gate_report_path": str(report_path.resolve()),
                "gate_report_sha256": _sha(report_path),
            }
        ]
        manifest = {
            "schema_version": router.MANIFEST_SCHEMA,
            "complete": True,
            "row_count": 1,
            "rows": rows,
            "rows_digest": router.object_sha256(rows),
        }
        path = self.root / "gate-manifest.json"
        path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def test_build_publishes_hash_closed_release_and_validator_accepts_it(self) -> None:
        manifest = self._write_manifest(
            _gate_report(
                iid=self.iid,
                source=self.source,
                candidate=self.candidate,
                frozen_base=self.base,
                status="pass",
            )
        )
        audit_dir = self.root / "audit"
        audit_dir.mkdir()
        done = audit_dir / "done.json"
        done.write_text("{}\n", encoding="utf-8")
        done_sha = _sha(done)
        qwen_record = _qwen_record(
            iid=self.iid, source=self.source, candidate=self.candidate
        )
        audit = {
            "status": "VALID",
            "output_dir": str(audit_dir.resolve()),
            "records": [qwen_record],
            "production_backend": True,
            "input_sha256": "e" * 64,
            "model_identity_sha256": "f" * 64,
            "method_source_revision": "1" * 40,
        }
        output = self.root / "routing.jsonl"
        with patch.object(
            router.qwen_builder,
            "validate_published_audit",
            return_value=audit,
        ) as validator:
            receipt = router.build_routing(
                gate_manifest_path=manifest,
                expected_gate_manifest_sha256=_sha(manifest),
                qwen_audit_dir=audit_dir,
                expected_qwen_audit_done_sha256=done_sha,
                output_jsonl=output,
            )
            self.assertEqual(receipt["route_counts"], {"PROMOTE": 1})
            self.assertEqual(receipt["training_eligible_count"], 1)
            self.assertTrue(output.is_file())
            self.assertTrue(Path(f"{output}.receipt.json").is_file())
            self.assertTrue(Path(f"{output}.sha256").is_file())
            validated = router.validate_release(
                output,
                expected_receipt_sha256=_sha(Path(f"{output}.receipt.json")),
            )
        self.assertEqual(validator.call_count, 2)
        for observed in validator.call_args_list:
            self.assertEqual(observed.kwargs["expected_done_sha256"], done_sha)
            self.assertTrue(observed.kwargs["require_production"])
        self.assertEqual(validated["status"], "VALID")

    def test_manifest_hash_is_caller_pinned(self) -> None:
        manifest = self._write_manifest(
            _gate_report(
                iid=self.iid,
                source=self.source,
                candidate=self.candidate,
                frozen_base=self.base,
                status="pass",
            )
        )
        with self.assertRaisesRegex(router.QualityRouterV4Error, "declared hash"):
            router.load_gate_manifest(manifest, expected_sha256="0" * 64)

    def test_cli_exposes_route_and_validate_commands(self) -> None:
        parser = router._parser()
        route = parser.parse_args(
            [
                "route",
                "--gate-manifest",
                "/tmp/gate.json",
                "--expected-gate-manifest-sha256",
                "0" * 64,
                "--qwen-audit-dir",
                "/tmp/audit",
                "--expected-qwen-audit-done-sha256",
                "1" * 64,
                "--output-jsonl",
                "/tmp/route.jsonl",
            ]
        )
        self.assertEqual(route.command, "route")
        validate = parser.parse_args(
            [
                "validate",
                "--output-jsonl",
                "/tmp/route.jsonl",
                "--expected-receipt-sha256",
                "2" * 64,
            ]
        )
        self.assertEqual(validate.command, "validate")

    def test_formal_json_schema_is_present_and_source_bound(self) -> None:
        schema = json.loads(router.JSON_SCHEMA_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            schema["$schema"], "https://json-schema.org/draft/2020-12/schema"
        )
        self.assertEqual(
            schema["$defs"]["route_record"]["properties"]["schema_version"]["const"],
            router.RECORD_SCHEMA,
        )
        self.assertEqual(
            schema["$defs"]["input_manifest"]["properties"]["schema_version"]["const"],
            router.MANIFEST_SCHEMA,
        )


if __name__ == "__main__":
    unittest.main()
