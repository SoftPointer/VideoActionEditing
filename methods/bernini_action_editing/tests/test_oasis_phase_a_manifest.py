from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = METHOD_ROOT / "tools"
for path in (METHOD_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import build_oasis_phase_a_manifest as builder
import oasis_phase_a_manifest as manifest


def _sha(character: str) -> str:
    return character * 64


class OASISManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.sources = []
        for index in range(4):
            path = self.root / f"source-{index}.mp4"
            path.write_bytes(f"source-{index}".encode("ascii"))
            self.sources.append(path)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def captions(self, prefix: str):
        return {
            branch: f"{prefix} caption for {branch}"
            for branch in manifest.BRANCH_ORDER
        }

    def draft(self, *, calibration_status: str = "unresolved"):
        rows = []
        ordinal = 0
        for family in manifest.FAMILY_ORDER:
            for split in manifest.SPLIT_ORDER:
                rows.append(
                    {
                        "sample_id": f"sample-{ordinal}",
                        "family": family,
                        "analysis_split": split,
                        "source_video_path": str(self.sources[ordinal]),
                        "source_caption": f"source caption {ordinal}",
                        "complete_action_caption": f"action caption {ordinal}",
                        "actor_binding": "the main foreground actor",
                        "raw_caption_by_branch": self.captions(f"s{ordinal}"),
                        "calibration_candidate_id": f"calibration-{ordinal}",
                        "calibration_event_receipt_digest": f"{ordinal + 4:x}" * 64,
                        "weak_wrongref_diagnostic": {
                            "available": False,
                            "proxy_kind": "none",
                            "known_confounds": [],
                        },
                    }
                )
                ordinal += 1
        return {
            "checkpoint_tree_sha256": _sha("1"),
            "t2v_scalar_calibration": {"status": calibration_status},
            "seed_order": [20260808, 20260809],
            "samples": rows,
        }

    def formal_authorization(self) -> tuple[dict, Path]:
        unsigned = {
            "schema_version": (
                "bernini-pair-v5-t2v-calibration-mainline-authorization-v4"
            ),
            "source_bank_spec_sha256": _sha("2"),
            "source_bank_receipt_digest": _sha("3"),
            "formal_score_provenance_set_digest": _sha("4"),
            "formal_score_schema": "formal-score-v3",
            "formal_score_filename": "formal-score.json",
            "formal_score_scalar_definition": "global-known-target-MACE",
            "formal_score_arithmetic_contract_digest": _sha("5"),
            "preregistration_digest": _sha("7"),
            "calibration_receipt_digest": _sha("8"),
            "family_mapping_set_digest": _sha("9"),
            "checkpoint_tree_sha256": _sha("1"),
            "score_count": 40,
            "branch_order": list(manifest.BRANCH_ORDER),
            "action_family_order": [
                "dog-sit-facing-camera",
                "human-rise-to-stand",
            ],
            "all_formal_scalar_provenance_recomputed": True,
            "formal_receipts_validated_by_active_v4_canonical_code": True,
            "active_repository_score_schema_consumed": True,
            "legacy_v3_compatibility_score_consumed": False,
            "initialization_ablation_teacher_or_adapter_artifact_consumed": False,
            "t2v_media_latent_gaussian_or_proposal_exported_to_native_scorer": False,
            "only_family_maps_threshold_prompts_and_scalar_digests_exported": True,
            "calibration_maps_authorized": True,
            "native_rv2v_optimizer_authorized": False,
            "scientific_action_editing_claim": False,
        }
        value = {**unsigned, "authorization_digest": manifest.object_sha256(unsigned)}
        path = self.root / "formal-mainline-authorization.json"
        path.write_bytes(manifest.canonical_json_bytes(value) + b"\n")
        return value, path

    def dedicated_evidence(self, draft: dict) -> tuple[dict, Path]:
        authorization, authorization_path = self.formal_authorization()
        family_ids = {
            "dog_sit_hold": "dog-sit-facing-camera",
            "human_stand_hold": "human-rise-to-stand",
        }
        gates = {}
        for family in manifest.FAMILY_ORDER:
            fit_ids = [
                row["calibration_candidate_id"]
                for row in draft["samples"]
                if row["family"] == family and row["analysis_split"] == "fit"
            ]
            confirmation_ids = [
                row["calibration_candidate_id"]
                for row in draft["samples"]
                if row["family"] == family
                and row["analysis_split"] == "confirmation"
            ]
            gate_unsigned = {
                "family": family,
                "formal_action_family_id": family_ids[family],
                "fit_candidate_ids": fit_ids,
                "confirmation_candidate_ids": confirmation_ids,
                "branch_order": list(manifest.BRANCH_ORDER),
                "score_definition": (
                    "known_target_velocity_global_MACE_action_vs_all_nine_stable_log_ratio"
                ),
                "minimum_robust_action_log_ratio": 0.25,
                "minimum_action_log_ratio_by_negative": {
                    branch: 0.10 for branch in manifest.BRANCH_ORDER[1:]
                },
                "fit_event_and_scalar_gate_passed": True,
                "confirmation_event_and_scalar_gate_passed": True,
            }
            gates[family] = {
                **gate_unsigned,
                "gate_digest": manifest.object_sha256(gate_unsigned),
            }
        unsigned = {
            "schema_version": manifest.SCALAR_CALIBRATION_EVIDENCE_SCHEMA,
            "checkpoint_tree_sha256": _sha("1"),
            "family_order": list(manifest.FAMILY_ORDER),
            "formal_scalar_source": {
                "validator_id": "validate_pair_v5_t2v_calibration_mainline_v3",
                "path": str(authorization_path),
                "file_sha256": manifest.file_sha256(authorization_path),
                "authorization_digest": authorization["authorization_digest"],
                "formal_validator_recomputed": True,
            },
            "family_gates": gates,
            "scalar_calibration_only": True,
            "source_media_consumed_by_oasis_runtime": False,
            "source_latent_or_gaussian_consumed_by_oasis_runtime": False,
            "source_media_or_latent_used_as_teacher": False,
            "optimizer_authorized": False,
            "frozen_controller_prompt_family_qualified": True,
            "scientific_action_editing_success_claim": False,
        }
        value = {**unsigned, "evidence_digest": manifest.object_sha256(unsigned)}
        path = self.root / "oasis-scalar-evidence.json"
        path.write_bytes(manifest.canonical_json_bytes(value) + b"\n")
        return value, path

    def build_and_write(self, draft=None):
        value = builder.build_manifest(self.draft() if draft is None else draft)
        path = self.root / "manifest.json"
        path.write_bytes(manifest.canonical_json_bytes(value) + b"\n")
        return value, path, manifest.file_sha256(path)

    def test_builds_exact_four_source_only_cells_and_24_matched_rollouts(self) -> None:
        value, path, digest = self.build_and_write()
        checked = manifest.load_phase_a_manifest(path, digest, verify_files=True)
        self.assertEqual(len(checked.samples), 4)
        self.assertEqual(checked.scalar_calibration_status, "unresolved")
        self.assertEqual(
            tuple((cell.family, cell.analysis_split) for cell in checked.samples),
            tuple(
                (family, split)
                for family in manifest.FAMILY_ORDER
                for split in manifest.SPLIT_ORDER
            ),
        )
        contract = manifest.static_contract()
        self.assertEqual(contract["rollout_count"], 24)
        self.assertEqual(contract["rollout_count_per_family"], 12)
        self.assertEqual(contract["arm_order"], list(manifest.ARM_ORDER))
        self.assertFalse(contract["old_cagd_authority_accepted"])
        self.assertFalse(value["information_flow"]["paired_target_video_or_latent"])
        self.assertFalse(value["information_flow"]["t2v_media_or_latent_to_rv2v"])
        # Runner compatibility aliases are derived from rich sealed fields.
        self.assertEqual(checked.samples[0].edit_instruction, "action caption 0")
        self.assertEqual(len(checked.samples[0].edit_instruction_sha256), 64)
        self.assertEqual(
            len(checked.samples[0].source_instruction_binding_digest), 64
        )

    def test_unresolved_scalar_calibration_fails_closed(self) -> None:
        _value, path, digest = self.build_and_write()
        checked = manifest.load_phase_a_manifest(path, digest, verify_files=True)
        with self.assertRaisesRegex(
            manifest.OASISManifestError,
            "OASIS_T2V_SCALAR_CALIBRATION_UNRESOLVED",
        ):
            manifest.load_dedicated_scalar_calibration_evidence(checked)

    def test_resolved_dedicated_scalar_evidence_validates_v4_authority(self) -> None:
        draft = self.draft()
        evidence, evidence_path = self.dedicated_evidence(draft)
        draft["t2v_scalar_calibration"] = {
            "status": "resolved",
            "path": str(evidence_path),
        }
        _value, path, digest = self.build_and_write(draft)
        checked = manifest.load_phase_a_manifest(path, digest, verify_files=True)
        loaded = manifest.load_dedicated_scalar_calibration_evidence(checked)
        self.assertEqual(loaded["evidence_digest"], evidence["evidence_digest"])
        self.assertFalse(loaded["optimizer_authorized"])
        self.assertFalse(loaded["source_media_or_latent_used_as_teacher"])

    def test_formal_authorization_tamper_fails_even_if_resealed(self) -> None:
        draft = self.draft()
        _evidence, evidence_path = self.dedicated_evidence(draft)
        wrapper = json.loads(evidence_path.read_text(encoding="ascii"))
        formal_path = Path(wrapper["formal_scalar_source"]["path"])
        formal = json.loads(formal_path.read_text(encoding="ascii"))
        formal["native_rv2v_optimizer_authorized"] = True
        unsigned_formal = dict(formal)
        unsigned_formal.pop("authorization_digest")
        formal["authorization_digest"] = manifest.object_sha256(unsigned_formal)
        formal_path.write_bytes(manifest.canonical_json_bytes(formal) + b"\n")
        wrapper["formal_scalar_source"]["file_sha256"] = manifest.file_sha256(
            formal_path
        )
        wrapper["formal_scalar_source"]["authorization_digest"] = formal[
            "authorization_digest"
        ]
        unsigned_wrapper = dict(wrapper)
        unsigned_wrapper.pop("evidence_digest")
        wrapper["evidence_digest"] = manifest.object_sha256(unsigned_wrapper)
        evidence_path.write_bytes(manifest.canonical_json_bytes(wrapper) + b"\n")
        draft["t2v_scalar_calibration"] = {
            "status": "resolved",
            "path": str(evidence_path),
        }
        _value, path, digest = self.build_and_write(draft)
        checked = manifest.load_phase_a_manifest(path, digest, verify_files=True)
        with self.assertRaisesRegex(manifest.OASISManifestError, "authorization differs"):
            manifest.load_dedicated_scalar_calibration_evidence(checked)

    def test_family_major_order_and_source_hash_are_fail_closed(self) -> None:
        draft = self.draft()
        draft["samples"][0], draft["samples"][1] = (
            draft["samples"][1],
            draft["samples"][0],
        )
        _value, path, digest = self.build_and_write(draft)
        with self.assertRaisesRegex(manifest.OASISManifestError, "family-major"):
            manifest.load_phase_a_manifest(path, digest, verify_files=True)

        _value, path, digest = self.build_and_write()
        self.sources[0].write_bytes(b"changed")
        with self.assertRaisesRegex(manifest.OASISManifestError, "source video SHA"):
            manifest.load_phase_a_manifest(path, digest, verify_files=True)

    def test_wrongref_proxy_is_diagnostic_only(self) -> None:
        draft = self.draft()
        decoy = self.root / "decoy.mp4"
        decoy.write_bytes(b"decoy")
        draft["samples"][0]["weak_wrongref_diagnostic"] = {
            "available": True,
            "path": str(decoy),
            "proxy_kind": "same_class_confounded_background_scale_species",
            "known_confounds": ["background", "scale", "species"],
        }
        value, path, digest = self.build_and_write(draft)
        weak = value["samples"][0]["weak_wrongref_diagnostic"]
        self.assertFalse(weak["identity_only_claim"])
        self.assertFalse(weak["used_for_authorization"])
        checked = manifest.load_phase_a_manifest(path, digest, verify_files=True)
        self.assertTrue(checked.samples[0].weak_wrongref_diagnostic.available)

    def test_target_and_old_authority_fields_are_rejected(self) -> None:
        draft = self.draft()
        draft["samples"][0]["target_video_path"] = str(self.sources[1])
        with self.assertRaisesRegex(builder.OASISManifestBuildError, "field closure"):
            builder.build_manifest(draft)
        draft = self.draft()
        draft["cagd_v3_evidence"] = {"path": "/tmp/forbidden"}
        with self.assertRaisesRegex(
            builder.OASISManifestBuildError, "root field closure"
        ):
            builder.build_manifest(draft)

    def test_manifest_digest_and_file_digest_are_both_checked(self) -> None:
        _value, path, _digest = self.build_and_write()
        with self.assertRaisesRegex(manifest.OASISManifestError, "file SHA"):
            manifest.load_phase_a_manifest(path, _sha("f"), verify_files=True)
        tampered = json.loads(path.read_text(encoding="ascii"))
        tampered["seed_order"] = [1, 2]
        bad = self.root / "tampered.json"
        bad.write_bytes(manifest.canonical_json_bytes(tampered) + b"\n")
        with self.assertRaisesRegex(manifest.OASISManifestError, "digest/schema"):
            manifest.load_phase_a_manifest(
                bad, manifest.file_sha256(bad), verify_files=True
            )


if __name__ == "__main__":
    unittest.main()
