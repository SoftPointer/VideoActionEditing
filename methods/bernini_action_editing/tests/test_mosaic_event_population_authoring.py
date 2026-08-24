from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import mosaic_event_population_authoring as population  # noqa: E402


REGISTRY_PATH = (
    METHOD_ROOT
    / "assets"
    / "mosaic_event_population_compact6_topup20_v1.json"
)


def _registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def _bound_inherited_rows(composition: dict) -> list[dict]:
    rows = []
    for cell in composition["inherited_cells"]:
        for branch, candidate_id in zip(
            population.SEMANTIC_BRANCHES, cell["candidate_ids"]
        ):
            caption = (
                f"A complete continuous exact eighty one frame review scene shows the "
                f"registered primary actor and the independently authored {branch} event "
                "under stable illumination with an explicit temporal ending and no cuts."
            )
            rows.append(
                {
                    "generation_receipt_digest": hashlib.sha256(
                        f"generation-{candidate_id}".encode("ascii")
                    ).hexdigest(),
                    "candidate": {
                        "candidate_id": candidate_id,
                        "analysis_split": cell["analysis_split"],
                        "action_family_id": cell["legacy_action_family_id"],
                        "calibration_group_id": cell["cell_id"],
                        "actor_group_id": cell["actor_group_id"],
                        "scene_group_id": cell["scene_group_id"],
                        "action_group_id": cell["action_group_id"],
                        "semantic_branch": branch,
                        "seed": cell["seed"],
                        "full_t2v_caption": caption,
                        "full_t2v_caption_utf8_sha256": hashlib.sha256(
                            caption.encode("utf-8")
                        ).hexdigest(),
                    }
                }
            )
    return rows


def _generation_receipt(path: Path, request: dict, media: Path) -> tuple[str, str]:
    media_sha = hashlib.sha256(media.read_bytes()).hexdigest()
    unsigned = {
        "candidate": {
            "candidate_id": request["candidate_id"],
            "semantic_branch": request["requested_semantic_branch"],
            "full_t2v_caption_utf8_sha256": request[
                "full_t2v_caption_utf8_sha256"
            ],
        },
        "artifacts": {"mp4": {"path": str(media), "sha256": media_sha}},
    }
    value = {**unsigned, "receipt_digest": population.object_sha256(unsigned)}
    raw = population.canonical_json_bytes(value) + b"\n"
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest(), value["receipt_digest"]


def _accepted_sidecar(root: Path, request: dict) -> dict:
    branch = request["requested_semantic_branch"]
    media = root / f"{request['candidate_id']}.mp4"
    media.write_bytes(f"decoded exact81 media {request['candidate_id']}".encode("ascii"))
    receipt = root / f"{request['candidate_id']}.receipt.json"
    receipt_sha, receipt_digest = _generation_receipt(receipt, request, media)
    evidence_file = root / f"{request['candidate_id']}.audit-evidence.json"
    evidence_file.write_bytes(
        json.dumps(
            {
                "candidate": request["candidate_id"],
                "blind_review": True,
                "observed_class": branch,
            },
            sort_keys=True,
        ).encode("ascii")
    )
    action = branch == "action"
    checks = {
        "video_quality_pass": True,
        "continuous_no_cut": True,
        "primary_actor_trackable": True,
        "family_start_state_observed": action,
        "family_transition_observed": action,
        "family_terminal_state_observed": action,
        "family_terminal_hold_observed": action,
        "full_target_event_observed": action,
        "full_target_event_false_confirmed": not action,
        "requested_branch_mechanism_observed": True,
        "actor_binding": "secondary" if branch == "wrong_actor" else "primary",
        "object_binding": (
            "distractor"
            if branch == "wrong_object"
            else "not_applicable" if action else "none"
        ),
        "camera_class": (
            "camera_only_motion" if branch == "camera_only" else "locked_or_natural"
        ),
        "appearance_only_observed": branch == "appearance_only",
    }
    evidence = {
        "start_frames": [0, 4] if action else [0],
        "transition_frames": [20, 35] if action else [],
        "terminal_frames": [50] if action else [],
        "terminal_hold_frames": [60, 67] if action else [],
        "branch_mechanism_frames": [20, 35] if action else [20],
        "written_observation": (
            "The blinded reviewer watched the complete video and recorded visible temporal "
            "evidence without seeing the generation prompt or requested semantic branch."
        ),
    }
    unsigned = {
        "schema_version": population.AUDIT_SIDECAR_SCHEMA,
        "request_digest": request["request_digest"],
        "candidate_id": request["candidate_id"],
        "rendered_media_path": str(media),
        "rendered_media_sha256": hashlib.sha256(media.read_bytes()).hexdigest(),
        "generation_receipt_path": str(receipt),
        "generation_receipt_sha256": receipt_sha,
        "generation_receipt_digest": receipt_digest,
        "audit_evidence_path": str(evidence_file),
        "audit_evidence_sha256": hashlib.sha256(evidence_file.read_bytes()).hexdigest(),
        "audit_protocol_sha256": request["audit_protocol_sha256"],
        "auditor_id": "blind-reviewer-01",
        "auditor_method": "human_blind_video_review_v1",
        "audited_at_utc": "2026-08-08T12:00:00Z",
        "generation_prompt_or_requested_branch_disclosed": False,
        "entire_video_viewed": True,
        "hidden_feature_extraction_started_before_audit": False,
        "observed_class": branch,
        "checks": checks,
        "evidence": evidence,
        "eligibility_decision": "accept",
        "rejection_reasons": [],
    }
    return {**unsigned, "sidecar_digest": population.object_sha256(unsigned)}


class CompactPopulationTopologyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = population.validate_registry(_registry())

    def test_sealed_asset_is_six_atomic_families_three_identities_two_seeds(self) -> None:
        raw_sha = hashlib.sha256(REGISTRY_PATH.read_bytes()).hexdigest()
        loaded, observed = population.load_sealed_registry(
            REGISTRY_PATH.resolve(), raw_sha
        )
        self.assertEqual(observed, raw_sha)
        self.assertEqual(loaded, self.registry)
        self.assertEqual(len(self.registry["action_families"]), 6)
        self.assertNotIn(
            "articulated-pose-transition",
            {family["family_id"] for family in self.registry["action_families"]},
        )
        for family in self.registry["action_families"]:
            identities = [
                *family["inherited_identity_scenes"],
                *family["topup_identity_scenes"],
            ]
            self.assertEqual(len(identities), 3)
            self.assertEqual({row["analysis_split"] for row in identities}, {"fit", "confirmation"})
            if family["population_role"] == "pilot_development":
                self.assertEqual(
                    [row["analysis_split"] for row in identities].count("fit"), 2
                )
                self.assertEqual(
                    [row["analysis_split"] for row in identities].count("confirmation"), 1
                )
            self.assertTrue(all(len(row["seeds"]) == 2 for row in identities))
            self.assertEqual(len({seed for row in identities for seed in row["seeds"]}), 6)

    def test_exact_existing_inflight_16_cell_mapping_and_atomic_relabel(self) -> None:
        bundle = population.build_stage_bundle(
            self.registry, stage_id="full_topup20"
        )
        composition = population.validate_composition_plan(bundle["composition"])
        inherited = composition["inherited_cells"]
        self.assertEqual(len(inherited), 16)
        self.assertEqual(
            {row["source_iid"] for row in inherited},
            {
                "7b88a1ca1f804f41",
                "841b5e0080a1441d",
                "a35b590961d24694",
                "a66e6818e4144928",
                "00435ad621c44fac",
                "0c6915018a5f4d9b",
                "33322eb8ec1e4703",
                "71ba57892bd043df",
            },
        )
        self.assertEqual(
            sum(row["source_bank_profile"] == "core4-v2" for row in inherited), 8
        )
        self.assertEqual(
            sum(row["source_bank_profile"] == "reserve4-v1" for row in inherited), 8
        )
        reserve_profile = next(
            row
            for row in composition["inherited_bank_profiles"]
            if row["profile_id"] == "reserve4-v1"
        )
        self.assertEqual(
            reserve_profile["seed1_root_spec_raw_sha256"],
            "2861b1021531896d387b0dccb945b9fc2516bf01472982c0fe2f7c1377ca7bab",
        )
        self.assertEqual(
            reserve_profile["seed2_root_spec_raw_sha256"],
            "0578cd6c39cdb625e69cf04164ffce29487b81e852974619f0ee43325e49398e",
        )
        self.assertTrue(
            all(not row["source_root_spec_hash_must_be_bound_before_use"] for row in inherited)
        )
        reserve = [row for row in inherited if row["source_bank_profile"] == "reserve4-v1"]
        self.assertEqual(
            {row["atomic_action_family_id"] for row in reserve},
            {
                "human-arms-raised-to-hands-on-hips",
                "human-head-turn-forward-and-smile",
                "human-peace-sign-to-open-palm-wave",
                "human-left-fist-to-forward-palm-down",
            },
        )
        self.assertTrue(
            all(row["legacy_action_family_id"] == "articulated-pose-transition" for row in reserve)
        )
        self.assertTrue(all(row["atomic_family_relabel_is_metadata_only"] for row in reserve))
        inherited_ids = {candidate for row in inherited for candidate in row["candidate_ids"]}
        topup_ids = {
            candidate
            for row in composition["new_topup_cells"]
            for candidate in row["candidate_ids"]
        }
        self.assertTrue(inherited_ids.isdisjoint(topup_ids))

    def test_staged_generation_cost_is_120_plus_80_not_regenerated_360(self) -> None:
        expected = {
            "pilot_topup12": (12, 120, 147.0, 28),
            "holdout_topup8": (8, 80, 98.0, 24),
            "full_topup20": (20, 200, 245.0, 36),
        }
        for stage, (cells, clips, minutes, composed) in expected.items():
            with self.subTest(stage=stage):
                bundle = population.build_stage_bundle(self.registry, stage_id=stage)
                self.assertEqual(bundle["authoring"]["expected_cell_count"], cells)
                self.assertEqual(bundle["cost"]["generated_clip_count"], clips)
                self.assertEqual(
                    bundle["cost"]["estimated_wall_minutes_at_reference_rate"], minutes
                )
                self.assertEqual(
                    bundle["composition"]["composed_cell_count_after_stage"], composed
                )
                self.assertTrue(bundle["cost"]["inherited_clips_are_reused_not_regenerated"])
                self.assertTrue(bundle["cost"]["compact_population_is_pilot_only"])
                self.assertFalse(bundle["cost"]["editor_optimizer_authorized"])
        full = population.build_stage_bundle(self.registry, stage_id="full_topup20")
        self.assertEqual(full["cost"]["sp4_generated_clip_counts"], {"sp4-a": 100, "sp4-b": 100})
        self.assertAlmostEqual(
            full["cost"]["estimated_eight_gpu_hours_at_reference_rate"],
            32.666666666666664,
        )

    def test_topup_fragment_is_composition_closed_and_geometry_only(self) -> None:
        bundle = population.build_stage_bundle(self.registry, stage_id="full_topup20")
        authoring = population.validate_topup_authoring(
            bundle["authoring"], bundle["composition"]
        )
        self.assertEqual(authoring["schema_version"], population.TOPUP_AUTHORING_SCHEMA)
        self.assertEqual(len(authoring["cells"]), 20)
        self.assertEqual(
            {cell["geometry_source_video"] for cell in authoring["cells"]},
            {self.registry["geometry_source_video"]},
        )
        self.assertFalse(authoring["geometry_source_pixels_enter_transformer"])
        self.assertFalse(authoring["geometry_source_vae_latent_created"])
        self.assertFalse(authoring["generated_media_editor_use_authorized"])
        self.assertTrue(
            all(
                list(cell["branch_descriptions"]) == list(population.SEMANTIC_BRANCHES)
                for cell in authoring["cells"]
            )
        )

    def test_registry_rejects_broad_family_duplicate_seed_and_policy_drift(self) -> None:
        mutations = []
        broad = deepcopy(_registry())
        broad["action_families"][0]["family_id"] = "articulated-pose-transition"
        mutations.append(broad)
        duplicate_seed = deepcopy(_registry())
        duplicate_seed["action_families"][0]["topup_identity_scenes"][0]["seeds"][0] = 2026080825
        mutations.append(duplicate_seed)
        label_leak = deepcopy(_registry())
        label_leak["audit_policy"]["requested_branch_is_label"] = True
        mutations.append(label_leak)
        geometry_leak = deepcopy(_registry())
        geometry_leak["authoring_contract"]["geometry_source_pixels_enter_transformer"] = True
        mutations.append(geometry_leak)
        for index, value in enumerate(mutations):
            with self.subTest(index=index), self.assertRaises(
                population.MosaicEventPopulationError
            ):
                population.validate_registry(value)


class GenericAuditAuthoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = population.validate_registry(_registry())
        self.bundle = population.build_stage_bundle(
            self.registry, stage_id="full_topup20"
        )

    def test_inherited_profiler_is_order_independent_null_label_and_atomic(self) -> None:
        rows = _bound_inherited_rows(self.bundle["composition"])
        random.Random(17).shuffle(rows)
        manifest = population.build_inherited_audit_requests_from_authenticated_rows(
            self.registry, self.bundle["composition"], rows
        )
        checked = population.validate_audit_request_manifest(manifest)
        self.assertEqual(checked["candidate_count"], 160)
        self.assertFalse(checked["critic_feature_extraction_authorized_before_audits"])
        self.assertTrue(
            all(row["critic_label_before_audit"] is None for row in checked["candidate_requests"])
        )
        self.assertTrue(
            all(
                row["authenticated_generation_receipt_digest"] is not None
                for row in checked["candidate_requests"]
            )
        )
        reserve_requests = [
            row
            for row in checked["candidate_requests"]
            if row["candidate_id"].startswith("pair5-t2v-reserve4")
        ]
        self.assertNotIn(
            "articulated-pose-transition",
            {row["action_family_id"] for row in reserve_requests},
        )
        merged = population.merge_composed_audit_request_manifests(
            self.bundle["composition"], checked, self.bundle["audit_requests"]
        )
        self.assertEqual(merged["candidate_count"], 360)
        self.assertTrue(
            all(row["critic_label_before_audit"] is None for row in merged["candidate_requests"])
        )

    def test_inherited_profiler_rejects_prompt_family_or_candidate_substitution(self) -> None:
        rows = _bound_inherited_rows(self.bundle["composition"])
        reserve = next(
            row
            for row in rows
            if row["candidate"]["candidate_id"].startswith("pair5-t2v-reserve4")
        )
        reserve["candidate"]["action_family_id"] = "human-motion"
        with self.assertRaisesRegex(
            population.MosaicEventPopulationError, "identity/order"
        ):
            population.build_inherited_audit_requests_from_authenticated_rows(
                self.registry, self.bundle["composition"], rows
            )

    def test_sidecar_is_required_and_only_complete_ten_branch_cell_unlocks_features(self) -> None:
        manifest = self.bundle["audit_requests"]
        empty = population.build_eligibility_index(manifest, {})
        self.assertEqual(empty["accepted_candidate_count"], 0)
        self.assertEqual(empty["eligible_cell_ids"], [])
        self.assertEqual(empty["feature_extraction_authorized_candidate_ids"], [])
        self.assertFalse(empty["prompt_branch_used_as_label"])
        self.assertFalse(empty["editor_optimizer_authorized"])

        first_cell = manifest["candidate_requests"][0]["cell_id"]
        first_requests = [
            row for row in manifest["candidate_requests"] if row["cell_id"] == first_cell
        ]
        self.assertEqual(len(first_requests), 10)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            sidecars = {
                request["candidate_id"]: _accepted_sidecar(root, request)
                for request in first_requests
            }
            provisional = population.build_eligibility_index(manifest, sidecars)
            self.assertEqual(provisional["eligible_cell_ids"], [])
            self.assertEqual(
                provisional["audit_accept_but_bank_authentication_missing_count"], 10
            )
            observed_digests = {
                candidate_id: sidecar["generation_receipt_digest"]
                for candidate_id, sidecar in sidecars.items()
            }
            authenticated_rows = []
            for request in manifest["candidate_requests"]:
                candidate_id = request["candidate_id"]
                authenticated_rows.append(
                    {
                        "generation_receipt_digest": observed_digests.get(
                            candidate_id,
                            hashlib.sha256(
                                f"authenticated-{candidate_id}".encode("ascii")
                            ).hexdigest(),
                        ),
                        "candidate": {
                            "candidate_id": candidate_id,
                            "analysis_split": request["analysis_split"],
                            "action_family_id": request["action_family_id"],
                            "calibration_group_id": request["cell_id"],
                            "actor_group_id": request["actor_group_id"],
                            "scene_group_id": request["scene_group_id"],
                            "action_group_id": request["action_group_id"],
                            "semantic_branch": request["requested_semantic_branch"],
                            "seed": request["seed"],
                            "full_t2v_caption_utf8_sha256": request[
                                "full_t2v_caption_utf8_sha256"
                            ],
                        },
                    }
                )
            bound_manifest = population.bind_audit_requests_to_authenticated_rows(
                manifest, authenticated_rows
            )
            bound_requests = {
                row["candidate_id"]: row for row in bound_manifest["candidate_requests"]
            }
            # Request digests change when the independent strong-bank receipt
            # digest is attached, so the detached sidecars must bind that seal.
            sidecars = {
                candidate_id: _accepted_sidecar(root, bound_requests[candidate_id])
                for candidate_id in sidecars
            }
            partial = dict(sidecars)
            partial.pop(first_requests[-1]["candidate_id"])
            incomplete = population.build_eligibility_index(bound_manifest, partial)
            self.assertEqual(incomplete["eligible_cell_ids"], [])
            self.assertEqual(
                incomplete["feature_extraction_authorized_candidate_ids"], []
            )
            failed_id = first_requests[0]["candidate_id"]
            rejected = deepcopy(sidecars[failed_id])
            rejected["checks"]["family_terminal_hold_observed"] = False
            rejected["eligibility_decision"] = "reject"
            rejected["rejection_reasons"] = ["target terminal hold was not observed"]
            unsigned = dict(rejected)
            unsigned.pop("sidecar_digest")
            rejected["sidecar_digest"] = population.object_sha256(unsigned)
            one_failed = dict(sidecars)
            one_failed[failed_id] = rejected
            failed_cell = population.build_eligibility_index(
                bound_manifest, one_failed
            )
            self.assertEqual(failed_cell["rejected_candidate_count"], 1)
            self.assertEqual(failed_cell["eligible_cell_ids"], [])
            self.assertEqual(
                failed_cell["feature_extraction_authorized_candidate_ids"], []
            )
            complete = population.build_eligibility_index(bound_manifest, sidecars)
            self.assertEqual(complete["eligible_cell_ids"], [first_cell])
            self.assertEqual(
                len(complete["feature_extraction_authorized_candidate_ids"]), 10
            )
            labels = {
                row["requested_semantic_branch"]: row["critic_label"]
                for row in complete["candidate_rows"]
                if row["cell_id"] == first_cell
            }
            self.assertEqual(labels["action"], 1)
            self.assertTrue(all(labels[branch] == 0 for branch in population.SEMANTIC_BRANCHES[1:]))

    def test_prompt_disclosure_or_failed_action_conjunction_cannot_be_resealed_accept(self) -> None:
        request = self.bundle["audit_requests"]["candidate_requests"][0]
        self.assertEqual(request["requested_semantic_branch"], "action")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            disclosed = _accepted_sidecar(root, request)
            disclosed["generation_prompt_or_requested_branch_disclosed"] = True
            unsigned = dict(disclosed); unsigned.pop("sidecar_digest")
            disclosed["sidecar_digest"] = population.object_sha256(unsigned)
            with self.assertRaisesRegex(
                population.MosaicEventPopulationError, "prompt-blind"
            ):
                population.validate_event_audit(disclosed, request)

            failed = _accepted_sidecar(root, request)
            failed["checks"]["family_terminal_hold_observed"] = False
            unsigned = dict(failed); unsigned.pop("sidecar_digest")
            failed["sidecar_digest"] = population.object_sha256(unsigned)
            with self.assertRaisesRegex(
                population.MosaicEventPopulationError, "does not satisfy"
            ):
                population.validate_event_audit(failed, request)


if __name__ == "__main__":
    unittest.main()
