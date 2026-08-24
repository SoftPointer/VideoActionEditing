#!/usr/bin/env python3
"""Local trust, claim-boundary, and review regressions for v15c-r9."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = METHOD_ROOT.parents[1]
AUTHORITY_PATH = (
    METHOD_ROOT / "assets/e00_source_four_role_authority_v15c_r9.json"
)
RELEASE_PATH = (
    METHOD_ROOT / "assets/e00_source_four_role_authority_v15c_r9_release.json"
)
ROLE_ASSET_PATH = (
    METHOD_ROOT / "assets/interaction_e00_source_instance_role_token_spans_v15b.json"
)
CORE_PATH = METHOD_ROOT / "source_role_authority_v15c_r9.py"
VALIDATOR_PATH = METHOD_ROOT / "validate_source_role_authority_v15c_r9.py"
RUNNER_PATH = METHOD_ROOT / "run_source_four_role_authority_v15c_r9.py"
OWNERSHIP_PATH = METHOD_ROOT / "materialize_source_role_ownership_v15c_r9.py"
POSTFLIGHT_PATH = METHOD_ROOT / "postflight_source_four_role_authority_v15c_r9.py"
BUILDER_PATH = METHOD_ROOT / "tools/build_source_four_role_authority_v15c_r9_review.py"
PRODUCTION_PATHS = (
    CORE_PATH,
    VALIDATOR_PATH,
    RUNNER_PATH,
    OWNERSHIP_PATH,
    POSTFLIGHT_PATH,
    BUILDER_PATH,
)
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))


def load_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("module loader unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


VALIDATOR = load_path("validate_source_role_authority_v15c_r9_test", VALIDATOR_PATH)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AuthorityClosureTests(unittest.TestCase):
    def test_exact_authority_token_source_and_r8_base_validate(self):
        authority, base, role_asset = VALIDATOR.validate_authority(
            root=REPO_ROOT, authority_path=AUTHORITY_PATH
        )
        self.assertEqual(authority["source"]["video_sha256"], base["source"]["sha256"])
        self.assertEqual(
            authority["token_source_authority"]["asset_internal_sha256"],
            role_asset["asset_sha256"],
        )
        self.assertEqual(
            [row["r9_role"] for row in authority["token_source_authority"]["role_channel_binding"]],
            ["human_agent", "old_actor", "new_actor", "recipient"],
        )
        self.assertFalse(authority["claim_limits"]["route_authorized"])
        self.assertFalse(authority["claim_limits"]["training_authorized"])
        self.assertFalse(authority["claim_limits"]["scientific_claim_authorized"])
        self.assertFalse(authority["claim_limits"]["mechanical_candidate_qualified"])

    def test_mutated_authority_or_role_asset_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in (
                VALIDATOR.AUTHORITY_RELATIVE_PATH,
                VALIDATOR.ROLE_ASSET_RELATIVE_PATH,
                VALIDATOR.BASE_SPEC_RELATIVE_PATH,
                VALIDATOR.BASE_RELEASE_RELATIVE_PATH,
            ):
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(REPO_ROOT / relative, destination)
            authority_path = root / VALIDATOR.AUTHORITY_RELATIVE_PATH
            mutated = json.loads(authority_path.read_text(encoding="utf-8"))
            mutated["source"]["video_sha256"] = "0" * 64
            authority_path.write_text(json.dumps(mutated), encoding="utf-8")
            with self.assertRaises(VALIDATOR.ValidateSourceRoleAuthorityV15CR9Error):
                VALIDATOR.validate_authority(root=root, authority_path=authority_path)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in (
                VALIDATOR.AUTHORITY_RELATIVE_PATH,
                VALIDATOR.ROLE_ASSET_RELATIVE_PATH,
                VALIDATOR.BASE_SPEC_RELATIVE_PATH,
                VALIDATOR.BASE_RELEASE_RELATIVE_PATH,
            ):
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(REPO_ROOT / relative, destination)
            role_path = root / VALIDATOR.ROLE_ASSET_RELATIVE_PATH
            mutated = json.loads(role_path.read_text(encoding="utf-8"))
            mutated["event"]["roles"][0]["token_ids"][0] += 1
            role_path.write_text(json.dumps(mutated), encoding="utf-8")
            with self.assertRaises(VALIDATOR.ValidateSourceRoleAuthorityV15CR9Error):
                VALIDATOR.validate_authority(
                    root=root,
                    authority_path=root / VALIDATOR.AUTHORITY_RELATIVE_PATH,
                )

    def test_common_null_is_no_go_and_future_joint_max_t_is_preregistered(self):
        authority = json.loads(AUTHORITY_PATH.read_text(encoding="utf-8"))
        roles = authority["role_assignment"]
        r6 = authority["r6_affinity_authority"]
        self.assertEqual(r6["null_tensor_shape"], [5, 64, 21, 37, 25])
        self.assertFalse(r6["four_role_joint_null_axis_available"])
        self.assertFalse(r6["common_null_broadcast_for_certification"])
        self.assertEqual(r6["fwer_status"], "FOUR_ROLE_JOINT_FWER_UNCERTIFIED")
        self.assertIn("role_indexed_null_tensor", roles["global_multiple_comparison_control"])
        self.assertAlmostEqual(roles["future_joint_null_minimum_attainable_p"], 1.0 / 65.0)
        self.assertFalse(roles["four_role_joint_fwer_certified"])
        self.assertTrue(roles["mechanical_candidate_must_remain_false"])
        self.assertTrue(roles["vessel_three_role_bonferroni_additional_gate"])
        source = CORE_PATH.read_text(encoding="utf-8")
        self.assertIn("global_max_null_track", source)
        self.assertIn("global_four_role_max_t_fwer", source)
        self.assertIn("vessel_three_role_bonferroni_extra_gate", source)
        self.assertIn("common_null_broadcast_used_for_certification", source)


class OwnershipAndReviewStaticTests(unittest.TestCase):
    def test_overlap_is_not_fed_directly_to_v15b(self):
        source = CORE_PATH.read_text(encoding="utf-8")
        self.assertIn("raw_overlapping_proposals_passed_to_v15b", source)
        self.assertIn("strict_argmax_of_replayed_raw_signed_valued_sam2_logits_else_unassigned", source)
        self.assertIn("ownership_hole_and_component_topology_matches_proposal_every_frame", source)
        self.assertIn("morphological_repair_applied", source)
        self.assertIn("contact_relation_mask_is_independent", source)

    def test_ownership_reopens_both_raw_logit_runs_and_exact_positive_masks(self):
        source = OWNERSHIP_PATH.read_text(encoding="utf-8")
        self.assertIn("for run_index, run in enumerate(evidence[\"runs\"])", source)
        self.assertIn("selected replayed raw logits differ across runs", source)
        self.assertIn("selected mask is not replayed raw logit positive set", source)
        self.assertIn("observer_evidence.strict_safetensors", source)
        self.assertNotIn("external_signature_or_tee_verified\": True", source)

    def test_postflight_recomputes_partition_and_has_non_decorative_gates(self):
        source = POSTFLIGHT_PATH.read_text(encoding="utf-8")
        self.assertIn("partition_source_role_ownership_v15c_r9", source)
        self.assertIn("adapt_qualified_ownership_to_v15b_v15c_r9", source)
        self.assertIn("selected_rows", source)
        self.assertNotIn("{key: True for key in GATE_KEYS}", source)
        self.assertIn("raw_overlapping_proposals_passed_to_v15b", source)

    def test_overlay_has_exact_six_views_two_rows_and_five_frame_sheets(self):
        source = BUILDER_PATH.read_text(encoding="utf-8")
        self.assertIn('VIEW_KEYS = ("source", "all", *core.ROLE_NAMES)', source)
        self.assertIn('(\"source\", \"all\", \"human_agent\")', source)
        self.assertIn('(\"old_actor\", \"new_actor\", \"recipient\")', source)
        self.assertIn("DISPLAY_FRAMES = (0, 20, 40, 60, 80)", source)
        self.assertIn("contact magenta", source)
        self.assertIn("reject_only", source)
        self.assertIn("approve_action_available", source)
        self.assertNotIn("Approve candidate", source)

    def test_production_modules_have_no_assert_or_authorization_true_literals(self):
        for path in PRODUCTION_PATHS:
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("assert ", source, path.name)
            for forbidden in (
                '"remote_worker_execution_verified": True',
                '"observer_execution_authorized": True',
                '"localization_semantically_certified": True',
                '"scientific_claim_authorized": True',
                '"route_authorized": True',
                '"decode_authorized": True',
                '"training_authorized": True',
            ):
                self.assertNotIn(forbidden, source, path.name)
        called = set()
        for path in PRODUCTION_PATHS:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            called.update(
                node.func.attr
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
            )
        for forbidden in ("backward", "step", "zero_grad", "decode"):
            self.assertNotIn(forbidden, called)


class ReleaseManifestTests(unittest.TestCase):
    def test_release_manifest_exactly_pins_every_runtime_member(self):
        release = json.loads(RELEASE_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            release["schema_version"],
            "bernini-source-four-role-authority-v15c-r9-local-release",
        )
        self.assertEqual(release["tag"], "v15c-r9-local-source-observer-only")
        self.assertEqual(release["member_count"], len(release["members"]))
        self.assertEqual(release["member_count"], 14)
        self.assertEqual(
            release["status"],
            "LOCAL_COMMON_NULL_DIAGNOSTIC_ONLY_FOUR_ROLE_JOINT_FWER_UNCERTIFIED",
        )
        self.assertEqual(release["r6_common_null_scope"], "COMMON_NULL_DIAGNOSTIC_ONLY")
        self.assertFalse(release["four_role_joint_fwer_certified"])
        self.assertFalse(release["mechanical_candidate_qualified"])
        self.assertFalse(release["observer_execution_authorized"])
        self.assertFalse(release["scientific_claim_authorized"])
        self.assertFalse(release["route_authorized"])
        self.assertFalse(release["decode_authorized"])
        self.assertFalse(release["training_authorized"])
        paths = []
        for row in release["members"]:
            path = REPO_ROOT / row["path"]
            paths.append(row["path"])
            self.assertTrue(path.is_file())
            self.assertFalse(path.is_symlink())
            self.assertEqual(digest(path), row["sha256"])
            self.assertEqual(path.stat().st_size, row["size"])
        self.assertEqual(paths, sorted(paths))
        payload = dict(release)
        claimed = payload.pop("release_sha256")
        self.assertEqual(claimed, hashlib.sha256(
            (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")
        ).hexdigest())


if __name__ == "__main__":
    unittest.main()
