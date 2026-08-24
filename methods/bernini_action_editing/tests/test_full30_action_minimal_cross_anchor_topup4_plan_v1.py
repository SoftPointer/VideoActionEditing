from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import full30_action_minimal_cross_anchor_topup4_plan_v1 as plan_mod
import pair_v5_t2v_calibration_bank_spec as pair_contract


SELECTION = METHOD_ROOT / "assets/full30_action_minimal_cross_anchor_topup4_v1.json"
REGISTRY = METHOD_ROOT / "assets/mosaic_event_population_compact6_topup20_v1.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MinimalCrossAnchorTopup4PlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = Path(tempfile.mkdtemp(prefix="topup4-plan-test-")).resolve(
            strict=True
        )
        self.addCleanup(lambda: shutil.rmtree(self.temp, ignore_errors=True))

    def build(self, name: str = "plan") -> tuple[dict, Path]:
        output = self.temp / name
        value = plan_mod.materialize_plan(
            selection_path=SELECTION,
            expected_selection_sha256=sha(SELECTION),
            registry_path=REGISTRY,
            expected_registry_sha256=sha(REGISTRY),
            output_dir=output,
        )
        return value, output

    def test_exact_four_cells_eight_candidates_and_pair_envelopes(self) -> None:
        value, output = self.build()
        self.assertEqual(value["generation_invocation_count"], 8)
        self.assertEqual(value["seed_cell_count"], 4)
        self.assertEqual(len(value["tasks"]), 8)
        self.assertEqual(
            [row["candidate_count"] for row in value["shards"]], [4, 4]
        )
        self.assertEqual(
            [row["seed"] for row in value["cell_proofs"]],
            [2026081205, 2026081209, 2026081213, 2026081217],
        )
        self.assertEqual(
            [row["semantic_branch"] for row in value["tasks"]],
            list(plan_mod.REQUIRED_BRANCH_ORDER) * 4,
        )
        root_sha = value["root_spec"]["file_sha256"]
        for task in value["tasks"]:
            envelope = pair_contract.load_candidate_envelope(
                task["candidate_spec_path"], root_sha
            )
            self.assertEqual(envelope["candidate"]["candidate_id"], task["candidate_id"])
            self.assertEqual(envelope["candidate"]["analysis_split"], "fit")
            self.assertEqual(
                envelope["candidate"]["geometry_source_video_sha256"],
                plan_mod.GEOMETRY_SOURCE_SHA256,
            )
        plan_path = Path(value["_path"])
        replay, resolved, observed = plan_mod.load_plan(plan_path, sha(plan_path))
        self.assertEqual(resolved, plan_path)
        self.assertEqual(observed, sha(plan_path))
        self.assertEqual(replay["plan_digest"], value["plan_digest"])
        self.assertTrue((output / "candidate-plan").is_dir())

    def test_execution_authority_is_optimizer_free_and_requires_later_gates(self) -> None:
        value, _ = self.build()
        execution = value["execution_contract"]
        self.assertFalse(execution["generated_media_is_editor_input_or_target"])
        self.assertFalse(execution["optimizer_authorized"])
        self.assertTrue(execution["independent_full81_review_required"])
        self.assertTrue(execution["same_state_six_sigma_materialization_required_after_review"])
        self.assertTrue(
            execution[
                "same_state_noop_camera_appearance_and_wrong_control_forwards_required"
            ]
        )
        self.assertEqual(
            execution["diagnostic_only_rendered_branches_not_optimizer_required"],
            list(plan_mod.DIAGNOSTIC_ONLY_BRANCHES),
        )
        selection = json.loads(SELECTION.read_text(encoding="utf-8"))
        self.assertEqual(selection["optimizer_updates"], 0)
        self.assertFalse(selection["generated_media_may_train_editor"])
        self.assertFalse(selection["generalization_claim_authorized"])

    def test_candidate_and_audit_surfaces_are_exact_mechanical_projection(self) -> None:
        value, _ = self.build()
        root = json.loads(Path(value["root_spec"]["path"]).read_text(encoding="ascii"))
        validated = plan_mod.validate_root_value(root)
        candidates = [row for group in validated["groups"] for row in group["candidates"]]
        requests = validated["audit_requests"]
        self.assertEqual(len(candidates), len(requests))
        for candidate, request in zip(candidates, requests):
            self.assertEqual(
                request["candidate_id"],
                candidate["candidate_id"].replace(
                    "topup4-", "mosaic-full_topup20-v1-", 1
                ),
            )
            self.assertEqual(
                request["requested_semantic_branch"], candidate["semantic_branch"]
            )
            self.assertEqual(
                request["full_t2v_caption_utf8_sha256"],
                candidate["full_t2v_caption_utf8_sha256"],
            )
            self.assertEqual(
                hashlib.sha256(candidate["full_t2v_caption"].encode("utf-8")).hexdigest(),
                candidate["full_t2v_caption_utf8_sha256"],
            )

    def test_two_materializations_have_identical_root_and_envelope_bytes(self) -> None:
        first, first_root = self.build("one")
        second, second_root = self.build("two")
        self.assertEqual(
            Path(first["root_spec"]["path"]).read_bytes(),
            Path(second["root_spec"]["path"]).read_bytes(),
        )
        first_envelopes = sorted((first_root / "candidate-plan").glob("*/*.json"))
        second_envelopes = sorted((second_root / "candidate-plan").glob("*/*.json"))
        self.assertEqual(len(first_envelopes), 8)
        self.assertEqual(
            [path.name for path in first_envelopes],
            [path.name for path in second_envelopes],
        )
        self.assertEqual(
            [path.read_bytes() for path in first_envelopes],
            [path.read_bytes() for path in second_envelopes],
        )

    def test_tampered_candidate_envelope_is_rejected(self) -> None:
        value, _ = self.build()
        path = Path(value["tasks"][0]["candidate_spec_path"])
        os.chmod(path, 0o600)
        row = json.loads(path.read_text(encoding="ascii"))
        row["candidate"]["seed"] += 1
        path.write_bytes(plan_mod.canonical_json_bytes(row) + b"\n")
        os.chmod(path, 0o400)
        with self.assertRaisesRegex(
            plan_mod.MinimalCrossAnchorPlanError, "candidate envelope SHA"
        ):
            plan_mod.load_plan(value["_path"], sha(Path(value["_path"])))

    def test_root_projection_tamper_is_rejected_even_when_plan_is_resigned(self) -> None:
        value, _ = self.build()
        root_path = Path(value["root_spec"]["path"])
        os.chmod(root_path, 0o600)
        root = json.loads(root_path.read_text(encoding="ascii"))
        root["groups"][0]["cells"][0]["actor_group_id"] = "forged-actor"
        root_path.write_bytes(plan_mod.canonical_json_bytes(root) + b"\n")
        os.chmod(root_path, 0o400)
        plan_path = Path(value["_path"])
        os.chmod(plan_path, 0o600)
        plan = json.loads(plan_path.read_text(encoding="ascii"))
        plan["root_spec"]["file_sha256"] = sha(root_path)
        unsigned = dict(plan)
        unsigned.pop("plan_digest")
        plan["plan_digest"] = plan_mod.object_sha256(unsigned)
        plan_path.write_bytes(plan_mod.canonical_json_bytes(plan) + b"\n")
        os.chmod(plan_path, 0o400)
        with self.assertRaises(plan_mod.MinimalCrossAnchorPlanError):
            plan_mod.load_plan(plan_path, sha(plan_path))

    def test_selection_seed_tamper_is_rejected_even_with_new_expected_hash(self) -> None:
        selection_path = self.temp / "selection.json"
        selection = json.loads(SELECTION.read_text(encoding="utf-8"))
        selection["selected_seed_cells"][0]["seed"] += 1
        selection_path.write_bytes(plan_mod.canonical_json_bytes(selection) + b"\n")
        os.chmod(selection_path, 0o400)
        with self.assertRaisesRegex(
            plan_mod.MinimalCrossAnchorPlanError,
            "selection file authority differs",
        ):
            plan_mod.materialize_plan(
                selection_path=selection_path,
                expected_selection_sha256=sha(selection_path),
                registry_path=REGISTRY,
                expected_registry_sha256=sha(REGISTRY),
                output_dir=self.temp / "tampered-seed-plan",
            )

    def test_registry_tamper_cannot_bypass_selection_binding(self) -> None:
        registry_path = self.temp / "registry.json"
        registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
        registry["registry_id"] = "forged-registry"
        registry_path.write_bytes(plan_mod.canonical_json_bytes(registry) + b"\n")
        os.chmod(registry_path, 0o400)
        with self.assertRaisesRegex(
            plan_mod.MinimalCrossAnchorPlanError,
            "parent registry file authority differs",
        ):
            plan_mod.materialize_plan(
                selection_path=SELECTION,
                expected_selection_sha256=sha(SELECTION),
                registry_path=registry_path,
                expected_registry_sha256=sha(registry_path),
                output_dir=self.temp / "forged-registry-plan",
            )

    def test_all_materialized_files_are_read_only_regular_files(self) -> None:
        value, output = self.build()
        files = [path for path in output.rglob("*") if path.is_file()]
        self.assertEqual(len(files), 10)
        for path in files:
            self.assertFalse(path.is_symlink())
            self.assertEqual(path.stat().st_mode & 0o777, 0o400)
        self.assertEqual(Path(value["_path"]).stat().st_mode & 0o777, 0o400)


if __name__ == "__main__":
    unittest.main()
