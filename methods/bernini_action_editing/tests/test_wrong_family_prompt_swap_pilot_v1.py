from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import wrong_family_prompt_swap_pilot_v1 as pilot  # noqa: E402


REGISTRY = METHOD_ROOT / "assets" / pilot.REGISTRY_ASSET_BASENAME
SOURCE_BANK = METHOD_ROOT / "assets" / pilot.SOURCE_BANK_BASENAME
ASSET_ROOT = METHOD_ROOT / "assets"


def _reseal(value: dict, digest_field: str) -> dict:
    result = copy.deepcopy(value)
    result.pop(digest_field, None)
    result[digest_field] = pilot.sha256_bytes(pilot.canonical_json_bytes(result))
    return result


class WrongFamilyRegistryAndPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry, cls.registry_digest = pilot.load_registry(
            REGISTRY.resolve(), pilot.REGISTRY_RAW_SHA256
        )
        cls.source_bank, cls.source_digest = pilot.load_source_bank(
            SOURCE_BANK.resolve(), cls.registry
        )
        cls.plan = pilot.build_generation_plan(
            registry_path=REGISTRY.resolve(),
            expected_registry_sha256=pilot.REGISTRY_RAW_SHA256,
            source_bank_path=SOURCE_BANK.resolve(),
            seed_scan_roots=[ASSET_ROOT.resolve()],
        )

    def test_pinned_authorities_and_retrospective_media_are_nonanalytic(self) -> None:
        self.assertEqual(self.registry_digest, pilot.REGISTRY_RAW_SHA256)
        self.assertEqual(self.source_digest, pilot.SOURCE_BANK_RAW_SHA256)
        self.assertEqual(
            [cell["iid"] for cell in self.registry["cells"][:2]],
            list(pilot.RETROSPECTIVE_IIDS),
        )
        for cell in self.registry["cells"]:
            self.assertTrue(cell["source_media_seen_before_seal"])
            for prompt in cell["query_prompts"].values():
                self.assertEqual(
                    hashlib.sha256(prompt["utf8_text"].encode("utf-8")).hexdigest(),
                    prompt["utf8_sha256"],
                )
        for excluded in self.plan["retrospective_discovery_exclusions"]:
            self.assertTrue(excluded["media_seen_before_registry_seal"])
            self.assertEqual(excluded["allowed_role"], "rubric_discovery_only")
            self.assertFalse(excluded["fit_confirmation_threshold_or_optimizer_use"])
            self.assertEqual(len(excluded["source_candidate_ids"]), 10)

    def test_plan_is_exact20_generation_exact24_a10_plus_b2_audit(self) -> None:
        self.assertEqual(self.plan["candidate_count"], 20)
        self.assertEqual(self.plan["judgment_count"], 24)
        self.assertFalse(self.plan["query_prompts_are_generation_captions"])
        self.assertEqual(
            self.plan["official_gaussian_binding_status"],
            "required_post_render_before_any_query_or_audit",
        )
        self.assertFalse(self.plan["interpretation_contract"]["editor"])
        self.assertFalse(self.plan["interpretation_contract"]["scientific_critic"])
        self.assertFalse(self.plan["interpretation_contract"]["optimizer_authorized"])
        for cell in self.plan["prospective_cells"]:
            self.assertEqual(cell["fresh_seed"], pilot.FRESH_SEEDS[cell["iid"]])
            candidates = cell["generation_candidates"]
            self.assertEqual(
                [row["semantic_branch"] for row in candidates],
                list(pilot.BRANCH_ORDER),
            )
            self.assertEqual(len(candidates), 10)
            requirements = cell["judgment_requirements"]
            self.assertEqual(len(requirements), 12)
            a_family = cell["family_rubrics"]["a"]["evaluated_family_id"]
            b_family = cell["family_rubrics"]["b"]["evaluated_family_id"]
            a_rows = [row for row in requirements if row["evaluated_family_id"] == a_family]
            b_rows = [row for row in requirements if row["evaluated_family_id"] == b_family]
            self.assertEqual([row["semantic_branch"] for row in a_rows], list(pilot.BRANCH_ORDER))
            self.assertEqual(
                {row["semantic_branch"]: row["required_outcome"] for row in a_rows},
                {branch: branch == "action" for branch in pilot.BRANCH_ORDER},
            )
            self.assertEqual(
                [(row["semantic_branch"], row["required_outcome"]) for row in b_rows],
                [("action", False), ("reverse", True)],
            )
            null_hashes = {row["common_null_utf8_sha256"] for row in requirements}
            self.assertEqual(
                null_hashes, {cell["query_prompts"]["common_null"]["utf8_sha256"]}
            )
            self.assertTrue(
                all(
                    row["full_t2v_caption_utf8_sha256"]
                    not in {
                        prompt["utf8_sha256"]
                        for prompt in cell["query_prompts"].values()
                    }
                    for row in candidates
                )
            )

    def test_generation_captions_and_geometry_copy_only_the_sealed_source(self) -> None:
        source_rows = pilot._source_rows_by_iid(self.source_bank)
        mutable = {"candidate_id", "calibration_group_id", "seed"}
        for cell in self.plan["prospective_cells"]:
            for old, new in zip(source_rows[cell["iid"]], cell["generation_candidates"]):
                self.assertEqual(
                    {key: value for key, value in old.items() if key not in mutable},
                    {key: value for key, value in new.items() if key not in mutable},
                )
                self.assertEqual(new["seed"], pilot.FRESH_SEEDS[cell["iid"]])

    def test_registry_and_plan_mutations_fail_closed(self) -> None:
        changed = copy.deepcopy(self.registry)
        changed["cells"][2]["fresh_seed"] += 1
        with self.assertRaises(pilot.WrongFamilyPromptSwapError):
            pilot.validate_registry(changed)

        changed = copy.deepcopy(self.registry)
        changed["cells"][2]["query_prompts"]["common_null"]["utf8_text"] += " changed"
        with self.assertRaises(pilot.WrongFamilyPromptSwapError):
            pilot.validate_registry(changed)

        changed_plan = copy.deepcopy(self.plan)
        changed_plan["prospective_cells"][0]["fresh_seed"] += 1
        changed_plan = _reseal(changed_plan, "generation_plan_digest")
        with self.assertRaises(pilot.WrongFamilyPromptSwapError):
            pilot.validate_generation_plan(changed_plan)

        changed_plan = copy.deepcopy(self.plan)
        changed_plan["interpretation_contract"]["optimizer_authorized"] = True
        changed_plan = _reseal(changed_plan, "generation_plan_digest")
        with self.assertRaises(pilot.WrongFamilyPromptSwapError):
            pilot.validate_generation_plan(changed_plan)

    def test_full_inventory_seed_collision_nulls_tuple_without_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            collision = Path(directory).resolve() / "other-bank.json"
            collision.write_text(
                json.dumps({"unrelated": {"seed": 2026081301}}), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                pilot.WrongFamilyPromptSwapError, "on-site replacement is forbidden"
            ):
                pilot.build_generation_plan(
                    registry_path=REGISTRY.resolve(),
                    expected_registry_sha256=pilot.REGISTRY_RAW_SHA256,
                    source_bank_path=SOURCE_BANK.resolve(),
                    seed_scan_roots=[ASSET_ROOT.resolve(), Path(directory).resolve()],
                )

    def test_seed_inventory_must_include_the_authenticated_registry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            harmless = Path(directory).resolve() / "bank.json"
            harmless.write_text(json.dumps({"seed": 123}), encoding="utf-8")
            with self.assertRaisesRegex(
                pilot.WrongFamilyPromptSwapError, "must include the authenticated registry"
            ):
                pilot.build_seed_collision_audit(
                    registry_path=REGISTRY.resolve(),
                    registry_raw_sha256=pilot.REGISTRY_RAW_SHA256,
                    scan_roots=[Path(directory).resolve()],
                )


class WrongFamilyGaussianAndDetachedAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.registry, _ = pilot.load_registry(REGISTRY.resolve(), pilot.REGISTRY_RAW_SHA256)
        self.plan = pilot.build_generation_plan(
            registry_path=REGISTRY.resolve(),
            expected_registry_sha256=pilot.REGISTRY_RAW_SHA256,
            source_bank_path=SOURCE_BANK.resolve(),
            seed_scan_roots=[ASSET_ROOT.resolve()],
        )
        bindings = []
        for cell_index, cell in enumerate(self.plan["prospective_cells"]):
            raw_digest = hashlib.sha256(("raw-cell-%d" % cell_index).encode()).hexdigest()
            content_digest = hashlib.sha256(("content-cell-%d" % cell_index).encode()).hexdigest()
            shape = [1, 16, 21, 116 + cell_index, 96 + cell_index]
            for candidate in cell["generation_candidates"]:
                stem = candidate["candidate_id"]
                receipt = self.root / (stem + ".receipt.json")
                mp4 = self.root / (stem + ".mp4")
                gaussian = self.root / (stem + ".gaussian.safetensors")
                mp4.write_bytes(("mp4:" + stem).encode())
                gaussian.write_bytes(("gaussian-container:" + stem).encode())
                row = {
                    "candidate_id": stem,
                    "seed": candidate["seed"],
                    "candidate_receipt_path": str(receipt),
                    "candidate_receipt_sha256": "0" * 64,
                    "mp4_path": str(mp4),
                    "mp4_sha256": pilot.file_sha256(mp4),
                    "official_gaussian_path": str(gaussian),
                    "official_gaussian_artifact_sha256": pilot.file_sha256(gaussian),
                    "raw_value_sha256": raw_digest,
                    "content_sha256": content_digest,
                    "tensor_key": "official_initial_gaussian",
                    "shape": shape,
                    "dtype": "torch.float32",
                    "stored_dtype": "torch.float32",
                    "generator_initial_seed": candidate["seed"],
                    "captured_from_native_sampler": True,
                    "external_initial_noise_injection": False,
                    "source_or_target_derived": False,
                }
                renderer_receipt = {
                    "schema_version": "test-native-renderer-receipt-v1",
                    "candidate": candidate,
                    "sampling_contract": self.plan["sampling_contract"],
                    "artifacts": {
                        "mp4": {
                            "path": row["mp4_path"],
                            "sha256": row["mp4_sha256"],
                            "frame_count": 81,
                            "fps": 25,
                        },
                        "official_initial_gaussian": {
                            "path": row["official_gaussian_path"],
                            "sha256": row["official_gaussian_artifact_sha256"],
                            "raw_value_sha256": row["raw_value_sha256"],
                            "content_sha256": row["content_sha256"],
                            "tensor_key": row["tensor_key"],
                            "shape": row["shape"],
                            "dtype": row["dtype"],
                            "stored_dtype": row["stored_dtype"],
                            "generator_initial_seed": row["generator_initial_seed"],
                            "captured_from_native_sampler": True,
                            "external_initial_noise_injection": False,
                            "source_or_target_derived": False,
                        },
                    },
                }
                renderer_receipt["receipt_digest"] = pilot.sha256_bytes(
                    pilot.canonical_json_bytes(renderer_receipt)
                )
                receipt.write_bytes(pilot.canonical_json_bytes(renderer_receipt) + b"\n")
                row["candidate_receipt_sha256"] = pilot.file_sha256(receipt)
                bindings.append(row)
        self.binding = pilot.build_gaussian_binding_manifest(self.plan, bindings)
        self.audit_plan = pilot.build_audit_plan(self.plan, self.binding)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _completed(self) -> dict:
        return {
            "schema_version": pilot.COMPLETED_AUDIT_SCHEMA,
            "audit_plan_digest": self.audit_plan["audit_plan_digest"],
            "judgments": [
                {
                    "candidate_id": row["candidate_id"],
                    "evaluated_family_id": row["evaluated_family_id"],
                    "decision": "true" if row["required_outcome"] else "false",
                    "full_exact81_viewed": True,
                    "generation_prompt_hidden": True,
                    "reviewer_id": "detached-reviewer-01",
                    "review_notes": "full clip reviewed under the sealed family rubric",
                    "mp4_sha256": row["mp4_sha256"],
                    "rubric_sha256": row["rubric_sha256"],
                }
                for row in self.audit_plan["judgments"]
            ],
        }

    def _refresh_renderer_receipt(self, row: dict) -> None:
        path = Path(row["candidate_receipt_path"])
        receipt = json.loads(path.read_text(encoding="utf-8"))
        receipt.pop("receipt_digest")
        gaussian = receipt["artifacts"]["official_initial_gaussian"]
        for field in (
            "path",
            "raw_value_sha256",
            "content_sha256",
            "tensor_key",
            "shape",
            "dtype",
            "stored_dtype",
            "generator_initial_seed",
            "captured_from_native_sampler",
            "external_initial_noise_injection",
            "source_or_target_derived",
        ):
            source_field = (
                "official_gaussian_path" if field == "path" else field
            )
            gaussian[field] = row[source_field]
        gaussian["sha256"] = row["official_gaussian_artifact_sha256"]
        receipt["receipt_digest"] = pilot.sha256_bytes(
            pilot.canonical_json_bytes(receipt)
        )
        path.write_bytes(pilot.canonical_json_bytes(receipt) + b"\n")
        row["candidate_receipt_sha256"] = pilot.file_sha256(path)

    def test_gaussian_binding_is_per_cell_exact_and_audit_plan_is_24(self) -> None:
        checked = pilot.validate_gaussian_binding(self.binding, self.plan)
        self.assertEqual(len(checked["bindings"]), 20)
        self.assertEqual(self.audit_plan["judgment_count"], 24)
        self.assertEqual(
            len(
                {
                    (row["candidate_id"], row["evaluated_family_id"])
                    for row in self.audit_plan["judgments"]
                }
            ),
            24,
        )
        for row in self.audit_plan["judgments"]:
            self.assertTrue(row["full_exact81_required"])
            self.assertTrue(row["generation_prompt_must_be_hidden"])

    def test_one_branch_gaussian_value_or_artifact_tamper_fails(self) -> None:
        changed = copy.deepcopy(self.binding)
        changed["bindings"][1]["raw_value_sha256"] = "e" * 64
        self._refresh_renderer_receipt(changed["bindings"][1])
        changed = _reseal(changed, "gaussian_binding_digest")
        with self.assertRaisesRegex(
            pilot.WrongFamilyPromptSwapError, "reuse one exact Gaussian"
        ):
            pilot.validate_gaussian_binding(changed, self.plan)

        changed = copy.deepcopy(self.binding)
        Path(changed["bindings"][0]["official_gaussian_path"]).write_bytes(b"tampered")
        with self.assertRaisesRegex(pilot.WrongFamilyPromptSwapError, "bytes differ"):
            pilot.validate_gaussian_binding(changed, self.plan)

    def test_seed_only_or_cross_cell_gaussian_alias_fails(self) -> None:
        changed = copy.deepcopy(self.binding)
        changed["bindings"][0].pop("raw_value_sha256")
        changed = _reseal(changed, "gaussian_binding_digest")
        with self.assertRaises(pilot.WrongFamilyPromptSwapError):
            pilot.validate_gaussian_binding(changed, self.plan)

        changed = copy.deepcopy(self.binding)
        first_raw = changed["bindings"][0]["raw_value_sha256"]
        first_content = changed["bindings"][0]["content_sha256"]
        for row in changed["bindings"][10:]:
            row["raw_value_sha256"] = first_raw
            row["content_sha256"] = first_content
            self._refresh_renderer_receipt(row)
        changed = _reseal(changed, "gaussian_binding_digest")
        with self.assertRaisesRegex(pilot.WrongFamilyPromptSwapError, "colliding Gaussian"):
            pilot.validate_gaussian_binding(changed, self.plan)

    def test_passing_audit_remains_editor_scientific_optimizer_false(self) -> None:
        receipt = pilot.validate_completed_audit(self.audit_plan, self._completed())
        self.assertTrue(receipt["prospective_tuple_pass"])
        self.assertTrue(receipt["pilot_family_swap_evidence_usable"])
        self.assertFalse(receipt["editor"])
        self.assertFalse(receipt["scientific_critic"])
        self.assertFalse(receipt["optimizer_authorized"])
        self.assertFalse(receipt["fresh_confirmation_enters_optimizer"])

    def test_unknown_ambiguous_wrong_or_missing_judgment_fails_whole_tuple(self) -> None:
        for decision in ("unknown", "ambiguous"):
            completed = self._completed()
            completed["judgments"][0]["decision"] = decision
            with self.subTest(decision=decision):
                with self.assertRaisesRegex(
                    pilot.WrongFamilyPromptSwapError, "entire prospective tuple"
                ):
                    pilot.validate_completed_audit(self.audit_plan, completed)

        completed = self._completed()
        completed["judgments"][0]["decision"] = (
            "false" if completed["judgments"][0]["decision"] == "true" else "true"
        )
        with self.assertRaisesRegex(
            pilot.WrongFamilyPromptSwapError, "violates preregistered outcome"
        ):
            pilot.validate_completed_audit(self.audit_plan, completed)

        completed = self._completed()
        completed["judgments"].pop()
        with self.assertRaisesRegex(pilot.WrongFamilyPromptSwapError, "exactly 24"):
            pilot.validate_completed_audit(self.audit_plan, completed)

    def test_blinding_full81_media_and_rubric_bindings_are_hard_gates(self) -> None:
        mutations = {
            "full_exact81_viewed": False,
            "generation_prompt_hidden": False,
            "mp4_sha256": "0" * 64,
            "rubric_sha256": "1" * 64,
        }
        for field, replacement in mutations.items():
            completed = self._completed()
            completed["judgments"][0][field] = replacement
            with self.subTest(field=field):
                with self.assertRaises(pilot.WrongFamilyPromptSwapError):
                    pilot.validate_completed_audit(self.audit_plan, completed)


if __name__ == "__main__":
    unittest.main()
