from __future__ import annotations

import copy
import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

import torch


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import full644_target_free_preference_v1 as target_free


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sealed(value: dict, field: str) -> dict:
    result = copy.deepcopy(value)
    result[field] = target_free.object_sha256(result)
    return result


def source_row(index: int) -> dict:
    instruction = (
        f"Keep source identity and scene while the primary actor performs "
        f"action family {index % target_free.ACTION_FAMILY_COUNT:02d}."
    )
    value = {
        "schema_version": target_free.SOURCE_ROW_SCHEMA,
        "row_id": f"row-{index:04d}",
        "group_id": f"group-{index:04d}",
        "action_family": f"family-{index % target_free.ACTION_FAMILY_COUNT:02d}",
        "source_video_path": f"/dataset/source/{index:04d}.mp4",
        "source_video_sha256": sha(f"source-{index}"),
        "source_frame_count": target_free.FRAME_COUNT,
        "source_fps": target_free.FPS,
        "instruction": instruction,
        "instruction_sha256": hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
        "upstream_preview_row_digest": sha(f"upstream-preview-row-{index}"),
    }
    return sealed(value, "row_digest")


def source_manifest() -> dict:
    value = {
        "schema_version": target_free.SOURCE_SCHEMA,
        "training_mode": target_free.TRAINING_MODE,
        "source_count": target_free.SOURCE_COUNT,
        "action_family_count": target_free.ACTION_FAMILY_COUNT,
        "rows": [source_row(index) for index in range(target_free.SOURCE_COUNT)],
        "source_authority": dict(target_free.PINNED_FULL644_SOURCE_AUTHORITY),
        "row_order": "ascii_ascending_row_id",
        "input_closure": dict(target_free.SOURCE_INPUT_CLOSURE),
    }
    return sealed(value, "manifest_digest")


def rollout(
    name: str,
    *,
    policy_sha: str,
    round_index: int,
    passed: bool,
    source: dict,
) -> dict:
    axes = {axis: True for axis in target_free.HARD_AXES}
    failures: list[str] = []
    if not passed:
        axes["terminal_hold"] = False
        failures = ["terminal_hold_failed"]
    value = {
        "schema_version": target_free.ROLLOUT_SCHEMA,
        "rollout_id": name,
        "policy_sha256": policy_sha,
        "round_index": round_index,
        "seed": 100 if passed else 101,
        "source_row_id": source["row_id"],
        "source_video_sha256": source["source_video_sha256"],
        "instruction_sha256": source["instruction_sha256"],
        "trajectory_receipt_path": f"/rollouts/{name}/trajectory.json",
        "trajectory_receipt_sha256": sha(f"{name}-trajectory"),
        "output_media_path": f"/rollouts/{name}/output.mp4",
        "output_media_sha256": sha(f"{name}-media"),
        "verifier_receipt_path": f"/rollouts/{name}/verifier.json",
        "verifier_receipt_sha256": sha(f"{name}-verifier"),
        "axis_pass": axes,
        "failure_tags": failures,
    }
    return sealed(value, "rollout_digest")


def preference_manifest(*, include_pair: bool = True) -> dict:
    policy_sha = sha("policy-before-round-0")
    pairs: list[dict] = []
    if include_pair:
        source = source_row(0)
        value = {
            "schema_version": target_free.PREFERENCE_PAIR_SCHEMA,
            "pair_id": "pair-0000",
            "source_row_id": source["row_id"],
            "source_video_sha256": source["source_video_sha256"],
            "instruction_sha256": source["instruction_sha256"],
            "chosen_rollout": rollout(
                "chosen-0000",
                policy_sha=policy_sha,
                round_index=0,
                passed=True,
                source=source,
            ),
            "rejected_rollout": rollout(
                "rejected-0000",
                policy_sha=policy_sha,
                round_index=0,
                passed=False,
                source=source,
            ),
        }
        pairs.append(sealed(value, "pair_digest"))
    value = {
        "schema_version": target_free.PREFERENCE_SCHEMA,
        "training_mode": target_free.TRAINING_MODE,
        "behavior_policy_sha256": policy_sha,
        "source_manifest_sha256": sha("source-manifest-file"),
        "source_manifest_digest": source_manifest()["manifest_digest"],
        "round_index": 0,
        "pair_count": len(pairs),
        "pairs": pairs,
        "verifier_qualification": {
            "schema_version": target_free.VERIFIER_QUALIFICATION_SCHEMA,
            "verifier_release_sha256": sha("verifier-release"),
            "verifier_model_sha256": sha("verifier-model"),
            "qualification_set_sha256": sha("verifier-qualification-set"),
            "independent_from_student": True,
            "hard_axis_conjunction": list(target_free.HARD_AXES),
            "scalar_compensation_allowed": False,
        },
        "input_closure": dict(target_free.PREFERENCE_INPUT_CLOSURE),
    }
    return sealed(value, "preference_set_digest")


def reseal_preference(value: dict) -> dict:
    result = copy.deepcopy(value)
    for pair in result["pairs"]:
        for role in ("chosen_rollout", "rejected_rollout"):
            pair[role].pop("rollout_digest", None)
            pair[role] = sealed(pair[role], "rollout_digest")
        pair.pop("pair_digest", None)
        pair.update(sealed(pair, "pair_digest"))
    result.pop("preference_set_digest", None)
    return sealed(result, "preference_set_digest")


class Full644TargetFreePreferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source_value = source_manifest()
        cls.source = target_free.validate_source_catalog_value(
            cls.source_value,
            manifest_sha256=sha("source-manifest-file"),
        )

    def preference(self, *, include_pair: bool = True):
        value = preference_manifest(include_pair=include_pair)
        return target_free.validate_preference_set_value(
            value,
            source_catalog=self.source,
            preference_set_sha256=sha("preference-set-file"),
        )

    def test_source_catalog_is_exact644_exact28_and_has_no_target_fields(self) -> None:
        self.assertEqual(len(self.source.rows), 644)
        self.assertEqual(len({row.action_family for row in self.source.rows}), 28)
        serialized = target_free.canonical_json_bytes(self.source_value).decode("ascii")
        self.assertNotIn("video_vae_latents", serialized)
        self.assertNotIn("target_video_path", serialized)
        self.assertNotIn("synthetic_target", serialized)

    def test_old_pair_container_and_renamed_target_are_rejected_before_access(self) -> None:
        for forbidden_key in (
            "video_vae_latents",
            "target_video",
            "positive_video",
            "teacher_unit",
            "flow_target",
        ):
            value = copy.deepcopy(self.source_value)
            value["rows"][0][forbidden_key] = object()
            with self.subTest(forbidden_key=forbidden_key):
                with self.assertRaisesRegex(
                    target_free.TargetFreeTrainingError, "fields differ"
                ):
                    target_free.validate_source_catalog_value(
                        value, manifest_sha256=sha("source-manifest-file")
                    )

    def test_preference_pair_is_same_source_same_instruction_current_policy(self) -> None:
        preference = self.preference()
        self.assertEqual(len(preference.pairs), 1)
        pair = preference.pairs[0]
        self.assertTrue(pair.chosen.passes_all_axes)
        self.assertFalse(pair.rejected.passes_all_axes)
        self.assertEqual(pair.chosen.policy_sha256, preference.behavior_policy_sha256)
        self.assertEqual(pair.source.row_id, "row-0000")

    def test_preference_source_catalog_and_rollout_source_binding_are_exact(self) -> None:
        wrong_catalog = preference_manifest()
        wrong_catalog["source_manifest_sha256"] = sha("another-source-manifest")
        wrong_catalog = reseal_preference(wrong_catalog)
        with self.assertRaisesRegex(
            target_free.TargetFreeTrainingError, "source catalogue binding"
        ):
            target_free.validate_preference_set_value(
                wrong_catalog,
                source_catalog=self.source,
                preference_set_sha256=sha("preference-set-file"),
            )

        wrong_rollout = preference_manifest()
        wrong_rollout["pairs"][0]["chosen_rollout"]["source_row_id"] = "row-0001"
        wrong_rollout = reseal_preference(wrong_rollout)
        with self.assertRaisesRegex(
            target_free.TargetFreeTrainingError, "source/instruction binding"
        ):
            target_free.validate_preference_set_value(
                wrong_rollout,
                source_catalog=self.source,
                preference_set_sha256=sha("preference-set-file"),
            )

    def test_preference_types_and_verifier_schema_are_closed(self) -> None:
        for mutation, message in (
            (
                lambda value: value["input_closure"].__setitem__(
                    "paired_edited_target_present", 0
                ),
                "target-free closure",
            ),
            (lambda value: value.__setitem__("round_index", False), "round_index"),
            (lambda value: value.__setitem__("pair_count", True), "pair count"),
            (
                lambda value: value["verifier_qualification"].__setitem__(
                    "teacher_unit", sha("forbidden")
                ),
                "fields differ",
            ),
        ):
            value = preference_manifest()
            mutation(value)
            value = reseal_preference(value)
            with self.subTest(message=message):
                with self.assertRaisesRegex(
                    target_free.TargetFreeTrainingError, message
                ):
                    target_free.validate_preference_set_value(
                        value,
                        source_catalog=self.source,
                        preference_set_sha256=sha("preference-set-file"),
                    )

    def test_failed_axis_and_failure_tag_must_match_exactly(self) -> None:
        value = preference_manifest()
        value["pairs"][0]["rejected_rollout"]["failure_tags"] = [
            "camera_failed"
        ]
        value = reseal_preference(value)
        with self.assertRaisesRegex(
            target_free.TargetFreeTrainingError, "exactly name failed axes"
        ):
            target_free.validate_preference_set_value(
                value,
                source_catalog=self.source,
                preference_set_sha256=sha("preference-set-file"),
            )

    def test_unqualified_chosen_and_all_pass_rejected_are_rejected(self) -> None:
        chosen_bad = preference_manifest()
        chosen_bad["pairs"][0]["chosen_rollout"] = rollout(
            "chosen-bad",
            policy_sha=chosen_bad["behavior_policy_sha256"],
            round_index=0,
            passed=False,
            source=source_row(0),
        )
        chosen_bad["pairs"][0].pop("pair_digest")
        chosen_bad["pairs"][0] = sealed(chosen_bad["pairs"][0], "pair_digest")
        chosen_bad.pop("preference_set_digest")
        chosen_bad = sealed(chosen_bad, "preference_set_digest")
        with self.assertRaisesRegex(
            target_free.TargetFreeTrainingError, "chosen rollout did not pass"
        ):
            target_free.validate_preference_set_value(
                chosen_bad,
                source_catalog=self.source,
                preference_set_sha256=sha("preference-set-file"),
            )

        rejected_bad = preference_manifest()
        rejected_bad["pairs"][0]["rejected_rollout"] = rollout(
            "rejected-bad",
            policy_sha=rejected_bad["behavior_policy_sha256"],
            round_index=0,
            passed=True,
            source=source_row(0),
        )
        rejected_bad["pairs"][0].pop("pair_digest")
        rejected_bad["pairs"][0] = sealed(rejected_bad["pairs"][0], "pair_digest")
        rejected_bad.pop("preference_set_digest")
        rejected_bad = sealed(rejected_bad, "preference_set_digest")
        with self.assertRaisesRegex(
            target_free.TargetFreeTrainingError, "rejected rollout has no named"
        ):
            target_free.validate_preference_set_value(
                rejected_bad,
                source_catalog=self.source,
                preference_set_sha256=sha("preference-set-file"),
            )

    def test_behavior_policy_must_equal_preupdate_student(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        with self.assertRaisesRegex(
            target_free.TargetFreeTrainingError, "behavior policy is not"
        ):
            target_free.run_target_free_preference_update(
                preference_set=self.preference(),
                student_before_sha256=sha("different-policy"),
                trainable_parameters=[parameter],
            )

    def test_empty_preference_set_is_exact_zero_update_and_no_optimizer(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float32))
        before = parameter.detach().clone()

        receipt = target_free.run_target_free_preference_update(
            preference_set=self.preference(include_pair=False),
            student_before_sha256=sha("policy-before-round-0"),
            trainable_parameters=[parameter],
        )
        self.assertEqual(receipt["status"], "ZERO_UPDATE_NO_QUALIFIED_PAIR")
        self.assertFalse(receipt["optimizer_constructed"])
        self.assertFalse(receipt["optimizer_step_executed"])
        self.assertEqual(receipt["function_owned_data_file_open_count"], 0)
        self.assertTrue(torch.equal(parameter.detach(), before))

    def test_nonempty_preference_is_blocked_before_any_update(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        before = parameter.detach().clone()
        with self.assertRaisesRegex(
            target_free.TargetFreeTrainingError,
            target_free.PRODUCTION_RUNTIME_STATUS,
        ):
            target_free.run_target_free_preference_update(
                preference_set=self.preference(),
                student_before_sha256=sha("policy-before-round-0"),
                trainable_parameters=[parameter],
            )
        self.assertTrue(torch.equal(parameter.detach(), before))

    def test_production_update_has_no_callable_seam_and_rejects_forged_object(self) -> None:
        names = set(
            inspect.signature(target_free.run_target_free_preference_update).parameters
        )
        self.assertEqual(
            names,
            {"preference_set", "student_before_sha256", "trainable_parameters"},
        )
        parameter = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        forged = SimpleNamespace(
            behavior_policy_sha256=sha("policy-before-round-0"), pairs=()
        )
        with self.assertRaisesRegex(
            target_free.TargetFreeTrainingError, "closed PreferenceSetV1"
        ):
            target_free.run_target_free_preference_update(
                preference_set=forged,
                student_before_sha256=sha("policy-before-round-0"),
                trainable_parameters=[parameter],
            )

    def test_private_objective_helper_updates_only_synthetic_parameter(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        def provider(_):
            chosen = parameter.expand(target_free.TRAJECTORY_STEPS) + 0.0
            rejected = (-parameter).expand(target_free.TRAJECTORY_STEPS) + 0.0
            return chosen, rejected

        receipt = target_free._run_objective_unit_test_update_v1(
            preference_set=self.preference(),
            trainable_parameters=[parameter],
            optimizer_factory=lambda params: torch.optim.SGD(params, lr=0.01),
            logprob_provider=provider,
        )
        self.assertEqual(
            receipt["status"], "OBJECTIVE_UNIT_TEST_UPDATE_COMPLETE_NOT_PRODUCTION"
        )
        self.assertTrue(receipt["optimizer_step_executed"])
        self.assertNotEqual(receipt["parameter_digest_before"], receipt["parameter_digest_after"])
        self.assertGreater(float(parameter.detach().item()), 0.0)

    def test_swapping_chosen_and_rejected_flips_gradient_sign(self) -> None:
        first = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)
        chosen = (first + 0.0).expand(1, target_free.TRAJECTORY_STEPS)
        rejected = (-first + 0.0).expand(1, target_free.TRAJECTORY_STEPS)
        target_free._pairwise_preference_objective_math_unit_v1(
            chosen, rejected, beta=1.0
        ).backward()
        first_gradient = float(first.grad.item())

        second = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)
        chosen_swapped = (-second + 0.0).expand(1, target_free.TRAJECTORY_STEPS)
        rejected_swapped = (second + 0.0).expand(1, target_free.TRAJECTORY_STEPS)
        target_free._pairwise_preference_objective_math_unit_v1(
            chosen_swapped, rejected_swapped, beta=1.0
        ).backward()
        second_gradient = float(second.grad.item())
        self.assertLess(first_gradient, 0.0)
        self.assertGreater(second_gradient, 0.0)
        self.assertEqual(first_gradient, -second_gradient)

    def test_objective_math_helper_has_no_target_teacher_or_frozen_argument(self) -> None:
        names = set(
            inspect.signature(
                target_free._pairwise_preference_objective_math_unit_v1
            ).parameters
        )
        forbidden_fragments = ("target", "teacher", "frozen", "anchor", "latent")
        self.assertTrue(
            all(
                not any(fragment in name for fragment in forbidden_fragments)
                for name in names
            )
        )

    def test_stable_loader_opens_only_the_declared_source_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source_path = root / "source.mp4"
            poison_path = root / "synthetic-target-index1.bin"
            source_path.write_bytes(b"source-only")
            poison_path.write_bytes(b"must-not-be-opened")
            source_path.chmod(0o444)
            poison_path.chmod(0o000)
            observed = target_free._read_stable_file(
                source_path,
                expected_sha256=hashlib.sha256(b"source-only").hexdigest(),
                label="source-only fixture",
            )
            self.assertEqual(observed, b"source-only")
            self.assertEqual(poison_path.stat().st_size, len(b"must-not-be-opened"))


if __name__ == "__main__":
    unittest.main()
