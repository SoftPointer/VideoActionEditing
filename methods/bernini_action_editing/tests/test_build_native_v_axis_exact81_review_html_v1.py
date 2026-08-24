from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import build_native_v_axis_exact81_review_html_v1 as builder  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _signed(raw: dict) -> dict:
    receipt = copy.deepcopy(raw)
    receipt["receipt_digest"] = hashlib.sha256(
        builder.canonical_json_bytes(receipt)
    ).hexdigest()
    return receipt


def _arm_contract(arm: str) -> dict:
    return {
        "arm": arm,
        "full_video_condition_role": "wrong" if arm == "wrong-V" else "correct",
        "omega_video": 0.0 if arm == "V-off" else 1.25,
        "omega_image": 4.5,
        "omega_text": 4.0,
        "correct_image_references": True,
        "same_instruction": True,
        "same_scheduler": True,
        "same_target_geometry": True,
        "intervention": {
            "V-on": "native_no_numerical_intervention",
            "V-off": "zero_standalone_vV_minus_v0_coefficient_only",
            "wrong-V": "replace_full_video_condition_only",
        }[arm],
        "v_vi_u_minus_v_v_term_retained": True,
    }


class NativeVAxisReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.dog = self._cell("dog", confound=False)
        self.human = self._cell("human", confound=True)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _cell(self, cell_id: str, *, confound: bool) -> Path:
        root = self.root / f"input-{cell_id}"
        root.mkdir()
        correct = root / "source-correct.mp4"
        wrong = root / "source-wrong-V.mp4"
        correct.write_bytes(f"{cell_id}-correct-source".encode())
        wrong.write_bytes(f"{cell_id}-wrong-source".encode())
        seeds = [2026080801, 2026080901] if cell_id == "dog" else [2026080802, 2026080902]
        instruction = (
            "The grey dog bends its hind legs, lowers its hips, and holds a stable sit."
            if cell_id == "dog"
            else "The woman rises smoothly from kneeling and holds a stable upright pose."
        )
        instruction_sha = hashlib.sha256(instruction.encode("utf-8")).hexdigest()
        outputs: dict[str, dict] = {}
        candidates: list[dict] = []
        traces: dict[str, dict] = {}
        for seed_index, seed in enumerate(seeds):
            gaussian = ("a" if seed_index == 0 else "b") * 64
            for arm_index, arm in enumerate(builder.ARM_ORDER):
                key = f"seed-{seed}__{arm}"
                video = root / f"{key}.mp4"
                video.write_bytes(f"{cell_id}-{seed}-{arm}".encode())
                outputs[key] = {
                    "path": str(video),
                    "sha256": _sha(video),
                    "frame_count": 81,
                    "fps": 25,
                }
                trace_digest = str(arm_index + 1) * 64
                candidate = {
                    "candidate_key": key,
                    "seed": seed,
                    "arm": arm,
                    "arm_contract": _arm_contract(arm),
                    "trace_digest": trace_digest,
                    "exact40_trace_gate": {
                        "passed": True,
                        "arm": arm,
                        "step_count": 40,
                        "four_native_branch_calls_per_step": True,
                        "one_original_unipc_call_per_step": True,
                    },
                    "official_initial_gaussian_raw_value_sha256": gaussian,
                    "score": None,
                    "rank": None,
                    "selected": False,
                }
                candidate["candidate_receipt_digest"] = hashlib.sha256(
                    builder.canonical_json_bytes(candidate)
                ).hexdigest()
                candidates.append(candidate)
                traces[key] = {"step_count": 40, "trace_digest": trace_digest}
        receipt = _signed(
            {
                "schema_version": builder.INPUT_RECEIPT_SCHEMA,
                "method": builder.METHOD,
                "stage": builder.STAGE,
                "cell_spec": {
                    "file_sha256": "c" * 64,
                    "contract": {
                        "method": builder.METHOD,
                        "frame_count": 81,
                        "latent_phases": 21,
                        "fps": 25,
                        "num_inference_steps": 40,
                        "guidance_mode": "rv2v",
                        "native_velocity_formula": builder.NATIVE_FORMULA,
                        "v_off_velocity_formula": builder.V_OFF_FORMULA,
                        "arm_order": list(builder.ARM_ORDER),
                        "wrong_v_replaces_full_video_condition_only": True,
                        "wrong_v_keeps_correct_image_references_and_text": True,
                        "training": False,
                        "optimizer": False,
                        "feature_scorer": False,
                        "reward": False,
                        "ranking": False,
                        "selection": False,
                    },
                    "cell": {
                        "cell_id": cell_id,
                        "actor_kind": cell_id,
                        "source_iid": f"{cell_id}-correct-iid",
                        "wrong_source_iid": f"{cell_id}-wrong-iid",
                        "wrong_source_geometry_confound": confound,
                        "wrong_source_pure_identity_control": False,
                        "action_caption": instruction,
                        "action_caption_utf8_sha256": instruction_sha,
                        "seeds": seeds,
                        "selected_before_generation": True,
                    },
                },
                "runtime_source": {
                    "revision": "d" * 40,
                },
                "checkpoint": {
                    "tree_sha256": "e" * 64,
                    "opened_read_only": True,
                },
                "correct_source": {
                    "sha256": _sha(correct),
                    "snapshot_mp4": str(correct),
                },
                "wrong_V_source": {
                    "sha256": _sha(wrong),
                    "snapshot_mp4": str(wrong),
                    "used_only_as_full_video_condition_in_wrong_V": True,
                    "used_as_image_reference": False,
                    "pure_identity_control": False,
                    "geometry_confound_present": confound,
                },
                "prompt": {
                    "action_caption": instruction,
                    "action_caption_utf8_sha256": instruction_sha,
                    "same_across_all_arms_and_seeds": True,
                },
                "sampling": {
                    "seeds": seeds,
                    "exact40": True,
                    "exact81": True,
                    "frame_count": 81,
                    "latent_phases": 21,
                    "fps": 25,
                    "num_inference_steps": 40,
                    "arm_order": list(builder.ARM_ORDER),
                    "same_official_gaussian_within_seed": True,
                    "same_x_t_t_target_geometry_within_seed": True,
                    "hook_contract": {
                        "native_formula": (
                            "v0+1.25*(vV-v0)+4.5*(vVIu-vV)+4.0*(vVIc-vVIu)"
                        ),
                        "v_off_formula": (
                            "v0+0.0*(vV-v0)+4.5*(vVIu-vV)+4.0*(vVIc-vVIu)"
                        ),
                        "transformer_forwards_per_step": 4,
                        "original_unipc_calls_per_step": 1,
                        "training": False,
                        "optimizer": False,
                        "feature_scorer": False,
                        "selection": False,
                    },
                },
                "candidates": candidates,
                "traces": traces,
                "outputs": outputs,
                "interpretation": {
                    "training_performed": False,
                    "trainer_instantiated": False,
                    "optimizer": None,
                    "backward": False,
                    "model_weights_written": False,
                    "adapter_loaded": False,
                    "target_video": False,
                    "feature_scorer_consumed": False,
                    "reward_computed": False,
                    "score_computed": False,
                    "ranking_performed": False,
                    "best_arm_selected": False,
                    "visual_selection_performed": False,
                    "action_success_evaluated": False,
                    "preservation_success_evaluated": False,
                    "scientific_claim_authorized_before_blind_review": False,
                    "wrong_V_changes_only_full_video_condition": True,
                    "V_off_zeros_only_standalone_vV_minus_v0_coefficient": True,
                    "V_off_retains_vVIu_minus_vV_term": True,
                },
            }
        )
        (root / "receipt.json").write_text(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")),
            encoding="ascii",
        )
        return root

    def _rewrite_receipt(self, root: Path, mutator: object) -> None:
        path = root / "receipt.json"
        receipt = json.loads(path.read_text(encoding="ascii"))
        receipt.pop("receipt_digest")
        mutator(receipt)  # type: ignore[operator]
        receipt = _signed(receipt)
        path.write_text(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")),
            encoding="ascii",
        )

    def test_builds_self_contained_cell_seed_five_column_review(self) -> None:
        output = self.root / "review"
        index = builder.build(
            dog_output=self.dog,
            human_output=self.human,
            output_dir=output,
        )
        self.assertEqual(index, output / "index.html")
        page = index.read_text(encoding="utf-8")
        self.assertEqual(page.count("<video controls"), 20)
        for expected in (
            "完整 editing instruction / action caption",
            "Correct source · 完整编辑输入",
            "Wrong-V source · 完整视频",
            "V-on / native",
            "V-off",
            "Wrong-V",
            "唯一变量：",
            "exact40 · 40 UniPC steps · 81 frames",
            "base checkpoint tree SHA-256",
            "source MP4 SHA-256",
            builder.NATIVE_FORMULA,
            builder.V_OFF_FORMULA,
            "geometry confound: PRESENT",
            "不是 pure identity control",
            'src="cells/dog/source-correct.mp4"',
            'src="cells/human/seed-2026080802__wrong-V.mp4"',
            'href="cells/dog/receipt.json"',
        ):
            self.assertIn(expected, page)
        self.assertNotIn("<dt>value</dt>", page.lower())
        self.assertNotIn(">value<", page.lower())
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["authority"], builder.AUTHORITY)
        self.assertEqual(len(manifest["cells"]), 2)
        self.assertTrue((output / "cells/dog/source-correct.mp4").is_file())
        self.assertTrue((output / "cells/human/seed-2026080802__wrong-V.mp4").is_file())
        self.assertEqual(
            _sha(output / "cells/dog/receipt.json"),
            _sha(self.dog / "receipt.json"),
        )

    def test_corrupt_mp4_fails_without_publishing_partial_directory(self) -> None:
        (self.dog / "seed-2026080801__V-off.mp4").write_bytes(b"corrupt")
        output = self.root / "failed-review"
        with self.assertRaisesRegex(builder.NativeVAxisReviewError, "MP4 differs"):
            builder.build(
                dog_output=self.dog,
                human_output=self.human,
                output_dir=output,
            )
        self.assertFalse(output.exists())

    def test_reward_or_scoring_authority_is_rejected(self) -> None:
        self._rewrite_receipt(
            self.human,
            lambda receipt: receipt["interpretation"].__setitem__(
                "reward_computed", True
            ),
        )
        output = self.root / "failed-authority"
        with self.assertRaisesRegex(builder.NativeVAxisReviewError, "authority"):
            builder.build(
                dog_output=self.dog,
                human_output=self.human,
                output_dir=output,
            )
        self.assertFalse(output.exists())

    def test_wrong_source_confound_flag_must_match_receipt_and_cell(self) -> None:
        self._rewrite_receipt(
            self.human,
            lambda receipt: receipt["wrong_V_source"].__setitem__(
                "geometry_confound_present", False
            ),
        )
        with self.assertRaisesRegex(builder.NativeVAxisReviewError, "confound"):
            builder.build(
                dog_output=self.dog,
                human_output=self.human,
                output_dir=self.root / "failed-confound",
            )


if __name__ == "__main__":
    unittest.main()
