from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
TOOLS = METHOD_ROOT / "tools"
for entry in (str(METHOD_ROOT), str(TOOLS)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import build_schedule_block_source_edge_pilot_review_html_v2 as builder  # noqa: E402
import schedule_block_causal_policy_v1 as policy  # noqa: E402
import schedule_block_source_edge_ablation_v2 as edge  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _signed(raw: dict, *, field: str) -> dict:
    value = copy.deepcopy(raw)
    value[field] = builder.object_sha256(value)
    return value


class SourceEdgePilotReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.original_bindings = copy.deepcopy(builder.FAMILY_BINDINGS)
        self.dog = self._family("dog")
        self.human = self._family("human")

    def tearDown(self) -> None:
        builder.FAMILY_BINDINGS.clear()
        builder.FAMILY_BINDINGS.update(self.original_bindings)
        self.temporary.cleanup()

    def _trace(self, hook: str) -> tuple[dict, str | None, str]:
        unsigned_native = {
            "steps": [
                {
                    "step_index": index,
                    "transformer_forward_count": 4,
                    "native_formula_exact_parity": True,
                    "original_scheduler_call_count": 1,
                }
                for index in range(builder.NUM_STEPS)
            ],
            "step_count": builder.NUM_STEPS,
            "observed_transformer_forwards": 4 * builder.NUM_STEPS,
        }
        native_digest = builder.object_sha256(unsigned_native)
        trace = {**unsigned_native, "trace_digest": native_digest}
        if hook == "native-unhooked":
            return trace, None, native_digest
        blocks = []
        for index in range(30):
            selected = index in builder.PILOT_BLOCKS
            deleted = 3 if hook == "source-off" and selected else 0
            source_on = 3 if hook == "source-on" and selected else 0
            geometry = None
            if hook == "source-off" and selected:
                geometry = {
                    "schedule_index": builder.PILOT_SCHEDULE,
                    "band_name": builder.PILOT_BAND,
                    "branch_name": "VI_cond",
                    "total_tokens": 130,
                    "source_tokens": 30,
                    "target_tokens": 100,
                    "source_query_rows_from_native_full_attention": True,
                    "target_query_rows_from_target_KV_only_attention": True,
                    "post_rope_token_order_unchanged": True,
                }
            blocks.append(
                {
                    "block_index": index,
                    "official_delegate_calls": 160 - deleted,
                    "active_edge_deletion_calls": deleted,
                    "active_source_on_calls": source_on,
                    "branch_calls": {
                        name: builder.NUM_STEPS for name in builder.NATIVE_BRANCH_ORDER
                    },
                    "schedule_calls": {
                        str(step): 4 for step in range(builder.NUM_STEPS)
                    },
                    "last_active_geometry": geometry,
                }
            )
        edge_receipt = _signed(
            {
                "contract": edge.intervention_contract(),
                "edge_mode": hook,
                "registered_schedule_index": builder.PILOT_SCHEDULE,
                "band_name": builder.PILOT_BAND,
                "selected_blocks": list(builder.PILOT_BLOCKS),
                "source_bearing_branches": list(builder.SOURCE_BEARING_BRANCHES),
                "expected_active_calls_per_selected_block": 3,
                "per_block": blocks,
                "native_trace_digest": native_digest,
            },
            field="digest",
        )
        combined = builder.object_sha256({"native": native_digest, "edge": edge_receipt})
        trace["source_edge"] = edge_receipt
        trace["source_edge_trace_digest"] = combined
        return trace, edge_receipt["digest"], combined

    def _family(self, family: str) -> Path:
        root = self.root / f"input-{family}"
        root.mkdir()
        correct_source = root / "source-correct.mp4"
        wrong_source = root / "source-wrong-owner.mp4"
        correct_source.write_bytes(f"{family}-correct-source-81f".encode())
        wrong_source.write_bytes(f"{family}-wrong-source-81f".encode())
        binding = builder.FAMILY_BINDINGS[family]
        binding["correct_sha256"] = _sha(correct_source)
        binding["wrong_sha256"] = _sha(wrong_source)

        descriptions = {
            "action": f"The {family} performs the complete requested action.",
            "noop": f"The {family} remains still and does not perform the action.",
            "reverse": f"The {family} performs the requested action in reverse order.",
            "incomplete": f"The {family} starts but does not finish the requested action.",
            "camera_only": "Only the camera moves while the actor remains still.",
            "appearance_only": "Only the actor appearance changes while motion remains still.",
        }
        correct = {
            "iid": binding["correct_iid"],
            "analysis_split": "fit",
            "action_family_id": f"{family}-action-family",
            "actor_group_id": f"{family}-actor-a",
            "scene_group_id": f"{family}-scene-a",
            "action_group_id": f"{family}-action-a",
            "execution_group": "pilot",
            "geometry_source_video": str(correct_source),
            "seed": binding["seed"],
            "scene_caption": f"A registered {family} stands in the source scene.",
            "branch_descriptions": descriptions,
            "camera_caption": "The camera remains fixed and the background stays unchanged.",
        }
        wrong = {
            **correct,
            "iid": binding["wrong_iid"],
            "analysis_split": "confirmation",
            "actor_group_id": f"{family}-actor-b",
            "scene_group_id": f"{family}-scene-b",
            "geometry_source_video": str(wrong_source),
        }
        captions = builder._branch_captions(correct)
        prompts = {
            branch: {
                "caption": caption,
                "caption_utf8_sha256": _hash_text(caption),
                "native_prompt_utf8_sha256": _hash_text(f"native-wrapper::{caption}"),
            }
            for branch, caption in captions.items()
        }
        plan = builder._expected_plan()
        outputs: dict[str, dict] = {}
        generated_identities: dict[str, dict] = {}
        traces: dict[str, dict] = {}
        candidates: list[dict] = []
        gaussian = _hash_text(f"{family}-official-gaussian")
        parity_raw = _hash_text(f"{family}-native-forward-latent")
        for ordinal, row in enumerate(plan):
            key = row["key"]
            path = root / f"{key}.mp4"
            path.write_bytes(f"{family}-{key}-decoded-exact81".encode())
            outputs[key] = {
                "path": str(path),
                "sha256": _sha(path),
                "frame_count": 81,
                "fps": 25,
                "height": 480,
                "width": 832,
                "normalized_clean_latent": {"path": f"unused-{key}.safetensors"},
            }
            raw_sha = (
                parity_raw
                if key in (
                    "native-correct-forward",
                    "parity-source-on-s16-early-forward",
                )
                else _hash_text(f"{family}-{key}-latent")
            )
            identity = {
                "all_rank_exact": True,
                "identity": {
                    "shape": [1, 16, 21, 60, 104],
                    "dtype": "torch.float32",
                    "numel": 2096640,
                    "byte_count": 8386560,
                    "content_sha256": _hash_text(f"content::{raw_sha}"),
                    "raw_storage_sha256": raw_sha,
                    "finite": True,
                    "label": f"generated_{key}",
                },
            }
            generated_identities[key] = identity
            trace, edge_digest, trace_identity = self._trace(row["hook"])
            traces[key] = trace
            gate = _signed(
                {
                    "passed": True,
                    "hook": row["hook"],
                    "step_count": 40,
                    "transformer_forward_count": 160,
                    "edge_receipt_digest": edge_digest,
                },
                field="digest",
            )
            candidate = {
                **row,
                "seed": binding["seed"],
                "prompt_sha256": prompts[row["text_branch"]][
                    "native_prompt_utf8_sha256"
                ],
                "initial_gaussian_raw_sha256": gaussian,
                "generated_identity": identity,
                "trace_gate": gate,
                "trace_all_rank": {"all_rank_exact": True, "value": trace_identity},
                "score": None,
                "rank": None,
                "selected": False,
            }
            candidates.append(_signed(candidate, field="candidate_digest"))

        prompt_guard = {
            "schema_version": "bernini-model-mutation-guard-v1",
            "state_tensor_count": 200,
            "process_local_storage_and_version_sha256": "1" * 64,
            "no_parameter_or_buffer_bytes_copied_to_host": True,
        }
        sampling_guard = {
            "schema_version": "bernini-model-mutation-guard-v1",
            "state_tensor_count": 100,
            "process_local_storage_and_version_sha256": "2" * 64,
            "no_parameter_or_buffer_bytes_copied_to_host": True,
        }
        receipt = _signed(
            {
                "schema_version": builder.INPUT_RECEIPT_SCHEMA,
                "method": builder.METHOD,
                "stage": builder.STAGE,
                "registered_schedule_block_policy": policy.default_policy().receipt(),
                "intervention_contract": edge.intervention_contract(),
                "full_grid_contract": edge.decoded_grid_contract(),
                "shard": {
                    "family": family,
                    "schedule_indices": [16],
                    "block_bands": ["early"],
                    "full_registered_grid": False,
                    "candidate_count": len(plan),
                    "plan": plan,
                },
                "authority": {
                    "path": str(root / "authoring.json"),
                    "sha256": builder.AUTHORING_SHA256,
                    "schema_version": builder.AUTHORING_SCHEMA,
                    "bank_id": "pair5-t2v-first8-v1",
                    "correct_row": correct,
                    "wrong_owner_row": wrong,
                },
                "runtime_source": {
                    "revision": "d" * 40,
                    "closure_sha256": "e" * 64,
                    "launcher_sha256": "f" * 64,
                },
                "checkpoint": {
                    "path": "/read-only/checkpoint",
                    "tree_sha256": "a" * 64,
                    "content_identity": {"manifest_sha256": "b" * 64},
                    "opened_read_only": True,
                },
                "source": {
                    "correct_path": str(correct_source),
                    "correct_sha256": binding["correct_sha256"],
                    "correct_snapshot": str(correct_source),
                    "wrong_owner_path": str(wrong_source),
                    "wrong_owner_sha256": binding["wrong_sha256"],
                    "wrong_owner_snapshot": str(wrong_source),
                    "wrong_owner_same_action_family": True,
                    "wrong_owner_identity_only_control": False,
                    "scene_and_geometry_confound_acknowledged": True,
                },
                "prompts": prompts,
                "sampling": {
                    "seed": binding["seed"],
                    "exact40": True,
                    "exact81": True,
                    "scheduler": "native-UniPC-flow-shift-5",
                    "same_initial_gaussian_all_candidates": True,
                    "shared_initial_gaussian_raw_sha256": gaussian,
                    "source_on_native_parity_raw_sha256": parity_raw,
                    "source_on_native_parity_bit_exact": True,
                },
                "candidates": candidates,
                "traces": traces,
                "generated_identities": generated_identities,
                "outputs": outputs,
                "frozen_model": {
                    "rank_zero_full_byte_certificate": {"passed": True},
                    "prompt_guard": prompt_guard,
                    "sampling_guard_before": sampling_guard,
                    "sampling_guard_after": copy.deepcopy(sampling_guard),
                    "unchanged": True,
                },
                "interpretation": {
                    "decoded_complete_video_required": True,
                    "manual_joint_action_and_preservation_review_pending": True,
                    "hidden_or_feature_metric_authorizes_route": False,
                    "score_computed": False,
                    "reward_computed": False,
                    "ranking_performed": False,
                    "selection_performed": False,
                    "training_performed": False,
                    "optimizer_present": False,
                    "backward_performed": False,
                    "parameter_update": False,
                    "stage_B_authorized_by_runtime_alone": False,
                },
            },
            field="receipt_digest",
        )
        (root / "receipt.json").write_text(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")),
            encoding="ascii",
        )
        return root

    def _rewrite_receipt(self, root: Path, mutator: object, *, resign: bool = True) -> None:
        path = root / "receipt.json"
        receipt = json.loads(path.read_text(encoding="ascii"))
        receipt.pop("receipt_digest")
        mutator(receipt)  # type: ignore[operator]
        if resign:
            receipt["receipt_digest"] = builder.object_sha256(receipt)
        path.write_text(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")),
            encoding="ascii",
        )

    def test_builds_complete_self_contained_pilot_review(self) -> None:
        output = self.root / "review"
        index = builder.build(
            dog_output=self.dog,
            human_output=self.human,
            output_dir=output,
        )
        self.assertEqual(index, output / "index.html")
        page = index.read_text(encoding="utf-8")
        self.assertEqual(page.count("<video controls"), 32)
        for expected in (
            "Correct source · 编辑输入",
            "Wrong-owner source · 完整视频",
            "Forward 的完整 editing instruction",
            "Native typed controls（correct owner）",
            "Source-on forward parity",
            "Source-off typed controls（s16 × early）",
            "forward · native wrong-owner context",
            "真正关闭的 edge：",
            "target noisy-suffix queries",
            "source-query rows 仍采用原生全 attention 输出",
            "pre-decode FP32 latent 与 native forward bit-exact",
            "seed 2026080825 · exact40 · exact81",
            'src="cells/dog/source-correct.mp4"',
            'src="cells/human/off-s16-early-appearance_only.mp4"',
            'href="cells/dog/receipt.json"',
        ):
            self.assertIn(expected, page)
        lowered = page.lower()
        self.assertNotIn("<dt>value</dt>", lowered)
        self.assertNotIn("scalar", lowered)
        self.assertNotIn("verdict", lowered)
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["authority"], builder.AUTHORITY)
        self.assertEqual(len(manifest["cells"]), 2)
        self.assertEqual(manifest["cells"][0]["pilot_cell"]["block_indices"], list(range(8)))
        self.assertTrue(
            manifest["cells"][0]["source_on_predecode_bit_exact_with_native_forward"]
        )
        self.assertTrue((output / "cells/dog/source-wrong-owner.mp4").is_file())
        self.assertTrue(
            (output / "cells/human/parity-source-on-s16-early-forward.mp4").is_file()
        )

    def test_corrupt_decoded_mp4_fails_without_partial_publish(self) -> None:
        (self.dog / "off-s16-early-reverse.mp4").write_bytes(b"corrupt")
        output = self.root / "failed-corrupt"
        with self.assertRaisesRegex(builder.SourceEdgePilotReviewError, "MP4 differs"):
            builder.build(
                dog_output=self.dog,
                human_output=self.human,
                output_dir=output,
            )
        self.assertFalse(output.exists())

    def test_resigned_policy_mutation_still_fails_registered_pin(self) -> None:
        def mutate(receipt: dict) -> None:
            registered = receipt["registered_schedule_block_policy"]
            registered.pop("receipt_digest")
            registered["optimizer_authorized"] = True
            registered["receipt_digest"] = builder.object_sha256(registered)

        self._rewrite_receipt(self.human, mutate)
        output = self.root / "failed-policy"
        with self.assertRaisesRegex(builder.SourceEdgePilotReviewError, "pinned registered"):
            builder.build(
                dog_output=self.dog,
                human_output=self.human,
                output_dir=output,
            )
        self.assertFalse(output.exists())

    def test_reward_or_exact81_authority_is_rejected(self) -> None:
        self._rewrite_receipt(
            self.human,
            lambda receipt: receipt["interpretation"].__setitem__("reward_computed", True),
        )
        with self.assertRaisesRegex(builder.SourceEdgePilotReviewError, "authority"):
            builder.build(
                dog_output=self.dog,
                human_output=self.human,
                output_dir=self.root / "failed-reward",
            )

    def test_source_on_parity_identity_mismatch_is_rejected(self) -> None:
        self._rewrite_receipt(
            self.dog,
            lambda receipt: receipt["sampling"].__setitem__(
                "source_on_native_parity_raw_sha256", "9" * 64
            ),
        )
        with self.assertRaisesRegex(builder.SourceEdgePilotReviewError, "not bit-exact"):
            builder.build(
                dog_output=self.dog,
                human_output=self.human,
                output_dir=self.root / "failed-parity",
            )

    def test_nonpilot_shard_and_invalid_receipt_digest_are_rejected(self) -> None:
        original = (self.dog / "receipt.json").read_bytes()
        self._rewrite_receipt(
            self.dog,
            lambda receipt: receipt["shard"].__setitem__("schedule_indices", [29]),
        )
        with self.assertRaisesRegex(builder.SourceEdgePilotReviewError, "not exactly"):
            builder.build(
                dog_output=self.dog,
                human_output=self.human,
                output_dir=self.root / "failed-cell",
            )
        (self.dog / "receipt.json").write_bytes(original)
        self._rewrite_receipt(self.dog, lambda receipt: None, resign=False)
        with self.assertRaisesRegex(builder.SourceEdgePilotReviewError, "digest"):
            builder.build(
                dog_output=self.dog,
                human_output=self.human,
                output_dir=self.root / "failed-digest",
            )


if __name__ == "__main__":
    unittest.main()
