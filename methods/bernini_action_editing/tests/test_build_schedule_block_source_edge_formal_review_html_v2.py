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

import build_schedule_block_source_edge_formal_review_html_v2 as builder  # noqa: E402
import build_schedule_block_source_edge_pilot_review_html_v2 as common  # noqa: E402
import schedule_block_causal_policy_v1 as policy  # noqa: E402
import schedule_block_source_edge_ablation_v2 as edge  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text_sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _signed(raw: dict, field: str) -> dict:
    value = copy.deepcopy(raw)
    value[field] = common.object_sha256(value)
    return value


class FormalSourceEdgeReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.original_bindings = copy.deepcopy(common.FAMILY_BINDINGS)
        self.dog = self._family("dog")
        self.human = self._family("human")

    def tearDown(self) -> None:
        common.FAMILY_BINDINGS.clear()
        common.FAMILY_BINDINGS.update(self.original_bindings)
        self.temporary.cleanup()

    def _trace(self, plan: dict) -> tuple[dict, str | None, str]:
        native = {
            "steps": [
                {
                    "step_index": index,
                    "transformer_forward_count": 4,
                    "native_formula_exact_parity": True,
                    "original_scheduler_call_count": 1,
                }
                for index in range(40)
            ],
            "step_count": 40,
            "observed_transformer_forwards": 160,
        }
        native_digest = common.object_sha256(native)
        trace = {**native, "trace_digest": native_digest}
        if plan["hook"] == "native-unhooked":
            return trace, None, native_digest
        schedule, band, selected_blocks = builder._coord_for_plan(plan)
        blocks = []
        for index in range(30):
            selected = index in selected_blocks
            deleted = 3 if plan["hook"] == "source-off" and selected else 0
            delegated_on = 3 if plan["hook"] == "source-on" and selected else 0
            geometry = None
            if deleted:
                geometry = {
                    "schedule_index": schedule,
                    "band_name": band,
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
                    "active_source_on_calls": delegated_on,
                    "branch_calls": {
                        name: 40 for name in common.NATIVE_BRANCH_ORDER
                    },
                    "schedule_calls": {str(step): 4 for step in range(40)},
                    "last_active_geometry": geometry,
                }
            )
        edge_receipt = _signed(
            {
                "contract": edge.intervention_contract(),
                "edge_mode": plan["hook"],
                "registered_schedule_index": schedule,
                "band_name": band,
                "selected_blocks": list(selected_blocks),
                "source_bearing_branches": list(common.SOURCE_BEARING_BRANCHES),
                "expected_active_calls_per_selected_block": 3,
                "per_block": blocks,
                "native_trace_digest": native_digest,
            },
            "digest",
        )
        combined = common.object_sha256({"native": native_digest, "edge": edge_receipt})
        trace["source_edge"] = edge_receipt
        trace["source_edge_trace_digest"] = combined
        return trace, edge_receipt["digest"], combined

    def _family(self, family: str) -> Path:
        root = self.root / f"formal-{family}"
        root.mkdir()
        correct_source = root / "source-correct.mp4"
        wrong_source = root / "source-wrong-owner.mp4"
        correct_source.write_bytes(f"{family}-correct-source".encode())
        wrong_source.write_bytes(f"{family}-wrong-source".encode())
        binding = common.FAMILY_BINDINGS[family]
        binding["correct_sha256"] = _sha(correct_source)
        binding["wrong_sha256"] = _sha(wrong_source)
        descriptions = {
            "action": f"The {family} completes the requested forward action.",
            "noop": f"The {family} stays still and performs no action.",
            "reverse": f"The {family} executes the action in reverse temporal order.",
            "incomplete": f"The {family} starts but does not complete the action.",
            "camera_only": "Only the camera moves while the actor remains still.",
            "appearance_only": "Only appearance changes while action and camera stay still.",
        }
        correct = {
            "iid": binding["correct_iid"],
            "analysis_split": "fit",
            "action_family_id": f"{family}-family",
            "actor_group_id": f"{family}-actor-a",
            "scene_group_id": f"{family}-scene-a",
            "action_group_id": f"{family}-action-a",
            "execution_group": "formal",
            "geometry_source_video": str(correct_source),
            "seed": binding["seed"],
            "scene_caption": f"A registered {family} is visible in the source scene.",
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
        captions = common._branch_captions(correct)
        prompts = {
            branch: {
                "caption": caption,
                "caption_utf8_sha256": _text_sha(caption),
                "native_prompt_utf8_sha256": _text_sha(f"wrapper::{caption}"),
            }
            for branch, caption in captions.items()
        }
        plan = builder._expected_plan()
        gaussian = _text_sha(f"{family}-gaussian")
        parity_raw = _text_sha(f"{family}-parity-latent")
        candidates = []
        traces = {}
        identities = {}
        outputs = {}
        for row in plan:
            key = row["key"]
            video = root / f"{key}.mp4"
            video.write_bytes(f"{family}-{key}-exact81".encode())
            outputs[key] = {
                "path": str(video),
                "sha256": _sha(video),
                "frame_count": 81,
                "fps": 25,
                "height": 480,
                "width": 832,
            }
            raw_sha = (
                parity_raw
                if key
                in ("native-correct-forward", "parity-source-on-s16-early-forward")
                else _text_sha(f"{family}-{key}-latent")
            )
            identity = {
                "all_rank_exact": True,
                "identity": {
                    "shape": [1, 16, 21, 60, 104],
                    "dtype": "torch.float32",
                    "numel": 2096640,
                    "byte_count": 8386560,
                    "content_sha256": _text_sha(f"content::{raw_sha}"),
                    "raw_storage_sha256": raw_sha,
                    "finite": True,
                    "label": f"generated_{key}",
                },
            }
            identities[key] = identity
            trace, edge_digest, trace_identity = self._trace(row)
            traces[key] = trace
            gate = _signed(
                {
                    "passed": True,
                    "hook": row["hook"],
                    "step_count": 40,
                    "transformer_forward_count": 160,
                    "edge_receipt_digest": edge_digest,
                },
                "digest",
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
            candidates.append(_signed(candidate, "candidate_digest"))
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
                "schema_version": common.INPUT_RECEIPT_SCHEMA,
                "method": common.METHOD,
                "stage": common.STAGE,
                "registered_schedule_block_policy": policy.default_policy().receipt(),
                "intervention_contract": edge.intervention_contract(),
                "full_grid_contract": edge.decoded_grid_contract(),
                "shard": {
                    "family": family,
                    "schedule_indices": list(builder.FORMAL_SCHEDULES),
                    "block_bands": list(builder.FORMAL_BANDS),
                    "full_registered_grid": False,
                    "candidate_count": 56,
                    "plan": plan,
                },
                "authority": {
                    "path": str(root / "authoring.json"),
                    "sha256": common.AUTHORING_SHA256,
                    "schema_version": common.AUTHORING_SCHEMA,
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
                "generated_identities": identities,
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
            "receipt_digest",
        )
        (root / "receipt.json").write_text(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")),
            encoding="ascii",
        )
        return root

    def _mutate(self, root: Path, mutator: object) -> None:
        path = root / "receipt.json"
        receipt = json.loads(path.read_text(encoding="ascii"))
        receipt.pop("receipt_digest")
        mutator(receipt)  # type: ignore[operator]
        receipt["receipt_digest"] = common.object_sha256(receipt)
        path.write_text(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")),
            encoding="ascii",
        )

    def test_builds_complete_formal_grid_and_first_screen_definitions(self) -> None:
        output = self.root / "formal-review"
        index = builder.build(
            dog_output=self.dog,
            human_output=self.human,
            output_dir=output,
        )
        self.assertEqual(index, output / "index.html")
        page = index.read_text(encoding="utf-8")
        self.assertEqual(page.count("<video controls"), 116)
        first_screen = page.split("</section>", 1)[0]
        for expected in (
            "Source 是什么",
            "Full instruction 是什么",
            "Schedule index / sigma",
            "Block band 是什么",
            "Native",
            "Source-on",
            "Source-off",
            "Wrong-owner",
        ):
            self.assertIn(expected, first_screen)
        for expected in (
            "s16",
            "0.8825258612632751",
            "s29",
            "0.6555827856063843",
            "s35",
            "0.41860657930374146",
            "s38",
            "0.21162153780460358",
            "early_middle · blocks 8–15",
            "late_middle · blocks 16–22",
            "完整 instruction",
            'src="cells/dog/off-s29-early_middle-forward.mp4"',
            'src="cells/human/off-s38-late_middle-appearance_only.mp4"',
        ):
            self.assertIn(expected, page)
        self.assertNotIn("scalar", page.lower())
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["authority"], builder.AUTHORITY)
        self.assertEqual(manifest["experiment"]["candidate_count_per_family"], 56)
        self.assertFalse(manifest["experiment"]["training"])
        self.assertEqual(len(manifest["cells"][0]["outputs"]), 56)

    def test_corrupt_mp4_fails_without_partial_publish(self) -> None:
        (self.human / "off-s35-late_middle-reverse.mp4").write_bytes(b"corrupt")
        output = self.root / "failed-corrupt"
        with self.assertRaisesRegex(builder.FormalSourceEdgeReviewError, "MP4 differs"):
            builder.build(
                dog_output=self.dog,
                human_output=self.human,
                output_dir=output,
            )
        self.assertFalse(output.exists())

    def test_wrong_grid_or_candidate_count_is_rejected(self) -> None:
        self._mutate(
            self.dog,
            lambda receipt: receipt["shard"].__setitem__(
                "block_bands", ["early", "late_middle"]
            ),
        )
        with self.assertRaisesRegex(builder.FormalSourceEdgeReviewError, "strict formal"):
            builder.build(
                dog_output=self.dog,
                human_output=self.human,
                output_dir=self.root / "failed-grid",
            )

    def test_training_or_reward_authority_is_rejected(self) -> None:
        self._mutate(
            self.human,
            lambda receipt: receipt["interpretation"].__setitem__(
                "training_performed", True
            ),
        )
        with self.assertRaisesRegex(builder.FormalSourceEdgeReviewError, "authority"):
            builder.build(
                dog_output=self.dog,
                human_output=self.human,
                output_dir=self.root / "failed-authority",
            )

    def test_source_off_coordinate_mismatch_is_rejected_even_when_resigned(self) -> None:
        def mutate(receipt: dict) -> None:
            key = "off-s29-early_middle-forward"
            trace = receipt["traces"][key]
            edge_receipt = trace["source_edge"]
            edge_receipt.pop("digest")
            edge_receipt["registered_schedule_index"] = 35
            edge_receipt["digest"] = common.object_sha256(edge_receipt)
            trace["source_edge_trace_digest"] = common.object_sha256(
                {"native": trace["trace_digest"], "edge": edge_receipt}
            )
            for candidate in receipt["candidates"]:
                if candidate["key"] != key:
                    continue
                candidate.pop("candidate_digest")
                gate = candidate["trace_gate"]
                gate.pop("digest")
                gate["edge_receipt_digest"] = edge_receipt["digest"]
                gate["digest"] = common.object_sha256(gate)
                candidate["trace_all_rank"]["value"] = trace[
                    "source_edge_trace_digest"
                ]
                candidate["candidate_digest"] = common.object_sha256(candidate)
                break

        self._mutate(self.dog, mutate)
        with self.assertRaisesRegex(builder.FormalSourceEdgeReviewError, "coordinate"):
            builder.build(
                dog_output=self.dog,
                human_output=self.human,
                output_dir=self.root / "failed-coordinate",
            )


if __name__ == "__main__":
    unittest.main()
