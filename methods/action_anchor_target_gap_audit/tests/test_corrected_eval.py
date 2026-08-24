from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
import tempfile
import unittest

from methods.action_anchor_target_gap_audit import corrected_eval
from methods.action_anchor_target_gap_audit import build_corrected_review_html


ROOT = Path(__file__).resolve().parents[1]


class CorrectedEvaluationTest(unittest.TestCase):
    def test_manual_contracts_are_complete_and_freeze_reported_corrections(self) -> None:
        contracts = corrected_eval.load_contracts(ROOT / "manual_action_contracts_v2.json")
        self.assertEqual(len(contracts), 16)
        self.assertEqual(Counter(row["manual_winner"] for row in contracts.values()), Counter({"anchor": 14, "tie": 2}))
        self.assertIn("full_spin_cycle", {row["id"] for row in contracts["81533c9e56ec"]["required_predicates"]})
        self.assertIn("source_bow_retained", {row["id"] for row in contracts["d908552fb691"]["forbidden_behaviors"]})
        self.assertIn("picks_up_smartphone_afterward", {row["id"] for row in contracts["5e83a9279951"]["required_predicates"]})
        self.assertIn("phone_remains_in_hand", {row["id"] for row in contracts["40712e1341dc"]["forbidden_behaviors"]})
        dog_contract = contracts["3fadbe0c2961"]
        self.assertEqual([row["id"] for row in dog_contract["required_predicates"]], ["head_turn_right"])
        self.assertNotIn("frame right", dog_contract["target_action"].lower())

    def test_prompt_reversal_changes_order_not_contract(self) -> None:
        contract = corrected_eval.load_contracts(ROOT / "manual_action_contracts_v2.json")["81533c9e56ec"]
        normal = corrected_eval.make_prompt(contract, False)
        reverse = corrected_eval.make_prompt(contract, True)
        self.assertNotEqual(normal, reverse)
        for item in contract["required_predicates"] + contract["forbidden_behaviors"]:
            self.assertEqual(normal.count(item["id"]), 1)
            self.assertEqual(reverse.count(item["id"]), 1)
        self.assertNotIn("SOURCE", normal)
        self.assertNotIn("REAL TARGET", normal)

    def test_neutral_trace_stage_has_no_instruction_and_trace_judge_is_trace_only(self) -> None:
        self.assertNotIn("spin", corrected_eval.TRACE_USER_PROMPT.lower())
        self.assertNotIn("requested action", corrected_eval.TRACE_USER_PROMPT.lower())
        trace = {
            "schema_version": corrected_eval.TRACE_SCHEMA, "visual_quality": "yes",
            "dense_temporal_observations": [
                {
                    "index": index, "phase": "early" if index < 4 else "middle" if index < 8 else "late",
                    "actor_orientation": "front", "body_pose": "standing",
                    "hands_and_objects": "scarf held", "continuity_from_previous": "start" if index == 0 else "continuous",
                    "observation": "front-facing standing pose",
                }
                for index in range(12)
            ],
            "ambiguous_or_unseen": ["back orientation unseen"],
            "neutral_summary": "The woman raises a scarf.",
        }
        corrected_eval.validate_trace(trace)
        contract = corrected_eval.load_contracts(ROOT / "manual_action_contracts_v2.json")["81533c9e56ec"]
        prompt = corrected_eval.make_trace_judge_prompt(contract, trace, False)
        self.assertIn("back orientation unseen", prompt)
        self.assertIn("Do not add any transition absent", prompt)

    def test_noncompensatory_gate(self) -> None:
        contract = corrected_eval.load_contracts(ROOT / "manual_action_contracts_v2.json")["d908552fb691"]
        value = {
            "schema_version": corrected_eval.OBSERVATION_SCHEMA,
            "action_observable": "yes",
            "requested_action_complete": "yes",
            "required_predicates": [
                {"id": "hands_move_together", "result": "yes", "evidence": [{"phase": "middle", "observation": "hands approach"}]},
                {"id": "clap_contact", "result": "no", "evidence": [{"phase": "late", "observation": "hands remain apart"}]},
            ],
            "forbidden_behaviors": [
                {"id": "source_bow_retained", "result": "no", "evidence": [{"phase": "middle", "observation": "torso upright"}]},
            ],
            "summary": "No clap contact.",
        }
        parsed = corrected_eval.validate_observation(value, contract)
        self.assertEqual(corrected_eval.observation_gate(parsed), 0)
        value["required_predicates"][1]["result"] = "uncertain"
        self.assertEqual(corrected_eval.observation_gate(value), 2)
        value["required_predicates"][1]["result"] = "yes"
        value["forbidden_behaviors"][0]["result"] = "yes"
        self.assertEqual(corrected_eval.observation_gate(value), 0)

    def test_coverage_only_breaks_equal_strict_gates(self) -> None:
        better_partial = (0, 3.0)
        worse_partial = (0, 1.0)
        self.assertEqual(corrected_eval._winner(better_partial, worse_partial), "anchor")
        # A higher strict gate remains lexicographically dominant regardless
        # of the secondary mean evidence coverage.
        self.assertEqual(corrected_eval._winner((2, 0.5), (0, 4.0)), "anchor")

    def test_pair_repair_atomically_replaces_complete_packet(self) -> None:
        prefix = "3fadbe0c2961"
        rows = []
        for role in corrected_eval.ROLES:
            for pass_index in range(2 if role in {"anchor", "frozen_base"} else 1):
                rows.append({
                    "schema_version": corrected_eval.RECORD_SCHEMA,
                    "pair_prefix": prefix, "role": role, "pass_index": pass_index,
                    "marker": "old",
                })
        repairs = [{**row, "marker": "new"} for row in rows]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "shard.jsonl"
            repair = Path(directory) / "repair.jsonl"
            corrected_eval.write_jsonl(base, rows)
            corrected_eval.write_jsonl(repair, repairs)
            args = type("Args", (), {"base": str(base), "repair": str(repair), "output": str(base)})
            corrected_eval.merge_qwen_repair(args)
            merged = corrected_eval._read_jsonl(base)
        self.assertEqual(len(merged), 8)
        self.assertEqual({row["marker"] for row in merged}, {"new"})

    def test_parser_accepts_fenced_json_and_rejects_invented_ids(self) -> None:
        contract = corrected_eval.load_contracts(ROOT / "manual_action_contracts_v2.json")["3fadbe0c2961"]
        value = {
            "schema_version": corrected_eval.OBSERVATION_SCHEMA,
            "action_observable": "yes", "requested_action_complete": "yes",
            "required_predicates": [
                {"id": row["id"], "result": "yes", "evidence": [{"phase": "middle", "observation": "visible"}]}
                for row in contract["required_predicates"]
            ],
            "forbidden_behaviors": [
                {"id": row["id"], "result": "no", "evidence": [{"phase": "late", "observation": "absent"}]}
                for row in contract["forbidden_behaviors"]
            ],
            "summary": "Action visible.",
        }
        parsed = corrected_eval.parse_observation("```json\n" + json.dumps(value) + "\n```", contract)
        self.assertEqual(corrected_eval.observation_gate(parsed), 4)
        value["required_predicates"][0]["id"] = "invented"
        with self.assertRaises(ValueError):
            corrected_eval.validate_observation(value, contract)

    def test_protected_write_guard_applies_to_controls(self) -> None:
        args = type("Args", (), {
            "manifest": "/does/not/matter", "contracts": "/does/not/matter",
            "ffmpeg": "/does/not/matter", "output_dir": str(corrected_eval.MEV_PROTECTED_ROOT / "forbidden"),
        })
        # The shared guard itself is what build-controls invokes before creating output.
        with self.assertRaises(ValueError):
            corrected_eval.assert_not_protected_write(args.output_dir)

    def test_corrected_html_has_four_synchronized_videos_and_old_audit_link(self) -> None:
        prefix = "81533c9e56ec"
        observation = {
            "schema_version": corrected_eval.OBSERVATION_SCHEMA,
            "action_observable": "yes", "requested_action_complete": "yes",
            "required_predicates": [
                {"id": item["id"], "result": "yes", "evidence": [{"phase": "middle", "observation": "visible"}]}
                for item in corrected_eval.load_contracts(ROOT / "manual_action_contracts_v2.json")[prefix]["required_predicates"]
            ],
            "forbidden_behaviors": [
                {"id": item["id"], "result": "no", "evidence": [{"phase": "late", "observation": "absent"}]}
                for item in corrected_eval.load_contracts(ROOT / "manual_action_contracts_v2.json")[prefix]["forbidden_behaviors"]
            ], "summary": "visible",
        }
        role_records = {
            role: [{"pass_index": index, "observation": observation, "parse_error": None} for index in range(2)]
            for role in ("anchor", "frozen_base")
        }
        case = {
            "pair_prefix": prefix, "instruction": "spin", "source_caption": "stand",
            "target_caption": "spin", "target_action": "full spin", "manual_winner": "anchor",
            "qwen_winner": "anchor", "agrees": True, "human_note": "manual", "pass_winners": ["anchor", "anchor"],
            "gate_scores": {"anchor": [4, 4], "frozen_base": [0, 0]},
            "coverage_scores": {"anchor": [4.0, 4.0], "frozen_base": [1.0, 1.0]},
            "qwen_roles": role_records,
            "intern_scores": {
                role: {"target_action_text_cosine": .1, "source_action_text_cosine": .05, "target_minus_source_action_margin": .05}
                for role in corrected_eval.ROLES
            },
        }
        qwen = {
            "manual_agreement_count": 1, "pair_count": 1, "manual_agreement_rate": 1.0,
            "winner_counts": {"anchor": 1},
            "control_calibration": {"target_forward_strict_pass_count": 1, "source_noop_strict_pass_count": 0, "reverse_below_forward_count": 1, "shuffle_below_forward_count": 1},
        }
        intern = {"calibration": {"admitted_for_candidate_ranking": False, "forward_over_reverse_count": 0, "forward_over_shuffle_count": 0, "forward_over_source_count": 1}}
        rendered = build_corrected_review_html.render_html([case], qwen, intern, "../old/media")
        self.assertEqual(rendered.count("<video "), 4)
        self.assertIn("data-act=\"play\"", rendered)
        self.assertIn("../20260819_anchor_gap16_review/index.html", rendered)
        self.assertIn("rejected diagnostic", rendered)


if __name__ == "__main__":
    unittest.main()
