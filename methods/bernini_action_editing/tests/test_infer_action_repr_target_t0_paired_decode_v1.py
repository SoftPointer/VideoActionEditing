from __future__ import annotations

from pathlib import Path
import copy
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_action_repr_target_t0_paired_decode_v1 as paired
import infer_action_repr_target_t0_matched_decode_v1 as single


def cells(*, baseline_hash: str = "a" * 64) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for index, (key, step, route) in enumerate(paired.COORDINATES):
        digest = baseline_hash if key in paired.BASELINE_KEYS else f"{index + 1:064x}"
        result.append(
            {
                "key": key,
                "checkpoint_step": step,
                "route_kind": route,
                "latent_sha256": digest,
                "four_rank_latent_hashes_equal": True,
                "shared_step_calls": 2 * single.NUM_INFERENCE_STEPS,
                "paired_cfg_timestep_digests_equal": True,
                "video_generated": True,
                "video_sha256": "b" * 64,
                "decoded_rgb24_sha256": "c" * 64,
            }
        )
    return result


class TargetT0PairedDecodeTests(unittest.TestCase):
    def test_matrix_has_complete_baselines_and_step1_controls(self) -> None:
        self.assertEqual(len(paired.COORDINATES), 11)
        self.assertEqual(
            paired.BASELINE_KEYS,
            (
                "s0_route_off_a",
                "s0_zero",
                "s0_correct",
                "s1_route_off",
                "s1_zero",
                "s0_route_off_b",
            ),
        )
        self.assertEqual(
            {route for _, step, route in paired.COORDINATES if step == 1},
            {
                "route_off",
                "zero",
                "correct",
                "temporal_shuffle",
                "reverse",
                "incomplete",
                "wrong_action",
            },
        )

    def test_paired_gate_requires_all_hard_negative_latents_exact(self) -> None:
        rows = cells()
        gate = paired.paired_gate(rows)
        self.assertTrue(gate["baseline_gate_passed"])
        self.assertTrue(gate["step1_correct_latent_changed_from_baseline"])
        self.assertFalse(gate["quality_success_claimed"])
        broken = copy.deepcopy(rows)
        next(row for row in broken if row["key"] == "s1_route_off")[
            "latent_sha256"
        ] = "d" * 64
        failed = paired.paired_gate(broken)
        self.assertFalse(failed["baseline_gate_passed"])
        self.assertFalse(failed["baseline_negative_controls_exact"]["s1_route_off"])

    def test_paired_gate_rejects_missing_or_duplicate_coordinates(self) -> None:
        rows = cells()
        with self.assertRaises(paired.PairedDecodeError):
            paired.paired_gate(rows[:-1])
        duplicated = copy.deepcopy(rows)
        duplicated[-1]["key"] = duplicated[0]["key"]
        with self.assertRaises(paired.PairedDecodeError):
            paired.paired_gate(duplicated)

    def test_receipt_replay_keeps_claims_false(self) -> None:
        rows = cells()
        receipt: dict[str, object] = {
            "schema_version": paired.SCHEMA_VERSION,
            "complete": True,
            "case_id": single.CASE_ID,
            "paired_gate": paired.paired_gate(rows),
            "cells": rows,
            "runtime": {
                "one_model_construction": True,
                "one_native_main_call": True,
                "sample_calls": len(paired.COORDINATES),
                "world_size": 4,
                "ulysses_size": 4,
                "strict_deterministic_algorithms": True,
            },
            "claim_boundary": {
                "ours_claimed": False,
                "quality_success_claimed": False,
                "route_selectivity_claimed": False,
            },
        }
        receipt["receipt_digest"] = single.object_sha256(receipt)
        self.assertIs(paired.validate_paired_receipt(receipt), receipt)
        changed = copy.deepcopy(receipt)
        changed["claim_boundary"]["ours_claimed"] = True
        unsigned = dict(changed)
        unsigned.pop("receipt_digest")
        changed["receipt_digest"] = single.object_sha256(unsigned)
        with self.assertRaises(paired.PairedDecodeError):
            paired.validate_paired_receipt(changed)

    def test_source_has_no_target_media_cli_and_uses_real_per_sample_clamp(self) -> None:
        source = Path(paired.__file__).read_text(encoding="utf-8")
        self.assertNotIn('add_argument("--target-video"', source)
        self.assertNotIn('add_argument("--target-image"', source)
        self.assertIn("with original_clamp(", source)
        self.assertIn("torch.distributed.all_gather_object", source)
        self.assertIn("allow_abbrev=False", source)
        self.assertIn("expand_target_only_route_for_native_mv2v", source)
        self.assertIn("paired native mv2v source+target token layout differs", source)
        self.assertIn("latent_progress.json", source)

    def test_prepared_output_root_requires_exact_empty_cell_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "paired"
            for key, _, _ in paired.COORDINATES:
                (root / "cells" / key).mkdir(parents=True, exist_ok=True)
            self.assertEqual(paired._prepared_output_root(root), root.resolve())
            (root / "cells" / paired.COORDINATES[0][0] / "unexpected.txt").write_text(
                "not allowed", encoding="utf-8"
            )
            with self.assertRaises(paired.PairedDecodeError):
                paired._prepared_output_root(root)


if __name__ == "__main__":
    unittest.main()
