from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock


try:
    import torch  # noqa: F401
except ModuleNotFoundError as error:
    raise unittest.SkipTest("v16 online-anchor tests require torch") from error


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

import train_online_anchor_attention_full644_dynamic_static_v16 as method


WORKER = ROOT / "scripts/auh_train_online_anchor_full644_dynamic_static_v16.sh"
CONTROLLER = ROOT / "scripts/auh_launch_online_anchor_full644_dynamic_static_v16.sh"


def manifest_fixture() -> tuple[dict, list[dict]]:
    rows = []
    for ordinal in range(method.FULL644_ROWS):
        iid = f"{ordinal:016x}"
        rows.append(
            {
                "iid": iid,
                "family": f"family_{ordinal % 28:02d}",
                "instruction": f"perform action {ordinal}",
                "noop_instruction": "keep the source unchanged",
                "strict_selection_gates_all_true": ordinal < 359,
                "posterior_pair": {
                    "source_role_index": 0,
                    "action_anchor_role_index": 1,
                    "parquet_path": f"/sealed/{iid}.parquet",
                    "parquet_sha256": "1" * 64,
                    "source_blob_sha256": "2" * 64,
                    "action_anchor_blob_sha256": "3" * 64,
                },
            }
        )
    manifest = {
        "schema_version": method.FULL644_SCHEMA,
        "authorization_label": method.FULL644_AUTHORIZATION,
        "row_count": method.FULL644_ROWS,
        "strict_row_count": 359,
        "broad_row_count": 285,
        "optimizer_schedule": "exact644_unique_rows_once",
        "source_anchor_role": (
            "identity_appearance_background_camera_and_non_target_preservation"
        ),
        "self_generated_action_anchor_role": "dense_action_trajectory_supervision",
        "paired_ground_truth_claimed": False,
        "qwen_or_other_verifier_controls_optimizer_admission": False,
        "production_claim_forbidden": True,
        "scientific_claim_authorized": False,
        "manifest_digest": "4" * 64,
    }
    return manifest, rows


class Full644DynamicStaticV16Test(unittest.TestCase):
    def setUp(self) -> None:
        method._RUNTIME_AUDIT = method._empty_runtime_audit()
        method._PAIR_CACHE.clear()

    def args(self, output: Path, *, steps: int = 64):
        manifest = "/tmp/full644_action_anchor_manifest_v1.json"
        return method.build_parser().parse_args(
            [
                "--bernini-root", "/tmp/bernini",
                "--veomni-root", "/tmp/veomni",
                "--checkpoint", "/tmp/Bernini-R-1.3B-Diffusers-ff4c5d4",
                "--pair-manifest", manifest,
                "--authoring", manifest,
                "--output", str(output),
                "--profile", "dynamic_static",
                "--route-operator", method.v15.ROUTE_OPERATOR,
                "--max-steps", str(steps),
                "--micro-records", "2",
                "--source-variant", "not_applicable",
                "--route-strength", "0.25",
                "--teacher-route-strength", "0.50",
                "--training-objective", method.v15.OBJECTIVE,
                "--training-interface", "first_phase_caption_i2v",
                "--paired-target-fm-weight", "0",
                "--real-source-manifest", manifest,
                "--real-source-manifest-sha256", "8" * 64,
                "--full644-manifest-sha256", "8" * 64,
                "--teacher-delta-mode", "raw",
                "--routed-teacher-mode", "same_action_route_only",
                "--source-reconstruction-weight", "0.025",
                "--replay-combine-mode", method.v15.REPLAY_COMBINE_MODE,
                "--source-reconstruction-prompt", "action",
                "--learning-rate", "1e-5",
                "--method-source-revision", "1" * 64,
                "--method-source-archive-sha256", "2" * 64,
            ]
        )

    def test_validation_requires_one_fresh_continuous_s644(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            method.validate_args(self.args(root / "fresh-v16-s644", steps=644))
            with self.assertRaises(method.base.OnlineAnchorTrainingError):
                method.validate_args(self.args(root / "fresh-v16-s64", steps=64))
            with self.assertRaises(method.base.OnlineAnchorTrainingError):
                method.validate_args(self.args(root / "fresh-v15-s644", steps=644))

    def test_exact644_manifest_contract_admits_strict_and_broad_rows(self):
        manifest, rows = manifest_fixture()
        method._validate_manifest_document(manifest, rows)
        self.assertEqual(sum(row["strict_selection_gates_all_true"] for row in rows), 359)
        malformed = list(rows)
        malformed[0] = {**malformed[0], "posterior_pair": {**malformed[0]["posterior_pair"], "action_anchor_role_index": 0}}
        with self.assertRaises(method.base.OnlineAnchorTrainingError):
            method._validate_manifest_document(manifest, malformed)

    def test_family_round_robin_closes_all_families_in_first28(self):
        _manifest, rows = manifest_fixture()
        ordered = method.family_round_robin_rows_v16(rows)
        self.assertEqual(len(ordered), 644)
        self.assertEqual(len({row["family"] for row in ordered[:28]}), 28)
        self.assertEqual(len({row["iid"] for row in ordered}), 644)

    def test_donor_is_same_iid_role1_binding(self):
        target = {"iid": "t", "family": "walk", "_v16_ordinal": 0}
        other = {"iid": "o", "family": "walk", "_v16_ordinal": 1}
        registry = method.Full644Registry([target, other])
        selected = method.donor_row_full644_v16(
            target, registry, donor_index=2
        )
        self.assertIs(selected, target)
        self.assertEqual(method._RUNTIME_AUDIT["same_iid_donor_count"], 1)
        self.assertEqual(method._RUNTIME_AUDIT["donor_iids"], {"t"})

    def test_receipt_binds_exact_prefix_and_no_manual_admission_gate(self):
        iids = tuple(f"{index:016x}" for index in range(644))
        targets = set(iids[:64])
        strict = set(iids[:64:2])
        broad = targets - strict
        method._RUNTIME_AUDIT.update(
            {
                "manifest_path": "/sealed/full644.json",
                "manifest_sha256": "8" * 64,
                "manifest_digest": "9" * 64,
                "manifest_iids": iids,
                "manifest_families": tuple(f"family_{i:02d}" for i in range(28)),
                "strict_manifest_count": 359,
                "broad_manifest_count": 285,
                "target_iids": targets,
                "target_families": {f"family_{i:02d}" for i in range(28)},
                "strict_target_iids": strict,
                "broad_target_iids": broad,
                "donor_iids": targets,
                "donor_families": {f"family_{i:02d}" for i in range(28)},
                "donor_selection_count": 128,
                "same_iid_donor_count": 128,
                "observed_latent_shapes": {(1, 16, 21, 72, 52)},
                "pair_decode_count": 64,
                "pair_cache_hit_count": 300,
            }
        )
        inherited = {
            "schema_version": method.r2.RECEIPT_SCHEMA,
            "global_step": 64,
            "training_contract": {
                "method": method.r2.METHOD,
                "anchor_dynamic_static_pairs_audited": 128,
            },
        }
        with mock.patch.object(
            method, "_R2_CHECKPOINT_RECEIPT", return_value=inherited
        ):
            receipt = method.checkpoint_receipt(args=object())
        contract = receipt["training_contract"]
        summary = receipt["v16_full644_summary"]
        self.assertEqual(receipt["schema_version"], method.RECEIPT_SCHEMA)
        self.assertEqual(contract["actual_distinct_target_iid_count"], 64)
        self.assertFalse(contract["strict_selection_flag_filters_optimizer_rows"])
        self.assertTrue(contract["broad_and_strict_rows_are_both_optimizer_admitted"])
        self.assertFalse(contract["anchor_cross_appearance"])
        self.assertTrue(contract["anchor_source_and_donor_share_iid"])
        self.assertNotIn("actual_distinct_cross_appearance_donor_iids", contract)
        self.assertNotIn("actual_distinct_target_events", contract)
        self.assertFalse(summary["manual_or_visual_review_controls_optimizer_admission"])
        self.assertEqual(summary["donor_selection_count"], 128)
        self.assertEqual(summary["same_iid_role1_donor_count"], 128)
        self.assertFalse(summary["all_full644_rows_targeted_exactly_once"])

    def test_worker_controller_bind_dynamic_job_node_gpus_and_run_one_s644(self):
        worker = WORKER.read_text(encoding="utf-8")
        controller = CONTROLLER.read_text(encoding="utf-8")
        self.assertIn("ONLINE_ANCHOR_JOB_ID:?", worker)
        self.assertIn("ONLINE_ANCHOR_NODE:?", worker)
        self.assertIn("ONLINE_ANCHOR_GPU_DEVICES:-0,1,2,3", worker)
        self.assertIn('ROCR_VISIBLE_DEVICES="$gpu_devices"', worker)
        self.assertIn("ONLINE_ANCHOR_SOURCE_TREE:?", worker)
        self.assertIn("ONLINE_ANCHOR_RELEASE:?", worker)
        self.assertIn("--full644-manifest-sha256", worker)
        self.assertIn("ONLINE_ANCHOR_JOB_ID:?", controller)
        self.assertIn("squeue --noheader --jobs=\"$job\"", controller)
        self.assertIn('ONLINE_ANCHOR_GPU_DEVICES="$gpu_devices"', controller)
        self.assertIn("run_exact644 644", controller)
        invocation_lines = {
            line.strip()
            for line in controller.splitlines()
            if line.startswith("run_exact644 ")
        }
        self.assertEqual(invocation_lines, {"run_exact644 644"})
        self.assertIn("anchor_cross_appearance == false", controller)
        self.assertIn("manual_or_visual_review_controls_optimizer_admission == false", controller)

    def test_main_patches_the_base_parser_seen_by_the_final_entrypoint(self):
        original = method.base.build_parser

        def observe_base_parser(_argv):
            destinations = {
                action.dest for action in method.base.build_parser()._actions
            }
            self.assertIn("full644_manifest_sha256", destinations)
            return 0

        with mock.patch.object(method.r2, "main", side_effect=observe_base_parser):
            self.assertEqual(method.main([]), 0)
        self.assertIs(method.base.build_parser, original)


if __name__ == "__main__":
    unittest.main()
