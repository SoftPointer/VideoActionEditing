from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_schedule_block_source_edge_localization_v2 as runner  # noqa: E402
import stage_a_source_edge_confirmation_contract_v1 as contract  # noqa: E402


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _trace(plan: dict) -> tuple[dict, str]:
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
    native_digest = contract.object_sha256(native)
    trace = {**native, "trace_digest": native_digest}
    if plan["hook"] == "native-unhooked":
        return trace, native_digest
    cell = contract.admitted_cell(plan["schedule_index"], plan["band_name"])
    selected = set(cell["block_indices"])
    per_block = []
    for index in range(30):
        active = index in selected
        deletion = 3 if active and plan["hook"] == "source-off" else 0
        source_on = 3 if active and plan["hook"] == "source-on" else 0
        geometry = None
        if deletion:
            geometry = {
                "schedule_index": plan["schedule_index"],
                "band_name": plan["band_name"],
                "branch_name": "VI_cond",
                "total_tokens": 150,
                "source_tokens": 50,
                "target_tokens": 100,
                "source_query_rows_from_native_full_attention": True,
                "target_query_rows_from_target_KV_only_attention": True,
                "post_rope_token_order_unchanged": True,
            }
        per_block.append(
            {
                "block_index": index,
                "branch_calls": {name: 40 for name in runner.edge.NATIVE_BRANCH_ORDER},
                "schedule_calls": {str(step): 4 for step in range(40)},
                "active_edge_deletion_calls": deletion,
                "active_source_on_calls": source_on,
                "official_delegate_calls": 160 - deletion,
                "last_active_geometry": geometry,
            }
        )
    edge_unsigned = {
        "contract": runner.edge.intervention_contract(),
        "edge_mode": plan["hook"],
        "registered_schedule_index": plan["schedule_index"],
        "band_name": plan["band_name"],
        "selected_blocks": cell["block_indices"],
        "source_bearing_branches": list(runner.edge.SOURCE_BEARING_BRANCHES),
        "expected_active_calls_per_selected_block": 3,
        "per_block": per_block,
        "native_trace_digest": native_digest,
    }
    edge_receipt = {**edge_unsigned, "digest": contract.object_sha256(edge_unsigned)}
    combined = contract.object_sha256({"native": native_digest, "edge": edge_receipt})
    return {
        **trace,
        "source_edge": edge_receipt,
        "source_edge_trace_digest": combined,
    }, combined


class InferStageASourceEdgeConfirmationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.output = self.root / "fresh-output"
        self.manifest_path = self.root / "manifest.json"
        self.manifest_path.write_text("{}\n", encoding="utf-8")
        self.cell = dict(contract.admitted_cell(29, "early_middle"))
        self.plan = [dict(row) for row in contract.build_confirmation_plan(29, "early_middle")]
        self.sentinel = {
            "sentinel_id": contract.SENTINEL_ORDER[0],
            "diversity_role": "animal",
            "source_entity_type": "animal",
            "iid": "iid-correct",
            "action_family": "pick",
            "source_caption": "a dog in a field",
            "source_video": str(self.root / "correct-original.mp4"),
            "source_video_sha256": "1" * 64,
            "latent_shape": [1, 16, 21, 60, 62],
            "seed": 52005001,
            "instructions": {branch: f"complete {branch} instruction" for branch in contract.BRANCHES},
            "instruction_sha256": {},
            "wrong_owner_sentinel_id": contract.SENTINEL_ORDER[2],
            "wrong_owner_iid": "iid-wrong",
            "wrong_owner_source_video": str(self.root / "wrong-original.mp4"),
            "wrong_owner_source_video_sha256": "2" * 64,
            "wrong_owner_latent_shape": [1, 16, 21, 60, 62],
            "wrong_owner_is_equal_geometry_cross_sentinel_control": True,
            "wrong_owner_is_pure_identity_control": False,
        }
        self.sentinel["instruction_sha256"] = {
            branch: _sha_text(text)
            for branch, text in self.sentinel["instructions"].items()
        }
        self.wrong = {
            **self.sentinel,
            "sentinel_id": contract.SENTINEL_ORDER[2],
            "iid": "iid-wrong",
            "source_video": self.sentinel["wrong_owner_source_video"],
            "source_video_sha256": "2" * 64,
        }
        self.manifest = {
            "manifest_digest": "3" * 64,
            "admitted_cell": self.cell,
            "plan": self.plan,
            "sentinels": [self.sentinel, self.wrong],
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _base_cli(self) -> list[str]:
        return [
            "--confirmation-manifest", str(self.manifest_path),
            "--expected-confirmation-manifest-sha256", "4" * 64,
            "--sentinel-id", contract.SENTINEL_ORDER[0],
            "--bernini-root", "/tmp/bernini",
            "--veomni-root", "/tmp/veomni",
            "--checkpoint", "/tmp/checkpoint",
            "--checkpoint-content-manifest", "/tmp/checkpoint-manifest",
            "--expected-checkpoint-content-manifest-sha256", runner.native.source_audit.CHECKPOINT_CONTENT_MANIFEST_SHA256,
            "--expected-checkpoint-tree-sha256", runner.native.legacy.trainer.CHECKPOINT_TREE_SHA256,
            "--output-dir", str(self.output),
            "--runtime-source-revision", "5" * 40,
            "--runtime-source-closure-sha256", "6" * 64,
            "--launcher-source-sha256", "7" * 64,
        ]

    def test_confirmation_cli_has_no_cell_override(self) -> None:
        args = runner.build_parser().parse_args(self._base_cli())
        output, schedules, bands, mode = runner.validate_cli(args)
        self.assertTrue(mode)
        self.assertEqual(output, self.output)
        self.assertEqual(schedules, ())
        self.assertEqual(bands, ())
        args = runner.build_parser().parse_args(
            self._base_cli() + ["--schedule-indices", "29"]
        )
        with self.assertRaisesRegex(runner.DecodedSourceEdgeError, "overrides"):
            runner.validate_cli(args)

    def test_legacy_explicit_empty_grid_axis_still_fails_closed(self) -> None:
        common_tail = self._base_cli()[6:]
        for option in ("--schedule-indices", "--block-bands"):
            args = runner.build_parser().parse_args(
                [
                    "--authoring-spec",
                    str(self.manifest_path),
                    "--family",
                    "dog",
                    option,
                    "",
                    *common_tail,
                ]
            )
            with self.subTest(option=option), self.assertRaises(
                runner.DecodedSourceEdgeError
            ):
                runner.validate_cli(args)

    def test_runtime_authority_uses_manifest_cell_and_full_instructions(self) -> None:
        with mock.patch.object(
            runner.confirmation, "load_manifest", return_value=self.manifest
        ):
            loaded = runner.load_confirmation_runtime_authority(
                self.manifest_path,
                expected_sha256="4" * 64,
                sentinel_id=contract.SENTINEL_ORDER[0],
            )
        _, row, correct, wrong, binding, captions, plan, path, digest = loaded
        self.assertEqual(row["iid"], "iid-correct")
        self.assertEqual(correct["geometry_source_video"], self.sentinel["source_video"])
        self.assertEqual(wrong["iid"], "iid-wrong")
        self.assertEqual(binding["latent_shape"], [1, 16, 21, 60, 62])
        self.assertEqual(captions, self.sentinel["instructions"])
        self.assertEqual(list(plan), self.plan)
        self.assertEqual(path, self.manifest_path)
        self.assertEqual(digest, "4" * 64)

    def test_evaluator_free_receipt_roundtrip(self) -> None:
        stage = self.root / "stage"
        stage.mkdir()
        correct_snapshot = stage / "source-correct.mp4"
        wrong_snapshot = stage / "source-wrong-owner.mp4"
        correct_snapshot.write_bytes(b"correct")
        wrong_snapshot.write_bytes(b"wrong")
        prompt_records = {
            branch: {
                "caption": self.sentinel["instructions"][branch],
                "caption_utf8_sha256": self.sentinel["instruction_sha256"][branch],
                "native_prompt_utf8_sha256": _sha_text(f"native {branch}"),
            }
            for branch in contract.BRANCHES
        }
        candidates = []
        traces = {}
        identities = {}
        captures = {}
        outputs = {}
        gaussian = "8" * 64
        native_endpoint = "9" * 64
        for index, plan in enumerate(self.plan):
            trace, consensus = _trace(plan)
            traces[plan["key"]] = trace
            endpoint = native_endpoint if plan["hook"] == "source-on" or plan["key"] == "native-correct-forward" else f"{index:x}" * 64
            endpoint = endpoint[:64]
            identities[plan["key"]] = {
                "all_rank_exact": True,
                "identity": {"raw_storage_sha256": endpoint},
            }
            captures[plan["key"]] = {"all_rank_exact": True, "identity": {}}
            video = stage / f"{plan['key']}.mp4"
            video.write_bytes(plan["key"].encode("utf-8"))
            outputs[plan["key"]] = {
                "path": str(video),
                "sha256": contract.file_sha256(video),
                "frame_count": 81,
                "fps": 25,
                "height": 480,
                "width": 496,
            }
            candidates.append(
                {
                    **plan,
                    "initial_gaussian_raw_sha256": gaussian,
                    "trace_all_rank": {"all_rank_exact": True, "value": consensus},
                }
            )
        receipt = runner._confirmation_receipt(
            output_dir=self.output,
            stage=stage,
            confirmation_manifest=self.manifest,
            confirmation_manifest_path=self.manifest_path,
            confirmation_manifest_sha256="4" * 64,
            sentinel=self.sentinel,
            plan=self.plan,
            prompt_records=prompt_records,
            candidates=candidates,
            traces=traces,
            generated_identities=identities,
            capture_identities=captures,
            outputs=outputs,
            correct_snapshot=correct_snapshot,
            wrong_snapshot=wrong_snapshot,
            shared_gaussian_sha256=gaussian,
            forward_endpoint_sha256=native_endpoint,
            checkpoint=self.root / "checkpoint",
            checkpoint_tree_sha256="a" * 64,
            checkpoint_identity={"identity": True},
            bernini_revision="b" * 40,
            veomni_revision="c" * 40,
            wan_sha256="d" * 64,
            inference_hashes={"file": "e" * 64},
            runtime_revision="f" * 40,
            runtime_closure_sha256="0" * 64,
            launcher_sha256="1" * 64,
            freeze_certificate={"frozen": True},
            prompt_guard={"guard": "same"},
            sampling_guard_before={"guard": "same"},
            sampling_guard_after={"guard": "same"},
            host_trim_after_load={"trimmed": True},
            runtime_versions={"torch": "test"},
        )
        payload = json.dumps(receipt, sort_keys=True).lower()
        for forbidden in ('"score"', '"reward"', '"ranking"', '"scalar"', '"optimizer"'):
            self.assertNotIn(forbidden, payload)
        contract.validate_receipt(
            receipt,
            manifest_value=self.manifest,
            manifest_path=self.manifest_path,
            manifest_file_sha256="4" * 64,
            sentinel_id=contract.SENTINEL_ORDER[0],
            media_root=stage,
            verify_media=False,
        )
        self.assertEqual(len(receipt["records"]), 14)
        self.assertFalse(receipt["frozen_execution"]["stage_b_admission"])

        bad = copy.deepcopy(receipt)
        bad_record = bad["records"][8]
        bad_record["memory_source_video_sha256"] = "2" * 64
        bad_record.pop("record_digest")
        bad_record["record_digest"] = contract.object_sha256(bad_record)
        bad.pop("receipt_digest")
        bad["receipt_digest"] = contract.object_sha256(bad)
        with self.assertRaisesRegex(
            contract.SourceEdgeConfirmationError, "source/instruction/seed"
        ):
            contract.validate_receipt(
                bad,
                manifest_value=self.manifest,
                manifest_path=self.manifest_path,
                manifest_file_sha256="4" * 64,
                sentinel_id=contract.SENTINEL_ORDER[0],
                media_root=stage,
                verify_media=False,
            )

        bad = copy.deepcopy(receipt)
        parity_index = next(
            index for index, row in enumerate(bad["records"])
            if row["hook"] == "source-on"
        )
        bad_record = bad["records"][parity_index]
        bad_record["predecode_endpoint_sha256"] = "e" * 64
        bad_record.pop("record_digest")
        bad_record["record_digest"] = contract.object_sha256(bad_record)
        bad.pop("receipt_digest")
        bad["receipt_digest"] = contract.object_sha256(bad)
        with self.assertRaisesRegex(
            contract.SourceEdgeConfirmationError, "Gaussian/parity"
        ):
            contract.validate_receipt(
                bad,
                manifest_value=self.manifest,
                manifest_path=self.manifest_path,
                manifest_file_sha256="4" * 64,
                sentinel_id=contract.SENTINEL_ORDER[0],
                media_root=stage,
                verify_media=False,
            )

        bad = copy.deepcopy(receipt)
        bad["records"][0]["score"] = None
        bad.pop("receipt_digest")
        bad["receipt_digest"] = contract.object_sha256(bad)
        with self.assertRaisesRegex(
            contract.SourceEdgeConfirmationError, "forbidden"
        ):
            contract.validate_receipt(
                bad,
                manifest_value=self.manifest,
                manifest_path=self.manifest_path,
                manifest_file_sha256="4" * 64,
                sentinel_id=contract.SENTINEL_ORDER[0],
                media_root=stage,
                verify_media=False,
            )

        bad = copy.deepcopy(receipt)
        bad_record = bad["records"][1]
        bad_record["relative_mp4"] = bad["records"][0]["relative_mp4"]
        bad_record.pop("record_digest")
        bad_record["record_digest"] = contract.object_sha256(bad_record)
        bad.pop("receipt_digest")
        bad["receipt_digest"] = contract.object_sha256(bad)
        with self.assertRaisesRegex(
            contract.SourceEdgeConfirmationError, "fourteen distinct"
        ):
            contract.validate_receipt(
                bad,
                manifest_value=self.manifest,
                manifest_path=self.manifest_path,
                manifest_file_sha256="4" * 64,
                sentinel_id=contract.SENTINEL_ORDER[0],
                media_root=stage,
                verify_media=False,
            )


if __name__ == "__main__":
    unittest.main()
