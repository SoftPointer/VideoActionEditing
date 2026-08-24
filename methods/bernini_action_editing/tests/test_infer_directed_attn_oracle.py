from __future__ import annotations

import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import directed_source_attention as directed  # noqa: E402
import infer_directed_attn_oracle as inference  # noqa: E402


def _training_source_identity() -> dict[str, str]:
    return {
        "training_method_source_revision": "a" * 40,
        "training_method_source_archive_sha256": "b" * 64,
    }


def _legacy_receipt() -> dict:
    value = {
        "schema_version": inference.legacy_inference.INFERENCE_RECEIPT_SCHEMA,
        "adapter": {
            "training_global_step": 644,
            "adapter_model_sha256": inference.FULL644_ADAPTER_SHA256,
            "tensor_count": 480,
            "strictly_reloaded": True,
            "safe_merged_for_inference": True,
        },
        "input": {
            "accepted_model_conditions": ["source_video", "edit_instruction"],
            "target_video_argument": False,
            "target_accessed_by_inference": False,
            "external_mask_or_swept_tube": False,
            "external_tracking_pose_or_trajectory": False,
            "reference_image_or_video": False,
            "external_shared_i0": False,
        },
        "sampling": {"num_frames": 81, "num_inference_steps": 40},
        "experimental_inference": True,
        "production_claim_forbidden": True,
        "scientific_claim_authorized": False,
    }
    value["receipt_digest"] = inference.legacy_inference.object_sha256(value)
    return value


class _ReceiptProcessor:
    def __init__(self, index: int) -> None:
        self.index = index

    def statistics(self) -> dict:
        return {
            "block_index": self.index,
            "call_count": 80,
            "full_sequence_length": 12,
            "source_sequence_length": 6,
            "target_sequence_length": 6,
            "ulysses_observed": True,
        }


class _ReceiptTransformer:
    blocks = tuple(object() for _ in range(30))


class _ReceiptHandle:
    selection = "late"
    indices = tuple(range(20, 30))
    processors = tuple(_ReceiptProcessor(index) for index in indices)
    transformer = _ReceiptTransformer()
    restored = True

    receipt = directed.DirectedAttentionPatchHandle.receipt


class _BaseProcessor:
    def __init__(self, index: int) -> None:
        self.index = index

    def _project_qkv(self, *args, **kwargs):
        raise AssertionError("the unit restore test must not execute attention")


class _Attention:
    def __init__(self, index: int) -> None:
        self.processor = _BaseProcessor(index)

    def set_processor(self, processor) -> None:
        self.processor = processor


class _Block:
    def __init__(self, index: int) -> None:
        self.attn1 = _Attention(index)


class _MergedModel:
    def __init__(self) -> None:
        self.blocks = tuple(_Block(index) for index in range(30))

    def patch_vae_latent(self):
        raise AssertionError("resolver checks only that this method exists")


class WrapperContractTests(unittest.TestCase):
    def test_oracle_parameter_is_removed_before_legacy_parser(self) -> None:
        oracle, remaining = inference.split_oracle_arguments(
            [
                "--source-video",
                "/x/source.mp4",
                "--directed-attn-blocks",
                "mid",
                "--instruction",
                "move",
            ]
        )
        self.assertEqual(oracle.directed_attn_blocks, "mid")
        self.assertEqual(
            remaining,
            ["--source-video", "/x/source.mp4", "--instruction", "move"],
        )

    def test_receipt_is_full644_source_only_untrained_and_digest_bound(self) -> None:
        original = _legacy_receipt()
        value = inference.augment_inference_receipt(
            original,
            handle=_ReceiptHandle(),
            training_source_identity=_training_source_identity(),
        )
        self.assertEqual(
            value["schema_version"], inference.INFERENCE_RECEIPT_SCHEMA
        )
        self.assertTrue(value["untrained_oracle"])
        self.assertTrue(value["source_and_instruction_only"])
        self.assertTrue(value["production_claim_forbidden"])
        self.assertTrue(value["scientific_claim_forbidden"])
        self.assertFalse(value["scientific_claim_authorized"])
        oracle = value["oracle"]
        self.assertEqual(oracle["probe_arm"], "C_late_attn1")
        self.assertTrue(oracle["zero_training"])
        self.assertEqual(oracle["trained_parameters"], 0)
        self.assertTrue(oracle["full644_adapter_frozen"])
        self.assertEqual(
            oracle["training_method_source_revision"], "a" * 40
        )
        self.assertEqual(
            oracle["training_method_source_archive_sha256"], "b" * 64
        )
        attention = oracle["directed_source_attention"]
        self.assertEqual(attention["block_selection"], "late")
        self.assertEqual(attention["block_indices"], list(range(20, 30)))
        self.assertTrue(attention["runtime"]["restored"])
        certificate = oracle["runtime_execution_certificate"]
        self.assertTrue(certificate["validated"])
        self.assertEqual(certificate["expected_calls_per_selected_block"], 80)
        self.assertEqual(certificate["common_source_sequence_length"], 6)
        self.assertTrue(
            certificate["ulysses_observed_on_every_selected_block"]
        )
        self.assertEqual(
            value["input"]["accepted_model_conditions"],
            ["source_video", "edit_instruction"],
        )
        digest = value.pop("receipt_digest")
        self.assertEqual(digest, inference.legacy_inference.object_sha256(value))
        self.assertEqual(original, _legacy_receipt())

    def test_non_full644_and_privileged_receipts_fail_closed(self) -> None:
        for mutate in (
            lambda value: value["adapter"].update(training_global_step=643),
            lambda value: value["adapter"].update(adapter_model_sha256="0" * 64),
            lambda value: value["input"].update(target_video_argument=True),
        ):
            with self.subTest(mutate=mutate):
                value = _legacy_receipt()
                mutate(value)
                with self.assertRaises(inference.DirectedOracleInferenceError):
                    inference.augment_inference_receipt(
                        value,
                        handle=_ReceiptHandle(),
                        training_source_identity=_training_source_identity(),
                    )

    def test_zero_call_wrong_layout_or_non_ulysses_runtime_fails_closed(self) -> None:
        valid = _ReceiptHandle().receipt()
        mutations = (
            lambda item: item.update(call_count=0),
            lambda item: item.update(target_sequence_length=5),
            lambda item: item.update(full_sequence_length=13),
            lambda item: item.update(ulysses_observed=False),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                runtime = copy.deepcopy(valid)
                mutate(runtime["runtime"]["per_block"][0])
                handle = SimpleNamespace(
                    restored=True,
                    selection="late",
                    receipt=lambda runtime=runtime: runtime,
                )
                with self.assertRaises(
                    inference.DirectedOracleInferenceError
                ):
                    inference.augment_inference_receipt(
                        _legacy_receipt(),
                        handle=handle,
                        training_source_identity=_training_source_identity(),
                    )

    def test_wrapper_patches_only_after_strict_merge_and_restores_processors(self) -> None:
        merged = _MergedModel()
        originals = tuple(block.attn1.processor for block in merged.blocks)
        written: dict = {}
        observed_legacy_args: list[str] = []
        events: list[str] = []

        def strict_loader(base_model, adapter, expected_targets):
            del base_model, adapter, expected_targets
            self.assertTrue(
                all(
                    block.attn1.processor is original
                    for block, original in zip(merged.blocks, originals)
                )
            )
            events.append("strict_full644_load_and_safe_merge")
            return merged, 480

        def atomic_writer(path, value):
            written["path"] = path
            written["receipt"] = copy.deepcopy(value)
            events.append("receipt_write")

        def legacy_main(argv):
            observed_legacy_args.extend(argv)
            model, count = inference.legacy_inference._strict_load_and_merge_adapter(
                object(), object(), []
            )
            self.assertIs(model, merged)
            self.assertEqual(count, 480)
            self.assertEqual(events, ["strict_full644_load_and_safe_merge"])
            self.assertTrue(
                all(
                    isinstance(merged.blocks[index].attn1.processor, directed.DirectedSourceSelfAttnProcessor)
                    for index in range(10, 20)
                )
            )
            self.assertTrue(
                all(
                    merged.blocks[index].attn1.processor is originals[index]
                    for index in tuple(range(10)) + tuple(range(20, 30))
                )
            )
            for index in range(10, 20):
                processor = merged.blocks[index].attn1.processor
                processor.call_count = 80
                processor.full_sequence_length = 12
                processor.source_sequence_length = 6
                processor.saw_ulysses = True
            inference.legacy_inference._atomic_write_json(
                Path("/tmp/oracle-receipt.json"),
                _legacy_receipt(),
            )
            self.assertTrue(
                all(
                    block.attn1.processor is original
                    for block, original in zip(merged.blocks, originals)
                )
            )
            return 17

        with mock.patch.object(
            inference.legacy_inference,
            "_strict_load_and_merge_adapter",
            new=strict_loader,
        ), mock.patch.object(
            inference.legacy_inference,
            "_atomic_write_json",
            new=atomic_writer,
        ), mock.patch.object(
            inference.legacy_inference,
            "main",
            new=legacy_main,
        ), mock.patch.object(
            inference,
            "validate_full644_adapter_bundle",
            return_value=_training_source_identity(),
        ):
            status = inference.main(
                [
                    "--directed-attn-blocks",
                    "mid",
                    "--source-video",
                    "/x/source.mp4",
                ]
            )
            self.assertIs(
                inference.legacy_inference._strict_load_and_merge_adapter,
                strict_loader,
            )
            self.assertIs(inference.legacy_inference._atomic_write_json, atomic_writer)

        self.assertEqual(status, 17)
        self.assertEqual(
            observed_legacy_args,
            ["--source-video", "/x/source.mp4"],
        )
        self.assertEqual(events, ["strict_full644_load_and_safe_merge", "receipt_write"])
        self.assertEqual(
            written["receipt"]["oracle"]["directed_source_attention"][
                "block_selection"
            ],
            "mid",
        )
        self.assertEqual(written["receipt"]["oracle"]["probe_arm"], "B_mid_attn1")
        self.assertTrue(
            written["receipt"]["oracle"]["directed_source_attention"]["runtime"][
                "restored"
            ]
        )
        self.assertTrue(
            all(
                block.attn1.processor is original
                for block, original in zip(merged.blocks, originals)
            )
        )

    def test_wrapper_restores_processors_when_legacy_inference_fails(self) -> None:
        merged = _MergedModel()
        originals = tuple(block.attn1.processor for block in merged.blocks)

        def strict_loader(base_model, adapter, expected_targets):
            del base_model, adapter, expected_targets
            return merged, 480

        def failing_legacy_main(argv):
            del argv
            inference.legacy_inference._strict_load_and_merge_adapter(
                object(), object(), []
            )
            self.assertTrue(
                all(
                    isinstance(
                        block.attn1.processor,
                        directed.DirectedSourceSelfAttnProcessor,
                    )
                    for block in merged.blocks
                )
            )
            raise RuntimeError("synthetic generation failure")

        with mock.patch.object(
            inference.legacy_inference,
            "_strict_load_and_merge_adapter",
            new=strict_loader,
        ), mock.patch.object(
            inference.legacy_inference,
            "main",
            new=failing_legacy_main,
        ), mock.patch.object(
            inference,
            "validate_full644_adapter_bundle",
            return_value=_training_source_identity(),
        ):
            with self.assertRaisesRegex(RuntimeError, "synthetic generation failure"):
                inference.main(["--directed-attn-blocks", "all"])

        self.assertTrue(
            all(
                block.attn1.processor is original
                for block, original in zip(merged.blocks, originals)
            )
        )

    def test_launcher_requires_git_archive_and_runtime_certificate(self) -> None:
        script = (
            METHOD_ROOT
            / "scripts"
            / "auh_infer_directed_attn_oracle.sbatch"
        ).read_text(encoding="utf-8")
        for needle in (
            "BERNINI_ACTION_SOURCE_REPOSITORY",
            "git -C \"${source_repository}\" archive",
            "test_directed_source_attention.py",
            "expected_calls_per_selected_block",
            "ulysses_observed_on_every_selected_block",
            "training_method_source_revision",
            "training_method_source_archive_sha256",
        ):
            self.assertIn(needle, script)


if __name__ == "__main__":
    unittest.main()
