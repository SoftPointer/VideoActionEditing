from __future__ import annotations

import argparse
from dataclasses import asdict
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from spt_v2 import infer_spt as inference  # noqa: E402
from spt_v2 import phase_query_planner as phase_query  # noqa: E402
from spt_v2 import unipc_projection as projection  # noqa: E402


SHA1 = "1" * 40
SHA256 = "2" * 64


def _args(**overrides):
    values = {
        "instruction": "Make the actor crouch.",
        "num_inference_steps": 40,
        "seed": 42,
        "max_generate_fraction": 0.12,
        "expected_bernini_commit": inference.trainer.BERNINI_OFFICIAL_COMMIT,
        "expected_veomni_commit": inference.trainer.VEOMNI_TESTED_COMMIT,
        "expected_checkpoint_tree_sha256": inference.trainer.CHECKPOINT_TREE_SHA256,
        "method_source_revision": SHA1,
        "method_source_archive_sha256": SHA256,
        "checkpoint": "/checkpoint/Bernini-R-1.3B-Diffusers",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _valid_metadata(config=None, *, parameter_names=None, parameter_count=1):
    config = config or phase_query.PhaseQueryPlannerConfig()
    raw_config = asdict(config)
    parameter_names = parameter_names or ["phase_queries"]
    immutable_value = {
        "method": inference.student_train.METHOD_NAME,
        "planner_architecture": phase_query.ARCHITECTURE_NAME,
        "checkpoint_tree_sha256": inference.trainer.CHECKPOINT_TREE_SHA256,
        "planner_config": json.loads(
            inference.base.canonical_json_bytes(raw_config).decode("utf-8")
        ),
        "student_semantic_inputs": ["source_video", "edit_instruction"],
        "instruction_representation": "full_unpadded_t5_token_sequence",
        "instruction_pooling": None,
        "phase_query_count": 21,
        "cross_attention_layers": 2,
        "target_used_by_student": False,
        "target_used_by_training_teacher_only": True,
    }
    receipt = {
        "schema_version": inference.student_train.RECEIPT_SCHEMA,
        "method": inference.student_train.METHOD_NAME,
        "global_step": 17,
        "immutable_contract": {
            "value": immutable_value,
            "digest": inference.base.object_sha256(immutable_value),
        },
        "planner": {
            "class": "PhaseQueryPlanner",
            "architecture": phase_query.ARCHITECTURE_NAME,
            "parameter_count": parameter_count,
            "parameter_names": parameter_names,
            "parameter_names_sha256": inference.base.object_sha256(parameter_names),
        },
        "supervision": {
            "student_api": ["source", "instruction_tokens"],
            "instruction_representation": "full_unpadded_t5_token_sequence",
            "instruction_pooling": None,
            "learned_phase_queries": 21,
            "explicit_sinusoidal_phase_encoding": True,
            "cross_attention_layers": 2,
            "student_target_argument_exists": False,
            "target_used_by_oracle_teacher_only": True,
            "external_mask_track_pose_flow": False,
            "max_generate_fraction_per_phase": 0.12,
            "generate_budget_reject_fallback": "preserve",
            "latent_phases": 21,
        },
        "dataset": {"diagnostic_subset": False},
        "production_claim_forbidden": True,
        "scientific_claim_authorized": False,
    }
    receipt["receipt_digest"] = inference.base.object_sha256(receipt)
    return raw_config, receipt


class PureSPTInferenceContractTests(unittest.TestCase):
    def test_cli_is_formal_source_instruction_planner_only(self) -> None:
        parser = inference.build_parser()
        destinations = {action.dest for action in parser._actions}
        self.assertTrue(
            {
                "planner_checkpoint",
                "source_video",
                "instruction",
                "num_inference_steps",
                "max_generate_fraction",
            }
            <= destinations
        )
        forbidden = {
            "adapter_checkpoint",
            "lora",
            "target_video",
            "oracle_plan",
            "oracle",
            "mask",
            "track",
            "pose",
            "flow",
            "trajectory",
            "first_frame_anchor",
        }
        self.assertTrue(destinations.isdisjoint(forbidden))
        args = parser.parse_args(
            [
                "--bernini-root", "/b",
                "--veomni-root", "/v",
                "--checkpoint", "/c",
                "--planner-checkpoint", "/p",
                "--source-video", "/source.mp4",
                "--instruction", "move",
                "--output", "/out.mp4",
                "--method-source-revision", SHA1,
                "--method-source-archive-sha256", SHA256,
            ]
        )
        self.assertEqual(args.num_inference_steps, 40)
        self.assertEqual(args.max_generate_fraction, 0.12)

    def test_formal_cli_rejects_sampler_and_generate_budget_drift(self) -> None:
        inference.validate_cli(_args())
        for changed in (
            {"num_inference_steps": 41},
            {"max_generate_fraction": 0.5},
            {"instruction": "\x00"},
        ):
            with self.subTest(changed=changed), self.assertRaises(
                inference.SPTInferenceError
            ):
                inference.validate_cli(_args(**changed))

    def test_sampler_is_exact81f_official_v2v_apg_unipc(self) -> None:
        contract = inference.exact_sampler_contract(seed=7)
        self.assertEqual(contract["num_frames"], 81)
        self.assertEqual(contract["num_inference_steps"], 40)
        self.assertEqual(contract["guidance_mode"], "v2v_apg")
        self.assertEqual(contract["flow_shift"], 5.0)
        self.assertEqual(contract["seed"], 7)
        self.assertEqual(inference.base.ULYSSES_SIZE, 4)

    def test_student_facing_helpers_have_no_target_or_oracle_argument(self) -> None:
        names = set(inspect.signature(inference.encode_student_instruction).parameters)
        self.assertTrue({"renderer", "input_ids", "attention_mask", "device"} <= names)
        self.assertTrue(names.isdisjoint({"target", "oracle", "mask", "track", "pose"}))
        pack_names = set(inspect.signature(inference.pack_clean_source).parameters)
        self.assertEqual(pack_names, {"source_latent"})

    def test_phase_query_v2_receipt_and_config_are_strict(self) -> None:
        raw_config, receipt = _valid_metadata()
        config, identity = inference.validate_planner_metadata(raw_config, receipt)
        self.assertEqual(config.architecture, phase_query.ARCHITECTURE_NAME)
        self.assertEqual(identity["global_step"], 17)

        old = dict(raw_config)
        old["architecture"] = "global_mean_film_v1"
        with self.assertRaises(inference.SPTInferenceError):
            inference.validate_planner_metadata(old, receipt)

        tampered = json.loads(json.dumps(receipt))
        tampered["supervision"]["student_target_argument_exists"] = True
        tampered["receipt_digest"] = inference.base.object_sha256(
            {key: value for key, value in tampered.items() if key != "receipt_digest"}
        )
        with self.assertRaisesRegex(inference.SPTInferenceError, "student_target"):
            inference.validate_planner_metadata(raw_config, tampered)

    def test_planner_bundle_rejects_adapter_contamination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "checkpoint-00000017"
            root.mkdir()
            for name in (
                "planner.safetensors",
                "planner_config.json",
                "receipt.json",
                "optimizer.pt",
            ):
                (root / name).write_bytes(b"x")
            bundle = inference.resolve_planner_checkpoint(root)
            self.assertEqual(bundle.root, root)
            (root / "adapter").mkdir()
            with self.assertRaisesRegex(inference.SPTInferenceError, "LoRA/PEFT"):
                inference.resolve_planner_checkpoint(root)

    def test_projection_trace_requires_every_official_step_and_budget(self) -> None:
        records = []
        for index in range(40):
            records.append(
                projection.ProjectionStepRecord(
                    step_index=index,
                    timestep=float(1000 - 20 * index),
                    sigma=1.0 - index / 41.0,
                    projection_applied=True,
                    correction_rms=0.1,
                    preserve_fraction=0.5,
                    transport_fraction=0.45,
                    generate_fraction=0.05,
                    max_sample_generate_fraction=0.05,
                    max_phase_generate_fraction=0.08,
                    generate_budget=0.12,
                )
            )
        trace = projection.ProjectionTrace(
            records=records, max_generate_fraction=0.12, oracle_ablation=False
        )
        payload = inference.validate_projection_trace(trace)
        self.assertEqual(payload["step_count"], 40)

        trace.records[3] = projection.ProjectionStepRecord(
            **{**asdict(trace.records[3]), "max_phase_generate_fraction": 0.5}
        )
        with self.assertRaisesRegex(inference.SPTInferenceError, "generate budget"):
            inference.validate_projection_trace(trace)

    def test_receipt_states_base_only_and_no_privileged_condition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            bundle = inference.PlannerBundle(
                root=root / "checkpoint-00000017",
                weights_path=root / "checkpoint-00000017/planner.safetensors",
                config_path=root / "checkpoint-00000017/planner_config.json",
                receipt_path=root / "checkpoint-00000017/receipt.json",
                optimizer_path=root / "checkpoint-00000017/optimizer.pt",
            )
            receipt = inference.build_inference_receipt(
                args=_args(),
                source_path=root / "source.mp4",
                source_sha256="3" * 64,
                source_metadata={"source_derived_bucket_hw": [480, 496]},
                output_path=root / "out.mp4",
                output_sha256="4" * 64,
                planner_bundle=bundle,
                planner_identity={
                    "weights_sha256": "5" * 64,
                    "strictly_reloaded": True,
                    "optimizer_loaded": False,
                },
                planner_config=phase_query.PhaseQueryPlannerConfig(),
                instruction_token_count=23,
                trace={"step_count": 40},
                bernini_revision=inference.trainer.BERNINI_OFFICIAL_COMMIT,
                veomni_revision=inference.trainer.VEOMNI_TESTED_COMMIT,
                inference_file_hashes={},
                runtime_versions={},
            )
        self.assertFalse(receipt["base_model"]["peft_or_lora_loaded"])
        self.assertFalse(receipt["base_model"]["cdf_adapter_loaded"])
        self.assertFalse(receipt["base_model"]["p3t_adapter_loaded"])
        self.assertEqual(
            receipt["input"]["accepted_model_conditions"],
            ["source_video", "edit_instruction"],
        )
        self.assertFalse(receipt["input"]["target_accessed_by_inference"])
        self.assertFalse(receipt["input"]["oracle_plan_loaded"])
        candidate = dict(receipt)
        declared = candidate.pop("receipt_digest")
        self.assertEqual(inference.base.object_sha256(candidate), declared)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is unavailable")
class TensorSPTInferenceContractTests(unittest.TestCase):
    def test_source_packing_matches_21_phase_64_channel_geometry(self) -> None:
        source = torch.randn(1, 16, 21, 4, 6)
        packed, video, height, width = inference.pack_clean_source(source)
        self.assertEqual(tuple(packed.shape), (1, 21 * 2 * 3, 64))
        self.assertEqual(tuple(video.shape), (1, 21, 2, 3, 64))
        self.assertEqual((height, width), (2, 3))
        self.assertTrue(torch.equal(inference.spt.video_to_packed(video), packed))

    def test_strict_planner_reload_binds_all_tensors_and_parameter_names(self) -> None:
        try:
            from safetensors.torch import save_file
        except Exception as error:
            self.skipTest(f"safetensors unavailable: {error}")
        config = phase_query.PhaseQueryPlannerConfig(
            text_channels=12,
            hidden_channels=32,
            attention_heads=4,
        )
        original = phase_query.PhaseQueryPlanner(config)
        names = [name for name, _ in original.named_parameters()]
        count = sum(int(parameter.numel()) for _, parameter in original.named_parameters())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "checkpoint-00000017"
            root.mkdir()
            weights = root / "planner.safetensors"
            save_file(
                {
                    key: value.detach().cpu().contiguous()
                    for key, value in original.state_dict().items()
                },
                str(weights),
            )
            for name in ("planner_config.json", "receipt.json", "optimizer.pt"):
                (root / name).write_bytes(b"x")
            bundle = inference.resolve_planner_checkpoint(root)
            loaded, identity = inference.strict_load_planner(
                bundle,
                config,
                {"parameter_names": names, "parameter_count": count},
                device=torch.device("cpu"),
            )
            self.assertFalse(loaded.training)
            self.assertTrue(all(not parameter.requires_grad for parameter in loaded.parameters()))
            self.assertTrue(identity["strictly_reloaded"])
            self.assertFalse(identity["optimizer_loaded"])
            self.assertEqual(identity["state_key_count"], len(original.state_dict()))


if __name__ == "__main__":
    unittest.main()
