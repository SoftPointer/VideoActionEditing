from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

try:
    import torch
except ImportError as error:  # pragma: no cover
    raise unittest.SkipTest("torch unavailable") from error


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_fewshot_motion_code as inference


SOURCE_PATH = METHOD_ROOT / "infer_fewshot_motion_code.py"


def _valid_tensors() -> dict[str, torch.Tensor]:
    phase = torch.linspace(-0.75, 0.75, 21, dtype=torch.float32).reshape(1, 21)
    phase[:, 0].zero_()
    block = torch.linspace(-0.5, 0.5, 16, dtype=torch.float32).reshape(1, 16, 1)
    block = block.expand(1, 16, 12).clone().contiguous()
    return {"phase_gates": phase.contiguous(), "block_head_gates": block}


def _receipt(
    *,
    state_filename: str,
    state_sha256: str,
    gate: str = "GO",
) -> dict[str, object]:
    code, tied = inference.validate_tied_prototype_tensors(_valid_tensors())
    return inference.build_prototype_training_receipt(
        state_filename=state_filename,
        state_file_sha256=state_sha256,
        motion_code=code,
        tied_code_36d=tied,
        support_tied_code_36d_sha256=("1" * 64, "2" * 64),
        training_gate_receipt_sha256="3" * 64,
        representability_gate=gate,
    )


def _write_json(path: Path, value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


class FewShotMotionInferenceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOURCE_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def _parser_flags(self) -> list[str]:
        return [
            node.args[0].value
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ]

    def test_frozen_81_frame_five_arm_contract(self) -> None:
        self.assertEqual(
            inference.ARM_ORDER,
            ("B0", "Z0", "PROTO", "REVERSE", "SHUFFLE"),
        )
        self.assertNotIn("ORACLE_HELDOUT", inference.ARM_ORDER)
        self.assertEqual(inference.EXPECTED_FRAMES, 81)
        self.assertEqual(inference.EXPECTED_FPS, 25)
        self.assertEqual(inference.EXPECTED_BUCKET_HW, (480, 496))
        self.assertEqual(inference.EXPECTED_LATENT_SHAPE, (1, 16, 21, 60, 62))
        self.assertEqual(inference.PROPOSAL_SEED, 2027)
        self.assertEqual(inference.RENDER_SEED, 2028)
        self.assertEqual(
            set(inference.ARM_OUTER_GATES), set(inference.ARM_ORDER)
        )
        for arm in inference.ARM_ORDER[1:]:
            self.assertEqual(
                inference.ARM_OUTER_GATES[arm], inference.OUTER_CPMR_GATE
            )

    def test_cli_has_no_privileged_or_spatial_condition(self) -> None:
        flags = self._parser_flags()
        joined = " ".join(flags)
        for forbidden in (
            "target",
            "support",
            "mask",
            "flow",
            "pose",
            "track",
            "trajectory",
            "reference",
        ):
            self.assertNotIn(forbidden, joined)
        self.assertIn("--source-video", flags)
        self.assertIn("--instruction", flags)
        self.assertIn("--prototype-state", flags)
        self.assertIn("--prototype-receipt", flags)
        self.assertIn("--allow-no-go-diagnostic", flags)
        self.assertNotIn("--arm", flags)
        self.assertEqual(
            inference.EXTERNAL_SEMANTIC_INPUTS, ("source_video", "instruction")
        )

    def test_real_v11_and_fewshot_runtime_paths_are_wired(self) -> None:
        for fragment in (
            "v11._sample_kwargs(",
            "BerniniRendererModel(config)",
            "legacy._tokenize_training_prompt(",
            "_vae_encode(",
            "_vae_decode(vae, latent)",
            "carrier_core.build_carrier_from_proposal_latents(",
            "motion_branch.install_fewshot_motion_branch(model)",
            "motion_branch.fewshot_motion_code_context(",
            "motion_runtime.cpmr_final_render_hook(",
        ):
            self.assertIn(fragment, self.source)
        self.assertEqual(self.source.count("seed=PROPOSAL_SEED"), 2)
        self.assertIn("for arm in PATCHED_ARM_ORDER", self.source)
        self.assertIn("seed=RENDER_SEED", self.source)
        self.assertIn("input_ids=noop_ids", self.source)
        self.assertIn("Z0 differs bytewise from B0", self.source)

    def test_state_is_exact_cpu_fp32_tied_36d(self) -> None:
        code, tied = inference.validate_tied_prototype_tensors(_valid_tensors())
        self.assertEqual(tuple(code.phase_gates.shape), (1, 21))
        self.assertEqual(tuple(code.block_head_gates.shape), (1, 16, 12))
        self.assertEqual(tuple(tied.shape), (1, 36))
        self.assertEqual(tied.dtype, torch.float32)
        self.assertEqual(tied.device.type, "cpu")
        self.assertTrue(tied.is_contiguous())
        expected = torch.cat(
            (code.phase_gates[:, 1:], code.block_head_gates[:, :, 0]), dim=1
        )
        self.assertTrue(torch.equal(tied, expected))

    def test_state_rejects_unknown_key_shape_dtype_bounds_and_untied_heads(self) -> None:
        cases: list[dict[str, torch.Tensor]] = []
        unknown = _valid_tensors()
        unknown["extra"] = torch.zeros(1)
        cases.append(unknown)
        shape = _valid_tensors()
        shape["phase_gates"] = torch.zeros(1, 20)
        cases.append(shape)
        dtype = _valid_tensors()
        dtype["block_head_gates"] = dtype["block_head_gates"].to(torch.float64)
        cases.append(dtype)
        bounds = _valid_tensors()
        bounds["phase_gates"][0, 1] = 1.01
        cases.append(bounds)
        untied = _valid_tensors()
        untied["block_head_gates"][0, 3, 7] += 0.01
        cases.append(untied)
        for tensors in cases:
            with self.subTest(keys=tuple(tensors), shapes=[tuple(x.shape) for x in tensors.values()]):
                with self.assertRaises(inference.FewShotMotionInferenceError):
                    inference.validate_tied_prototype_tensors(tensors)

    def test_state_rejects_negative_zero_phase_zero_and_head_zero_mismatch(self) -> None:
        negative_phase = _valid_tensors()
        negative_phase["phase_gates"][0, 0] = -0.0
        with self.assertRaises(inference.FewShotMotionInferenceError):
            inference.validate_tied_prototype_tensors(negative_phase)

        head_sign = _valid_tensors()
        head_sign["block_head_gates"][0, 8].zero_()
        head_sign["block_head_gates"][0, 8, 5] = -0.0
        with self.assertRaises(inference.FewShotMotionInferenceError):
            inference.validate_tied_prototype_tensors(head_sign)

    def test_training_receipt_binds_support_codes_gate_and_deployment(self) -> None:
        state_sha = "a" * 64
        receipt = _receipt(
            state_filename="prototype.safetensors", state_sha256=state_sha
        )
        code, tied = inference.validate_tied_prototype_tensors(_valid_tensors())
        validated = inference.validate_prototype_training_receipt(
            receipt,
            state_filename="prototype.safetensors",
            state_file_sha256=state_sha,
            motion_code=code,
            tied_code_36d=tied,
        )
        training = validated["training_provenance"]
        self.assertEqual(
            training["support_iids"], list(inference.EXPECTED_SUPPORT_IIDS)
        )
        self.assertEqual(
            training["support_tied_code_36d_sha256"], ["1" * 64, "2" * 64]
        )
        self.assertEqual(training["training_gate_receipt_sha256"], "3" * 64)
        self.assertEqual(training["representability_gate"], "GO")
        self.assertEqual(
            training["heldout_use_definition"],
            "optimizer_or_model_tensor_use; hash_or_metadata_preflight_is_not_use",
        )
        self.assertTrue(training["heldout_hash_or_metadata_preflight_allowed"])
        deployment = validated["deployment_contract"]
        self.assertTrue(deployment["source_instruction_only"])
        self.assertFalse(deployment["support_available_at_inference"])
        self.assertFalse(deployment["target_available_at_inference"])
        self.assertFalse(deployment["heldout_oracle_available_at_inference"])

    def test_identical_support_code_digests_are_scientifically_legal(self) -> None:
        code, tied = inference.validate_tied_prototype_tensors(_valid_tensors())
        receipt = inference.build_prototype_training_receipt(
            state_filename="prototype.safetensors",
            state_file_sha256="a" * 64,
            motion_code=code,
            tied_code_36d=tied,
            support_tied_code_36d_sha256=("1" * 64, "1" * 64),
            training_gate_receipt_sha256="3" * 64,
        )
        validated = inference.validate_prototype_training_receipt(
            receipt,
            state_filename="prototype.safetensors",
            state_file_sha256="a" * 64,
            motion_code=code,
            tied_code_36d=tied,
        )
        self.assertEqual(
            validated["training_provenance"]["support_tied_code_36d_sha256"],
            ["1" * 64, "1" * 64],
        )

    def test_rehashed_privileged_receipt_is_still_rejected_before_state_load(self) -> None:
        with tempfile.TemporaryDirectory() as root_value:
            root = Path(root_value)
            state_path = root / "prototype.safetensors"
            state_path.write_bytes(b"not-deserialized")
            state_sha = hashlib.sha256(state_path.read_bytes()).hexdigest()
            receipt = _receipt(
                state_filename=state_path.name, state_sha256=state_sha
            )
            receipt["training_provenance"]["heldout_target_used"] = True
            receipt.pop("receipt_digest")
            receipt["receipt_digest"] = inference._object_sha256(receipt)
            receipt_path = root / "prototype.receipt.json"
            receipt_sha = _write_json(receipt_path, receipt)
            with mock.patch.object(inference, "_load_safetensors_cpu") as loader:
                with self.assertRaises(inference.FewShotMotionInferenceError):
                    inference.load_prototype_bundle(
                        state_path,
                        receipt_path,
                        expected_state_sha256=state_sha,
                        expected_receipt_sha256=receipt_sha,
                    )
                loader.assert_not_called()

    def test_no_go_requires_explicit_diagnostic_flag(self) -> None:
        with tempfile.TemporaryDirectory() as root_value:
            root = Path(root_value)
            state_path = root / "prototype.safetensors"
            state_path.write_bytes(b"mock-safetensors")
            state_sha = hashlib.sha256(state_path.read_bytes()).hexdigest()
            receipt = _receipt(
                state_filename=state_path.name,
                state_sha256=state_sha,
                gate="NO_GO",
            )
            receipt_path = root / "prototype.receipt.json"
            receipt_sha = _write_json(receipt_path, receipt)
            with mock.patch.object(
                inference, "_load_safetensors_cpu", return_value=_valid_tensors()
            ):
                with self.assertRaises(inference.FewShotMotionInferenceError):
                    inference.load_prototype_bundle(
                        state_path,
                        receipt_path,
                        expected_state_sha256=state_sha,
                        expected_receipt_sha256=receipt_sha,
                    )
                bundle = inference.load_prototype_bundle(
                    state_path,
                    receipt_path,
                    expected_state_sha256=state_sha,
                    expected_receipt_sha256=receipt_sha,
                    allow_no_go_diagnostic=True,
                )
            self.assertEqual(bundle.representability_gate, "NO_GO")

    def test_go_rejects_unnecessary_no_go_override(self) -> None:
        with tempfile.TemporaryDirectory() as root_value:
            root = Path(root_value)
            state_path = root / "prototype.safetensors"
            state_path.write_bytes(b"mock-safetensors")
            state_sha = hashlib.sha256(state_path.read_bytes()).hexdigest()
            receipt = _receipt(
                state_filename=state_path.name, state_sha256=state_sha, gate="GO"
            )
            receipt_path = root / "prototype.receipt.json"
            receipt_sha = _write_json(receipt_path, receipt)
            with mock.patch.object(
                inference, "_load_safetensors_cpu", return_value=_valid_tensors()
            ):
                with self.assertRaises(inference.FewShotMotionInferenceError):
                    inference.load_prototype_bundle(
                        state_path,
                        receipt_path,
                        expected_state_sha256=state_sha,
                        expected_receipt_sha256=receipt_sha,
                        allow_no_go_diagnostic=True,
                    )

    def test_arm_codes_are_zero_prototype_reverse_and_shuffle_only(self) -> None:
        prototype, _ = inference.validate_tied_prototype_tensors(_valid_tensors())
        arms = inference.build_arm_motion_codes(prototype)
        self.assertEqual(tuple(arms), inference.PATCHED_ARM_ORDER)
        arms["Z0"].validate(require_noop=True)
        self.assertTrue(torch.equal(arms["PROTO"].phase_gates, prototype.phase_gates))
        self.assertTrue(
            torch.equal(
                arms["REVERSE"].phase_gates,
                prototype.phase_gates[:, list(inference.epmc.REVERSE_PHASE_INDICES)],
            )
        )
        self.assertTrue(
            torch.equal(
                arms["SHUFFLE"].phase_gates,
                prototype.phase_gates[:, list(inference.epmc.SHUFFLE_PHASE_INDICES)],
            )
        )
        for arm in ("REVERSE", "SHUFFLE"):
            self.assertTrue(
                torch.equal(arms[arm].block_head_gates, prototype.block_head_gates)
            )

    def test_z0_latent_parity_is_fail_closed(self) -> None:
        base = torch.zeros(inference.EXPECTED_LATENT_SHAPE, dtype=torch.bfloat16)
        values = {name: base.clone() for name in inference.ARM_ORDER}
        result = inference.validate_arm_latents(values)
        self.assertTrue(result["z0_full_latent_byte_exact_b0"])
        values["Z0"].reshape(-1)[0] = 1
        with self.assertRaisesRegex(
            inference.FewShotMotionInferenceError, "Z0 differs bytewise"
        ):
            inference.validate_arm_latents(values)

    def test_receipt_labels_no_go_as_diagnostic_and_never_claims_science(self) -> None:
        for fragment in (
            '"scientific_claim": False',
            '"video_quality_claim": False',
            '"source_instruction_only_inference": True',
            '"heldout_oracle_arm_exists": False',
            '"heldout_oracle_used": False',
            '"diagnostic_only": prototype_bundle.representability_gate == "NO_GO"',
            '"no_go_diagnostic_override": args.allow_no_go_diagnostic',
            '"every_output_is_81_frames_25fps"',
        ):
            self.assertIn(fragment, self.source)

    def test_cli_hash_and_seed_validation_is_fail_closed(self) -> None:
        instruction = "Have the animal sit and turn its head."
        instruction_sha = hashlib.sha256(instruction.encode("utf-8")).hexdigest()
        with mock.patch.object(
            inference, "EXPECTED_INSTRUCTION_SHA256", instruction_sha
        ):
            args = inference.build_parser().parse_args(
                [
                    "--bernini-root", "/b",
                    "--veomni-root", "/v",
                    "--checkpoint", "/c",
                    "--checkpoint-content-manifest", "/m",
                    "--source-video", "/s.mp4",
                    "--instruction", instruction,
                    "--prototype-state", "/prototype.safetensors",
                    "--prototype-receipt", "/prototype.receipt.json",
                    "--expected-prototype-state-sha256", "4" * 64,
                    "--expected-prototype-receipt-sha256", "5" * 64,
                    "--output-dir", "/out/epmc",
                    "--method-source-revision", "a" * 40,
                    "--method-source-archive-sha256", "b" * 64,
                ]
            )
            inference.validate_cli(args)
            args.render_seed += 1
            with self.assertRaises(inference.FewShotMotionInferenceError):
                inference.validate_cli(args)


if __name__ == "__main__":
    unittest.main()
