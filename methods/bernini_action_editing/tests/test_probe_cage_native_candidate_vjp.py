from __future__ import annotations

import copy
import hashlib
import inspect
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(METHOD_ROOT))

try:
    import torch  # noqa: E402
except ImportError:
    torch = None

if torch is not None:
    import cage_candidate_action_energy_vjp as cage  # noqa: E402
    import mace_candidate_action_energy as mace  # noqa: E402
    import probe_cage_native_candidate_vjp as probe  # noqa: E402
    import score_pair_v5_t2v_energy_bank_v3 as frozen_runtime  # noqa: E402
else:  # pragma: no cover - dependency-light environments
    cage = None
    mace = None
    probe = None
    frozen_runtime = None


POPULATION_PATH = (
    METHOD_ROOT / "assets/pair_v5_native_rv2v4_core4_action_population_v1.json"
)
T2V_PATH = METHOD_ROOT / "assets/pair_v5_t2v_calibration_core4_bank_v2.json"


if torch is not None:

    def _pack_spatial(value: torch.Tensor) -> torch.Tensor:
        batch, channels, phases, height, width = value.shape
        return (
            value.reshape(
                batch,
                channels,
                phases,
                1,
                height // 2,
                2,
                width // 2,
                2,
            )
            .permute(0, 2, 4, 6, 3, 5, 7, 1)
            .reshape(batch, phases * (height // 2) * (width // 2), 64)
        )


    class _Config:
        patch_size = (1, 2, 2)


    class _Transformer(torch.nn.Module):
        dtype = torch.float32
        config = _Config()

        def __init__(self) -> None:
            super().__init__()
            self.patch_calls: list[dict[str, object]] = []

        def patch_vae_latent(self, value, *, source_id):
            self.patch_calls.append(
                {
                    "source_id": source_id,
                    "grad_enabled": torch.is_grad_enabled(),
                    "requires_grad": value.requires_grad,
                }
            )
            packed = _pack_spatial(value)
            padding = torch.zeros(
                1,
                int(packed.shape[1]),
                1536 - 64,
                dtype=packed.dtype,
                device=packed.device,
            )
            tokens = torch.cat((packed, padding), dim=2)
            count = int(tokens.shape[1])
            real = torch.arange(
                count * 64, dtype=torch.float64, device=value.device
            ).reshape(1, 1, count, 64)
            rotary = torch.complex(real, torch.zeros_like(real))
            return tokens, rotary


    class _Diffusion(torch.nn.Module):
        def __init__(self, transformer: _Transformer, *, detach_output: bool = False):
            super().__init__()
            self.transformer = transformer
            self.transformer_2 = None
            self.gain = torch.nn.Parameter(
                torch.tensor(0.31, dtype=torch.float32), requires_grad=False
            )
            self.detach_output = detach_output
            self.calls: list[dict[str, object]] = []

        def shared_step(self, **kwargs):
            condition_scalar = kwargs["cond_embeds"][0, 0, 0].float()
            timestep_scalar = kwargs["timesteps"][0].float()
            prediction = (
                kwargs["noisy_latents"][:, :, :64].float()
                * (self.gain + 0.00007 * timestep_scalar)
                + 0.071 * condition_scalar
                + 0.000003 * timestep_scalar.square()
            )
            if self.detach_output:
                prediction = prediction.detach()
            self.calls.append(
                {
                    "timestep": float(kwargs["timesteps"].item()),
                    "grad_enabled": torch.is_grad_enabled(),
                    "tokens_require_grad": kwargs["noisy_latents"].requires_grad,
                    "prediction_requires_grad": prediction.requires_grad,
                    "batch_vae_seqlen": tuple(kwargs["batch_vae_seqlen"]),
                    "batch_text_seqlen": tuple(kwargs["batch_text_seqlen"]),
                }
            )
            return prediction


    def _prompts() -> dict[str, str]:
        return {
            branch: f"sealed native Phase-A prompt for {branch}"
            for branch in mace.BRANCH_ORDER
        }


    def _condition(value: float) -> torch.Tensor:
        return torch.tensor(value, dtype=torch.float32).reshape(1, 1, 1).expand(
            1, 512, 4096
        )


    def _conditions() -> dict[str, torch.Tensor]:
        return {
            branch: _condition(0.2 + 0.13 * index)
            for index, branch in enumerate(mace.BRANCH_ORDER)
        }


    def _bridge(
        *, detach_output: bool = False, inference_conditions: bool = False
    ):
        transformer = _Transformer()
        diffusion = _Diffusion(transformer, detach_output=detach_output).eval()
        if inference_conditions:
            with torch.inference_mode():
                conditions = _conditions()
            self_check = list(conditions.values())
            if not all(torch.is_inference(value) for value in self_check):
                raise AssertionError("test fixture did not create inference tensors")
        else:
            conditions = _conditions()
        bridge = probe.NativeMultiSigmaFrozenT2VInputVJPBridge(
            diffusion,
            transformer,
            _prompts(),
            conditions,
            frozen_model_receipt_digest="a" * 64,
        )
        return bridge, transformer, diffusion


class _FakeDistributed:
    def __init__(self, *, mismatch: bool = False) -> None:
        self.mismatch = mismatch

    @staticmethod
    def get_world_size() -> int:
        return 4

    def all_gather_object(self, rows, value) -> None:
        for index in range(4):
            rows[index] = value
        if self.mismatch:
            rows[-1] = {"different": True}


@unittest.skipIf(torch is None, "torch is unavailable")
class NativeScheduleAndCore4ContractTests(unittest.TestCase):
    def test_phase_a_uses_exact_native_indices_sigmas_and_timesteps(self) -> None:
        coordinates = probe.native_probe_coordinates()
        self.assertEqual(
            tuple(item.schedule_index for item in coordinates), (20, 28, 33)
        )
        self.assertEqual(
            tuple(item.native_timestep for item in coordinates), (833, 682, 516)
        )
        self.assertEqual(
            tuple(item.sigma for item in coordinates),
            (
                0.8336109519004822,
                0.6825404167175293,
                0.5161304473876953,
            ),
        )
        for item in coordinates:
            receipt = item.receipt()
            self.assertEqual(receipt["schedule_index"], item.schedule_index)
            self.assertEqual(
                receipt["native_scheduler_timestep"], item.native_timestep
            )
            self.assertTrue(receipt["legacy_1000_sigma_timestep_rejected"])
            self.assertNotEqual(
                float(item.native_timestep), 1000.0 * float(item.sigma)
            )

    def test_real_sealed_core4_population_matches_action_plus_nine(self) -> None:
        population = json.loads(POPULATION_PATH.read_text(encoding="utf-8"))
        t2v = json.loads(T2V_PATH.read_text(encoding="utf-8"))
        candidate_ids = [
            row["candidate_id"]
            for group in population["groups"]
            for row in group["candidates"]
        ]
        self.assertEqual(len(candidate_ids), 8)
        cells: dict[str, int] = {}
        for candidate_id in candidate_ids:
            matched = probe.match_core4_native_candidate(
                population, t2v, candidate_id=candidate_id
            )
            self.assertEqual(
                list(matched["caption_by_branch"]), list(mace.BRANCH_ORDER)
            )
            self.assertEqual(
                matched["caption_by_branch"]["action"],
                matched["candidate"]["complete_caption"],
            )
            cell_id = matched["cell"]["calibration_group_id"]
            cells[cell_id] = cells.get(cell_id, 0) + 1
        self.assertEqual(len(cells), 4)
        self.assertEqual(set(cells.values()), {2})
        self.assertEqual(
            probe.file_sha256(T2V_PATH), probe.PINNED_T2V_CORE4_V2_SHA256
        )
        self.assertEqual(
            probe.file_sha256(POPULATION_PATH),
            probe.PINNED_NATIVE_CORE4_POPULATION_SHA256,
        )

    def test_public_tensor_boundary_forbids_proposal_media(self) -> None:
        receipt = probe.contract_receipt()
        self.assertFalse(receipt["pure_t2v_media_consumed"])
        self.assertFalse(
            receipt["proposal_media_target_condition_noise_or_donor_consumed"]
        )
        self.assertEqual(receipt["scan_calls"], 30)
        self.assertEqual(receipt["selected_input_vjp_replay_calls"], 2)
        self.assertTrue(
            receipt["native_clean_absent_value_hashes_never_compared_as_none"]
        )
        self.assertTrue(
            receipt["official_gaussian_recorded_raw_content_identity_required"]
        )
        for function in (
            probe.make_energy_coordinates,
            probe.compute_native_candidate_cotangent,
            probe.verify_authenticated_native_clean_tensor_identity,
            probe.save_cotangent_safetensors,
        ):
            self.assertFalse(
                set(inspect.signature(function).parameters)
                & probe.FORBIDDEN_PUBLIC_INPUT_NAMES
            )

        population = json.loads(POPULATION_PATH.read_text(encoding="utf-8"))
        t2v = json.loads(T2V_PATH.read_text(encoding="utf-8"))
        candidate_id = population["groups"][0]["candidates"][0]["candidate_id"]
        bad_t2v = copy.deepcopy(t2v)
        bad_t2v["semantic_input_closure"]["proposal_media_as_noise"] = True
        with self.assertRaisesRegex(
            probe.CAGENativeCandidateVJPProbeError,
            "semantic input closure differs",
        ):
            probe.match_core4_native_candidate(
                population, bad_t2v, candidate_id=candidate_id
            )
        bad_population = copy.deepcopy(population)
        bad_population["semantic_input_closure"]["t2v_proposal_media"] = True
        with self.assertRaisesRegex(
            probe.CAGENativeCandidateVJPProbeError,
            "semantic input closure differs",
        ):
            probe.match_core4_native_candidate(
                bad_population, t2v, candidate_id=candidate_id
            )


@unittest.skipIf(torch is None, "torch is unavailable")
class NativeFrozenInputVJPBridgeTests(unittest.TestCase):
    def test_world4_scan30_then_only_action_and_hardest_input_vjp(self) -> None:
        bridge, transformer, diffusion = _bridge(inference_conditions=True)
        self.assertTrue(
            all(
                not torch.is_inference(value)
                for value in bridge._conditions.values()
            )
        )
        clean = torch.linspace(
            -0.7, 0.8, 1 * 16 * 21 * 2 * 2, dtype=torch.float32
        ).reshape(1, 16, 21, 2, 2)
        gaussian = torch.linspace(
            0.9, -0.4, clean.numel(), dtype=torch.float32
        ).reshape_as(clean)
        result, execution = probe.compute_native_candidate_cotangent(
            clean,
            gaussian,
            _prompts(),
            bridge,
            config=cage.EnergyVJPConfig(
                target_margin=0.13,
                temperature=0.83,
            ),
        )

        self.assertEqual(execution["scan_shared_step_calls"], 30)
        self.assertEqual(execution["scan_patch_vae_latent_calls"], 3)
        self.assertEqual(execution["replay_shared_step_calls"], 2)
        self.assertEqual(execution["replay_patch_vae_latent_calls"], 2)
        self.assertEqual(
            execution["replay_branches"],
            [mace.ACTION_BRANCH, result.scan.selected_negative_branch],
        )
        self.assertTrue(execution["existing_no_grad_scorer_bypassed"])
        self.assertEqual(len(diffusion.calls), 32)
        self.assertEqual(len(transformer.patch_calls), 5)
        self.assertEqual(
            [row["timestep"] for row in diffusion.calls[:30]],
            [833.0] * 10 + [682.0] * 10 + [516.0] * 10,
        )
        self.assertTrue(
            all(not row["grad_enabled"] for row in diffusion.calls[:30])
        )
        self.assertTrue(
            all(row["grad_enabled"] for row in diffusion.calls[30:])
        )
        self.assertTrue(
            all(not row["tokens_require_grad"] for row in diffusion.calls[:30])
        )
        self.assertTrue(
            all(row["tokens_require_grad"] for row in diffusion.calls[30:])
        )
        self.assertTrue(result.finite)
        self.assertTrue(result.nonzero)
        self.assertGreater(result.gradient_norm, 0.0)
        result_receipt = probe.vjp_result_receipt(result)
        unsigned_result_receipt = dict(result_receipt)
        result_digest = unsigned_result_receipt.pop("digest")
        self.assertEqual(result_digest, probe.object_sha256(unsigned_result_receipt))
        self.assertEqual(
            result_receipt["candidate_clean_cotangent_identity"]["shape"],
            [1, 16, 21, 2, 2],
        )
        recomposed = (
            result.direct_flow_target_gradient
            + (1.0 - result.scan.selected_sigma)
            * (result.action_input_vjp + result.negative_input_vjp)
        )
        torch.testing.assert_close(result.gradient, recomposed, rtol=0, atol=0)
        self.assertEqual(
            [id(coordinate.epsilon) for coordinate in probe.make_energy_coordinates(gaussian)],
            [id(gaussian)] * 3,
        )

    def test_detached_legacy_style_replay_fails_closed(self) -> None:
        bridge, _, _ = _bridge(detach_output=True)
        clean = torch.zeros(1, 16, 21, 2, 2, dtype=torch.float32)
        gaussian = torch.ones_like(clean)
        with self.assertRaisesRegex(
            probe.CAGENativeCandidateVJPProbeError,
            "shared_step output graph/shape",
        ):
            probe.compute_native_candidate_cotangent(
                clean, gaussian, _prompts(), bridge
            )


@unittest.skipIf(torch is None, "torch is unavailable")
class DistributedFreezeAndArtifactTests(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("safetensors") is not None,
        "safetensors is unavailable",
    )
    def test_clean_identity_uses_authenticated_container_not_absent_hashes(self) -> None:
        from safetensors.torch import save_file

        value = torch.linspace(
            -0.5, 0.75, 1 * 16 * 21 * 2 * 2, dtype=torch.float32
        ).reshape(1, 16, 21, 2, 2).contiguous()

        def storage_identity(tensor):
            cpu = tensor.detach().to(device="cpu").contiguous().clone()
            raw = bytes(cpu.untyped_storage())
            metadata = {
                "shape": [int(item) for item in cpu.shape],
                "dtype": str(cpu.dtype),
                "numel": int(cpu.numel()),
                "byte_count": len(raw),
            }
            return {
                **metadata,
                "raw_value_sha256": hashlib.sha256(raw).hexdigest(),
                "content_sha256": hashlib.sha256(
                    probe.canonical_json_bytes(metadata) + b"\x00" + raw
                ).hexdigest(),
            }

        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw).resolve() / "native-clean.safetensors"
            save_file(
                {"normalized_clean_latent": value},
                str(path),
                metadata={
                    "coordinate": "bernini_normalized_clean_vae_latent",
                    "frame_contract": "exact81_latent21",
                    "artifact_role": "native_sampler_proposal",
                    "source": "native_sampler_before_vae_decode",
                },
            )
            artifact = {
                "path": str(path),
                "sha256": probe.file_sha256(path),
                "tensor_key": "normalized_clean_latent",
                "shape": [1, 16, 21, 2, 2],
                "stored_dtype": "torch.float32",
                "sampler_return_dtype": "torch.float32",
                "coordinate": "bernini_normalized_clean_vae_latent",
                "artifact_role": "native_sampler_proposal",
                "origin": "native_sampler_before_vae_decode",
                "native_sampler_before_vae_decode": True,
                "source_video_vae_encode_before_any_decode": False,
                "mp4_decode_reencode_used": False,
                "roundtrip_byte_exact_fp32": True,
            }
            self.assertNotIn("raw_value_sha256", artifact)
            self.assertNotIn("content_sha256", artifact)

            def verify(candidate_value=value, candidate_artifact=artifact):
                with mock.patch.object(
                    frozen_runtime,
                    "native_tensor_value_identity",
                    side_effect=storage_identity,
                ):
                    return probe.verify_authenticated_native_clean_tensor_identity(
                        candidate_value,
                        candidate_artifact,
                        label="normalized clean latent",
                    )

            identity = verify()
            expected = storage_identity(value)
            self.assertEqual(
                identity["raw_value_sha256"], expected["raw_value_sha256"]
            )
            self.assertEqual(
                identity["content_sha256"], expected["content_sha256"]
            )
            self.assertFalse(identity["recorded_value_hashes_present"])
            self.assertFalse(identity["native_receipt_value_hashes_synthesized"])
            self.assertTrue(
                identity[
                    "observed_value_hashes_recomputed_after_authenticated_reopen"
                ]
            )
            self.assertEqual(
                identity["authenticated_container_sha256"], artifact["sha256"]
            )
            unsigned = dict(identity)
            binding_digest = unsigned.pop("binding_digest")
            self.assertEqual(binding_digest, probe.object_sha256(unsigned))
            receipt = probe._make_receipt(
                args=mock.Mock(
                    candidate_id="native-clean-test",
                    expected_checkpoint_tree_sha256="1" * 64,
                ),
                row={
                    "candidate": {
                        "candidate_id": "native-clean-test",
                        "seed": 17,
                        "source_video_sha256": "2" * 64,
                        "complete_caption_sha256": "3" * 64,
                    },
                    "group_id": "sp4-a",
                    "ordinal": 0,
                    "cell": {
                        "analysis_split": "fit",
                        "action_family_id": "test-action",
                        "calibration_group_id": "test-cell",
                    },
                    "native_artifacts": {
                        "native_receipt_digest": "4" * 64,
                        "predecode_clean_latent": artifact,
                        "official_initial_gaussian": {"sha256": "5" * 64},
                    },
                    "population_spec_path": "/sealed/population.json",
                    "population_spec_sha256": "6" * 64,
                    "t2v_spec_path": "/sealed/t2v.json",
                    "t2v_spec_sha256": "7" * 64,
                    "pair_receipt_path": "/sealed/pair.json",
                    "pair_receipt_file_sha256": "8" * 64,
                    "pair_receipt_digest": "9" * 64,
                    "native_receipt_path": "/sealed/native.json",
                    "native_receipt_file_sha256": "a" * 64,
                    "source_video_path": "/sealed/source.mp4",
                },
                prompt_binding={},
                checkpoint_identity={},
                checkpoint_binding={},
                freeze_before={},
                freeze_after={},
                clean_identity=identity,
                gaussian_identity={"raw_value_sha256": "b" * 64},
                result_receipt={},
                execution_receipt={},
                cotangent_artifact={},
                runtime_versions={},
                bernini_revision="c" * 40,
                veomni_revision="d" * 40,
            )
            sealed_clean = receipt["candidate_coordinate"][
                "normalized_clean_latent_identity"
            ]
            self.assertEqual(
                sealed_clean["raw_value_sha256"], identity["raw_value_sha256"]
            )
            self.assertEqual(
                sealed_clean["content_sha256"], identity["content_sha256"]
            )
            self.assertFalse(sealed_clean["native_receipt_value_hashes_synthesized"])

            wrong_role = dict(artifact)
            wrong_role["artifact_role"] = "source_video_condition"
            with self.assertRaisesRegex(
                probe.CAGENativeCandidateVJPProbeError,
                "artifact field artifact_role differs",
            ):
                verify(candidate_artifact=wrong_role)

            for field, wrong_value, message in (
                ("shape", [1, 16, 21, 4, 2], "metadata differs"),
                ("stored_dtype", "torch.float16", "metadata differs"),
                (
                    "roundtrip_byte_exact_fp32",
                    False,
                    "roundtrip_byte_exact_fp32 differs",
                ),
            ):
                bad_artifact = dict(artifact)
                bad_artifact[field] = wrong_value
                with self.subTest(field=field), self.assertRaisesRegex(
                    probe.CAGENativeCandidateVJPProbeError, message
                ):
                    verify(candidate_artifact=bad_artifact)

            invented_recorded_hash = dict(artifact)
            invented_recorded_hash["raw_value_sha256"] = "0" * 64
            with self.assertRaisesRegex(
                probe.CAGENativeCandidateVJPProbeError,
                "native artifact field closure differs",
            ):
                verify(candidate_artifact=invented_recorded_hash)

            wrong_key_path = Path(raw).resolve() / "wrong-key.safetensors"
            save_file(
                {"some_other_tensor": value},
                str(wrong_key_path),
                metadata={
                    "coordinate": "bernini_normalized_clean_vae_latent",
                    "frame_contract": "exact81_latent21",
                    "artifact_role": "native_sampler_proposal",
                    "source": "native_sampler_before_vae_decode",
                },
            )
            wrong_key_artifact = dict(artifact)
            wrong_key_artifact.update(
                path=str(wrong_key_path),
                sha256=probe.file_sha256(wrong_key_path),
            )
            with self.assertRaisesRegex(
                probe.CAGENativeCandidateVJPProbeError,
                "container key closure differs",
            ):
                verify(candidate_artifact=wrong_key_artifact)

            negative_zero_path = Path(raw).resolve() / "negative-zero.safetensors"
            negative_zero = torch.zeros_like(value)
            negative_zero.reshape(-1)[0] = -0.0
            save_file(
                {"normalized_clean_latent": negative_zero},
                str(negative_zero_path),
                metadata={
                    "coordinate": "bernini_normalized_clean_vae_latent",
                    "frame_contract": "exact81_latent21",
                    "artifact_role": "native_sampler_proposal",
                    "source": "native_sampler_before_vae_decode",
                },
            )
            negative_zero_artifact = dict(artifact)
            negative_zero_artifact.update(
                path=str(negative_zero_path),
                sha256=probe.file_sha256(negative_zero_path),
            )
            # torch.equal(+0.0, -0.0) is true.  Raw identity must still reject
            # this as not byte-exact round-trip evidence.
            self.assertTrue(torch.equal(torch.zeros_like(value), negative_zero))
            with self.assertRaisesRegex(
                probe.CAGENativeCandidateVJPProbeError,
                "actual tensor/container metadata differs",
            ):
                verify(
                    candidate_value=torch.zeros_like(value),
                    candidate_artifact=negative_zero_artifact,
                )

            with self.assertRaisesRegex(
                probe.CAGENativeCandidateVJPProbeError,
                "loaded value differs from authenticated container",
            ):
                probe.verify_authenticated_native_clean_tensor_identity(
                    value + 0.125, artifact, label="normalized clean latent"
                )

            save_file(
                {"normalized_clean_latent": value + 1.0},
                str(path),
                metadata={
                    "coordinate": "bernini_normalized_clean_vae_latent",
                    "frame_contract": "exact81_latent21",
                    "artifact_role": "native_sampler_proposal",
                    "source": "native_sampler_before_vae_decode",
                },
            )
            with self.assertRaisesRegex(
                probe.CAGENativeCandidateVJPProbeError,
                "container SHA-256 differs",
            ):
                verify()

    def test_main_keeps_gaussian_on_full_recorded_value_verifier(self) -> None:
        source = inspect.getsource(probe.main)
        self.assertIn(
            "clean_identity = verify_authenticated_native_clean_tensor_identity(",
            source,
        )
        self.assertIn(
            "gaussian_identity = frozen_runtime.verify_native_tensor_value_identity(",
            source,
        )

    def test_world4_contract_and_exact_consensus(self) -> None:
        contract = probe.world4_sp4_contract(
            {
                "WORLD_SIZE": "4",
                "RANK": "2",
                "LOCAL_RANK": "2",
                "LOCAL_WORLD_SIZE": "4",
            }
        )
        self.assertEqual(contract.world_size, 4)
        self.assertEqual(contract.ulysses_size, 4)
        self.assertEqual(contract.rank, 2)
        self.assertEqual(
            probe.require_sp4_object_consensus(
                {"digest": "a" * 64},
                label="toy",
                distributed_module=_FakeDistributed(),
            ),
            {"digest": "a" * 64},
        )
        with self.assertRaisesRegex(
            probe.CAGENativeCandidateVJPProbeError, "differs across SP4"
        ):
            probe.require_sp4_object_consensus(
                {"digest": "a" * 64},
                label="toy",
                distributed_module=_FakeDistributed(mismatch=True),
            )
        with self.assertRaisesRegex(
            probe.CAGENativeCandidateVJPProbeError, "WORLD4"
        ):
            probe.world4_sp4_contract(
                {
                    "WORLD_SIZE": "8",
                    "RANK": "0",
                    "LOCAL_RANK": "0",
                    "LOCAL_WORLD_SIZE": "8",
                }
            )

    def test_freeze_certificate_requires_no_parameter_grad(self) -> None:
        model = torch.nn.Linear(3, 2).requires_grad_(False).eval()
        base = {
            "base_frozen": True,
            "trainable_parameter_tensors": 0,
            "trainable_parameter_elements": 0,
            "lora_module_count": 0,
        }
        certificate = probe.model_freeze_runtime_certificate(
            model, base_certificate=base
        )
        self.assertEqual(certificate["parameter_grad_tensors"], 0)
        self.assertTrue(certificate["input_vjp_only"])
        next(model.parameters()).grad = torch.ones_like(next(model.parameters()))
        with self.assertRaisesRegex(
            probe.CAGENativeCandidateVJPProbeError, "acquired trainable"
        ):
            probe.model_freeze_runtime_certificate(
                model, base_certificate=base
            )

    @unittest.skipUnless(
        importlib.util.find_spec("safetensors") is not None,
        "safetensors is unavailable",
    )
    def test_cotangent_safetensors_is_exact_fp32_single_key(self) -> None:
        from safetensors import safe_open

        value = torch.linspace(
            -1.0, 1.0, 1 * 16 * 21 * 2 * 2, dtype=torch.float32
        ).reshape(1, 16, 21, 2, 2)

        # Some local legacy Torch builds cannot expose NumPy because their
        # NumPy ABI is newer than the wheel.  The production helper is tested
        # elsewhere; use the same byte/content definition without NumPy so
        # this test still exercises the real safetensors create/reopen path.
        def storage_identity(tensor):
            cpu = tensor.detach().to(device="cpu").contiguous().clone()
            raw = bytes(cpu.untyped_storage())
            metadata = {
                "shape": [int(item) for item in cpu.shape],
                "dtype": str(cpu.dtype),
                "numel": int(cpu.numel()),
                "byte_count": len(raw),
            }
            return {
                **metadata,
                "raw_value_sha256": hashlib.sha256(raw).hexdigest(),
                "content_sha256": hashlib.sha256(
                    probe.canonical_json_bytes(metadata) + b"\x00" + raw
                ).hexdigest(),
            }

        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw).resolve() / probe.COTANGENT_FILENAME
            with mock.patch.object(
                frozen_runtime,
                "native_tensor_value_identity",
                side_effect=storage_identity,
            ):
                artifact = probe.save_cotangent_safetensors(path, value)
            self.assertEqual(artifact["tensor_key"], probe.COTANGENT_KEY)
            self.assertFalse(artifact["proposal_media_consumed"])
            self.assertEqual(artifact["sha256"], probe.file_sha256(path))
            with safe_open(str(path), framework="pt", device="cpu") as opened:
                self.assertEqual(list(opened.keys()), [probe.COTANGENT_KEY])
                restored = opened.get_tensor(probe.COTANGENT_KEY)
            self.assertTrue(torch.equal(restored, value))
            self.assertEqual(
                artifact["raw_value_sha256"],
                storage_identity(value)["raw_value_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
