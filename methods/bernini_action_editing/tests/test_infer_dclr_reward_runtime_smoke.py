from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import struct
from types import SimpleNamespace
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

try:
    import torch  # noqa: E402
except ImportError:
    torch = None

if torch is not None:
    import infer_dclr_reward_runtime_smoke as smoke  # noqa: E402
else:  # pragma: no cover - dependency-light environments
    smoke = None


class SourceOnlyWrongSourceGuards(unittest.TestCase):
    def test_wrong_source_never_loads_full_parquet_row(self) -> None:
        source = (
            METHOD_ROOT / "infer_dclr_reward_runtime_smoke.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("dataset[args.wrong_source_row_index]", source)
        self.assertNotIn("wrong_full_row", source)
        self.assertIn("--wrong-source-clean-latent", source)
        self.assertIn("validate_source_condition_provenance", source)


@unittest.skipIf(torch is None, "torch is unavailable")
class FlowCoordinateTests(unittest.TestCase):
    def test_explicit_sigma_maps_to_one_mode_shared_timestep(self) -> None:
        point = smoke.flow_query_point(0.35)
        expected_sigma = struct.unpack("!f", struct.pack("!f", 0.35))[0]
        self.assertEqual(point.sigma, expected_sigma)
        self.assertEqual(point.timestep, 350.0)
        self.assertEqual(point.sigma_float32_bits_hex, struct.pack("!f", expected_sigma).hex())
        self.assertEqual(point.timestep_float32_bits_hex, struct.pack("!f", 350.0).hex())

        # There is intentionally no shift or mode argument: shift changes the
        # sampling density, not the explicit transformer query coordinate.
        self.assertNotIn("shift", point.as_dict())
        self.assertNotIn("mode", point.as_dict())

    def test_sigma_request_is_fail_closed_and_weights_are_shared(self) -> None:
        points, weights = smoke.validate_sigma_request([0.8, 0.35], None)
        self.assertEqual(len(points), 2)
        self.assertEqual(weights, (1.0, 1.0))

        points, weights = smoke.validate_sigma_request(
            [0.8, 0.35, 0.15], [1.0, 2.0, 1.0]
        )
        self.assertEqual(len(points), 3)
        self.assertEqual(weights, (1.0, 2.0, 1.0))

        invalid = (
            ([0.5], None),
            ([0.5, 0.5], None),
            ([0.0, 0.5], None),
            ([0.5, 1.0], None),
            ([0.5, float("nan")], None),
            ([0.5, 0.7], [1.0]),
            ([0.5, 0.7], [0.0, 0.0]),
            ([0.5, 0.7], [1.0, -1.0]),
        )
        for sigmas, weights in invalid:
            with self.subTest(sigmas=sigmas, weights=weights):
                with self.assertRaises(smoke.DCLRRuntimeSmokeError):
                    smoke.validate_sigma_request(sigmas, weights)


@unittest.skipIf(torch is None, "torch is unavailable")
class QueryGeometryTests(unittest.TestCase):
    class FakeTransformer:
        dtype = torch.float32

        def patch_vae_latent(self, hidden_states, source_id=None):
            tokens = smoke.pack_spatial_velocity(hidden_states)
            count = int(tokens.shape[1])
            positions = torch.arange(
                count * smoke.ROPE_COMPLEX_DIM,
                dtype=torch.float32,
                device=hidden_states.device,
            )
            real = positions.reshape(1, 1, count, smoke.ROPE_COMPLEX_DIM)
            imag = torch.full_like(real, float(source_id))
            rope = torch.complex(real.double(), imag.double())
            return tokens, rope

    def setUp(self) -> None:
        torch.manual_seed(7)
        shape = (1, 16, 21, 4, 4)
        self.correct = torch.randn(shape, dtype=torch.float32)
        self.wrong = torch.randn(shape, dtype=torch.float32) + 3.0
        self.student = torch.randn(shape, dtype=torch.float32)
        self.noise = torch.randn(shape, dtype=torch.float32)
        self.point = smoke.flow_query_point(0.4)

    def _bundle(self):
        return smoke.build_same_state_query_bundle(
            self.FakeTransformer(),
            correct_source_spatial=self.correct,
            wrong_source_spatial=self.wrong,
            student_clean_spatial=self.student,
            epsilon_spatial=self.noise,
            point=self.point,
        )

    def test_pack_spatial_velocity_uses_pt_ph_pw_c_order(self) -> None:
        spatial = torch.tensor(
            [
                [
                    [[[1.0, 2.0], [3.0, 4.0]]],
                    [[[10.0, 20.0], [30.0, 40.0]]],
                ]
            ]
        )
        packed = smoke.pack_spatial_velocity(spatial)
        self.assertEqual(tuple(packed.shape), (1, 1, 8))
        self.assertTrue(
            torch.equal(
                packed[0, 0],
                torch.tensor([1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0]),
            )
        )

    def test_t2v_is_direct_correct_mv2v_tail_and_wrong_changes_prefix_only(self) -> None:
        bundle = self._bundle()
        evidence = smoke.validate_query_bundle(bundle)
        count = bundle.target_tokens
        self.assertEqual(count, 21 * 2 * 2)
        self.assertEqual(tuple(bundle.t2v_noisy_latents.shape[:2]), (1, count))
        self.assertEqual(
            tuple(bundle.mv2v_correct_noisy_latents.shape[:2]), (1, 2 * count)
        )
        self.assertTrue(
            torch.equal(
                bundle.t2v_noisy_latents,
                bundle.mv2v_correct_noisy_latents[:, count:, :],
            )
        )
        self.assertEqual(
            bundle.t2v_noisy_latents.untyped_storage().data_ptr(),
            bundle.mv2v_correct_noisy_latents.untyped_storage().data_ptr(),
        )
        self.assertTrue(
            torch.equal(
                bundle.mv2v_wrong_noisy_latents[:, count:, :],
                bundle.mv2v_correct_noisy_latents[:, count:, :],
            )
        )
        self.assertFalse(
            torch.equal(
                bundle.mv2v_wrong_noisy_latents[:, :count, :],
                bundle.mv2v_correct_noisy_latents[:, :count, :],
            )
        )
        self.assertTrue(evidence["t2v_is_correct_mv2v_target_tail_view"])
        self.assertTrue(evidence["correct_wrong_full_rotary_exact_equal"])
        self.assertEqual(evidence["target_source_id"], 0)
        self.assertEqual(evidence["source_source_id"], 1)

        expected_noisy = (
            (1.0 - self.point.sigma) * self.student
            + self.point.sigma * self.noise
        )
        self.assertTrue(torch.equal(bundle.noisy_target_spatial, expected_noisy))
        expected_velocity = self.noise.float() - self.student.float()
        self.assertTrue(
            torch.equal(
                bundle.true_velocity_packed,
                smoke.pack_spatial_velocity(expected_velocity),
            )
        )

    def test_bundle_rejects_non_exact81_and_identical_wrong_source(self) -> None:
        with self.assertRaisesRegex(
            smoke.DCLRRuntimeSmokeError, "exact81"
        ):
            smoke.build_same_state_query_bundle(
                self.FakeTransformer(),
                correct_source_spatial=self.correct[:, :, :20],
                wrong_source_spatial=self.wrong[:, :, :20],
                student_clean_spatial=self.student[:, :, :20],
                epsilon_spatial=self.noise[:, :, :20],
                point=self.point,
            )
        with self.assertRaisesRegex(
            smoke.DCLRRuntimeSmokeError, "tensor-identical"
        ):
            smoke.build_same_state_query_bundle(
                self.FakeTransformer(),
                correct_source_spatial=self.correct,
                wrong_source_spatial=self.correct.clone(),
                student_clean_spatial=self.student,
                epsilon_spatial=self.noise,
                point=self.point,
            )

    def test_validator_rejects_copied_t2v_tail_and_changed_wrong_tail(self) -> None:
        bundle = self._bundle()
        copied = replace(
            bundle,
            t2v_noisy_latents=bundle.t2v_noisy_latents.clone(),
        )
        with self.assertRaisesRegex(smoke.DCLRRuntimeSmokeError, "direct"):
            smoke.validate_query_bundle(copied)

        changed_wrong = bundle.mv2v_wrong_noisy_latents.clone()
        changed_wrong[:, bundle.target_tokens :, :] += 1.0
        changed = replace(bundle, mv2v_wrong_noisy_latents=changed_wrong)
        with self.assertRaisesRegex(smoke.DCLRRuntimeSmokeError, "target tail"):
            smoke.validate_query_bundle(changed)

        changed_noise = replace(bundle, epsilon_spatial=bundle.epsilon_spatial + 1.0)
        with self.assertRaisesRegex(smoke.DCLRRuntimeSmokeError, "noisy target"):
            smoke.validate_query_bundle(changed_noise)

        changed_rope = bundle.mv2v_wrong_rotary_embs.clone()
        changed_rope[:, :, 0, :] += complex(0.0, 1.0)
        with self.assertRaisesRegex(smoke.DCLRRuntimeSmokeError, "full"):
            smoke.validate_query_bundle(
                replace(bundle, mv2v_wrong_rotary_embs=changed_rope)
            )

    def test_runtime_rejects_float_rope_that_cannot_match_pinned_bernini(self) -> None:
        with self.assertRaisesRegex(smoke.DCLRRuntimeSmokeError, "complex128"):
            smoke._require_rotary_tensor(
                torch.zeros(1, 1, 4, smoke.ROPE_COMPLEX_DIM),
                label="fake rope",
            )


@unittest.skipIf(torch is None, "torch is unavailable")
class DirectSharedStepTests(unittest.TestCase):
    class Decoder:
        def __init__(self):
            self.calls = []

        def shared_step(self, **kwargs):
            self.calls.append(kwargs)
            return kwargs["noisy_latents"] + 5.0

    class Renderer:
        def __init__(self):
            self.diff_dec = DirectSharedStepTests.Decoder()

        def forward(self, *args, **kwargs):  # pragma: no cover - must never run
            raise AssertionError("renderer.forward scalar loss is forbidden")

    def test_calls_direct_shared_step_once_and_selects_target_tail(self) -> None:
        renderer = self.Renderer()
        latents = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4)
        rope = torch.zeros(
            1, 1, 6, smoke.ROPE_COMPLEX_DIM, dtype=torch.complex128
        )
        condition = smoke.TextCondition(
            text_lens=[3],
            text_embs=torch.zeros(1, 3, 4),
            prompt_sha256="a" * 64,
            instruction_sha256="b" * 64,
            task_name="mv2v",
        )
        timestep = torch.tensor([400.0], dtype=torch.float32)
        result = smoke.shared_step_target_prediction(
            renderer,
            model_id="transformer_1",
            noisy_latents=latents,
            rotary_embs=rope,
            target_tokens=3,
            target_mask=torch.tensor(
                [False, False, False, True, True, True], dtype=torch.bool
            ),
            timestep=timestep,
            condition=condition,
        )
        self.assertEqual(len(renderer.diff_dec.calls), 1)
        self.assertTrue(torch.equal(result, latents[:, 3:, :] + 5.0))
        call = renderer.diff_dec.calls[0]
        self.assertIs(call["timesteps"], timestep)
        self.assertEqual(call["batch_vae_seqlen"], [6])
        self.assertEqual(call["batch_text_seqlen"], [3])

    def test_positive_text_uses_actual_input_lens_then_padded_512_metadata(self) -> None:
        class TextRenderer:
            max_sequence_length = 512

            def get_t5_text_embeddings(self, ids, attention, lens):
                self.seen = (ids.clone(), attention.clone(), lens.clone())
                return [512], torch.zeros(1, 512, 8, dtype=torch.float32)

        def encode(messages, tokenizer, *, task_name, **kwargs):
            self.assertEqual(task_name, "t2v")
            self.assertEqual(messages[1]["text"], "make the dog sit")
            return {
                "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
                "attention_mask": torch.ones(3, dtype=torch.long),
                "t5_input_lens": torch.tensor([3], dtype=torch.long),
            }

        sample = {
            "inputs": json.dumps(
                [
                    {"type": "video", "has_loss": 0},
                    {"type": "text", "has_loss": 0, "text": "make the dog sit"},
                    {"type": "video_gen", "has_loss": 1},
                ]
            )
        }
        renderer = TextRenderer()
        condition = smoke._tokenize_positive_condition(
            renderer=renderer,
            tokenizer=object(),
            encode_renderer_messages=encode,
            sample=sample,
            task_name="t2v",
            device=torch.device("cpu"),
        )
        self.assertEqual(condition.text_lens, [512])
        self.assertEqual(tuple(condition.text_embs.shape), (1, 512, 8))
        self.assertEqual(tuple(renderer.seen[0].shape), (1, 3))
        self.assertEqual(tuple(renderer.seen[2].shape), (1, 1))

        renderer.max_sequence_length = 256
        with self.assertRaisesRegex(smoke.DCLRRuntimeSmokeError, "max=512"):
            smoke._tokenize_positive_condition(
                renderer=renderer,
                tokenizer=object(),
                encode_renderer_messages=encode,
                sample=sample,
                task_name="t2v",
                device=torch.device("cpu"),
            )


@unittest.skipIf(torch is None, "torch is unavailable")
class WrongSourceManifestTests(unittest.TestCase):
    def manifest(self):
        return {
            "schema_version": smoke.WRONG_SOURCE_MATCH_SCHEMA,
            "candidate_iid": "candidate",
            "candidate_source_video_sha256": "c" * 64,
            "wrong_source_iid": "donor",
            "wrong_source_video_sha256": "d" * 64,
            "criteria": {name: True for name in smoke.MATCH_CRITERIA},
            "declared_use": "reward_calibration",
            "reviewer": "manual-reviewer",
        }

    def test_accepts_exact_manually_reviewed_matching_contract(self) -> None:
        result = smoke.validate_wrong_source_match_manifest(
            self.manifest(),
            candidate_iid="candidate",
            candidate_source_video_sha256="c" * 64,
            wrong_source_iid="donor",
            wrong_source_video_sha256="d" * 64,
        )
        self.assertEqual(result["candidate_iid"], "candidate")
        self.assertEqual(result["wrong_source_iid"], "donor")
        self.assertTrue(result["source_reward_calibration_authorized"])
        self.assertRegex(result["manifest_digest"], r"^[0-9a-f]{64}$")

    def test_rejects_missing_false_extra_or_same_identity(self) -> None:
        missing = self.manifest()
        missing["criteria"].pop("same_camera_class")
        false = self.manifest()
        false["criteria"]["manual_reviewed"] = False
        extra = self.manifest()
        extra["criteria"]["same_background"] = True
        for value in (missing, false, extra):
            with self.subTest(value=value):
                with self.assertRaises(smoke.DCLRRuntimeSmokeError):
                    smoke.validate_wrong_source_match_manifest(
                        value,
                        candidate_iid="candidate",
                        candidate_source_video_sha256="c" * 64,
                        wrong_source_iid="donor",
                        wrong_source_video_sha256="d" * 64,
                    )
        with self.assertRaisesRegex(smoke.DCLRRuntimeSmokeError, "distinct"):
            smoke.validate_wrong_source_match_manifest(
                self.manifest(),
                candidate_iid="same",
                candidate_source_video_sha256="c" * 64,
                wrong_source_iid="same",
                wrong_source_video_sha256="d" * 64,
            )

    def test_confounded_camera_donor_is_runtime_only_not_reward_evidence(self) -> None:
        value = self.manifest()
        value["declared_use"] = "runtime_plumbing_only"
        value["criteria"]["same_camera_class"] = False
        value["criteria"]["same_composition_class"] = False
        result = smoke.validate_wrong_source_match_manifest(
            value,
            candidate_iid="candidate",
            candidate_source_video_sha256="c" * 64,
            wrong_source_iid="donor",
            wrong_source_video_sha256="d" * 64,
        )
        self.assertFalse(result["scientific_eligibility"])
        self.assertFalse(result["source_reward_calibration_authorized"])

        value["declared_use"] = "reward_calibration"
        with self.assertRaisesRegex(smoke.DCLRRuntimeSmokeError, "every"):
            smoke.validate_wrong_source_match_manifest(
                value,
                candidate_iid="candidate",
                candidate_source_video_sha256="c" * 64,
                wrong_source_iid="donor",
                wrong_source_video_sha256="d" * 64,
            )


@unittest.skipIf(torch is None, "torch is unavailable")
class CleanLatentArtifactTests(unittest.TestCase):
    def _write(self, path: Path, *, role: str) -> str:
        from safetensors.torch import save_file

        origin = (
            "native_sampler_before_vae_decode"
            if role == "native_sampler_proposal"
            else "source_video_vae_encode_before_any_decode"
        )
        save_file(
            {"normalized_clean_latent": torch.zeros(1, 16, 21, 4, 4)},
            str(path),
            metadata={
                "coordinate": "bernini_normalized_clean_vae_latent",
                "frame_contract": "exact81_latent21",
                "artifact_role": role,
                "source": origin,
            },
        )
        return smoke._file_sha256(path)

    def test_loads_only_content_bound_role_typed_predecode_fp32(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "proposal.safetensors"
            digest = self._write(path, role="native_sampler_proposal")
            tensor, receipt = smoke.load_normalized_clean_latent_artifact(
                path,
                expected_sha256=digest,
                expected_role="native_sampler_proposal",
            )
            self.assertEqual(tuple(tensor.shape), (1, 16, 21, 4, 4))
            self.assertEqual(receipt["file_sha256"], digest)
            self.assertFalse(receipt["mp4_decode_reencode_used"])
            with self.assertRaisesRegex(smoke.DCLRRuntimeSmokeError, "metadata"):
                smoke.load_normalized_clean_latent_artifact(
                    path,
                    expected_sha256=digest,
                    expected_role="source_video_condition",
                )
            with self.assertRaisesRegex(smoke.DCLRRuntimeSmokeError, "SHA"):
                smoke.load_normalized_clean_latent_artifact(
                    path,
                    expected_sha256="0" * 64,
                    expected_role="native_sampler_proposal",
                )

    def test_native_receipts_bind_candidate_and_source_predecode_latents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proposal_path = root / "r2v.safetensors"
            source_path = root / "source.safetensors"
            proposal_sha = self._write(
                proposal_path, role="native_sampler_proposal"
            )
            source_sha = self._write(
                source_path, role="source_video_condition"
            )
            video_sha = "4" * 64
            prompt_sha = "5" * 64
            tree_sha = smoke.legacy.CHECKPOINT_TREE_SHA256
            checkpoint = {
                "tree_sha256": tree_sha,
                "content": {
                    "every_file_sha256_verified": True,
                    "verified_entries_digest": "6" * 64,
                },
            }

            def base(arms):
                return {
                    "schema_version": "bernini-native-identity-generation-canary-v1",
                    "experimental_canary": True,
                    "scientific_claim_authorized": False,
                    "arms": arms,
                    "input": {
                        "source_video_sha256": video_sha,
                        "action_prompt_utf8_sha256": prompt_sha,
                        "target_video": False,
                    },
                    "checkpoint": checkpoint,
                    "outputs": {},
                    "source_condition_artifact": None,
                }

            candidate = base(["r2v"])
            candidate["outputs"]["r2v"] = {
                "normalized_clean_latent": {
                    "path": str(proposal_path),
                    "sha256": proposal_sha,
                    "artifact_role": "native_sampler_proposal",
                    "native_sampler_before_vae_decode": True,
                    "mp4_decode_reencode_used": False,
                }
            }
            source = base(["rv2v"])
            source["source_condition_artifact"] = {
                "path": str(source_path),
                "sha256": source_sha,
                "artifact_role": "source_video_condition",
                "source_video_vae_encode_before_any_decode": True,
                "mp4_decode_reencode_used": False,
            }
            source_only = {
                "schema_version": smoke.SOURCE_ONLY_VAE_RECEIPT_SCHEMA,
                "source_only": True,
                "scientific_claim_authorized": False,
                "input": {
                    "source_iid": "wrong-source-iid",
                    "source_video_sha256": video_sha,
                },
                "access_audit": {
                    "source_columns_accessed": [
                        "iid",
                        "source_video",
                        "source_video_sha256",
                    ],
                    "target_columns_accessed": [],
                    "target_media_accessed": False,
                    "paired_target_accessed": False,
                },
                "checkpoint": checkpoint,
                "source_condition_artifact": {
                    "path": str(source_path),
                    "sha256": source_sha,
                    "tensor_key": "normalized_clean_latent",
                    "shape": [1, 16, 21, 4, 4],
                    "stored_dtype": "torch.float32",
                    "coordinate": "bernini_normalized_clean_vae_latent",
                    "frame_contract": "exact81_latent21",
                    "artifact_role": "source_video_condition",
                    "source_video_vae_encode_before_any_decode": True,
                    "mp4_decode_reencode_used": False,
                },
            }

            def write_receipt(name, value):
                path = root / name
                value = dict(value)
                value["receipt_digest"] = smoke.legacy.object_sha256(value)
                path.write_text(json.dumps(value), encoding="utf-8")
                return path, smoke._file_sha256(path)

            candidate_receipt, candidate_receipt_sha = write_receipt(
                "candidate.json", candidate
            )
            source_receipt, source_receipt_sha = write_receipt(
                "source.json", source
            )
            source_only_receipt, source_only_receipt_sha = write_receipt(
                "source-only.json", source_only
            )
            leaky_source_only = json.loads(json.dumps(source_only))
            leaky_source_only["access_audit"]["target_columns_accessed"] = [
                "target_video"
            ]
            leaky_receipt, leaky_receipt_sha = write_receipt(
                "source-only-leaky.json", leaky_source_only
            )
            result = smoke.validate_native_rollout_provenance(
                candidate_receipt_path=candidate_receipt,
                expected_candidate_receipt_sha256=candidate_receipt_sha,
                source_receipt_path=source_receipt,
                expected_source_receipt_sha256=source_receipt_sha,
                candidate_arm="r2v",
                candidate_artifact_path=proposal_path,
                candidate_artifact_sha256=proposal_sha,
                source_artifact_path=source_path,
                source_artifact_sha256=source_sha,
                expected_source_video_sha256=video_sha,
                expected_action_prompt_sha256=prompt_sha,
                expected_checkpoint_tree_sha256=tree_sha,
            )
            self.assertEqual(result["candidate_arm"], "r2v")
            self.assertFalse(result["paired_target_accessed"])
            self.assertFalse(result["mp4_decode_reencode_used"])

            source_only = smoke.validate_source_condition_provenance(
                source_receipt_path=source_only_receipt,
                expected_source_receipt_sha256=source_only_receipt_sha,
                source_iid="wrong-source-iid",
                source_artifact_path=source_path,
                source_artifact_sha256=source_sha,
                expected_source_video_sha256=video_sha,
                expected_checkpoint_tree_sha256=tree_sha,
            )
            self.assertEqual(source_only["source_iid"], "wrong-source-iid")
            self.assertEqual(source_only["source_video_sha256"], video_sha)
            self.assertEqual(
                source_only["checkpoint_content_identity"], checkpoint["content"]
            )
            self.assertFalse(source_only["paired_target_accessed"])
            self.assertEqual(source_only["target_columns_accessed"], [])
            self.assertFalse(source_only["target_media_accessed"])
            with self.assertRaisesRegex(
                smoke.DCLRRuntimeSmokeError, "schema/access closure"
            ):
                smoke.validate_source_condition_provenance(
                    source_receipt_path=leaky_receipt,
                    expected_source_receipt_sha256=leaky_receipt_sha,
                    source_iid="wrong-source-iid",
                    source_artifact_path=source_path,
                    source_artifact_sha256=source_sha,
                    expected_source_video_sha256=video_sha,
                    expected_checkpoint_tree_sha256=tree_sha,
                )


@unittest.skipIf(torch is None, "torch is unavailable")
class ReceiptAndCliTests(unittest.TestCase):
    def local_evidence(self):
        return {
            "forward_implementation": smoke.FORWARD_IMPLEMENTATION,
            "branch_order": list(smoke.BRANCH_ORDER),
            "num_frames": 81,
            "latent_phases": 21,
            "sigma_records": [{"sigma": 0.8}, {"sigma": 0.35}],
            "forwards_per_rank": 8,
            "adapter_state": "absent_frozen_base",
            "geometry": {"verified": True},
            "reward_reduction": "none",
            "candidate": {
                "message_template_columns_loaded": [
                    "iid",
                    "inputs",
                    "source_video_sha256",
                ],
                "proposal_origin": "native_rollout_predecode_latent",
                "proposal_artifact": {"file_sha256": "a" * 64},
                "native_provenance": {"candidate_arm": "r2v"},
                "paired_target_accessed": False,
                "positive_control_only": False,
            },
            "correct_source_artifact": {"file_sha256": "b" * 64},
            "wrong_source": {
                "paired_target_accessed": False,
                "source_artifact": {"file_sha256": "c" * 64},
                "source_provenance": {
                    "target_columns_accessed": [],
                    "target_media_accessed": False,
                    "paired_target_accessed": False,
                },
                "match_manifest": {
                    "source_reward_calibration_authorized": False
                }
            },
        }

    def rank_records(self, evidence):
        digest = smoke._object_sha256(evidence)
        return [
            {
                "rank": rank,
                "world_size": 4,
                "ulysses_size": 4,
                "local_evidence_digest": digest,
            }
            for rank in range(4)
        ]

    def test_sp4_receipt_requires_identical_all_gather_not_reduction(self) -> None:
        evidence = self.local_evidence()
        receipt = smoke.assemble_sp4_receipt(
            evidence, self.rank_records(evidence)
        )
        self.assertEqual(receipt["distributed"]["reward_reduction"], "none")
        self.assertTrue(receipt["distributed"]["all_gather_evidence_only"])
        self.assertFalse(receipt["scientific_claim_authorized"])
        self.assertFalse(receipt["paired_target_positive_control"])
        self.assertFalse(receipt["wrong_source_paired_target_accessed"])
        self.assertFalse(receipt["training_pair_authorized"])
        self.assertFalse(receipt["source_reward_calibration_authorized"])

        with self.assertRaisesRegex(smoke.DCLRRuntimeSmokeError, "four"):
            smoke.assemble_sp4_receipt(
                evidence, self.rank_records(evidence)[:3]
            )
        divergent = self.rank_records(evidence)
        divergent[3] = dict(divergent[3])
        divergent[3]["local_evidence_digest"] = "f" * 64
        with self.assertRaisesRegex(smoke.DCLRRuntimeSmokeError, "identical"):
            smoke.assemble_sp4_receipt(evidence, divergent)

        reduced = dict(evidence)
        reduced["reward_reduction"] = "mean"
        with self.assertRaisesRegex(smoke.DCLRRuntimeSmokeError, "reduction"):
            smoke.assemble_sp4_receipt(
                reduced, self.rank_records(reduced)
            )

    def valid_args(self):
        action = "make the dog pick up and hold the bone"
        hard = "make the actor turn left"
        return SimpleNamespace(
            num_frames=81,
            sigmas=[0.8, 0.35],
            sigma_weights=[1.0, 1.0],
            noise_seed=7,
            candidate_row_index=1,
            wrong_source_row_index=2,
            expected_candidate_iid="candidate",
            proposal_source_iid="source-candidate",
            expected_wrong_source_iid="donor",
            wrong_source_clean_latent="/tmp/wrong-source.safetensors",
            expected_wrong_source_clean_latent_sha256="4" * 64,
            wrong_source_provenance_receipt="/tmp/wrong-source-receipt.json",
            expected_wrong_source_provenance_receipt_sha256="5" * 64,
            expected_wrong_source_video_sha256="6" * 64,
            checkpoint_content_manifest="/tmp/checkpoint.sha256",
            action_instruction=action,
            hard_negative_instruction=hard,
            expected_hard_negative_instruction_sha256=hashlib.sha256(
                hard.encode("utf-8")
            ).hexdigest(),
            expected_action_instruction_sha256=hashlib.sha256(
                action.encode("utf-8")
            ).hexdigest(),
            expected_wrong_source_match_sha256="b" * 64,
            expected_checkpoint_tree_sha256=smoke.legacy.CHECKPOINT_TREE_SHA256,
            method_source_archive_sha256="c" * 64,
            expected_bernini_commit=smoke.legacy.BERNINI_OFFICIAL_COMMIT,
            expected_veomni_commit=smoke.legacy.VEOMNI_TESTED_COMMIT,
            method_source_revision="d" * 40,
            candidate_clean_latent="/tmp/proposal.safetensors",
            expected_candidate_clean_latent_sha256="e" * 64,
            correct_source_clean_latent="/tmp/source.safetensors",
            expected_correct_source_clean_latent_sha256="f" * 64,
            positive_control_paired_target=False,
            candidate_arm="r2v",
            candidate_provenance_receipt="/tmp/candidate-receipt.json",
            expected_candidate_provenance_receipt_sha256="1" * 64,
            source_provenance_receipt="/tmp/source-receipt.json",
            expected_source_provenance_receipt_sha256="2" * 64,
            expected_proposal_source_video_sha256="3" * 64,
        )

    def test_cli_rejects_hash_sigma_and_row_contract_drift(self) -> None:
        points, weights = smoke.validate_cli(self.valid_args())
        self.assertEqual(len(points), 2)
        self.assertEqual(weights, (1.0, 1.0))

        mutations = (
            ("sigmas", [0.8]),
            ("sigma_weights", [1.0]),
            ("noise_seed", -1),
            ("wrong_source_row_index", 1),
            ("expected_wrong_source_iid", "source-candidate"),
            ("expected_hard_negative_instruction_sha256", "0" * 64),
            ("expected_action_instruction_sha256", "not-a-hash"),
            ("wrong_source_clean_latent", "relative.safetensors"),
            ("expected_wrong_source_video_sha256", "not-a-hash"),
            ("method_source_revision", "short"),
        )
        for name, value in mutations:
            with self.subTest(name=name):
                args = self.valid_args()
                setattr(args, name, value)
                with self.assertRaises(smoke.DCLRRuntimeSmokeError):
                    smoke.validate_cli(args)

    def test_paired_target_requires_explicit_positive_control_and_no_artifacts(self) -> None:
        args = self.valid_args()
        args.candidate_clean_latent = None
        args.expected_candidate_clean_latent_sha256 = None
        args.correct_source_clean_latent = None
        args.expected_correct_source_clean_latent_sha256 = None
        args.candidate_arm = None
        args.candidate_provenance_receipt = None
        args.expected_candidate_provenance_receipt_sha256 = None
        args.source_provenance_receipt = None
        args.expected_source_provenance_receipt_sha256 = None
        args.expected_proposal_source_video_sha256 = None
        args.proposal_source_iid = args.expected_candidate_iid
        args.positive_control_paired_target = True
        smoke.validate_cli(args)

        args.candidate_clean_latent = "/tmp/leak.safetensors"
        with self.assertRaisesRegex(smoke.DCLRRuntimeSmokeError, "cannot accept"):
            smoke.validate_cli(args)

    def test_paired_target_never_authorizes_source_calibration(self) -> None:
        evidence = self.local_evidence()
        evidence["candidate"] = dict(evidence["candidate"])
        evidence["candidate"].update(
            {
                "proposal_origin": "paired_target_positive_control",
                "proposal_artifact": None,
                "native_provenance": None,
                "paired_target_accessed": True,
                "positive_control_only": True,
            }
        )
        evidence["correct_source_artifact"] = None
        evidence["wrong_source"] = dict(evidence["wrong_source"])
        evidence["wrong_source"]["match_manifest"] = {
            "source_reward_calibration_authorized": True
        }
        receipt = smoke.assemble_sp4_receipt(
            evidence, self.rank_records(evidence)
        )
        self.assertTrue(receipt["paired_target_positive_control"])
        self.assertFalse(receipt["wrong_source_paired_target_accessed"])
        self.assertFalse(receipt["source_reward_calibration_authorized"])

    def test_receipt_rejects_wrong_source_target_access(self) -> None:
        for key, value in (
            ("target_columns_accessed", ["target_video"]),
            ("target_media_accessed", True),
            ("paired_target_accessed", True),
        ):
            with self.subTest(key=key):
                evidence = self.local_evidence()
                evidence["wrong_source"] = dict(evidence["wrong_source"])
                provenance = dict(
                    evidence["wrong_source"]["source_provenance"]
                )
                provenance[key] = value
                evidence["wrong_source"]["source_provenance"] = provenance
                with self.assertRaisesRegex(
                    smoke.DCLRRuntimeSmokeError,
                    "wrong-source eligibility evidence is absent",
                ):
                    smoke.assemble_sp4_receipt(
                        evidence, self.rank_records(evidence)
                    )


if __name__ == "__main__":
    unittest.main()
