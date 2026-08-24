from __future__ import annotations

from pathlib import Path
import hashlib
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


METHOD_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = METHOD_ROOT / "audit_pair_v7_phase_a_geometry.py"
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

try:
    import torch

    import audit_pair_v7_phase_a_geometry as phase_a
    import pair_v7_dual_coordinate_nullspace_transport as v7

    _TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    phase_a = None  # type: ignore[assignment]
    v7 = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


def _sha(character: str) -> str:
    return character * 64


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


class PhaseAStaticClosureTests(unittest.TestCase):
    def test_runtime_has_real_vjps_and_no_update_primitive(self) -> None:
        source = RUNTIME_PATH.read_text(encoding="utf-8")
        self.assertIn("cagd.collect_same_state_predictions(", source)
        self.assertIn("cagd.build_bounded_teacher(", source)
        self.assertNotIn("cagd.build_distill_objective(", source)
        self.assertIn("cagd.replay_student_vjp(", source)
        self.assertNotIn("GuidanceEligibility", source)
        self.assertNotIn("run_same_state_cell", source)
        self.assertNotIn("authorize_action_arm_with_v3_evidence", source)
        self.assertIn("_build_deployment_v2v_source_pack(", source)
        self.assertIn("runtime.replay_post_apg(", source)
        self.assertNotIn("spatial.requires_grad_(True)", source)
        self.assertIn("raw replay is detached from Action-LoRA", source)
        self.assertIn(
            "world_rank0_cpu_union_project_and_audit_action_families(", source
        )
        self.assertIn('"solver_device": "cpu"', source)
        self.assertIn('"replicated_gpu_solver_used": False', source)
        self.assertIn("_assert_world_receipt_field_consensus(", source)
        self.assertIn("vendor.normalized_guidance(", source)
        self.assertNotIn("native.build_native_rv2v_pack(", source)
        self.assertNotIn("runtime.replay_reference_branch(", source)
        self.assertNotIn("torch.optim", source)
        self.assertNotIn("optimizer.step(", source)
        self.assertNotIn("parameter.add_(", source)
        self.assertIn(
            "**deployment_infer.inference_renderer_config_overrides(checkpoint)",
            source,
        )
        self.assertNotIn("**legacy.renderer_config_overrides(checkpoint)", source)
        self.assertNotIn("train_pair_v6_scaid", source)
        self.assertNotIn("import pair_v6_scaid", source)

    def test_correct_source_is_in_fit_manifest_and_wrong_source_schema_is_absent(self) -> None:
        source = RUNTIME_PATH.read_text(encoding="utf-8")
        self.assertNotIn("source_by_role", source)
        self.assertNotIn("wrong_source_video_path", source)
        self.assertNotIn("SOURCE_BINDING_SCHEMA", source)
        self.assertIn("pre.action_manifest.events[contract.arm_index]", source)
        self.assertIn('"wrong_source_fields_present": False', source)

    def test_launcher_acknowledgements_are_mandatory_in_preflight(self) -> None:
        source = RUNTIME_PATH.read_text(encoding="utf-8")
        self.assertIn("--ack-root-reviewed-phase-a-launch", source)
        self.assertIn("--ack-no-parameter-mutation-no-success-claim", source)
        self.assertIn("if args.ack_root_reviewed_phase_a_launch is not True", source)
        self.assertIn(
            "if args.ack_no_parameter_mutation_no_success_claim is not True", source
        )

    def test_source_tree_closure_binds_safe_tracked_symlink_bytes(self) -> None:
        source = RUNTIME_PATH.read_text(encoding="utf-8")
        for token in (
            '"bernini-pair-v7-source-tree-binding-v2"',
            '"tracked_relative_symlink"',
            '"link_text_sha256"',
            '"target_sha256"',
            '"git_blob_sha1"',
            '"tracked_relative_symlink_link_and_target_bytes_verified"',
        ):
            self.assertIn(token, source)

    def test_action_proposal_replay_uses_inference_tokenization_authority(self) -> None:
        source = RUNTIME_PATH.read_text(encoding="utf-8")
        self.assertIn("deployment_infer._tokenize_training_prompt(", source)
        self.assertNotIn("legacy._tokenize_training_prompt(", source)


@unittest.skipUnless(_TORCH_AVAILABLE, "torch and Bernini audit dependencies are required")
class PhaseAGeometryTests(unittest.TestCase):
    checkpoint_digest = _sha("1")
    parameter_state_digest = _sha("2")
    authorization_digest = _sha("3")
    parameter_name = "blocks.0.attn2.to_q.action_lora_b.weight"
    parameter_dim = 10

    def test_source_tree_closure_accepts_only_safe_tracked_relative_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "AGENTS.md").write_text("same tracked bytes\n", encoding="utf-8")
            link = root / "CLAUDE.md"
            link.symlink_to("AGENTS.md")
            subprocess.run(
                ["git", "init", "-q", str(root)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                ["git", "-C", str(root), "add", "AGENTS.md", "CLAUDE.md"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            closure = phase_a._source_tree_file_closure(root)
            self.assertEqual(closure["closure_mode"], "git_tracked_files")
            self.assertEqual(closure["file_count"], 2)
            self.assertEqual(closure["regular_file_count"], 1)
            self.assertEqual(closure["tracked_relative_symlink_count"], 1)
            self.assertTrue(
                closure[
                    "tracked_relative_symlink_link_and_target_bytes_verified"
                ]
            )

            link.unlink()
            link.symlink_to("../escape.md")
            with self.assertRaisesRegex(
                phase_a.PairV7PhaseAError, "symlink target differs"
            ):
                phase_a._source_tree_file_closure(root)

    def named(self, values):
        return {self.parameter_name: torch.tensor(values, dtype=torch.float32)}

    def metadata(self, family: str, candidate: str, gradient, ordinal: int):
        return {
            "candidate_id": candidate,
            "action_family": family,
            "event_digest": _sha(("4", "5")[ordinal]),
            "gradient_sha256": phase_a._named_gradient_sha256(gradient),
            "gradient_computation_receipt_digest": _sha(("6", "7")[ordinal]),
            "checkpoint_content_receipt_digest": self.checkpoint_digest,
            "parameter_state_sha256": self.parameter_state_digest,
        }

    def vector(self, *entries):
        value = [0.0] * self.parameter_dim
        for index, coefficient in entries:
            value[index] = float(coefficient)
        return value

    def default_probe_vectors(self):
        # Four K4 groups: DP2 source coordinates x two deployed identity
        # families.  Every group has rank three, while their union spans
        # e0..e7 and therefore has global effective rank eight.
        axes_by_group = (
            (0, 1, 2),
            (2, 3, 4),
            (4, 5, 6),
            (6, 7, 0),
        )
        vectors = []
        for axes in axes_by_group:
            vectors.extend(self.vector((axis, 1.0)) for axis in axes)
            vectors.append(self.vector(*((axis, 1.0) for axis in axes)))
        return vectors

    def probes(self, vectors=None):
        vectors = self.default_probe_vectors() if vectors is None else list(vectors)
        expected = (
            phase_a.DP_SIZE
            * len(v7.REQUIRED_IDENTITY_FAMILIES)
            * phase_a.IDENTITY_SKETCHES_PER_FAMILY
        )
        self.assertEqual(len(vectors), expected)
        rows = []
        vector_index = 0
        for arm_index, coordinate_digest in enumerate((_sha("c"), _sha("d"))):
            for family in v7.REQUIRED_IDENTITY_FAMILIES:
                for sketch_index in range(phase_a.IDENTITY_SKETCHES_PER_FAMILY):
                    rows.append(
                        v7.IdentityGradientProbe(
                            probe_id=f"arm{arm_index}-{family}-k{sketch_index}",
                            family=family,
                            gradient_by_parameter=self.named(vectors[vector_index]),
                            feature_sketch_sha256=_digest(
                                f"feature:{arm_index}:{family}:{sketch_index}"
                            ),
                            source_coordinate_receipt_digest=coordinate_digest,
                            gradient_computation_receipt_digest=_digest(
                                f"vjp:{arm_index}:{family}:{sketch_index}"
                            ),
                            checkpoint_content_receipt_digest=self.checkpoint_digest,
                            parameter_state_sha256=self.parameter_state_digest,
                        )
                    )
                    vector_index += 1
        return rows

    def union(self, left, right, vectors=None):
        gradients = {
            "sit": self.named(left),
            "turn": self.named(right),
        }
        metadata = {
            "sit": self.metadata("sit", "fit-sit", gradients["sit"], 0),
            "turn": self.metadata("turn", "fit-turn", gradients["turn"], 1),
        }
        return phase_a.union_project_and_audit_action_families(
            action_gradient_by_family=gradients,
            action_metadata_by_family=metadata,
            identity_probes=self.probes(vectors),
            checkpoint_content_receipt_digest=self.checkpoint_digest,
            parameter_state_sha256=self.parameter_state_digest,
            fit_only_geometry_authority_digest=self.authorization_digest,
        )

    def test_phase_a_is_only_the_preregistered_single_cell(self) -> None:
        for index in (38, 39):
            with self.assertRaisesRegex(phase_a.PairV7PhaseAError, "must be 33"):
                phase_a.phase_a_schedule_policy(index)
        first = phase_a.phase_a_schedule_policy(33)
        self.assertTrue(first["is_preregistered_first_phase_a_cell"])
        self.assertEqual(first["first_phase_a_schedule_index"], 33)
        self.assertTrue(first["single_fit_only_geometry_cell"])
        self.assertFalse(first["parameter_update_authorized"])

    def test_only_no_extra_carrier_is_exposed(self) -> None:
        receipt = phase_a.source_carrier_extension_contract("none")
        self.assertFalse(receipt["extra_carrier_tensor_accepted"])
        with self.assertRaisesRegex(phase_a.PairV7PhaseAError, "not validated"):
            phase_a.source_carrier_extension_contract("latent")

    def test_native_prompts_use_deployed_mv2v_prefix_and_negative_authority(self) -> None:
        raw = {
            branch: f"canonical {branch} edit instruction"
            for branch in phase_a.cagd.BRANCH_ORDER
        }

        def cleaner(value):
            return f"cleaned {value}"

        t2v, deployment_v2v, receipt = phase_a.build_task_prompt_registry(
            raw, prompt_cleaner=cleaner
        )
        self.assertEqual(set(t2v), set(phase_a.cagd.BRANCH_ORDER))
        self.assertEqual(set(deployment_v2v), set(phase_a.cagd.BRANCH_ORDER))
        for branch in phase_a.cagd.BRANCH_ORDER:
            self.assertEqual(
                deployment_v2v[branch],
                phase_a.deployment_infer.MV2V_SYSTEM_PROMPT + cleaner(raw[branch]),
            )
            self.assertFalse(
                deployment_v2v[branch].startswith(
                    phase_a.native_infer.TASK_SYSTEM_PROMPTS["vr2v"]
                )
            )
            self.assertNotIn("image0, image1, image2, and image3", deployment_v2v[branch])
        self.assertEqual(
            receipt["deployment_v2v_prompt_authority"],
            "infer_lora.build_training_prompt",
        )
        source = RUNTIME_PATH.read_text(encoding="utf-8")
        self.assertIn("deployment_infer.DEFAULT_NEGATIVE_PROMPT", source)
        self.assertNotIn("legacy.DEFAULT_NEGATIVE_PROMPT", source)

    def test_action_proposal_replay_calls_inference_tokenizer_authority(self) -> None:
        prompts = {
            branch: f"sealed T2V prompt {index}"
            for index, branch in enumerate(phase_a.cagd.BRANCH_ORDER)
        }
        prompt_to_token = {
            prompt: index + 1 for index, prompt in enumerate(prompts.values())
        }

        def tokenize(_tokenizer, prompt):
            token = prompt_to_token[prompt]
            ids = torch.full((1, 2), token, dtype=torch.int64)
            return ids, torch.ones_like(ids)

        class Renderer:
            @staticmethod
            def encode_prompt(ids, _mask):
                return ids.float().unsqueeze(-1).expand(-1, -1, 3).contiguous()

        with patch.object(
            phase_a.deployment_infer,
            "_tokenize_training_prompt",
            side_effect=tokenize,
        ) as tokenization_authority:
            with patch.object(phase_a, "_broadcast_sp") as broadcast:
                with patch.object(
                    phase_a.t2v_runtime, "PINNED_TEXT_TOKENS", 2
                ), patch.object(phase_a.t2v_runtime, "PINNED_TEXT_DIM", 3):
                    conditions = phase_a._encode_action_prompt_bank(
                        renderer=Renderer(),
                        tokenizer=object(),
                        prompt_by_branch=prompts,
                        device=torch.device("cpu"),
                        parallel=object(),
                    )

        self.assertEqual(list(conditions), list(phase_a.cagd.BRANCH_ORDER))
        self.assertEqual(tokenization_authority.call_count, len(prompts))
        self.assertEqual(broadcast.call_count, len(prompts))
        for branch, embedding in conditions.items():
            self.assertEqual(tuple(embedding.shape), (1, 2, 3))
            self.assertTrue(
                torch.equal(
                    embedding,
                    torch.full_like(embedding, float(prompt_to_token[prompts[branch]])),
                )
            )

    def test_feature_sketches_are_mask_free_deterministic_and_distinct(self) -> None:
        reference = torch.zeros((1, 16, 21, 2, 3), dtype=torch.float32)
        digests = []
        for family in v7.REQUIRED_IDENTITY_FAMILIES:
            for sketch_index in range(phase_a.IDENTITY_SKETCHES_PER_FAMILY):
                left, left_receipt = phase_a.build_mask_free_feature_sketch(
                    reference,
                    family=family,
                    sketch_index=sketch_index,
                    seed_digest=_sha("a"),
                )
                right, right_receipt = phase_a.build_mask_free_feature_sketch(
                    reference,
                    family=family,
                    sketch_index=sketch_index,
                    seed_digest=_sha("a"),
                )
                self.assertTrue(torch.equal(left, right))
                self.assertEqual(
                    left_receipt["tensor_sha256"], right_receipt["tensor_sha256"]
                )
                self.assertEqual(left_receipt["sketch_index"], sketch_index)
                self.assertFalse(left_receipt["spatial_region_mask_used"])
                self.assertFalse(left_receipt["flow_pose_track_or_trajectory_used"])
                self.assertAlmostEqual(
                    float(torch.linalg.vector_norm(left)), 1.0, places=5
                )
                digests.append(left_receipt["tensor_sha256"])
        self.assertEqual(
            len(set(digests)),
            len(v7.REQUIRED_IDENTITY_FAMILIES)
            * phase_a.IDENTITY_SKETCHES_PER_FAMILY,
        )
        for invalid in (-1, phase_a.IDENTITY_SKETCHES_PER_FAMILY, True):
            with self.assertRaisesRegex(
                phase_a.PairV7PhaseAError, "feature-sketch index differs"
            ):
                phase_a.build_mask_free_feature_sketch(
                    reference,
                    family="deploy_noop_identity",
                    sketch_index=invalid,
                    seed_digest=_sha("a"),
                )

    def test_deployed_v_only_term_registry(self) -> None:
        expected = {
            "deploy_noop_identity": (("post_apg", "V", "noop", 1.0),),
            "deploy_camera_delta": (
                ("post_apg", "V", "camera_only", 1.0),
                ("post_apg", "V", "noop", -1.0),
            ),
        }
        self.assertEqual(set(expected), set(v7.REQUIRED_IDENTITY_FAMILIES))
        for family, terms in expected.items():
            self.assertEqual(phase_a._identity_term_spec(family), terms)
            self.assertEqual({term[0] for term in terms}, {"post_apg"})
            self.assertEqual({term[1] for term in terms}, {"V"})
            self.assertNotIn("action", {term[2] for term in terms})

    def test_deployment_v_pack_patches_only_source_one_then_target_zero(self) -> None:
        class FakeTransformer:
            dtype = torch.float32

            def __init__(self):
                self.calls = []

            def patch_vae_latent(self, value, *, source_id):
                self.calls.append((float(source_id), value.detach().clone()))
                latent = torch.full(
                    (1, 3, 64), float(source_id), dtype=torch.float32
                )
                rotary = torch.full(
                    (1, 1, 3, 2), float(source_id), dtype=torch.float32
                )
                return latent, rotary

        transformer = FakeTransformer()
        source = torch.ones((1, 16, 21, 2, 2), dtype=torch.float32)
        target = torch.zeros_like(source)
        pack = phase_a._build_deployment_v2v_source_pack(
            transformer, source_video=source, noisy_target=target
        )
        self.assertEqual([row[0] for row in transformer.calls], [1.0, 0.0])
        self.assertTrue(torch.equal(transformer.calls[0][1], source))
        self.assertTrue(torch.equal(transformer.calls[1][1], target))
        self.assertEqual(pack.video.source_ids, (1.0, 0.0))
        self.assertEqual(pack.video.condition_tokens, 3)
        self.assertEqual(pack.video.total_tokens, 6)
        self.assertEqual(pack.video.target_mask.tolist(), [False] * 3 + [True] * 3)
        self.assertEqual(pack.receipt["patch_call_source_ids"], [1.0, 0.0])
        self.assertEqual(pack.receipt["image_reference_count"], 0)

    def test_identity_protocol_uses_scheduler_cell_33_with_fresh_step_index(self) -> None:
        class FakeScheduler:
            def __init__(self):
                self.step_index = None
                self.calls = []
                self.config = {"flow_shift": 5.0}

            def set_timesteps(self, count):
                self.calls.append(count)
                self.sigmas = torch.tensor(
                    phase_a.native.NATIVE_UNIPC40_SIGMAS, dtype=torch.float32
                )
                self.timesteps = torch.tensor(
                    phase_a.native.NATIVE_UNIPC40_TIMESTEPS, dtype=torch.float32
                )

        scheduler = FakeScheduler()
        diffusion = SimpleNamespace(scheduler=scheduler)
        coordinate = phase_a.PhaseASourceCoordinate(
            x_sigma=torch.zeros((1, 16, 21, 2, 2), dtype=torch.float32),
            timestep=torch.tensor([516.0], dtype=torch.float32),
            sigma=float(phase_a.native.NATIVE_UNIPC40_SIGMAS[33]),
            schedule_index=33,
            receipt={},
        )
        sigma, receipt = phase_a._prepare_identity_deployment_protocol(
            diffusion, coordinate
        )
        self.assertEqual(scheduler.calls, [40])
        self.assertIsNone(scheduler.step_index)
        self.assertTrue(torch.equal(sigma, scheduler.sigmas[33]))
        self.assertEqual(receipt["schedule_index"], 33)
        self.assertEqual(receipt["timestep"], 516)
        self.assertEqual(receipt["guidance_mode"], "v2v_apg")
        self.assertEqual(receipt["visual_condition"], "source_video_only_V")
        self.assertEqual(receipt["forward_order_per_field"], ["V_negative", "V_positive"])
        self.assertEqual(receipt["omega_txt"], 4.0)
        self.assertEqual(receipt["eta"], 0.5)
        self.assertEqual(receipt["norm_threshold"], 50.0)
        self.assertEqual(receipt["momentum"], 0.0)
        self.assertEqual(receipt["flow_shift"], 5.0)
        self.assertTrue(receipt["fresh_zero_momentum_history_equivalent"])
        self.assertFalse(receipt["full_sampler_trajectory_equivalent"])

    def test_identity_protocol_rejects_training_shift_three_scheduler(self) -> None:
        class ShiftThreeScheduler:
            config = {"flow_shift": 3.0}

            @staticmethod
            def set_timesteps(_count):
                raise AssertionError("shift-3 schedule must fail before materialization")

        coordinate = phase_a.PhaseASourceCoordinate(
            x_sigma=torch.zeros((1, 16, 21, 2, 2), dtype=torch.float32),
            timestep=torch.tensor([516.0], dtype=torch.float32),
            sigma=float(phase_a.native.NATIVE_UNIPC40_SIGMAS[33]),
            schedule_index=33,
            receipt={},
        )
        with self.assertRaisesRegex(
            phase_a.PairV7PhaseAError, "deployment flow-shift 5"
        ):
            phase_a._prepare_identity_deployment_protocol(
                SimpleNamespace(scheduler=ShiftThreeScheduler()), coordinate
            )

    def test_post_apg_uses_vendor_parameters_and_has_exact_leaf_vjp(self) -> None:
        calls = []
        vendor = ModuleType("bernini.models.wan_diffusion")

        class MomentumBuffer:
            def __init__(self, momentum):
                self.momentum = float(momentum)
                self.running_average = 0

            def update(self, value):
                self.running_average = value + self.momentum * self.running_average

        def normalized_guidance(
            *,
            pred_cond,
            pred_uncond,
            guidance_scale,
            momentum_buffer,
            eta,
            norm_threshold,
        ):
            calls.append(
                {
                    "guidance_scale": guidance_scale,
                    "momentum": momentum_buffer.momentum,
                    "eta": eta,
                    "norm_threshold": norm_threshold,
                }
            )
            momentum_buffer.update(pred_cond - pred_uncond)
            # A differentiable mock keeps the exact negative/condition leaf
            # dependence visible: post velocity = 4*condition - 3*negative.
            return pred_uncond + guidance_scale * momentum_buffer.running_average

        vendor.MomentumBuffer = MomentumBuffer
        vendor.normalized_guidance = normalized_guidance
        FakeDiffusion = type(
            "FakeDiffusion", (), {"__module__": "bernini.models.wan_diffusion"}
        )
        runtime = object.__new__(phase_a.NativeFeatureVJPRuntime)
        runtime.diffusion = FakeDiffusion()
        runtime.coordinate = SimpleNamespace(
            x_sigma=torch.zeros((1, 16, 21, 1, 1), dtype=torch.float32)
        )
        runtime.sigma = torch.tensor(0.5, dtype=torch.float32)
        negative = torch.full(
            (1, 16, 21, 1, 1), 0.25, dtype=torch.bfloat16, requires_grad=True
        )
        condition = torch.full(
            (1, 16, 21, 1, 1), 0.75, dtype=torch.bfloat16, requires_grad=True
        )
        with patch.dict(sys.modules, {"bernini.models.wan_diffusion": vendor}):
            post_velocity = runtime._post_apg_from_raw(negative, condition)
        expected = 4.0 * condition.float() - 3.0 * negative.float()
        self.assertEqual(post_velocity.dtype, torch.float32)
        self.assertTrue(torch.equal(post_velocity, expected))
        negative_vjp, condition_vjp = torch.autograd.grad(
            post_velocity,
            (negative, condition),
            grad_outputs=torch.ones_like(post_velocity),
        )
        self.assertTrue(torch.equal(negative_vjp, torch.full_like(negative, -3.0)))
        self.assertTrue(torch.equal(condition_vjp, torch.full_like(condition, 4.0)))
        self.assertEqual(
            calls,
            [
                {
                    "guidance_scale": 4.0,
                    "momentum": 0.0,
                    "eta": 0.5,
                    "norm_threshold": 50.0,
                }
            ],
        )

    def test_final_source_binding_keeps_full_sp4_consensus_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = []
            gathered = []
            for arm_index in range(2):
                video = root / f"source-{arm_index}.mp4"
                video.write_bytes(f"source-{arm_index}".encode("ascii"))
                video_sha = hashlib.sha256(video.read_bytes()).hexdigest()
                event = SimpleNamespace(
                    source_sample_id=f"source-{arm_index}",
                    event_digest=_sha(("c", "d")[arm_index]),
                    source_video=SimpleNamespace(path=video, sha256=video_sha),
                )
                events.append(event)
                source_receipt = phase_a._seal(
                    {
                        "video_path": str(video),
                        "video_sha256": video_sha,
                        "clean_latent_sha256": _sha(("e", "f")[arm_index]),
                        "deployment_visual_condition": "source_video_only_V",
                        "image_reference_count": 0,
                        "reference_indices": [],
                        "reference_latent_sha256": [],
                        "frame_count": 81,
                        "fps": 25.0,
                    }
                )
                for sp_rank, rank in enumerate(
                    phase_a.distributed_runtime.SP_GROUP_RANKS[arm_index]
                ):
                    gathered.append(
                        {
                            "rank": rank,
                            "arm_index": arm_index,
                            "sp_rank": sp_rank,
                            "source_sample_id": event.source_sample_id,
                            "source_event_digest": event.event_digest,
                            "source_receipt_digest": source_receipt["receipt_digest"],
                            "source_receipt": source_receipt,
                        }
                    )
            selected = phase_a._select_source_receipts_by_arm(
                gathered, SimpleNamespace(events=tuple(events))
            )
            self.assertEqual(len(selected), 2)
            self.assertTrue(all(row["sp4_receipt_consensus"] for row in selected))
            self.assertEqual(
                selected[0]["source_receipt"]["reference_indices"],
                [],
            )
            self.assertEqual(
                selected[0]["source_receipt"]["reference_latent_sha256"], []
            )

    def test_fit_only_objective_is_numerically_equal_without_optimizer_receipt(self) -> None:
        gate_name, gate_weight = phase_a.action_adapter.sigma_gate(33)
        x_sigma = torch.zeros((1, 1, 1, 1, 4), dtype=torch.float32)
        sigma = torch.tensor([0.25], dtype=torch.float32)
        timestep = torch.tensor([250.0], dtype=torch.float32)
        query = phase_a.cagd.SameStateQuery(
            sample_id="fit-only-toy",
            x_sigma=x_sigma,
            sigma=sigma,
            timestep=timestep,
            schedule_index=33,
            gate_name=gate_name,
            gate_weight=float(gate_weight),
            coordinate_digest=_sha("a"),
            x_sigma_object_id=id(x_sigma),
            sigma_object_id=id(sigma),
            timestep_object_id=id(timestep),
            x_sigma_version=int(x_sigma._version),
            sigma_version=int(sigma._version),
            timestep_version=int(timestep._version),
        )
        base = {
            branch: torch.zeros_like(x_sigma)
            for branch in phase_a.cagd.BRANCH_ORDER
        }
        base["action"] = torch.tensor(
            [[[[[0.5, -0.5, 1.0, -1.0]]]]], dtype=torch.float32
        )
        student = {
            branch: (value + (0.15 if branch == "action" else 0.01))
            .detach()
            .requires_grad_(True)
            for branch, value in base.items()
        }
        packet = phase_a.cagd.PredictionPacket(
            query=query,
            base_by_branch=base,
            student_by_branch=student,
            prompt_bank_digest=_sha("b"),
            shared_query_object_all_forwards=True,
            call_order=tuple(
                [f"base:{branch}" for branch in phase_a.cagd.BRANCH_ORDER]
                + [f"student:{branch}" for branch in phase_a.cagd.BRANCH_ORDER]
            ),
            leaf_vjp_mode=True,
        )
        measured, receipt = phase_a.build_fit_only_measurement_objective(packet)
        legacy = phase_a.cagd.build_distill_objective(packet)
        self.assertTrue(torch.equal(measured.loss, legacy.loss))
        self.assertTrue(
            torch.equal(measured.action_match_loss, legacy.action_match_loss)
        )
        self.assertTrue(
            torch.equal(measured.negative_parity_loss, legacy.negative_parity_loss)
        )
        self.assertTrue(torch.equal(measured.trust_penalty, legacy.trust_penalty))
        self.assertFalse(receipt["cagd_build_distill_objective_called"])
        self.assertFalse(receipt["legacy_optimizer_authority_consumed"])
        for field in (
            "global_population_go",
            "optimizer_authorized",
            "parameter_update_authorized",
            "action_success_claimed",
        ):
            self.assertFalse(receipt[field])

    def test_dp2_union_projection_passes_partial_conflict(self) -> None:
        result = self.union(
            self.vector((0, 1.0), (8, 1.0)),
            self.vector((1, 1.0), (8, 2.0)),
        )
        self.assertTrue(result.geometry_audit_passed, result.receipt["failure_codes"])
        self.assertEqual(result.receipt["identity_probe_union_count"], 16)
        self.assertEqual(result.receipt["identity_global_effective_rank"], 8)
        self.assertTrue(
            all(
                row["effective_rank"] >= 3
                for row in result.receipt["identity_source_family_rank_gate"]
            )
        )
        self.assertEqual(result.receipt["projection_count_after_union"], 1)
        self.assertFalse(result.receipt["local_project_then_average"])
        self.assertTrue(result.receipt["unprojected_dp2_action_gradients_exchanged"])
        self.assertTrue(
            all(row["passed"] for row in result.receipt["per_family_action_descent"])
        )

    def test_union_before_projection_rejects_local_projection_counterexample(self) -> None:
        result = self.union(
            self.vector((0, 1.0), (1, 1.0)),
            self.vector((0, 1.0), (1, 1.0)),
        )
        self.assertFalse(result.geometry_audit_passed)
        self.assertIn(
            "UNION_IDENTITY_NULLSPACE_GEOMETRY_NO_GO", result.receipt["failure_codes"]
        )
        local_project_then_average = torch.tensor([0.5, 0.5], dtype=torch.float64)
        self.assertGreater(abs(float(local_project_then_average[0])), 0.0)
        self.assertGreater(abs(float(local_project_then_average[1])), 0.0)

    def test_each_action_family_must_descend_even_if_mean_descends(self) -> None:
        result = self.union(
            self.vector((8, 2.0)),
            self.vector((8, -0.2)),
        )
        self.assertTrue(result.transport.geometry_authorized)
        self.assertFalse(result.geometry_audit_passed)
        self.assertIn("PER_FAMILY_ACTION_DESCENT_FAILED:turn", result.receipt["failure_codes"])

    def _world_union_inputs(self):
        gradients = {
            "sit": self.named(self.vector((0, 1.0), (8, 1.0))),
            "turn": self.named(self.vector((1, 1.0), (8, 2.0))),
        }
        metadata = {
            "sit": self.metadata("sit", "fit-sit", gradients["sit"], 0),
            "turn": self.metadata("turn", "fit-turn", gradients["turn"], 1),
        }
        return gradients, metadata, self.probes()

    @staticmethod
    def _replicate_world_status(output, value, group) -> None:
        del group
        for rank in range(phase_a.WORLD_SIZE):
            row = dict(value)
            row["rank"] = rank
            output[rank] = row

    def test_world_union_input_receipt_is_order_stable_and_byte_bound(self) -> None:
        gradients, metadata, probes = self._world_union_inputs()
        kwargs = {
            "checkpoint_content_receipt_digest": self.checkpoint_digest,
            "parameter_state_sha256": self.parameter_state_digest,
            "fit_only_geometry_authority_digest": self.authorization_digest,
            "config": v7.TransportConfig(),
        }
        left = phase_a._world_union_input_receipt(
            action_gradient_by_family=gradients,
            action_metadata_by_family=metadata,
            identity_probes=probes,
            **kwargs,
        )
        right = phase_a._world_union_input_receipt(
            action_gradient_by_family=dict(reversed(tuple(gradients.items()))),
            action_metadata_by_family=dict(reversed(tuple(metadata.items()))),
            identity_probes=list(reversed(probes)),
            **kwargs,
        )
        self.assertEqual(left, right)
        changed = dict(gradients)
        changed["sit"] = self.named(self.vector((0, 1.0), (8, 1.125)))
        changed_receipt = phase_a._world_union_input_receipt(
            action_gradient_by_family=changed,
            action_metadata_by_family=metadata,
            identity_probes=probes,
            **kwargs,
        )
        self.assertNotEqual(
            left["receipt_digest"], changed_receipt["receipt_digest"]
        )
        self.assertEqual(len(left["action_rows"]), 2)
        self.assertEqual(len(left["identity_rows"]), 16)

    def test_world_union_solver_runs_once_on_rank0_cpu_and_nonroot_only_receives(self) -> None:
        gradients, metadata, probes = self._world_union_inputs()
        world_group = object()
        root_parallel = SimpleNamespace(
            contract=SimpleNamespace(rank=0), world_group=world_group
        )
        captured = {}

        def root_broadcast(values, src, group):
            self.assertEqual(src, 0)
            self.assertIs(group, world_group)
            captured["payload"] = values[0]

        with patch(
            "torch.distributed.all_gather_object",
            side_effect=self._replicate_world_status,
        ) as gather, patch(
            "torch.distributed.broadcast_object_list", side_effect=root_broadcast
        ) as broadcast, patch.object(
            phase_a,
            "union_project_and_audit_action_families",
            wraps=phase_a.union_project_and_audit_action_families,
        ) as solver, patch.object(
            phase_a.distributed_runtime, "digest_consensus"
        ) as consensus:
            root = phase_a.world_rank0_cpu_union_project_and_audit_action_families(
                action_gradient_by_family=gradients,
                action_metadata_by_family=metadata,
                identity_probes=probes,
                checkpoint_content_receipt_digest=self.checkpoint_digest,
                parameter_state_sha256=self.parameter_state_digest,
                fit_only_geometry_authority_digest=self.authorization_digest,
                parallel=root_parallel,
            )
        self.assertEqual(gather.call_count, 2)
        broadcast.assert_called_once()
        consensus.assert_called_once()
        solver.assert_called_once()
        solved = solver.call_args.kwargs
        self.assertTrue(
            all(
                tensor.device.type == "cpu"
                for mapping in solved["action_gradient_by_family"].values()
                for tensor in mapping.values()
            )
        )
        self.assertTrue(
            all(
                tensor.device.type == "cpu"
                for probe in solved["identity_probes"]
                for tensor in probe.gradient_by_parameter.values()
            )
        )
        self.assertEqual(root.authority_receipt["authoritative_world_rank"], 0)
        self.assertEqual(root.authority_receipt["solver_execution_count"], 1)
        self.assertEqual(root.authority_receipt["solver_device"], "cpu")
        self.assertFalse(root.authority_receipt["replicated_gpu_solver_used"])
        self.assertTrue(root.authority_receipt["world_result_digest_consensus"])
        phase_a._sealed_receipt_digest(
            root.authority_receipt, label="test root authority"
        )

        nonroot_parallel = SimpleNamespace(
            contract=SimpleNamespace(rank=3), world_group=world_group
        )

        def nonroot_broadcast(values, src, group):
            self.assertEqual(src, 0)
            self.assertIs(group, world_group)
            values[0] = captured["payload"]

        with patch(
            "torch.distributed.all_gather_object",
            side_effect=self._replicate_world_status,
        ), patch(
            "torch.distributed.broadcast_object_list", side_effect=nonroot_broadcast
        ), patch.object(
            phase_a,
            "union_project_and_audit_action_families",
            side_effect=AssertionError("non-root must not solve"),
        ) as nonroot_solver, patch.object(
            phase_a.distributed_runtime, "digest_consensus"
        ):
            nonroot = phase_a.world_rank0_cpu_union_project_and_audit_action_families(
                action_gradient_by_family=gradients,
                action_metadata_by_family=metadata,
                identity_probes=probes,
                checkpoint_content_receipt_digest=self.checkpoint_digest,
                parameter_state_sha256=self.parameter_state_digest,
                fit_only_geometry_authority_digest=self.authorization_digest,
                parallel=nonroot_parallel,
            )
        nonroot_solver.assert_not_called()
        self.assertEqual(nonroot.receipt, root.receipt)
        self.assertEqual(nonroot.transport_receipt, root.transport_receipt)
        self.assertEqual(nonroot.authority_receipt, root.authority_receipt)

    def test_world_union_input_mismatch_stops_before_solver_and_broadcast(self) -> None:
        gradients, metadata, probes = self._world_union_inputs()
        parallel = SimpleNamespace(
            contract=SimpleNamespace(rank=0), world_group=object()
        )

        def mismatch(output, value, group):
            del group
            for rank in range(phase_a.WORLD_SIZE):
                row = dict(value)
                row["rank"] = rank
                if rank == phase_a.WORLD_SIZE - 1:
                    row["digest"] = _sha("0")
                output[rank] = row

        with patch(
            "torch.distributed.all_gather_object", side_effect=mismatch
        ), patch(
            "torch.distributed.broadcast_object_list"
        ) as broadcast, patch.object(
            phase_a, "union_project_and_audit_action_families"
        ) as solver:
            with self.assertRaisesRegex(
                phase_a.PairV7PhaseAError, "input differs across WORLD8"
            ):
                phase_a.world_rank0_cpu_union_project_and_audit_action_families(
                    action_gradient_by_family=gradients,
                    action_metadata_by_family=metadata,
                    identity_probes=probes,
                    checkpoint_content_receipt_digest=self.checkpoint_digest,
                    parameter_state_sha256=self.parameter_state_digest,
                    fit_only_geometry_authority_digest=self.authorization_digest,
                    parallel=parallel,
                )
        solver.assert_not_called()
        broadcast.assert_not_called()

    def test_world_union_root_failure_is_broadcast_as_one_fail_closed_code(self) -> None:
        gradients, metadata, probes = self._world_union_inputs()
        parallel = SimpleNamespace(
            contract=SimpleNamespace(rank=0), world_group=object()
        )

        with patch(
            "torch.distributed.all_gather_object",
            side_effect=self._replicate_world_status,
        ), patch(
            "torch.distributed.broadcast_object_list"
        ), patch.object(
            phase_a,
            "union_project_and_audit_action_families",
            side_effect=RuntimeError("synthetic root solver error"),
        ):
            with self.assertRaisesRegex(
                phase_a.PairV7PhaseAError, "ROOT_CPU_UNION_SOLVE_FAILED"
            ):
                phase_a.world_rank0_cpu_union_project_and_audit_action_families(
                    action_gradient_by_family=gradients,
                    action_metadata_by_family=metadata,
                    identity_probes=probes,
                    checkpoint_content_receipt_digest=self.checkpoint_digest,
                    parameter_state_sha256=self.parameter_state_digest,
                    fit_only_geometry_authority_digest=self.authorization_digest,
                    parallel=parallel,
                )

    def test_final_receipt_field_consensus_names_the_divergent_field(self) -> None:
        parallel = SimpleNamespace(world_group=object())

        def gather(output, value, group):
            self.assertIs(group, parallel.world_group)
            for rank in range(phase_a.WORLD_SIZE):
                output[rank] = dict(value)
            output[-1]["union_projection_receipt"] = _sha("0")

        with patch("torch.distributed.all_gather_object", side_effect=gather):
            with self.assertRaisesRegex(
                phase_a.PairV7PhaseAError,
                "union_projection_receipt",
            ):
                phase_a._assert_world_receipt_field_consensus(
                    {
                        "union_projection_receipt": {"value": 1},
                        "rank_runtime_provenance": [{"rank": 0}],
                    },
                    parallel=parallel,
                )

    def test_probe_parameter_state_mismatch_is_fail_closed(self) -> None:
        probes = self.probes()
        probes[0] = v7.IdentityGradientProbe(
            probe_id=probes[0].probe_id,
            family=probes[0].family,
            gradient_by_parameter=probes[0].gradient_by_parameter,
            feature_sketch_sha256=probes[0].feature_sketch_sha256,
            source_coordinate_receipt_digest=probes[0].source_coordinate_receipt_digest,
            gradient_computation_receipt_digest=(
                probes[0].gradient_computation_receipt_digest
            ),
            checkpoint_content_receipt_digest=probes[0].checkpoint_content_receipt_digest,
            parameter_state_sha256=_sha("0"),
        )
        gradients = {
            "sit": self.named(self.vector((8, 1.0))),
            "turn": self.named(self.vector((8, 2.0))),
        }
        metadata = {
            "sit": self.metadata("sit", "fit-sit", gradients["sit"], 0),
            "turn": self.metadata("turn", "fit-turn", gradients["turn"], 1),
        }
        with self.assertRaisesRegex(v7.PairV7TransportError, "state differs"):
            phase_a.union_project_and_audit_action_families(
                action_gradient_by_family=gradients,
                action_metadata_by_family=metadata,
                identity_probes=probes,
                checkpoint_content_receipt_digest=self.checkpoint_digest,
                parameter_state_sha256=self.parameter_state_digest,
                fit_only_geometry_authority_digest=self.authorization_digest,
            )

    def test_fixed_gauge_freezes_a_and_exposes_zero_b_only(self) -> None:
        class FakeTransformer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.base = torch.nn.Parameter(torch.ones(1), requires_grad=False)
                self.lora_a = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float32))
                self.lora_b = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

        class FakeHandle:
            def __init__(self):
                self.transformer = FakeTransformer()

            def trainable_named_parameters(self):
                return (
                    (
                        "blocks.0.attn2.to_q.action_lora_a.weight",
                        self.transformer.lora_a,
                    ),
                    (
                        "blocks.0.attn2.to_q.action_lora_b.weight",
                        self.transformer.lora_b,
                    ),
                )

        gauge = phase_a.configure_fixed_a_b_only_gauge(FakeHandle())
        self.assertFalse(gauge.frozen_a_named[0][1].requires_grad)
        self.assertTrue(gauge.trainable_b_named[0][1].requires_grad)
        self.assertEqual(gauge.receipt["gauge"], "freeze_action_lora_A_train_zero_init_B_only")
        self.assertFalse(gauge.receipt["parameter_mutation_authorized"])

    def test_sp4_bundle_binds_every_rank_receipt_to_averaged_gradient(self) -> None:
        gradient = self.named([1.0, 2.0])
        unsigned = {
            "schema_version": phase_a.IDENTITY_VJP_RECEIPT_SCHEMA,
            "sp_rank": 0,
            "gradient_sha256": phase_a._named_gradient_sha256(gradient),
            "checkpoint_content_receipt_digest": self.checkpoint_digest,
            "parameter_state_sha256": self.parameter_state_digest,
        }
        local = phase_a._seal(unsigned)
        parallel = SimpleNamespace(
            contract=SimpleNamespace(sp_rank=0), sp_group=object()
        )

        def gather(output, value, group):
            self.assertIs(group, parallel.sp_group)
            for rank in range(4):
                row = dict(value)
                row.pop("receipt_digest")
                row["sp_rank"] = rank
                output[rank] = phase_a._seal(row)

        with patch("torch.distributed.all_gather_object", side_effect=gather), patch.object(
            phase_a.distributed_runtime, "digest_consensus"
        ) as consensus:
            bundle = phase_a._bundle_sp4_vjp_receipts(
                local_receipt=local,
                averaged_gradient=gradient,
                parallel=parallel,
                label="toy-identity",
                common_fields=(
                    "schema_version",
                    "gradient_sha256",
                    "checkpoint_content_receipt_digest",
                    "parameter_state_sha256",
                ),
            )
        self.assertEqual(bundle["sp_size"], 4)
        self.assertEqual(len(bundle["sp_rank_receipts"]), 4)
        self.assertTrue(bundle["all_four_rank_local_vjps_bound"])
        self.assertTrue(bundle["sp4_arithmetic_average_bound"])
        consensus.assert_called_once()

    def test_sp4_bundle_rejects_one_rank_with_different_gradient_binding(self) -> None:
        gradient = self.named([1.0, 2.0])
        local = phase_a._seal(
            {
                "schema_version": phase_a.IDENTITY_VJP_RECEIPT_SCHEMA,
                "sp_rank": 0,
                "gradient_sha256": phase_a._named_gradient_sha256(gradient),
                "checkpoint_content_receipt_digest": self.checkpoint_digest,
                "parameter_state_sha256": self.parameter_state_digest,
            }
        )
        parallel = SimpleNamespace(
            contract=SimpleNamespace(sp_rank=0), sp_group=object()
        )

        def gather(output, value, group):
            for rank in range(4):
                row = dict(value)
                row.pop("receipt_digest")
                row["sp_rank"] = rank
                if rank == 3:
                    row["gradient_sha256"] = _sha("f")
                output[rank] = phase_a._seal(row)

        with patch("torch.distributed.all_gather_object", side_effect=gather):
            with self.assertRaisesRegex(phase_a.PairV7PhaseAError, "common field differs"):
                phase_a._bundle_sp4_vjp_receipts(
                    local_receipt=local,
                    averaged_gradient=gradient,
                    parallel=parallel,
                    label="toy-identity",
                    common_fields=(
                        "schema_version",
                        "gradient_sha256",
                        "checkpoint_content_receipt_digest",
                        "parameter_state_sha256",
                    ),
                )


if __name__ == "__main__":
    unittest.main()
