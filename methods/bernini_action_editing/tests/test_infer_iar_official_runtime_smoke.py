from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
import struct
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

try:
    import torch  # noqa: E402
except ImportError:  # pragma: no cover
    torch = None

if torch is not None:
    import identity_anchored_action_residual as iar_core  # noqa: E402
    import infer_dclr_reward_runtime_smoke as dclr  # noqa: E402
    import infer_iar_official_runtime_smoke as smoke  # noqa: E402
else:  # pragma: no cover
    iar_core = None
    dclr = None
    smoke = None


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class DependencyLightSourceGuards(unittest.TestCase):
    def test_untrusted_callback_scaffold_is_replaced(self) -> None:
        old_runtime = METHOD_ROOT / "infer_iar_field_feasibility.py"
        old_test = METHOD_ROOT / "tests/test_infer_iar_field_feasibility.py"
        new_runtime = METHOD_ROOT / "infer_iar_official_runtime_smoke.py"
        self.assertFalse(old_runtime.exists())
        self.assertFalse(old_test.exists())
        self.assertTrue(new_runtime.is_file())
        source = new_runtime.read_text(encoding="utf-8")
        for forbidden in (
            "FrozenFieldForwardResponse",
            "bind_forward_response",
            "make_official_shared_step_callback",
            ".backward(",
            "optimizer.step(",
            "torch.save(",
            "positive_control_paired_target",
            '"--candidate-row-index"',
            '"--expected-candidate-iid"',
            "candidate_row =",
            "replace_edit_instruction(",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn("dclr.shared_step_target_prediction(", source)
        self.assertIn("iar_core.compute_frozen_identity_anchored_teacher(", source)
        self.assertIn("build_canonical_renderer_sample(", source)
        self.assertIn("_tokenize_canonical_condition(", source)

    def test_cell_api_has_no_replaceable_forward_core_or_energy_slot(self) -> None:
        if smoke is None:
            return
        parameters = inspect.signature(smoke._run_official_cell).parameters
        for forbidden in (
            "forward_callback",
            "query_builder",
            "response",
            "core",
            "iar_core",
            "hard_negative_energies",
            "sigma_state",
            "noised_state",
        ):
            self.assertNotIn(forbidden, parameters)
        self.assertIn("renderer", parameters)
        self.assertIn("bundles", parameters)
        self.assertTrue(
            all(
                parameter.kind is inspect.Parameter.KEYWORD_ONLY
                for parameter in parameters.values()
            )
        )


@unittest.skipIf(torch is None, "torch is unavailable")
class BridgeAndManifestTests(unittest.TestCase):
    def manifest(self):
        action = "the dog picks up and holds the bone"
        noop = "the dog remains still without touching the bone"
        incomplete = "the dog approaches but does not pick up the bone"
        return action, noop, {
            "schema_version": smoke.HARD_NEGATIVE_MANIFEST_SCHEMA,
            "action_instruction_sha256": sha(action),
            "hard_negatives": [
                {
                    "condition_id": "semantic-noop",
                    "instruction": noop,
                    "instruction_sha256": sha(noop),
                },
                {
                    "condition_id": "incomplete-event",
                    "instruction": incomplete,
                    "instruction_sha256": sha(incomplete),
                },
            ],
            "energy_semantics": smoke.ENERGY_SEMANTICS,
            "declared_use": smoke.HARD_NEGATIVE_DECLARED_USE,
        }

    def test_bridge_is_fp32_exact_and_preserves_requested_traversal(self) -> None:
        self.assertEqual(
            smoke.validate_bridge_fractions([1.0, 0.5, 0.0]),
            (1.0, 0.5, 0.0),
        )
        self.assertEqual(
            smoke.validate_bridge_fractions([0.0, 0.25, 1.0]),
            (0.0, 0.25, 1.0),
        )
        for bad in (
            [0.0],
            [0.0, 0.5],
            [0.0, 0.75, 0.25, 1.0],
            [1.0, 0.5, 0.5, 0.0],
            [1.0, float("nan"), 0.0],
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(smoke.IAROfficialRuntimeSmokeError):
                    smoke.validate_bridge_fractions(bad)

        source = torch.randn(1, 16, 21, 4, 4, dtype=torch.float32)
        proposal = torch.randn_like(source)
        self.assertIs(smoke.construct_bridge_clean(source, proposal, 0.0), source)
        self.assertIs(smoke.construct_bridge_clean(source, proposal, 1.0), proposal)
        middle = smoke.construct_bridge_clean(source, proposal, 0.5)
        self.assertTrue(torch.equal(middle, 0.5 * source + 0.5 * proposal))
        self.assertEqual(middle.dtype, torch.float32)

    def test_manifest_has_text_only_and_no_external_energy_escape_hatch(self) -> None:
        action, noop, manifest = self.manifest()
        result = smoke.validate_hard_negative_manifest(
            manifest,
            action_instruction_sha256=sha(action),
            noop_instruction_sha256=sha(noop),
        )
        self.assertEqual(result["hard_negative_count"], 2)
        self.assertTrue(result["noop_present"])
        self.assertFalse(result["energies_supplied_externally"])

        top_energy = dict(manifest)
        top_energy["energies"] = [0.1, 0.2]
        item_energy = json.loads(json.dumps(manifest))
        item_energy["hard_negatives"][0]["energy"] = 0.1
        too_few = json.loads(json.dumps(manifest))
        too_few["hard_negatives"] = too_few["hard_negatives"][:1]
        missing_noop = json.loads(json.dumps(manifest))
        missing_noop["hard_negatives"][0] = {
            "condition_id": "reverse",
            "instruction": "the dog drops the bone",
            "instruction_sha256": sha("the dog drops the bone"),
        }
        duplicate = json.loads(json.dumps(manifest))
        duplicate["hard_negatives"][1] = dict(duplicate["hard_negatives"][0])
        for bad in (top_energy, item_energy, too_few, missing_noop, duplicate):
            with self.subTest(bad=bad):
                with self.assertRaises(smoke.IAROfficialRuntimeSmokeError):
                    smoke.validate_hard_negative_manifest(
                        bad,
                        action_instruction_sha256=sha(action),
                        noop_instruction_sha256=sha(noop),
                    )

    def test_negative_energy_is_actual_fp32_mse_to_true_velocity(self) -> None:
        negatives = torch.tensor(
            [[[[1.0, 2.0]], [[3.0, 4.0]]]], dtype=torch.bfloat16
        )
        target = torch.tensor([[[0.0, 1.0]]], dtype=torch.float32)
        actual = smoke.hard_negative_energies(negatives, target)
        expected = (
            negatives.float() - target[:, None]
        ).square().mean(dim=(2, 3))
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(actual.dtype, torch.float32)
        self.assertFalse(actual.requires_grad)

    def test_canonical_messages_are_dataset_independent_and_encoder_guarded(self) -> None:
        instruction = "the dog picks up and holds the bone"
        sample = smoke.build_canonical_renderer_sample(
            instruction,
            expected_instruction_sha256=sha(instruction),
        )
        self.assertEqual(set(sample), {"inputs"})
        self.assertEqual(
            json.loads(sample["inputs"]),
            [
                {"type": "video", "has_loss": 0},
                {
                    "type": "text",
                    "text": instruction,
                    "has_loss": 0,
                },
                {"type": "video_gen", "has_loss": 1},
            ],
        )
        with self.assertRaises(smoke.IAROfficialRuntimeSmokeError):
            smoke.build_canonical_renderer_sample(
                instruction,
                expected_instruction_sha256="f" * 64,
            )
        with self.assertRaises(smoke.IAROfficialRuntimeSmokeError):
            smoke._canonical_sample_identity(
                {**sample, "iid": "forbidden-dataset-row-iid"},
                expected_instruction_sha256=sha(instruction),
            )

        calls = []

        def encode(messages, tokenizer, **kwargs):
            calls.append((json.loads(json.dumps(messages)), tokenizer, dict(kwargs)))
            return {
                "input_ids": torch.tensor([11, 12, 13], dtype=torch.long),
                "attention_mask": torch.ones(3, dtype=torch.long),
                "t5_input_lens": torch.tensor([3], dtype=torch.long),
            }

        class Renderer:
            max_sequence_length = 512

            @staticmethod
            def get_t5_text_embeddings(ids, attention, lens):
                del attention, lens
                return [512], torch.zeros(
                    (int(ids.shape[0]), 512, 2), dtype=torch.float32
                )

        tokenizer = object()
        condition = smoke._tokenize_canonical_condition(
            renderer=Renderer(),
            tokenizer=tokenizer,
            encode_renderer_messages=encode,
            sample=sample,
            expected_instruction_sha256=sha(instruction),
            task_name="t2v",
            device=torch.device("cpu"),
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], json.loads(sample["inputs"]))
        self.assertIs(calls[0][1], tokenizer)
        self.assertEqual(
            calls[0][2],
            {
                "task_name": "t2v",
                "drop_text": False,
                "drop_video": False,
                "drop_img": False,
            },
        )
        self.assertEqual(condition.instruction_sha256, sha(instruction))


@unittest.skipIf(torch is None, "torch is unavailable")
class OfficialCellHarness(unittest.TestCase):
    class FakeTransformer:
        # Class bodies execute before unittest's skip decorator is consulted.
        dtype = torch.float32 if torch is not None else None

        def patch_vae_latent(self, hidden_states, source_id=None):
            tokens = dclr.pack_spatial_velocity(hidden_states)
            count = int(tokens.shape[1])
            positions = torch.arange(
                count * dclr.ROPE_COMPLEX_DIM,
                dtype=torch.float64,
                device=hidden_states.device,
            ).reshape(1, 1, count, dclr.ROPE_COMPLEX_DIM)
            rope = torch.complex(
                positions, torch.full_like(positions, float(source_id))
            )
            return tokens, rope

    class FakeDecoder:
        def __init__(self):
            self.calls = []

        def shared_step(self, **kwargs):
            self.calls.append(kwargs)
            noisy = kwargs["noisy_latents"]
            scalar = kwargs["cond_embeds"].reshape(-1)[0].to(noisy.dtype)
            total = int(noisy.shape[1])
            index = torch.arange(
                noisy.numel(), dtype=noisy.dtype, device=noisy.device
            ).reshape_as(noisy)
            pattern = torch.sin(index * 0.013) + 0.3 * torch.cos(index * 0.031)
            if total == 2 * (21 * 2 * 2):
                prefix = noisy[:, : total // 2, :].mean(dim=1, keepdim=True)
                prefix = prefix.repeat(1, total, 1)
            else:
                prefix = torch.zeros_like(noisy)
            # Text/source interaction makes action-minus-noop source residuals
            # observable, while keeping every output deterministic.
            return (
                0.07 * noisy
                + scalar * pattern
                + 0.11 * prefix
                + 0.02 * scalar * prefix * pattern
            )

    class FakeRenderer:
        def __init__(self):
            self.diff_dec = OfficialCellHarness.FakeDecoder()

        def forward(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError("renderer.forward must not run")

    @staticmethod
    def condition(instruction: str, task: str, scalar: float):
        return dclr.TextCondition(
            text_lens=[1],
            text_embs=torch.tensor([[[scalar]]], dtype=torch.float32),
            prompt_sha256=sha(f"{task}:{instruction}"),
            instruction_sha256=sha(instruction),
            task_name=task,
        )

    def setUp(self) -> None:
        generator = torch.Generator().manual_seed(17)
        shape = (1, 16, 21, 4, 4)
        self.source = torch.randn(shape, generator=generator)
        self.proposal = torch.randn(shape, generator=generator)
        self.wrong = torch.randn(shape, generator=generator) + 0.5
        self.epsilon = torch.randn(shape, generator=generator)
        self.transformer = self.FakeTransformer()
        self.action_text = "the dog picks up and holds the bone"
        self.noop_text = "the dog remains still without touching the bone"
        self.incomplete_text = "the dog approaches but does not lift the bone"
        self.action_t2v = self.condition(self.action_text, "t2v", 1.4)
        self.negatives = (
            self.condition(self.noop_text, "t2v", -0.2),
            self.condition(self.incomplete_text, "t2v", 0.45),
        )
        self.noop_mv2v = self.condition(self.noop_text, "mv2v", -0.2)
        self.action_mv2v = self.condition(self.action_text, "mv2v", 1.4)

    def bundle(self, bridge_fraction=0.5, sigma=0.6):
        bridge = smoke.construct_bridge_clean(
            self.source, self.proposal, bridge_fraction
        )
        return dclr.build_same_state_query_bundle(
            self.transformer,
            correct_source_spatial=self.source,
            wrong_source_spatial=self.wrong,
            student_clean_spatial=bridge,
            epsilon_spatial=self.epsilon,
            point=dclr.flow_query_point(sigma),
        )

    def run_cell(self, bridge_fraction=0.5, sigma=0.6):
        renderer = self.FakeRenderer()
        bundle = self.bundle(bridge_fraction, sigma)
        result = smoke._run_official_cell(
            renderer=renderer,
            model_id="transformer_1",
            bundles=(bundle,),
            bridge_fraction=bridge_fraction,
            action_t2v_condition=self.action_t2v,
            hard_negative_t2v_conditions=self.negatives,
            noop_mv2v_condition=self.noop_mv2v,
            action_mv2v_condition=self.action_mv2v,
            correct_source_sha256="c" * 64,
            wrong_source_sha256s=("d" * 64,),
        )
        return renderer, bundle, result

    def test_exact_seven_direct_forwards_and_noop_only_projection(self) -> None:
        renderer, bundle, result = self.run_cell()
        expected = iar_core.expected_frozen_branch_names(2, 1)
        self.assertEqual(len(renderer.diff_dec.calls), 7)
        self.assertEqual(result.branch_names, expected)
        self.assertEqual(tuple(result.record["branch_order"]), expected)
        self.assertEqual(result.record["forward_count"], 7)
        self.assertTrue(result.record["direct_official_shared_step"])
        self.assertTrue(result.record["no_forward_callback"])

        calls = renderer.diff_dec.calls
        # action + two negatives share the literal T2V target-tail view.
        self.assertTrue(all(call["noisy_latents"] is bundle.t2v_noisy_latents for call in calls[:3]))
        target_count = bundle.target_tokens
        for call in calls[3:]:
            self.assertTrue(
                torch.equal(
                    call["noisy_latents"][:, -target_count:, :],
                    bundle.t2v_noisy_latents,
                )
            )
        self.assertTrue(
            torch.equal(
                result.frozen_fields.hard_negative_energies,
                smoke.hard_negative_energies(
                    result.frozen_fields.frozen_t2v_hard_negatives,
                    bundle.true_velocity_packed,
                ),
            )
        )
        expected_action_energy = (
            result.frozen_fields.frozen_t2v_action.float()
            - bundle.true_velocity_packed.float()
        ).square().mean(dim=(1, 2))
        expected_margins = (
            result.frozen_fields.hard_negative_energies
            - expected_action_energy[:, None]
        )
        self.assertEqual(
            result.record["action_energy_EA"],
            [float(item) for item in expected_action_energy.tolist()],
        )
        self.assertEqual(
            result.record["ordering_margins_Ek_minus_EA"],
            expected_margins.tolist(),
        )
        self.assertTrue(
            result.record["rf_squared_error_proxy_not_likelihood_or_free_energy"]
        )
        self.assertTrue(
            result.record["ordering_is_diagnostic_not_training_authorization"]
        )
        expected_tangent = (
            result.frozen_fields.frozen_identity_noop_correct[:, None]
            - result.frozen_fields.frozen_identity_noop_wrong_sources
        ).float()
        self.assertTrue(
            torch.allclose(
                result.teacher_result.diagnostics.identity_tangents,
                expected_tangent,
            )
        )
        self.assertTrue(result.record["independent_recompute"]["verified"])
        self.assertTrue(
            result.record["independent_recompute"][
                "projection_uses_noop_source_swaps_only"
            ]
        )
        self.assertFalse(
            result.record["metrics"][
                "source_action_invariance_calibration_authorized"
            ]
        )
        self.assertTrue(result.record["metrics"]["M_equals_one_plumbing_only"])

    def test_action_diagnostic_uses_residuals_not_absolute_action_fields(self) -> None:
        _, _, result = self.run_cell()
        fields = result.frozen_fields
        correct = (
            fields.frozen_identity_action_correct
            - fields.frozen_identity_noop_correct
        ).float()
        wrong = (
            fields.frozen_identity_action_wrong_sources
            - fields.frozen_identity_noop_wrong_sources
        ).float()
        self.assertTrue(
            torch.allclose(
                result.teacher_result.diagnostics.source_conditioned_action_residual_correct,
                correct,
            )
        )
        self.assertTrue(
            torch.allclose(
                result.teacher_result.diagnostics.source_conditioned_action_residual_wrong_sources,
                wrong,
            )
        )
        self.assertFalse(
            result.teacher_result.receipt["action_source_invariance_diagnostic"][
                "training_authorized_by_this_diagnostic"
            ]
        )

    def test_independent_cap_audit_rejects_forged_teacher_and_low_sigma_leak(self) -> None:
        _, bundle, result = self.run_cell(sigma=0.6)
        config = iar_core.IARConfig()
        forged_teacher = replace(
            result.teacher_result,
            teacher_action_residual=(
                result.teacher_result.teacher_action_residual + 0.25
            ),
        )
        with self.assertRaisesRegex(
            smoke.IAROfficialRuntimeSmokeError, "teacher_action_residual"
        ):
            smoke._independent_core_recompute(
                result.frozen_fields,
                forged_teacher,
                config,
                true_velocity=bundle.true_velocity_packed,
            )

        _, low_bundle, low = self.run_cell(sigma=0.15)
        self.assertTrue(
            torch.equal(
                low.teacher_result.teacher_action_residual,
                torch.zeros_like(low.teacher_result.teacher_action_residual),
            )
        )
        leaked = replace(
            low.teacher_result,
            teacher_action_residual=torch.full_like(
                low.teacher_result.teacher_action_residual, 1.0e-3
            ),
        )
        with self.assertRaises(smoke.IAROfficialRuntimeSmokeError):
            smoke._independent_core_recompute(
                low.frozen_fields,
                leaked,
                config,
                true_velocity=low_bundle.true_velocity_packed,
            )

    def test_bundle_tampering_or_missing_noop_wrong_fails_before_forward(self) -> None:
        bundle = self.bundle()
        copied = replace(bundle, t2v_noisy_latents=bundle.t2v_noisy_latents.clone())
        renderer = self.FakeRenderer()
        kwargs = dict(
            renderer=renderer,
            model_id="transformer_1",
            bridge_fraction=0.5,
            action_t2v_condition=self.action_t2v,
            hard_negative_t2v_conditions=self.negatives,
            noop_mv2v_condition=self.noop_mv2v,
            action_mv2v_condition=self.action_mv2v,
            correct_source_sha256="c" * 64,
            wrong_source_sha256s=("d" * 64,),
        )
        with self.assertRaises(smoke.IAROfficialRuntimeSmokeError):
            smoke._run_official_cell(bundles=(copied,), **kwargs)
        self.assertEqual(renderer.diff_dec.calls, [])
        with self.assertRaisesRegex(
            smoke.IAROfficialRuntimeSmokeError, "at least one"
        ):
            smoke._run_official_cell(
                bundles=(),
                **{**kwargs, "wrong_source_sha256s": ()},
            )
        self.assertEqual(renderer.diff_dec.calls, [])

    def grid(self):
        renderer = self.FakeRenderer()
        fractions = (1.0, 0.0)
        points = (dclr.flow_query_point(0.8), dclr.flow_query_point(0.35))
        results = []
        for fraction in fractions:
            bridge = smoke.construct_bridge_clean(
                self.source, self.proposal, fraction
            )
            for point in points:
                bundle = dclr.build_same_state_query_bundle(
                    self.transformer,
                    correct_source_spatial=self.source,
                    wrong_source_spatial=self.wrong,
                    student_clean_spatial=bridge,
                    epsilon_spatial=self.epsilon,
                    point=point,
                )
                results.append(
                    smoke._run_official_cell(
                        renderer=renderer,
                        model_id="transformer_1",
                        bundles=(bundle,),
                        bridge_fraction=fraction,
                        action_t2v_condition=self.action_t2v,
                        hard_negative_t2v_conditions=self.negatives,
                        noop_mv2v_condition=self.noop_mv2v,
                        action_mv2v_condition=self.action_mv2v,
                        correct_source_sha256="c" * 64,
                        wrong_source_sha256s=("d" * 64,),
                    )
                )
        return renderer, fractions, points, results

    def local_evidence(self):
        renderer, fractions, points, results = self.grid()
        hard_negative_texts = (self.noop_text, self.incomplete_text)
        manifest = {
            "action_instruction_sha256": sha(self.action_text),
            "noop_instruction_sha256": sha(self.noop_text),
            "hard_negatives": [
                {"instruction_sha256": sha(item)}
                for item in hard_negative_texts
            ],
            "hard_negative_count": 2,
            "energies_supplied_externally": False,
            "energy_semantics": smoke.ENERGY_SEMANTICS,
            "noop_present": True,
        }
        action_sample = smoke.build_canonical_renderer_sample(
            self.action_text,
            expected_instruction_sha256=sha(self.action_text),
        )
        negative_samples = tuple(
            smoke.build_canonical_renderer_sample(
                item, expected_instruction_sha256=sha(item)
            )
            for item in hard_negative_texts
        )
        noop_sample = smoke.build_canonical_renderer_sample(
            self.noop_text,
            expected_instruction_sha256=sha(self.noop_text),
        )
        message_schema = smoke.canonical_message_schema_evidence(
            action_sample=action_sample,
            action_instruction_sha256=sha(self.action_text),
            negative_samples=negative_samples,
            hard_negative_instruction_sha256s=tuple(
                sha(item) for item in hard_negative_texts
            ),
            noop_sample=noop_sample,
            noop_instruction_sha256=sha(self.noop_text),
        )
        branch_order = list(iar_core.expected_frozen_branch_names(2, 1))
        cells = [dict(item.record) for item in results]
        continuity = smoke.build_bridge_continuity(
            results, bridge_fractions=fractions, points=points
        )
        evidence = {
            "method": smoke.METHOD_NAME,
            "launcher_source_sha256": "9" * 64,
            "forward_implementation": smoke.FORWARD_IMPLEMENTATION,
            "num_frames": 81,
            "latent_phases": 21,
            "patch_size": [1, 2, 2],
            "hard_negative_manifest": manifest,
            "candidate": {
                "canonical_message_schema": message_schema,
                "proposal_source_iid": "proposal-source-iid",
                "proposal_origin": "native_rollout_predecode_latent",
                "proposal_artifact": {},
                "correct_source_artifact": {},
                "native_provenance": {},
                "paired_target_accessed": False,
            },
            "hard_negative_count": 2,
            "wrong_source_count": 1,
            "branch_order": branch_order,
            "bridge_fractions": list(fractions),
            "sigmas": [point.as_dict() for point in points],
            "cell_records": cells,
            "forwards_per_rank": len(cells) * len(branch_order),
            "homotopy": {
                "formula": "q=(1-sigma)*((1-lambda)*S+lambda*P)+sigma*epsilon",
                "constructed_inside_runtime": True,
                "caller_provided_sigma_states": False,
                "one_proposal_P_for_all_cells": True,
                "one_correct_source_S_for_all_cells": True,
                "one_epsilon_for_all_cells": True,
                "proposal_P": {"content_sha256": "a" * 64},
                "correct_source_S": {"content_sha256": "b" * 64},
                "epsilon": {"content_sha256": "e" * 64},
            },
            "continuity": continuity,
            "iar_core": {
                "direct_corrected_core_call": True,
                "replaceable_core": False,
                "projection_nuisance": "mv2v_noop_correct_minus_mv2v_noop_wrong",
                "action_source_swaps_diagnostic_only": True,
            },
            "donor_plumbing_only": True,
            "source_reward_calibration_authorized": False,
            "source_action_invariance_calibration_authorized": False,
            "training_authorized": False,
            "training_pair_authorized": False,
            "scientific_claim_authorized": False,
            "production_claim_forbidden": True,
            "paired_target_accessed": False,
            "forward_callback_present": False,
            "custom_core_present": False,
            "training": {
                "forward_only": True,
                "backward_performed": False,
                "optimizer_present": False,
                "checkpoint_saved": False,
                "adapter_present": False,
            },
        }
        self.assertEqual(len(renderer.diff_dec.calls), 28)
        return evidence

    def test_sp4_receipt_requires_full_grid_digest_consensus(self) -> None:
        evidence = self.local_evidence()
        digest = smoke._object_sha256(evidence)
        ranks = [
            {
                "rank": rank,
                "world_size": 4,
                "ulysses_size": 4,
                "local_evidence_digest": digest,
            }
            for rank in range(4)
        ]
        receipt = smoke.assemble_sp4_receipt(evidence, ranks)
        self.assertEqual(receipt["distributed"]["world_size"], 4)
        self.assertTrue(receipt["engineering_smoke_only"])
        self.assertTrue(receipt["donor_plumbing_only"])
        self.assertFalse(receipt["training_authorized"])
        self.assertFalse(receipt["source_reward_calibration_authorized"])
        self.assertFalse(
            receipt["source_action_invariance_calibration_authorized"]
        )
        self.assertEqual(evidence["forwards_per_rank"], 28)

        missing_launcher = json.loads(json.dumps(evidence))
        missing_launcher.pop("launcher_source_sha256")
        missing_digest = smoke._object_sha256(missing_launcher)
        with self.assertRaisesRegex(
            smoke.IAROfficialRuntimeSmokeError, "launcher source"
        ):
            smoke.assemble_sp4_receipt(
                missing_launcher,
                [
                    {**item, "local_evidence_digest": missing_digest}
                    for item in ranks
                ],
            )

        forged_schema = json.loads(json.dumps(evidence))
        forged_schema["candidate"]["canonical_message_schema"][
            "candidate_dataset_row_accessed"
        ] = True
        unsigned_schema = dict(
            forged_schema["candidate"]["canonical_message_schema"]
        )
        unsigned_schema.pop("schema_evidence_digest")
        forged_schema["candidate"]["canonical_message_schema"][
            "schema_evidence_digest"
        ] = smoke._object_sha256(unsigned_schema)
        forged_digest = smoke._object_sha256(forged_schema)
        with self.assertRaisesRegex(
            smoke.IAROfficialRuntimeSmokeError, "construction/encoder"
        ):
            smoke.assemble_sp4_receipt(
                forged_schema,
                [
                    {**item, "local_evidence_digest": forged_digest}
                    for item in ranks
                ],
            )

        divergent = [dict(item) for item in ranks]
        divergent[-1]["local_evidence_digest"] = "f" * 64
        with self.assertRaisesRegex(
            smoke.IAROfficialRuntimeSmokeError, "identical"
        ):
            smoke.assemble_sp4_receipt(evidence, divergent)
        tampered = json.loads(json.dumps(evidence))
        tampered["cell_records"][0]["hard_negative_energies"][0] += 1.0
        bad_digest = smoke._object_sha256(tampered)
        bad_ranks = [
            {**item, "local_evidence_digest": bad_digest} for item in ranks
        ]
        with self.assertRaisesRegex(
            smoke.IAROfficialRuntimeSmokeError, "cell digest"
        ):
            smoke.assemble_sp4_receipt(tampered, bad_ranks)


@unittest.skipIf(torch is None, "torch is unavailable")
class ParserContractTests(unittest.TestCase):
    def test_native_only_cli_has_exact81_homotopy_and_no_external_energy(self) -> None:
        parser = smoke.build_parser()
        actions = {action.dest: action for action in parser._actions}
        for required in (
            "candidate_clean_latent",
            "correct_source_clean_latent",
            "wrong_source_clean_latent",
            "hard_negative_manifest",
            "noop_instruction",
            "bridge_fractions",
            "sigmas",
            "noise_seed",
            "launcher_source_sha256",
        ):
            self.assertIn(required, actions)
        for forbidden in (
            "positive_control_paired_target",
            "hard_negative_energies",
            "forward_callback",
            "target_clean_latent",
            "target_video",
            "candidate_row_index",
            "expected_candidate_iid",
        ):
            self.assertNotIn(forbidden, actions)
        self.assertTrue(actions["launcher_source_sha256"].required)
        self.assertEqual(
            tuple(actions["bridge_fractions"].default),
            (1.0, 0.5, 0.0),
        )
        self.assertEqual(tuple(actions["sigmas"].default), smoke.DEFAULT_SIGMAS)
        self.assertEqual(actions["num_frames"].default, 81)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
