from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path
import sys
import unittest

import torch


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import braid_stage_a_source_reconstruction_v1 as stage_a


class FakeNativeTransformer:
    dtype = torch.float32

    def __init__(self) -> None:
        self.calls: list[tuple[int, float]] = []

    def patch_vae_latent(
        self, value: torch.Tensor, *, source_id: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = int(value.shape[2] * value.shape[3] * value.shape[4])
        self.calls.append((tokens, float(source_id)))
        latent = torch.full((1, tokens, 4), float(source_id), device=value.device)
        latent[..., 1] = value.mean()
        rotary = torch.full(
            (1, 2, tokens, 3), float(source_id), device=value.device
        )
        return latent, rotary


class FakeWanDiffusion:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def shared_step(self, **kwargs: object) -> torch.Tensor:
        self.calls.append(dict(kwargs))
        return kwargs["noisy_latents"] + 1.0  # type: ignore[operator]


def _video(value: float, *, phases: int = 21) -> torch.Tensor:
    return torch.full((1, 16, phases, 2, 2), value, dtype=torch.float32)


def _references(base: float) -> tuple[torch.Tensor, ...]:
    return tuple(_video(base + index, phases=1) for index in range(4))


def _valid_authorization() -> dict[str, object]:
    evidence = {
        name: {
            "status": "PASS",
            "job_id": 140000 + index,
            "evidence_sha256": f"{index + 1:x}" * 64,
            "checkpoint_content_manifest_sha256": (
                stage_a.PINNED_CHECKPOINT_CONTENT_MANIFEST_SHA256
            ),
            "method_source_revision": "a" * 40,
            "exact81_registry_sha256": "b" * 64,
            "native_schedule_digest": (
                stage_a.native.PINNED_NATIVE_UNIPC40_SCHEDULE_DIGEST
            ),
            "parameter_updates_executed": 0,
        }
        for index, name in enumerate(stage_a.REQUIRED_STAGE0_CANARIES)
    }
    body: dict[str, object] = {
        "schema_version": stage_a.STAGE0_AUTHORIZATION_SCHEMA_VERSION,
        "decision": "AUTHORIZED_ONE_STAGE_A_SHADOW_UPDATE",
        "checkpoint_content_manifest_sha256": (
            stage_a.PINNED_CHECKPOINT_CONTENT_MANIFEST_SHA256
        ),
        "bernini_revision": stage_a.PINNED_BERNINI_REVISION,
        "veomni_revision": stage_a.PINNED_VEOMNI_REVISION,
        "method_source_revision": "a" * 40,
        "exact81_registry_sha256": "b" * 64,
        "native_schedule_digest": stage_a.native.PINNED_NATIVE_UNIPC40_SCHEDULE_DIGEST,
        "parallel_contract": {
            "node_count": 1,
            "visible_gpu_count": 8,
            "world_size": 8,
            "sequence_parallel_size": 4,
            "data_parallel_size": 2,
        },
        "canary_evidence": evidence,
        "optimizer_created": False,
        "parameter_updates_executed": 0,
        "stage_a_shadow_updates_authorized": 1,
    }
    return {**body, "authorization_receipt_sha256": stage_a.object_sha256(body)}


class BRAIDStageASourceReconstructionTests(unittest.TestCase):
    def test_exact81_shifted_states_velocity_and_predicted_clean(self) -> None:
        clean = _video(1.0)
        epsilon = _video(5.0)
        batch = stage_a.prepare_teacher_forced_source_batch(
            clean,
            epsilon,
            indices=(0, 17, 39),
            source_video_sha256="1" * 64,
            noop_caption_utf8_sha256="2" * 64,
        )
        self.assertEqual(tuple(batch.states.indices), (0, 17, 39))
        self.assertEqual(batch.states.timesteps.tolist(), [999.0, 871.0, 117.0])
        self.assertTrue(
            torch.equal(
                batch.states.target_velocity,
                (epsilon - clean).unsqueeze(0).expand_as(batch.states.noisy),
            )
        )
        predicted_clean = stage_a.predicted_clean_from_velocity(
            batch.states, batch.states.target_velocity
        )
        torch.testing.assert_close(
            predicted_clean,
            clean.unsqueeze(0).expand_as(predicted_clean),
            rtol=0.0,
            atol=3.0e-7,
        )
        receipt = batch.receipt()
        self.assertEqual(receipt["flow_shift"], 5.0)
        self.assertEqual(receipt["prediction_type"], "flow_prediction")
        self.assertEqual(
            receipt["native_schedule_digest"],
            stage_a.native.PINNED_NATIVE_UNIPC40_SCHEDULE_DIGEST,
        )
        self.assertEqual(receipt["training_target_role"], "same_raw_source_latent")
        self.assertFalse(receipt["separate_edited_target_consumed"])

    def test_velocity_pack_matches_official_pt_ph_pw_c_order(self) -> None:
        value = torch.arange(1 * 16 * 21 * 2 * 2, dtype=torch.float32).reshape(
            1, 16, 21, 2, 2
        )
        packed = stage_a.pack_exact81_velocity(value.contiguous())
        expected = (
            value.squeeze(0)
            .reshape(16, 21, 1, 2, 1, 2)
            .permute(1, 2, 4, 0, 3, 5)
            .reshape(21, 16, 1, 2, 2)
            .permute(0, 2, 3, 4, 1)
            .reshape(1, 21, 64)
            .contiguous()
        )
        self.assertTrue(torch.equal(packed, expected))

    def test_exact81_is_the_only_clean_endpoint_input(self) -> None:
        bad = _video(1.0, phases=20)
        with self.assertRaisesRegex(stage_a.BRAIDStageAError, "21"):
            stage_a.prepare_teacher_forced_source_batch(
                bad,
                bad.clone(),
                indices=(0,),
                source_video_sha256="1" * 64,
                noop_caption_utf8_sha256="2" * 64,
            )
        signature = inspect.signature(stage_a.prepare_teacher_forced_source_batch)
        self.assertNotIn("edited_target", signature.parameters)
        self.assertNotIn("target_video", signature.parameters)

    def test_native_correct_drop_wrong_share_target_and_forward_coordinate(self) -> None:
        batch = stage_a.prepare_teacher_forced_source_batch(
            _video(1.0),
            _video(9.0),
            indices=(17,),
            source_video_sha256="1" * 64,
            noop_caption_utf8_sha256="2" * 64,
        )
        correct_refs = _references(10.0)
        wrong = _video(3.0)
        wrong_refs = _references(20.0)
        admission = stage_a.WrongSourceAdmission(
            correct_source_video_sha256="1" * 64,
            wrong_source_video_sha256="3" * 64,
            correct_source_latent_sha256=stage_a.source_runtime.tensor_sha256(
                batch.clean_source
            ),
            wrong_source_latent_sha256=stage_a.source_runtime.tensor_sha256(wrong),
            correct_reference_latent_sha256s=tuple(
                stage_a.source_runtime.tensor_sha256(item) for item in correct_refs
            ),
            wrong_reference_latent_sha256s=tuple(
                stage_a.source_runtime.tensor_sha256(item) for item in wrong_refs
            ),
            correct_identity_group="dog.source",
            wrong_identity_group="dog.decoy",
            semantic_class_id="dog",
            scene_camera_bucket_id="floor-static-medium",
            selection_evidence_sha256="4" * 64,
        )
        transformer = FakeNativeTransformer()
        queries = stage_a.build_native_stage_a_queries(
            transformer,
            batch,
            sigma_position=0,
            correct_references=correct_refs,
            wrong_source=wrong,
            wrong_references=wrong_refs,
            wrong_source_admission=admission,
        )
        self.assertEqual(list(queries.branches()), ["correct", "drop", "wrong"])
        self.assertEqual(queries.correct.name, "VI")
        self.assertEqual(queries.drop.name, "none")
        self.assertEqual(queries.wrong.name, "VI")  # type: ignore[union-attr]
        self.assertEqual(queries.schedule_index, 17)
        self.assertEqual(queries.timestep, 871)
        self.assertTrue(queries.receipt()["target_patch_and_rotary_shared_exactly"])

        diffusion = FakeWanDiffusion()
        predictions = stage_a.forward_native_stage_a_queries(
            diffusion,
            queries,
            noop_cond_embeds=torch.zeros((1, 5, 8), dtype=torch.float32),
        )
        self.assertEqual(list(predictions), ["correct", "drop", "wrong"])
        self.assertEqual(len(diffusion.calls), 3)
        self.assertTrue(
            all(call["timesteps"].tolist() == [871.0] for call in diffusion.calls)
        )
        self.assertTrue(
            all(call["cond_embeds"] is diffusion.calls[0]["cond_embeds"] for call in diffusion.calls)
        )
        with self.assertRaisesRegex(stage_a.BRAIDStageAError, "latent hashes differ"):
            stage_a.build_native_stage_a_queries(
                transformer,
                batch,
                sigma_position=0,
                correct_references=correct_refs,
                wrong_source=wrong,
                wrong_references=wrong_refs,
                wrong_source_admission=replace(
                    admission, wrong_source_latent_sha256="5" * 64
                ),
            )

    def test_optional_wrong_margin_is_omitted_instead_of_fabricated(self) -> None:
        target = torch.zeros((2, 3), dtype=torch.float32)
        correct = torch.ones((2, 3), dtype=torch.float32, requires_grad=True)
        drop = torch.ones((2, 3), dtype=torch.float32, requires_grad=True)
        result = stage_a.build_authorized_stage_a_objective(
            {"correct": correct, "drop": drop},
            target,
            sigma_weights=torch.tensor([0.5, 0.5], dtype=torch.float32),
            stage0_authorization=_valid_authorization(),
            drop_margin=0.25,
        )
        self.assertFalse(result.wrong_source_margin_used)
        self.assertEqual(set(result.gap_by_counterfactual), {"drop"})
        self.assertEqual(set(result.hinge_by_counterfactual), {"drop"})

    def test_authorized_objective_has_correct_counterfactual_gradient_signs(self) -> None:
        target = torch.zeros((4, 2), dtype=torch.float32)
        predictions = {
            name: torch.ones((4, 2), dtype=torch.float32, requires_grad=True)
            for name in stage_a.QUERY_NAMES
        }
        result = stage_a.build_authorized_stage_a_objective(
            predictions,
            target,
            sigma_weights=torch.full((4,), 0.25, dtype=torch.float32),
            stage0_authorization=_valid_authorization(),
            drop_margin=0.25,
            wrong_margin=0.25,
        )
        result.loss.backward()
        self.assertTrue(bool((predictions["correct"].grad > 0).all()))
        self.assertTrue(bool((predictions["drop"].grad < 0).all()))
        self.assertTrue(bool((predictions["wrong"].grad < 0).all()))
        self.assertTrue(result.wrong_source_margin_used)

    def test_stage0_authorization_is_explicit_sealed_and_three_way(self) -> None:
        with self.assertRaisesRegex(stage_a.BRAIDStageAError, "non-closed"):
            stage_a.verify_stage0_authorization(None)

        valid = _valid_authorization()
        token = stage_a.verify_stage0_authorization(valid)
        self.assertEqual(token.authorized_shadow_updates, 1)

        tampered = dict(valid)
        tampered["stage_a_shadow_updates_authorized"] = 2
        with self.assertRaisesRegex(stage_a.BRAIDStageAError, "seal"):
            stage_a.verify_stage0_authorization(tampered)

        failed = _valid_authorization()
        failed_body = {
            key: value
            for key, value in failed.items()
            if key != "authorization_receipt_sha256"
        }
        failed_evidence = {
            key: dict(value)
            for key, value in failed_body["canary_evidence"].items()  # type: ignore[union-attr]
        }
        failed_evidence[stage_a.REQUIRED_STAGE0_CANARIES[0]]["status"] = "FAIL"
        failed_body["canary_evidence"] = failed_evidence
        failed = {
            **failed_body,
            "authorization_receipt_sha256": stage_a.object_sha256(failed_body),
        }
        with self.assertRaisesRegex(stage_a.BRAIDStageAError, "did not pass"):
            stage_a.verify_stage0_authorization(failed)

    def test_core_has_no_optimizer_backward_or_parameter_update_surface(self) -> None:
        source = Path(stage_a.__file__).read_text()
        self.assertNotIn("torch.optim", source)
        self.assertNotIn(".backward(", source)
        self.assertNotIn("optimizer.step(", source)
        banned_external_parameters = {
            "mask",
            "track",
            "pose",
            "flow",
            "edited_target",
            "target_video",
        }
        for name in stage_a.__all__:
            value = getattr(stage_a, name)
            if inspect.isfunction(value):
                self.assertTrue(
                    banned_external_parameters.isdisjoint(
                        inspect.signature(value).parameters
                    )
                )


if __name__ == "__main__":
    unittest.main()
