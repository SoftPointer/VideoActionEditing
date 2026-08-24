from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import infer_preservation_residual_action_canary_v1 as canary
    import inference_sigma_strata as exact40
    import load_preservation_residual_v1 as loader
    import preservation_residual_action_patch_v1 as patching
    import train_preservation_residual_v1 as training


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@unittest.skipUnless(TORCH_AVAILABLE, "the AUH Bernini runtime provides torch")
class PreservationResidualActionTests(unittest.TestCase):
    def test_config_is_unit_gain_exact40_without_a_reward_parameter(self) -> None:
        config = patching.PreservationPatchConfig(
            target_latent_shape=(1, 16, 21, 30, 52), sequence_parallel_size=4
        )
        config.validate()
        self.assertEqual(config.expected_steps, 40)
        self.assertNotIn("scale", config.__dataclass_fields__)
        self.assertNotIn("threshold", config.__dataclass_fields__)
        self.assertNotIn("reward", config.__dataclass_fields__)

    def test_patch_composes_every_exact40_boundary_at_unit_gain(self) -> None:
        import torch

        class Scheduler:
            def __init__(self) -> None:
                self.sigmas = [torch.tensor(value) for value in exact40.PINNED_POSITIVE_SIGMAS]
                self.step_index = 0
                self.seen = []

            def step(self, model_output, timestep, sample):
                self.seen.append(model_output.detach().clone())
                self.step_index += 1
                return model_output

        class Transformer:
            pass

        class Diffusion:
            use_unipc = True
            transformer_2 = None

            def __init__(self) -> None:
                self.transformer = Transformer()
                self.scheduler = Scheduler()

            def shared_step(self, **_kwargs):
                raise AssertionError("the fake query replaces shared_step")

            def sample(self, guidance_mode, num_inference_steps):
                sample = torch.zeros((1, 1, 64), dtype=torch.float32)
                for timestep in exact40.PINNED_TIMESTEPS:
                    official = torch.full_like(sample, 5.0)
                    self.scheduler.step(official, torch.tensor(timestep), sample)
                return sample

        class Adapter:
            def __init__(self, transformer) -> None:
                self.transformer = transformer

        diffusion = Diffusion()
        adapter = Adapter(diffusion.transformer)
        patch = patching.NativeRV2VPreservationResidualPatch(
            diffusion,
            adapter=adapter,
            noop_prompt_embeds=torch.zeros((1, 3, 8)),
            noop_text_lens=[3],
            source_latent=torch.zeros((1, 16, 1, 2, 2)),
            source_references=[torch.zeros((1, 16, 1, 2, 2)) for _ in range(3)],
            rope=object(),
            config=patching.PreservationPatchConfig(
                target_latent_shape=(1, 16, 1, 2, 2),
                sequence_parallel_size=4,
            ),
        )

        def fake_query(_sample, _timestep, _sigma, *, enabled):
            patch.noop_forwards += 1
            value = 3.0 if enabled else 1.0
            return torch.full((1, 1, 64), value)

        patch._query = fake_query
        patch.install()
        diffusion.sample(guidance_mode="v2v_apg", num_inference_steps=40)
        patch.restore()
        receipt = patch.finalize()
        self.assertEqual(len(diffusion.scheduler.seen), 40)
        self.assertTrue(all(torch.equal(value, torch.full_like(value, 7.0)) for value in diffusion.scheduler.seen))
        self.assertEqual(receipt["composition"], "v_native_action+(v_adapted_noop-v_frozen_noop)")
        self.assertTrue(receipt["unit_gain"])
        self.assertFalse(receipt["feature_reward"])

    def test_rotary_pack_matches_training_pretranspose_layout(self) -> None:
        import torch

        def rope(value, *, source_id):
            del source_id
            tokens = int(value.shape[2]) * (int(value.shape[3]) // 2) * (
                int(value.shape[4]) // 2
            )
            return torch.ones((1, 1, tokens, 64), dtype=torch.complex128)

        donor = torch.zeros((1, 16, 3, 4, 6))
        references = [torch.zeros((1, 16, 1, 4, 6)) for _ in range(3)]
        target = torch.zeros((1, 16, 3, 4, 6))
        expected_tokens = 18 + 3 * 6 + 18
        packed = patching._packed_rotary(
            rope,
            donor,
            references,
            target,
            expected_tokens=expected_tokens,
        )
        self.assertEqual(tuple(packed.shape), (1, 1, expected_tokens, 64))
        # This is the same conversion performed at training call sites.
        training_form = (
            torch.cat(
                (
                    rope(donor, source_id=1),
                    *(rope(value, source_id=index + 2) for index, value in enumerate(references)),
                    rope(target, source_id=0),
                ),
                dim=2,
            )
            .squeeze(0)
            .permute(1, 0, 2)
            .permute(1, 0, 2)
            .unsqueeze(0)
        )
        self.assertTrue(torch.equal(packed, training_form))

    def test_bundle_resolver_closes_training_contract(self) -> None:
        for optimizer_steps in training.LOADABLE_CHECKPOINT_STEPS:
            with self.subTest(optimizer_steps=optimizer_steps), tempfile.TemporaryDirectory() as value:
                root = Path(value)
                adapter = root / "adapter.safetensors"
                adapter.write_bytes(b"adapter")
                (root / "optimizer.pt").write_bytes(b"optimizer")
                (root / "history.json").write_text("{}", encoding="ascii")
                adapter_sha = _sha(adapter)
                receipt = {
                    "schema_version": training.RUN_RECEIPT_SCHEMA,
                    "method": training.METHOD_NAME,
                    "complete": True,
                    "mode": training.MODE,
                    "optimizer_steps": optimizer_steps,
                    "registered_schedule_indices": list(training.REGISTERED_SCHEDULE_INDICES),
                    "training_schedule_indices": list(
                        training.REGISTERED_SCHEDULE_INDICES[:optimizer_steps]
                    ),
                    "base_frozen": True,
                    "frozen_base_action_prior_not_retrained": True,
                    "adapter_rank": training.MAIN_LORA_RANK,
                    "final_adapter_sha256": "0" * 64,
                    "objective": {
                        "name": "single_preservation_residual_mse",
                        "action_reward": False,
                        "feature_reward": False,
                        "synthetic_target": False,
                    },
                    "artifacts": {"adapter.safetensors": adapter_sha},
                }
                receipt["receipt_digest"] = hashlib.sha256(
                    json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
                ).hexdigest()
                receipt_path = root / "receipt.json"
                receipt_path.write_text(
                    json.dumps(receipt, sort_keys=True, separators=(",", ":")),
                    encoding="ascii",
                )
                bundle = loader.resolve_bundle(
                    root,
                    expected_adapter_sha256=adapter_sha,
                    expected_receipt_sha256=_sha(receipt_path),
                )
                self.assertEqual(bundle.adapter_rank, training.MAIN_LORA_RANK)
                self.assertEqual(bundle.adapter_sha256, adapter_sha)
                self.assertEqual(bundle.receipt["optimizer_steps"], optimizer_steps)

    def test_registry_reuse_does_not_reuse_old_scale_arms(self) -> None:
        source = "source action"
        target = "target action"
        cell = {
            "cell_id": "dog",
            "source_video": "/tmp/source.mp4",
            "source_video_sha256": "0" * 64,
            "source_action_caption": source,
            "source_action_caption_sha256": canary._sha256_text(source),
            "target_action_caption": target,
            "target_action_caption_sha256": canary._sha256_text(target),
            "seed": 1,
            "bucket_hw": [480, 832],
            "latent_shape": [1, 16, 21, 30, 52],
        }
        registry = {
            "schema_version": "bernini-self-guided-action-field-core2-v1",
            "arm_scales": {"obsolete": 99.0},
            "contract": {"native_guidance_mode": "v2v_apg"},
            "cells": [cell, {**cell, "cell_id": "human"}],
        }
        self.assertIs(canary._registry_cell(registry, cell_id="dog"), cell)
        self.assertEqual(canary.ARMS, ("native-rv2v", "preservation-residual"))


if __name__ == "__main__":
    unittest.main()
