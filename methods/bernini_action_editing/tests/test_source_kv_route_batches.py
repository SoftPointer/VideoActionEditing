from __future__ import annotations

from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import source_kv_route_batches as batches


try:
    import torch
except ImportError:  # pragma: no cover - lightweight local environment
    torch = None


class StaticBatchContractTests(unittest.TestCase):
    def test_exact_noop_text_and_hash_are_locked(self) -> None:
        self.assertEqual(
            batches.validate_noop_instruction(batches.EXACT_NOOP_INSTRUCTION),
            batches.EXACT_NOOP_INSTRUCTION_SHA256,
        )
        with self.assertRaises(batches.SourceKVRouteBatchError):
            batches.validate_noop_instruction(batches.EXACT_NOOP_INSTRUCTION + " ")


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class TensorBatchTests(unittest.TestCase):
    def _pair(self, *, noop: bool):
        total = 8
        value = {
            "input_ids": torch.tensor([[10 if noop else 20, 11 if noop else 21]]),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
            "t5_input_lens": torch.tensor([[2]], dtype=torch.long),
            "input_vae_latents": torch.arange(total * 3, dtype=torch.float32).reshape(total, 3),
            "input_vae_rope": torch.arange(total * 2 * 3, dtype=torch.float32).reshape(total, 2, 3),
            "vae_latents_mask": torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=torch.bool),
            "vae_seqlen": torch.tensor([[total, 0]], dtype=torch.long),
            "timesteps": torch.tensor([[117]], dtype=torch.long),
            "target_velocity": torch.zeros((4, 3), dtype=torch.float32),
            "target_lens": torch.tensor([[4, 0]], dtype=torch.long),
        }
        return value

    def test_carrier_is_exact_source_prefix_with_no_target_fields(self) -> None:
        action = self._pair(noop=False)
        noop = self._pair(noop=True)
        result = batches.build_source_only_carrier_batch(
            action_pair_batch=action,
            noop_pair_batch=noop,
            noop_instruction=batches.EXACT_NOOP_INSTRUCTION,
        )
        carrier = result.batch
        self.assertEqual(result.source_tokens, 4)
        self.assertEqual(set(carrier), set(batches.CARRIER_MODEL_FIELDS))
        self.assertTrue(torch.equal(carrier["input_vae_latents"], noop["input_vae_latents"][:4]))
        self.assertTrue(torch.equal(carrier["input_vae_rope"], noop["input_vae_rope"][:4]))
        self.assertEqual(carrier["vae_seqlen"].reshape(-1).tolist(), [4, 0])
        self.assertNotIn("vae_latents_mask", carrier)
        self.assertNotIn("target_velocity", carrier)
        self.assertNotIn("target_lens", carrier)

    def test_pair_state_mismatch_and_noncontiguous_selector_fail(self) -> None:
        action = self._pair(noop=False)
        noop = self._pair(noop=True)
        noop["timesteps"] = torch.tensor([[116]], dtype=torch.long)
        with self.assertRaises(batches.SourceKVRouteBatchError):
            batches.validate_equal_pair_batches(action, noop)
        noop = self._pair(noop=True)
        noop["vae_latents_mask"] = torch.tensor(
            [[0, 1, 0, 0, 1, 1, 0, 1]], dtype=torch.bool
        )
        with self.assertRaises(batches.SourceKVRouteBatchError):
            batches.validate_equal_pair_batches(action, noop)

    def test_forbidden_external_condition_is_rejected(self) -> None:
        action = self._pair(noop=False)
        noop = self._pair(noop=True)
        noop["optical_flow"] = torch.zeros(1)
        with self.assertRaises(batches.SourceKVRouteBatchError):
            batches.build_source_only_carrier_batch(
                action_pair_batch=action,
                noop_pair_batch=noop,
                noop_instruction=batches.EXACT_NOOP_INSTRUCTION,
            )

    def test_full_prediction_and_target_selection_use_pinned_geometry(self) -> None:
        class Transformer:
            @staticmethod
            def patch_embedding(value):
                return value

        class Decoder:
            transformer = Transformer()
            transformer_2 = None

            def __init__(self):
                self.call = None

            def shared_step(self, **kwargs):
                self.call = kwargs
                return kwargs["noisy_latents"] + 1.0

        class Renderer:
            def __init__(self):
                self.diff_dec = Decoder()

            @staticmethod
            def get_t5_text_embeddings(input_ids, attention_mask, t5_input_lens):
                del input_ids, attention_mask, t5_input_lens
                return [2], torch.zeros((1, 2, 3), dtype=torch.float32)

        action = self._pair(noop=False)
        noop = self._pair(noop=True)
        carrier = batches.build_source_only_carrier_batch(
            action_pair_batch=action,
            noop_pair_batch=noop,
            noop_instruction=batches.EXACT_NOOP_INSTRUCTION,
        ).batch
        renderer = Renderer()
        carrier_prediction = batches.renderer_full_velocity_prediction(renderer, carrier)
        self.assertEqual(tuple(carrier_prediction.shape), (1, 4, 3))
        self.assertEqual(renderer.diff_dec.call["batch_vae_seqlen"], [4])
        self.assertEqual(renderer.diff_dec.call["batch_text_seqlen"], [2])
        self.assertEqual(renderer.diff_dec.call["model_id"], "transformer_1")
        self.assertEqual(tuple(renderer.diff_dec.call["rotary_embs"].shape), (1, 2, 4, 3))

        full_prediction = batches.renderer_full_velocity_prediction(renderer, action)
        selected = batches.select_target_velocity(full_prediction, action)
        self.assertEqual(tuple(full_prediction.shape), (1, 8, 3))
        self.assertTrue(torch.equal(selected, full_prediction[:, 4:, :]))

    def test_full_prediction_rejects_wrong_shared_step_length(self) -> None:
        class Transformer:
            @staticmethod
            def patch_embedding(value):
                return value

        class Decoder:
            transformer = Transformer()
            transformer_2 = None

            @staticmethod
            def shared_step(**kwargs):
                return kwargs["noisy_latents"][:, :-1]

        class Renderer:
            diff_dec = Decoder()

            @staticmethod
            def get_t5_text_embeddings(*args):
                del args
                return [2], torch.zeros((1, 2, 3), dtype=torch.float32)

        with self.assertRaises(batches.SourceKVRouteBatchError):
            batches.renderer_full_velocity_prediction(Renderer(), self._pair(noop=False))


if __name__ == "__main__":
    unittest.main()
