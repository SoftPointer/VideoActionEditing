from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

try:
    import torch

    import source_self_native_ref_contrastive_v3 as native
    import source_self_native_rv2v_guidance as guidance
    import source_self_native_target_adapter as target_adapter

    _TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    native = None  # type: ignore[assignment]
    guidance = None  # type: ignore[assignment]
    target_adapter = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


if _TORCH_AVAILABLE:
    class _Transformer:
        dtype = torch.float32

        def patch_vae_latent(self, value, *, source_id):
            tokens = int(value.shape[2] * value.shape[3] * value.shape[4])
            latent = torch.full(
                (1, tokens, 4), float(source_id), requires_grad=True
            )
            rotary = torch.full((1, 2, tokens, 3), float(source_id))
            return latent, rotary


    class _Diffusion:
        def shared_step(self, **kwargs):
            value = kwargs["noisy_latents"]
            text = kwargs["cond_embeds"]
            return value + text.mean() + 1.0


    class _Adapter:
        restored = False

        @contextmanager
        def route(self, route):
            with target_adapter.activate_route(route):
                yield


@unittest.skipUnless(_TORCH_AVAILABLE, "AUH vace torch runtime is required")
class NativeRV2VGuidanceTests(unittest.TestCase):
    def test_formula_matches_vendor_four_forward_chain(self) -> None:
        transformer = _Transformer()
        video = torch.zeros((1, 16, 21, 2, 2))
        refs = [torch.zeros((1, 16, 1, 2, 2)) for _ in range(4)]
        target = torch.zeros((1, 16, 21, 2, 2))
        pack = native.build_native_rv2v_pack(
            transformer,
            donor_video=video,
            image_references=refs,
            noisy_target=target,
        )
        # The production type check is intentional.  Use a minimal object with
        # the same dataclass identity without installing model wrappers.
        fake = object.__new__(target_adapter.NativeTargetAdapterHandle)
        fake.transformer = object()
        fake.q_wrappers = ()
        fake.o_wrappers = ()
        fake.original_q = ()
        fake.original_o = ()
        fake.block_indices = ()
        fake.original_patch_embedding_id = 0
        fake.restored = False
        fake.route = _Adapter().route
        result = guidance.forward_native_rv2v_guidance(
            _Diffusion(),
            pack,
            timestep=torch.tensor([999.0]),
            cond_embeds=torch.ones((1, 2, 3)),
            uncond_embeds=torch.zeros((1, 2, 3)),
            adapter=fake,
            sequence_parallel_rank=0,
            sequence_parallel_size=1,
        )
        c = result.components
        expected = (
            c["none_uncond"]
            + 1.25 * (c["V_uncond"] - c["none_uncond"])
            + 4.5 * (c["VI_uncond"] - c["V_uncond"])
            + 4.0 * (c["VI_cond"] - c["VI_uncond"])
        )
        torch.testing.assert_close(result.guided, expected)
        self.assertEqual(
            list(c), ["none_uncond", "V_uncond", "VI_uncond", "VI_cond"]
        )
        self.assertTrue(result.receipt["image_only_axis_built_but_not_forwarded_by_rv2v"])
        self.assertEqual(pack.reference_count, 4)
        self.assertEqual(pack.patch_call_source_ids, native.PATCH_CALL_SOURCE_IDS)
        self.assertEqual(pack.patch_call_roles, native.PATCH_CALL_ROLES)
        self.assertEqual(pack.image.concat_order, native.BRANCH_CONCAT_ORDER["I"])
        self.assertEqual(
            pack.video_image.concat_order, native.BRANCH_CONCAT_ORDER["VI"]
        )
        self.assertEqual(pack.receipt()["rotary_concat_dim"], 2)
        self.assertEqual(
            pack.receipt()["native_rv2v4_reference_contract_digest"],
            native.PINNED_NATIVE_RV2V4_REFERENCE_CONTRACT_DIGEST,
        )

    def test_static_formula_is_pinned(self) -> None:
        receipt = guidance.guidance_receipt()
        self.assertEqual(receipt["omega_video_hex"], (1.25).hex())
        self.assertEqual(receipt["omega_image_hex"], (4.5).hex())
        self.assertEqual(receipt["omega_text_hex"], (4.0).hex())
        self.assertEqual(receipt["schema_version"], "bernini-native-rv2v-guidance-training-v2")
        self.assertEqual(
            receipt["native_rv2v4_reference_contract_digest"],
            native.native_rv2v4_reference_contract()["digest"],
        )


if __name__ == "__main__":
    unittest.main()
